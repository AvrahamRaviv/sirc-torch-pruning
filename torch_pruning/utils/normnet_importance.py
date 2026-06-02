"""NormalizedNetImportance: adapt normalize-net contribution scores to tp's pruner.

`tp.pruner.{Magnitude,VBP}Pruner` ranks channels by asking an `tp.Importance` object for a
per-channel score on each pruning Group. This adapter feeds it the normalize-net score
instead of weight magnitude:

  - per_layer    → ‖σ·v‖ per input channel = √NCI         (MeanResidual/BNResidual manager)
  - propagation  → global I^l = W̄^l … W̄^L I^o (variance p=2)  (mean variant only)

Both criteria are PER-INPUT-CHANNEL. In channel pruning a prunable channel is the output
of one layer = the input of its consumer(s). We attach our score on the *in_channels* side
of every reparam'd consumer present in the group, then reuse tp's own group reduction /
normalization (we subclass GroupMagnitudeImportance). Groups with no reparam'd consumer
fall back to plain weight magnitude so every group still ranks.

This lives in the package (not benchmarks/) so the DDP harness (`torch_pruning.utils.
pruning_utils`) can import it without a layering violation; `benchmarks/vbp/
normalized_net_importance.py` re-exports it for the single-GPU `prune_e2` path.

Flow (matches the experiment pipeline):

    mgr.reparameterize(loader); train_normalized(...)              # train in σ-space
    scores = extract_normnet_scores(mgr, "per_layer")             # WHILE ACTIVE
    mgr.merge_back()                                              # → plain nn.Conv2d/Linear
    imp = NormalizedNetImportance(model, scores)                 # plain model + frozen scores
    pruner = tp.pruner.VBPPruner(model, example_inputs, importance=imp, ...)
    pruner.step()                                                # prune, then FT

Scores are extracted BEFORE merge_back (they read the reparam modules) and keyed by module
dotted-name, which is stable across merge_back (only the module *type* at that name changes,
not the name). The adapter maps group member modules → names on the merged model.
"""
from collections import OrderedDict

import torch

from torch_pruning.pruner import function
from torch_pruning.pruner.importance import GroupMagnitudeImportance

# in_channels pruning fns: the side where our per-input-channel score applies.
_IN_FNS = (function.prune_conv_in_channels, function.prune_linear_in_channels)


def extract_input_channel_scores(mgr, mode="per_layer", *, example_inputs=None,
                                 I_out=None, p=2, conv_reduction="frobenius",
                                 on_mismatch="warn", relative=True):
    """Pull per-input-channel scores from an ACTIVE reparam manager.

    mode="per_layer"   → mgr.input_channel_scores()  (‖σ·v‖ = √NCI, the §2 criterion).
    mode="propagation" → mgr.propagation_importance() (the §3 global criterion).
        Pass example_inputs to build the residual/DAG topology (build_propagation_topology
        with the SAME p); omit it for a pure sequential walk.
        `relative` picks the PDF's two derivations: True (default) → W̄=M^p·D (column-
        normalized, mass-preserving); False → W̄=M^p (non-relative, magnitudes compound
        through depth). Ignored for mode="per_layer".

    Returns OrderedDict[name → 1-D tensor], one score per input channel of that layer.
    Must be called before mgr.merge_back().
    """
    if mode == "per_layer":
        return mgr.input_channel_scores()
    if mode == "propagation":
        topo = None
        if example_inputs is not None:
            topo = mgr.build_propagation_topology(example_inputs, p=p)
        return mgr.propagation_importance(
            I_out=I_out, p=p, conv_reduction=conv_reduction,
            on_mismatch=on_mismatch, topology=topo, relative=relative)
    raise ValueError(f"mode must be 'per_layer' or 'propagation', got {mode!r}")


def extract_normnet_scores(mgr, mode, example_inputs=None, *, p=2,
                           conv_reduction="frobenius", on_mismatch="warn",
                           relative=True):
    """Score extraction with the propagation-needs-mean-variant guard, shared by the
    single-GPU (prune_e2) and DDP (pruning_utils) paths.

    propagation needs per-branch output std (sigma_out_x), which only the mean variant
    tracks → propagation_importance is a mean-manager method. The bn (canonical) variant
    has no propagation_importance yet, so guard with a clear error.

    `relative` (propagation only): True → relative W̄=M^p·D (default); False → non-
    relative W̄=M^p (raw normalized-weight product, compounds through depth).
    """
    if mode == "propagation" and not hasattr(mgr, "propagation_importance"):
        raise ValueError(
            "propagation scorer needs the mean variant (σ_out branch weighting); the bn "
            "variant has no propagation_importance. Use mode='per_layer', or reparameterize "
            "with the mean variant (--sparse_mode reparam / --reparam_variant mean).")
    return extract_input_channel_scores(
        mgr, mode=mode, example_inputs=example_inputs, p=p,
        conv_reduction=conv_reduction, on_mismatch=on_mismatch, relative=relative)


class NormalizedNetImportance(GroupMagnitudeImportance):
    """tp.Importance that ranks channels by normalize-net contribution scores.

    Args:
        model: the (merged, plain) model the pruner operates on. Used to map group
            member modules back to dotted names.
        scores: dict[name → 1-D per-input-channel tensor] from extract_normnet_scores(...).
            Frozen at construction (detached, cpu).
        group_reduction / normalizer: passed to GroupMagnitudeImportance — how scores from
            multiple group members combine ("mean" default) and are normalized.
        fallback: if True (default), groups with no reparam'd consumer fall back to plain
            weight-magnitude (super().__call__). If False, return None for them.
        p: norm degree for the magnitude fallback only (default 2).
    """

    def __init__(self, model, scores, *, group_reduction="mean", normalizer="mean",
                 fallback=True, p=2):
        super().__init__(p=p, group_reduction=group_reduction, normalizer=normalizer)
        self.scores_by_name = OrderedDict(
            (n, s.detach().float().cpu().reshape(-1)) for n, s in scores.items())
        self.name_by_module = {m: n for n, m in model.named_modules()}
        self.fallback = fallback

    @torch.no_grad()
    def __call__(self, group):
        group_imp, group_idxs = [], []
        for i, (dep, idxs) in enumerate(group):
            layer = dep.layer
            if dep.pruning_fn not in _IN_FNS:
                continue
            name = self.name_by_module.get(layer)
            if name is None or name not in self.scores_by_name:
                continue
            s = self.scores_by_name[name]
            sel = s[list(idxs)]                 # align to this member's channel order
            group_imp.append(sel)
            group_idxs.append(group[i].root_idxs)

        if not group_imp:
            # No reparam'd consumer in this group → magnitude fallback (or skip).
            return super().__call__(group) if self.fallback else None

        reduced = self._reduce(group_imp, group_idxs)
        out = self._normalize(reduced, self.normalizer)
        # normalizer="mean" divides by the group mean with no eps: a fully zeroed group
        # (e.g. a heavily λ-regularized layer) → mean≈0 → nan, which would corrupt the
        # global pruning sort. nan→0 makes those dead channels rank as prunable (correct).
        return torch.nan_to_num(out, nan=0.0)
