"""
normnet_main.py — the normalize-net pipeline in its simplest form.

Once a net is put into NORMALIZED form (input divided by σ via post-activation
BN(affine=False), μ folded to bias, weight re-expressed as v_tilde = σ·W), the contribution
of each channel equals the magnitude of its normalized weight. So everything downstream is
STANDARD: train with plain weight decay (= contribution decay), prune by plain magnitude
(= NCI). The only non-stock piece is the one-shot normalize transform, which already lives
in torch_pruning/utils/reparam.py (NormalizedResidualManager).

Four optional steps (skip any with its epochs/flag):

  1. TRAIN        regular training (from scratch with no --checkpoint, or continue one)
  2. NORMALIZE    one-shot transform: fold + post-act BN(affine=False) + v_tilde=σW
     (NORM-FT)    optional fine-tune IN normalized coordinates (train v_tilde, WD on σW)
  3. PRUNE        magnitude of the normalized weight = NCI → stock MagnitudePruner
  4. FINE-TUNE    regular post-prune recovery (plain net, plain WD)

Examples
  # zero-shot: train from scratch, normalize, fine-tune in normalized coords
  python normnet_main.py --cnn_arch resnet50 --data_path <in1k> \
    --epochs_train 90 --lr_train 0.1 --epochs_norm_ft 0 --no_prune --epochs_ft 0

  # prune a trained net by NCI: load, normalize, prune 50%, recover
  python normnet_main.py --cnn_arch resnet50 --checkpoint <plain.pth> --data_path <in1k> \
    --epochs_train 0 --pruning_ratio 0.5 --epochs_ft 90 --lr_ft 0.02
"""
import argparse
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import torch
import torch_pruning as tp

from vbp_common import (
    setup_logging, build_dataloaders, build_calib_loader, load_model, validate,
    build_whole_net_reparam_layers,
)
from normalize_net import (
    build_reparam_manager, train_normalized, log_info, get_device,
    append_metrics, write_run, save_normnet_checkpoint, is_main,
)
from normalized_net_importance import NormalizedNetImportance, extract_normnet_scores
from torch_pruning.utils.pruning_utils import _recalibrate_bn


def _count(model, ex):
    macs, params = tp.utils.count_ops_and_params(model, ex)
    return macs / 1e9, params / 1e6


def _load_any(args, device):
    """Load whatever the --checkpoint points at, returning a ready model:
      - normnet-ckpt bundle (ckpt.py)  → load_ckpt (full object; pruned/merged/ema nets).
        Uses the EMA model by default (the deployable one) — set --load_raw for the endpoint.
      - VNR ckpt (.meta.json format=vnr) → load_normnet_checkpoint (reparam reload).
      - plain state_dict / None         → load_model (rebuild arch + load, or random init)."""
    ckpt = args.checkpoint
    if ckpt:
        from ckpt import is_bundle, load_ckpt
        if is_bundle(ckpt):
            prefer = "raw" if getattr(args, "load_raw", False) else "ema"
            log_info(f"normnet-ckpt bundle → {ckpt} (prefer={prefer})")
            return load_ckpt(ckpt, device, prefer=prefer)
        meta = os.path.splitext(ckpt)[0] + ".meta.json"
        if os.path.exists(meta):
            import json
            with open(meta) as f:
                if json.load(f).get("format") == "vnr":
                    from normalize_net import load_normnet_checkpoint
                    log_info(f"VNR checkpoint detected → {ckpt}")
                    args.checkpoint = None       # inner load_model random-inits the arch;
                    return load_normnet_checkpoint(ckpt, device, args)  # strict-load overwrites
    return load_model(args, device)


def _ratio_for_mac(model, ex, imp_factory, ignored_factory, target_g, global_pruning,
                   max_pruning_ratio=1.0, lo=0.0, hi=0.95, iters=12):
    """Binary-search the global channel pruning_ratio whose pruned MACs ≤ target_g (GMAC).
    Pure search on deepcopies with the SAME frozen scores → caller does the real prune at
    the returned ratio. Deterministic, so every DDP rank lands on the identical ratio.
    max_pruning_ratio caps how much ANY single layer may be pruned (per-layer floor)."""
    import copy
    best = hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        trial = copy.deepcopy(model)
        tp.pruner.MagnitudePruner(
            trial, ex, importance=imp_factory(trial), global_pruning=global_pruning,
            pruning_ratio=mid, max_pruning_ratio=max_pruning_ratio,
            ignored_layers=ignored_factory(trial)).step()
        g = tp.utils.count_ops_and_params(trial, ex)[0] / 1e9
        del trial
        if g <= target_g:
            best, hi = mid, mid          # feasible → tighten downward (less pruning)
        else:
            lo = mid                     # still too big → prune more
    return best


def _classifier(model):
    """The logits head: resnet/mobilenet → .fc, convnext/timm → .head, HF ViT → .classifier.
    None if neither."""
    for attr in ("fc", "head", "classifier"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Linear):
            return m
        if isinstance(m, torch.nn.Sequential):                 # mobilenet_v2: classifier =
            for sub in reversed(m):                            # Sequential(Dropout, Linear) →
                if isinstance(sub, torch.nn.Linear):           # the last Linear is the logits head
                    return sub
    return None


# --- ViT/DeiT MLP-layer predicates (arch-agnostic: HF transformers ViT/DeiT AND timm) ----
def _vit_is_attn(name):
    return ".attention." in name or ".attn." in name


def _vit_fc1(name):
    """The MLP intermediate/expand layer (output = the prunable hidden dim)."""
    return name.endswith(".intermediate.dense") or name.endswith(".mlp.fc1")


def _vit_fc2(name):
    """The MLP output/project layer (input = the prunable hidden dim). Exclude attention's
    own '.output.dense' (HF attention proj shares the .output.dense suffix)."""
    return (name.endswith(".output.dense") and not _vit_is_attn(name)) \
        or name.endswith(".mlp.fc2")


def _propagation_branch_scale(model):
    """Per-channel scale between a residual branch's output and its add, for the
    propagation topology's join shares (PDF σ_a^p/(σ_a^p+σ_b^p) needs σ AT the add).

    ConvNeXt: x = input + gamma * pwconv2(...) — layer-scale gamma (init 1e-6, trained
    per-channel) sits AFTER pwconv2, whose sigma_out_x is measured PRE-gamma. Without
    this, branch shares are skewed by relative gamma^p across blocks. Detect by module
    shape (has both a per-channel `gamma` Parameter and a `pwconv2` child) so it works
    on the reparam'd model too (pwconv2 swapped in place, gamma untouched).

    Returns {pwconv2_dotted_name → |gamma|} or None (resnet etc. — when native BN is
    folded pre-reparam, sigma_out_x already includes it; nothing else sits before the add).
    """
    scale = {}
    for name, mod in model.named_modules():
        g = getattr(mod, "gamma", None)
        if isinstance(g, torch.nn.Parameter) and g.dim() == 1 and hasattr(mod, "pwconv2"):
            scale[f"{name}.pwconv2"] = g.detach().abs()
    return scale or None


def _residual_blocks(model):
    """Map each residual Block module → its residual-branch terminal reparam-layer name, for
    collect_join_covariance. The block's forward must be c = input + branch (a = first input,
    c = output), so hooking it gives a, c and b = c − a at the ADD. ConvNeXt Block detected by
    the same (per-channel `gamma` Parameter + `pwconv2` child) signature as the branch scale;
    the branch terminal is `{block}.pwconv2` (out-channels = residual stream width = c)."""
    blocks = {}
    for name, mod in model.named_modules():
        g = getattr(mod, "gamma", None)
        if isinstance(g, torch.nn.Parameter) and g.dim() == 1 and hasattr(mod, "pwconv2"):
            blocks[mod] = f"{name}.pwconv2"
    return blocks


def _prunable_input_layers(model_type, score_names):
    """The layers whose INPUT channels the pruner actually removes — the greedy candidate pool.
    convnext: pwconv2 inputs (= pwconv1 outputs, the only pruned dim, matches _ignored_layers).
    Other model types: all scored layers (the TP ignored_layers filter still applies at prune)."""
    if model_type == "convnext":
        return [n for n in score_names if n.endswith(".pwconv2")]
    if model_type == "vit":
        return [n for n in score_names if _vit_fc2(n)]   # fc2 inputs = the pruned hidden dim
    return list(score_names)


def _apply_imp_normalizer(s, normalizer):
    """Per-layer importance normalizer matching NormalizedNetImportance/_normalize, applied
    to the greedy candidate scores so the greedy ranks channels the SAME way the global
    threshold would. Folding it in here lets the pruner run with normalizer='none' on the
    emitted removal-rank scores (else the rank order would be re-scaled and broken)."""
    eps = 1e-12
    if normalizer in (None, "none"):
        return s
    if normalizer == "width":
        return s * s.numel()                       # undo the 1/width per-channel dilution
    if normalizer == "mean":
        return s / (s.mean() + eps)
    if normalizer == "max":
        return s / (s.max() + eps)
    if normalizer == "sum":
        return s / (s.sum() + eps)
    return s                                       # gaussian/standarization: leave (rare here)


def _iterative_propagation_scores(mgr, ex, extract_kwargs, *, model_type, normalizer,
                                  drop_per_round=1, max_frac=1.0, ignored_names=frozenset(),
                                  log=print):
    """Greedy iterative propagation pruning (v2 §'Importance score updating'), GROUP-aware.

    Loops: (1) score with the current channels masked (propagation_importance(keep=...) → the
    'variances forced to 1' update), (2) remove the global-least-important prunable GROUP-channel
    (ranked by the SAME reduce+normalizer the pruner uses), (3) repeat. Emits a per-input-channel
    REMOVAL-RANK score: pruned-earliest = lowest → the unchanged global MAC/ratio threshold
    (run with normalizer='none') prunes exactly the greedy prefix at any target.

    GROUP-aware (vs per-layer): a physical prunable dimension can feed SEVERAL in-channel
    consumers (mnv2 residual stream: G64 = features.8/9/10/11.conv.0.0 — ONE dim, 4 consumers).
    Scoring/masking per layer-name double-counts it in the budget AND lets the keep state for the
    same channel diverge across views → degenerate greedy (this was why iter HURT mnv2: 0.11 vs
    one-shot 0.346, while convnext — pwconv2 one-consumer-per-group — was clean). We build the
    real TP groups (on a merged clone; mgr stays active) and greedily remove a group-channel at a
    time: score = reduce(mean) over the group's consumers (matching NormalizedNetImportance),
    mask sets keep=False for that channel in EVERY consumer, rank emitted to every consumer.

    Returns (scores OrderedDict[name → rank tensor], removal_order list[(group_idx, ch)])."""
    import copy as _copy
    import torch as _t
    import torch_pruning as _tp
    import torch_pruning.pruner.function as _F
    from torch_pruning.utils.normnet_importance import _classifier_seed
    _IN = (_F.prune_conv_in_channels, _F.prune_linear_in_channels)
    # Build the DAG topology + classifier seed ONCE — `keep` masks channels INSIDE the backward
    # recursion (it does not change the graph topology or the logits seed), so rebuilding them
    # every round (as extract_normnet_scores would) is pure waste. Score via the masked
    # propagation_importance directly in the loop.
    ek = dict(extract_kwargs)
    p = ek.get("p", 2)
    topo = mgr.build_propagation_topology(
        ex, p=p, use_measured_sigma_c=ek.get("use_measured_sigma_c", False),
        branch_out_scale=ek.get("branch_out_scale"), join_cov=ek.get("join_cov"))
    clf = ek.get("classifier")
    seed = _classifier_seed(mgr, topo, clf, p) if clf is not None else None

    def _score(keep):
        return mgr.propagation_importance(
            I_out=seed, p=p, topology=topo, relative=ek.get("relative", True),
            use_measured_var=ek.get("use_measured_var", False),
            input_cov=ek.get("input_cov"), keep=keep)

    S0 = _score(None)

    # --- real prunable groups (deduped multi-consumer dims) from the TP dependency graph ---
    # Build on a merged plain CLONE: the reparam-active model exposes MeanResidualConv2d (not
    # nn.Conv2d) so the DepGraph can't trace coupling on it. deepcopy+merge_back leaves the live
    # mgr untouched (verified). Group canonical channel j ↔ each member's local idxs[j] (TP keeps
    # every dep's idxs position-aligned to the group's prune order), so masking idxs[j] in all
    # members removes exactly one physical dim, consistently.
    _clone = _copy.deepcopy(mgr)
    _clone.merge_back()
    DG = _tp.DependencyGraph().build_dependency(_clone.model, ex)
    _name_of = {m: n for n, m in _clone.model.named_modules()}
    _ig = [m for n, m in _clone.model.named_modules() if n in ignored_names]
    groups = []                                        # list[ list[(name, idxs list)] ]
    for g in DG.get_all_groups(ignored_layers=_ig,
                               root_module_types=[_t.nn.Conv2d, _t.nn.Linear]):
        members, seen = [], set()
        for dep, idxs in g:
            if dep.handler in _IN:                     # in-channel side = where our score lives
                nm = _name_of.get(dep.target.module)
                if nm in S0 and nm not in seen and S0[nm].numel() > 0:
                    members.append((nm, list(idxs)))
                    seen.add(nm)
        if members:
            groups.append(members)

    pool_names = sorted({nm for mem in groups for nm, _ in mem})
    keep = {n: _t.ones(S0[n].numel(), dtype=_t.bool) for n in pool_names}
    g_keep = [_t.ones(len(mem[0][1]), dtype=_t.bool) for mem in groups]

    def _group_score(S, mem):                          # reduce(mean over consumers) → normalize
        acc = None                                     # (matches NormalizedNetImportance)
        for nm, idxs in mem:
            v = S[nm].float().cpu()[_t.as_tensor(idxs)]
            acc = v if acc is None else acc + v
        return _apply_imp_normalizer(acc / len(mem), normalizer)

    total = sum(int(k.sum()) for k in g_keep)
    target = max(0, min(int(round(max_frac * total)), total))
    log(f"prop_iterative: {len(groups)} groups, {total} group-channels; "
        f"greedy target={target} (drop_per_round={drop_per_round})")
    for _gi, _mem in enumerate(groups):                # gi → layer names + round-0 score stats
        _s0 = _group_score(S0, _mem)
        log(f"  group g{_gi}: n={len(_mem[0][1])} score0[min={float(_s0.min()):.3e} "
            f"mean={float(_s0.mean()):.3e} max={float(_s0.max()):.3e}] "
            f"members={[nm for nm, _ in _mem]}")

    order, milestone = [], max(1, target // 10)
    while len(order) < target:
        S = _score(keep)
        cand = []
        for gi, mem in enumerate(groups):
            gs = _t.nan_to_num(_group_score(S, mem), nan=float("inf"))
            for j in _t.nonzero(g_keep[gi]).flatten().tolist():
                cand.append((float(gs[j]), gi, j))
        cand.sort(key=lambda t: t[0])                  # nan→inf above keeps it total-ordered
        k = min(drop_per_round, target - len(order), len(cand))
        for _, gi, j in cand[:k]:
            g_keep[gi][j] = False
            order.append((gi, j))
            for nm, idxs in groups[gi]:                # mask the SAME physical dim in all views
                keep[nm][idxs[j]] = False
        if len(order) % milestone < k:
            log(f"  iter: pruned {len(order)}/{target}  (global-min {cand[0][0]:.3e} "
                f"@ g{cand[0][1]}[{cand[0][2]}])")

    # rank scores: removed group-channel → its removal step (lower = pruned first), written to
    # EVERY consumer of the group; survivors → total + normalized final importance (> all removed).
    Sf = _score(keep)
    out = OrderedDict()
    rank = {n: _t.full((S0[n].numel(),), float(total), dtype=_t.float) for n in pool_names}
    for step, (gi, j) in enumerate(order):
        for nm, idxs in groups[gi]:
            rank[nm][idxs[j]] = float(step)
    for gi, mem in enumerate(groups):
        surv = g_keep[gi]
        if surv.any():
            gsf = _t.nan_to_num(_group_score(Sf, mem), nan=0.0)
            gsf = float(total) + gsf / (gsf.abs().max() + 1e-12)
            for nm, idxs in mem:
                for j in _t.nonzero(surv).flatten().tolist():
                    rank[nm][idxs[j]] = float(gsf[j])
    for n, s in S0.items():
        out[n] = rank[n] if n in rank else s.float()   # non-pool (ignored at prune) — left as-is
    return out, order


class _FixedCalib:
    """Re-iterable loader over a fixed list of (img, label) CPU batches. Used by --calib_tensor
    to feed BYTE-IDENTICAL calibration input across machines (every stage that consumes the
    calib loader — reparameterize / collect_input_covariance / join_cov / bias_comp — re-iterates
    it, so the batches must replay each pass)."""
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


def _pin_calib(loader, path, max_batches, log):
    """DEBUG: pin the calibration input. file missing → materialize first max_batches batches,
    save, and replay them; file present → load and replay. Returns a _FixedCalib either way, so
    the WHOLE run also becomes calib-deterministic (shuffle/order frozen)."""
    import os as _os
    if _os.path.exists(path):
        batches = torch.load(path, map_location="cpu")
        log(f"calib_tensor: LOADED {len(batches)} fixed calib batches ← {path}")
    else:
        batches = []
        for bi, b in enumerate(loader):
            if bi >= max_batches:
                break
            x = b[0] if isinstance(b, (list, tuple)) else b
            y = b[1] if isinstance(b, (list, tuple)) and len(b) > 1 else torch.zeros(len(x), dtype=torch.long)
            batches.append((x.cpu(), y.cpu()))
        torch.save(batches, path)
        log(f"calib_tensor: DUMPED {len(batches)} calib batches → {path} "
            f"(sum={sum(float(x.sum()) for x, _ in batches):.4f})")
    return _FixedCalib(batches)


def _dump_artifacts(out_dir, stats, icov, scores, mean_dict, log):
    """DEBUG: save per-layer reparam stats + input_cov + scores + mean_dict for cross-machine
    diffing (probe_compare.py). Stats are the calib-derived quantities every score is built on;
    comparing them per-layer localizes WHICH stage + layer first diverges across platforms.
    `stats` is snapshotted by the caller BEFORE mgr.merge_back() clears the reparam modules."""
    import os as _os
    _os.makedirs(out_dir, exist_ok=True)
    torch.save(stats or {}, _os.path.join(out_dir, "stats.pt"))
    if icov is not None:
        torch.save({k: v.detach().cpu() for k, v in icov.items()},
                   _os.path.join(out_dir, "cov.pt"))
    if scores is not None:
        torch.save({k: v.detach().cpu() for k, v in scores.items()},
                   _os.path.join(out_dir, "scores.pt"))
    if mean_dict is not None:
        torch.save({k: v.detach().cpu() for k, v in mean_dict.items()},
                   _os.path.join(out_dir, "mean_dict.pt"))
    log(f"dump_artifacts: saved stats({len(stats)}) "
        f"cov({0 if icov is None else len(icov)}) "
        f"scores({0 if scores is None else len(scores)}) "
        f"mean_dict({0 if mean_dict is None else len(mean_dict)}) → {out_dir}")


def _ignored_layers(model, model_type, interior_only=False):
    """Layers the global pruner must NOT touch.

    cnn (resnet): only the classifier (.fc) — interior convs are all prunable.
        interior_only=True reproduces the old vbp_imagenet_pat scope: ALSO ignore the
        residual-stream width — the stem conv1, every block's conv3 (bottleneck output),
        and every downsample conv. Leaves only conv1/conv2 of each bottleneck prunable
        (the 32-group scope). Use this to match the proven NCI runs; full scope additionally
        cuts the wide 2048/1024/512 residual dims (param-dense, MAC-cheap → strips params at
        fixed MAC, wrecks pre-FT acc).
    convnext: MLP-only, exactly like ViT. Prune ONLY pwconv1's output (= the block's
        4×dim intermediate, = pwconv2's input). Ignore the head, every downsample/stem
        conv (their out-channels are the residual stream width), the depthwise dwconv, and
        pwconv2 (its out-channels = residual stream). Mirrors vbp_imagenet.py's convnext arm.
    """
    ig = []
    clf = _classifier(model)
    if clf is not None:
        ig.append(clf)
    if model_type == "convnext":
        for ds in model.downsample_layers:
            ig += [m for m in ds.modules() if isinstance(m, torch.nn.Conv2d)]
        for stage in model.stages:
            for block in stage:
                ig += [block.dwconv, block.pwconv2]
    elif model_type == "vit":
        # MLP-only (mirrors convnext + vbp_imagenet_pat build_layers_to_prune): prune ONLY the
        # MLP intermediate/fc1 output (the 4×dim hidden = fc2 input). Ignore everything else —
        # attention (q/k/v + proj), the MLP fc2 (its out = residual-stream width), patch_embed,
        # and the head. Attention is also kept OUT of the propagation reparam set (see main:
        # softmax/head reshape isn't channel-linear → would break the propagation DAG).
        for name, m in model.named_modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)) and not _vit_fc1(name):
                ig.append(m)
    elif model_type == "cnn" and interior_only:
        # Protect the RESIDUAL-STREAM out-channels; leave only the bottleneck interior prunable.
        #   ResNet: stem conv1 + every blockX.Y.conv3 (bottleneck output) + downsample conv.
        #   MobileNetV2: stem features.0.0 + every project conv (blocks 2-17 '.conv.2',
        #     block-1 'features.1.conv.1') + final 'features.18.0'. Leaves the inverted-residual
        #     EXPANSION (conv.0.0 1×1 expand + conv.1.0 3×3 depthwise) prunable — the expanded
        #     hidden dim (= conv.2 input), the mnv2 analogue of resnet conv1/conv2 / convnext
        #     pwconv1. Protecting '.conv.2' stops global pruning gutting the stream to 1 channel
        #     (the collapse seen at aggressive MAC like 0.15G).
        for name, m in model.named_modules():
            if not isinstance(m, torch.nn.Conv2d):
                continue
            resnet_stream = (name == "conv1" or name.endswith(".conv3")
                             or ".downsample." in name)
            mbv2_stream = (name == "features.0.0" or name.endswith(".conv.2")
                           or name == "features.1.conv.1" or name == "features.18.0")
            if resnet_stream or mbv2_stream:
                ig.append(m)
    return ig


def _post_act_target_layers(model, model_type, ex):
    """(producer_module, post_act_fn) pairs for POST-ACTIVATION stats collection — shared by
    tp_variance (the criterion's sqrt(output var)) and bias_comp (μ = consumer input mean).

    Both need the producer's output AFTER its activation (ReLU/GELU), keyed by the producer
    (= the pruner group root). Without target_layers, collect_statistics hooks the RAW
    pre-activation output → wrong variance / wrong μ → ranking + compensation diverge from the
    proven vbp_imagenet_pat behavior. convnext: (pwconv1, GELU) — plain act, NO NHWC→NCHW
    permute (the current isinstance(Linear) hook reads channels from the LAST dim, so a permute
    would misread spatial as channels). cnn: walk the DG to compose each conv's BN+activation."""
    if model_type == "convnext":
        return [(blk.pwconv1, blk.act) for stage in model.stages for blk in stage]
    if model_type == "vit":
        # Arch-agnostic (HF ViT/DeiT `.intermediate` + `.intermediate_act_fn`; timm `.mlp`
        # with `.fc1` + `.act`) — pair each MLP's fc1 producer with its activation.
        out = []
        for name, m in model.named_modules():
            if name.endswith(".intermediate") and hasattr(m, "dense") \
                    and hasattr(m, "intermediate_act_fn"):
                out.append((m.dense, m.intermediate_act_fn))
            elif name.endswith(".mlp") and hasattr(m, "fc1") and hasattr(m, "act"):
                out.append((m.fc1, m.act))
        if out:
            return out
    from torch_pruning.pruner.importance import build_cnn_target_layers
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=ex)
    return build_cnn_target_layers(model, dg)


def _layer_widths(model):
    """name → output width (out_channels / out_features) for every Conv2d / Linear."""
    w = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            w[name] = m.out_channels
        elif isinstance(m, torch.nn.Linear):
            w[name] = m.out_features
    return w


@torch.no_grad()
def compute_nci_cov_scores(model, loader, device, max_batches=50):
    """Covariance-aware NCI per input channel of every Conv2d/Linear consumer.

    Channel-removal cost = drop-one change in the consumer's total output variance:
        ΔVar(c) = 2 Σ_k M_ck Σ_ck − M_cc Σ_cc
    M = input-channel weight Gram (⟨W[:,c], W[:,k]⟩ over out + kernel), Σ = input-channel
    covariance on calib data. Independent NCI = M_cc Σ_cc (only the diagonal of Σ). For conv,
    M uses the kernel-aligned Frobenius inner product (channel-stationary approx); Σ pools batch
    and space. Returns {consumer_name: per-input-channel score} for NormalizedNetImportance —
    these score the consumer's INPUTS = the producer's pruned OUTPUT channels.
    """
    # Prunable consumers only: Linear, or dense (groups==1) Conv2d. Depthwise convs
    # (groups != 1) are NOT consumers of a pruned producer channel (their weight has 1
    # input channel/group → no C×C Gram) and are skipped by build_whole_net_reparam_layers
    # too. Including them → M (1×1) vs cov (C×C) shape crash.
    targets = {n: m for n, m in model.named_modules()
               if (isinstance(m, torch.nn.Linear) and m.in_features > 1)
               or (isinstance(m, torch.nn.Conv2d) and m.groups == 1 and m.in_channels > 1)}
    acc = {n: None for n in targets}          # [sum_AtA (C×C), sum_A (C), count]
    name_of = {m: n for n, m in targets.items()}

    def hook(mod, inp):
        # Dispatch on module TYPE, not tensor ndim: convnext feeds its pwconv (Linear) a 4D
        # channels-LAST (N,H,W,C) tensor — channel is the LAST dim, not dim 1. A conv's input
        # is channels-FIRST (N,C,H,W). Keying off x.dim()==4 misreads the Linear's spatial dim
        # as channels (the 96-vs-56 mismatch).
        x = inp[0].detach()
        if isinstance(mod, torch.nn.Conv2d):  # channels-first (N,C,H,W) → (N*H*W, C)
            a = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        else:                                 # Linear: channel = last dim (handles (N,C) and (N,H,W,C))
            a = x.reshape(-1, x.shape[-1])
        a = a.float()
        n = name_of[mod]
        AtA = a.t() @ a
        As = a.sum(0)
        cnt = a.shape[0]
        if acc[n] is None:
            acc[n] = [AtA, As, cnt]
        else:
            acc[n][0] += AtA; acc[n][1] += As; acc[n][2] += cnt

    handles = [m.register_forward_pre_hook(hook) for m in targets.values()]
    model.eval()
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device))
    for h in handles:
        h.remove()

    ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    scores = {}
    for n, m in targets.items():
        if acc[n] is None:
            continue
        AtA, As, cnt = acc[n]
        if ddp:                               # sync raw moments across ranks (CUDA tensors)
            cnt_t = torch.tensor([float(cnt)], device=AtA.device)
            torch.distributed.all_reduce(AtA, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(As, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)
            cnt = float(cnt_t.item())
        mean = As / cnt
        cov = AtA / cnt - torch.outer(mean, mean)          # (C,C) input-channel covariance
        W = m.weight.data
        if W.dim() == 4:                                   # (out,in,kh,kw) → Gram over (out,kh,kw)
            Wr = W.permute(1, 0, 2, 3).reshape(W.shape[1], -1)
        else:
            Wr = W.t()                                     # (in,out)
        M = (Wr @ Wr.t()).to(cov.dtype)                    # (C,C) weight Gram over input channels
        nci_cov = 2.0 * (M * cov).sum(1) - torch.diag(M) * torch.diag(cov)
        scores[n] = nci_cov.clamp_min(0).detach().cpu()
    return scores


def log_prune_distribution(pre, post):
    """Log per-layer kept/removed channels + the global summary (ported from prune_e2).
    Returns (per_layer dict {name: {pre, post, kept_pct}}, global_kept_pct) for the
    structured prune summary."""
    log_info("per-layer pruning (out width pre → post, kept%):")
    tot_pre = tot_post = 0
    per_layer = {}
    for name in pre:
        a, b = pre[name], post.get(name, pre[name])
        tot_pre += a; tot_post += b
        per_layer[name] = {"pre": a, "post": b, "kept_pct": round(100.0 * b / max(a, 1), 1)}
        if b != a:
            log_info(f"  {name}: {a} → {b}  ({100.0*b/a:.0f}% kept, -{a-b})")
    global_kept = round(100.0 * tot_post / max(tot_pre, 1), 1)
    log_info(f"channels total: {tot_pre} → {tot_post} ({global_kept}% kept)")
    return per_layer, global_kept


def log_contribution_norms(mgr, args, stage):
    """Snapshot the per-channel contribution norm ‖ṽ‖=‖σv‖ (= the thing λ-reg drives to 0).
    Call BEFORE and AFTER the norm-ft/reg phase: the delta is the reg effect — how many
    channels got pushed toward 0 (= prunable). Logs global + per-layer fraction below
    thresholds, saves <tag>_contrib_<stage>.json. mgr must be active (pre merge_back)."""
    scores = extract_normnet_scores(mgr, "per_layer")           # ‖σv‖ per input channel
    allv = torch.cat([s.float().flatten() for s in scores.values()]) if scores else torch.zeros(1)
    glob = {"mean": float(allv.mean()), "median": float(allv.median()),
            "frac_below_1e-3": float((allv < 1e-3).float().mean()),
            "frac_below_1e-2": float((allv < 1e-2).float().mean()),
            "frac_below_1e-1": float((allv < 1e-1).float().mean())}
    log_info(f"[reg-track {stage}] ‖ṽ‖ global: mean={glob['mean']:.3e} median={glob['median']:.3e} "
             f"| prunable frac <1e-3={glob['frac_below_1e-3']:.3f} <1e-2={glob['frac_below_1e-2']:.3f} "
             f"<1e-1={glob['frac_below_1e-1']:.3f}  ({allv.numel()} channels)")
    per_layer = {}
    for name, s in scores.items():
        s = s.float()
        per_layer[name] = {"mean": float(s.mean()),
                           "frac_below_1e-2": float((s < 1e-2).float().mean())}
    if is_main():
        import json as _json
        with open(os.path.join(args.save_dir, f"{args.save_tag}_contrib_{stage}.json"), "w") as f:
            _json.dump({"stage": stage, "global": glob, "per_layer": per_layer}, f, indent=2)
    return glob


def log_score_distribution(scores, args):
    """Per-layer propagation/per_layer SCORE stats (BEFORE pruning) → log + <tag>_scores.json.
    This is the real-data measurement: the per-layer MEAN reveals depth-compounding (does the
    score span orders of magnitude early→late?), CV the within-layer spread. Scores ordered by
    the manager = forward order, so reading top→bottom is shallow→deep."""
    log_info("per-layer score distribution (mean / CV / min / max, forward order):")
    per_layer = {}
    means = []
    for name, s in scores.items():
        s = s.float()
        mean = float(s.mean()); std = float(s.std()) if s.numel() > 1 else 0.0
        cv = std / (mean + 1e-12)
        rec = {"width": int(s.numel()), "mean": mean, "cv": cv,
               "min": float(s.min()), "max": float(s.max())}
        per_layer[name] = rec
        means.append((name, mean))
        log_info(f"  {name}: w={s.numel():4d}  mean={mean:.3e}  cv={cv:.3f}  "
                 f"min={float(s.min()):.3e}  max={float(s.max()):.3e}")
    # cross-layer span = how many orders of magnitude the per-layer means cover (the
    # compounding signature: a large span → global ranking sorts ~by depth, not importance).
    mvals = [m for _, m in means if m > 0]
    span = (max(mvals) / min(mvals)) if len(mvals) > 1 else 1.0
    shallowest = means[0] if means else ("", 0.0)
    deepest = means[-1] if means else ("", 0.0)
    log_info(f"score cross-layer span (max layer-mean / min layer-mean) = {span:.3e}  "
             f"[shallow {shallowest[0]}={shallowest[1]:.3e} → deep {deepest[0]}={deepest[1]:.3e}]")
    import json as _json
    with open(os.path.join(args.save_dir, f"{args.save_tag}_scores.json"), "w") as f:
        _json.dump({"scorer": args.scorer, "relative": not args.prop_non_relative,
                    "cross_layer_span": span, "per_layer": per_layer}, f, indent=2)


def _run_phase(model, mgr, loaders, args, device, use_ddp, *, epochs, lr, tag, teacher=None):
    """Run one train/FT phase via the shared train_normalized loop. mgr=None → plain
    training; mgr active → training in normalized coordinates (WD acts on v_tilde). Swaps
    args.epochs / args.lr for the phase, restores after. Returns best val acc."""
    if epochs <= 0:
        return None
    train_loader, val_loader, train_sampler = loaders
    # train_normalized reads args.epochs/args.lr; normnet_main only has phase-specific
    # flags (epochs_norm_ft/lr_ft/…), so set these transiently (getattr-safe restore).
    saved = (getattr(args, "epochs", None), getattr(args, "lr", None))
    args.epochs, args.lr = epochs, lr
    log_info(f"[{tag}] {epochs} epochs, lr={lr}, "
             f"{'normalized (v_tilde)' if mgr is not None and mgr.is_active else 'plain'}")
    best = train_normalized(model, mgr, train_loader, val_loader, train_sampler,
                            args, device, use_ddp, teacher=teacher)
    args.epochs, args.lr = saved
    return best


def main(argv):
    args = parse_args(argv[1:])
    use_ddp = not args.disable_ddp and "RANK" in os.environ
    if use_ddp:
        from normalize_net import setup_distributed
        device = setup_distributed(args)
    else:
        device = get_device()
        args.rank = 0; args.world_size = 1; args.local_rank = 0
    setup_logging(args.save_dir)
    log_info("=" * 60)
    log_info("normnet pipeline: train → normalize → prune → fine-tune")
    log_info("=" * 60)
    for k, v in vars(args).items():
        log_info(f"  {k}: {v}")

    loaders = build_dataloaders(args, use_ddp=use_ddp)
    train_loader, val_loader, _ = loaders
    had_ckpt = args.checkpoint is not None       # _load_any nulls it for vnr → capture first
    model = _load_any(args, device)             # plain ckpt, VNR ckpt, or random init
    ex = torch.randn(1, 3, 224, 224).to(device)
    dense_macs, dense_params = _count(model, ex)
    write_run(args, {"status": "running", "config": vars(args),
                     "dense_macs_g": dense_macs, "dense_params_m": dense_params})

    # KD teacher = frozen copy of the loaded DENSE net (before normalize/prune). Skipped for
    # from-scratch (no ckpt → a random teacher would be worse than useless). train_one_epoch
    # gates KD on (args.use_kd and teacher is not None), so teacher=None ≡ plain CE.
    teacher = None
    if args.use_kd and had_ckpt:
        import copy
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        log_info("KD teacher = frozen dense net (pre-prune)")
    elif args.use_kd:
        log_info("--use_kd set but no checkpoint (from-scratch) → KD off (no teacher)")

    # -- 1. TRAIN (plain) ---------------------------------------------------------------
    _run_phase(model, None, loaders, args, device, use_ddp,
               epochs=args.epochs_train, lr=args.lr_train, tag="TRAIN")
    if args.epochs_train > 0:
        acc, _ = validate(model, val_loader, device, args.model_type)
        log_info(f"post-train acc={acc:.4f}")

    # -- fold native Conv->BN into the conv (BEFORE reparameterize) so the BN scale lands in
    # M and sigma_out is measured POST-BN. Function-preserving (teacher copied above keeps its
    # own BN; identical logits either way). Fresh BN reinserted after prune, before FT.
    folded_bn_locations = None
    if args.fold_native_bn:
        from torch_pruning.utils.reparam import fold_all_conv_bn
        n_fold, folded_bn_locations = fold_all_conv_bn(model)
        if args.skip_norm_eval:
            log_info(f"fold_native_bn: folded {n_fold} Conv->BN (post-fold eval skipped)")
        else:
            a_pre, _ = validate(model, val_loader, device, args.model_type)
            log_info(f"fold_native_bn: folded {n_fold} Conv->BN | post-fold acc={a_pre:.4f} "
                     f"(function-preserving — should match pre-fold)")

    # -- 2. NORMALIZE (one-shot transform) ----------------------------------------------
    # Classical baselines (magnitude / variance=VBP / tp_variance / bn_scale) must score the
    # ORIGINAL net. The normalize transform scales weights to v_tilde=σW → magnitude-on-σW IS
    # NCI, not true magnitude; and reparam→merge_back is not bit-exact on a BN-free residual net
    # (convnext) → perturbs the baseline. Skip the transform for these scorers entirely.
    do_normalize = args.scorer not in ("magnitude", "variance", "tp_variance", "bn_scale")
    names = build_whole_net_reparam_layers(
        model, exclude_classifier=True, exclude_stem=args.exclude_stem)
    if args.model_type == "vit":
        # MLP-only reparam: keep ONLY fc1/fc2; drop attention (q/k/v + proj) + patch_embed.
        # Attention's qkv→reshape→softmax→proj is not channel-linear (576≠192 fan-out) and
        # would break the propagation DAG; the MLP chain (fc1→GELU→fc2) is channel-linear and
        # carries the prunable hidden dim. Scores still propagate block→block via the residual
        # stream (attention traversed transparently as a non-reparam'd pass-through).
        names = [n for n in names if _vit_fc1(n) or _vit_fc2(n)]
    args.max_batches = args.calib_batches
    mgr = build_reparam_manager(model, names, device, args)
    # Calibrate σ/μ on CLEAN center-crop images, NOT the augmented train loader (scale-0.08
    # RandomResizedCrop distorts σ — and σ defines the normalized weight every score uses).
    calib_loader = build_calib_loader(args, use_ddp=use_ddp)
    if args.calib_tensor:                              # DEBUG: pin calib input (see --calib_tensor)
        calib_loader = _pin_calib(calib_loader, args.calib_tensor, args.calib_batches, log_info)
    if do_normalize:
        mgr.reparameterize(calib_loader)            # post-act BN(affine=False) + v_tilde=σW
        if use_ddp:
            from vbp_common import broadcast_model_state
            broadcast_model_state(model)
        if args.skip_norm_eval:
            log_info(f"NORMALIZE: {len(names)} layers (post-transform eval SKIPPED "
                     "— --skip_norm_eval fast-screen)")
        else:
            acc, _ = validate(model, val_loader, device, args.model_type)
            log_info(f"NORMALIZE: {len(names)} layers, post-transform acc={acc:.4f} "
                     f"(function-preserving — should match post-train)")
    else:
        log_info(f"classical scorer ({args.scorer}): SKIP normalize transform — "
                 f"prune original net (no reparam to norm form)")

    # -- 2b. optional FT in normalized coordinates (train v_tilde; WD on σW = contrib reg)
    # Bracket the reg phase with ‖ṽ‖ snapshots → the delta = how much λ-reg made channels
    # prunable (contribution pushed toward 0). Only meaningful when a reg phase actually runs.
    reg_active = do_normalize and args.epochs_norm_ft > 0
    if reg_active:
        log_contribution_norms(mgr, args, "pre_reg")
    if do_normalize:
        _run_phase(model, mgr, loaders, args, device, use_ddp,
                   epochs=args.epochs_norm_ft, lr=args.lr_norm_ft, tag="NORM-FT", teacher=teacher)
    if reg_active:
        log_contribution_norms(mgr, args, "post_reg")

    # -- 3. PRUNE — normnet criterion (per_layer/propagation) OR a classical tp baseline ---
    _NORMNET_SCORERS = ("per_layer", "propagation")
    if not args.no_prune:
        scores = None
        if args.scorer in _NORMNET_SCORERS:
            # DDP: σ buffers diverge per rank (local EMA); sync, extract, then broadcast the
            # scores rank0→all so every rank prunes the SAME mask (else shape-mismatch hang).
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                mgr._sync_bn_stats()
            # PDF seed: propagation runs to the classifier (I^o = 𝟙 over classes, back through
            # W̄^fc) → final-stage features get real importance instead of a uniform tie.
            clf = _classifier(model)
            # ConvNeXt layer-scale gamma sits between pwconv2 and the residual add —
            # fold it into the join branch shares (σ at the add, per the PDF).
            bscale = _propagation_branch_scale(model) if args.scorer == "propagation" else None
            if bscale:
                log_info(f"propagation: branch_out_scale (layer-scale gamma) on {len(bscale)} branches")
            # Full covariance fix: input correlation Σ̂ per reparam'd layer → cov numerator
            # whose colsum reconstructs true Var(Z) (numerator+denominator together).
            icov = None
            if args.scorer == "propagation" and args.prop_cov:
                if args.prop_p != 2:
                    raise SystemExit("--prop_cov requires --prop_p 2 (variance decomposition)")
                icov = mgr.collect_input_covariance(calib_loader, max_batches=args.calib_batches)
                off = [float((s - torch.eye(s.shape[0], device=s.device, dtype=s.dtype)).abs().sum() / s.abs().sum()) for s in icov.values()]
                log_info(f"prop_cov: input correlation on {len(icov)} layers "
                         f"({args.calib_batches} calib batches), "
                         f"median offdiag mass={sorted(off)[len(off) // 2]:.2f}")
            elif args.scorer == "propagation" and args.prop_measured_var:
                log_info("WARNING: --prop_measured_var without --prop_cov = denominator-only "
                         "(numerator stays independence) → mass leaks → depth bias returns. "
                         "Ablation rung only; add --prop_cov for the full fix.")
            # EXACT residual-join covariance share Cov(b,c)/Var(c) (mass-conserving) — the
            # join-level analogue of prop_cov, replacing the non-conserving skip_sigma_c rescale.
            jcov = None
            if args.scorer == "propagation" and args.prop_join_cov:
                rblocks = _residual_blocks(model)
                if not rblocks:
                    log_info("WARNING: --prop_join_cov set but no residual blocks detected "
                             "(γ+pwconv2) — independence join shares used.")
                else:
                    if args.skip_sigma_c:
                        log_info("NOTE: --prop_join_cov overrides --skip_sigma_c at joins "
                                 "(exact mass-conserving Cov(b,c)/Var(c) vs the σ_c rescale).")
                    jcov = mgr.collect_join_covariance(
                        calib_loader, rblocks, max_batches=args.calib_batches)
                    sums = []  # Σ_branch weight per join should be ≈1 (mass-conserved)
                    log_info(f"prop_join_cov: exact join share on {len(jcov)} residual branches "
                             f"({args.calib_batches} calib batches), "
                             f"median weight_b={sorted(float(w.median()) for w in jcov.values())[len(jcov)//2]:.3f}")
            _extract_kwargs = dict(
                p=args.prop_p, relative=not args.prop_non_relative, classifier=clf,
                use_measured_sigma_c=args.skip_sigma_c,
                use_measured_var=args.prop_measured_var,
                branch_out_scale=bscale, input_cov=icov, join_cov=jcov)
            if args.scorer == "propagation" and args.prop_iterative:
                if args.prop_iter_drop < 1:
                    raise SystemExit("--prop_iter_drop must be ≥ 1")
                _prot_ids = {id(m) for m in
                             _ignored_layers(model, args.model_type,
                                             interior_only=args.interior_only)}
                _prot_names = {n for n, m in model.named_modules() if id(m) in _prot_ids}
                scores, _iter_order = _iterative_propagation_scores(
                    mgr, ex, _extract_kwargs, model_type=args.model_type,
                    normalizer=args.imp_normalizer, drop_per_round=args.prop_iter_drop,
                    max_frac=args.prop_iter_max_frac, ignored_names=_prot_names, log=log_info)
                # the normalizer is now folded into the greedy removal order (rank scores) →
                # the pruner must NOT re-apply it, else the rank order is re-scaled and broken.
                if args.imp_normalizer != "none":
                    log_info(f"prop_iterative: normalizer '{args.imp_normalizer}' folded into "
                             f"greedy ranking → pruner runs with normalizer='none'")
                    args.imp_normalizer = "none"
            else:
                scores = extract_normnet_scores(
                    mgr, args.scorer,
                    example_inputs=(ex if args.scorer == "propagation" else None),
                    **_extract_kwargs)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                for k in list(scores.keys()):
                    t = scores[k].detach().to(device).contiguous()   # NCCL needs CUDA tensors
                    torch.distributed.broadcast(t, src=0)
                    scores[k] = t
            n0 = sum(int((s < 0.1).sum()) for s in scores.values())
            ntot = sum(s.numel() for s in scores.values())
            cvs = [float(s.std() / (s.mean() + 1e-12)) for s in scores.values() if s.numel() > 1]
            cv_med = sorted(cvs)[len(cvs) // 2] if cvs else 0.0
            log_info(f"scores ({args.scorer}): {ntot} channels, {n0} below 0.1 "
                     f"({100.0 * n0 / max(ntot, 1):.1f}%), median per-layer CV={cv_med:.3f}")
            if is_main():
                log_score_distribution(scores, args)
        elif args.scorer == "nci_cov":
            log_info("nci_cov: covariance-aware NCI — scores computed post-merge on plain model")
        else:
            log_info(f"classical scorer ({args.scorer}) — stock tp importance, no normnet scores")

        # W̄ mass diagnostics (Q2 cov colsum, Q3 recon-vs-measured) — read-only, before merge_back.
        if args.dump_wbar and args.scorer == "propagation":
            _icov_diag = icov if icov is not None else mgr.collect_input_covariance(
                calib_loader, max_batches=args.calib_batches)
            _diag = mgr.propagation_diagnostics(input_cov=_icov_diag)
            if is_main():
                os.makedirs(args.dump_wbar, exist_ok=True)
                _wp = os.path.join(args.dump_wbar, f"{args.save_tag}_wbar.pt")
                torch.save(_diag, _wp)
                _cov_n = sum(1 for r in _diag.values() if r.get("has_cov"))
                _leaks = [float(r["leak"].median()) for r in _diag.values() if "leak" in r]
                _cols = [float(r["cov_colsum"].mean()) for r in _diag.values() if "cov_colsum" in r]
                log_info(f"dump_wbar: {len(_diag)} layers, {_cov_n} with cov numerator "
                         f"({len(_diag)-_cov_n} independence-fallback, e.g. depthwise); "
                         f"median leak(recon/meas)={sorted(_leaks)[len(_leaks)//2] if _leaks else float('nan'):.3f} "
                         f"mean cov_colsum={(sum(_cols)/len(_cols)) if _cols else float('nan'):.3f} → {_wp}")

        _stats_snap = None
        if args.dump_artifacts:                 # snapshot reparam stats BEFORE merge_back clears them
            _stats_snap = {name: {k: getattr(rp, k).detach().cpu()
                                  for k in ("mu_x", "sigma_x", "sigma_out_x") if hasattr(rp, k)}
                           for name, rp in getattr(mgr, "_reparam_modules", {}).items()}
        if do_normalize:
            mgr.merge_back()                    # back to plain modules for the tp pruner
        # Save the pre-prune (post-norm-ft, merged) DENSE net so a prune+FT can be retried
        # without redoing training: --checkpoint <preprune> --epochs_norm_ft 0.
        if is_main() and not args.no_save_preprune:
            from ckpt import save_ckpt
            pp = os.path.join(args.save_dir, f"{args.save_tag}_preprune.pth")
            save_ckpt(pp, model, kind="merged", arch=args.cnn_arch,
                      meta={"stage": "preprune", "note": "post-norm-ft merged dense net"})
            log_info(f"saved pre-prune checkpoint → {pp}")
        pre_w = _layer_widths(model)            # widths before the structural edit

        # tp_variance ABLATION (old vbp_imagenet_pat criterion): group-L2(both sides) ×
        # sqrt(conv-OUTPUT activation var). Needs a stats pass on the merged plain model. Build
        # + collect ONCE (expensive); _imp returns it. DDP: per-rank shards → per-rank var →
        # divergent masks/hang. Sync self.variance/means across ranks (all-reduce mean) so every
        # rank ranks identically. imp_normalizer=mean → norm_per_layer=True (the old setting).
        var_imp = None
        var_stats_by_name = None              # tp_variance: {name: (variance, means)} — rebind
        if args.scorer in ("tp_variance", "variance"):  # per _imp call so it survives _ratio_for_mac's copies
            # scorer "variance" = ORIGINAL VBP paper criterion = pure activation variance σ²
            # (no weight term). scorer "tp_variance" = group-L2(both sides) × σ.
            var_imp = tp.importance.VarianceImportance(
                norm_per_layer=(args.imp_normalizer == "mean"), importance_mode=args.scorer)
            # POST-activation output variance (the proven vbp_imagenet_pat criterion). Without
            # target_layers collect_statistics hooks the raw pre-activation output → different
            # ranking → different prune distribution → bad retention (convnext esp.).
            var_imp.collect_statistics(model, calib_loader, device,
                                       target_layers=_post_act_target_layers(model, args.model_type, ex),
                                       max_batches=args.calib_batches)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ws = torch.distributed.get_world_size()
                for d in (var_imp.variance, var_imp.means):
                    for k in list(d.keys()):
                        t = d[k].detach().to(device).contiguous()   # NCCL needs CUDA tensors
                        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                        d[k] = (t / ws).to(d[k].device)
            # Re-key stats by module NAME (stable across deepcopy; module objects are not).
            var_stats_by_name = {}
            for nm, m in model.named_modules():
                if m in var_imp.variance:
                    var_stats_by_name[nm] = (var_imp.variance[m], var_imp.means.get(m))
            log_info(f"{args.scorer}: collected activation stats on {len(var_imp.variance)} layers "
                     f"({args.calib_batches} calib batches), norm_per_layer={var_imp.norm_per_layer}")

        # nci_cov: covariance-aware NCI. Hook every Conv2d/Linear, accumulate channel covariance
        # of its INPUT activations on calib data, combine with the weight Gram → per-input-channel
        # drop-one output-variance change. Keyed by consumer name; fed through NormalizedNetImportance
        # (the per_layer/propagation path) so the global threshold + normalizer apply uniformly.
        if args.scorer == "nci_cov":
            scores = compute_nci_cov_scores(model, calib_loader, device, args.calib_batches)
            n0 = sum(int((s < 0.1 * s.mean()).sum()) for s in scores.values() if s.numel())
            ntot = sum(s.numel() for s in scores.values())
            log_info(f"nci_cov: scored {len(scores)} layers, {ntot} channels "
                     f"({args.calib_batches} calib batches)")

        # Bias compensation: removing input channel c shifts each consumer's output by
        # E[Δy]=W[:,c]·μ_c. Adding that to the consumer bias BEFORE pruning preserves the
        # expected output (base_pruner._apply_compensation, auto-applied when mean_dict set).
        # μ MUST be the consumer's INPUT mean = the producer's output AFTER its activation
        # (ReLU/GELU), keyed by the producer (the group root). Collect with post-act target_layers
        # (like vbp_imagenet) — NOT raw collect_activation_means (pre-act → wrong reference; on a
        # BN-free net like convnext nothing absorbs the error → corrupted biases, acc≈0).
        mean_dict = None
        if args.bias_comp:
            from torch_pruning.pruner.importance import VarianceImportance
            tl = _post_act_target_layers(model, args.model_type, ex)
            vi_bc = VarianceImportance()
            vi_bc.collect_statistics(model, calib_loader, device,
                                     target_layers=tl, max_batches=args.calib_batches)
            mean_dict = dict(vi_bc.means)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ws = torch.distributed.get_world_size()
                for k in list(mean_dict.keys()):
                    t = mean_dict[k].detach().to(device).contiguous()   # NCCL needs CUDA
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                    mean_dict[k] = t / ws
            log_info(f"bias_comp: post-activation means on {len(mean_dict)} layers "
                     f"({args.calib_batches} calib batches)")

        if args.dump_artifacts and is_main():          # DEBUG: cross-machine stage diffing
            _name_by_mod = {m: n for n, m in model.named_modules()}
            _md_named = (None if mean_dict is None else
                         {_name_by_mod.get(k, str(k)): v for k, v in mean_dict.items()})
            _dump_artifacts(args.dump_artifacts, _stats_snap,
                            icov if args.scorer in _NORMNET_SCORERS else None,
                            scores, _md_named, log_info)

        # GLOBAL pruning ranks all groups against ONE threshold (base_pruner cats every group's
        # importance). A per-group normalizer ("mean" → each layer forced to mean-1) ERASES the
        # cross-layer scale before that global ranking → reduces global pruning to ~per-layer-
        # uniform AND kills the propagation criterion's σ_out^p inter-layer transfer (which is
        # what makes non-relative I globally comparable BY DESIGN). normalizer=None keeps raw
        # cross-layer-comparable scores → the global criterion is actually tested. Same setting
        # for every scorer so the comparison is apples-to-apples.
        norm = None if args.imp_normalizer == "none" else args.imp_normalizer

        def _imp(mdl):
            if args.scorer == "magnitude":
                return tp.importance.GroupMagnitudeImportance(p=2, group_reduction="mean", normalizer=norm)
            if args.scorer == "bn_scale":
                return tp.importance.BNScaleImportance(normalizer=norm)
            if args.scorer in ("tp_variance", "variance"):
                # Rebind the collected stats onto THIS model's modules (mdl may be a
                # _ratio_for_mac deepcopy → different module objects, same names). Without this
                # the lookup misses → uniform ones → degenerate global threshold → ratio≈0.
                vi = tp.importance.VarianceImportance(
                    norm_per_layer=var_imp.norm_per_layer, importance_mode=args.scorer)
                for nm, m in mdl.named_modules():
                    if nm in var_stats_by_name:
                        var, mean = var_stats_by_name[nm]
                        vi.variance[m] = var
                        if mean is not None:
                            vi.means[m] = mean
                return vi
            # fallback=False: groups the normnet scorer did NOT score (e.g. the classifier-
            # adjacent head group — classifier is excluded from the reparam set, so the layer
            # feeding it has no propagation score) return importance=None → the global pruner
            # SKIPS them (base_pruner: `if imp is None: continue`) instead of ranking them by
            # raw weight-magnitude on an incomparable scale. Without this, a per-layer cap
            # (--max_prune_ratio) redistributes the MAC budget into that unscored head group and
            # GUTS it (features.18.0 out → cap floor) → pre-FT collapse (~0.001). We only prune
            # what the criterion actually scored.
            return NormalizedNetImportance(mdl, scores, group_reduction="mean", normalizer=norm,
                                           fallback=False)

        def _ignored(mdl):
            return _ignored_layers(mdl, args.model_type, interior_only=args.interior_only)

        # MAC target overrides the raw channel ratio: 2G on RN50 (4.1G) ≈ 30% channels,
        # not 50% — channel-ratio ≠ MAC-ratio. Binary-search the global ratio whose
        # pruned MACs ≤ target_g, then prune the real model at it. Deterministic across
        # ranks (scores already broadcast), so DDP masks stay identical.
        # per-layer floor: cap how much ANY layer may be pruned so global ranking can't empty a
        # cheap-but-critical layer (e.g. the 64-ch stem → ~0 MAC saving, huge acc cost). 1.0=off.
        mpr = args.max_prune_ratio if args.max_prune_ratio > 0 else 1.0
        ratio = args.pruning_ratio
        if args.mac_target_g > 0:
            ratio = _ratio_for_mac(model, ex, _imp, _ignored,
                                   args.mac_target_g, args.global_pruning, max_pruning_ratio=mpr)
            log_info(f"mac_target {args.mac_target_g:.2f}G → global pruning_ratio={ratio:.4f} "
                     f"(max_prune_ratio={mpr})")

        _pruner = tp.pruner.MagnitudePruner(
            model, ex, importance=_imp(model), global_pruning=args.global_pruning,
            pruning_ratio=ratio, max_pruning_ratio=mpr, ignored_layers=_ignored(model),
            mean_dict=mean_dict)
        _pruner.step()                                    # normal global prune (TP records DG history)
        if args.dump_prune and is_main():
            # Dump the EXACT applied mask. TP records pruning_history DURING step(); read that.
            # (Earlier this used step(interactive=True)+manual g.prune(), which produced a
            # DIFFERENT, broken mask vs the normal global step() → dumped a dead net. Replaying
            # that JSON gave 0.002 while the real step() gave 0.28. Use the recorded history.)
            import json as _json
            hist = [[n, bool(is_out), [int(i) for i in idxs]]
                    for n, is_out, idxs in _pruner.pruning_history()]
            pruned_idxs = {n: idxs for n, is_out, idxs in hist if is_out}   # out-ops (back-compat)
            torch.save({"scores": {k: v.detach().cpu() for k, v in scores.items()},
                        "pruned_idxs": pruned_idxs, "ratio": ratio,
                        "imp_normalizer": args.imp_normalizer}, args.dump_prune)
            log_info(f"dump_prune: saved scores + pruned_idxs → {args.dump_prune}")
            # canonical, portable replay format (JSON): reconstructs the EXACT pruned model on a
            # fresh net via DG.load_pruning_history (see apply_prune.py).
            json_path = (args.dump_prune[:-4] if args.dump_prune.endswith(".pth")
                         else args.dump_prune) + ".json"
            with open(json_path, "w") as f:
                _json.dump({"pruning_history": hist, "ratio": round(ratio, 6),
                            "mac_target_g": args.mac_target_g,
                            "imp_normalizer": args.imp_normalizer,
                            "model_type": args.model_type, "cnn_arch": args.cnn_arch},
                           f, indent=2)
            log_info(f"dump_prune: saved pruning_history JSON ({len(hist)} ops) → {json_path}")
        model.to(device)
        per_layer_dist, global_kept = log_prune_distribution(pre_w, _layer_widths(model))
        # reinsert fresh BN at the PRUNED widths (the native BN we folded is gone) so FT has
        # working normalization; _recalibrate_bn below then populates its running stats.
        if folded_bn_locations is not None and args.fold_no_reinsert:
            log_info("fold_no_reinsert: BN stays folded into conv → BN-FREE FT (folded scale "
                     "baked in; keeps BN gain in the propagation score, no fresh-BN reset).")
        elif folded_bn_locations is not None:
            from torch_pruning.utils.reparam import reinsert_bn
            n_re = reinsert_bn(model, folded_bn_locations)
            model.to(device)
            log_info(f"fold_native_bn: reinserted {n_re} fresh BN at pruned widths (pre-FT)")
        if not args.no_bn_recalib:
            log_info(f"recalibrating BN running stats ({args.calib_batches} batches)...")
            _recalibrate_bn(model, train_loader, device, max_batches=args.calib_batches)
            log_info("BN recalibration done")
        else:
            log_info("BN recalibration SKIPPED (--no_bn_recalib)")
        pr_macs, pr_params = _count(model, ex)
        acc, pre_ft_nll = validate(model, val_loader, device, args.model_type)
        tgt = f"mac={args.mac_target_g}G" if args.mac_target_g > 0 else f"ratio={args.pruning_ratio}"
        log_info(f"PRUNE ({args.scorer}, {tgt}): pre-FT acc={acc:.4f}  "
                 f"{pr_macs:.2f}G ({100*pr_macs/dense_macs:.0f}%) {pr_params:.2f}M params")
        # Structured prune summary → dedicated <tag>_prune.json (NOT metrics.jsonl, which is
        # per-epoch only — an epochless record there breaks the curve parsers). Holds the
        # per-layer pruning ratios the user wants.
        if is_main():
            import json as _json
            with open(os.path.join(args.save_dir, f"{args.save_tag}_prune.json"), "w") as f:
                _json.dump({"scorer": args.scorer, "relative": not args.prop_non_relative,
                            "imp_normalizer": args.imp_normalizer,
                            "global_pruning": args.global_pruning, "target": tgt,
                            "global_ratio": round(ratio, 4), "pre_ft_val_acc": round(acc, 6),
                            "macs_g": round(pr_macs, 4), "macs_pct": round(100*pr_macs/dense_macs, 1),
                            "params_m": round(pr_params, 4), "global_kept_pct": global_kept,
                            "per_layer": per_layer_dist}, f, indent=2)
    else:
        mgr.merge_back()
        # no prune: still must restore the folded BN before FT (else FT runs BN-less).
        if folded_bn_locations is not None and not args.fold_no_reinsert:
            from torch_pruning.utils.reparam import reinsert_bn
            n_re = reinsert_bn(model, folded_bn_locations)
            model.to(device)
            if not args.no_bn_recalib:
                log_info(f"recalibrating BN running stats ({args.calib_batches} batches)...")
                _recalibrate_bn(model, train_loader, device, max_batches=args.calib_batches)
                log_info("BN recalibration done")
            log_info(f"fold_native_bn: reinserted {n_re} fresh BN (no-prune path, pre-FT)")

    # -- 4. FINE-TUNE (plain post-prune / post-normalize recovery) ----------------------
    best = _run_phase(model, None, loaders, args, device, use_ddp,
                      epochs=args.epochs_ft, lr=args.lr_ft, tag="FINE-TUNE", teacher=teacher)

    # -- autoresearch ledger: one row per run (Karpathy results.tsv) --------------------
    if args.results_tsv and is_main():
        if best is None and "pre_ft_nll" in locals():   # no FT → score = retention (skip dup eval)
            score, val_nll = float(locals().get("acc")), float(pre_ft_nll)
        else:
            score, val_nll = validate(model, val_loader, device, args.model_type)
        _ret = float(locals().get("acc", float("nan")))            # pre-FT retention (prune branch)
        _macs = float(locals().get("pr_macs", float("nan")))
        _pld = locals().get("per_layer_dist", {}) or {}
        _min_kept = min((v.get("kept_pct", 100.0) for v in _pld.values()), default=float("nan"))
        _desc = getattr(args, "run_desc", "") or args.save_tag
        _new = not os.path.exists(args.results_tsv)
        with open(args.results_tsv, "a") as f:
            if _new:
                f.write("tag\tscore\tval_nll\tretention\tmac_g\tmin_kept_pct\t"
                        "ft_batches\tstatus\tdesc\n")
            f.write(f"{args.save_tag}\t{score:.6f}\t{val_nll:.6f}\t{_ret:.6f}\t{_macs:.4f}\t"
                    f"{_min_kept:.1f}\t{int(getattr(args,'ft_proxy_batches',0))}\t\t{_desc}\n")
        log_info(f"results.tsv += {args.save_tag}  score={score:.4f} nll={val_nll:.4f} "
                 f"ret={_ret:.4f} mac={_macs:.3f}G minkept={_min_kept:.1f}% → {args.results_tsv}")

    # -- save (rank 0) -------------------------------------------------------------------
    if is_main():
        from ckpt import save_ckpt
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, f"{args.save_tag}.pth")
        final_acc, _ = validate(model, val_loader, device, args.model_type)
        fmacs, fparams = _count(model, ex)
        # Pruned model → save the FULL object (reduced channel dims can't rebuild from a bare
        # state_dict). load_ckpt(path) returns it ready to infer / ft-more.
        kind = "dense" if args.no_prune else "pruned"
        save_ckpt(path, model, kind=kind, arch=args.cnn_arch,
                  meta={"final_val_acc": final_acc, "best_ft_val_acc": best,
                        "macs_g": fmacs, "params_m": fparams, "scorer": args.scorer})
        log_info(f"DONE: final acc={final_acc:.4f}  best_ft={best}  {fmacs:.2f}G  {fparams:.2f}M  → {path}")
        write_run(args, {"status": "done", "config": vars(args),
                         "final_val_acc": final_acc, "best_ft_val_acc": best,
                         "final_macs_g": fmacs, "final_params_m": fparams, "checkpoint": path})
    if use_ddp:
        from normalize_net import cleanup
        cleanup()
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser(description="normalize-net pipeline (train→normalize→prune→ft)")
    # model / data
    p.add_argument("--model_type", default="cnn", choices=["cnn", "vit", "convnext"])
    p.add_argument("--model_name", default="resnet50")
    p.add_argument("--cnn_arch", default="resnet50")
    p.add_argument("--checkpoint", default=None, help="None → random init (from scratch). "
                   "Accepts a normnet-ckpt bundle, a VNR ckpt, or a plain state_dict.")
    p.add_argument("--load_raw", action="store_true",
                   help="when --checkpoint is a bundle, load the raw (trajectory-endpoint) "
                        "model instead of the EMA (default: EMA, the deployable one)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--exclude_stem", action="store_true")
    p.add_argument("--interior_only", action="store_true",
                   help="resnet: prune ONLY conv1/conv2 of each bottleneck (ignore stem conv1, "
                        "every conv3, every downsample = the residual-stream width). Reproduces "
                        "the old vbp_imagenet_pat 32-group scope / proven NCI runs. Without it the "
                        "global pruner also cuts the wide residual dims.")
    # 1. train
    p.add_argument("--epochs_train", type=int, default=0, help="plain training epochs (0=skip)")
    p.add_argument("--lr_train", type=float, default=0.1)
    # 2. normalize + optional normalized-FT
    p.add_argument("--reparam_variant", default="bn", choices=["bn", "mean"])
    p.add_argument("--norm_bn_momentum", type=float, default=0.01)
    p.add_argument("--reparam_lambda", type=float, default=0.0, help="extra λ‖σv‖ during norm-ft")
    p.add_argument("--epochs_norm_ft", type=int, default=0, help="FT in normalized coords (0=skip)")
    p.add_argument("--lr_norm_ft", type=float, default=0.01)
    p.add_argument("--calib_batches", type=int, default=50)
    p.add_argument("--fold_native_bn", action="store_true",
                   help="fold each native Conv->BN into the conv weight BEFORE reparameterize, "
                        "so the BN scale (gamma/sigma_run) enters M and the propagation transfer "
                        "(sigma_out) is measured POST-BN. Without it the native post-conv BN is "
                        "absent from the score (it only shows up as the NEXT layer's input sigma). "
                        "Fresh BN is reinserted after prune (recalibrated + FT'd).")
    p.add_argument("--fold_no_reinsert", action="store_true",
                   help="with --fold_native_bn: do NOT reinsert fresh BN after prune → FT a "
                        "BN-FREE net (folded scale baked into conv). Keeps the BN gain in the "
                        "propagation score WITHOUT the fresh-BN reset that hurts recovery. "
                        "No-op without --fold_native_bn.")
    # 3. prune
    p.add_argument("--no_prune", action="store_true")
    p.add_argument("--scorer", default="per_layer",
                   choices=["per_layer", "propagation", "magnitude", "bn_scale", "tp_variance",
                            "variance", "nci_cov"],
                   help="normnet: per_layer (‖σv‖=√NCI) / propagation (I). classical baselines "
                        "(same harness, no normnet scores): magnitude (group L2) / bn_scale "
                        "(network-slimming BN γ) / tp_variance (OLD vbp_imagenet_pat criterion: "
                        "group-L2(both sides) × sqrt(conv-output activation var); ABLATION vs nci). "
                        "nci_cov: COVARIANCE-aware NCI — drop-one output-variance change "
                        "2Σ_k M_ck Σ_ck − M_cc Σ_cc (M=weight Gram, Σ=channel cov); NCI is its "
                        "independence approximation (off-diagonal Σ dropped).")
    p.add_argument("--prop_non_relative", action="store_true",
                   help="propagation: non-relative column norm D=1/σ_pre^p (PDF steps 7-8) "
                        "instead of column-stochastic D=1/Σ_iM^p. NOTE: at p=2 the two are "
                        "IDENTICAL (PDF: 'variance propagation yields the same relative "
                        "importance criterion for p=2') — NO-OP unless --prop_p 1, where the "
                        "forms genuinely differ (and nonrel colsums ΣM/σ_pre ≥ 1 → mass "
                        "inflates per step → depth bias; that's the point of the ablation).")
    p.add_argument("--prop_p", type=int, default=2, choices=[1, 2],
                   help="propagation exponent: 2 = variance (default; rel ≡ nonrel; cov-fix "
                        "capable), 1 = std (rel/nonrel genuinely differ; no cov form exists — "
                        "std does not decompose additively over channel pairs).")
    p.add_argument("--prop_cov", action="store_true",
                   help="propagation FULL covariance fix (p=2 only): numerator AND denominator "
                        "together. Collects per-layer input correlation Σ̂ on calib data; "
                        "numerator becomes the signed cov share N[c,j]=w̃_jc(w̃Σ̂)_jc whose "
                        "colsum reconstructs true Var(Z_j) → W̄ exactly column-stochastic → "
                        "mass conserved with COVARIANCE-correct shares (independence numerator "
                        "= diag-only special case; convnext offdiag = 53-89%% of Var). With "
                        "--prop_measured_var the denom uses measured σ_out_x² instead of the "
                        "recon colsum (recon≈meas).")
    p.add_argument("--skip_sigma_c", action="store_true",
                   help="propagation: at residual joins use the MEASURED post-add std σ_c as the "
                        "branch-weight denominator (PDF σ_c^p/(σ_a^p+σ_b^p) skip factor) instead of "
                        "the independence sum Σσ_branch^p. HEURISTIC: rescales every branch by the "
                        "same σ_c^p/Σσ^p (shares stay ∝ own variance, sum to σ_c^p/Σσ^p ≠ 1 → "
                        "spurious join mass). Prefer --prop_join_cov (exact, mass-conserving).")
    p.add_argument("--prop_join_cov", action="store_true",
                   help="propagation: EXACT residual-join branch share = Cov(b,c)/Var(c) measured "
                        "at the add (mass-conserving: Σ_branches = 1; attributes 2·Cov(a,b) to the "
                        "right path). Join-level analogue of --prop_cov; supersedes the "
                        "--skip_sigma_c rescale (overrides it if both set).")
    p.add_argument("--prop_measured_var", action="store_true",
                   help="propagation: per-layer denominator = MEASURED output variance σ_out_x^p "
                        "(true Var(Z_j)) instead of the computed independence colsum Σ_i(σ_i W_ij)^p. "
                        "WARNING — WITHOUT --prop_cov this is a DENOMINATOR-ONLY fix (numerator "
                        "stays independence ΣM²): colsums = indep/meas ≈ 1/2…1/17, per-output "
                        "varying → mass leaks per step → depth bias RETURNS, worse than no fix. "
                        "Fix numerator+denominator together (--prop_cov) or neither. Kept as "
                        "ablation rung.")
    p.add_argument("--prop_iterative", action="store_true",
                   help="propagation: ITERATIVE (greedy) score updating (v2 §'Importance score "
                        "updating'). Instead of scoring once, greedily (1) score, (2) remove the "
                        "global-least-important prunable channel, (3) re-score with it masked out "
                        "(propagation_importance(keep=...), the 'variances forced to 1' update — "
                        "no forward variance pass, no transfer-fn assumption), repeat. Accounts "
                        "for inter-layer dependency the one-shot score ignores. Emits a per-channel "
                        "REMOVAL-RANK score (pruned-earliest = lowest) → the existing global "
                        "MAC/ratio threshold prunes the greedy prefix. propagation scorer only.")
    p.add_argument("--prop_iter_drop", type=int, default=1,
                   help="iterative: channels removed per round (re-score every k removals). 1 = "
                        "pure one-at-a-time greedy (spec default); >1 = batched bottom-k speed knob "
                        "(L·M² per round; larger k → fewer rounds, coarser update).")
    p.add_argument("--prop_iter_max_frac", type=float, default=1.0,
                   help="iterative: prune this fraction of the prunable channel pool greedily "
                        "(removal order beyond it is irrelevant — the threshold never reaches). "
                        "1.0 = full order (safe for any ratio); lower = faster when the MAC target "
                        "is shallow.")
    p.add_argument("--pruning_ratio", type=float, default=0.5)
    p.add_argument("--mac_target_g", type=float, default=0.0,
                   help="target MACs in GMAC (e.g. 2.0). >0 overrides --pruning_ratio: "
                        "binary-search the global ratio that hits it. 0 = use ratio.")
    p.add_argument("--global_pruning", action="store_true")
    p.add_argument("--imp_normalizer", default="none",
                   choices=["none", "mean", "max", "sum", "standarization", "gaussian", "width"],
                   help="per-group importance normalizer (applied BEFORE the global threshold). "
                        "Default 'none' = keep raw cross-layer-comparable scores (REQUIRED for "
                        "global pruning + the propagation criterion). 'mean' = old behavior "
                        "(per-layer mean-1 → global pruning collapses to per-layer-uniform). "
                        "'width' = score × layer width: undoes the 1/width per-channel dilution "
                        "of the mass-conserving (column-stochastic) propagation score while "
                        "KEEPING the cross-layer mass signal — the fix for global pruning "
                        "gutting the widest layers (convnext stage3 pwconv1=3072).")
    p.add_argument("--no_bn_recalib", action="store_true")
    p.add_argument("--bias_comp", action="store_true",
                   help="bias compensation: add E[Δy]=W[:,c]·μ_c to each consumer bias before "
                        "removing channel c, preserving the expected output. μ = per-channel "
                        "activation mean on calib data. NOTE: a BN after the consumer that gets "
                        "recalibrated (default) re-centers and absorbs most of this — pair with "
                        "--no_bn_recalib to see the compensation effect cleanly.")
    p.add_argument("--no_save_preprune", action="store_true",
                   help="skip saving the pre-prune dense checkpoint (default: save it)")
    p.add_argument("--dump_prune", default="",
                   help="path to torch.save {raw per-channel scores, pruned out-channel idxs "
                        "per layer, ratio} for cluster-vs-local mask diffing")
    p.add_argument("--calib_split", default="train", choices=["train", "val"],
                   help="data split for reparam σ/μ + cov + measured-Var calibration. "
                        "'val' matches the research harness that built the mnv2 leaderboard")
    # ---- cross-platform DEBUG instrumentation (default off → byte-identical behavior) ----
    p.add_argument("--calib_tensor", default="",
                   help="DEBUG: pin the calibration input. If file missing → dump the first "
                        "--calib_batches post-transform calib batches there; if present → LOAD "
                        "them and use as calib (bypass the loader). Lets you feed IDENTICAL calib "
                        "input on local+cluster to split DATA-divergence from CODE/NUMERICS.")
    p.add_argument("--dump_artifacts", default="",
                   help="DEBUG: dir to save per-layer reparam stats (mu_x/sigma_x/sigma_out_x), "
                        "input_cov, raw scores, mean_dict for cross-machine diffing (probe_compare.py).")
    p.add_argument("--max_prune_ratio", type=float, default=0.0,
                   help="per-layer floor: cap any single layer's prune fraction (e.g. 0.8 = "
                        "keep ≥20%% of every layer). Stops global pruning gutting cheap-but-"
                        "critical layers (stem/early). 0 = off.")
    # autoresearch (Karpathy-style) FT proxy harness — all default-off → no behavior change
    p.add_argument("--ft_proxy_batches", type=int, default=0,
                   help="FT-proxy budget: stop FT after this many train batches (epoch 1). "
                        "0 = full FT. The fixed compute budget for the autoresearch ledger.")
    p.add_argument("--results_tsv", default="",
                   help="append one ledger row (tag, score, val_nll, retention, mac, min_kept%%, "
                        "ft_batches, status, desc) per run — the autoresearch results.tsv.")
    p.add_argument("--run_desc", default="", help="one-line description of THIS run's single change "
                   "(for the results.tsv ledger). Defaults to save_tag.")
    p.add_argument("--dump_wbar", default="",
                   help="DEBUG dir: per-layer W̄ mass diagnostics (indep/recon/measured var, cov "
                        "colsum, leak) for the cov(Q2)/measured-vs-recon(Q3) audit.")
    p.add_argument("--val_limit", type=int, default=0,
                   help="use only the first N val samples (deterministic proxy subset). 0 = all.")
    p.add_argument("--skip_norm_eval", action="store_true",
                   help="fast-screen: skip the function-preserving post-fold/post-transform "
                        "validates (sanity checks). Keeps only the pre-FT acc (the ledger score).")
    p.add_argument("--seed", type=int, default=0,
                   help="seed the train-shuffle generator → reproducible K-batch FT proxy. 0 = off.")
    # 4. fine-tune
    p.add_argument("--epochs_ft", type=int, default=0, help="post-prune plain FT (0=skip)")
    p.add_argument("--lr_ft", type=float, default=0.02)
    # shared training knobs (consumed by train_normalized / build_optimizer / loaders)
    p.add_argument("--opt", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay = contribution decay on σW")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--ft_eta_min", type=float, default=1e-6)
    p.add_argument("--ft_warmup_epochs", type=float, default=0)
    p.add_argument("--use_kd", action="store_true")
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--val_batch_size", type=int, default=256)
    p.add_argument("--val_resize", type=int, default=232)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--mu_ema_momentum", type=float, default=0.0)
    # io / ddp
    p.add_argument("--save_dir", required=True)
    p.add_argument("--save_tag", default="normnet")
    p.add_argument("--disable_ddp", action="store_true")
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                   help="set by torch.distributed.launch")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
