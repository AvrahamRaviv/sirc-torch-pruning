"""MUTABLE experiment file (karpathy/autoresearch `train.py` analogue).

Each custom direction is fn(ctx, params) -> dict[layer_name -> 1-D score tensor] (per input
channel, higher = more important / pruned later). The harness builds NormalizedNetImportance
and prunes at matched MAC. Add hypotheses to SPECS; re-run
`python -m research.harness --phase experiments`. Findings → JOURNAL.md.

BASELINE FINDINGS (5k val, pre-FT, MAC 2.94G):
  nci_cov 0.447 > prop_p2_cov 0.404 > prop_rel_p2(plain) 0.242 > variance 0.180
  > prop_cov_joincov 0.176 > prop_joincov 0.137 > tp_variance 0.028 > magnitude 0.009
  ⇒ within-layer COV numerator = big win; JOIN_COV = big loss (its signed residual-join
    shares go negative → mislead the ascending prune). Build on cov, NOT cov+joincov.
    Target: push propagation past nci_cov (0.447).
"""
import torch
from normalized_net_importance import extract_normnet_scores


def _prop(ctx, cov=True, join_cov=False, p=2, relative=True, measured_var=False,
          jcov_override=None):
    ek = ctx.extract_kwargs(cov=cov, join_cov=join_cov, p=p, relative=relative,
                            measured_var=measured_var)
    if jcov_override is not None:
        ek["join_cov"] = jcov_override
    return extract_normnet_scores(ctx.mgr, "propagation", example_inputs=ctx.ex, **ek)


def _geom(a, b):
    """Per-channel geometric mean of two name→[in] score dicts (shared keys/shapes)."""
    out = {}
    for n, va in a.items():
        vb = b.get(n)
        if vb is not None and vb.numel() == va.numel():
            out[n] = (va.abs().clamp_min(1e-12) * vb.abs().clamp_min(1e-12)).sqrt()
        else:
            out[n] = va.abs()
    return out


# --- rescue join_cov: clamp its negative branch shares ≥0 before propagation (harness-side,
#     no TP edit). Negatives were the pre-FT killer; positive-only shares keep the routing. ---
def prop_cov_clampjoin(ctx, p):
    jcov = {n: w.clamp_min(0.0) for n, w in ctx.jcov.items()}
    return _prop(ctx, cov=True, join_cov=True, jcov_override=jcov)


def prop_cov_absjoin(ctx, p):
    jcov = {n: w.abs() for n, w in ctx.jcov.items()}
    return _prop(ctx, cov=True, join_cov=True, jcov_override=jcov)


# --- denominator rung: cov numerator + measured Var(Z) ---
def prop_cov_measured(ctx, p):
    return _prop(ctx, cov=True, join_cov=False, measured_var=True)


# --- hybrids: prop_cov (global routing) × local covariance-aware nci_cov / variance ---
def prop_cov_x_nci(ctx, p):
    return _geom(_prop(ctx, cov=True, join_cov=False), ctx.nci_scores())


def prop_cov_x_nci_join(ctx, p):
    jcov = {n: w.clamp_min(0.0) for n, w in ctx.jcov.items()}
    return _geom(_prop(ctx, cov=True, join_cov=True, jcov_override=jcov), ctx.nci_scores())


# ---------------------------------------------------------------------------
# BATCH 2 — built on the batch-1 winners (cov + measured Var(Z) denom = 0.690;
# geom(prop_cov, nci) = 0.686; both > nci_cov 0.447). Two open questions:
#   (a) is the measured-Var win from the DENOMINATOR alone, or cov-num × measured-den?
#       → ablate measured_only (cov off).
#   (b) normalizer interacts strongly (cov: mean 0.585 > width 0.404 > none 0.315) —
#       sweep mean/none on the measured base, and blend the two top winners.
# ---------------------------------------------------------------------------
def prop_measured_only(ctx, p):
    """Ablation: measured Var(Z) denominator WITHOUT the cov numerator (indep M²)."""
    return _prop(ctx, cov=False, join_cov=False, measured_var=True)


def prop_cov_measured_x_nci(ctx, p):
    """Blend the two batch-1 leaders: geom(cov+measuredVar, nci_cov)."""
    return _geom(_prop(ctx, cov=True, join_cov=False, measured_var=True), ctx.nci_scores())


# ---------------------------------------------------------------------------
# BATCH 3 — the PDF flagship: ITERATIVE score-updating (greedy: score → drop
# least-important → re-propagate with f≈const) ON the cov+measured champion base.
# extract_kwargs sets use_measured_var, which _iterative_propagation_scores reads,
# so iterative + cov + measured Var(Z) composes. normalizer folds into the greedy
# ranking → prune with 'none'.
# ---------------------------------------------------------------------------
def prop_iter_cov_measured(ctx, p):
    import normnet_main as NM
    ek = ctx.extract_kwargs(cov=True, join_cov=False, p=2, relative=True,
                            measured_var=p.get("measured_var", True))
    mtype = getattr(ctx, "mtype", "convnext")
    _ig_ids = {id(m) for m in NM._ignored_layers(ctx.norm_model, mtype)}
    _ig_names = {n for n, m in ctx.norm_model.named_modules() if id(m) in _ig_ids}
    scores, _ = NM._iterative_propagation_scores(
        ctx.mgr, ctx.ex, ek, model_type=mtype,
        normalizer=p.get("iter_norm", "width"),
        drop_per_round=p.get("iter_drop", 128), max_frac=p.get("iter_max_frac", 0.6),
        ignored_names=_ig_names, log=NM.log_info)
    return scores


# ---------------------------------------------------------------------------
# ViT/DeiT residual-join DAG. Attention is NOT channel-linear (softmax mixes TOKENS with
# data-dependent weights → can't propagate per-channel variance through it) and the residual
# stream is a GLOBAL running sum x_final = x0 + Σ_i(attn_i + mlp_i) read by the head. So:
#   (1) propagate WITHIN each MLP branch normally (fc1→GELU→fc2 is channel-linear, like
#       convnext) → per-hidden-channel scores;
#   (2) replace the (wrong) cross-block weighting with the MEASURED global stream share
#       g_i = mean_c Cov(mlp_branch_i, x_final)/Var(x_final)  (captures attention's effect
#       empirically, no softmax model; branch shares of Var(x_final) sum to 1).
# new_score[block_i hidden] = within_block_relative(base) × g_i.
# ---------------------------------------------------------------------------
def _vit_mlp_stream_shares(ctx):
    """Per-block g_i = mean_c Cov(mlp_branch_i, x_final)/Var(x_final), measured on calib.
    timm DeiT: block.mlp output = the mlp branch (no LayerScale on deit_tiny, drop_path=id in
    eval); x_final = input to model.norm (the post-all-adds residual stream the head reads)."""
    model = ctx.dense
    caps, xf = {}, []
    hs = []
    for i, blk in enumerate(model.blocks):
        def mk(i):
            def h(m, inp, out): caps.setdefault(i, []).append(out.detach())
            return h
        hs.append(blk.mlp.register_forward_hook(mk(i)))
    hs.append(model.norm.register_forward_pre_hook(
        lambda m, inp: xf.append(inp[0].detach())))
    model.eval()
    with torch.no_grad():
        for bi, (x, y) in enumerate(ctx.calib):
            if bi >= ctx.args.calib_batches:
                break
            model(x)
    for h in hs:
        h.remove()
    Xf = torch.cat(xf, 0).reshape(-1, xf[0].shape[-1])          # [S, C]
    xf_c = Xf - Xf.mean(0, keepdim=True)
    var_f = xf_c.var(0) + 1e-8
    g = {}
    for i in caps:
        B = torch.cat(caps[i], 0).reshape(-1, Xf.shape[-1])
        b_c = B - B.mean(0, keepdim=True)
        cov = (b_c * xf_c).mean(0)
        g[i] = float((cov / var_f).clamp_min(0).mean())          # block scalar share ≥0
    return g


def prop_vit_joinshare(ctx, p):
    """cov+measured within-MLP propagation, cross-block budget = measured stream share g_i."""
    base = _prop(ctx, cov=p.get("cov", True), join_cov=False,
                 measured_var=p.get("measured_var", True))
    g = _vit_mlp_stream_shares(ctx)
    out = {}
    for n, v in base.items():
        v = v.float().clone()
        if n.endswith(".mlp.fc2"):
            bi = int(n.split(".")[1])
            v = v / (v.mean() + 1e-12) * g.get(bi, 1.0)          # unit-mean per block × g_i
        out[n] = v
    return out


SPECS = [
    # --- MATCHED-NORMALIZER ordering test (iter vs cov vs prop p2) ---
    # cached @ width: iter 0.608 > prop 0.579 > cov 0.563. These fill the missing cells so
    # the iter>cov>prop claim can be checked at a SINGLE normalizer (and each at its own best).
    dict(name="exp_prop_mean", kind="propagation",
         params=dict(normalizer="mean", cov=False, join_cov=False, relative=True, p=2),
         note="plain prop p2 @ mean"),
    dict(name="exp_iter_mean", kind="propagation_iterative",
         params=dict(normalizer="mean", cov=True, join_cov=False, iter_drop=128, iter_max_frac=0.6),
         note="iterative cov @ mean"),
    dict(name="exp_iter_width", kind="propagation_iterative",
         params=dict(normalizer="width", cov=True, join_cov=False, iter_drop=128, iter_max_frac=0.6),
         note="iterative cov (no joincov) @ width"),
    # normalizer sweep on the winning base (prop_cov, no joincov)
    dict(name="exp_cov_normNone", kind="propagation",
         params=dict(normalizer="none", cov=True, join_cov=False), note="prop_cov, norm none"),
    dict(name="exp_cov_normMean", kind="propagation",
         params=dict(normalizer="mean", cov=True, join_cov=False), note="prop_cov, norm mean"),
    # join_cov rescue
    dict(name="exp_cov_clampjoin", kind="custom",
         params=dict(fn="prop_cov_clampjoin", prune_normalizer="width"),
         note="cov + clamp(joincov≥0)"),
    dict(name="exp_cov_absjoin", kind="custom",
         params=dict(fn="prop_cov_absjoin", prune_normalizer="width"),
         note="cov + |joincov|"),
    # denominator
    dict(name="exp_cov_measured", kind="custom",
         params=dict(fn="prop_cov_measured", prune_normalizer="width"),
         note="cov + measured Var(Z)"),
    # hybrids vs nci_cov (the leader)
    dict(name="exp_cov_x_nci", kind="custom",
         params=dict(fn="prop_cov_x_nci", prune_normalizer="width"),
         note="geom(prop_cov, nci_cov)"),
    dict(name="exp_cov_x_nci_join", kind="custom",
         params=dict(fn="prop_cov_x_nci_join", prune_normalizer="width"),
         note="geom(cov+clampjoin, nci_cov)"),
    # ---- BATCH 2: build on cov+measured (0.690) ----
    # normalizer sweep on the measured base (width gave 0.690)
    dict(name="exp_cov_measured_mean", kind="custom",
         params=dict(fn="prop_cov_measured", prune_normalizer="mean"),
         note="cov + measured Var(Z), norm mean"),
    dict(name="exp_cov_measured_none", kind="custom",
         params=dict(fn="prop_cov_measured", prune_normalizer="none"),
         note="cov + measured Var(Z), norm none"),
    # ablation: is the win the measured DENOM alone (no cov numerator)?
    dict(name="exp_measured_only", kind="custom",
         params=dict(fn="prop_measured_only", prune_normalizer="width"),
         note="measured Var(Z) denom, indep num (cov off)"),
    dict(name="exp_measured_only_mean", kind="custom",
         params=dict(fn="prop_measured_only", prune_normalizer="mean"),
         note="measured Var(Z) denom, indep num, norm mean"),
    # blend the two batch-1 leaders
    dict(name="exp_cov_measured_x_nci", kind="custom",
         params=dict(fn="prop_cov_measured_x_nci", prune_normalizer="width"),
         note="geom(cov+measuredVar, nci_cov)"),
    # ---- BATCH 3: iterative score-updating on the champion base ----
    dict(name="exp_iter_cov_measured", kind="custom",
         params=dict(fn="prop_iter_cov_measured", prune_normalizer="none",
                     iter_norm="width", iter_drop=128, iter_max_frac=0.6),
         note="ITERATIVE cov + measured Var(Z), width"),
    # ---- BATCH 4: finer greedy (smaller drop_per_round) on the iterative champion ----
    dict(name="exp_iter_cov_measured_d64", kind="custom",
         params=dict(fn="prop_iter_cov_measured", prune_normalizer="none",
                     iter_norm="width", iter_drop=64, iter_max_frac=0.6),
         note="ITERATIVE cov+measured, drop=64"),
    dict(name="exp_iter_cov_measured_d32", kind="custom",
         params=dict(fn="prop_iter_cov_measured", prune_normalizer="none",
                     iter_norm="width", iter_drop=32, iter_max_frac=0.6),
         note="ITERATIVE cov+measured, drop=32"),
    # ---- COVFIX rerun: reparam._layer_N cov-leak fix (mask v's in-cols before Σ_c' cov sum).
    # Distinct names force a fresh run (done-cache keyed by name). recon = mass-conserving
    # (theory-clean under fix); measured = stale-denom rung (gap-2, expect still weak).
    dict(name="exp_iter_cov_recon_covfix", kind="custom",
         params=dict(fn="prop_iter_cov_measured", prune_normalizer="none", measured_var=False,
                     iter_norm="width", iter_drop=128, iter_max_frac=0.6),
         note="ITERATIVE cov RECON (no measured), covfix"),
    dict(name="exp_iter_cov_measured_covfix", kind="custom",
         params=dict(fn="prop_iter_cov_measured", prune_normalizer="none", measured_var=True,
                     iter_norm="width", iter_drop=128, iter_max_frac=0.6),
         note="ITERATIVE cov+measured, covfix"),
]
