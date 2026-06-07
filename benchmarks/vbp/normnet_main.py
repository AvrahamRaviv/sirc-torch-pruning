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
    """The logits head: resnet/mobilenet → .fc, convnext/timm → .head. None if neither."""
    for attr in ("fc", "head"):
        m = getattr(model, attr, None)
        if isinstance(m, torch.nn.Linear):
            return m
    return None


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
    elif model_type == "cnn" and interior_only:
        # residual-stream out-channels = stem conv1 + every blockX.Y.conv3 + downsample conv.
        for name, m in model.named_modules():
            if not isinstance(m, torch.nn.Conv2d):
                continue
            if name == "conv1" or name.endswith(".conv3") or ".downsample." in name:
                ig.append(m)
    return ig


def _layer_widths(model):
    """name → output width (out_channels / out_features) for every Conv2d / Linear."""
    w = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            w[name] = m.out_channels
        elif isinstance(m, torch.nn.Linear):
            w[name] = m.out_features
    return w


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
        a_pre, _ = validate(model, val_loader, device, args.model_type)
        log_info(f"fold_native_bn: folded {n_fold} Conv->BN | post-fold acc={a_pre:.4f} "
                 f"(function-preserving — should match pre-fold)")

    # -- 2. NORMALIZE (one-shot transform) ----------------------------------------------
    names = build_whole_net_reparam_layers(
        model, exclude_classifier=True, exclude_stem=args.exclude_stem)
    args.max_batches = args.calib_batches
    mgr = build_reparam_manager(model, names, device, args)
    # Calibrate σ/μ on CLEAN center-crop images, NOT the augmented train loader (scale-0.08
    # RandomResizedCrop distorts σ — and σ defines the normalized weight every score uses).
    calib_loader = build_calib_loader(args, use_ddp=use_ddp)
    mgr.reparameterize(calib_loader)            # post-act BN(affine=False) + v_tilde=σW
    if use_ddp:
        from normalize_net import _broadcast_model_state
        _broadcast_model_state(model)
    acc, _ = validate(model, val_loader, device, args.model_type)
    log_info(f"NORMALIZE: {len(names)} layers, post-transform acc={acc:.4f} "
             f"(function-preserving — should match post-train)")

    # -- 2b. optional FT in normalized coordinates (train v_tilde; WD on σW = contrib reg)
    # Bracket the reg phase with ‖ṽ‖ snapshots → the delta = how much λ-reg made channels
    # prunable (contribution pushed toward 0). Only meaningful when a reg phase actually runs.
    reg_active = args.epochs_norm_ft > 0
    if reg_active:
        log_contribution_norms(mgr, args, "pre_reg")
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
            scores = extract_normnet_scores(
                mgr, args.scorer, example_inputs=(ex if args.scorer == "propagation" else None),
                relative=not args.prop_non_relative, classifier=clf)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                for k in list(scores.keys()):
                    t = scores[k].contiguous()
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
        else:
            log_info(f"classical scorer ({args.scorer}) — stock tp importance, no normnet scores")

        mgr.merge_back()                        # back to plain modules for the tp pruner
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
        if args.scorer == "tp_variance":
            var_imp = tp.importance.VarianceImportance(
                norm_per_layer=(args.imp_normalizer == "mean"), importance_mode="tp_variance")
            var_imp.collect_statistics(model, calib_loader, device, max_batches=args.calib_batches)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                ws = torch.distributed.get_world_size()
                for d in (var_imp.variance, var_imp.means):
                    for k in list(d.keys()):
                        t = d[k].detach().to(device).contiguous()   # NCCL needs CUDA tensors
                        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                        d[k] = (t / ws).to(d[k].device)
            log_info(f"tp_variance: collected activation stats on {len(var_imp.variance)} layers "
                     f"({args.calib_batches} calib batches), norm_per_layer={var_imp.norm_per_layer}")

        # Bias compensation: removing input channel c shifts each consumer's output by
        # E[Δy]=W[:,c]·μ_c. Adding that to the consumer bias BEFORE pruning preserves the
        # expected output (base_pruner._apply_compensation, auto-applied when mean_dict set).
        # μ = per-channel activation mean on calib data. Reuse tp_variance's already-collected
        # (and DDP-synced) means; else collect + all-reduce so every rank edits biases identically.
        mean_dict = None
        if args.bias_comp:
            if var_imp is not None and var_imp.means:
                mean_dict = dict(var_imp.means)
            else:
                from torch_pruning.pruner.importance import collect_activation_means
                mean_dict = collect_activation_means(
                    model, calib_loader, device, max_batches=args.calib_batches)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    ws = torch.distributed.get_world_size()
                    for k in list(mean_dict.keys()):
                        t = mean_dict[k].detach().to(device).contiguous()   # NCCL needs CUDA
                        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                        mean_dict[k] = t / ws
            log_info(f"bias_comp: activation means on {len(mean_dict)} layers "
                     f"({args.calib_batches} calib batches)")

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
            if args.scorer == "tp_variance":
                return var_imp                      # prebuilt + stats-collected (old criterion)
            return NormalizedNetImportance(mdl, scores, group_reduction="mean", normalizer=norm)

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

        tp.pruner.MagnitudePruner(
            model, ex, importance=_imp(model), global_pruning=args.global_pruning,
            pruning_ratio=ratio, max_pruning_ratio=mpr, ignored_layers=_ignored(model),
            mean_dict=mean_dict).step()
        model.to(device)
        per_layer_dist, global_kept = log_prune_distribution(pre_w, _layer_widths(model))
        # reinsert fresh BN at the PRUNED widths (the native BN we folded is gone) so FT has
        # working normalization; _recalibrate_bn below then populates its running stats.
        if folded_bn_locations is not None:
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
        acc, _ = validate(model, val_loader, device, args.model_type)
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
        if folded_bn_locations is not None:
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
    # 3. prune
    p.add_argument("--no_prune", action="store_true")
    p.add_argument("--scorer", default="per_layer",
                   choices=["per_layer", "propagation", "magnitude", "bn_scale", "tp_variance"],
                   help="normnet: per_layer (‖σv‖=√NCI) / propagation (I). classical baselines "
                        "(same harness, no normnet scores): magnitude (group L2) / bn_scale "
                        "(network-slimming BN γ) / tp_variance (OLD vbp_imagenet_pat criterion: "
                        "group-L2(both sides) × sqrt(conv-output activation var); ABLATION vs nci).")
    p.add_argument("--prop_non_relative", action="store_true",
                   help="propagation: non-relative criterion W̄=M^p (raw product, no column-norm "
                        "→ cross-layer, compounds). Default = relative W̄=M^p·D (within-layer).")
    p.add_argument("--pruning_ratio", type=float, default=0.5)
    p.add_argument("--mac_target_g", type=float, default=0.0,
                   help="target MACs in GMAC (e.g. 2.0). >0 overrides --pruning_ratio: "
                        "binary-search the global ratio that hits it. 0 = use ratio.")
    p.add_argument("--global_pruning", action="store_true")
    p.add_argument("--imp_normalizer", default="none",
                   choices=["none", "mean", "max", "sum", "standarization", "gaussian"],
                   help="per-group importance normalizer (applied BEFORE the global threshold). "
                        "Default 'none' = keep raw cross-layer-comparable scores (REQUIRED for "
                        "global pruning + the propagation criterion). 'mean' = old behavior "
                        "(per-layer mean-1 → global pruning collapses to per-layer-uniform).")
    p.add_argument("--no_bn_recalib", action="store_true")
    p.add_argument("--bias_comp", action="store_true",
                   help="bias compensation: add E[Δy]=W[:,c]·μ_c to each consumer bias before "
                        "removing channel c, preserving the expected output. μ = per-channel "
                        "activation mean on calib data. NOTE: a BN after the consumer that gets "
                        "recalibrated (default) re-centers and absorbs most of this — pair with "
                        "--no_bn_recalib to see the compensation effect cleanly.")
    p.add_argument("--no_save_preprune", action="store_true",
                   help="skip saving the pre-prune dense checkpoint (default: save it)")
    p.add_argument("--max_prune_ratio", type=float, default=0.0,
                   help="per-layer floor: cap any single layer's prune fraction (e.g. 0.8 = "
                        "keep ≥20%% of every layer). Stops global pruning gutting cheap-but-"
                        "critical layers (stem/early). 0 = off.")
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
