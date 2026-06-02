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
  # zero-shot §7 demo: train from scratch, normalize, fine-tune in normalized coords
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
    setup_logging, build_dataloaders, load_model, validate,
    build_whole_net_reparam_layers,
)
from normalize_net import (
    build_reparam_manager, train_normalized, log_info, get_device,
    append_metrics, write_run, save_normnet_checkpoint,
)
from normalized_net_importance import NormalizedNetImportance, extract_normnet_scores
from torch_pruning.utils.pruning_utils import _recalibrate_bn


def _count(model, ex):
    macs, params = tp.utils.count_ops_and_params(model, ex)
    return macs / 1e9, params / 1e6


def _load_any(args, device):
    """Load a plain ckpt (load_model), or a normnet VNR ckpt when the .meta.json sidecar
    says format=='vnr'. The latter lets Option B reuse the completed RN_bn sparse ckpts
    straight onto the DDP path (the merged_biased save is broken on the cluster)."""
    ckpt = args.checkpoint
    if ckpt:
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
                   lo=0.0, hi=0.95, iters=12):
    """Binary-search the global channel pruning_ratio whose pruned MACs ≤ target_g (GMAC).
    Pure search on deepcopies with the SAME frozen scores → caller does the real prune at
    the returned ratio. Deterministic, so every DDP rank lands on the identical ratio."""
    import copy
    best = hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        trial = copy.deepcopy(model)
        tp.pruner.MagnitudePruner(
            trial, ex, importance=imp_factory(trial), global_pruning=global_pruning,
            pruning_ratio=mid, ignored_layers=ignored_factory(trial)).step()
        g = tp.utils.count_ops_and_params(trial, ex)[0] / 1e9
        del trial
        if g <= target_g:
            best, hi = mid, mid          # feasible → tighten downward (less pruning)
        else:
            lo = mid                     # still too big → prune more
    return best


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

    # -- 2. NORMALIZE (one-shot transform) ----------------------------------------------
    names = build_whole_net_reparam_layers(
        model, exclude_classifier=True, exclude_stem=args.exclude_stem)
    args.max_batches = args.calib_batches
    mgr = build_reparam_manager(model, names, device, args)
    mgr.reparameterize(train_loader)            # post-act BN(affine=False) + v_tilde=σW
    if use_ddp:
        from normalize_net import _broadcast_model_state
        _broadcast_model_state(model)
    acc, _ = validate(model, val_loader, device, args.model_type)
    log_info(f"NORMALIZE: {len(names)} layers, post-transform acc={acc:.4f} "
             f"(function-preserving — should match post-train)")

    # -- 2b. optional FT in normalized coordinates (train v_tilde; WD on σW = contrib reg)
    _run_phase(model, mgr, loaders, args, device, use_ddp,
               epochs=args.epochs_norm_ft, lr=args.lr_norm_ft, tag="NORM-FT", teacher=teacher)

    # -- 3. PRUNE (magnitude of normalized weight = NCI, via stock MagnitudePruner) ------
    if not args.no_prune:
        # DDP: σ buffers diverge per rank (local EMA); sync them, extract, then broadcast
        # the scores rank0→all so every rank prunes the SAME mask (else shape-mismatch hang).
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            mgr._sync_bn_stats()
        # PDF seed: propagation runs to the classifier (I^o = 𝟙 over classes, back through
        # W̄^fc). Pass the classifier so the final-stage features get real importance instead
        # of a uniform tie (which global pruning guts).
        clf = model.fc if hasattr(model, "fc") else None
        scores = extract_normnet_scores(
            mgr, args.scorer, example_inputs=(ex if args.scorer == "propagation" else None),
            relative=not args.prop_non_relative, classifier=clf)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for k in list(scores.keys()):
                t = scores[k].contiguous()
                torch.distributed.broadcast(t, src=0)
                scores[k] = t
        # score distribution (how many channels the criterion ranks near-zero, per layer
        # + global) — tells whether the ranking is concentrated or flat/degenerate.
        n0 = sum(int((s < 0.1).sum()) for s in scores.values())
        ntot = sum(s.numel() for s in scores.values())
        cvs = [float(s.std() / (s.mean() + 1e-12)) for s in scores.values() if s.numel() > 1]
        cv_med = sorted(cvs)[len(cvs) // 2] if cvs else 0.0
        log_info(f"scores ({args.scorer}): {ntot} channels, {n0} below 0.1 "
                 f"({100.0 * n0 / max(ntot, 1):.1f}%), median per-layer CV={cv_med:.3f}")

        mgr.merge_back()                        # back to plain modules for the tp pruner
        # Save the pre-prune (post-norm-ft, merged) DENSE net so a prune+FT can be retried
        # without redoing training: --checkpoint <preprune> --epochs_norm_ft 0 → normalize →
        # prune → FT. Plain state_dict (σ re-calibrated on reload). rank 0 only.
        if is_main() and not args.no_save_preprune:
            pp = os.path.join(args.save_dir, f"{args.save_tag}_preprune.pth")
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, pp)
            log_info(f"saved pre-prune checkpoint → {pp}")
        pre_w = _layer_widths(model)            # widths before the structural edit

        def _imp(mdl):
            return NormalizedNetImportance(mdl, scores, group_reduction="mean", normalizer="mean")

        def _ignored(mdl):
            return [mdl.fc] if hasattr(mdl, "fc") else []

        # MAC target overrides the raw channel ratio: 2G on RN50 (4.1G) ≈ 30% channels,
        # not 50% — channel-ratio ≠ MAC-ratio. Binary-search the global ratio whose
        # pruned MACs ≤ target_g, then prune the real model at it. Deterministic across
        # ranks (scores already broadcast), so DDP masks stay identical.
        ratio = args.pruning_ratio
        if args.mac_target_g > 0:
            ratio = _ratio_for_mac(model, ex, _imp, _ignored,
                                   args.mac_target_g, args.global_pruning)
            log_info(f"mac_target {args.mac_target_g:.2f}G → global pruning_ratio={ratio:.4f}")

        tp.pruner.MagnitudePruner(
            model, ex, importance=_imp(model), global_pruning=args.global_pruning,
            pruning_ratio=ratio, ignored_layers=_ignored(model)).step()
        model.to(device)
        per_layer_dist, global_kept = log_prune_distribution(pre_w, _layer_widths(model))
        if not args.no_bn_recalib:
            _recalibrate_bn(model, train_loader, device, max_batches=args.calib_batches)
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
                            "global_pruning": args.global_pruning, "target": tgt,
                            "global_ratio": round(ratio, 4), "pre_ft_val_acc": round(acc, 6),
                            "macs_g": round(pr_macs, 4), "macs_pct": round(100*pr_macs/dense_macs, 1),
                            "params_m": round(pr_params, 4), "global_kept_pct": global_kept,
                            "per_layer": per_layer_dist}, f, indent=2)
    else:
        mgr.merge_back()

    # -- 4. FINE-TUNE (plain post-prune / post-normalize recovery) ----------------------
    best = _run_phase(model, None, loaders, args, device, use_ddp,
                      epochs=args.epochs_ft, lr=args.lr_ft, tag="FINE-TUNE", teacher=teacher)

    # -- save (rank 0) -------------------------------------------------------------------
    from normalize_net import is_main
    if is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, f"{args.save_tag}.pth")
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)
        final_acc, _ = validate(model, val_loader, device, args.model_type)
        fmacs, fparams = _count(model, ex)
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
    p.add_argument("--checkpoint", default=None, help="None → random init (from scratch)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--exclude_stem", action="store_true")
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
    # 3. prune
    p.add_argument("--no_prune", action="store_true")
    p.add_argument("--scorer", default="per_layer", choices=["per_layer", "propagation"])
    p.add_argument("--prop_non_relative", action="store_true",
                   help="propagation: W̄=M^p (non-relative, drop column-normalizer D). "
                        "Default = relative W̄=M^p·D.")
    p.add_argument("--pruning_ratio", type=float, default=0.5)
    p.add_argument("--mac_target_g", type=float, default=0.0,
                   help="target MACs in GMAC (e.g. 2.0). >0 overrides --pruning_ratio: "
                        "binary-search the global ratio that hits it. 0 = use ratio.")
    p.add_argument("--global_pruning", action="store_true")
    p.add_argument("--no_bn_recalib", action="store_true")
    p.add_argument("--no_save_preprune", action="store_true",
                   help="skip saving the pre-prune dense checkpoint (default: save it)")
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
