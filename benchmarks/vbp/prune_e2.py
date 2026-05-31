"""
E2 — prune a trained net by ‖σ·v‖ / propagation vs ‖v‖ magnitude, then short FT.

Tests paper §2 (‖σ·v‖ > ‖v‖) and §3-4 (propagation > per-layer) empirically:

    dense ckpt → [calibrate σ] → score channels → prune @ ratio → short FT → eval

Three scorers (--scorer), all share the SAME pruner / reduction / normalizer so only the
per-channel base score differs:
  - magnitude    : tp GroupMagnitudeImportance, ‖v‖           (the §2 baseline)
  - per_layer    : NormalizedNetImportance ‖σ·v‖              (§2 criterion)
  - propagation  : NormalizedNetImportance I^l (variance p=2) (§3-4 criterion)

Single-process (pruning is a structural edit — no DDP). KD teacher = the frozen dense
net (self-distillation). Uniform per-layer ratio by default (global_pruning=False) so the
sparsity is matched per layer and the comparison isolates ranking quality; pass
--global_pruning for the cross-layer (criterion-4) setting.

Example (one scorer, RN50):
  python benchmarks/vbp/prune_e2.py --model_type cnn --cnn_arch resnet50 \
    --checkpoint <dir>/<tag>_merged_biased.pth --data_path <imagenet> \
    --scorer per_layer --pruning_ratio 0.5 --epochs_ft 10 --use_kd \
    --save_dir <out> --save_tag E2_per_layer_50
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import torch
import torch_pruning as tp

from vbp_common import (
    is_main, setup_logging, build_dataloaders, load_model, validate,
    forward_logits, build_ft_scheduler, train_one_epoch,
    build_whole_net_reparam_layers,
)
from normalize_net import (
    load_normnet_checkpoint, log_info, get_device, append_metrics, write_run,
)
from torch_pruning.utils.reparam import MeanResidualManager
from torch_pruning.pruner.importance import GroupMagnitudeImportance
from normalized_net_importance import (
    NormalizedNetImportance, extract_input_channel_scores,
)


def _count(model, ex):
    macs, params = tp.utils.count_ops_and_params(model, ex)
    return macs / 1e9, params / 1e6


def build_scorer(scorer, model, calib_loader, device, args, ex):
    """Return a tp.Importance for the chosen scorer. For per_layer / propagation this
    calibrates σ on a fresh MeanResidualManager, extracts scores, then merges back so
    the pruner sees plain Conv/Linear."""
    if scorer == "magnitude":
        return GroupMagnitudeImportance(p=2, group_reduction="mean", normalizer="mean")

    names = build_whole_net_reparam_layers(
        model, exclude_classifier=True, exclude_stem=args.exclude_stem)
    log_info(f"σ calibration: {len(names)} reparam layers")
    mgr = MeanResidualManager(model, names, device, lambda_reg=0.0,
                              max_batches=args.calib_batches)
    mgr.reparameterize(calib_loader)
    mode = "per_layer" if scorer == "per_layer" else "propagation"
    scores = extract_input_channel_scores(
        mgr, mode=mode, example_inputs=(ex if mode == "propagation" else None), p=2)
    mgr.merge_back()
    n0 = sum(int((s < 0.1).sum()) for s in scores.values())
    ntot = sum(s.numel() for s in scores.values())
    log_info(f"scores extracted ({mode}): {ntot} channels, {n0} below 0.1 "
             f"({100.0 * n0 / max(ntot, 1):.1f}%)")
    return NormalizedNetImportance(model, scores, group_reduction="mean",
                                   normalizer="mean")


def main(argv):
    ap = argparse.ArgumentParser(description="E2: prune-by-score + short FT")
    # model / ckpt
    ap.add_argument("--model_type", default="cnn", choices=["vit", "convnext", "cnn"])
    ap.add_argument("--model_name", default="resnet50")
    ap.add_argument("--cnn_arch", default="resnet50")
    ap.add_argument("--checkpoint", required=True, help="merged_biased .pth (+ .meta.json)")
    ap.add_argument("--data_path", required=True)
    # scorer / pruning
    ap.add_argument("--scorer", required=True,
                    choices=["magnitude", "per_layer", "propagation"])
    ap.add_argument("--pruning_ratio", type=float, default=0.5)
    ap.add_argument("--global_pruning", action="store_true",
                    help="cross-layer global ranking (criterion-4 setting). "
                         "Default off = uniform per-layer ratio.")
    ap.add_argument("--exclude_stem", action="store_true",
                    help="don't reparam/score the stem conv")
    ap.add_argument("--calib_batches", type=int, default=50)
    # FT
    ap.add_argument("--epochs_ft", type=int, default=10)
    ap.add_argument("--opt", default="sgd")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--ft_eta_min", type=float, default=1e-6)
    ap.add_argument("--ft_warmup_epochs", type=int, default=0)
    # KD (teacher = frozen dense net)
    ap.add_argument("--use_kd", action="store_true")
    ap.add_argument("--kd_alpha", type=float, default=0.25)
    ap.add_argument("--kd_T", type=float, default=4.0)
    # data / loop knobs (consumed by build_dataloaders / train_one_epoch)
    ap.add_argument("--train_batch_size", type=int, default=128)
    ap.add_argument("--val_batch_size", type=int, default=256)
    ap.add_argument("--val_resize", type=int, default=232)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--sparse_mode", default="none")
    ap.add_argument("--var_loss_weight", type=float, default=0.0)
    # io
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--save_tag", required=True)
    ap.add_argument("--disable_ddp", action="store_true", default=True)
    args = ap.parse_args(argv[1:])

    # single-process only (pruning = structural edit)
    args.rank = 0; args.world_size = 1; args.local_rank = 0
    device = get_device()
    setup_logging(args.save_dir)
    log_info("=" * 60)
    log_info(f"E2 prune-and-FT: scorer={args.scorer} ratio={args.pruning_ratio} "
             f"global={args.global_pruning}")
    log_info("=" * 60)
    for k, v in vars(args).items():
        log_info(f"  {k}: {v}")

    train_loader, val_loader, _ = build_dataloaders(args, use_ddp=False)

    # Dense model from the merged_biased checkpoint. Null args.checkpoint so the inner
    # load_model() random-inits the bare arch; load_normnet_checkpoint's strict load
    # (after attach_biases) then fully populates weights + biases from the sidecar.
    ckpt_path = args.checkpoint
    args.checkpoint = None
    model = load_normnet_checkpoint(ckpt_path, device, args).to(device).eval()
    ex = torch.randn(1, 3, 224, 224).to(device)
    dense_macs, dense_params = _count(model, ex)
    dense_acc, _ = validate(model, val_loader, device, args.model_type)
    log_info(f"DENSE: acc={dense_acc:.4f}  {dense_macs:.2f}G MACs  {dense_params:.2f}M params")

    # KD teacher = frozen dense snapshot (before pruning).
    teacher = None
    if args.use_kd:
        teacher = copy.deepcopy(model).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    write_run(args, {"arm": f"prune_{args.scorer}", "status": "running",
                     "config": vars(args), "dense_val_acc": dense_acc,
                     "dense_macs_g": dense_macs, "dense_params_m": dense_params})

    # Build scorer (calibrates σ for per_layer/propagation), then prune.
    importance = build_scorer(args.scorer, model, train_loader, device, args, ex)
    ignored = [model.fc] if hasattr(model, "fc") else []
    pruner = tp.pruner.MagnitudePruner(
        model, ex, importance=importance, global_pruning=args.global_pruning,
        pruning_ratio=args.pruning_ratio, ignored_layers=ignored)
    pruner.step()
    model.to(device)

    pr_macs, pr_params = _count(model, ex)
    acc_pruned, _ = validate(model, val_loader, device, args.model_type)
    log_info(f"PRUNED (pre-FT): acc={acc_pruned:.4f}  {pr_macs:.2f}G MACs "
             f"({100*pr_macs/dense_macs:.1f}%)  {pr_params:.2f}M params "
             f"({100*pr_params/dense_params:.1f}%)  Δacc={acc_pruned-dense_acc:+.4f}")
    append_metrics(args, {"arm": f"prune_{args.scorer}", "epoch": 0,
                          "epochs": args.epochs_ft, "val_acc": round(acc_pruned, 6),
                          "stage": "post_prune", "macs_g": round(pr_macs, 4),
                          "params_m": round(pr_params, 4)})

    # Short FT (KD from frozen dense teacher).
    params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.wd)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = len(train_loader)
    scheduler, step_per_batch = build_ft_scheduler(
        optimizer, args.epochs_ft, steps, eta_min=args.ft_eta_min,
        warmup_epochs=args.ft_warmup_epochs)

    best = acc_pruned
    for epoch in range(args.epochs_ft):
        train_loss, _ = train_one_epoch(
            model, train_loader, None, optimizer, scheduler, device, epoch, args,
            teacher=teacher, step_per_batch=step_per_batch, phase="E2-FT")
        acc, val_loss = validate(model, val_loader, device, args.model_type)
        best = max(best, acc)
        cur_lr = optimizer.param_groups[0]["lr"]
        log_info(f"[E2-FT] Epoch {epoch+1}/{args.epochs_ft}: train_loss={train_loss:.4f} "
                 f"val_acc={acc:.4f} best={best:.4f} lr={cur_lr:.2e}")
        append_metrics(args, {"arm": f"prune_{args.scorer}", "epoch": epoch + 1,
                              "epochs": args.epochs_ft, "train_loss": round(train_loss, 6),
                              "val_acc": round(acc, 6), "val_loss": round(val_loss, 6),
                              "best_val_acc": round(best, 6), "lr": cur_lr})

    log_info(f"RESULT scorer={args.scorer} ratio={args.pruning_ratio} "
             f"dense={dense_acc:.4f} pruned_preFT={acc_pruned:.4f} bestFT={best:.4f} "
             f"Δvs_dense={best-dense_acc:+.4f} MACs={pr_macs:.2f}G")
    write_run(args, {"arm": f"prune_{args.scorer}", "status": "done",
                     "config": vars(args), "dense_val_acc": dense_acc,
                     "dense_macs_g": dense_macs, "dense_params_m": dense_params,
                     "pruned_preft_val_acc": acc_pruned, "best_val_acc": best,
                     "pruned_macs_g": pr_macs, "pruned_params_m": pr_params,
                     "pruning_ratio": args.pruning_ratio, "scorer": args.scorer})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
