"""
VBP Pruning-Aware Training (PAT) via the Pruning class from pruning_utils.py.

This script demonstrates iterative VBP pruning using the standard Pruning
class interface (channel_pruning + slice_pruning), with epoch-based gradual
pruning and optional variance concentration loss.

Usage:
    # Single GPU
    python benchmarks/vbp/vbp_imagenet_pat.py \
        --model_type vit \
        --model_name google/vit-base-patch16-224 \
        --data_path /path/to/imagenet \
        --keep_ratio 0.65 \
        --global_pruning \
        --pat_steps 5 \
        --epochs 15 \
        --disable_ddp

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet_pat.py \
        --model_type vit \
        --model_name /path/to/deit_tiny \
        --data_path /path/to/imagenet \
        --keep_ratio 0.65 \
        --global_pruning \
        --pat_steps 5 \
        --epochs 15 \
        --use_kd
"""

import argparse
import copy
import datetime
import json
import logging
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_pruning as tp
from torch_pruning.utils.pruning_utils import Pruning

# Reuse infrastructure from vbp_imagenet
try:
    from .vbp_imagenet import (
        build_dataloaders, load_model, forward_logits, validate,
        setup_distributed, cleanup, is_main, log_info, setup_logging,
        build_ft_scheduler, VarianceConcentrationHooks, logger,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vbp_imagenet import (
        build_dataloaders, load_model, forward_logits, validate,
        setup_distributed, cleanup, is_main, log_info, setup_logging,
        build_ft_scheduler, VarianceConcentrationHooks, logger,
    )


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def build_vbp_layers_to_prune(model, model_type):
    """Return list of module names that VBP should prune (fc1 / pwconv1)."""
    layers = []
    if model_type == "vit":
        for name, m in model.named_modules():
            if name.endswith(".intermediate.dense"):
                layers.append(name)
    elif model_type == "convnext":
        for name, m in model.named_modules():
            if hasattr(m, "pwconv1"):
                layers.append(name + ".pwconv1")
    return layers


def build_pruning_config(args, model, config_dir):
    """Programmatically create pruning_config.json for the Pruning class."""
    layers = build_vbp_layers_to_prune(model, args.model_type)
    epochs_per_step = max(1, args.epochs // args.pat_steps)
    # epoch_rate: prune every N epochs (maps pat_steps to epoch schedule)
    epoch_rate = epochs_per_step

    config = {
        "channel_sparsity_args": {
            "is_prune": True,
            "pruning_method": "VBP",
            "global_pruning": args.global_pruning,
            "block_size": 1,
            "start_epoch": 0,
            "end_epoch": args.epochs - 1,
            "epoch_rate": epoch_rate,
            "global_prune_rate": 1.0 - args.keep_ratio,
            "max_pruning_rate": 0.95,
            "prune_channels_at_init": False,
            "infer": False,
            "input_shape": [1, 3, 224, 224],
            "layers": layers,
            "mac_target": 0.0,
            "regularize": {"reg": 0, "mac_reg": 0},
            "MAC_params": {},
            # VBP-specific
            "model_type": args.model_type,
            "max_batches": args.max_batches,
            "var_loss_weight": args.var_loss_weight,
            "norm_per_layer": args.norm_per_layer,
            "verbose": 1,
        },
        "slice_sparsity_args": None,
    }

    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "pruning_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log_info(f"Pruning config written to {config_path}")
    log_info(f"  VBP layers: {len(layers)}, epoch_rate={epoch_rate}, "
             f"total_prune_rate={1.0 - args.keep_ratio:.3f}")
    return config_dir


# ---------------------------------------------------------------------------
# Training loop (simplified, with var loss support)
# ---------------------------------------------------------------------------
def train_one_epoch_pat(model, train_loader, train_sampler, optimizer,
                        scheduler, device, epoch, args,
                        teacher=None, step_per_batch=True,
                        var_hooks=None):
    """Training epoch with optional KD and variance concentration loss."""
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    use_kd = args.use_kd and teacher is not None
    use_var_loss = var_hooks is not None and args.var_loss_weight > 0

    total_loss = 0.0
    total_var = 0.0
    num_batches = 0
    total = len(train_loader)
    log_interval = max(total // 20, 1)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = forward_logits(model, images, args.model_type)
        ce_loss = F.cross_entropy(logits, labels)

        if use_kd:
            with torch.no_grad():
                teacher_logits = forward_logits(teacher, images, args.model_type)
            kd_loss = F.kl_div(
                F.log_softmax(logits / args.kd_T, dim=1),
                F.softmax(teacher_logits / args.kd_T, dim=1),
                reduction="batchmean"
            ) * (args.kd_T ** 2)
            loss = args.kd_alpha * ce_loss + (1 - args.kd_alpha) * kd_loss
        else:
            loss = ce_loss

        if use_var_loss:
            var_loss = var_hooks.compute_loss()
            loss = loss + args.var_loss_weight * var_loss
            total_var += var_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step_per_batch:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if is_main() and (batch_idx % log_interval == 0 or batch_idx == total - 1):
            avg_loss = total_loss / num_batches
            parts = [f"loss={avg_loss:.4f}"]
            if use_var_loss:
                parts.append(f"var={total_var / num_batches:.4f}")
            log_info(f"Epoch {epoch+1} [{batch_idx+1}/{total}] {' '.join(parts)}")

    if not step_per_batch:
        scheduler.step()
    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Setup DDP or single GPU
    if args.disable_ddp:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        log_info("Running in single GPU mode")
    else:
        device = setup_distributed(args)

    setup_logging(args.save_dir)

    if not is_main():
        import warnings
        warnings.filterwarnings("ignore")

    if is_main():
        log_info("=" * 60)
        log_info("VBP PAT via Pruning class")
        log_info("=" * 60)
        for k, v in vars(args).items():
            logger.info(f"  {k}: {v}")

    # Build dataloaders
    train_loader, val_loader, train_sampler = build_dataloaders(
        args, use_ddp=not args.disable_ddp)

    # Load model
    model = load_model(args, device)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    # Baseline
    if is_main():
        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        log_info(f"Baseline: {base_macs / 1e9:.2f}G MACs, {base_params / 1e6:.2f}M params")

        acc_orig, _ = validate(model, val_loader, device, args.model_type)
        log_info(f"Original accuracy: {acc_orig:.4f}")
    else:
        base_macs = base_params = acc_orig = None

    # Teacher for KD
    teacher = None
    if args.use_kd:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        log_info("Created teacher model for KD")

    # Build pruning config and Pruning class
    config_dir = os.path.join(args.save_dir, "pruning_config")
    build_pruning_config(args, model, config_dir)

    output_transform = None
    if args.model_type == "vit":
        output_transform = lambda out: out.logits.sum()
    elif args.model_type == "convnext":
        output_transform = lambda out: out.sum()

    pruner = Pruning(model, config_dir, forward_fn=output_transform, device=device)
    # Inject train_loader for VBP stats collection
    pruner.channel_pruner.train_loader = train_loader

    # Variance concentration hooks (optional)
    var_hooks = None
    if args.var_loss_weight > 0:
        var_hooks = VarianceConcentrationHooks(model, args.model_type)

    # Training loop with epoch-based pruning
    use_ddp = not args.disable_ddp and dist.is_initialized()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler, step_per_batch = build_ft_scheduler(
        optimizer, args.epochs, len(train_loader))

    # Wrap in DDP
    train_model = model
    if use_ddp:
        train_model = DDP(model, device_ids=[args.local_rank],
                          output_device=args.local_rank)
        dist.barrier()

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Prune via Pruning class (handles epoch_rate scheduling internally)
        # Must prune on unwrapped model
        eval_model = train_model.module if isinstance(train_model, DDP) else train_model
        pruner.prune(eval_model, epoch, log=logger, mask_only=False)

        # Re-wrap in DDP if pruning changed the model structure
        if use_ddp and pruner.channel_pruner.reset_optimizer:
            del train_model
            train_model = DDP(model, device_ids=[args.local_rank],
                              output_device=args.local_rank)
            optimizer = torch.optim.AdamW(train_model.parameters(),
                                          lr=args.lr, weight_decay=0.01)
            scheduler, step_per_batch = build_ft_scheduler(
                optimizer, args.epochs - epoch, len(train_loader))
            pruner.channel_pruner.reset_optimizer = False

        # Train
        train_loss = train_one_epoch_pat(
            train_model, train_loader, train_sampler,
            optimizer, scheduler, device, epoch, args,
            teacher=teacher, step_per_batch=step_per_batch,
            var_hooks=var_hooks,
        )

        # Validate
        if is_main():
            eval_model = train_model.module if isinstance(train_model, DDP) else train_model
            acc, val_loss = validate(eval_model, val_loader, device, args.model_type)
            pruned_macs, pruned_params = tp.utils.count_ops_and_params(eval_model, example_inputs)
            log_info(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, "
                     f"val_acc={acc:.4f}, MACs={pruned_macs / 1e9:.2f}G")

            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(args.save_dir, "vbp_pat_best.pth")
                torch.save(eval_model.state_dict(), save_path)
                log_info(f"New best! Saved to {save_path}")

        if use_ddp:
            dist.barrier()

    # Cleanup var hooks
    if var_hooks is not None:
        var_hooks.remove()

    # Final summary
    if is_main():
        eval_model = train_model.module if isinstance(train_model, DDP) else train_model
        acc_final, _ = validate(eval_model, val_loader, device, args.model_type)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(eval_model, example_inputs)

        log_info("=" * 60)
        log_info("Summary")
        log_info("=" * 60)
        log_info(f"Base MACs:    {base_macs / 1e9:.2f}G -> Pruned: {pruned_macs / 1e9:.2f}G "
                 f"({pruned_macs / base_macs * 100:.1f}%)")
        log_info(f"Base Params:  {base_params / 1e6:.2f}M -> Pruned: {pruned_params / 1e6:.2f}M "
                 f"({pruned_params / base_params * 100:.1f}%)")
        log_info(f"Original Acc: {acc_orig:.4f}")
        log_info(f"Final Acc:    {acc_final:.4f}")
        log_info(f"Best Acc:     {best_acc:.4f}")

        save_path = os.path.join(args.save_dir, "vbp_pat_final.pth")
        torch.save(eval_model.state_dict(), save_path)
        log_info(f"Final model saved to {save_path}")

    if not args.disable_ddp:
        cleanup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="VBP PAT via Pruning class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model_type", default="vit", choices=["vit", "convnext"])
    parser.add_argument("--model_name", default="/algo/NetOptimization/outputs/VBP/DeiT_tiny")

    # Data
    parser.add_argument("--data_path", default="/algo/NetOptimization/outputs/VBP/")
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=200)

    # Pruning
    parser.add_argument("--keep_ratio", type=float, default=0.65)
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--norm_per_layer", action="store_true")

    # PAT
    parser.add_argument("--pat_steps", type=int, default=5,
                        help="Number of prune steps (maps to epoch_rate)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--var_loss_weight", type=float, default=0.0)

    # KD
    parser.add_argument("--use_kd", action="store_true")
    parser.add_argument("--kd_alpha", type=float, default=0.7)
    parser.add_argument("--kd_T", type=float, default=2.0)

    # DDP
    parser.add_argument("--disable_ddp", action="store_true")
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))

    # Output
    parser.add_argument("--save_dir", default="./output/vbp_pat")

    return parser.parse_args()


if __name__ == "__main__":
    main()
