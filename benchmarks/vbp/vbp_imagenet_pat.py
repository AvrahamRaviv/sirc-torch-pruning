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
import json
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_pruning as tp
from torch_pruning.utils.pruning_utils import Pruning

# Reuse infrastructure from vbp_imagenet
try:
    from .vbp_imagenet import (
        build_dataloaders, load_model, forward_logits, validate,
        setup_distributed, cleanup, is_main, log_info, setup_logging,
        build_ft_scheduler, VarianceConcentrationHooks, train_one_epoch,
        logger,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vbp_imagenet import (
        build_dataloaders, load_model, forward_logits, validate,
        setup_distributed, cleanup, is_main, log_info, setup_logging,
        build_ft_scheduler, VarianceConcentrationHooks, train_one_epoch,
        logger,
    )


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def build_vbp_layers_to_prune(model, model_type, architecture=None, interior_only=True):
    """Return list of module names that VBP should prune (fc1 / pwconv1 / interior convs)."""
    layers = []
    if model_type == "vit":
        for name, m in model.named_modules():
            if name.endswith(".intermediate.dense"):
                layers.append(name)
    elif model_type == "convnext":
        for name, m in model.named_modules():
            if hasattr(m, "pwconv1"):
                layers.append(name + ".pwconv1")
    elif model_type == "cnn":
        from torch_pruning.pruner.importance import build_cnn_ignored_layers
        import torch.nn as nn
        ignored = build_cnn_ignored_layers(model, architecture, interior_only)
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m not in ignored:
                layers.append(name)
    return layers


def build_pruning_config(args, model, config_dir):
    """Programmatically create pruning_config.json for the Pruning class."""
    layers = build_vbp_layers_to_prune(
        model, args.model_type,
        architecture=getattr(args, 'cnn_arch', None),
        interior_only=getattr(args, 'interior_only', True))

    # Sparse pre-training shifts the pruning start epoch
    start_epoch = args.epochs_sparse if args.sparse_mode != "none" else 0
    pat_total = args.epochs - start_epoch

    if args.pat_steps <= 1:
        # One-shot: single prune at start_epoch
        epoch_rate = 1
        end_epoch = start_epoch
    else:
        # PAT: distribute pat_steps across available epochs
        epoch_rate = max(1, pat_total // args.pat_steps)
        end_epoch = start_epoch + (args.pat_steps - 1) * epoch_rate

    config = {
        "channel_sparsity_args": {
            "is_prune": True,
            "pruning_method": "VBP",
            "global_pruning": args.global_pruning,
            "block_size": 1,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "epoch_rate": epoch_rate,
            "global_prune_rate": 1.0 - args.keep_ratio,
            "mac_target": args.keep_ratio if args.mac_target else 0.0,
            "max_pruning_rate": 0.95,
            "prune_channels_at_init": False,
            "infer": False,
            "input_shape": [1, 3, 224, 224],
            "layers": layers,
            "regularize": {"reg": 0, "mac_reg": 0},
            "MAC_params": {},
            # VBP-specific
            "model_type": args.model_type,
            "max_batches": args.max_batches,
            "var_loss_weight": args.var_loss_weight,
            "norm_per_layer": args.norm_per_layer,
            "verbose": 1,
            # Schedule and internal features
            "pruning_schedule": args.pruning_schedule,
            "bn_recalibration": args.model_type == "cnn",
            "sparse_mode": args.sparse_mode,
            "sparse_l1_lambda": args.l1_lambda,
            "sparse_gmp_target": args.gmp_target_sparsity,
        },
        "slice_sparsity_args": None,
    }

    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "pruning_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log_info(f"Pruning config written to {config_path}")
    log_info(f"  VBP layers: {len(layers)}, start={start_epoch}, end={end_epoch}, "
             f"epoch_rate={epoch_rate}, keep_ratio={args.keep_ratio:.3f}, "
             f"mac_target={'yes' if args.mac_target else 'no'}")
    if args.sparse_mode != "none":
        log_info(f"  Sparse: mode={args.sparse_mode}, start_epoch={start_epoch}")
    return config_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv):
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

    pruner = Pruning(model, config_dir, device=device)
    # Inject train_loader for VBP stats collection
    pruner.channel_pruner.train_loader = train_loader

    # Variance concentration hooks (optional)
    var_hooks = None
    if args.var_loss_weight > 0:
        cnn_target_layers = None
        if args.model_type == "cnn":
            from torch_pruning.pruner.importance import build_cnn_target_layers
            temp_DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
            cnn_target_layers = build_cnn_target_layers(model, temp_DG)
        var_hooks = VarianceConcentrationHooks(model, args.model_type,
                                               target_layers=cnn_target_layers)

    # Training loop with epoch-based pruning
    use_ddp = not args.disable_ddp and dist.is_initialized()

    # Pre-loop prune: one-shot prunes here; PAT does step 1
    pruner.prune(model, epoch=0, log=logger, mask_only=False)

    # Log retention accuracy after final pruning (one-shot finishes here)
    retention_logged = False
    if is_main() and not pruner.channel_pruner.prune_channels:
        acc_ret, _ = validate(model, val_loader, device, args.model_type)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
        log_info(f"Retention accuracy: {acc_ret:.4f} "
                 f"(MACs={pruned_macs / 1e9:.2f}G, Params={pruned_params / 1e6:.2f}M)")
        retention_logged = True

    # Optimizer + scheduler AFTER prune (parameters may have changed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler, step_per_batch = build_ft_scheduler(
        optimizer, args.epochs, len(train_loader))

    # Wrap in DDP after initial prune
    train_model = model
    if use_ddp:
        train_model = DDP(model, device_ids=[args.local_rank],
                          output_device=args.local_rank)
        dist.barrier()

    # Track pruning step for DDP rebuild detection
    if pruner.channel_pruner.pruning_schedule == 'geometric':
        prev_step = pruner.channel_pruner._geometric_step
    else:
        prev_step = pruner.channel_pruner.pruner.current_step if pruner.channel_pruner.pruner else 0

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(
            train_model, train_loader, train_sampler,
            optimizer, scheduler, device, epoch, args,
            teacher=teacher, step_per_batch=step_per_batch,
            phase="PAT", var_hooks=var_hooks,
            regularize_fn=pruner.channel_regularize,
        )

        # End-of-epoch prune (skip epoch 0, already done pre-loop)
        if epoch > 0:
            pruner.prune(model, epoch, log=logger, mask_only=False)

            # Log retention after final pruning step (PAT finishes here)
            if is_main() and not retention_logged and not pruner.channel_pruner.prune_channels:
                acc_ret, _ = validate(model, val_loader, device, args.model_type)
                pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
                log_info(f"Retention accuracy: {acc_ret:.4f} "
                         f"(MACs={pruned_macs / 1e9:.2f}G, Params={pruned_params / 1e6:.2f}M)")
                retention_logged = True

            # Re-wrap DDP + rebuild optimizer if pruning step advanced
            if pruner.channel_pruner.pruning_schedule == 'geometric':
                cur_step = pruner.channel_pruner._geometric_step
            else:
                cur_step = pruner.channel_pruner.pruner.current_step if pruner.channel_pruner.pruner else 0
            if cur_step > prev_step:
                prev_step = cur_step
                if use_ddp:
                    del train_model
                    train_model = DDP(model, device_ids=[args.local_rank],
                                      output_device=args.local_rank)
                optimizer = torch.optim.AdamW(train_model.parameters(),
                                              lr=args.lr, weight_decay=0.01)
                scheduler, step_per_batch = build_ft_scheduler(
                    optimizer, args.epochs - epoch - 1, len(train_loader))

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
    parser.add_argument("--model_type", default="vit", choices=["vit", "convnext", "cnn"])
    parser.add_argument("--model_name", default="/algo/NetOptimization/outputs/VBP/DeiT_tiny")
    parser.add_argument("--cnn_arch", default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v2"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--interior_only", action="store_true", default=True)

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
    parser.add_argument("--mac_target", action="store_true",
                        help="Use MAC-target mode: analytically compute channel ratio from keep_ratio")

    # PAT
    parser.add_argument("--pat_steps", type=int, default=5,
                        help="Number of prune steps (0 or 1 = one-shot, >1 = PAT)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--var_loss_weight", type=float, default=0.0)
    parser.add_argument("--pruning_schedule", default="geometric",
                        choices=["geometric", "linear"],
                        help="Pruning schedule type")

    # Sparse pre-training
    parser.add_argument("--sparse_mode", default="none",
                        choices=["l1_group", "gmp", "none"],
                        help="Sparse pre-training mode (none = skip)")
    parser.add_argument("--epochs_sparse", type=int, default=0,
                        help="Sparse pre-training epochs (shifts pruning start)")
    parser.add_argument("--l1_lambda", type=float, default=1e-4,
                        help="L2,1 regularization strength (l1_group mode)")
    parser.add_argument("--gmp_target_sparsity", type=float, default=0.5,
                        help="Target weight sparsity for GMP mode")

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


if __name__ == '__main__':
    main(sys.argv[1:])
