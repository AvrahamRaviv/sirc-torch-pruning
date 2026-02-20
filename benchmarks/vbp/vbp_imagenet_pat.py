"""
Unified Pruning Pipeline via the Pruning class from pruning_utils.py.

One training loop, three phases managed by ChannelPruning:
  1. Sparse pre-training (optional): L2,1 / GMP / reparam
  2. PAT: iterative prune-then-train steps
  3. Post-prune fine-tuning

Epoch math:
    epoch_rate = max(1, pat_epochs_per_step + 1)
    start_epoch = epochs_sparse  (if sparse_mode != none)
    end_epoch   = start_epoch + (pat_steps - 1) * epoch_rate
    total       = end_epoch + 1 + epochs_ft

Supports multiple pruning criteria (variance/VBP, magnitude, LAMP, random).

Usage:
    # 5 PAT steps + 10 FT epochs (VBP, default)
    python benchmarks/vbp/vbp_imagenet_pat.py \
        --model_type vit \
        --model_name google/vit-base-patch16-224 \
        --data_path /path/to/imagenet \
        --keep_ratio 0.65 \
        --global_pruning \
        --pat_steps 5 \
        --epochs_ft 10 \
        --disable_ddp

    # Sparse (reparam) + PAT + FT
    python benchmarks/vbp/vbp_imagenet_pat.py \
        --sparse_mode reparam --epochs_sparse 3 \
        --pat_steps 5 --epochs_ft 10 \
        --keep_ratio 0.65 --global_pruning --disable_ddp \
        --model_type vit --model_name /path/to/deit_tiny \
        --data_path /path/to/imagenet
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

# Shared infrastructure
try:
    from .vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, validate,
        train_one_epoch, build_ft_scheduler,
        VarianceConcentrationHooks, build_layers_to_prune,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, validate,
        train_one_epoch, build_ft_scheduler,
        VarianceConcentrationHooks, build_layers_to_prune,
    )


_CRITERION_TO_METHOD = {
    "variance": "VBP",
    "magnitude": "GroupNormPruner",
    "lamp": "LAMP",
    "random": "Random",
}


def build_pruning_config(args, model, config_dir):
    """Programmatically create pruning_config.json for the Pruning class."""
    layers = build_layers_to_prune(
        model, args.model_type,
        architecture=getattr(args, 'cnn_arch', None),
        interior_only=getattr(args, 'interior_only', True))

    start_epoch = args.epochs_sparse if args.sparse_mode != "none" else 0
    epoch_rate = max(1, args.pat_epochs_per_step + 1)
    end_epoch = start_epoch + (args.pat_steps - 1) * epoch_rate

    config = {
        "channel_sparsity_args": {
            "is_prune": True,
            "pruning_method": _CRITERION_TO_METHOD[args.criterion],
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
            "no_compensation": args.no_compensation,
            "verbose": 1,
            # Schedule and features
            "pruning_schedule": args.pruning_schedule,
            "bn_recalibration": args.bn_recalibration,
            "sparse_mode": args.sparse_mode,
            "sparse_l1_lambda": args.l1_lambda,
            "sparse_gmp_target": args.gmp_target_sparsity,
            # Unified pipeline
            "epochs_ft": args.epochs_ft,
            "reparam_lambda": args.reparam_lambda,
        },
        "slice_sparsity_args": None,
    }

    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "pruning_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    total = end_epoch + 1 + args.epochs_ft
    log_info(f"Pruning config: {len(layers)} layers, start={start_epoch}, "
             f"end={end_epoch}, epoch_rate={epoch_rate}, total={total}, "
             f"keep_ratio={args.keep_ratio:.3f}, "
             f"mac_target={'yes' if args.mac_target else 'no'}")
    if args.sparse_mode != "none":
        log_info(f"  Sparse: mode={args.sparse_mode}, epochs={args.epochs_sparse}")
    return config_dir


# ---------------------------------------------------------------------------
# Optimizer with reparam-aware weight decay grouping
# ---------------------------------------------------------------------------
def build_optimizer(model, args, reparam_manager=None):
    """Build optimizer with optional param group splitting for reparam.

    When reparam is active, v and m parameters get weight_decay=0 (regularized
    via aux loss instead). Other parameters use standard weight decay.
    """
    params = list(model.parameters())
    if reparam_manager and reparam_manager.is_active:
        reparam_ids = reparam_manager.reparam_param_ids()
        base = [p for p in params if id(p) not in reparam_ids]
        reparam = [p for p in params if id(p) in reparam_ids]
        groups = [
            {"params": base, "weight_decay": args.wd},
            {"params": reparam, "weight_decay": 0.0},
        ]
    else:
        groups = [{"params": params, "weight_decay": args.wd}]

    if args.opt == "sgd":
        return torch.optim.SGD(groups, lr=args.lr, momentum=0.9)
    return torch.optim.AdamW(groups, lr=args.lr)


# ---------------------------------------------------------------------------
# DDP sync — broadcast rank 0's model after structural changes
# ---------------------------------------------------------------------------
def _broadcast_model_state(model):
    """Broadcast model parameters and buffers from rank 0.

    Called after prune/reparameterize so all DDP ranks have identical
    model state. Each rank may prune different channels (different stats
    from DistributedSampler), but rank 0's result is authoritative.
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf, src=0)


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
        log_info(f"Unified Pipeline (criterion={args.criterion})")
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

    pruner = Pruning(model, config_dir, device=device, train_loader=train_loader)
    cp = pruner.channel_pruner
    total = cp.total_epochs

    # Variance concentration hooks (optional, VBP only)
    var_hooks = None
    if args.criterion == "variance" and args.var_loss_weight > 0:
        cnn_target_layers = None
        if args.model_type == "cnn":
            from torch_pruning.pruner.importance import build_cnn_target_layers
            temp_DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
            cnn_target_layers = build_cnn_target_layers(model, temp_DG)
        var_hooks = VarianceConcentrationHooks(model, args.model_type,
                                               target_layers=cnn_target_layers)

    # --- Unified training loop ---
    use_ddp = not args.disable_ddp and dist.is_initialized()

    optimizer = build_optimizer(model, args, cp._reparam_manager)
    scheduler, step_per_batch = build_ft_scheduler(optimizer, total, len(train_loader))

    train_model = model
    if use_ddp:
        train_model = DDP(model, device_ids=[args.local_rank],
                          output_device=args.local_rank)
        dist.barrier()

    best_acc = 0.0
    for epoch in range(total):
        # 1. Prune / sparse lifecycle / no-op (phase decided internally)
        pruner.prune(model, epoch, log=logger, mask_only=False)

        # 2. Rebuild optimizer if model structure changed
        changed = cp.model_changed
        if use_ddp and changed:
            _broadcast_model_state(model)
        if changed:
            optimizer = build_optimizer(model, args, cp._reparam_manager)
            scheduler, step_per_batch = build_ft_scheduler(
                optimizer, total - epoch - 1, len(train_loader))
            if use_ddp:
                del train_model
                train_model = DDP(model, device_ids=[args.local_rank],
                                  output_device=args.local_rank)

        # 3. Train with phase-appropriate losses
        phase = cp.phase
        fc1 = cp._sparse_modules if phase == "Sparse" and args.sparse_mode == "l1_group" else None
        aux_fn = (cp._reparam_manager.regularization_loss
                  if phase == "Sparse" and cp._reparam_manager and cp._reparam_manager.is_active
                  else None)

        train_loss = train_one_epoch(
            train_model, train_loader, train_sampler,
            optimizer, scheduler, device, epoch, args,
            teacher=teacher, fc1_modules=fc1,
            step_per_batch=step_per_batch, phase=phase,
            var_hooks=var_hooks if phase == "PAT" else None,
            regularize_fn=pruner.channel_regularize,
            aux_loss_fn=aux_fn,
        )

        # 4. Validate + save best
        if is_main():
            eval_model = train_model.module if isinstance(train_model, DDP) else train_model
            acc, val_loss = validate(eval_model, val_loader, device, args.model_type)
            pruned_macs, pruned_params = tp.utils.count_ops_and_params(eval_model, example_inputs)
            log_info(f"[{phase}] Epoch {epoch+1}/{total}: train_loss={train_loss:.4f}, "
                     f"val_acc={acc:.4f}, MACs={pruned_macs / 1e9:.2f}G")

            if acc > best_acc:
                best_acc = acc
                save_path = os.path.join(args.save_dir, f"{args.criterion}_pat_best.pth")
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

        save_path = os.path.join(args.save_dir, f"{args.criterion}_pat_final.pth")
        torch.save(eval_model.state_dict(), save_path)
        log_info(f"Final model saved to {save_path}")

    if not args.disable_ddp:
        cleanup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified pruning pipeline (PAT/one-shot/sparse)",
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
    parser.add_argument("--criterion", default="variance",
                        choices=["variance", "magnitude", "lamp", "random"])
    parser.add_argument("--keep_ratio", type=float, default=0.65)
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument("--norm_per_layer", action="store_true")
    parser.add_argument("--mac_target", action="store_true",
                        help="Use MAC-target mode: analytically compute channel ratio from keep_ratio")
    parser.add_argument("--no_compensation", action="store_true",
                        help="Disable bias compensation after pruning")
    parser.add_argument("--bn_recalibration", action="store_true",
                        help="Recalibrate BN running stats after each pruning step")

    # PAT schedule
    parser.add_argument("--pat_steps", type=int, default=5,
                        help="Number of prune steps")
    parser.add_argument("--pat_epochs_per_step", type=int, default=0,
                        help="FT epochs between prune steps (0 = prune every epoch)")
    parser.add_argument("--epochs_ft", type=int, default=10,
                        help="Post-prune fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--wd", type=float, default=0.01,
                        help="Weight decay (applied to non-reparam params)")
    parser.add_argument("--var_loss_weight", type=float, default=0.0)
    parser.add_argument("--pruning_schedule", default="geometric",
                        choices=["geometric", "linear"])

    # Sparse pre-training
    parser.add_argument("--sparse_mode", default="none",
                        choices=["l1_group", "gmp", "reparam", "none"],
                        help="Sparse pre-training mode (none = skip)")
    parser.add_argument("--epochs_sparse", type=int, default=0,
                        help="Sparse pre-training epochs (shifts pruning start)")
    parser.add_argument("--l1_lambda", type=float, default=1e-4,
                        help="L2,1 regularization strength (l1_group mode)")
    parser.add_argument("--gmp_target_sparsity", type=float, default=0.5,
                        help="Target weight sparsity for GMP mode")
    parser.add_argument("--reparam_lambda", type=float, default=0.01,
                        help="L_{2,1} regularization strength for reparam mode")

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
