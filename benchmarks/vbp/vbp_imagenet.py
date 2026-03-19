"""
VBP (Variance-Based Pruning) ImageNet Reproduction Script

This script reproduces results from the VBP paper using the integrated
VarianceImportance class in Torch-Pruning, with DDP support.
Supports multiple importance criteria for comparison (VBP, magnitude, LAMP, random).

Reference: https://arxiv.org/pdf/2507.12988

Usage:
    # Single GPU — VBP (default)
    python benchmarks/vbp/vbp_imagenet.py \
        --model_type vit \
        --model_name google/vit-base-patch16-224 \
        --data_path /path/to/imagenet \
        --keep_ratio 0.65 \
        --global_pruning \
        --disable_ddp

    # Single GPU — magnitude baseline
    python benchmarks/vbp/vbp_imagenet.py \
        --model_type vit \
        --model_name google/vit-base-patch16-224 \
        --data_path /path/to/imagenet \
        --keep_ratio 0.65 \
        --global_pruning \
        --disable_ddp \
        --criterion magnitude

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet.py \
        --model_type vit \
        --model_name /path/to/deit_tiny \
        --data_path /path/to/imagenet \
        --keep_ratio 0.65 \
        --global_pruning \
        --epochs_ft 10 \
        --use_kd
"""

import argparse
import copy
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_pruning as tp

from torch_pruning.utils.sparse_utils import (
    get_fc1_modules, l21_regularization, gmp_sparsity_schedule,
    apply_unstructured_pruning, remove_pruning_reparametrization,
    compute_variance_entropy, compute_weight_sparsity,
)

# Shared infrastructure (extracted to vbp_common)
try:
    from .vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, forward_logits,
        validate, train_one_epoch, build_cosine_scheduler,
        VarianceConcentrationHooks, build_layers_to_prune, build_reparam_layers,
        build_consumer_weight_map, build_producer_weight_map,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vbp_common import (
        logger, is_main, log_info, setup_logging,
        setup_distributed, cleanup,
        build_dataloaders, load_model, forward_logits,
        validate, train_one_epoch, build_cosine_scheduler,
        VarianceConcentrationHooks, build_layers_to_prune, build_reparam_layers,
        build_consumer_weight_map, build_producer_weight_map,
    )

# ---------------------------------------------------------------------------
# ConvNeXt VBP helpers
# ---------------------------------------------------------------------------
def _make_post_gelu_nchw(act_fn):
    """Post-act fn: apply GELU then permute NHWC -> NCHW for stats hook."""
    def fn(x):
        return act_fn(x).permute(0, 3, 1, 2)
    return fn


# ---------------------------------------------------------------------------
# V-norm vs VBP correlation
# ---------------------------------------------------------------------------
def _log_vnorm_vbp_correlation(vnorm_path, model, imp, model_type, architecture=None):
    """Log Spearman rank correlation between V-norms and VBP importance.

    Both are per input-channel (d_intermediate) vectors — direct 1:1 mapping.
    For ViT: fc2 V-norm ↔ fc1 VBP importance for same block.
    """
    from scipy.stats import spearmanr

    vnorms = torch.load(vnorm_path, map_location="cpu", weights_only=True)

    # Build module→name map for VBP importance
    module_to_name = {m: n for n, m in model.named_modules()}

    correlations = []
    for fc2_name, v_col_norms in vnorms.items():
        # Map fc2 → fc1: "vit.encoder.layer.N.output.dense" → "vit.encoder.layer.N.intermediate.dense"
        if model_type == "vit":
            fc1_name = fc2_name.replace(".output.dense", ".intermediate.dense")
        elif model_type == "convnext":
            fc1_name = fc2_name.replace("pwconv2", "pwconv1")
        elif model_type == "cnn":
            if architecture and "mobilenet" in architecture.lower():
                # features.N.conv.2 → features.N.conv.0.0 (projection → expand)
                parts = fc2_name.rsplit('.conv.', 1)
                if len(parts) != 2:
                    continue
                fc1_name = parts[0] + ".conv.0.0"
            else:
                # ResNet: layerX.N.conv3 → layerX.N.conv1
                fc1_name = fc2_name.replace("conv3", "conv1")
        else:
            continue

        # Find fc1 module in imp.variance
        fc1_mod = None
        for mod, name in module_to_name.items():
            if name == fc1_name:
                fc1_mod = mod
                break
        if fc1_mod is None or fc1_mod not in imp.variance:
            continue

        vbp_imp = imp.variance[fc1_mod].cpu()
        if vbp_imp.shape != v_col_norms.shape:
            continue

        rho, pval = spearmanr(v_col_norms.numpy(), vbp_imp.numpy())
        correlations.append(rho)
        short = fc2_name.split('.')[-2] + '.' + fc2_name.split('.')[-1]
        log_info(f"V-norm↔VBP {short}: spearman_r={rho:.4f} (p={pval:.2e})")

    if correlations:
        import numpy as np
        log_info(f"V-norm↔VBP aggregate: mean_r={np.mean(correlations):.4f}, "
                 f"median_r={np.median(correlations):.4f}")


# ---------------------------------------------------------------------------
# Sparse pre-training
# ---------------------------------------------------------------------------
def run_sparse_pretraining(model, teacher, train_loader, train_sampler,
                           val_loader, device, args):
    """Optional sparse pre-training stage before VBP stats collection.

    Supports DDP: wraps model for training, unwraps after.
    Modifies model in-place. Uses KD from teacher if --use_kd is set.
    """
    fc1_pairs = get_fc1_modules(
        model, model_type=args.model_type,
        cnn_arch=getattr(args, 'cnn_arch', None))
    fc1_modules = [m for _, m in fc1_pairs]
    log_info(f"Sparse pre-training: mode={args.sparse_mode}, "
             f"{len(fc1_modules)} fc1 layers, epochs={args.epochs_sparse}")

    # Wrap in DDP for sparse training
    use_ddp = not args.disable_ddp and dist.is_initialized()
    if use_ddp:
        train_model = DDP(model, device_ids=[args.local_rank],
                          output_device=args.local_rank)
    else:
        train_model = model

    optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr_sparse,
                                  weight_decay=0.01)
    scheduler, step_per_batch = build_cosine_scheduler(
        optimizer, args.epochs_sparse, len(train_loader))

    for epoch in range(args.epochs_sparse):
        # GMP: apply sparsity mask before training epoch
        if args.sparse_mode == "gmp":
            target_s = gmp_sparsity_schedule(
                epoch, args.epochs_sparse,
                init_s=0.0, target_s=args.gmp_target_sparsity)
            apply_unstructured_pruning(fc1_modules, target_s)
            ws = compute_weight_sparsity(fc1_modules)
            log_info(f"GMP epoch {epoch+1}: target={target_s:.4f}, "
                     f"actual={ws['global']:.4f}")

        train_loss, aux_losses = train_one_epoch(
            train_model, train_loader, train_sampler, optimizer, scheduler,
            device, epoch, args, teacher=teacher, fc1_modules=fc1_modules,
            step_per_batch=step_per_batch, phase="Sparse")

        # Validate on unwrapped model
        if is_main():
            acc, val_loss = validate(model, val_loader, device, args.model_type)
            aux_str = " ".join(f"{k}={v:.4f}" for k, v in aux_losses.items()) if aux_losses else ""
            log_info(f"Sparse {epoch+1}/{args.epochs_sparse}: "
                     f"train_loss={train_loss:.4f}, val_acc={acc:.4f}"
                     f"{' | ' + aux_str if aux_str else ''}")
            if acc < 0.01:
                log_info("WARNING: Model accuracy collapsed below 1%!")

        if use_ddp:
            dist.barrier()

    # Cleanup DDP wrapper
    if use_ddp:
        del train_model

    # GMP: bake masks into weights
    if args.sparse_mode == "gmp":
        remove_pruning_reparametrization(fc1_modules)
        ws = compute_weight_sparsity(fc1_modules)
        log_info(f"GMP done — final weight sparsity: {ws['global']:.4f}")
        for m in fc1_modules:
            assert not hasattr(m, "weight_mask"), "weight_mask still present"

    # Post-sparse accuracy
    if is_main():
        acc_sparse, _ = validate(model, val_loader, device, args.model_type)
        log_info(f"Post-sparse accuracy: {acc_sparse:.4f}")


def run_reparam_pretraining(model, teacher, train_loader, train_sampler,
                            val_loader, device, args):
    """Mean-residual reparameterization sparse phase.

    Decomposes target layers into m + V^T(x - μ_x), then trains with L_{2,1}
    regularization on V to drive activation variance toward zero before pruning.
    Merges back to standard modules after training.
    """
    from torch_pruning.utils.reparam import MeanResidualManager

    reparam_target = getattr(args, 'reparam_target', 'fc2')
    target_names = build_reparam_layers(
        model, args.model_type,
        architecture=getattr(args, 'cnn_arch', None),
        reparam_target=reparam_target)
    log_info(f"Reparam pre-training: {len(target_names)} {reparam_target} layers, "
             f"epochs={args.epochs_sparse}, λ={args.reparam_lambda}")

    mgr = MeanResidualManager(
        model, target_names, device,
        lambda_reg=args.reparam_lambda,
        max_batches=args.max_batches,
        normalize=getattr(args, 'reparam_normalize', False),
        reparam_target=reparam_target)
    mgr.reparameterize(train_loader)

    # Wrap in DDP for training
    use_ddp = not args.disable_ddp and dist.is_initialized()
    if use_ddp:
        train_model = DDP(model, device_ids=[args.local_rank],
                          output_device=args.local_rank)
    else:
        train_model = model

    # Exclude reparam params from weight decay
    reparam_ids = mgr.reparam_param_ids()
    base_params = [p for p in train_model.parameters() if id(p) not in reparam_ids]
    reparam_params = [p for p in train_model.parameters() if id(p) in reparam_ids]
    optimizer = torch.optim.AdamW([
        {"params": base_params, "weight_decay": 0.01},
        {"params": reparam_params, "weight_decay": 0.0},
    ], lr=args.lr_sparse)
    scheduler, step_per_batch = build_cosine_scheduler(
        optimizer, args.epochs_sparse, len(train_loader))

    def _reparam_aux():
        return {"reg": mgr.regularization_loss()}

    for epoch in range(args.epochs_sparse):
        train_loss, aux_losses = train_one_epoch(
            train_model, train_loader, train_sampler, optimizer, scheduler,
            device, epoch, args, teacher=teacher,
            step_per_batch=step_per_batch, phase="Reparam",
            aux_loss_fn=_reparam_aux)

        # Periodic μ_x refresh (function-preserving)
        refresh = getattr(args, 'reparam_refresh_interval', 0)
        if refresh > 0 and (epoch + 1) % refresh == 0:
            mgr.refresh_mu(train_loader)

        if is_main():
            acc, _ = validate(model, val_loader, device, args.model_type)
            aux_str = " ".join(f"{k}={v:.4f}" for k, v in aux_losses.items()) if aux_losses else ""
            log_info(f"Reparam {epoch+1}/{args.epochs_sparse}: "
                     f"train_loss={train_loss:.4f}, val_acc={acc:.4f}"
                     f"{' | ' + aux_str if aux_str else ''}")
            mgr.log_channel_stats()

        if use_ddp:
            dist.barrier()

    # Cleanup DDP wrapper
    if use_ddp:
        del train_model

    # Save V-norm snapshot, validate pre/post merge-back for numerical verification
    if is_main():
        mgr.save_vnorm_snapshot(args.save_dir)
        acc_pre, _ = validate(model, val_loader, device, args.model_type)

    mgr.merge_back()

    if is_main():
        acc_post, _ = validate(model, val_loader, device, args.model_type)
        log_info(f"Merge-back delta: pre={acc_pre:.4f}, post={acc_post:.4f}, "
                 f"Δ={acc_post - acc_pre:+.4f}")
        if abs(acc_post - acc_pre) > 1e-4:
            log_info("WARNING: merge-back delta exceeds 1e-4 — possible numerical issue")


# ---------------------------------------------------------------------------
# Pruner setup
# ---------------------------------------------------------------------------
def build_importance(criterion, norm_per_layer=False, importance_mode="variance",
                     wv_base_mode="weight_variance", mag_guided_delta=0.2):
    """Map criterion string to importance object."""
    if criterion == "variance":
        return tp.importance.VarianceImportance(norm_per_layer=norm_per_layer,
                                                 importance_mode=importance_mode,
                                                 wv_base_mode=wv_base_mode,
                                                 mag_guided_delta=mag_guided_delta)
    elif criterion == "magnitude":
        return tp.importance.MagnitudeImportance(p=2)
    elif criterion == "lamp":
        return tp.importance.LAMPImportance(p=2)
    elif criterion == "random":
        return tp.importance.RandomImportance()
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def create_pruner(model, example_inputs, imp, args):
    """Create pruner: VBPPruner for variance criterion, BasePruner for others."""
    ignored_layers = []

    if args.model_type == "vit":
        # MLP-only pruning: ignore everything except fc1 (intermediate.dense)
        ignored_layers.append(model.classifier)
        ignored_layers.append(model.vit.embeddings.patch_embeddings.projection)

        for block in model.vit.encoder.layer:
            # Attention layers — don't prune
            ignored_layers.append(block.attention.attention.query)
            ignored_layers.append(block.attention.attention.key)
            ignored_layers.append(block.attention.attention.value)
            ignored_layers.append(block.attention.output.dense)
            # fc2 output channels are the residual stream — don't prune
            ignored_layers.append(block.output.dense)

        output_transform = lambda out: out.logits.sum()

    elif args.model_type == "convnext":
        # MLP-only: ignore everything except pwconv1 (intermediate dim)
        ignored_layers.append(model.head)
        for ds in model.downsample_layers:
            for m in ds.modules():
                if isinstance(m, nn.Conv2d):
                    ignored_layers.append(m)
        for stage in model.stages:
            for block in stage:
                ignored_layers.append(block.dwconv)
                ignored_layers.append(block.pwconv2)
        output_transform = lambda out: out.sum()

    elif args.model_type == "cnn":
        from torch_pruning.pruner.importance import build_cnn_ignored_layers
        ignored_layers = build_cnn_ignored_layers(
            model, args.cnn_arch, interior_only=args.interior_only)
        output_transform = lambda out: out.sum()

    else:
        output_transform = lambda out: out.sum()

    is_vbp = getattr(args, 'criterion', 'variance') == 'variance'
    if is_vbp:
        pruner = tp.pruner.VBPPruner(
            model,
            example_inputs,
            importance=imp,
            global_pruning=args.global_pruning,
            pruning_ratio=1.0 - args.keep_ratio,
            max_pruning_ratio=getattr(args, 'max_pruning_ratio', 1.0),
            ignored_layers=ignored_layers,
            output_transform=output_transform,
            mean_dict=imp.means,
            verbose=is_main(),
        )
    else:
        pruner = tp.pruner.BasePruner(
            model,
            example_inputs,
            importance=imp,
            global_pruning=args.global_pruning,
            pruning_ratio=1.0 - args.keep_ratio,
            max_pruning_ratio=getattr(args, 'max_pruning_ratio', 1.0),
            ignored_layers=ignored_layers,
            output_transform=output_transform,
            verbose=is_main(),
        )

    return pruner


# ---------------------------------------------------------------------------
# Pruning with VBP compensation
# ---------------------------------------------------------------------------
def recalibrate_bn(model, loader, device, max_batches=100):
    """Reset and recalibrate BN running stats on the pruned model.

    Essential for CNNs after structured pruning. Needs 1000+ samples
    (MobileNetV2 needs ~5000 for full recovery).

    Args:
        max_batches: Max batches to use (default 100 ≈ 6400 samples at bs=64).
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
    model.train()
    total = min(max_batches, len(loader))
    log_interval = max(total // 20, 1)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            from torch_pruning.utils.pruning_utils import _unpack_images
            images = _unpack_images(batch)
            model(images.to(device, non_blocking=True))
            if is_main() and (batch_idx % log_interval == 0 or batch_idx == total - 1):
                log_info(f"BN recalib [{batch_idx+1}/{total}]")
    model.eval()
    log_info(f"BN recalibration done ({total} batches)")


def prune_model(model, pruner, device, example_inputs,
                enable_compensation=True, is_vbp=True):
    """
    Apply pruning. For VBP (is_vbp=True), uses bias compensation via VBPPruner.
    For other criteria (is_vbp=False), uses BasePruner without compensation.
    """
    if is_vbp and enable_compensation:
        # Cache consumer inputs for compensation
        pruner.enable_meancheck(model)
        model.eval()
        with torch.no_grad():
            model(example_inputs)

    if is_vbp:
        pruner.step(interactive=False, enable_compensation=enable_compensation)
    else:
        pruner.step(interactive=False)

    if is_vbp and enable_compensation:
        pruner.disable_meancheck()
    log_info(f"Pruning complete (criterion={'VBP' if is_vbp else 'non-VBP'}, "
             f"compensation={'on' if is_vbp and enable_compensation else 'off'})")


# ---------------------------------------------------------------------------
# Statistics collection and DDP sync
# ---------------------------------------------------------------------------
def collect_and_sync_stats(model, train_loader, device, imp, args):
    """
    Collect variance statistics on rank 0 and broadcast to all ranks.
    """
    use_ddp = not args.disable_ddp and dist.is_initialized()

    # Collect statistics on main rank
    if is_main():
        # Build target_layers for post-GELU stats (fc1 only)
        target_layers = None
        if args.model_type == "vit":
            target_layers = [
                (block.intermediate.dense, block.intermediate.intermediate_act_fn)
                for block in model.vit.encoder.layer
            ]
        elif args.model_type == "convnext":
            target_layers = [
                (block.pwconv1, _make_post_gelu_nchw(block.act))
                for stage in model.stages for block in stage
            ]
        elif args.model_type == "cnn":
            from torch_pruning.pruner.importance import build_cnn_target_layers
            example = torch.randn(1, 3, 224, 224).to(device)
            temp_DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
            target_layers = build_cnn_target_layers(model, temp_DG)
            log_info(f"Auto-detected {len(target_layers)} CNN target layers for stats")
        log_info("Collecting activation variance statistics...")
        imp.collect_statistics(model, train_loader, device, target_layers=target_layers, max_batches=args.max_batches)
        log_info(f"Statistics collected for {len(imp.variance)} layers")

        # Debug: print variance stats
        for mod, var in list(imp.variance.items())[:5]:
            log_info(f"  {mod.__class__.__name__}: mean_var={var.mean().item():.6f}")

    # Sync across ranks
    if use_ddp:
        dist.barrier()

        # Build name→module map (same structure on all ranks)
        name_to_module = {n: m for n, m in model.named_modules()}
        module_to_name = {m: n for n, m in model.named_modules()}

        # Package as {name: cpu_tensor} — no module objects, no CUDA tensors
        if is_main():
            stats_dict = {
                "variance": {module_to_name[m]: v.cpu() for m, v in imp.variance.items()
                             if m in module_to_name},
                "means": {module_to_name[m]: v.cpu() for m, v in imp.means.items()
                           if m in module_to_name},
            }
        else:
            stats_dict = None

        stats_list = [stats_dict]
        dist.broadcast_object_list(stats_list, src=0)

        # Unpack on non-main ranks
        if not is_main():
            stats_dict = stats_list[0]
            for name, var in stats_dict["variance"].items():
                if name in name_to_module:
                    mod = name_to_module[name]
                    imp.variance[mod] = var.to(device)
                    if name in stats_dict["means"]:
                        imp.means[mod] = stats_dict["means"][name].to(device)

        dist.barrier()


# ---------------------------------------------------------------------------
# Pruning-Aware Training (PAT)
# ---------------------------------------------------------------------------
def finetune(model, teacher, train_loader, train_sampler, val_loader,
             device, args, epochs, epoch_offset=0, phase="FT",
             use_var_loss=False, aux_loss_fn=None, reparam_manager=None):
    """Fine-tune model for a given number of epochs.

    Args:
        aux_loss_fn: Optional callable returning a scalar tensor (e.g. reparam reg).
        reparam_manager: If provided, exclude its params from weight decay.

    Returns:
        best_acc achieved during fine-tuning (0.0 if epochs=0).
    """
    if epochs <= 0:
        return 0.0

    use_ddp = not args.disable_ddp and dist.is_initialized()

    train_model = model
    if use_ddp:
        train_model = DDP(model, device_ids=[args.local_rank],
                          output_device=args.local_rank)
        dist.barrier()

    # Build optimizer with optional reparam param group splitting
    wd = args.wd_ft if args.wd_ft is not None else (1e-4 if args.opt_ft == "sgd" else 0.01)
    params = list(train_model.parameters())
    if reparam_manager and reparam_manager.is_active:
        reparam_ids = reparam_manager.reparam_param_ids()
        base = [p for p in params if id(p) not in reparam_ids]
        reparam_p = [p for p in params if id(p) in reparam_ids]
        groups = [{"params": base, "weight_decay": wd},
                  {"params": reparam_p, "weight_decay": 0.0}]
    else:
        groups = [{"params": params, "weight_decay": wd}]

    if args.opt_ft == "sgd":
        optimizer = torch.optim.SGD(groups, lr=args.lr_ft, momentum=args.momentum_ft)
    else:
        optimizer = torch.optim.AdamW(groups, lr=args.lr_ft)
    scheduler, step_per_batch = build_cosine_scheduler(
        optimizer, epochs, len(train_loader))

    var_hooks = None
    if use_var_loss and args.var_loss_weight > 0:
        cnn_target_layers = None
        if args.model_type == "cnn":
            from torch_pruning.pruner.importance import build_cnn_target_layers
            example = torch.randn(1, 3, 224, 224).to(device)
            temp_DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
            cnn_target_layers = build_cnn_target_layers(model, temp_DG)
        var_hooks = VarianceConcentrationHooks(model, args.model_type,
                                               target_layers=cnn_target_layers)

    best_acc = 0.0
    for ep in range(epochs):
        global_epoch = epoch_offset + ep
        train_loss, aux_losses = train_one_epoch(
            train_model, train_loader, train_sampler,
            optimizer, scheduler, device, global_epoch, args,
            teacher=teacher, step_per_batch=step_per_batch,
            phase=phase, var_hooks=var_hooks,
            aux_loss_fn=aux_loss_fn,
        )

        # Log reparam stats if active
        if is_main() and reparam_manager and reparam_manager.is_active:
            reparam_manager.log_channel_stats()

        if is_main():
            eval_model = train_model.module if isinstance(train_model, DDP) else train_model
            acc_ft, loss_ft = validate(eval_model, val_loader, device, args.model_type)
            aux_str = " ".join(f"{k}={v:.4f}" for k, v in aux_losses.items()) if aux_losses else ""
            log_info(f"{phase} ep {ep+1}/{epochs}: "
                     f"train_loss={train_loss:.4f}, val_acc={acc_ft:.4f}"
                     f"{' | ' + aux_str if aux_str else ''}")

            if acc_ft > best_acc:
                best_acc = acc_ft
                save_path = os.path.join(args.save_dir, "vbp_best.pth")
                torch.save(eval_model.state_dict(), save_path)
                log_info(f"New best! Saved to {save_path}")

        if use_ddp:
            dist.barrier()

    if var_hooks is not None:
        var_hooks.remove()
    if use_ddp:
        del train_model

    return best_acc


def run_pat(model, teacher, train_loader, train_sampler, val_loader,
            device, example_inputs, args,
            base_macs=None, base_params=None, acc_orig=None):
    """Prune (one-shot or iterative) then optionally fine-tune.

    Pipeline: [pat_steps x (collect stats -> prune -> per-step FT)] -> post-prune FT
    Geometric schedule: per_step_keep^pat_steps = keep_ratio.
    """
    pat_steps = args.pat_steps
    epochs_per_step = args.pat_epochs_per_step
    per_step_keep = args.keep_ratio ** (1.0 / pat_steps)
    per_step_prune = 1.0 - per_step_keep

    is_vbp = args.criterion == "variance"
    log_info(f"PAT: {pat_steps} steps, per_step_keep={per_step_keep:.4f}, "
             f"epochs_per_step={epochs_per_step}, target_keep={args.keep_ratio}, "
             f"criterion={args.criterion}")

    # Track cumulative keep ratio for logging
    cumulative_keep = 1.0
    best_acc = 0.0

    for step_i in range(pat_steps):
        log_info(f"\n{'='*60}")
        log_info(f"PAT Step {step_i+1}/{pat_steps}")
        log_info(f"{'='*60}")

        # 1. Create importance and collect stats (VBP only)
        imp_mode = getattr(args, 'importance_mode', 'variance')
        imp = build_importance(args.criterion, norm_per_layer=args.norm_per_layer,
                               importance_mode=imp_mode,
                               wv_base_mode=getattr(args, 'wv_base_mode', 'weight_variance'),
                               mag_guided_delta=getattr(args, 'mag_guided_delta', 0.2))
        if is_vbp:
            collect_and_sync_stats(model, train_loader, device, imp, args)
            # Determine if consumer/producer weight norms are needed
            wv_base = getattr(args, 'wv_base_mode', 'weight_variance')
            _needs_consumer = imp_mode in ('weight_variance', 'weight_variance_both') or (
                imp_mode in ('rank_fusion', 'mag_guided') and wv_base in ('weight_variance', 'weight_variance_both'))
            _needs_producer = imp_mode == 'weight_variance_both' or (
                imp_mode in ('rank_fusion', 'mag_guided') and wv_base == 'weight_variance_both')
            if _needs_consumer:
                w_map = build_consumer_weight_map(model, args.model_type,
                                                   architecture=getattr(args, 'cnn_arch', None))
                if w_map:
                    imp.set_weight_norms(w_map)
                    log_info(f"Set consumer weight norms for {len(w_map)} layers")
            if _needs_producer:
                p_map = build_producer_weight_map(model, args.model_type,
                                                    architecture=getattr(args, 'cnn_arch', None))
                if p_map:
                    imp.set_producer_weight_norms(p_map)
                    log_info(f"Set producer weight norms for {len(p_map)} layers")

            if is_main():
                var_metrics = compute_variance_entropy(imp)
                log_info(f"Variance — entropy={var_metrics['entropy']:.4f}, "
                         f"cv={var_metrics['cv']:.4f}, gini={var_metrics['gini']:.4f}")

                # V-norm vs VBP importance correlation (if reparam snapshot exists)
                vnorm_path = os.path.join(args.save_dir, "reparam_vnorms.pt")
                if os.path.exists(vnorm_path) and step_i == 0:
                    _log_vnorm_vbp_correlation(vnorm_path, model, imp, args.model_type,
                                               architecture=getattr(args, 'cnn_arch', None))

        # 2. Create pruner with per-step ratio
        step_args = argparse.Namespace(**vars(args))
        step_args.keep_ratio = per_step_keep
        pruner = create_pruner(model, example_inputs, imp, step_args)

        # 3. Prune (compensation only for VBP)
        enable_comp = is_vbp and not getattr(args, 'no_compensation', False)
        log_info(f"Pruning: per_step_prune={per_step_prune:.4f}")
        prune_model(model, pruner, device, example_inputs,
                    enable_compensation=enable_comp, is_vbp=is_vbp)

        # 3b. BN recalibration for CNNs (essential after structured pruning)
        if args.model_type == "cnn" and not getattr(args, 'no_recalib', False):
            log_info("Recalibrating BN running stats...")
            recalibrate_bn(model, train_loader, device)

        cumulative_keep *= per_step_keep

        # 4. Evaluate retention
        if is_main():
            acc_ret, loss_ret = validate(model, val_loader, device, args.model_type)
            pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
            log_info(f"Step {step_i+1} retention: acc={acc_ret:.4f}, loss={loss_ret:.4f}")
            log_info(f"  cumulative_keep={cumulative_keep:.4f}, "
                     f"MACs={pruned_macs / 1e9:.2f}G, params={pruned_params / 1e6:.2f}M")

        # 5. Per-step fine-tuning (var_loss only for VBP)
        #    Optionally re-reparameterize for continued V-regularization during PAT
        pat_reparam_mgr = None
        pat_aux_fn = None
        use_reparam_pat = (getattr(args, 'reparam_during_pat', False)
                           and getattr(args, 'sparse_mode', 'none') == 'reparam'
                           and epochs_per_step > 0)
        if use_reparam_pat:
            from torch_pruning.utils.reparam import MeanResidualManager
            pat_reparam_target = getattr(args, 'reparam_target', 'fc2')
            target_names = build_reparam_layers(
                model, args.model_type,
                architecture=getattr(args, 'cnn_arch', None),
                reparam_target=pat_reparam_target)
            pat_reparam_mgr = MeanResidualManager(
                model, target_names, device,
                lambda_reg=args.reparam_lambda,
                max_batches=args.max_batches,
                normalize=getattr(args, 'reparam_normalize', False),
                reparam_target=pat_reparam_target)
            pat_reparam_mgr.reparameterize(train_loader)
            pat_aux_fn = pat_reparam_mgr.regularization_loss
            log_info(f"PAT-{step_i+1}: reparam active with λ={args.reparam_lambda}")

        epoch_offset = step_i * epochs_per_step
        step_best = finetune(
            model, teacher, train_loader, train_sampler, val_loader,
            device, args, epochs=epochs_per_step, epoch_offset=epoch_offset,
            phase=f"PAT-{step_i+1}", use_var_loss=is_vbp,
            aux_loss_fn=pat_aux_fn, reparam_manager=pat_reparam_mgr,
        )
        best_acc = max(best_acc, step_best)

        # Merge reparam back before next prune step
        if pat_reparam_mgr is not None and pat_reparam_mgr.is_active:
            if is_main():
                pat_reparam_mgr.save_vnorm_snapshot(args.save_dir)
            pat_reparam_mgr.merge_back()

    # 6. Post-prune fine-tuning
    if args.epochs_ft > 0:
        wd_display = args.wd_ft if args.wd_ft is not None else (1e-4 if args.opt_ft == "sgd" else 0.01)
        log_info(f"\nPost-prune fine-tuning for {args.epochs_ft} epochs "
                 f"({args.opt_ft}, lr={args.lr_ft}, wd={wd_display})...")
        epoch_offset = pat_steps * epochs_per_step
        ft_best = finetune(
            model, teacher, train_loader, train_sampler, val_loader,
            device, args, epochs=args.epochs_ft, epoch_offset=epoch_offset,
            phase="FT",
        )
        best_acc = max(best_acc, ft_best)

    # Final evaluation and summary
    if is_main():
        acc_final, _ = validate(model, val_loader, device, args.model_type)
        pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)

        log_info("=" * 60)
        log_info("Summary")
        log_info("=" * 60)
        if base_macs is not None:
            log_info(f"Base MACs:    {base_macs / 1e9:.2f}G -> Pruned: {pruned_macs / 1e9:.2f}G "
                     f"({pruned_macs / base_macs * 100:.1f}%)")
            log_info(f"Base Params:  {base_params / 1e6:.2f}M -> Pruned: {pruned_params / 1e6:.2f}M "
                     f"({pruned_params / base_params * 100:.1f}%)")
        else:
            log_info(f"Pruned: {pruned_macs / 1e9:.2f}G MACs, {pruned_params / 1e6:.2f}M params")
        if acc_orig is not None:
            log_info(f"Original Acc: {acc_orig:.4f}")
        log_info(f"Final Acc:    {acc_final:.4f}")
        if best_acc > 0:
            log_info(f"Best Acc:     {best_acc:.4f}")

        save_path = os.path.join(args.save_dir, "vbp_final.pth")
        torch.save(model.state_dict(), save_path)
        log_info(f"Final model saved to {save_path}")


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

    # Suppress warnings on non-main ranks
    if not is_main():
        import warnings
        warnings.filterwarnings("ignore")

    if is_main():
        log_info("=" * 60)
        log_info("VBP ImageNet Reproduction Script")
        log_info("=" * 60)
        for k, v in vars(args).items():
            logger.info(f"  {k}: {v}")

    # Build dataloaders
    train_loader, val_loader, train_sampler = build_dataloaders(
        args, use_ddp=not args.disable_ddp
    )

    # Load model
    model = load_model(args, device)
    example_inputs = torch.randn(1, 3, 224, 224).to(device)

    # Baseline evaluation
    if is_main():
        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        log_info(f"Baseline: {base_macs / 1e9:.2f}G MACs, {base_params / 1e6:.2f}M params")

        log_info("Evaluating original model...")
        acc_orig, loss_orig = validate(model, val_loader, device, args.model_type)
        log_info(f"Original accuracy: {acc_orig:.4f}, loss: {loss_orig:.4f}")
    else:
        base_macs = base_params = acc_orig = loss_orig = None

    # Create teacher for KD (deep copy of original pretrained model)
    teacher = None
    if args.use_kd:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        log_info("Created teacher model for knowledge distillation")

    # --- Sparse pre-training (optional, default: skip) ---
    if args.sparse_mode == "reparam":
        run_reparam_pretraining(model, teacher, train_loader, train_sampler,
                                val_loader, device, args)
    elif args.sparse_mode != "none":
        run_sparse_pretraining(model, teacher, train_loader, train_sampler,
                               val_loader, device, args)

    # --- Pruning: PAT (iterative) or one-shot (pat_steps=1) ---
    if not args.pat:
        args.pat_steps = 1
        args.pat_epochs_per_step = 0  # all FT via epochs_ft

    run_pat(model, teacher, train_loader, train_sampler, val_loader,
            device, example_inputs, args,
            base_macs=base_macs, base_params=base_params, acc_orig=acc_orig)

    # Cleanup
    if not args.disable_ddp:
        cleanup()


def parse_args():
    parser = argparse.ArgumentParser(
        description="VBP ImageNet pruning with Torch-Pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model_type", default="vit", choices=["vit", "convnext", "cnn"],
                             help="Model architecture type")
    model_group.add_argument("--model_name", default="/algo/NetOptimization/outputs/VBP/DeiT_tiny",
                             help="Architecture source (HF model ID/dir, ConvNeXt variant)")
    model_group.add_argument("--checkpoint", default=None,
                             help="Optional .pth checkpoint to load weights from")
    model_group.add_argument("--cnn_arch", default="resnet50",
                             choices=["resnet18", "resnet34", "resnet50", "resnet101", "mobilenet_v2"],
                             help="CNN architecture (only used when model_type=cnn)")
    model_group.add_argument("--pretrained", action="store_true", default=True,
                             help="Use pretrained weights for CNN models")
    model_group.add_argument("--interior_only", action="store_true", default=True,
                             help="Only prune block-interior channels (not residual stream)")

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_path", default="/algo/NetOptimization/outputs/VBP/",
                            help="Path to ImageNet root (with train/val subdirs)")
    data_group.add_argument("--train_batch_size", type=int, default=64,
                            help="Training batch size per GPU")
    data_group.add_argument("--val_batch_size", type=int, default=128,
                            help="Validation batch size")
    data_group.add_argument("--num_workers", type=int, default=4,
                            help="Number of data loading workers")
    data_group.add_argument("--max_batches", type=int, default=200,
                            help="Max batches for stats collection")

    # Pruning
    prune_group = parser.add_argument_group("Pruning")
    prune_group.add_argument("--keep_ratio", type=float, default=0.65,
                             help="Ratio of channels to keep (1 - pruning_ratio)")
    prune_group.add_argument("--global_pruning", action="store_true",
                             help="Use global pruning across all layers")
    prune_group.add_argument("--max_pruning_ratio", type=float, default=1.0,
                             help="Max fraction of channels to prune per layer (e.g. 0.8 = keep at least 20%%)")
    prune_group.add_argument("--norm_per_layer", action="store_true",
                             help="Normalize variance per layer")
    prune_group.add_argument("--no_compensation", action="store_true",
                             help="Disable VBP bias compensation")
    prune_group.add_argument("--no_recalib", action="store_true",
                             help="Skip BN recalibration after pruning (CNN mode only)")
    prune_group.add_argument("--criterion", default="variance",
                             choices=["variance", "magnitude", "lamp", "random"],
                             help="Importance criterion (variance=VBP, others use BasePruner)")
    prune_group.add_argument("--importance_mode", default="variance",
                             choices=["variance", "weight_variance", "weight_variance_both", "combined", "rank_fusion", "mag_guided"],
                             help="Importance scoring: variance (σ²), weight_variance (||W_fc2[:,k]||·σ_k), "
                                  "weight_variance_both, combined, rank_fusion, or mag_guided")
    prune_group.add_argument("--wv_base_mode", default="weight_variance",
                             choices=["variance", "weight_variance", "weight_variance_both"],
                             help="WV formula for rank_fusion/mag_guided modes (default: weight_variance)")
    prune_group.add_argument("--mag_guided_delta", type=float, default=0.2,
                             help="Tolerance for mag_guided mode (default: 0.2)")

    # Fine-tuning
    ft_group = parser.add_argument_group("Fine-tuning")
    ft_group.add_argument("--epochs_ft", type=int, default=10,
                          help="Number of fine-tuning epochs")
    ft_group.add_argument("--lr_ft", type=float, default=1.5e-5,
                          help="Fine-tuning learning rate")
    ft_group.add_argument("--opt_ft", type=str, default="adamw",
                          choices=["adamw", "sgd"],
                          help="Fine-tuning optimizer (default: adamw)")
    ft_group.add_argument("--momentum_ft", type=float, default=0.9,
                          help="SGD momentum (ignored for AdamW)")
    ft_group.add_argument("--wd_ft", type=float, default=None,
                          help="Fine-tuning weight decay (default: 0.01 for AdamW, 1e-4 for SGD)")

    # Knowledge Distillation
    kd_group = parser.add_argument_group("Knowledge Distillation")
    kd_group.add_argument("--use_kd", action="store_true",
                          help="Enable knowledge distillation from unpruned teacher")
    kd_group.add_argument("--kd_alpha", type=float, default=0.7,
                          help="Weight for CE loss in KD")
    kd_group.add_argument("--kd_T", type=float, default=2.0,
                          help="Temperature for KD softmax")

    # Pruning-Aware Training (PAT)
    pat_group = parser.add_argument_group("Pruning-Aware Training")
    pat_group.add_argument("--pat", action="store_true",
                           help="Enable iterative pruning-aware training mode")
    pat_group.add_argument("--pat_steps", type=int, default=5,
                           help="Number of iterative prune-then-train cycles")
    pat_group.add_argument("--pat_epochs_per_step", type=int, default=3,
                           help="Fine-tuning epochs between prune steps")
    pat_group.add_argument("--var_loss_weight", type=float, default=0.0,
                           help="Weight for variance concentration loss (0 = disabled)")
    pat_group.add_argument("--reparam_during_pat", action="store_true",
                           help="Keep reparam active during PAT per-step fine-tuning")

    # Sparse pre-training (optional, default: none = skip)
    sparse_group = parser.add_argument_group("Sparse Pre-training")
    sparse_group.add_argument("--sparse_mode", default="none",
                              choices=["l1_group", "gmp", "reparam", "none"],
                              help="Sparse pre-training mode (none = skip)")
    sparse_group.add_argument("--epochs_sparse", type=int, default=5,
                              help="Sparse pre-training epochs")
    sparse_group.add_argument("--lr_sparse", type=float, default=1e-4,
                              help="Learning rate for sparse phase")
    sparse_group.add_argument("--l1_lambda", type=float, default=1e-4,
                              help="L2,1 regularization strength (l1_group mode)")
    sparse_group.add_argument("--gmp_target_sparsity", type=float, default=0.5,
                              help="Target weight sparsity for GMP mode")
    sparse_group.add_argument("--reparam_lambda", type=float, default=0.01,
                              help="L_{2,1} regularization strength for reparam mode")
    sparse_group.add_argument("--reparam_refresh_interval", type=int, default=1,
                              help="Re-estimate μ_x every N epochs (0 = never)")
    sparse_group.add_argument("--reparam_normalize", action="store_true",
                              help="Normalize L_{2,1} by initial norms (scale-invariant)")
    sparse_group.add_argument("--reparam_target", default="fc2", choices=["fc1", "fc2"],
                              help="Which layer to reparameterize: fc1 (upstream, row norms) or fc2 (downstream, col norms)")

    # DDP
    ddp_group = parser.add_argument_group("Distributed")
    ddp_group.add_argument("--disable_ddp", action="store_true",
                           help="Disable DDP for single-GPU debugging")
    ddp_group.add_argument("--local_rank", type=int,
                           default=int(os.environ.get("LOCAL_RANK", 0)),
                           help="Local rank for DDP (set by torchrun)")

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--save_dir", default="./output/vbp",
                           help="Directory for saving outputs")

    return parser.parse_args()


if __name__ == '__main__':
    main(sys.argv[1:])
