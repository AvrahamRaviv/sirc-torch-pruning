"""
VBP (Variance-Based Pruning) ImageNet Reproduction Script

This script reproduces results from the VBP paper using the integrated
VarianceImportance class in Torch-Pruning, with DDP support.

Reference: https://arxiv.org/pdf/2507.12988

Usage:
    # Single GPU (debug)
    python benchmarks/vbp/vbp_imagenet.py \
        --model_type vit \
        --model_name google/vit-base-patch16-224 \
        --data_path /path/to/imagenet \
        --keep_ratio 0.65 \
        --global_pruning \
        --disable_ddp

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
import datetime
import logging
import os
import pickle
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import torchvision.transforms as T
from tqdm import tqdm

import torch_pruning as tp
from transformers import ViTForImageClassification

# Local ConvNeXt implementation (FB version)
# Handle both direct execution and module import
try:
    from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
except ImportError:
    # Running directly, add parent to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convnext import convnext_tiny, convnext_small, convnext_base, convnext_large

try:
    from .sparse_utils import (
        get_fc1_modules, l21_regularization, gmp_sparsity_schedule,
        apply_unstructured_pruning, remove_pruning_reparametrization,
        compute_variance_entropy, compute_weight_sparsity,
    )
except ImportError:
    from sparse_utils import (
        get_fc1_modules, l21_regularization, gmp_sparsity_schedule,
        apply_unstructured_pruning, remove_pruning_reparametrization,
        compute_variance_entropy, compute_weight_sparsity,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("vbp_imagenet")
logger.setLevel(logging.INFO)


def is_main():
    """Check if current process is the main rank."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log_info(msg: str):
    """Log message only on main rank."""
    if is_main():
        logger.info(msg)


def setup_logging(save_dir: str):
    """Configure logging with console and file handlers."""
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (main rank only)
    if is_main():
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, "vbp_imagenet.log")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Logging to {log_path}")


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def setup_distributed(args):
    """Initialize torch.distributed for DDP. Expects torchrun launch."""
    if "RANK" not in os.environ:
        raise RuntimeError("DDP mode requires torchrun. Use --disable_ddp for single GPU.")

    timeout = datetime.timedelta(seconds=7200)  # 2 hours
    dist.init_process_group(backend="nccl", timeout=timeout)

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    log_info(f"DDP initialized | rank={args.rank}, world_size={args.world_size}")
    return device


def cleanup():
    """Destroy the DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class FastImageNet(torch.utils.data.Dataset):
    """Fast ImageNet dataset using pre-cached sample list."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.loader = default_loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_transform(model_type):
    """Training transform with light augmentations."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(model_type):
    """Validation transform."""
    if model_type == "convnext":
        interpolation = T.InterpolationMode.BICUBIC
    else:
        interpolation = T.InterpolationMode.BILINEAR

    return T.Compose([
        T.Resize(256, interpolation=interpolation),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(args, use_ddp=True):
    """Build ImageNet train/val dataloaders with optional DDP samplers."""
    train_transform = get_train_transform(args.model_type)
    val_transform = get_val_transform(args.model_type)

    log_info("Loading ImageNet dataset...")

    # Check for cached pickle files (fast path)
    train_pkl = os.path.join(args.data_path, "train_samples.pkl")
    val_pkl = os.path.join(args.data_path, "val_samples.pkl")

    if os.path.exists(train_pkl) and os.path.exists(val_pkl):
        log_info("Using cached sample lists for fast loading")
        with open(train_pkl, "rb") as f:
            train_samples = pickle.load(f)
        with open(val_pkl, "rb") as f:
            val_samples = pickle.load(f)

        train_dst = FastImageNet(train_samples, transform=train_transform)
        val_dst = FastImageNet(val_samples, transform=val_transform)
    else:
        log_info("Using ImageFolder (slow path)")
        train_dst = ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
        val_dst = ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)

    # Create samplers
    if use_ddp:
        train_sampler = DistributedSampler(train_dst, shuffle=True)
        val_sampler = None  # Sequential for validation
    else:
        train_sampler = None
        val_sampler = None

    # Create loaders
    train_loader = DataLoader(
        train_dst,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dst,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    log_info(f"Train samples: {len(train_dst)}, Val samples: {len(val_dst)}")
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(args, device):
    """Load ViT (HuggingFace) or ConvNeXt (FB implementation)."""
    if args.model_type == "vit":
        # HuggingFace ViT
        model = ViTForImageClassification.from_pretrained(args.model_name, local_files_only=os.path.isdir(args.model_name))
        model = model.to(device)
        log_info(f"Loaded ViT from {args.model_name}")

    elif args.model_type == "convnext":
        # FB ConvNeXt implementation — model_name is the checkpoint path
        variant_map = {
            "convnext_tiny": convnext_tiny,
            "convnext_small": convnext_small,
            "convnext_base": convnext_base,
            "convnext_large": convnext_large,
        }

        # Determine variant from model_name (path or variant string)
        variant = args.model_name.lower()
        for key in variant_map:
            if key in variant:
                model_fn = variant_map[key]
                break
        else:
            model_fn = convnext_tiny
            log_info(f"Unknown ConvNeXt variant '{args.model_name}', defaulting to tiny")

        model = model_fn(pretrained=False)

        # Load checkpoint (model_name is the .pth path)
        if os.path.exists(args.model_name):
            state = torch.load(args.model_name, map_location="cpu", weights_only=True)
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=True)
            log_info(f"Loaded ConvNeXt checkpoint from {args.model_name}")
        else:
            log_info(f"WARNING: Checkpoint not found at {args.model_name}, using random weights")

        model = model.to(device)

    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    return model


def forward_logits(model, images, model_type: str):
    """Unified logits extraction for HF ViT and ConvNeXt."""
    out = model(images)
    if hasattr(out, "logits"):
        return out.logits
    return out


# ---------------------------------------------------------------------------
# ConvNeXt VBP helpers
# ---------------------------------------------------------------------------
def _make_post_gelu_nchw(act_fn):
    """Post-act fn: apply GELU then permute NHWC -> NCHW for stats hook."""
    def fn(x):
        return act_fn(x).permute(0, 3, 1, 2)
    return fn


# ---------------------------------------------------------------------------
# Variance concentration loss for PAT
# ---------------------------------------------------------------------------
class VarianceConcentrationHooks:
    """Register hooks on target layers, compute entropy of per-channel variance.

    During PAT fine-tuning, this penalizes flat variance distributions,
    encouraging the model to concentrate variance in fewer channels
    (improving the VBP pruning signal for subsequent steps).
    """

    def __init__(self, model, model_type):
        self.hooks = []
        self.activations = {}
        self._register(model, model_type)

    def _register(self, model, model_type):
        if model_type == "vit":
            for block in model.vit.encoder.layer:
                mod = block.intermediate.dense
                act_fn = block.intermediate.intermediate_act_fn
                self.hooks.append(mod.register_forward_hook(
                    self._make_hook(mod, act_fn)))
        elif model_type == "convnext":
            for stage in model.stages:
                for block in stage:
                    mod = block.pwconv1
                    act_fn = block.act
                    self.hooks.append(mod.register_forward_hook(
                        self._make_hook(mod, act_fn)))

    def _make_hook(self, mod, act_fn):
        def hook(module, inp, out):
            x = out.detach()
            if act_fn is not None:
                x = act_fn(x)
            self.activations[mod] = x
        return hook

    def compute_loss(self):
        """Compute entropy of normalized per-channel variance across hooked layers."""
        total_entropy = torch.tensor(0.0)
        for mod, act in self.activations.items():
            total_entropy = total_entropy.to(act.device)
            # Flatten spatial/sequence dims, keep channel last
            if act.dim() == 4:
                # Conv: [B, C, H, W] or NHWC -> per-channel over B,H,W
                var = act.var(dim=(0, 2, 3))
            elif act.dim() == 3:
                # Transformer: [B, T, C] -> per-channel over B,T
                var = act.var(dim=(0, 1))
            elif act.dim() == 2:
                var = act.var(dim=0)
            else:
                continue
            p = var / (var.sum() + 1e-8)
            entropy = -(p * torch.log(p + 1e-8)).sum()
            total_entropy = total_entropy + entropy
        self.activations.clear()
        return total_entropy

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.activations.clear()


# ---------------------------------------------------------------------------
# Sparse pre-training
# ---------------------------------------------------------------------------
def run_sparse_pretraining(model, teacher, train_loader, train_sampler,
                           val_loader, device, args):
    """Optional sparse pre-training stage before VBP stats collection.

    Supports DDP: wraps model for training, unwraps after.
    Modifies model in-place. Uses KD from teacher if --use_kd is set.
    """
    if args.model_type != "vit":
        log_info(f"WARNING: Sparse pre-training only supports ViT, "
                 f"skipping for {args.model_type}")
        return

    fc1_pairs = get_fc1_modules(model, model_type=args.model_type)
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
    scheduler, step_per_batch = build_ft_scheduler(
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

        train_loss = train_one_epoch(
            train_model, train_loader, train_sampler, optimizer, scheduler,
            device, epoch, args, teacher=teacher, fc1_modules=fc1_modules,
            step_per_batch=step_per_batch, phase="Sparse")

        # Validate on unwrapped model
        if is_main():
            acc, val_loss = validate(model, val_loader, device, args.model_type)
            log_info(f"Sparse {epoch+1}/{args.epochs_sparse}: "
                     f"train_loss={train_loss:.4f}, val_acc={acc:.4f}")
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


# ---------------------------------------------------------------------------
# Pruner setup
# ---------------------------------------------------------------------------
def create_pruner(model, example_inputs, imp, args):
    """Create VBPPruner with VarianceImportance and bias compensation."""
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

    else:
        output_transform = lambda out: out.sum()

    pruner = tp.pruner.VBPPruner(
        model,
        example_inputs,
        importance=imp,
        global_pruning=args.global_pruning,
        pruning_ratio=1.0 - args.keep_ratio,
        ignored_layers=ignored_layers,
        output_transform=output_transform,
        mean_dict=imp.means,
        verbose=is_main(),
    )

    return pruner


# ---------------------------------------------------------------------------
# Pruning with VBP compensation
# ---------------------------------------------------------------------------
def prune_model(model, pruner, device, example_inputs):
    """
    Apply VBP pruning with bias compensation.

    VBPPruner.step() handles compensation internally:
    1. Caches consumer inputs via forward hooks
    2. For each group: compensates consumer bias, then prunes
    """
    # Cache consumer inputs for compensation
    pruner.enable_meancheck(model)
    model.eval()
    with torch.no_grad():
        model(example_inputs)

    # VBPPruner.step(interactive=False) applies compensation + prune per group
    pruner.step(interactive=False, enable_compensation=True)

    pruner.disable_meancheck()
    log_info("Pruning with VBP compensation complete")


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
# Training
# ---------------------------------------------------------------------------
def build_ft_scheduler(optimizer, epochs, steps_per_epoch):
    """Build cosine LR scheduler with per-batch stepping.

    Returns:
        (scheduler, step_per_batch): scheduler and its stepping granularity.
        If step_per_batch=True, call scheduler.step() every batch.
        If step_per_batch=False, call scheduler.step() every epoch.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-8
    )
    return scheduler, True


def train_one_epoch(model, train_loader, train_sampler, optimizer,
                    scheduler, device, epoch, args,
                    teacher=None, fc1_modules=None,
                    step_per_batch=True, phase="Epoch",
                    var_hooks=None):
    """Unified training epoch: optional KD + optional L2,1 regularization + optional var loss.

    Args:
        teacher: If not None and args.use_kd, apply knowledge distillation.
        fc1_modules: If not None and args.sparse_mode == "l1_group",
            add L2,1 group regularization.
        phase: Log prefix ("Epoch" for fine-tuning, "Sparse" for sparse phase).
        var_hooks: If not None and args.var_loss_weight > 0, compute variance
            concentration loss from hooked activations.
    """
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    use_kd = args.use_kd and teacher is not None
    use_l21 = fc1_modules is not None and args.sparse_mode == "l1_group"
    use_var_loss = var_hooks is not None and getattr(args, 'var_loss_weight', 0) > 0

    total_loss = 0.0
    total_reg = 0.0
    total_var = 0.0
    num_batches = 0

    total = len(train_loader)
    log_interval = max(total // 20, 1)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = forward_logits(model, images, args.model_type)
        ce_loss = F.cross_entropy(logits, labels)

        # Knowledge distillation
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

        # L2,1 group regularization
        if use_l21:
            reg_loss = l21_regularization(fc1_modules, device)
            loss = loss + args.l1_lambda * reg_loss
            total_reg += reg_loss.item()

        # Variance concentration loss
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
            if use_l21:
                parts.append(f"L21={total_reg / num_batches:.2f}")
            if use_var_loss:
                parts.append(f"var={total_var / num_batches:.4f}")
            log_info(f"{phase} {epoch+1} [{batch_idx+1}/{total}] {' '.join(parts)}")

    if not step_per_batch:
        scheduler.step()
    return total_loss / max(num_batches, 1)


def validate(model, val_loader, device, model_type: str):
    """Validate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        total_val = len(val_loader)
        pbar = tqdm(val_loader, disable=not is_main(), desc="Validating", miniters=max(total_val // 20, 1))
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_logits(model, images, model_type)
            loss = F.cross_entropy(logits, labels, reduction="sum")

            _, pred = logits.max(1)
            correct += (pred == labels).sum().item()
            loss_sum += loss.item()
            total += images.size(0)

    acc = correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss


# ---------------------------------------------------------------------------
# Pruning-Aware Training (PAT)
# ---------------------------------------------------------------------------
def run_pat(model, teacher, train_loader, train_sampler, val_loader,
            device, example_inputs, args,
            base_macs=None, base_params=None, acc_orig=None):
    """Iterative pruning-aware training: prune in multiple steps with fine-tuning.

    Each step: collect fresh stats -> prune a fraction -> fine-tune.
    Geometric schedule: per_step_keep^pat_steps = keep_ratio.
    Also used for one-shot pruning (pat_steps=1).
    """
    pat_steps = args.pat_steps
    epochs_per_step = args.pat_epochs_per_step
    per_step_keep = args.keep_ratio ** (1.0 / pat_steps)
    per_step_prune = 1.0 - per_step_keep

    log_info(f"PAT: {pat_steps} steps, per_step_keep={per_step_keep:.4f}, "
             f"epochs_per_step={epochs_per_step}, target_keep={args.keep_ratio}")

    use_ddp = not args.disable_ddp and dist.is_initialized()

    # Track cumulative keep ratio for logging
    cumulative_keep = 1.0
    best_acc = 0.0

    for step_i in range(pat_steps):
        log_info(f"\n{'='*60}")
        log_info(f"PAT Step {step_i+1}/{pat_steps}")
        log_info(f"{'='*60}")

        # 1. Collect fresh VBP stats on current model
        imp = tp.importance.VarianceImportance(norm_per_layer=args.norm_per_layer)
        collect_and_sync_stats(model, train_loader, device, imp, args)

        if is_main():
            var_metrics = compute_variance_entropy(imp)
            log_info(f"Variance — entropy={var_metrics['entropy']:.4f}, "
                     f"cv={var_metrics['cv']:.4f}, gini={var_metrics['gini']:.4f}")

        # 2. Create pruner with per-step ratio
        step_args = argparse.Namespace(**vars(args))
        step_args.keep_ratio = per_step_keep
        pruner = create_pruner(model, example_inputs, imp, step_args)

        # 3. Prune with compensation
        log_info(f"Pruning: per_step_prune={per_step_prune:.4f}")
        prune_model(model, pruner, device, example_inputs)

        cumulative_keep *= per_step_keep

        # 4. Evaluate retention
        if is_main():
            acc_ret, loss_ret = validate(model, val_loader, device, args.model_type)
            pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
            log_info(f"Step {step_i+1} retention: acc={acc_ret:.4f}, loss={loss_ret:.4f}")
            log_info(f"  cumulative_keep={cumulative_keep:.4f}, "
                     f"MACs={pruned_macs / 1e9:.2f}G, params={pruned_params / 1e6:.2f}M")

        # 5. Fine-tune for epochs_per_step epochs
        if epochs_per_step > 0:
            # Wrap in DDP for fine-tuning
            train_model = model
            if use_ddp:
                train_model = DDP(model, device_ids=[args.local_rank],
                                  output_device=args.local_rank)
                dist.barrier()

            optimizer = torch.optim.AdamW(train_model.parameters(),
                                          lr=args.lr_ft, weight_decay=0.01)
            scheduler, step_per_batch = build_ft_scheduler(
                optimizer, epochs_per_step, len(train_loader))

            # Optional variance concentration hooks
            var_hooks = None
            if args.var_loss_weight > 0:
                var_hooks = VarianceConcentrationHooks(model, args.model_type)

            for ep in range(epochs_per_step):
                global_epoch = step_i * epochs_per_step + ep
                train_loss = train_one_epoch(
                    train_model, train_loader, train_sampler,
                    optimizer, scheduler, device, global_epoch, args,
                    teacher=teacher, step_per_batch=step_per_batch,
                    phase=f"PAT-{step_i+1}", var_hooks=var_hooks,
                )

                if is_main():
                    eval_model = train_model.module if isinstance(train_model, DDP) else train_model
                    acc_ft, loss_ft = validate(eval_model, val_loader, device, args.model_type)
                    log_info(f"PAT-{step_i+1} ep {ep+1}/{epochs_per_step}: "
                             f"train_loss={train_loss:.4f}, val_acc={acc_ft:.4f}")

                    if acc_ft > best_acc:
                        best_acc = acc_ft
                        save_path = os.path.join(args.save_dir, "vbp_best.pth")
                        torch.save(eval_model.state_dict(), save_path)
                        log_info(f"New best! Saved to {save_path}")

                if use_ddp:
                    dist.barrier()

            if var_hooks is not None:
                var_hooks.remove()

            # Unwrap DDP
            if use_ddp:
                del train_model

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
    if args.sparse_mode != "none":
        run_sparse_pretraining(model, teacher, train_loader, train_sampler,
                               val_loader, device, args)

    # --- Pruning: PAT (iterative) or one-shot (pat_steps=1) ---
    if not args.pat:
        # One-shot mode: override PAT args to single step
        args.pat_steps = 1
        args.pat_epochs_per_step = args.epochs_ft

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
    model_group.add_argument("--model_type", default="vit", choices=["vit", "convnext"],
                             help="Model architecture type")
    model_group.add_argument("--model_name", default="/algo/NetOptimization/outputs/VBP/DeiT_tiny",
                             help="Model name/path (HF model ID or ConvNeXt .pth path)")

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
    prune_group.add_argument("--norm_per_layer", action="store_true",
                             help="Normalize variance per layer")

    # Fine-tuning
    ft_group = parser.add_argument_group("Fine-tuning")
    ft_group.add_argument("--epochs_ft", type=int, default=10,
                          help="Number of fine-tuning epochs")
    ft_group.add_argument("--lr_ft", type=float, default=1.5e-5,
                          help="Fine-tuning learning rate (AdamW)")

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

    # Sparse pre-training (optional, default: none = skip)
    sparse_group = parser.add_argument_group("Sparse Pre-training")
    sparse_group.add_argument("--sparse_mode", default="none",
                              choices=["l1_group", "gmp", "none"],
                              help="Sparse pre-training mode (none = skip)")
    sparse_group.add_argument("--epochs_sparse", type=int, default=5,
                              help="Sparse pre-training epochs")
    sparse_group.add_argument("--lr_sparse", type=float, default=1e-4,
                              help="Learning rate for sparse phase")
    sparse_group.add_argument("--l1_lambda", type=float, default=1e-4,
                              help="L2,1 regularization strength (l1_group mode)")
    sparse_group.add_argument("--gmp_target_sparsity", type=float, default=0.5,
                              help="Target weight sparsity for GMP mode")

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
