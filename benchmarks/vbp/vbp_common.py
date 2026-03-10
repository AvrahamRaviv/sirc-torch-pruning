"""
Common infrastructure for VBP benchmark scripts.

Shared training, data loading, logging, and model loading functions
extracted from vbp_imagenet.py to avoid duplication across scripts.
"""

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

from transformers import ViTForImageClassification

# Local ConvNeXt implementation (FB version)
try:
    from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convnext import convnext_tiny, convnext_small, convnext_base, convnext_large

from torch_pruning.utils.sparse_utils import l21_regularization


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("vbp_imagenet")
logger.setLevel(logging.INFO)
logger.propagate = False  # prevent duplicate output via root logger


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
    """Configure logging with console and file handlers (main rank only)."""
    logger.handlers.clear()
    if not is_main():
        # Non-main ranks: no handlers → silent
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

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
def _merge_vnr_state_dict(state, eps=1e-5):
    """Convert VNR checkpoint (v_tilde, m, bn.*) to standard (weight, bias).

    If no VNR keys found, returns state unchanged.
    """
    # Find VNR layer prefixes: keys ending in .v_tilde
    prefixes = [k.rsplit(".v_tilde", 1)[0] for k in state if k.endswith(".v_tilde")]
    if not prefixes:
        return state

    new_state = {}
    consumed = set()
    for prefix in prefixes:
        vt = state[prefix + ".v_tilde"]
        m = state[prefix + ".m"]
        var = state[prefix + ".bn.running_var"]
        mu = state[prefix + ".bn.running_mean"]
        sigma = torch.sqrt(var + eps)

        # Linear: v_tilde [out, in], sigma [in]
        w_eff = vt / sigma[None, :]
        b_eff = m - w_eff @ mu

        new_state[prefix + ".weight"] = w_eff
        new_state[prefix + ".bias"] = b_eff
        consumed.update([
            prefix + ".v_tilde", prefix + ".m",
            prefix + ".bn.running_mean", prefix + ".bn.running_var",
            prefix + ".bn.num_batches_tracked",
        ])

    # Copy non-VNR keys as-is
    for k, v in state.items():
        if k not in consumed:
            new_state[k] = v

    log_info(f"Converted {len(prefixes)} VNR layers to standard weight/bias")
    return new_state


def load_model(args, device):
    """Load model architecture, optionally overriding weights from --checkpoint.

    Args:
        args.model_name: Architecture source.
            - vit: HuggingFace model ID or local dir (e.g. "google/vit-base-patch16-224")
            - convnext: variant string or path containing variant name
            - cnn: ignored (architecture from --cnn_arch)
        args.checkpoint: Optional .pth file to load weights from (overrides default weights).
    """
    checkpoint = getattr(args, 'checkpoint', None)
    if checkpoint and not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            args.model_name, local_files_only=os.path.isdir(args.model_name))
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = _merge_vnr_state_dict(state)
            model.load_state_dict(state, strict=True)
            log_info(f"Loaded ViT arch from {args.model_name}, weights from {checkpoint}")
        else:
            log_info(f"Loaded ViT from {args.model_name}")
        model = model.to(device)

    elif args.model_type == "convnext":
        variant_map = {
            "convnext_tiny": convnext_tiny,
            "convnext_small": convnext_small,
            "convnext_base": convnext_base,
            "convnext_large": convnext_large,
        }
        variant = args.model_name.lower()
        for key in variant_map:
            if key in variant:
                model_fn = variant_map[key]
                break
        else:
            model_fn = convnext_tiny
            log_info(f"Unknown ConvNeXt variant '{args.model_name}', defaulting to tiny")

        model = model_fn(pretrained=False)
        ckpt = checkpoint or args.model_name
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=True)
            log_info(f"Loaded ConvNeXt checkpoint from {ckpt}")
        else:
            log_info(f"WARNING: Checkpoint not found at {ckpt}, using random weights")
        model = model.to(device)

    elif args.model_type == "cnn":
        import torchvision.models as tv_models
        model_map = {
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
            "resnet101": tv_models.resnet101,
            "mobilenet_v2": tv_models.mobilenet_v2,
        }
        model_fn = model_map[args.cnn_arch]
        model = model_fn(pretrained=False)
        ckpt = checkpoint or args.model_name
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device)
        log_info(f"Loaded {args.cnn_arch} from {ckpt}")

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
# Variance concentration loss for PAT
# ---------------------------------------------------------------------------
class VarianceConcentrationHooks:
    """Register hooks on target layers, compute entropy of per-channel variance.

    During PAT fine-tuning, this penalizes flat variance distributions,
    encouraging the model to concentrate variance in fewer channels
    (improving the VBP pruning signal for subsequent steps).
    """

    def __init__(self, model, model_type, target_layers=None):
        self.hooks = []
        self.activations = {}
        self._register(model, model_type, target_layers)

    def _register(self, model, model_type, target_layers=None):
        if target_layers is not None:
            # Generic path: use provided (module, act_fn) pairs
            for mod, act_fn in target_layers:
                self.hooks.append(mod.register_forward_hook(
                    self._make_hook(mod, act_fn)))
        elif model_type == "vit":
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
# Layer name helpers
# ---------------------------------------------------------------------------
def build_reparam_layers(model, model_type, architecture=None, reparam_target="fc2"):
    """Return layer names for reparam.

    Args:
        reparam_target: "fc2" (downstream/output layer) or "fc1" (upstream/intermediate layer).
    """
    layers = []
    if model_type == "vit":
        suffix = ".output.dense" if reparam_target == "fc2" else ".intermediate.dense"
        for name, m in model.named_modules():
            if name.endswith(suffix) and ".attention." not in name:
                layers.append(name)
    elif model_type == "convnext":
        attr = "pwconv2" if reparam_target == "fc2" else "pwconv1"
        for name, m in model.named_modules():
            if hasattr(m, attr):
                layers.append(name + "." + attr)
    elif model_type == "cnn":
        if architecture and "mobilenet" in architecture.lower():
            for name, m in model.named_modules():
                if not isinstance(m, nn.Conv2d) or m.groups != 1 or m.kernel_size != (1, 1):
                    continue
                if reparam_target == "fc2" and m.in_channels > m.out_channels:
                    layers.append(name)  # project conv (narrows)
                elif reparam_target == "fc1" and m.out_channels > m.in_channels:
                    layers.append(name)  # expand conv (widens)
        else:
            # ResNet Bottleneck: fc2=conv3 (expand), fc1=conv1 (reduce)
            suffix = "conv3" if reparam_target == "fc2" else "conv1"
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d) and name.endswith(suffix) \
                   and m.kernel_size == (1, 1) and m.groups == 1:
                    layers.append(name)
    return layers


def build_consumer_weight_map(model, model_type, architecture=None):
    """Map fc1 modules to per-channel L2 norms of corresponding fc2 weight columns.

    Returns:
        Dict[nn.Module, torch.Tensor]: fc1_module → ||W_fc2[:,k]||₂ shape [d_intermediate].
    """
    weight_map = {}
    if model_type == "vit":
        for block in model.vit.encoder.layer:
            fc1 = block.intermediate.dense
            fc2 = block.output.dense
            # fc2.weight: [d_model, d_intermediate], column k = fc2.weight[:, k]
            weight_map[fc1] = fc2.weight.detach().norm(p=2, dim=0)
    elif model_type == "cnn":
        if architecture and "mobilenet" in architecture.lower():
            # MNV2 inverted residual: expand(fc1) → DW → project(fc2)
            for name, m in model.named_modules():
                if not hasattr(m, 'conv'):
                    continue
                expand, project = None, None
                for sub in m.conv.modules():
                    if isinstance(sub, nn.Conv2d) and sub.groups == 1 and sub.kernel_size == (1, 1):
                        if sub.out_channels > sub.in_channels:
                            expand = sub
                        elif sub.in_channels > sub.out_channels:
                            project = sub
                if expand is not None and project is not None:
                    # project.weight: [C_bottleneck, C_expand, 1, 1]
                    weight_map[expand] = project.weight.detach().squeeze(-1).squeeze(-1).norm(p=2, dim=0)
        else:
            # ResNet Bottleneck: conv1(fc1) → conv2 → conv3(fc2)
            for name, m in model.named_modules():
                if hasattr(m, 'conv1') and hasattr(m, 'conv3'):
                    fc1 = m.conv1
                    fc2 = m.conv3
                    # conv3.weight: [C_residual, C_mid, 1, 1]
                    weight_map[fc1] = fc2.weight.detach().squeeze(-1).squeeze(-1).norm(p=2, dim=0)
    if weight_map:
        log_info(f"Consumer weight map: {len(weight_map)} fc1→fc2 pairs ({model_type})")
    return weight_map


def build_layers_to_prune(model, model_type, architecture=None, interior_only=True):
    """Return list of module names to prune (fc1 / pwconv1 / interior convs)."""
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
        ignored = build_cnn_ignored_layers(model, architecture, interior_only)
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m not in ignored:
                layers.append(name)
    return layers


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_cosine_scheduler(optimizer, epochs, steps_per_epoch):
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


def build_sparse_scheduler(optimizer, epochs, steps_per_epoch):
    """Cosine scheduler for sparse pre-training phase."""
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-8
    )
    return scheduler, True


def build_ft_scheduler(optimizer, epochs, steps_per_epoch, eta_min=1e-8):
    """Cosine scheduler for fine-tuning phase.

    Args:
        eta_min: Minimum LR floor. Raise for short FT runs (e.g. 1e-5)
                 to keep the model learning through all epochs.
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=eta_min
    )
    return scheduler, True


def train_one_epoch(model, train_loader, train_sampler, optimizer,
                    scheduler, device, epoch, args,
                    teacher=None, fc1_modules=None,
                    step_per_batch=True, phase="Epoch",
                    var_hooks=None, regularize_fn=None,
                    aux_loss_fn=None):
    """Unified training epoch: optional KD + optional L2,1 regularization + optional var loss.

    Args:
        teacher: If not None and args.use_kd, apply knowledge distillation.
        fc1_modules: If not None and args.sparse_mode == "l1_group",
            add L2,1 group regularization.
        phase: Log prefix ("Epoch" for fine-tuning, "Sparse" for sparse phase).
        var_hooks: If not None and args.var_loss_weight > 0, compute variance
            concentration loss from hooked activations.
        aux_loss_fn: Optional callable returning a scalar tensor (e.g. reparam
            regularization). Added to loss before backward.
    """
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    use_kd = args.use_kd and teacher is not None
    use_l21 = fc1_modules is not None and getattr(args, 'sparse_mode', 'none') == "l1_group"
    use_var_loss = var_hooks is not None and getattr(args, 'var_loss_weight', 0) > 0

    total_loss = 0.0
    total_reg = 0.0
    total_var = 0.0
    total_aux = {}  # key → running sum (supports dict-returning aux_loss_fn)
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

        # Auxiliary loss (e.g. mean-residual regularization)
        # aux_loss_fn may return a scalar OR a dict {name: tensor} for per-component logging
        if aux_loss_fn is not None:
            aux = aux_loss_fn()
            if isinstance(aux, dict):
                for k, v in aux.items():
                    loss = loss + v
                    total_aux[k] = total_aux.get(k, 0.0) + v.item()
            else:
                loss = loss + aux
                total_aux["aux"] = total_aux.get("aux", 0.0) + aux.item()

        optimizer.zero_grad()
        loss.backward()
        if regularize_fn is not None:
            eval_m = model.module if isinstance(model, DDP) else model
            regularize_fn(eval_m)
        optimizer.step()
        if step_per_batch:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if is_main() and (batch_idx % log_interval == 0 or batch_idx == total - 1):
            avg_loss = total_loss / num_batches
            cur_lr = optimizer.param_groups[0]['lr']
            parts = [f"loss={avg_loss:.4f}", f"lr={cur_lr:.2e}"]
            if use_l21:
                parts.append(f"L21={total_reg / num_batches:.2f}")
            if use_var_loss:
                parts.append(f"var={total_var / num_batches:.4f}")
            if total_aux:
                for k, v in total_aux.items():
                    parts.append(f"{k}={v / num_batches:.4f}")
            log_info(f"{phase} {epoch+1} [{batch_idx+1}/{total}] {' '.join(parts)}")

    if not step_per_batch:
        scheduler.step()
    avg_loss = total_loss / max(num_batches, 1)
    avg_aux = {k: v / max(num_batches, 1) for k, v in total_aux.items()} if total_aux else {}
    return avg_loss, avg_aux


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
