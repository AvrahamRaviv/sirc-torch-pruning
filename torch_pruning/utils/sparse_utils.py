"""Sparse training primitives for VBP pre-conditioning.

Provides L2,1 group regularization, gradual magnitude pruning (GMP),
and variance distribution metrics to quantify VBP signal quality.

Reference: Zhu & Gupta 2017 — "To prune, or not to prune" (GMP schedule)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torch_pruning.pruner.importance import VarianceImportance


# ---------------------------------------------------------------------------
# fc1 module extraction
# ---------------------------------------------------------------------------
def get_fc1_modules(model, model_type="vit", cnn_arch=None) -> list[tuple[str, nn.Module]]:
    """Return (name, module) pairs for interior layers targeted by sparse training.

    Args:
        model: The model to extract modules from.
        model_type: Architecture type ("vit", "cnn").
        cnn_arch: CNN architecture name (e.g. "resnet50", "mobilenet_v2").
            Required when model_type == "cnn".

    Returns:
        List of (name, module) tuples for each target layer.
    """
    if model_type == "vit":
        result = []
        for name, module in model.named_modules():
            if name.endswith(".intermediate.dense"):
                result.append((name, module))
        return result

    if model_type == "cnn":
        if cnn_arch is None:
            raise ValueError("cnn_arch is required when model_type='cnn'")
        return _get_cnn_interior_modules(model, cnn_arch)

    raise ValueError(f"Unsupported model_type: {model_type}")


def _get_cnn_interior_modules(model, cnn_arch: str) -> list[tuple[str, nn.Module]]:
    """Return interior conv modules for CNN sparse training.

    ResNet-50 (Bottleneck): conv1 and conv2 from each Bottleneck block.
    MobileNetV2 (InvertedResidual): expand conv (first 1x1 Conv2d) in each block.
    """
    result = []

    if cnn_arch.startswith("resnet"):
        from torchvision.models.resnet import Bottleneck, BasicBlock
        for name, module in model.named_modules():
            if isinstance(module, Bottleneck):
                result.append((name + ".conv1", module.conv1))
                result.append((name + ".conv2", module.conv2))
            elif isinstance(module, BasicBlock):
                result.append((name + ".conv1", module.conv1))

    elif cnn_arch == "mobilenet_v2":
        try:
            from torchvision.models.mobilenetv2 import InvertedResidual
        except ImportError:
            from torchvision.models.mobilenet import InvertedResidual
        for name, module in model.named_modules():
            if isinstance(module, InvertedResidual):
                # Expand conv: first 1x1 Conv2d that increases channels (out > in).
                # Blocks with expansion=1 have no expand conv (skip them).
                for sub_name, sub in module.conv.named_modules():
                    if (isinstance(sub, nn.Conv2d) and sub.kernel_size == (1, 1)
                            and sub.groups == 1 and sub.out_channels > sub.in_channels):
                        result.append((f"{name}.conv.{sub_name}", sub))
                        break

    else:
        raise ValueError(f"Unsupported cnn_arch for sparse training: {cnn_arch}")

    return result


# ---------------------------------------------------------------------------
# L2,1 group regularization
# ---------------------------------------------------------------------------
def l21_regularization(modules: list[nn.Module], device) -> torch.Tensor:
    """Compute L2,1 norm: sum of L2 norms of each output neuron's weight row.

    For Linear [out, in] or Conv2d [out, in, k, k]:
      ||W||_{2,1} = sum_i ||w_i||_2
    where w_i is the i-th output channel flattened to a vector.
    This encourages entire output channels to shrink to zero.

    Args:
        modules: List of nn.Linear or nn.Conv2d modules.
        device: Device for the result tensor.

    Returns:
        Scalar tensor with L2,1 regularization loss.
    """
    reg = torch.tensor(0.0, device=device)
    for m in modules:
        # Flatten to [out_channels, -1] then L2 per row — works for both 2D and 4D
        reg = reg + m.weight.flatten(1).norm(p=2, dim=1).sum()
    return reg


# ---------------------------------------------------------------------------
# GMP (Gradual Magnitude Pruning)
# ---------------------------------------------------------------------------
def gmp_sparsity_schedule(epoch: int, total_epochs: int,
                          init_s: float = 0.0, target_s: float = 0.5) -> float:
    """Cubic sparsity schedule from Zhu & Gupta 2017.

    s_t = s_f + (s_i - s_f) * (1 - t/T)^3

    Args:
        epoch: Current epoch (0-indexed).
        total_epochs: Total number of sparse training epochs.
        init_s: Initial sparsity (usually 0).
        target_s: Final target sparsity.

    Returns:
        Target sparsity for this epoch.
    """
    if total_epochs <= 1:
        return target_s
    t = min(epoch / (total_epochs - 1), 1.0)
    return target_s + (init_s - target_s) * (1.0 - t) ** 3


def apply_unstructured_pruning(modules: list[nn.Module], sparsity: float) -> None:
    """Apply global unstructured L1 pruning across all modules.

    Removes any existing pruning masks before applying the new sparsity level,
    avoiding mask stacking. Uses global pruning to match VBP's global philosophy.

    Args:
        modules: List of nn.Linear or nn.Conv2d modules to prune.
        sparsity: Target fraction of weights to zero out (0 to 1).
    """
    if sparsity <= 0:
        return

    # Remove existing masks to avoid stacking
    for m in modules:
        if hasattr(m, "weight_mask"):
            prune.remove(m, "weight")

    # Global L1 unstructured pruning across all fc1 layers
    parameters_to_prune = [(m, "weight") for m in modules]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )


def remove_pruning_reparametrization(modules: list[nn.Module]) -> None:
    """Bake pruning masks into weights permanently.

    Call this after sparse pre-training, before VBP stats collection,
    so the model becomes a standard dense model with zero-valued weights.

    Args:
        modules: List of nn.Linear or nn.Conv2d modules with active pruning masks.
    """
    for m in modules:
        if hasattr(m, "weight_mask"):
            prune.remove(m, "weight")


# ---------------------------------------------------------------------------
# Variance distribution metrics
# ---------------------------------------------------------------------------
def compute_variance_entropy(imp: VarianceImportance) -> dict:
    """Compute metrics quantifying the sharpness of the variance distribution.

    A sharper (less uniform) distribution means VBP has a clearer signal
    for channel selection. Lower entropy / higher Gini = sharper.

    Args:
        imp: VarianceImportance with collected .variance dict.

    Returns:
        Dictionary with entropy, cv, gini, and top-K concentration metrics.
    """
    all_vars = []
    for var in imp.variance.values():
        all_vars.append(var.detach().cpu().float())

    if not all_vars:
        return {"entropy": float("nan"), "cv": float("nan"),
                "gini": float("nan"), "top10_pct": float("nan"),
                "top20_pct": float("nan"), "top50_pct": float("nan")}

    v = torch.cat(all_vars)
    n = v.numel()

    # Normalize to probability distribution
    v_sum = v.sum()
    if v_sum <= 0:
        return {"entropy": float("nan"), "cv": float("nan"),
                "gini": float("nan"), "top10_pct": float("nan"),
                "top20_pct": float("nan"), "top50_pct": float("nan")}

    p = v / v_sum

    # Entropy (lower = sharper)
    log_p = torch.log(p + 1e-12)
    entropy = -(p * log_p).sum().item()
    max_entropy = math.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Coefficient of variation (higher = sharper)
    cv = (v.std() / (v.mean() + 1e-12)).item()

    # Gini coefficient (higher = more concentrated)
    sorted_v, _ = v.sort()
    cumsum = sorted_v.cumsum(0)
    gini = 1.0 - 2.0 * cumsum.sum().item() / (n * v_sum.item()) + 1.0 / n

    # Top-K% concentration: fraction of total variance in top K% channels
    sorted_desc, _ = v.sort(descending=True)
    cumsum_desc = sorted_desc.cumsum(0)
    top10_k = max(1, int(0.10 * n))
    top20_k = max(1, int(0.20 * n))
    top50_k = max(1, int(0.50 * n))

    return {
        "entropy": normalized_entropy,
        "cv": cv,
        "gini": gini,
        "top10_pct": (cumsum_desc[top10_k - 1] / v_sum).item(),
        "top20_pct": (cumsum_desc[top20_k - 1] / v_sum).item(),
        "top50_pct": (cumsum_desc[top50_k - 1] / v_sum).item(),
    }


def compute_weight_sparsity(modules: list[nn.Module]) -> dict:
    """Compute fraction of zero weights per layer and globally.

    Args:
        modules: List of nn.Linear or nn.Conv2d modules.

    Returns:
        Dictionary with per-layer sparsity and global sparsity.
    """
    total_zeros = 0
    total_params = 0
    per_layer = {}

    for i, m in enumerate(modules):
        w = m.weight.data
        n_zeros = (w == 0).sum().item()
        n_total = w.numel()
        total_zeros += n_zeros
        total_params += n_total
        per_layer[f"layer_{i}"] = n_zeros / n_total

    global_sparsity = total_zeros / total_params if total_params > 0 else 0.0
    return {"global": global_sparsity, "per_layer": per_layer}
