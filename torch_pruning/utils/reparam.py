"""Reparameterization modules for variance-aware pruning.

Two variants, both sharing the BaseReparamManager ABC:

1. **NormalizedResidualManager** (VNR, BN-based) — CANONICAL ("the BN trick"):
   Inserts BN(affine=False) before the target layer: input normalized by the LIVE
   EMA σ (running stats), trainable v_tilde = σ_cal·W (σ-scaling applied once at
   init, then frozen into v_tilde), μ folded into the bias. ‖v_tilde[:,k]‖ is the
   contribution score, so plain weight-decay on v_tilde = the pruning regularizer
   (magnitude pruning on the normalized net). Gradient w.r.t v_tilde is on unit-
   variance input → well-scaled. σ's two roles per the spec: (1) normalize input,
   kept updated by EMA; (2) seed v_tilde=σW at init, not updated thereafter.

2. **MeanResidualManager** — ABLATION (input-normalization removed):
   Decomposes z = W^T x + b into z = m + V^T(x - μ_x). σ is ABSENT from the forward
   (input not normalized); σ is only a detached factor in the ‖σ·v‖ score / aux reg.
   Carries the propagation / σ_out machinery (M2–M4). Use to ablate normalization.

Training-time only — merge_back() restores standard nn.Linear/nn.Conv2d before
pruning so the TP dependency graph, pruner, and compensation code stay untouched.

Efficiency (both use the effective-bias trick, no actual BN reshape on the hot path):
    BNResidual:   z = F.linear(x, v_tilde/σ_run, m − (v_tilde/σ_run)@μ_run)
    MeanResidual: z = F.linear(x, V, m − V @ μ_x)
"""

import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.propagate = False  # prevent duplicate output via root logger


def _is_main_rank():
    """Check if this is rank 0 (or non-DDP)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _log_info(msg):
    if _is_main_rank():
        logger.info(msg)


def _log_warning(msg):
    if _is_main_rank():
        logger.warning(msg)

# Numerical stability constants
MIN_VARIANCE = 1e-12
MIN_SIGMA = 1e-6
EPSILON = 1e-8


# =====================================================================
# Reparam module classes
# =====================================================================

class MeanResidualLinear(nn.Module):
    """Linear layer decomposed as z = m + V^T(x - μ_x), computed via effective bias.

    Carries a frozen sigma_x buffer (per-input-channel std from calibration). σ does NOT
    enter the forward — it stays a stop-grad factor used only by the pruning score
    ‖σ·v‖ and the optional λ‖σ·v‖ regularizer. Folding σ into the trainable produces the
    1/σ² overshoot (the deprecated bn variant).
    """

    def __init__(self, in_features, out_features, v, m, mu_x, sigma_x=None,
                 sigma_out_x=None, ema_momentum=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v = nn.Parameter(v)       # [out, in] — residual weights
        self.m = nn.Parameter(m)       # [out]     — channel means (frozen from WD)
        self.register_buffer('mu_x', mu_x)  # [in] — input mean (M4: EMA when training)
        if sigma_x is None:
            sigma_x = torch.ones(in_features, device=mu_x.device, dtype=mu_x.dtype)
        self.register_buffer('sigma_x', sigma_x)  # [in] — input std (score-only)
        if sigma_out_x is None:
            sigma_out_x = torch.ones(out_features, device=mu_x.device, dtype=mu_x.dtype)
        # [out] — output std (M3: branch weighting at residual joins)
        self.register_buffer('sigma_out_x', sigma_out_x)
        self.ema_momentum = float(ema_momentum)  # M4: per-step EMA on (μ, σ_in, σ_out)
                                                  # 0 = frozen (default, M1-M3 behavior).

    def forward(self, x):
        # Snapshot mu_x to a fresh storage (clone) so the EMA in-place update below
        # (when enabled) can mutate self.mu_x without breaking the next backward.
        # detach()+share-storage isn't enough — autograd's version check tracks the
        # underlying storage.
        mu_snap = self.mu_x.detach().clone() if self.training and self.ema_momentum > 0 \
            else self.mu_x
        eff_bias = self.m - self.v @ mu_snap  # [out] — tiny vector
        out = F.linear(x, self.v, eff_bias)
        if self.training and self.ema_momentum > 0:
            with torch.no_grad():
                in_dims = tuple(range(x.dim() - 1))  # all but last (channel) dim
                batch_mean = x.mean(dim=in_dims)
                batch_var = x.var(dim=in_dims, unbiased=False).clamp_(min=0.0)
                out_dims = tuple(range(out.dim() - 1))
                out_var = out.var(dim=out_dims, unbiased=False).clamp_(min=0.0)
                mom = self.ema_momentum
                # μ EMA only — do NOT compensate m here. Mutating self.m (a Parameter)
                # between forward and backward bumps autograd's version counter and
                # breaks loss.backward(). Standard BN takes the same approach: running
                # stats update during forward, affine params (γ/β here ≈ m) drift via
                # their own gradient. The eff_bias = m - v @ μ_x in the next forward
                # automatically reflects the new μ. Function-preserving m correction
                # remains available via the manual refresh_stats() between epochs.
                self.mu_x.add_(mom * (batch_mean - self.mu_x))
                self.sigma_x.add_(mom * (torch.sqrt(batch_var + 1e-5) - self.sigma_x))
                self.sigma_out_x.add_(mom * (torch.sqrt(out_var + 1e-5) - self.sigma_out_x))
        return out

    def merge_params(self):
        """Return (weight, bias) in standard nn.Linear convention."""
        weight = self.v.data.clone()
        bias = (self.m - self.v @ self.mu_x).data.clone()
        return weight, bias

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features} [MeanResidual]'


class MeanResidualConv2d(nn.Module):
    """Conv2d (groups=1) decomposed as z = m + V*(x - μ_x), computed via effective bias.

    For Conv2d, μ_x is the spatial-averaged per-channel input mean [C_in], sigma_x is the
    matching per-channel std (pooled over batch + spatial). σ stays out of the forward —
    score-only, broadcast over [None,:,None,None] when computing ‖σ·v‖.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, v, m, mu_x, sigma_x=None, sigma_out_x=None,
                 ema_momentum=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.v = nn.Parameter(v)       # [C_out, C_in, kH, kW]
        self.m = nn.Parameter(m)       # [C_out]
        self.register_buffer('mu_x', mu_x)  # [C_in] — input mean (M4: EMA when training)
        if sigma_x is None:
            sigma_x = torch.ones(in_channels, device=mu_x.device, dtype=mu_x.dtype)
        self.register_buffer('sigma_x', sigma_x)  # [C_in] — input std (score-only)
        if sigma_out_x is None:
            sigma_out_x = torch.ones(out_channels, device=mu_x.device, dtype=mu_x.dtype)
        # [C_out] — output std (M3: branch weighting at residual joins)
        self.register_buffer('sigma_out_x', sigma_out_x)
        self.ema_momentum = float(ema_momentum)  # M4: per-step EMA on (μ, σ_in, σ_out)

    def forward(self, x):
        # v.sum(dim=(2,3)) → [C_out, C_in], then @ mu_x → [C_out].
        # Snapshot mu_x (clone) when EMA is on so in-place mutation doesn't break
        # the saved-for-backward version check.
        mu_snap = self.mu_x.detach().clone() if self.training and self.ema_momentum > 0 \
            else self.mu_x
        eff_bias = self.m - self.v.sum(dim=(2, 3)) @ mu_snap
        out = F.conv2d(x, self.v, eff_bias, self.stride, self.padding,
                       self.dilation, self.groups)
        if self.training and self.ema_momentum > 0:
            with torch.no_grad():
                batch_mean = x.mean(dim=(0, 2, 3))
                batch_var = x.var(dim=(0, 2, 3), unbiased=False).clamp_(min=0.0)
                out_var = out.var(dim=(0, 2, 3), unbiased=False).clamp_(min=0.0)
                mom = self.ema_momentum
                # μ EMA only — m compensation would mutate a Parameter mid-graph
                # (autograd version-counter conflict on backward). See the Linear
                # forward for the full rationale. refresh_stats() between epochs is
                # the function-preserving alternative.
                self.mu_x.add_(mom * (batch_mean - self.mu_x))
                self.sigma_x.add_(mom * (torch.sqrt(batch_var + 1e-5) - self.sigma_x))
                self.sigma_out_x.add_(mom * (torch.sqrt(out_var + 1e-5) - self.sigma_out_x))
        return out

    def merge_params(self):
        """Return (weight, bias) in standard nn.Conv2d convention."""
        weight = self.v.data.clone()
        bias = (self.m - self.v.sum(dim=(2, 3)) @ self.mu_x).data.clone()
        return weight, bias

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding} [MeanResidual]')


class BNResidualLinear(nn.Module):
    """BN(affine=False) + Linear with mean-residual decomposition.

    Boss spec: normalize input by the LIVE EMA σ (running stat, not batch), trainable
    v_tilde = σ_cal·W (σ-scaling applied once at init, then frozen into v_tilde). Forward
    uses the effective-bias trick z = linear(x, v_tilde/σ_run, m − (v_tilde/σ_run)@μ_run)
    = v_tilde·(x−μ)/σ — equals BN-normalize-then-linear but with RUNNING (EMA) stats so it
    matches the Conv variant and "keep σ updated with EMA". ‖v_tilde[:,k]‖ = contribution.
    """

    def __init__(self, in_features, out_features, v_tilde, m, bn_running_mean,
                 bn_running_var, bn_momentum=0.1, sigma_out_x=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = nn.BatchNorm1d(in_features, affine=False, momentum=bn_momentum)
        self.bn.running_mean.copy_(bn_running_mean)
        self.bn.running_var.copy_(bn_running_var)
        self.v_tilde = nn.Parameter(v_tilde)  # [out, in] — operates on normalized input
        self.m = nn.Parameter(m)              # [out] — channel means
        # Per-OUTPUT-channel std (calibration), for propagation branch weighting (M3).
        if sigma_out_x is None:
            sigma_out_x = torch.ones(out_features, device=m.device, dtype=m.dtype)
        self.register_buffer('sigma_out_x', sigma_out_x)

    def forward(self, x):
        # Update EMA running stats in train mode (no grad), then normalize by RUNNING
        # stats via the effective-bias trick — not batch stats (boss: normalize by EMA σ).
        if self.training:
            with torch.no_grad():
                D = x.shape[-1]
                self.bn(x.reshape(-1, D)) if x.dim() >= 3 else self.bn(x)
        sigma = torch.sqrt(self.bn.running_var + self.bn.eps)
        mu = self.bn.running_mean
        # Effective-bias trick: ṽ·(x−μ)/σ + m, computed without materializing the BN.
        w_eff = self.v_tilde / sigma[None, :]   # un-fold σ for the forward (ṽ/σ = W)
        eff_bias = self.m - w_eff @ mu          # un-fold μ:  m − W·μ = b
        return F.linear(x, w_eff, eff_bias)

    def merge_params(self):
        """Recover standard (weight, bias) using BN running stats."""
        sigma = torch.sqrt(self.bn.running_var + self.bn.eps)
        mu = self.bn.running_mean
        w_eff = self.v_tilde.data / sigma[None, :]
        b_eff = self.m.data - w_eff @ mu
        return w_eff.clone(), b_eff.clone()

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features} [BNResidual]'


class BNResidualConv2d(nn.Module):
    """BN2d(affine=False) stats tracker + Conv2d with effective-bias trick.

    BN auto-updates running stats during training (no frozen σ).
    Forward uses the effective-bias trick (not actual BN normalization)
    to avoid padding artifacts at borders.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, v_tilde, m, bn_running_mean, bn_running_var,
                 bn_momentum=0.1, sigma_out_x=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bn = nn.BatchNorm2d(in_channels, affine=False, momentum=bn_momentum)
        self.bn.running_mean.copy_(bn_running_mean)
        self.bn.running_var.copy_(bn_running_var)
        self.v_tilde = nn.Parameter(v_tilde)  # [C_out, C_in, kH, kW]
        self.m = nn.Parameter(m)              # [C_out]
        # Per-OUTPUT-channel std (calibration), for propagation branch weighting (M3).
        if sigma_out_x is None:
            sigma_out_x = torch.ones(out_channels, device=m.device, dtype=m.dtype)
        self.register_buffer('sigma_out_x', sigma_out_x)

    def forward(self, x):
        # Update BN running stats in training mode (no gradient, discard output)
        if self.training:
            with torch.no_grad():
                self.bn(x)
        # Use effective-bias trick (avoids padding border artifacts)
        sigma = torch.sqrt(self.bn.running_var + self.bn.eps)
        mu = self.bn.running_mean
        sigma_bc = sigma[None, :, None, None]
        w_eff = self.v_tilde / sigma_bc
        eff_bias = self.m - w_eff.sum(dim=(2, 3)) @ mu
        return F.conv2d(x, w_eff, eff_bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def merge_params(self):
        """Recover standard (weight, bias) using BN running stats."""
        sigma = torch.sqrt(self.bn.running_var + self.bn.eps)
        mu = self.bn.running_mean
        sigma_bc = sigma[None, :, None, None]
        w_eff = (self.v_tilde.data / sigma_bc).clone()
        b_eff = (self.m.data - w_eff.sum(dim=(2, 3)) @ mu).clone()
        return w_eff, b_eff

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding} [BNResidual]')


# =====================================================================
# BN folding utility
# =====================================================================

def fold_all_conv_bn(model: nn.Module):
    """Fold each BatchNorm that immediately follows a Conv2d/Linear into that layer's
    weights. Replaces the BN with nn.Identity.

    Walks the consecutive children of EVERY module (not just nn.Sequential) for
    Conv/Linear → BN2d/BN1d pairs, where child-definition order = forward order (holds for
    ResNet/ConvNeXt/VGG/MobileNet). The old Sequential-only walk MISSED ResNet bottleneck
    main-path BNs (conv1→bn1, conv2→bn2, conv3→bn3 and the stem conv1→bn1 are sibling block
    ATTRIBUTES, not in a Sequential) — only the downsample Sequential folded, so most BN
    scales never entered the scored weights. A num_features==out-channels guard rejects
    accidental non-adjacent pairs. conv_key/bn_key are siblings under one parent, so the
    (parent_name, conv_key, bn_key, bn_type) records stay valid for reinsert_bn.
    """
    folded = 0
    folded_locations = []
    named_modules = {id(m): n for n, m in model.named_modules()}

    for parent_module in list(model.modules()):
        parent_name = named_modules.get(id(parent_module), "")
        children = list(parent_module.named_children())
        for i in range(len(children) - 1):
            key_a, mod_a = children[i]
            key_b, mod_b = children[i + 1]
            if not isinstance(mod_a, (nn.Conv2d, nn.Linear)):
                continue
            if not isinstance(mod_b, (nn.BatchNorm2d, nn.BatchNorm1d)):
                continue
            # adjacency guard: BN channels must match the layer's output channels
            out_ch = mod_a.out_channels if isinstance(mod_a, nn.Conv2d) else mod_a.out_features
            if mod_b.num_features != out_ch:
                continue

            bn = mod_b
            bn_type = type(bn).__name__  # "BatchNorm2d" or "BatchNorm1d"
            eps = bn.eps
            mean = bn.running_mean          # [C_out]
            var = bn.running_var            # [C_out]
            sigma = torch.sqrt(var + eps)   # [C_out]

            if bn.affine:
                gamma, beta = bn.weight, bn.bias
            else:
                gamma = torch.ones_like(mean)
                beta = torch.zeros_like(mean)

            scale = gamma / sigma           # [C_out]

            # Fold into weight
            if isinstance(mod_a, nn.Conv2d):
                mod_a.weight.data.mul_(scale[:, None, None, None])
            else:
                mod_a.weight.data.mul_(scale[:, None])

            # Fold into bias (create if missing)
            b = mod_a.bias.data if mod_a.bias is not None else torch.zeros(
                scale.shape[0], device=mod_a.weight.device)
            b_eff = scale * (b - mean) + beta

            if mod_a.bias is None:
                mod_a.bias = nn.Parameter(b_eff.detach().clone())
            else:
                mod_a.bias.data.copy_(b_eff)

            # Replace BN with Identity
            setattr(parent_module, key_b, nn.Identity())
            folded_locations.append((parent_name, key_a, key_b, bn_type))
            _log_info(f"fold_all_conv_bn: folded {key_b}(BN) into {key_a}(Conv/Linear)")
            folded += 1

    _log_info(f"fold_all_conv_bn: total {folded} BNs folded into preceding layers")
    return folded, folded_locations


def reinsert_bn(model: nn.Module, folded_locations):
    """Re-insert fresh BatchNorm layers at locations previously folded by fold_all_conv_bn.

    Each Identity placeholder is replaced with a new BN (γ=1, β=0, default running stats).
    The preceding Conv/Linear's out_channels determines the BN num_features.

    Args:
        model: The (possibly pruned) model.
        folded_locations: List of (parent_name, conv_key, bn_key, bn_type) from fold_all_conv_bn.

    Returns:
        Number of BN layers re-inserted.
    """
    name_to_module = dict(model.named_modules())
    reinserted = 0
    for parent_name, conv_key, bn_key, bn_type in folded_locations:
        parent = name_to_module.get(parent_name)
        if parent is None:
            _log_warning(f"reinsert_bn: parent '{parent_name}' not found, skipping")
            continue
        current = getattr(parent, bn_key, None)
        if not isinstance(current, nn.Identity):
            _log_warning(f"reinsert_bn: {parent_name}.{bn_key} is {type(current).__name__}, not Identity, skipping")
            continue
        conv = getattr(parent, conv_key, None)
        if conv is None or not isinstance(conv, (nn.Conv2d, nn.Linear)):
            _log_warning(f"reinsert_bn: {parent_name}.{conv_key} is not Conv/Linear, skipping")
            continue
        num_features = conv.weight.shape[0]
        if bn_type == "BatchNorm2d":
            new_bn = nn.BatchNorm2d(num_features).to(conv.weight.device)
        else:
            new_bn = nn.BatchNorm1d(num_features).to(conv.weight.device)
        setattr(parent, bn_key, new_bn)
        reinserted += 1
    _log_info(f"reinsert_bn: re-inserted {reinserted} fresh BN layers")
    return reinserted


# =====================================================================
# Base class (ABC) for reparameterization managers
# =====================================================================

def _residual_weight(reparam):
    """Return the residual weight tensor (v_tilde if present, else v)."""
    if hasattr(reparam, 'v_tilde'):
        return reparam.v_tilde
    return reparam.v


def _contribution_weight(reparam):
    """Return the contribution weight σ·v — the quantity ‖·‖ of which is the paper's
    per-layer pruning score.

    - BN variant (v_tilde already absorbs σ): returns v_tilde directly.
    - Mean variant (σ kept as stop-grad buffer): returns sigma_x · v with broadcast.
    Falls back to _residual_weight when sigma_x is unavailable (back-compat).
    """
    w = _residual_weight(reparam)
    if hasattr(reparam, 'v_tilde'):
        return w
    sigma = getattr(reparam, 'sigma_x', None)
    if sigma is None:
        return w
    if w.dim() == 4:
        return w * sigma[None, :, None, None]
    return w * sigma[None, :]


def _channel_group_norm(w, norm_dim):
    """Per-channel L2 norm grouping weight `w` like a 2D [out, in] `norm(dim=norm_dim)`.

    norm_dim=0 → per-input-channel (reduce output dim); norm_dim=1 → per-output-channel.
    For conv (4D [out, in, kH, kW]) the kernel dims are reduced INTO the group, not
    merged into the channel axis. The old `w.flatten(1).norm(dim=0)` gave a
    [in·kH·kW] vector for k>1 convs (per kernel tap, not per channel) — wrong grouping
    for channel pruning / the §2 per-input-channel contribution score.
    """
    if w.dim() == 4:
        keep = 1 if norm_dim == 0 else 0
        reduce_dims = tuple(d for d in range(w.dim()) if d != keep)
        return w.pow(2).sum(dim=reduce_dims).sqrt()
    return w.norm(p=2, dim=norm_dim)


class BaseReparamManager(ABC):
    """Abstract lifecycle orchestrator for reparameterization managers.

    Concrete subclasses implement _calibrate, _make_reparam, refresh_stats,
    regularization_loss, and channel_stats.
    """

    def __init__(self, model, target_names, device, lambda_reg=0.01,
                 max_batches=200, scale_invariant=False, reparam_target="fc2"):
        if not target_names:
            raise ValueError("target_names cannot be empty")
        if max_batches <= 0:
            raise ValueError(f"max_batches must be positive, got {max_batches}")
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be non-negative, got {lambda_reg}")
        if reparam_target not in ("fc1", "fc2"):
            raise ValueError(f"reparam_target must be 'fc1' or 'fc2', got {reparam_target}")

        self.model = model
        self.target_names = target_names
        self.device = device
        self.lambda_reg = lambda_reg
        self.max_batches = max_batches
        self.scale_invariant = scale_invariant
        self.reparam_target = reparam_target
        # fc2: column norms (dim=0) → per input channel (= intermediate dim)
        # fc1: row norms (dim=1) → per output channel (= intermediate dim)
        self.norm_dim = 1 if reparam_target == "fc1" else 0
        self._active = False
        self._reparam_modules = OrderedDict()  # name → reparam module

    @property
    def is_active(self):
        return self._active

    def reparameterize(self, calibration_loader):
        """Calibrate statistics and replace target modules with reparam variants."""
        if self._active:
            raise RuntimeError("Already reparameterized. Call merge_back() first.")

        targets = self._resolve_targets()
        cal_dict = self._calibrate(targets, calibration_loader)

        for name, module in targets.items():
            reparam = self._make_reparam(module, cal_dict[name])
            # Store initial norms for scale-invariant L_{2,1}. Use the contribution
            # weight σ·v so the scale-invariance baseline matches the new ‖σ·v‖ score.
            if self.scale_invariant:
                w = _contribution_weight(reparam)
                init_norms = _channel_group_norm(w.detach(), self.norm_dim).clamp(min=EPSILON)
                reparam.register_buffer('v_init_norms', init_norms)
            self._replace_module(name, reparam)
            self._reparam_modules[name] = reparam

        self._active = True
        cls_name = type(self).__name__
        _log_info(f"{cls_name}: reparameterized {len(targets)} {self.reparam_target} modules"
                     f"{' (scale-invariant)' if self.scale_invariant else ''}"
                     f" [norm_dim={self.norm_dim}]")
        # Log initial V-norms as baseline for tracking regularization progress
        self.log_channel_stats(verbose=False)

    def _sync_bn_stats(self):
        """All-reduce reparam-module buffers across DDP ranks (no-op if not distributed).

        Covers both variants:
          - BN variant: reparam.bn.running_mean / running_var
          - Mean variant: reparam.mu_x / sigma_x / sigma_out_x
        Mean buffers diverge per-rank under M4 per-step EMA (each rank updates from
        its local DistributedSampler shard). Called at merge_back so the saved
        canonical state is rank-consistent.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return
        synced_count = 0
        for reparam in self._reparam_modules.values():
            if hasattr(reparam, 'bn'):  # BN variant
                torch.distributed.all_reduce(reparam.bn.running_mean)
                reparam.bn.running_mean.div_(world_size)
                torch.distributed.all_reduce(reparam.bn.running_var)
                reparam.bn.running_var.div_(world_size)
                synced_count += 1
            # Mean variant has these buffers; BN variant also has m (Parameter, sync via DDP)
            for buf_name in ('mu_x', 'sigma_x', 'sigma_out_x'):
                if hasattr(reparam, buf_name):
                    buf = getattr(reparam, buf_name)
                    torch.distributed.all_reduce(buf)
                    buf.div_(world_size)
        _log_info(f"{type(self).__name__}: synced reparam buffers across {world_size} ranks "
                  f"({len(self._reparam_modules)} layers, {synced_count} with BN)")

    def merge_back(self):
        """Restore standard nn.Linear/nn.Conv2d from reparam modules."""
        if not self._active:
            return

        self._sync_bn_stats()

        for name, reparam in self._reparam_modules.items():
            weight, bias = reparam.merge_params()
            standard = self._make_standard(reparam, weight, bias)
            self._replace_module(name, standard)

        self._reparam_modules.clear()
        self._active = False
        _log_info(f"{type(self).__name__}: merged back to standard modules")

    def reparam_param_ids(self):
        """Return set of id(p) for all trainable reparam parameters."""
        ids = set()
        for reparam in self._reparam_modules.values():
            w = _residual_weight(reparam)
            ids.add(id(w))
            ids.add(id(reparam.m))
        return ids

    def save_vnorm_snapshot(self, save_dir):
        """Save per-channel ‖σ·v‖ contribution norms as .pt file (BN variant: ‖v_tilde‖)."""
        vnorms = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            w = _contribution_weight(reparam).detach()
            vnorms[name] = _channel_group_norm(w, self.norm_dim).cpu()
        self._last_vnorms = vnorms

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "reparam_vnorms.pt")
        torch.save(vnorms, path)
        _log_info(f"Saved V-norm snapshot to {path} ({len(vnorms)} layers)")

    def log_channel_stats(self, verbose=True):
        """Log per-layer and aggregate residual-weight norm summary.

        Args:
            verbose: If True, log per-layer detail. Always logs aggregate summary.
        """
        stats = self.channel_stats()
        axis_desc = "row" if self.norm_dim == 1 else "col"
        target_desc = f"fc1 ({axis_desc}-norms)" if self.reparam_target == "fc1" else f"fc2 ({axis_desc}-norms)"

        if verbose:
            _log_info(f"V-norm per-layer ({target_desc}):")
            for name, s in stats.items():
                short = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
                _log_info(
                    f"  {short}: mean={s['v_col_norm_mean']:.4f} std={s['v_col_norm_std']:.4f} "
                    f"min={s['v_col_norm_min']:.4f} max={s['v_col_norm_max']:.4f} "
                    f"<0.01={s['frac_below_0.01']:.1%} <0.1={s['frac_below_0.1']:.1%} "
                    f"m={s['m_mean']:.4f}±{s['m_std']:.4f}"
                )

        # Aggregate summary across all layers (uses σ·v contribution norms)
        summ = self.vnorm_summary()
        if summ:
            _log_info(
                f"V-norm aggregate ({target_desc}, {summ['n_layers']} layers, "
                f"{summ['n_channels']} channels): "
                f"mean={summ['mean']:.4f} median={summ['median']:.4f} "
                f"std={summ['std']:.4f} "
                f"<0.01={summ['frac_below_0.01']:.1%} "
                f"<0.1={summ['frac_below_0.1']:.1%} "
                f"<1.0={summ['frac_below_1.0']:.1%}"
            )

    def vnorm_summary(self):
        """Global ‖σ·v‖ distribution across all reparam'd layers (one flat vector).

        Returns dict (or {} if inactive) with mean/median/std/min/max and the
        natural-sparsity fractions frac_below_{0.01,0.1,1.0} — the headline signal
        for the λ regularization sweep (E4): a growing left tail = induced channel
        sparsity. Machine-friendly (all python floats/ints) for metrics.jsonl.
        """
        if not self._reparam_modules:
            return {}
        all_norms = []
        for reparam in self._reparam_modules.values():
            v = _contribution_weight(reparam).detach()
            all_norms.append(_channel_group_norm(v, self.norm_dim))
        all_norms = torch.cat(all_norms)
        n_total = int(all_norms.numel())
        return {
            "n_layers": len(self._reparam_modules),
            "n_channels": n_total,
            "mean": all_norms.mean().item(),
            "median": all_norms.median().item(),
            "std": all_norms.std().item(),
            "min": all_norms.min().item(),
            "max": all_norms.max().item(),
            "frac_below_0.01": (all_norms < 0.01).sum().item() / n_total,
            "frac_below_0.1": (all_norms < 0.1).sum().item() / n_total,
            "frac_below_1.0": (all_norms < 1.0).sum().item() / n_total,
        }

    def input_channel_scores(self):
        """Per-INPUT-channel contribution norm ‖σ·v‖ per layer (variant-agnostic).

        Returns OrderedDict[name → tensor of length in_channels^l]. norm_dim is forced
        to 0 (reduce output + kernel → one score per input channel) regardless of
        self.norm_dim, because channel pruning always ranks input channels. Reads
        _contribution_weight: σ·v for the mean variant, v_tilde (= σ·W on normalized
        input) for the BN variant — both equal the contribution score. This is the
        "per_layer" criterion the E0 adapter (NormalizedNetImportance) consumes.

        Call while ACTIVE (before merge_back).
        """
        out = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            w = _contribution_weight(reparam).detach()
            out[name] = _channel_group_norm(w, 0)  # 0 → per input channel
        return out

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _calibrate(self, targets, loader):
        """Calibrate input statistics. Returns dict[name → calibration_data]."""
        ...

    @abstractmethod
    def _make_reparam(self, module, calibration_data):
        """Create reparam module from standard nn.Linear/Conv2d + calibration data."""
        ...

    @abstractmethod
    def refresh_stats(self, loader):
        """Re-estimate input statistics (function-preserving)."""
        ...

    @abstractmethod
    def regularization_loss(self):
        """Return scalar regularization loss on device."""
        ...

    @abstractmethod
    def channel_stats(self):
        """Return dict of {layer_name: {stat_name: value}}."""
        ...

    # ------------------------------------------------------------------
    # Shared internal helpers
    # ------------------------------------------------------------------
    def _resolve_targets(self):
        """Resolve target_names to an OrderedDict of name → module."""
        name_to_module = dict(self.model.named_modules())
        targets = OrderedDict()
        cls_name = type(self).__name__
        for name in self.target_names:
            if name not in name_to_module:
                _log_warning(f"{cls_name}: target '{name}' not found, skipping")
                continue
            module = name_to_module[name]
            if isinstance(module, nn.Linear):
                targets[name] = module
            elif isinstance(module, nn.Conv2d) and module.groups == 1:
                targets[name] = module
            else:
                _log_warning(f"{cls_name}: skipping '{name}' "
                               f"({type(module).__name__}, groups={getattr(module, 'groups', 'N/A')})")
        return targets

    def _calibrate_mu(self, targets, loader):
        """Back-compat shim: returns dict[name → μ_x tensor] only.

        Prefer _calibrate_stats for new code (returns (μ, σ_in, σ_out) tuples).
        """
        stats = self._calibrate_stats(targets, loader)
        return OrderedDict((name, mu) for name, (mu, _si, _so) in stats.items())

    def _calibrate_stats(self, targets, loader, eps=1e-5):
        """Estimate (μ_x, σ_x, σ_out_x) for each target module via forward hooks (one pass).

        Returns dict[name → (μ_x, σ_x, σ_out_x)]. Input stats per-input-channel; output std
        per-output-channel. All pooled over batch + spatial to match BatchNorm conventions.
        σ uses the same eps as the bn variant. Never trained; consumers must .detach().

        σ_out_x added in M3 for branch weighting at residual joins
        (σ_branch / Σ σ_branches).
        """
        accumulators = {}
        hooks = []

        def _per_channel_reduce(t, channel_dim):
            """Return (sum, sum_sq, count) reduced over all-but-channel axes.

            channel_dim is the axis index of the channel/feature dim:
              - Linear: channel = last dim (handles 2D [N,D], 3D [N,T,D], 4D
                channels-last [N,H,W,C]).
              - Conv2d: channel = dim 1 (4D [N,C,H,W]).
            """
            if t.dim() < 2:
                return None
            reduce_dims = tuple(i for i in range(t.dim()) if i != channel_dim)
            n = 1
            for i in reduce_dims:
                n *= t.shape[i]
            return t.sum(dim=reduce_dims), (t * t).sum(dim=reduce_dims), n

        for name, module in targets.items():
            acc = {
                'in_sum': None, 'in_sum_sq': None, 'in_count': 0,
                'out_sum': None, 'out_sum_sq': None, 'out_count': 0,
            }
            accumulators[name] = acc
            # Linear-like (channel = last dim, all ranks): nn.Linear, MeanResidualLinear,
            #   BNResidualLinear — covers initial calibration (plain modules) AND
            #   refresh_stats (reparam'd modules) for ViT/ConvNeXt 3D/4D inputs.
            # Conv-like (channel = dim 1, 4D NCHW): nn.Conv2d, MeanResidualConv2d,
            #   BNResidualConv2d.
            is_linear = isinstance(module, (nn.Linear, MeanResidualLinear, BNResidualLinear))
            ch_in = -1 if is_linear else 1
            ch_out = -1 if is_linear else 1

            def make_hook(acc_ref, ch_in=ch_in, ch_out=ch_out):
                def hook(mod, inp, out):
                    x = inp[0].detach()
                    ch_in_eff = (x.dim() - 1) if ch_in == -1 else ch_in
                    reduced = _per_channel_reduce(x, ch_in_eff)
                    if reduced is not None:
                        s, ss, n = reduced
                        if acc_ref['in_sum'] is None:
                            acc_ref['in_sum'] = s; acc_ref['in_sum_sq'] = ss
                        else:
                            acc_ref['in_sum'] += s; acc_ref['in_sum_sq'] += ss
                        acc_ref['in_count'] += n

                    o = out.detach()
                    ch_out_eff = (o.dim() - 1) if ch_out == -1 else ch_out
                    reduced_o = _per_channel_reduce(o, ch_out_eff)
                    if reduced_o is not None:
                        s_o, ss_o, n_o = reduced_o
                        if acc_ref['out_sum'] is None:
                            acc_ref['out_sum'] = s_o; acc_ref['out_sum_sq'] = ss_o
                        else:
                            acc_ref['out_sum'] += s_o; acc_ref['out_sum_sq'] += ss_o
                        acc_ref['out_count'] += n_o
                return hook

            hooks.append(module.register_forward_hook(make_hook(acc)))

        try:
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= self.max_batches:
                        break
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    else:
                        images = batch
                    images = images.to(self.device, non_blocking=True)
                    self.model(images)
        finally:
            for h in hooks:
                h.remove()

        def _finalize(s, ss, n, fallback_dim):
            if s is not None and n > 0:
                mu = s / n
                var = (ss / n) - mu * mu
                sigma = torch.sqrt(var.clamp_(min=0.0) + eps)
                return mu, sigma
            return (torch.zeros(fallback_dim, device=self.device),
                    torch.ones(fallback_dim, device=self.device))

        stats_dict = OrderedDict()
        for name, acc in accumulators.items():
            module = targets[name]
            if isinstance(module, nn.Linear):
                in_dim, out_dim = module.in_features, module.out_features
            elif isinstance(module, nn.Conv2d):
                in_dim, out_dim = module.in_channels, module.out_channels
            elif hasattr(module, 'mu_x'):
                in_dim = module.mu_x.numel()
                out_dim = module.sigma_out_x.numel() if hasattr(module, 'sigma_out_x') else in_dim
            else:
                in_dim = out_dim = 0

            mu_in, sigma_in = _finalize(acc['in_sum'], acc['in_sum_sq'], acc['in_count'], in_dim)
            _mu_out, sigma_out = _finalize(acc['out_sum'], acc['out_sum_sq'], acc['out_count'], out_dim)

            if acc['in_count'] == 0:
                _log_warning(f"{type(self).__name__}: no data for '{name}', using μ=0 σ=1")
            stats_dict[name] = (mu_in, sigma_in, sigma_out)

        avg = sum(a['in_count'] for a in accumulators.values()) // max(len(accumulators), 1)
        _log_info(f"{type(self).__name__}: calibrated (μ,σ_in,σ_out) for {len(stats_dict)} "
                     f"modules ({avg} samples/positions avg)")
        return stats_dict

    def _make_standard(self, reparam, weight, bias):
        """Create standard nn.Linear/Conv2d from merged params."""
        if isinstance(reparam, (MeanResidualLinear, BNResidualLinear)):
            linear = nn.Linear(reparam.in_features, reparam.out_features, bias=True)
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            return linear.to(self.device)
        elif isinstance(reparam, (MeanResidualConv2d, BNResidualConv2d)):
            conv = nn.Conv2d(
                reparam.in_channels, reparam.out_channels,
                kernel_size=reparam.kernel_size, stride=reparam.stride,
                padding=reparam.padding, dilation=reparam.dilation,
                groups=reparam.groups, bias=True)
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(bias)
            return conv.to(self.device)
        raise TypeError(f"Unsupported reparam type: {type(reparam)}")

    def _replace_module(self, dotted_name, new_module):
        """Replace a module in self.model by its dotted name path."""
        parts = dotted_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)


# =====================================================================
# MeanResidualManager (original)
# =====================================================================

class MeanResidualManager(BaseReparamManager):
    """Lifecycle orchestrator for mean-residual reparameterization.

    Trainable v = W (effective weight on raw input), forward v·(x−μ) + m. Gradient
    ∝ (x−μ): mean-centered, uniform W-space lr → fine-tunes like the plain baseline.
    This is the function-/gradient-correct parametrization (no 1/σ² overshoot, unlike
    NormalizedResidualManager's v_tilde = W·σ).

    Usage:
        mgr = MeanResidualManager(model, target_names, device, lambda_reg=0.01)
        mgr.reparameterize(train_loader)   # calibrate μ_x, replace modules
        ...  # train with mgr.regularization_loss() as aux loss
        mgr.merge_back()                   # restore standard modules before pruning

    TODO (pruning, deferred — base training is the current focus):
      The NCI contribution-variance score is ‖v·σ‖, but this manager carries no σ
      (it folds away σ entirely; σ cancels in the forward). To prune on the mean path:
        1. Calibrate and store a fixed per-input-channel σ buffer (BN-EMA) on each
           MeanResidual* module at reparameterize() time.
        2. Score channels by ‖v·σ‖ (not ‖v‖) in channel_stats()/importance.
        3. If contribution-variance *regularization* is wanted, add it as an EXPLICIT
           penalty λ‖v·σ‖ in regularization_loss() — do NOT fold σ into the trainable
           (that reintroduces the 1/σ² optimizer-geometry overshoot). This decouples the
           whitening penalty from the optimizer step.
    """

    def __init__(self, model, target_names, device, lambda_reg=0.01, max_batches=200,
                 normalize=False, scale_invariant=False, reparam_target="fc2",
                 ema_momentum=0.0):
        super().__init__(model, target_names, device, lambda_reg=lambda_reg,
                         max_batches=max_batches,
                         scale_invariant=(normalize or scale_invariant),
                         reparam_target=reparam_target)
        # M4: per-step EMA momentum for (μ, σ_in, σ_out) inside Mean module forward.
        # 0 = frozen at calibration (default, original M1–M3 behavior). Recommended
        # for from-scratch / long training: 0.01 (slow). Faster than 0.01 risks
        # aug-driven drift (see --norm_bn_momentum investigation).
        if not 0.0 <= float(ema_momentum) <= 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1], got {ema_momentum}")
        self.ema_momentum = float(ema_momentum)

    # Backward-compat alias
    def refresh_mu(self, loader):
        """Alias for refresh_stats (backward compatibility)."""
        return self.refresh_stats(loader)

    def sync_ema_buffers(self):
        """All-reduce (μ_x, σ_x, σ_out_x) across DDP ranks; no-op on single rank.

        Per-step M4 EMA updates each Mean module's buffers from the local rank's
        DistributedSampler shard, so without explicit sync the buffers diverge over
        training. Call this once per epoch (cheap) when --mu_ema_momentum > 0 to
        keep ranks coherent and propagation_importance / build_propagation_topology
        deterministic across ranks. merge_back's _sync_bn_stats also calls into the
        same all-reduce logic, so saved checkpoints are always rank-consistent.

        Known interactions (document for users):
          - Optimizer state staleness: after merge_back, the v / m Parameters are
            no longer part of self.model (replaced by plain weight/bias). Continuing
            optimizer.step() after merge_back updates the orphaned Parameters with
            no effect on the model. Don't keep training after merge_back; build a
            fresh optimizer for the merged model if needed.
          - Gradient accumulation: M4 per-step EMA fires once per FORWARD, not per
            optimizer step. With N micro-batches per effective step, the EMA
            absorbs N batches' worth of drift in one effective step — divide
            --mu_ema_momentum by N if you want comparable behavior to non-
            accumulating runs.
          - Mixed precision (autocast): not exercised. Forward inside autocast
            casts v to fp16/bf16; mu_snap.clone() preserves dtype; batch_var
            computed in fp16 risks catastrophic cancellation. Recommended: keep
            buffers in fp32 (default) and only autocast the matmul ops.
        """
        self._sync_bn_stats()

    def _calibrate(self, targets, loader):
        """Calibrate (μ_x, σ_x). Returns dict[name → (μ_x, σ_x)]."""
        return self._calibrate_stats(targets, loader)

    def _make_reparam(self, module, stats):
        """Create MeanResidual* module from standard nn.Linear/Conv2d.

        stats: (mu_x, sigma_x, sigma_out_x) tuple from _calibrate_stats. Both sigmas are
        stored as frozen buffers — sigma_x for ‖σ·v‖ scoring/regularization (M1),
        sigma_out_x for residual branch weighting (M3). Neither enters the forward.
        """
        mu_x, sigma_x, sigma_out_x = stats
        if isinstance(module, nn.Linear):
            w = module.weight.data.clone()
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_features, device=self.device)
            m = b + w @ mu_x
            reparam = MeanResidualLinear(
                module.in_features, module.out_features,
                v=w, m=m, mu_x=mu_x, sigma_x=sigma_x, sigma_out_x=sigma_out_x,
                ema_momentum=self.ema_momentum)
            return reparam.to(self.device)

        elif isinstance(module, nn.Conv2d):
            w = module.weight.data.clone()
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_channels, device=self.device)
            m = b + w.sum(dim=(2, 3)) @ mu_x
            reparam = MeanResidualConv2d(
                module.in_channels, module.out_channels,
                kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation,
                groups=module.groups, v=w, m=m, mu_x=mu_x, sigma_x=sigma_x,
                sigma_out_x=sigma_out_x, ema_momentum=self.ema_momentum)
            return reparam.to(self.device)

        raise TypeError(f"Unsupported module type: {type(module)}")

    def refresh_stats(self, loader):
        """Re-estimate (μ_x, σ_x). μ refresh is function-preserving (m adjusts to absorb
        Δμ); σ refresh is forward-irrelevant (σ does not enter forward — score-only).

        m_new = m_old + V @ (μ_old - μ_new); μ_x and σ_x buffers updated to fresh stats.
        """
        if not self._active:
            return

        targets = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            targets[name] = reparam

        stats_new_dict = self._calibrate_stats(targets, loader)

        for name, reparam in self._reparam_modules.items():
            mu_old = reparam.mu_x
            mu_new, sigma_new, sigma_out_new = stats_new_dict[name]
            # Function-preserving: forward = v(x − μ) + m ⇒ m_new = m_old + v·(μ_new − μ_old).
            delta = mu_new - mu_old
            with torch.no_grad():
                if hasattr(reparam, 'in_features'):  # Linear
                    reparam.m.data.add_(reparam.v.detach() @ delta)
                elif hasattr(reparam, 'in_channels'):  # Conv2d
                    reparam.m.data.add_(reparam.v.detach().sum(dim=(2, 3)) @ delta)
                reparam.mu_x.copy_(mu_new)
                reparam.sigma_x.copy_(sigma_new)
                if hasattr(reparam, 'sigma_out_x'):
                    reparam.sigma_out_x.copy_(sigma_out_new)

        _log_info("MeanResidualManager: refreshed (μ_x, σ_x, σ_out_x) — μ function-preserving, σ score-only")

    def regularization_loss(self):
        """L_{2,1} regularization on ‖σ_x·v‖ along self.norm_dim — the paper's
        contribution-score penalty.

        σ is detached (stop-grad) so the penalty pushes v down without backprop'ing
        through the data-measured spread. fc2 (dim=0): column norms → per input channel;
        fc1 (dim=1): row norms → per output channel. If scale_invariant=True, each norm
        is divided by its σ·v init value (set in BaseReparamManager.reparameterize).
        """
        loss = torch.tensor(0.0, device=self.device)
        for reparam in self._reparam_modules.values():
            v = reparam.v
            sigma = reparam.sigma_x.detach()
            if v.dim() == 4:                     # conv: [C_out, C_in, kH, kW]
                w_eff = v * sigma[None, :, None, None]
            else:                                 # linear: [out, in]
                w_eff = v * sigma[None, :]
            norms = _channel_group_norm(w_eff, self.norm_dim)
            if self.scale_invariant and hasattr(reparam, 'v_init_norms'):
                loss = loss + (norms / reparam.v_init_norms).sum()
            else:
                loss = loss + norms.sum()
        return self.lambda_reg * loss

    def build_propagation_topology(self, example_inputs, p=2):
        """Build the per-layer downstream + branch-weight topology via tp.DependencyGraph.

        Returns OrderedDict[name → list of (downstream_name, weight)]:
          - sequential: single downstream, weight 1.0;
          - residual join: each branch's single downstream is the merged consumer,
            with weight σ_branch^p / Σ σ_branches^p (paper's σ_c^p/(σ_a^p+σ_b^p));
          - fan-out (no add, layer feeds multiple consumers): each downstream weight = 1.0
            (importance accumulates additively across consumers);
          - terminal (no reparam'd consumer downstream): empty list — uses I_out seed.

        σ_branch is the per-branch mean of `sigma_out_x` (scalar). Per-channel branch
        weights are out of M3 scope.

        Args:
            p: skip-connection exponent — MUST match the `p` passed to
                propagation_importance (default 2 = variance / VBP-consistent; 1 = std).
            example_inputs: torch.Tensor (or list/dict) suitable for one forward pass —
                used by DepGraph to trace the computational graph on `self.model`.
        """
        try:
            import torch_pruning as tp
        except ImportError as e:
            raise RuntimeError("tp.DependencyGraph required for build_propagation_topology") from e

        # tp.DependencyGraph traces autograd-recognizable types (nn.Linear, nn.Conv2d, ...);
        # MeanResidual* are custom nn.Module subclasses → DG won't add nodes for them.
        # Workaround: temporarily swap each reparam'd module for a plain nn.Linear/Conv2d
        # built via merge_params (function-preserving at this point). Trace, then swap back.
        # The swap loop is inside the try/finally too so that ANY exception (including
        # mid-swap merge_params / .to(device) failures) still restores the original
        # reparam'd modules — never leave the user's model half-swapped.
        saved = OrderedDict()
        try:
            for name, rp in list(self._reparam_modules.items()):
                weight, bias = rp.merge_params()
                # Recognize by attribute so BOTH variants swap (MeanResidual* and
                # BNResidual* share in_features / in_channels). DG can't trace the custom
                # reparam modules, so we temporarily install function-preserving plain
                # nn.Linear/Conv2d, trace, then restore in the finally block.
                if hasattr(rp, 'in_features'):    # Linear (mean or bn)
                    tmp = nn.Linear(rp.in_features, rp.out_features, bias=True)
                    tmp.weight.data.copy_(weight); tmp.bias.data.copy_(bias)
                elif hasattr(rp, 'in_channels'):  # Conv2d (mean or bn)
                    tmp = nn.Conv2d(rp.in_channels, rp.out_channels,
                                    kernel_size=rp.kernel_size, stride=rp.stride,
                                    padding=rp.padding, dilation=rp.dilation,
                                    groups=rp.groups, bias=True)
                    tmp.weight.data.copy_(weight); tmp.bias.data.copy_(bias)
                else:
                    continue
                tmp = tmp.to(weight.device)
                saved[name] = rp
                self._replace_module(name, tmp)

            DG = tp.DependencyGraph().build_dependency(
                self.model, example_inputs, verbose=False)

            # After swap, look up plain modules by dotted name to map back to topology names.
            def _get_by_dotted(model, dotted_name):
                cur = model
                for p in dotted_name.split('.'):
                    cur = getattr(cur, p)
                return cur

            plain_by_name = {n: _get_by_dotted(self.model, n) for n in saved.keys()}
            name_by_module = {plain_by_name[n]: n for n in saved.keys()}

            return self._build_topology_from_dg(DG, name_by_module, saved, p=p)
        finally:
            # Swap reparam'd modules back regardless of outcome.
            for name, original_rp in saved.items():
                self._replace_module(name, original_rp)

    def _build_topology_from_dg(self, DG, name_by_module, saved, p=2):
        """Walk the DepGraph forward to assemble the (downstream_name, weight) topology.
        Branch weights at residual ADDs come from σ_out^p (mean per branch)."""

        def _walk_downstream(node, visited):
            """Forward BFS from node.outputs; collect reparam'd modules reached."""
            found = []
            for out_node in node.outputs:
                key = id(out_node)
                if key in visited:
                    continue
                visited.add(key)
                mod = out_node.module
                if mod in name_by_module:
                    found.append(name_by_module[mod])
                else:
                    found.extend(_walk_downstream(out_node, visited))
            return found

        # Look up plain (post-swap) module by name to query DepGraph.
        plain_by_name = {n: m for m, n in name_by_module.items()}

        # Forward sweep: src → list of downstream names (unweighted, may repeat)
        raw_topology = OrderedDict()
        for name in self._reparam_modules.keys():
            plain_mod = plain_by_name.get(name)
            node = DG.module2node.get(plain_mod) if plain_mod is not None else None
            if node is None:
                _log_warning(f"build_propagation_topology: '{name}' not in DepGraph; "
                             f"empty downstream list")
                raw_topology[name] = []
            else:
                raw_topology[name] = _walk_downstream(node, visited=set())

        # Reverse: dst_name → list of upstream src_names (the "branches" entering dst)
        upstream_map = {}
        for src, dsts in raw_topology.items():
            for d in dsts:
                upstream_map.setdefault(d, []).append(src)

        # Branch weights via σ_out^p (scalar per branch = mean over channels).
        # p=2 → σ_c²/(σ_a²+σ_b²) variance split; p=1 → σ_c/(σ_a+σ_b) std split.
        weights = {}  # (src, dst) → float
        for dst, branches in upstream_map.items():
            unique_branches = list(dict.fromkeys(branches))  # de-dup, preserve order
            if len(unique_branches) == 1:
                weights[(unique_branches[0], dst)] = 1.0
            else:
                sigma_per_branch = {}
                for b in unique_branches:
                    rp_b = self._reparam_modules[b]
                    sigma_per_branch[b] = float(rp_b.sigma_out_x.detach().mean().item()) ** p
                total = sum(sigma_per_branch.values()) + 1e-8
                for b in unique_branches:
                    weights[(b, dst)] = sigma_per_branch[b] / total

        # Assemble final topology with weights
        topology = OrderedDict()
        for src, dsts in raw_topology.items():
            unique_dsts = list(dict.fromkeys(dsts))  # de-dup per src
            topology[src] = [(d, weights[(src, d)]) for d in unique_dsts]

        return topology

    def propagation_importance(self, I_out=None, *,
                               conv_reduction="frobenius",
                               on_mismatch="warn",
                               topology=None,
                               p=2,
                               relative=True):
        """Per-input-channel global importance via reverse-walk recursion.

        Implements the paper's propagation criterion (Recap §3 — the best one):

            I^l = W̄^l · I^{l+1},   W̄ = (M)^p · D,
              M[i, j] = |σ_i · reduce_kernel(v[j, i, :])|       (in × out)
              D = Diag(1 / Σ_i M[i, j]^p)  clamp ε               (column-normalize per j)
            I^{L+1} = I_out (default uniform 1/out_dim on the last layer's outputs)

        `relative` selects the two PDF derivations of the SAME recursion:
          relative=True  (default) → W̄ = M^p · D  — each column normalized to sum 1
              (mass-preserving / probability redistribution per layer). Output importance
              is split among inputs by relative contribution; magnitude does NOT compound
              through depth.
          relative=False → W̄ = M^p  (the D column-normalizer dropped) — the raw
              normalized-weight product. The non-relative `I` of the PDF; magnitudes
              compound through depth (deep-layer contributions scale by ∏ column masses).
        Both share M, the kernel reduction, the DAG fan-in, and the I_out seed; they
        differ ONLY in the per-layer D.

        The exponent `p` selects the paper's two relative-contribution flavors
        (relative-contribution section): p=2 = variance `σ²w²/Σσ²w²` (DEFAULT,
        VBP-consistent, matches the L2 per-layer score), p=1 = std `σw/Σσw`. Both
        normalize each column to sum 1. (The paper notes `σw/√Σσw` "cannot" be used —
        it does not sum to 1 — so only p∈{1,2} are offered.)

        Walks self._reparam_modules in reverse forward order (sequential nets). Returns
        OrderedDict[name → I^l tensor of length in_features^l], in forward order.

        Args:
            I_out: Output seed, length = out_features^L. None → uniform.
            conv_reduction: kernel collapse to per-(i,j) scalar — "frobenius" (default,
                ‖v[j,i,:,:]‖_2) or "abs_sum".
            on_mismatch: when two consecutive reparam'd layers have out_l ≠ in_{l+1}
                (residual / branched boundary): "warn" (default) → log + per-layer
                fallback ‖σ·v‖ for that layer; "raise" → RuntimeError; "skip" → silent
                fallback.
            p: relative-contribution exponent (2 variance / 1 std). For residual nets
                build the topology with the SAME p (build_propagation_topology(p=...)).

        Notes:
            - topology=None → sequential walk (M2 behavior). For residual / branched
              nets, pass `topology=self.build_propagation_topology(example_inputs, p=p)`
              — M3 DepGraph-based DAG traversal with σ_out^p branch weighting.
            - σ stop-grad (buffer; .detach() for safety).
            - M is taken absolute so column sums are non-negative (the paper takes
              |w_ij| in §1; for p=2 the sign is irrelevant anyway).
        """
        if not self._active or not self._reparam_modules:
            return OrderedDict()
        if conv_reduction not in ("frobenius", "abs_sum"):
            raise ValueError(f"conv_reduction must be 'frobenius'/'abs_sum', got {conv_reduction!r}")
        if on_mismatch not in ("warn", "raise", "skip"):
            raise ValueError(f"on_mismatch must be 'warn'/'raise'/'skip', got {on_mismatch!r}")
        if p not in (1, 2):
            raise ValueError(f"p must be 1 (std) or 2 (variance), got {p!r}")

        eps = 1e-8

        def _in_dim(rp):
            return rp.in_features if hasattr(rp, "in_features") else rp.in_channels

        def _out_dim(rp):
            return rp.out_features if hasattr(rp, "out_features") else rp.out_channels

        def _layer_M(rp):
            """Build M[in, out] = |σ_i · reduce(v[j, i, *])| for one layer."""
            w = _contribution_weight(rp).detach()  # [out, in] or [out, in, kH, kW]
            if w.dim() == 4:
                if conv_reduction == "frobenius":
                    w_red = w.flatten(2).norm(p=2, dim=2)        # [out, in]
                else:  # abs_sum
                    w_red = w.abs().flatten(2).sum(dim=2)        # [out, in]
            else:
                w_red = w                                          # [out, in]
            return w_red.t().abs()                                 # [in, out]

        def _post_act_std(rp):
            """FALLBACK transfer (terminal / channel-mismatch only): the layer's own
            sigma_out_x. NOTE this is the PRE-activation std (= σ_pre = the D denominator);
            the real PDF transfer Σ^{l+1} is the POST-activation std, taken from the immediate
            on-path consumer's sigma_x in the caller. sigma_out_x is used ONLY when no aligned
            downstream exists. In a fully normalized net every σ→1 so the transfer →1 and
            non-relative collapses to relative (PDF: variance prop gives the same criterion
            for p=2)."""
            od = _out_dim(rp)
            s = getattr(rp, "sigma_out_x", None)
            if s is not None and s.numel() == od:
                return s.detach()
            dev = next(rp.parameters()).device
            return torch.ones(od, device=dev, dtype=next(rp.parameters()).dtype)

        layers = list(self._reparam_modules.items())  # forward order
        Ms = {name: _layer_M(rp) for name, rp in layers}

        # ---- DAG walk path (M3): topology provided ----
        if topology is not None:
            return self._propagate_dag(layers, Ms, topology, I_out, eps, p, relative,
                                       post_act_std=_post_act_std)

        # ---- Sequential path (M2): no topology ----
        # Seed I_out at last layer's output
        last_out_dim = _out_dim(layers[-1][1])
        last_M = Ms[layers[-1][0]]
        last_device = last_M.device
        last_dtype = last_M.dtype
        if I_out is None:
            I_next = torch.full((last_out_dim,), 1.0 / last_out_dim,
                                device=last_device, dtype=last_dtype)
        else:
            I_next = torch.as_tensor(I_out, device=last_device, dtype=last_dtype).flatten()
            if I_next.numel() != last_out_dim:
                raise ValueError(f"I_out has {I_next.numel()} entries, "
                                 f"expected {last_out_dim} (last layer's out_dim)")

        # Per-layer inter-layer transfer Σ^{l+1} = POST-activation std = the NEXT layer's
        # input std sigma_x (a^l feeds straight into it). NOT the layer's own sigma_out_x
        # (= PRE-activation = σ_pre = the D denominator → would cancel D → raw M^p). Fallback
        # to post_act_std at the terminal / channel-mismatch boundary.
        sigma_post = {}
        for _i, (name, rp) in enumerate(layers):
            od = _out_dim(rp)
            nxt_sx = getattr(layers[_i + 1][1], "sigma_x", None) if _i + 1 < len(layers) else None
            sigma_post[name] = (nxt_sx.detach() if (nxt_sx is not None and nxt_sx.numel() == od)
                                else _post_act_std(rp))

        results = []  # reverse-collected, flipped at end
        for idx in range(len(layers) - 1, -1, -1):
            name, rp = layers[idx]
            M = Ms[name]
            out_l = _out_dim(rp)

            if I_next.numel() != out_l:
                msg = (f"propagation chain mismatch at '{name}': out_dim={out_l} but "
                       f"I^{{l+1}} has {I_next.numel()} entries — likely residual / "
                       f"non-sequential boundary. Pass topology= for M3 DAG walk.")
                if on_mismatch == "raise":
                    raise RuntimeError(msg)
                if on_mismatch == "warn":
                    _log_warning(msg)
                # Fallback: per-layer ‖σ·v‖ on input channels (the M1 / L2 score —
                # variance-consistent, independent of p).
                I_l = M.norm(p=2, dim=1)  # [in]
            else:
                # SCORE RECURSION I^l = W̄^l·I^{l+1} (normalized_nets_pruning.pdf steps 7-8).
                # BOTH forms use the column-stochastic W̄ = M^p·D, D=1/col-sum = 1/σ^l_j^p
                # (cancels the pre-activation output variance). They differ ONLY by the
                # inter-layer transfer Σ^{l+1} = post-act output std^p: relative DROPS it
                # (within-layer/local — PDF p3); non-relative KEEPS it (D^lΣ^{l+1}=f_j the
                # measured activation gain → cross-layer). Raw M^p (no D) was a BUG: it
                # compounds the σ^l_j it should divide out (∏ col-sum ≪1 → vanish).
                Mp = M.pow(p)
                col_sums = Mp.sum(dim=0).clamp(min=eps)        # = σ^l_j^p (pre-act output)
                Wbar = Mp / col_sums[None, :]                  # · D: column-stochastic
                if not relative:
                    I_next = sigma_post[name].pow(p) * I_next  # · Σ^{l+1}: post-act transfer
                I_l = Wbar @ I_next                            # propagate one layer back → [in]

            results.append((name, I_l))
            I_next = I_l  # chains to layer l-1 (whose out_dim should equal in_l = I_next.numel())

        return OrderedDict(reversed(results))

    def _propagate_dag(self, layers, Ms, topology, I_out, eps, p=2, relative=True,
                       post_act_std=None):
        """DAG reverse-walk using a downstream+weight topology (M3).

        For each layer L (visited in reverse forward order — assumes named_modules order
        is a topological sort of the forward DAG, which holds for ResNet/VGG/MobileNet
        and other standard architectures), gather:
            I_next = Σ_{(d, w) in topology[L]} w · I[d]
        with I[d] already computed. Terminal layers (empty downstream list) seed from I_out.
        Then I[L] = W̄^L · I_next. `post_act_std(rp, consumer)` supplies the non-relative
        inter-layer transfer Σ^{l+1} (= consumer input std).
        """
        def _out_dim(rp):
            return rp.out_features if hasattr(rp, "out_features") else rp.out_channels

        # Identify terminal layers (no downstream reparam'd consumers)
        terminals = [name for name, dsts in topology.items() if not dsts]
        if not terminals:
            raise RuntimeError("propagation topology has no terminal layer "
                               "(every layer has downstreams) — cyclic or malformed graph")
        if I_out is not None and len(terminals) > 1:
            raise ValueError(f"I_out supplied as a single tensor, but the topology has "
                             f"{len(terminals)} terminal layers: {terminals}. Pass I_out "
                             f"as dict[name → tensor] when there are multiple terminals.")

        # Seed map: terminal_name → I_out vector for that terminal
        rp_by_name = dict(layers)
        device, dtype = next(iter(Ms.values())).device, next(iter(Ms.values())).dtype

        def _seed_for(name):
            rp = rp_by_name[name]
            out_dim = _out_dim(rp)
            if I_out is None:
                return torch.full((out_dim,), 1.0 / out_dim, device=device, dtype=dtype)
            if isinstance(I_out, dict):
                if name not in I_out:
                    raise ValueError(f"I_out dict missing entry for terminal '{name}'")
                t = torch.as_tensor(I_out[name], device=device, dtype=dtype).flatten()
            else:
                t = torch.as_tensor(I_out, device=device, dtype=dtype).flatten()
            if t.numel() != out_dim:
                raise ValueError(f"I_out[{name!r}] has {t.numel()} entries, expected {out_dim}")
            return t

        seeds = {name: _seed_for(name) for name in terminals}

        I_by_name = {}  # name → I^l (per-input-channel)
        # Reverse-topological order from the DAG itself: a layer is ready only once ALL its
        # downstream consumers are computed (I^l depends on I_by_name[d] for every d in dsts).
        # named_modules order is NOT a valid topo sort for branched nets (e.g. ConvNeXt lists
        # all downsample_layers before all stages, but downsample_layers.3 runs AFTER stages.2)
        # — so derive the order explicitly instead of assuming reversed(layers) works.
        rp_by_name = dict(layers)
        done, order, remaining = set(), [], [name for name, _ in layers]
        while remaining:
            ready = [n for n in remaining
                     if all(d in done for d, _ in topology.get(n, []))]
            if not ready:
                raise RuntimeError("propagation topology is cyclic or references unknown "
                                   f"layers; stuck on {remaining}")
            order.extend(ready); done.update(ready)
            remaining = [n for n in remaining if n not in done]

        # Non-relative inter-layer transfer Σ^{l+1} = POST-activation output std (PDF steps
        # 3-4: f = σ_post/σ_pre, the activation std-gain). This is the IMMEDIATE on-path
        # consumer's INPUT std sigma_x (a^l feeds straight into it), NOT the layer's own
        # sigma_out_x — which is the PRE-activation (= σ_pre = the colsum/D denominator), so
        # using it CANCELS D and collapses non-relative to the raw M^p product (the ~1e9
        # compound). To dodge off-path residual readers, pick the consumer EARLIEST in
        # forward order (the main path); fall back to post_act_std only when no channel-
        # aligned downstream exists (terminal / reshape boundary).
        sigma_post = {}
        if not relative and post_act_std is not None:
            fwd_idx = {nm: i for i, (nm, _) in enumerate(layers)}
            sigma_x_of = {nm: getattr(rp, "sigma_x", None) for nm, rp in layers}
            for name, rp in layers:
                od = _out_dim(rp)
                cands = [(fwd_idx[d], d) for d, _w in topology.get(name, [])
                         if d in fwd_idx and fwd_idx[d] > fwd_idx[name]
                         and sigma_x_of[d] is not None and sigma_x_of[d].numel() == od]
                sigma_post[name] = (sigma_x_of[min(cands)[1]].detach() if cands
                                    else post_act_std(rp))
        for name in order:
            rp = rp_by_name[name]
            M = Ms[name]
            out_l = _out_dim(rp)
            dsts = topology.get(name, [])

            if not dsts:
                I_next = seeds[name]
            else:
                I_next = torch.zeros(out_l, device=device, dtype=dtype)
                for d, w in dsts:
                    if d not in I_by_name:
                        raise RuntimeError(f"propagation order violation: '{name}' "
                                           f"downstream '{d}' not yet computed (DAG topology "
                                           f"may not match named_modules order)")
                    I_d = I_by_name[d]
                    if I_d.numel() != out_l:
                        raise RuntimeError(f"propagation shape mismatch at '{name}' → '{d}': "
                                           f"out_dim({name})={out_l} but I[{d!r}] has "
                                           f"{I_d.numel()} entries. Topology says these "
                                           f"layers chain but their channel counts disagree "
                                           f"— check for asymmetric fan-out / channel reshape "
                                           f"between them.")
                    I_next = I_next + float(w) * I_d

            # SCORE RECURSION I^l = W̄^l·I^{l+1} (normalized_nets_pruning.pdf steps 7-8). BOTH
            # forms use column-stochastic W̄ = M^p·D, D=1/col-sum=1/σ^l_j^p (divides out the
            # pre-act output variance). They differ ONLY by the inter-layer transfer Σ^{l+1} =
            # post-act output std^p: relative DROPS it (within-layer/local); non-relative KEEPS
            # it (D^lΣ^{l+1}=f_j the measured activation gain → cross-layer comparison, PDF p3).
            # This is distinct from the residual-JOIN weights `w` above (σ^p/Σσ^p fan-in split):
            # `w` mixes branches into I_next; Σ^{l+1} is THIS layer's per-channel output gain.
            Mp = M.pow(p)
            col_sums = Mp.sum(dim=0).clamp(min=eps)     # = σ^l_j^p (pre-act output)
            Wbar = Mp / col_sums[None, :]               # · D: column-stochastic
            if not relative:
                I_next = sigma_post[name].pow(p) * I_next   # · Σ^{l+1}: post-act transfer
            I_by_name[name] = Wbar @ I_next

        # Return in forward order
        return OrderedDict((name, I_by_name[name]) for name, _ in layers)

    def channel_stats(self):
        """Per-layer ‖σ_x·v‖ statistics along self.norm_dim, plus σ distribution."""
        stats = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            v = reparam.v.detach()
            sigma = reparam.sigma_x.detach()
            if v.dim() == 4:
                w_eff = v * sigma[None, :, None, None]
            else:
                w_eff = v * sigma[None, :]
            col_norms = _channel_group_norm(w_eff, self.norm_dim)
            m = reparam.m.detach()
            stats[name] = {
                'v_col_norm_mean': col_norms.mean().item(),
                'v_col_norm_std': col_norms.std().item(),
                'v_col_norm_min': col_norms.min().item(),
                'v_col_norm_max': col_norms.max().item(),
                'frac_below_0.01': (col_norms < 0.01).float().mean().item(),
                'frac_below_0.1': (col_norms < 0.1).float().mean().item(),
                'm_mean': m.mean().item(),
                'm_std': m.std().item(),
                'sigma_mean': sigma.mean().item(),
                'sigma_min': sigma.min().item(),
                'sigma_max': sigma.max().item(),
            }
        return stats


# The propagation criterion (build_propagation_topology / propagation_importance and
# helpers) is variant-agnostic: it reads only _contribution_weight + sigma_out_x, both
# present on the mean AND bn modules, and the topology swap recognizes either variant by
# attribute. It lives on MeanResidualManager for historical reasons; promote it onto the
# base class so NormalizedResidualManager (bn / canonical) inherits it too (Fix 2).
for _pm in ("build_propagation_topology", "_build_topology_from_dg",
            "propagation_importance", "_propagate_dag"):
    setattr(BaseReparamManager, _pm, getattr(MeanResidualManager, _pm))
del _pm


# =====================================================================
# NormalizedResidualManager (VNR)
# =====================================================================

class NormalizedResidualManager(BaseReparamManager):
    """BN-based reparameterization: inserts BN(affine=False) before the target layer.

    The trainable weight v_tilde = W·σ_cal operates on BN-normalized input.
    ‖v_tilde[:,k]‖ = σ_k·‖W[:,k]‖ measures variance contribution directly.
    BN auto-updates running stats during training — no frozen σ, no refresh needed.
    Compensation blocked: inflating σ upstream increases ‖Ṽ‖ → more penalty.
    Optional entropy_loss encourages uniform Ṽ column norms → balanced pruning.

    Usage:
        mgr = NormalizedResidualManager(model, target_names, device,
                                         lambda_reg=0.01, entropy_lambda=0.01)
        mgr.reparameterize(train_loader)
        ...  # train with mgr.regularization_loss() + mgr.entropy_loss()
        mgr.merge_back()
    """

    def __init__(self, model, target_names, device, lambda_reg=0.01, max_batches=200,
                 scale_invariant=False, entropy_lambda=0.0, reparam_target="fc2",
                 bn_momentum=0.1):
        super().__init__(model, target_names, device, lambda_reg=lambda_reg,
                         max_batches=max_batches, scale_invariant=scale_invariant,
                         reparam_target=reparam_target)
        self._entropy_lambda = entropy_lambda
        # σ EMA momentum for the input-normalizing BN (boss: "keep σ updated with EMA").
        # 0.1 = torch BN default; slower (e.g. 0.01) for long from-scratch training.
        if not 0.0 <= float(bn_momentum) <= 1.0:
            raise ValueError(f"bn_momentum must be in [0, 1], got {bn_momentum}")
        self.bn_momentum = float(bn_momentum)

    def _calibrate(self, targets, loader):
        """Calibrate (μ_x, σ_x, σ_out_x) via the shared per-layer stats hook (same routine
        the mean variant uses). The 3rd element — per-output-channel std — feeds propagation
        branch weighting (M3); v_tilde still uses σ_x (input std) for the BN-form weight."""
        return self._calibrate_stats(targets, loader)

    def _make_reparam(self, module, calibration_data):
        """Create BNResidual* module from standard nn.Linear/Conv2d."""
        mu_x, sigma_x, sigma_out_x = calibration_data
        BN_EPS = 1e-5  # default nn.BatchNorm eps

        if isinstance(module, nn.Linear):
            W = module.weight.data.clone()  # [out, in]
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_features, device=self.device)
            # _calibrate_stats returns σ_x = sqrt(var + eps); recover the raw variance so
            # bn_running_var + sigma_eff exactly match the pre-Fix-2 vnr init (no eps
            # double-count) → tp_variance / vnr stays numerically backward-compatible.
            bn_running_var = (sigma_x ** 2 - BN_EPS).clamp(min=0.0)
            # BN eval: x_bn = (x - μ) / sqrt(var + eps)
            # For v_tilde / sqrt(var + eps) = W: v_tilde = W * sqrt(var + eps)
            sigma_eff = torch.sqrt(bn_running_var + BN_EPS)
            v_tilde = W * sigma_eff[None, :]   # FOLD σ INTO WEIGHT: ṽ = σ·W (now acts on unit-var input)
            m = b + W @ mu_x                   # FOLD μ INTO BIAS:   m = b + W·μ
            reparam = BNResidualLinear(
                module.in_features, module.out_features,
                v_tilde=v_tilde, m=m,
                bn_running_mean=mu_x, bn_running_var=bn_running_var,
                bn_momentum=self.bn_momentum, sigma_out_x=sigma_out_x)
            return reparam.to(self.device)

        elif isinstance(module, nn.Conv2d):
            W = module.weight.data.clone()  # [C_out, C_in, kH, kW]
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_channels, device=self.device)
            # _calibrate_stats returns σ_x = sqrt(var + eps); recover the raw variance so
            # bn_running_var + sigma_eff exactly match the pre-Fix-2 vnr init (no eps
            # double-count) → tp_variance / vnr stays numerically backward-compatible.
            bn_running_var = (sigma_x ** 2 - BN_EPS).clamp(min=0.0)
            sigma_eff = torch.sqrt(bn_running_var + BN_EPS)
            v_tilde = W * sigma_eff[None, :, None, None]
            m = b + W.sum(dim=(2, 3)) @ mu_x
            reparam = BNResidualConv2d(
                module.in_channels, module.out_channels,
                kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation,
                groups=module.groups, v_tilde=v_tilde, m=m,
                bn_running_mean=mu_x, bn_running_var=bn_running_var,
                bn_momentum=self.bn_momentum, sigma_out_x=sigma_out_x)
            return reparam.to(self.device)

        raise TypeError(f"Unsupported module type: {type(module)}")

    def refresh_stats(self, loader):
        """No-op: BN(affine=False) auto-updates running stats during training."""
        if not self._active:
            return
        _log_info("NormalizedResidualManager: BN running stats auto-updated (no manual refresh needed)")

    def regularization_loss(self):
        """L_{2,1} on Ṽ along self.norm_dim. Ṽ=σW operates on unit-variance normalized
        input, so ‖Ṽ‖ IS the contribution score — this is plain magnitude pruning on the
        normalized net (boss recap #6), no separate σ factor needed."""
        loss = torch.tensor(0.0, device=self.device)
        for reparam in self._reparam_modules.values():
            norms = _channel_group_norm(reparam.v_tilde, self.norm_dim)
            if self.scale_invariant and hasattr(reparam, 'v_init_norms'):
                loss = loss + (norms / reparam.v_init_norms).sum()
            else:
                loss = loss + norms.sum()
        return self.lambda_reg * loss

    def entropy_loss(self):
        """Weight-space entropy on Ṽ column norms.

        H = -Σ_k p_k log(p_k), where p_k = ||Ṽ[:,k]||² / Σ_j ||Ṽ[:,j]||².
        Maximizing H encourages uniform column norms → balanced pruning decisions.
        Returns entropy_lambda * (-H) (minimize to maximize entropy).
        """
        if self._entropy_lambda == 0.0:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for reparam in self._reparam_modules.values():
            norms_sq = _channel_group_norm(reparam.v_tilde, self.norm_dim) ** 2
            p = norms_sq / (norms_sq.sum() + EPSILON)
            loss = loss - (p * (p + EPSILON).log()).sum()
        return self._entropy_lambda * loss

    def channel_stats(self):
        """Per-layer dual reporting: Ṽ norms AND w_eff norms along self.norm_dim."""
        stats = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            vt = reparam.v_tilde.detach()
            vt_col_norms = _channel_group_norm(vt, self.norm_dim)

            # w_eff from BN running stats
            sigma = torch.sqrt(reparam.bn.running_var + reparam.bn.eps)
            if hasattr(reparam, 'in_features'):  # Linear
                w_eff = vt / sigma[None, :]
            else:  # Conv2d
                w_eff = vt / sigma[None, :, None, None]
            weff_col_norms = _channel_group_norm(w_eff, self.norm_dim)

            m = reparam.m.detach()
            stats[name] = {
                'v_col_norm_mean': vt_col_norms.mean().item(),
                'v_col_norm_std': vt_col_norms.std().item(),
                'v_col_norm_min': vt_col_norms.min().item(),
                'v_col_norm_max': vt_col_norms.max().item(),
                'frac_below_0.01': (vt_col_norms < 0.01).float().mean().item(),
                'frac_below_0.1': (vt_col_norms < 0.1).float().mean().item(),
                'weff_col_norm_mean': weff_col_norms.mean().item(),
                'weff_col_norm_std': weff_col_norms.std().item(),
                'm_mean': m.mean().item(),
                'm_std': m.std().item(),
            }
        return stats
