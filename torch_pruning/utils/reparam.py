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
    ‖σ·v‖ and the optional λ‖σ·v‖ regularizer. Folding σ directly into a W-trainable
    (σ-divide) parametrization produces a 1/σ² optimizer-geometry overshoot, which this
    mean variant avoids by keeping σ out of the forward.
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

    Normalize input by the live EMA σ (running stat, not batch). Trainable
    v_tilde = σ_cal·W (σ-scaling applied once at init, then frozen into v_tilde). Forward
    uses the effective-bias trick z = linear(x, v_tilde/σ_run, m − (v_tilde/σ_run)@μ_run)
    = v_tilde·(x−μ)/σ — equals BN-normalize-then-linear but with running (EMA) stats so it
    matches the Conv variant. ‖v_tilde[:,k]‖ = contribution.
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
        # Update EMA running stats in train mode (no grad), then normalize by running
        # stats via the effective-bias trick — not batch stats.
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
        natural-sparsity fractions frac_below_{0.01,0.1,1.0} — the primary signal
        for the λ regularization sweep: a growing left tail = induced channel
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

    @torch.no_grad()
    def collect_input_covariance(self, loader, max_batches=50):
        """Per-layer input-channel CORRELATION matrix Σ̂ for the propagation cov numerator.

        Hooks every ACTIVE reparam module, accumulates raw input moments over `loader`
        (conv: channels-first NCHW, pooled over batch+space; linear: channel = LAST dim,
        which also handles convnext's channels-last 4D pwconv input), then
        Σ̂ = cov(x) / (σσᵀ) — the covariance of the NORMALIZED input x̂ that the
        contribution weight w̃ = σ·v acts on, so w̃ Σ̂ w̃ᵀ = true Var(Z) including the
        off-diagonal covariance the independence colsum drops (diag(Σ̂)=1 recovers it).
        σ here is the cov-pass std itself (NOT the stored calibration sigma_x) so the
        diagonal is exactly 1 and the matrix is variant-agnostic (mean AND bn).
        DDP: raw moments are all-reduced → identical Σ̂ on every rank.

        Call while ACTIVE. Returns OrderedDict[name → Σ̂ [in, in]] for
        propagation_importance(input_cov=...). O(in²) memory per layer (convnext
        worst case 3072² fp32 ≈ 38 MB).
        """
        if not self._active or not self._reparam_modules:
            return OrderedDict()
        acc = {name: None for name in self._reparam_modules}   # [AtA, sumA, count]
        name_of = {m: n for n, m in self._reparam_modules.items()}

        def hook(mod, inp):
            x = inp[0].detach()
            if hasattr(mod, "in_channels"):     # conv: channels-first (N,C,H,W) → (N·H·W, C)
                a = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            else:                               # linear: channel = last dim ((N,C) or (N,H,W,C))
                a = x.reshape(-1, x.shape[-1])
            a = a.float()
            n = name_of[mod]
            AtA, As, cnt = a.t() @ a, a.sum(0), a.shape[0]
            if acc[n] is None:
                acc[n] = [AtA, As, cnt]
            else:
                acc[n][0] += AtA; acc[n][1] += As; acc[n][2] += cnt

        handles = [m.register_forward_pre_hook(hook) for m in self._reparam_modules.values()]
        was_training = self.model.training
        self.model.eval()
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            self.model(x.to(self.device))
        for h in handles:
            h.remove()
        if was_training:
            self.model.train()

        ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        out = OrderedDict()
        for name in self._reparam_modules:
            if acc[name] is None:
                continue
            AtA, As, cnt = acc[name]
            if ddp:                             # sync raw moments (CUDA tensors for NCCL)
                cnt_t = torch.tensor([float(cnt)], device=AtA.device)
                torch.distributed.all_reduce(AtA, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(As, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)
                cnt = float(cnt_t.item())
            mean = As / cnt
            cov = AtA / cnt - torch.outer(mean, mean)
            # Variance is non-negative, but a near-constant channel (dead/saturated activation)
            # lands cov.diag ≈ 0 or slightly NEGATIVE via float cancellation (AtA/cnt ≈ mean²).
            # A naive correlation then divides by a hard-clamped sqrt and EXPLODES (diag → -1e4,
            # off-diag → 1e6), and WHICH channel crosses zero is float-order/platform dependent
            # → non-reproducible blow-up that measured_var amplifies through the propagation chain.
            # Clamp variance ≥ 0 with a RELATIVE floor, bound the correlation to its valid [-1,1]
            # range, and force an exact unit diagonal so diag(Σ̂)=1 (independence recovery) holds.
            var = cov.diag().clamp_min(0.0)
            floor = 1e-8 * var.max().clamp_min(1e-12)
            sig = var.clamp_min(floor).sqrt()
            corr = (cov / torch.outer(sig, sig)).clamp_(-1.0, 1.0)
            corr.diagonal().copy_(torch.ones_like(sig))
            out[name] = corr
        return out

    def collect_join_covariance(self, loader, residual_blocks, max_batches=50):
        """EXACT, mass-conserving residual-join branch shares from measured covariance.

        At a residual add c = a + b (skip a + branch b), importance flowing back to c must
        split by each branch's covariance with the sum, NOT by its own variance:

            weight_b = Cov(b, c) / Var(c)   (Σ_branches Cov(·,c)/Var(c) = Cov(c,c)/Var(c) = 1)

        which conserves mass and attributes the SHARED Cov(a,b) to the right path — unlike the
        independence share σ_b²/Σσ² (drops 2·Cov(a,b)) and unlike the use_measured_sigma_c
        rescale (σ_c²/Σσ² boost, keeps shares ∝ own variance, sums to σ_c²/Σσ² ≠ 1).

        Measured at the ADD itself: each block module is hooked, a = block input, c = block
        output, b = c − a. Per output-channel: Var(c) and Cov(b,c) = Var(c) − Cov(a,c).

        Args:
            residual_blocks: dict {block_nn_module → residual_terminal_name}. The block's
                forward must be c = a + (branch), with a = its first input and c = its output
                (e.g. ConvNeXt Block: x = input + drop_path(γ·pwconv2(...))). The terminal name
                is the reparam'd layer ending the residual branch (its out-channels = c's), used
                to key the returned weight onto the topology edge.
        Returns OrderedDict[residual_terminal_name → per-channel weight_b = Cov(b,c)/Var(c)].
        For _build_topology_from_dg(join_cov=...); the skip branch(es) take the remainder
        (1 − weight_b), so the join stays exactly mass-conserving.
        """
        if not self._active or not residual_blocks:
            return OrderedDict()
        acc = {name: None for name in residual_blocks.values()}   # name → [sa, sc, sac, scc, cnt]
        name_of = {mod: name for mod, name in residual_blocks.items()}

        def _chan_last(t):  # → [tokens, C]; NCHW conv vs channel-last/linear
            if t.dim() == 4:
                return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]).float()
            return t.reshape(-1, t.shape[-1]).float()

        def hook(mod, inp, out):
            a = _chan_last(inp[0].detach())
            c = _chan_last(out.detach())
            name = name_of[mod]
            sa, sc, sac, scc, cnt = a.sum(0), c.sum(0), (a * c).sum(0), (c * c).sum(0), a.shape[0]
            if acc[name] is None:
                acc[name] = [sa, sc, sac, scc, cnt]
            else:
                acc[name][0] += sa; acc[name][1] += sc
                acc[name][2] += sac; acc[name][3] += scc; acc[name][4] += cnt

        handles = [m.register_forward_hook(hook) for m in residual_blocks]
        was_training = self.model.training
        self.model.eval()
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            self.model(x.to(self.device))
        for h in handles:
            h.remove()
        if was_training:
            self.model.train()

        ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        out = OrderedDict()
        for name in residual_blocks.values():
            if acc[name] is None:
                continue
            sa, sc, sac, scc, cnt = acc[name]
            if ddp:
                cnt_t = torch.tensor([float(cnt)], device=sa.device)
                for t in (sa, sc, sac, scc):
                    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)
                cnt = float(cnt_t.item())
            ma, mc = sa / cnt, sc / cnt
            var_c = (scc / cnt - mc * mc).clamp(min=1e-12)        # Var(c)  per channel
            cov_ac = sac / cnt - ma * mc                          # Cov(a,c)
            cov_bc = var_c - cov_ac                               # Cov(b,c) = Var(c) − Cov(a,c)
            out[name] = cov_bc / var_c                            # weight_b (may be <0 or >1; skip takes 1−)
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

    Pruning on the mean path is supported: each MeanResidual* module stores a frozen
    per-input-channel sigma_x buffer (set at reparameterize()), input_channel_scores()
    ranks by ‖σ·v‖ (= √NCI), and regularization_loss() penalizes ‖σ·v‖ with σ detached.
    σ is NEVER folded into the trainable v — that would reintroduce the 1/σ²
    optimizer-geometry overshoot; the penalty stays decoupled from the optimizer step.
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

    def build_propagation_topology(self, example_inputs, p=2, use_measured_sigma_c=False,
                                   branch_out_scale=None, join_cov=None):
        """Build the per-layer downstream + branch-weight topology via tp.DependencyGraph.

        Returns OrderedDict[name → list of (downstream_name, weight)]:
          - sequential: single downstream, weight 1.0;
          - residual join: each branch's single downstream is the merged consumer,
            with weight σ_branch^p / Σ σ_branches^p (paper's σ_c^p/(σ_a^p+σ_b^p)),
            computed PER-CHANNEL (a tensor over the merged channel axis);
          - fan-out (no add, layer feeds multiple consumers): each downstream weight = 1.0
            (importance accumulates additively across consumers);
          - terminal (no reparam'd consumer downstream): empty list — uses I_out seed.

        Args:
            p: skip-connection exponent — MUST match the `p` passed to
                propagation_importance (default 2 = variance / VBP-consistent; 1 = std).
            example_inputs: torch.Tensor (or list/dict) suitable for one forward pass —
                used by DepGraph to trace the computational graph on `self.model`.
            branch_out_scale: optional {layer_name → per-channel scale tensor}. The PDF
                join share needs the branch std AT THE ADD, but sigma_out_x is measured at
                the reparam'd layer's RAW output — any per-channel scaling between them
                (ConvNeXt layer-scale gamma, an unfolded BN) is missed. Pass |scale| here
                to correct: branch σ = sigma_out_x · scale before the σ^p share.
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

            return self._build_topology_from_dg(DG, name_by_module, saved, p=p,
                                                use_measured_sigma_c=use_measured_sigma_c,
                                                branch_out_scale=branch_out_scale,
                                                join_cov=join_cov)
        finally:
            # Swap reparam'd modules back regardless of outcome.
            for name, original_rp in saved.items():
                self._replace_module(name, original_rp)

    def _build_topology_from_dg(self, DG, name_by_module, saved, p=2,
                                use_measured_sigma_c=False, branch_out_scale=None,
                                join_cov=None):
        """Walk the DepGraph forward to assemble the (downstream_name, weight) topology.
        Branch weights at residual ADDs come from σ_out^p (mean per branch). When
        use_measured_sigma_c, the join denominator is the measured post-add σ_c (PDF skip
        factor) instead of the independence sum Σσ_branch^p. branch_out_scale corrects
        each branch's σ for per-channel scaling between the layer output and the add
        (see build_propagation_topology)."""

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

        # Branch weights via σ_out^p, PER-CHANNEL (PDF σ_c^p/(σ_a^p+σ_b^p), the variance share
        # of each branch at the add — p=2 variance, p=1 std). Each branch's σ_out_x is per
        # output-channel and the branches share the merged channel axis, so the split is a
        # per-channel vector (NOT a scalar channel-mean — that was an approximation).
        weights = {}  # (src, dst) → float 1.0 (sequential/fan-out) or per-channel tensor (join)
        for dst, branches in upstream_map.items():
            unique_branches = list(dict.fromkeys(branches))  # de-dup, preserve order
            if len(unique_branches) == 1:
                weights[(unique_branches[0], dst)] = 1.0
            elif join_cov and any(b in join_cov for b in unique_branches):
                # EXACT covariance share (mass-conserving): residual branch b gets the measured
                # Cov(b,c)/Var(c); the skip branch(es) — not in join_cov — take the remainder
                # (1 − Σ weight_b), split by σ^p if more than one. Σ over branches = 1 exactly,
                # so no spurious mass at the join (unlike use_measured_sigma_c). See
                # collect_join_covariance.
                resid = [b for b in unique_branches if b in join_cov]
                skips = [b for b in unique_branches if b not in join_cov]
                wsum = None
                for b in resid:
                    wb = join_cov[b].detach()
                    weights[(b, dst)] = wb
                    wsum = wb if wsum is None else wsum + wb
                remainder = (1.0 - wsum) if wsum is not None else 1.0
                if skips:
                    ssig = {b: self._reparam_modules[b].sigma_out_x.detach().pow(p) for b in skips}
                    sbase = sum(ssig.values()) + 1e-8
                    for b in skips:
                        weights[(b, dst)] = remainder * (ssig[b] / sbase)
                continue
            else:
                # Branch σ at the ADD: sigma_out_x is measured at the layer's raw output;
                # apply any per-channel scale sitting between it and the add (ConvNeXt
                # layer-scale gamma) so the PDF σ_a^p/(σ_a^p+σ_b^p) share uses the std the
                # join actually sees. Missing this skews shares by relative gamma^p.
                def _branch_sigma(b):
                    s = self._reparam_modules[b].sigma_out_x.detach()
                    if branch_out_scale and b in branch_out_scale:
                        s = s * branch_out_scale[b].detach().abs().to(s.device, s.dtype)
                    return s
                sig_p = {b: _branch_sigma(b).pow(p) for b in unique_branches}
                base_sum = sum(sig_p.values()) + 1e-8         # Σσ_branch^p (independence node var)
                # Default branch share = σ_branch^p / Σσ_branch^p (assumes Var(C)=ΣVar(branch)).
                # use_measured_sigma_c: multiply by the PDF skip factor σ_c^p/(σ_a^p+σ_b^p), with
                # σ_c = MEASURED post-add std (= merged consumer's calibrated sigma_x). cov(A,B)>0
                # ⇒ σ_c^p > Σσ_branch^p ⇒ factor>1 ⇒ branch weight BOOSTED (correct direction —
                # brute-force drop-branch confirms; the earlier σ_branch^p/σ_c^p form was inverted).
                if use_measured_sigma_c:
                    factor = self._reparam_modules[dst].sigma_x.detach().pow(p) / base_sum
                    for b in unique_branches:
                        weights[(b, dst)] = (sig_p[b] / base_sum) * factor   # per-channel tensor
                else:
                    for b in unique_branches:
                        weights[(b, dst)] = sig_p[b] / base_sum              # per-channel tensor

        # Assemble final topology with weights
        topology = OrderedDict()
        for src, dsts in raw_topology.items():
            unique_dsts = list(dict.fromkeys(dsts))  # de-dup per src
            topology[src] = [(d, weights[(src, d)]) for d in unique_dsts]

        return topology

    def propagation_diagnostics(self, input_cov=None, conv_reduction="frobenius"):
        """Per-layer W̄ mass diagnostics for the cov/measured debug (read-only, p=2).

        For each reparam layer returns scalars + per-output-channel tensors:
          indep_var    = Σ_i M_ij²              (independence colsum = plain-prop denom)
          recon_var    = Σ_i clamp(N_ij, ≥0)    (cov-reconstructed Var(Z_j) = N.sum)
          measured_var = σ_out_x_j²             (directly measured post-act output var)
          leak         = recon_var / measured_var   (=1 ⇔ recon==measured; Q3 measured≈recon)
          cov_colsum   = (N / N.sum).sum         (=1 ⇔ recon W̄ column-stochastic; Q2 mass)
          has_cov      = whether a cov numerator applied (False ⇒ independence fallback, e.g.
                         depthwise where Σ̂[in,in] ≠ M's in/groups dim → shape-guard skip).
        Mirrors _layer_M / _layer_N exactly so the numbers match the live scorer.
        """
        out = OrderedDict()
        for name, rp in self._reparam_modules.items():
            w = _contribution_weight(rp).detach()
            if w.dim() == 4:
                w_red = (w.flatten(2).norm(p=2, dim=2) if conv_reduction == "frobenius"
                         else w.abs().flatten(2).sum(dim=2))
            else:
                w_red = w
            M = w_red.t().abs()                              # [in, out]
            indep_var = M.pow(2).sum(dim=0)                  # [out]
            rec = {"width_in": int(M.shape[0]), "width_out": int(M.shape[1]),
                   "indep_var": indep_var.detach().cpu(), "has_cov": False}
            S = input_cov.get(name) if input_cov else None
            if S is not None and S.shape[0] == M.shape[0]:
                S = S.to(w.device, w.dtype)
                if w.dim() == 4:
                    V = w.flatten(2)
                    T = torch.einsum('odk,dc->ock', V, S)
                    N = (V * T).sum(-1)
                else:
                    N = w * (w @ S)
                N = N.t().clamp_min(0.0)                     # [in, out]
                recon_var = N.sum(dim=0)                     # [out]
                rec["has_cov"] = True
                rec["recon_var"] = recon_var.detach().cpu()
                rec["cov_colsum"] = (N / recon_var.clamp(min=1e-8)).sum(dim=0).detach().cpu()
            if hasattr(rp, "sigma_out_x"):
                meas = rp.sigma_out_x.detach().pow(2).cpu()
                rec["measured_var"] = meas
                if "recon_var" in rec:
                    rec["leak"] = rec["recon_var"] / meas.clamp(min=1e-8)
            out[name] = rec
        return out

    def propagation_importance(self, I_out=None, *,
                               conv_reduction="frobenius",
                               on_mismatch="warn",
                               topology=None,
                               p=2,
                               relative=True,
                               use_measured_var=False,
                               input_cov=None,
                               keep=None):
        """Per-input-channel global importance via reverse-walk recursion.

        The paper's propagation criterion (normalized_nets_pruning.pdf steps 7-10):

            I^l = W̄^l · I^{l+1},   W̄ = M^p · D,
              M[i, j] = |σ_i · reduce_kernel(v[j, i, :])| = |W'[i,j]|   (in × out)
            I^{L+1} = I_out (default uniform 1/out_dim on the last layer's outputs)

        σ is folded into M ONCE; no separate per-hop σ transfer — the PDF's interleaved
        Σ^{l+1} is already absorbed into M^{l+1} = σ_in^{l+1}·W. `relative` selects D:

          relative=True  (default) → D = 1/Σ_i M[i,j]^p — column-stochastic, bounded,
              but mass-conserving → cross-layer ranking degenerates to 1/width (local
              metric, PDF p.3).
          relative=False → D = 1/(Σ_i M[i,j]^2)^{p/2} — the std/L2 column norm (PDF
              steps 7-8). Cross-layer in intent, but ρ≠1 → scale compounds ρ^depth →
              global sort degenerates to ranking-by-depth.

        For p=2 the two normalizers COINCIDE (PDF: "variance propagation yields the same
        relative importance criterion for p=2"); they differ only at p=1 (std). Neither
        form is a bounded global criterion; that is per-layer ‖σ·v‖ = √NCI
        (mode="per_layer").

        Walks self._reparam_modules in reverse forward order. Returns
        OrderedDict[name → I^l tensor of length in_features^l], in forward order.

        Args:
            I_out: Output seed, length = out_features^L. None → uniform.
            conv_reduction: kernel collapse — "frobenius" (default, ‖v[j,i,:,:]‖_2)
                or "abs_sum".
            on_mismatch: out_l ≠ in_{l+1} boundary (residual/branched): "warn" (default,
                per-layer ‖σ·v‖ fallback + log), "raise", or "skip" (silent fallback).
            p: 2 = variance (default, VBP-consistent) / 1 = std. For residual nets build
                the topology with the SAME p.
            topology: None → sequential walk (M2). Residual/branched nets: pass
                `self.build_propagation_topology(example_inputs, p=p)` (M3 DAG walk
                with σ_out^p branch weighting).
            input_cov: {name → Σ̂ [in,in]} from collect_input_covariance() — the FULL
                covariance fix (p=2 only). Replaces the independence numerator M^2 with
                the signed covariance share N[c,j] = w̃_jc·(w̃Σ̂)_jc (clamped ≥0), whose
                column sums reconstruct the true Var(Z_j). Default denominator = that
                colsum → W̄ exactly column-stochastic → mass conserved per backward
                step WITH correct (covariance-aware) shares. use_measured_var then
                swaps the denom for measured σ_out_x² (recon≈meas, both legal).
                NOTE: numerator and denominator must be fixed TOGETHER —
                use_measured_var alone (denom-only) breaks the colsum (indep/meas ≈
                1/2…1/17, per-output varying) → mass leaks → depth bias returns.

        σ is stop-grad throughout; M is taken absolute (PDF §1 |w_ij|; sign-free for
        p=2 anyway). The cov numerator keeps the SIGN of w̃ (cross terms cancel).
        """
        if not self._active or not self._reparam_modules:
            return OrderedDict()
        if conv_reduction not in ("frobenius", "abs_sum"):
            raise ValueError(f"conv_reduction must be 'frobenius'/'abs_sum', got {conv_reduction!r}")
        if on_mismatch not in ("warn", "raise", "skip"):
            raise ValueError(f"on_mismatch must be 'warn'/'raise'/'skip', got {on_mismatch!r}")
        if p not in (1, 2):
            raise ValueError(f"p must be 1 (std) or 2 (variance), got {p!r}")
        if input_cov and p != 2:
            raise ValueError("input_cov (covariance numerator) is a variance decomposition "
                             "— only valid at p=2. Drop input_cov or use p=2.")

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

        def _layer_N(rp, S, k=None):
            """SIGNED covariance numerator N [in, out]: N[c,j] = w̃_jc · (w̃ Σ̂)_jc.
            Row-sums over c reconstruct the true Var(Z_j) = w̃Σ̂w̃ᵀ (off-diagonals kept;
            the independence M² numerator is the diag(Σ̂)=1 special case). Conv uses the
            channel-stationary kernel-aligned form N[c,j] = Σ_c' ⟨v_jc,·,v_jc',·⟩ Σ̂_cc'
            (same approximation as nci_cov's Gram). Negative entries (removing the
            channel would INCREASE Var) clamp to 0; the column normalizer is taken
            AFTER the clamp so W̄ stays exactly column-stochastic.

            k: optional input-channel keep mask (iterative). The pruned channels' RESULT rows
            are zeroed AFTER the Σ_c' cov sum (post-hoc), keeping each SURVIVING channel's share
            computed against the original (full) couplings. This is deliberate: the keep= path is
            the v2 §"Importance score updating" footnote* "variances forced to 1 — updated variance
            does NOT forward-propagate" simplification, so the recon denom D_j=Σ_c N[c,j] stays the
            stable full Var(Z_j) reference (no per-round variance update). Excluding pruned k from a
            survivor's Σ_c' sum (the 2026-06-23 "cov-leak fix", reverted) imposes a partial variance
            update inconsistent with forced-to-1 and empirically degrades the greedy ranking
            (cov-iter 0.27→0.20 @ 66% MAC, mnv2). One op covers [out,in] and [out,in,kH,kW]."""
            w = _contribution_weight(rp).detach()
            S = S.to(w.device, w.dtype)
            if w.dim() == 4:
                V = w.flatten(2)                              # [out, in, kHkW]
                T = torch.einsum('odk,dc->ock', V, S)         # (w̃ Σ̂) kernel-aligned
                N = (V * T).sum(-1)                           # [out, in]
            else:
                N = w * (w @ S)                               # [out, in]
            Nt = N.t().clamp_min(0.0)                         # [in, out]
            if k is not None:
                Nt = Nt.clone(); Nt[~k.to(Nt.device)] = 0    # forced-to-1: zero pruned rows only
            return Nt

        def _Wbar(M, measured_denom=None, N=None):
            """Column-normalized W̄ = M^p · D (no σ transfer — σ is already in M).
            relative → D = 1/Σ_i M^p (column-stochastic). non-relative → D = 1/σ_pre^p
            = 1/(Σ_i M^2)^{p/2} (std/L2 norm). Identical for p=2.

            N (input_cov): FULL covariance fix — numerator = cov share N, denominator =
            its own colsum (exactly column-stochastic) or measured σ_out_x² when
            use_measured_var (recon ≈ meas). Numerator+denominator move TOGETHER.

            measured_denom WITHOUT N (use_measured_var alone) is the broken middle rung:
            numerator stays independence ΣM², denom becomes true Var → colsums =
            indep/meas ≈ 1/2…1/17 per-output varying → mass leaks unevenly per step →
            depth bias (compounding) returns. Kept for ablation only."""
            if N is not None:
                if measured_denom is not None:
                    denom = measured_denom.to(N.dtype).clamp(min=eps)      # true Var(Z_j)
                else:
                    denom = N.sum(dim=0).clamp(min=eps)                    # recon Var (exact)
                return N / denom[None, :]
            Mp = M.pow(p)
            if measured_denom is not None:
                denom = measured_denom.to(Mp.dtype).clamp(min=eps)        # true Var(Z_j)^{p/2}
            elif relative:
                denom = Mp.sum(dim=0).clamp(min=eps)                       # L1 of M^p
            else:
                denom = M.pow(2).sum(dim=0).clamp(min=eps).pow(p / 2.0)    # σ_pre^p (L2)
            return Mp / denom[None, :]

        layers = list(self._reparam_modules.items())  # forward order
        Ms = {name: _layer_M(rp) for name, rp in layers}
        # use_measured_var: per-layer MEASURED output node variance σ_out_x^p (true Var(Z_j)),
        # replaces the computed independence denominator Σ_i(σ_i W_ij)^p in _Wbar.
        meas_denom = ({name: rp.sigma_out_x.detach().pow(p) for name, rp in layers}
                      if use_measured_var else {})
        # keep: iterative-pruning input-channel mask (v2 "variances forced to 1" update).
        # A pruned hidden channel is an INPUT (row of M [in,out]) of its consumer; zero that
        # row so the column denominator Σ_i M^p re-normalizes over the survivors. Its producer
        # side is handled implicitly — the consumer's I^l[c]=0 flows back as the producer's
        # output seed. Default None ⇒ byte-identical to one-shot.
        def _keep_mask(name):
            if not keep:
                return None
            k = keep.get(name)
            return None if k is None else k.to(Ms[name].device).bool().reshape(-1)
        masks = {name: _keep_mask(name) for name, _ in layers} if keep else {}

        # input_cov: covariance numerators (full cov fix). The keep mask is passed INTO
        # _layer_N so pruned in-channels are zeroed in w̃ BEFORE the Σ_c' cov sum (a survivor
        # stops leaking covariance through already-pruned k — exact iterative update). Layers
        # missing from input_cov (or with a stale shape) fall back to the independence M^p num.
        cov_num = {}
        if input_cov:
            for name, rp in layers:
                S = input_cov.get(name)
                if S is None:
                    continue
                if S.shape[0] != Ms[name].shape[0]:
                    _log_warning(f"input_cov[{name!r}] is {tuple(S.shape)} but layer has "
                                 f"{Ms[name].shape[0]} input channels — skipping cov "
                                 f"numerator for this layer (independence fallback)")
                    continue
                cov_num[name] = _layer_N(rp, S, masks.get(name))

        if keep:
            for name in Ms:
                k = masks.get(name)
                if k is not None:
                    Ms[name] = Ms[name].clone(); Ms[name][~k] = 0

        # ---- DAG walk path (M3): topology provided ----
        if topology is not None:
            return self._propagate_dag(layers, Ms, topology, I_out, eps, p, relative,
                                       wbar_fn=_Wbar, meas_denom=meas_denom,
                                       cov_num=cov_num, keep_mask=_keep_mask)

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
                # SCORE RECURSION I^l = W̄^l·I^{l+1} (normalized_nets_pruning.pdf steps 7-10).
                # σ is already in M (=σ·W); no separate transfer. relative → column-stochastic
                # D=1/Σ_iM^p; non-relative → D=1/σ_pre^p (L2). Identical for p=2.
                I_l = _Wbar(M, meas_denom.get(name), cov_num.get(name)) @ I_next  # one layer back → [in]

            km = _keep_mask(name)
            if km is not None:
                I_l = I_l * km.to(I_l.dtype)        # pruned inputs carry no importance
            results.append((name, I_l))
            I_next = I_l  # chains to layer l-1 (whose out_dim should equal in_l = I_next.numel())

        return OrderedDict(reversed(results))

    def _propagate_dag(self, layers, Ms, topology, I_out, eps, p=2, relative=True,
                       wbar_fn=None, meas_denom=None, cov_num=None, keep_mask=None):
        """DAG reverse-walk using a downstream+weight topology (M3).

        For each layer L (visited in reverse forward order — assumes named_modules order
        is a topological sort of the forward DAG, which holds for ResNet/VGG/MobileNet
        and other standard architectures), gather:
            I_next = Σ_{(d, w) in topology[L]} w · I[d]
        with I[d] already computed. Terminal layers (empty downstream list) seed from I_out.
        Then I[L] = W̄^L · I_next, with W̄ built by `wbar_fn` (column-stochastic for
        relative, σ_pre/L2 for non-relative; σ already folded into M, no transfer).
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
                    # w is 1.0 (sequential/fan-out) or a per-channel σ^p-share tensor (residual
                    # join) aligned to this layer's out / d's in axis.
                    w_t = w if torch.is_tensor(w) else float(w)
                    I_next = I_next + w_t * I_d

            # SCORE RECURSION I^l = W̄^l·I^{l+1} (normalized_nets_pruning.pdf steps 7-10).
            # σ already in M (=σ·W); no separate transfer. relative → column-stochastic
            # (D=1/Σ_iM^p); non-relative → D=1/σ_pre^p (L2). Identical for p=2. The residual
            # fan-in weights `w` above (σ^p/Σσ^p branch split) are a separate, orthogonal mix.
            md = meas_denom.get(name) if meas_denom else None
            cn = cov_num.get(name) if cov_num else None
            I_l = wbar_fn(M, md, cn) @ I_next
            km = keep_mask(name) if keep_mask else None
            if km is not None:
                I_l = I_l * km.to(I_l.dtype)        # pruned inputs carry no importance
            I_by_name[name] = I_l

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
        # σ EMA momentum for the input-normalizing BN.
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
        normalized net, no separate σ factor needed."""
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
