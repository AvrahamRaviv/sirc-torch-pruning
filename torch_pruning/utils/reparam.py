"""Reparameterization modules for variance-aware pruning.

Two variants, both sharing the BaseReparamManager ABC:

1. **MeanResidualManager** (original):
   Decomposes z = W^T x + b into z = m + V^T(x - μ_x), regularizes V → 0.
   Drives activation variance → 0 while preserving learned means m.

2. **NormalizedResidualManager** (VNR):
   Decomposes into z = m + Ṽ^T((x - μ_x)/σ_x), regularizes Ṽ → 0.
   σ_x normalization ensures L_{2,1} on Ṽ directly penalizes variance contribution,
   preventing the network from compensating via input scale.

Training-time only — merge_back() restores standard nn.Linear/nn.Conv2d before
pruning so the TP dependency graph, pruner, and compensation code stay untouched.

Efficiency: uses the effective-bias trick to avoid materializing centered/normalized inputs:
    MeanResidual:     z = F.linear(x, V, m - V @ μ_x)
    NormalizedResidual: z = F.linear(x, Ṽ/σ_x, m - (Ṽ/σ_x) @ μ_x)
Overhead is O(out × in) per layer per batch for the bias, negligible vs matmul.
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

# Numerical stability constants
MIN_VARIANCE = 1e-12
MIN_SIGMA = 1e-6
EPSILON = 1e-8


# =====================================================================
# Reparam module classes
# =====================================================================

class MeanResidualLinear(nn.Module):
    """Linear layer decomposed as z = m + V^T(x - μ_x), computed via effective bias."""

    def __init__(self, in_features, out_features, v, m, mu_x):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v = nn.Parameter(v)       # [out, in] — residual weights
        self.m = nn.Parameter(m)       # [out]     — channel means (frozen from WD)
        self.register_buffer('mu_x', mu_x)  # [in] — frozen input mean

    def forward(self, x):
        eff_bias = self.m - self.v @ self.mu_x  # [out] — tiny vector
        return F.linear(x, self.v, eff_bias)

    def merge_params(self):
        """Return (weight, bias) in standard nn.Linear convention."""
        weight = self.v.data.clone()
        bias = (self.m - self.v @ self.mu_x).data.clone()
        return weight, bias

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features} [MeanResidual]'


class MeanResidualConv2d(nn.Module):
    """Conv2d (groups=1) decomposed as z = m + V*(x - μ_x), computed via effective bias.

    For Conv2d, μ_x is the spatial-averaged per-channel input mean [C_in].
    The effective bias absorbs the correction: m - V.sum(dim=(2,3)) @ μ_x.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, v, m, mu_x):
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
        self.register_buffer('mu_x', mu_x)  # [C_in]

    def forward(self, x):
        # v.sum(dim=(2,3)) → [C_out, C_in], then @ mu_x → [C_out]
        eff_bias = self.m - self.v.sum(dim=(2, 3)) @ self.mu_x
        return F.conv2d(x, self.v, eff_bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def merge_params(self):
        """Return (weight, bias) in standard nn.Conv2d convention."""
        weight = self.v.data.clone()
        bias = (self.m - self.v.sum(dim=(2, 3)) @ self.mu_x).data.clone()
        return weight, bias

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding} [MeanResidual]')


class NormalizedResidualLinear(nn.Module):
    """Linear layer decomposed as z = m + Ṽ^T((x - μ_x)/σ_x).

    Forward: w_eff = Ṽ / σ_x, b_eff = m - w_eff @ μ_x, z = F.linear(x, w_eff, b_eff)
    """

    def __init__(self, in_features, out_features, v_tilde, m, mu_x, sigma_x):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v_tilde = nn.Parameter(v_tilde)  # [out, in] — normalized residual weights
        self.m = nn.Parameter(m)              # [out] — channel means
        self.register_buffer('mu_x', mu_x)    # [in] — input mean
        self.register_buffer('sigma_x', sigma_x)  # [in] — input std

    def forward(self, x):
        w_eff = self.v_tilde / self.sigma_x[None, :]  # [out, in]
        eff_bias = self.m - w_eff @ self.mu_x          # [out]
        return F.linear(x, w_eff, eff_bias)

    def merge_params(self):
        """Return (weight, bias) in standard nn.Linear convention."""
        w = self.v_tilde.data / self.sigma_x[None, :]
        bias = (self.m - w @ self.mu_x).data.clone()
        return w.clone(), bias

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features} [NormalizedResidual]'


class NormalizedResidualConv2d(nn.Module):
    """Conv2d (groups=1) decomposed as z = m + Ṽ*((x - μ_x)/σ_x).

    σ_x broadcast: [None, :, None, None] for [C_out, C_in, kH, kW].
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, v_tilde, m, mu_x, sigma_x):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.v_tilde = nn.Parameter(v_tilde)  # [C_out, C_in, kH, kW]
        self.m = nn.Parameter(m)              # [C_out]
        self.register_buffer('mu_x', mu_x)    # [C_in]
        self.register_buffer('sigma_x', sigma_x)  # [C_in]

    def forward(self, x):
        sigma_bc = self.sigma_x[None, :, None, None]           # [1, C_in, 1, 1]
        w_eff = self.v_tilde / sigma_bc                         # [C_out, C_in, kH, kW]
        eff_bias = self.m - w_eff.sum(dim=(2, 3)) @ self.mu_x  # [C_out]
        return F.conv2d(x, w_eff, eff_bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def merge_params(self):
        """Return (weight, bias) in standard nn.Conv2d convention."""
        sigma_bc = self.sigma_x[None, :, None, None]
        w = (self.v_tilde.data / sigma_bc).clone()
        bias = (self.m - w.sum(dim=(2, 3)) @ self.mu_x).data.clone()
        return w, bias

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding} [NormalizedResidual]')


# =====================================================================
# Base class (ABC) for reparameterization managers
# =====================================================================

def _residual_weight(reparam):
    """Return the residual weight tensor (v_tilde if present, else v)."""
    if hasattr(reparam, 'v_tilde'):
        return reparam.v_tilde
    return reparam.v


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
            # Store initial norms for scale-invariant L_{2,1}
            if self.scale_invariant:
                w = _residual_weight(reparam)
                init_norms = w.detach().flatten(1).norm(p=2, dim=self.norm_dim).clamp(min=EPSILON)
                reparam.register_buffer('v_init_norms', init_norms)
            self._replace_module(name, reparam)
            self._reparam_modules[name] = reparam

        self._active = True
        cls_name = type(self).__name__
        logger.info(f"{cls_name}: reparameterized {len(targets)} {self.reparam_target} modules"
                     f"{' (scale-invariant)' if self.scale_invariant else ''}"
                     f" [norm_dim={self.norm_dim}]")

    def merge_back(self):
        """Restore standard nn.Linear/nn.Conv2d from reparam modules."""
        if not self._active:
            return

        for name, reparam in self._reparam_modules.items():
            weight, bias = reparam.merge_params()
            standard = self._make_standard(reparam, weight, bias)
            self._replace_module(name, standard)

        self._reparam_modules.clear()
        self._active = False
        logger.info(f"{type(self).__name__}: merged back to standard modules")

    def reparam_param_ids(self):
        """Return set of id(p) for all trainable reparam parameters."""
        ids = set()
        for reparam in self._reparam_modules.values():
            w = _residual_weight(reparam)
            ids.add(id(w))
            ids.add(id(reparam.m))
        return ids

    def save_vnorm_snapshot(self, save_dir):
        """Save per-channel residual-weight norms as .pt file."""
        vnorms = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            w = _residual_weight(reparam).detach()
            vnorms[name] = w.flatten(1).norm(p=2, dim=self.norm_dim).cpu()
        self._last_vnorms = vnorms

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "reparam_vnorms.pt")
        torch.save(vnorms, path)
        logger.info(f"Saved V-norm snapshot to {path} ({len(vnorms)} layers)")

    def log_channel_stats(self):
        """Log per-layer residual-weight norm summary (one line per layer)."""
        stats = self.channel_stats()
        axis_desc = "row-wise, per output channel" if self.norm_dim == 1 else "column-wise, per input channel"
        logger.info(f"V-norm stats ({axis_desc}):")
        for name, s in stats.items():
            short = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            logger.info(
                f"  {short}: mean={s['v_col_norm_mean']:.4f} std={s['v_col_norm_std']:.4f} "
                f"min={s['v_col_norm_min']:.4f} max={s['v_col_norm_max']:.4f} "
                f"<0.01={s['frac_below_0.01']:.1%} <0.1={s['frac_below_0.1']:.1%} "
                f"m={s['m_mean']:.4f}±{s['m_std']:.4f}"
            )

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
                logger.warning(f"{cls_name}: target '{name}' not found, skipping")
                continue
            module = name_to_module[name]
            if isinstance(module, nn.Linear):
                targets[name] = module
            elif isinstance(module, nn.Conv2d) and module.groups == 1:
                targets[name] = module
            else:
                logger.warning(f"{cls_name}: skipping '{name}' "
                               f"({type(module).__name__}, groups={getattr(module, 'groups', 'N/A')})")
        return targets

    def _calibrate_mu(self, targets, loader):
        """Estimate μ_x for each target module via forward hooks.

        Returns dict[name → μ_x tensor]. Shared by both subclasses.
        """
        accumulators = {}
        hooks = []

        for name, module in targets.items():
            acc = {'sum': None, 'count': 0}
            accumulators[name] = acc

            def make_hook(acc_ref):
                def hook(mod, inp, out):
                    x = inp[0].detach()
                    if x.dim() == 4:
                        batch_mean = x.mean(dim=(0, 2, 3))
                    elif x.dim() == 3:
                        batch_mean = x.mean(dim=(0, 1))
                    elif x.dim() == 2:
                        batch_mean = x.mean(dim=0)
                    else:
                        return
                    n = x.shape[0]
                    if acc_ref['sum'] is None:
                        acc_ref['sum'] = batch_mean * n
                    else:
                        acc_ref['sum'] += batch_mean * n
                    acc_ref['count'] += n
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

        mu_dict = OrderedDict()
        for name, acc in accumulators.items():
            if acc['sum'] is not None and acc['count'] > 0:
                mu_dict[name] = acc['sum'] / acc['count']
            else:
                module = targets[name]
                if isinstance(module, nn.Linear):
                    mu_dict[name] = torch.zeros(module.in_features, device=self.device)
                elif isinstance(module, nn.Conv2d):
                    mu_dict[name] = torch.zeros(module.in_channels, device=self.device)
                elif hasattr(module, 'mu_x'):
                    mu_dict[name] = torch.zeros_like(module.mu_x)
                logger.warning(f"{type(self).__name__}: no data for '{name}', using zero μ_x")

        logger.info(f"{type(self).__name__}: calibrated μ_x for {len(mu_dict)} modules "
                     f"({sum(a['count'] for a in accumulators.values()) // max(len(accumulators), 1)} samples avg)")
        return mu_dict

    def _make_standard(self, reparam, weight, bias):
        """Create standard nn.Linear/Conv2d from merged params."""
        if isinstance(reparam, (MeanResidualLinear, NormalizedResidualLinear)):
            linear = nn.Linear(reparam.in_features, reparam.out_features, bias=True)
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            return linear.to(self.device)
        elif isinstance(reparam, (MeanResidualConv2d, NormalizedResidualConv2d)):
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

    Usage:
        mgr = MeanResidualManager(model, target_names, device, lambda_reg=0.01)
        mgr.reparameterize(train_loader)   # calibrate μ_x, replace modules
        ...  # train with mgr.regularization_loss() as aux loss
        mgr.merge_back()                   # restore standard modules before pruning
    """

    def __init__(self, model, target_names, device, lambda_reg=0.01, max_batches=200,
                 normalize=False, scale_invariant=False, reparam_target="fc2"):
        super().__init__(model, target_names, device, lambda_reg=lambda_reg,
                         max_batches=max_batches,
                         scale_invariant=(normalize or scale_invariant),
                         reparam_target=reparam_target)

    # Backward-compat alias
    def refresh_mu(self, loader):
        """Alias for refresh_stats (backward compatibility)."""
        return self.refresh_stats(loader)

    def _calibrate(self, targets, loader):
        """Calibrate μ_x. Returns dict[name → μ_x]."""
        return self._calibrate_mu(targets, loader)

    def _make_reparam(self, module, mu_x):
        """Create MeanResidual* module from standard nn.Linear/Conv2d."""
        if isinstance(module, nn.Linear):
            w = module.weight.data.clone()
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_features, device=self.device)
            m = b + w @ mu_x
            reparam = MeanResidualLinear(
                module.in_features, module.out_features,
                v=w, m=m, mu_x=mu_x)
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
                groups=module.groups, v=w, m=m, mu_x=mu_x)
            return reparam.to(self.device)

        raise TypeError(f"Unsupported module type: {type(module)}")

    def refresh_stats(self, loader):
        """Re-estimate μ_x (function-preserving): adjust m so output is unchanged.

        m_new = m_old + V @ (μ_old - μ_new), then update mu_x buffer.
        """
        if not self._active:
            return

        targets = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            targets[name] = reparam

        mu_new_dict = self._calibrate_mu(targets, loader)

        for name, reparam in self._reparam_modules.items():
            mu_old = reparam.mu_x
            mu_new = mu_new_dict[name]
            delta = mu_old - mu_new
            with torch.no_grad():
                if hasattr(reparam, 'in_features'):  # Linear
                    reparam.m.add_(reparam.v @ delta)
                elif hasattr(reparam, 'in_channels'):  # Conv2d
                    reparam.m.add_(reparam.v.sum(dim=(2, 3)) @ delta)
                reparam.mu_x.copy_(mu_new)

        logger.info("MeanResidualManager: refreshed μ_x (function-preserving)")

    def regularization_loss(self):
        """L_{2,1} regularization on V along self.norm_dim.

        fc2 (dim=0): column norms → per input channel (intermediate dim).
        fc1 (dim=1): row norms → per output channel (intermediate dim).
        If scale_invariant=True, each norm is divided by its initial value.
        """
        loss = torch.tensor(0.0, device=self.device)
        for reparam in self._reparam_modules.values():
            v = reparam.v
            norms = v.flatten(1).norm(p=2, dim=self.norm_dim)
            if self.scale_invariant and hasattr(reparam, 'v_init_norms'):
                loss = loss + (norms / reparam.v_init_norms).sum()
            else:
                loss = loss + norms.sum()
        return self.lambda_reg * loss

    def channel_stats(self):
        """Per-layer V-norm statistics along self.norm_dim."""
        stats = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            v = reparam.v.detach()
            col_norms = v.flatten(1).norm(p=2, dim=self.norm_dim)
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
            }
        return stats


# =====================================================================
# NormalizedResidualManager (VNR)
# =====================================================================

class NormalizedResidualManager(BaseReparamManager):
    """Variance-normalized reparameterization: z = m + Ṽ^T((x - μ_x)/σ_x).

    L_{2,1} on Ṽ directly penalizes variance contribution (||Ṽ[:,j]||² ∝ var contrib).
    Optional entropy_loss encourages uniform Ṽ column norms → balanced pruning.

    Usage:
        mgr = NormalizedResidualManager(model, target_names, device,
                                         lambda_reg=0.01, entropy_lambda=0.01)
        mgr.reparameterize(train_loader)
        ...  # train with mgr.regularization_loss() + mgr.entropy_loss()
        mgr.merge_back()
    """

    def __init__(self, model, target_names, device, lambda_reg=0.01, max_batches=200,
                 scale_invariant=False, entropy_lambda=0.0, reparam_target="fc2"):
        super().__init__(model, target_names, device, lambda_reg=lambda_reg,
                         max_batches=max_batches, scale_invariant=scale_invariant,
                         reparam_target=reparam_target)
        self._entropy_lambda = entropy_lambda

    def _calibrate(self, targets, loader):
        """Calibrate μ_x and σ_x. Returns dict[name → (μ_x, σ_x)]."""
        accumulators = {}
        hooks = []

        for name, module in targets.items():
            acc = {'sum': None, 'sum_sq': None, 'count': 0}
            accumulators[name] = acc

            def make_hook(acc_ref):
                def hook(mod, inp, out):
                    x = inp[0].detach()
                    if x.dim() == 4:
                        batch_mean = x.mean(dim=(0, 2, 3))
                        batch_mean_sq = (x * x).mean(dim=(0, 2, 3))
                    elif x.dim() == 3:
                        batch_mean = x.mean(dim=(0, 1))
                        batch_mean_sq = (x * x).mean(dim=(0, 1))
                    elif x.dim() == 2:
                        batch_mean = x.mean(dim=0)
                        batch_mean_sq = (x * x).mean(dim=0)
                    else:
                        return
                    n = x.shape[0]
                    if acc_ref['sum'] is None:
                        acc_ref['sum'] = batch_mean * n
                        acc_ref['sum_sq'] = batch_mean_sq * n
                    else:
                        acc_ref['sum'] += batch_mean * n
                        acc_ref['sum_sq'] += batch_mean_sq * n
                    acc_ref['count'] += n
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

        cal_dict = OrderedDict()
        cls_name = type(self).__name__
        for name, acc in accumulators.items():
            if acc['sum'] is not None and acc['count'] > 0:
                mu_x = acc['sum'] / acc['count']
                mean_sq = acc['sum_sq'] / acc['count']
                variance = (mean_sq - mu_x * mu_x).clamp(min=MIN_VARIANCE)
                sigma_x = variance.sqrt().clamp(min=MIN_SIGMA)
                cal_dict[name] = (mu_x, sigma_x)
            else:
                module = targets[name]
                if isinstance(module, nn.Linear):
                    d = module.in_features
                elif isinstance(module, nn.Conv2d):
                    d = module.in_channels
                elif hasattr(module, 'mu_x'):
                    d = module.mu_x.shape[0]
                else:
                    d = 1
                cal_dict[name] = (
                    torch.zeros(d, device=self.device),
                    torch.ones(d, device=self.device),
                )
                logger.warning(f"{cls_name}: no data for '{name}', using zero μ_x / unit σ_x")

        logger.info(f"{cls_name}: calibrated μ_x, σ_x for {len(cal_dict)} modules "
                     f"({sum(a['count'] for a in accumulators.values()) // max(len(accumulators), 1)} samples avg)")
        return cal_dict

    def _make_reparam(self, module, calibration_data):
        """Create NormalizedResidual* module from standard nn.Linear/Conv2d."""
        mu_x, sigma_x = calibration_data

        if isinstance(module, nn.Linear):
            W = module.weight.data.clone()  # [out, in]
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_features, device=self.device)
            # v_tilde = W * sigma_x[None,:] so that W = v_tilde / sigma_x
            v_tilde = W * sigma_x[None, :]
            m = b + W @ mu_x
            reparam = NormalizedResidualLinear(
                module.in_features, module.out_features,
                v_tilde=v_tilde, m=m, mu_x=mu_x, sigma_x=sigma_x)
            return reparam.to(self.device)

        elif isinstance(module, nn.Conv2d):
            W = module.weight.data.clone()  # [C_out, C_in, kH, kW]
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_channels, device=self.device)
            sigma_bc = sigma_x[None, :, None, None]
            v_tilde = W * sigma_bc
            m = b + W.sum(dim=(2, 3)) @ mu_x
            reparam = NormalizedResidualConv2d(
                module.in_channels, module.out_channels,
                kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation,
                groups=module.groups, v_tilde=v_tilde, m=m,
                mu_x=mu_x, sigma_x=sigma_x)
            return reparam.to(self.device)

        raise TypeError(f"Unsupported module type: {type(module)}")

    def refresh_stats(self, loader):
        """Re-estimate μ_x, σ_x (function-preserving).

        Steps:
        1. Compute w_eff_old from current v_tilde / sigma_x
        2. Calibrate new mu, sigma
        3. v_tilde *= sigma_new / sigma_old
        4. m += w_eff_old @ (mu_old - mu_new)
        5. Update buffers
        """
        if not self._active:
            return

        targets = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            targets[name] = reparam

        cal_new = self._calibrate(targets, loader)

        for name, reparam in self._reparam_modules.items():
            mu_old = reparam.mu_x.clone()
            sigma_old = reparam.sigma_x.clone()
            mu_new, sigma_new = cal_new[name]

            with torch.no_grad():
                # Step 1: effective weight before change
                if hasattr(reparam, 'in_features'):  # Linear
                    w_eff_old = reparam.v_tilde / sigma_old[None, :]
                    # Step 3: rescale v_tilde
                    reparam.v_tilde.mul_(sigma_new[None, :] / sigma_old[None, :])
                    # Step 4: adjust m
                    reparam.m.add_(w_eff_old @ (mu_old - mu_new))
                elif hasattr(reparam, 'in_channels'):  # Conv2d
                    sigma_old_bc = sigma_old[None, :, None, None]
                    sigma_new_bc = sigma_new[None, :, None, None]
                    w_eff_old = reparam.v_tilde / sigma_old_bc
                    reparam.v_tilde.mul_(sigma_new_bc / sigma_old_bc)
                    reparam.m.add_(w_eff_old.sum(dim=(2, 3)) @ (mu_old - mu_new))

                # Step 5: update buffers
                reparam.mu_x.copy_(mu_new)
                reparam.sigma_x.copy_(sigma_new)

        logger.info("NormalizedResidualManager: refreshed μ_x, σ_x (function-preserving)")

    def regularization_loss(self):
        """L_{2,1} on Ṽ along self.norm_dim."""
        loss = torch.tensor(0.0, device=self.device)
        for reparam in self._reparam_modules.values():
            v = reparam.v_tilde
            norms = v.flatten(1).norm(p=2, dim=self.norm_dim)
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
            v = reparam.v_tilde
            norms_sq = v.flatten(1).norm(p=2, dim=self.norm_dim) ** 2
            p = norms_sq / (norms_sq.sum() + EPSILON)
            loss = loss - (p * (p + EPSILON).log()).sum()
        return self._entropy_lambda * loss

    def channel_stats(self):
        """Per-layer dual reporting: Ṽ norms AND w_eff norms along self.norm_dim."""
        stats = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            vt = reparam.v_tilde.detach()
            vt_col_norms = vt.flatten(1).norm(p=2, dim=self.norm_dim)

            # w_eff = v_tilde / sigma_x
            if hasattr(reparam, 'in_features'):
                w_eff = vt / reparam.sigma_x[None, :]
            else:
                w_eff = vt / reparam.sigma_x[None, :, None, None]
            weff_col_norms = w_eff.flatten(1).norm(p=2, dim=self.norm_dim)

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
