"""Mean-Residual Reparameterization for variance-aware pruning.

Decomposes z = W^T x + b into z = m + V^T(x - μ_x), then regularizes only V
toward zero. This drives activation variance → 0 while preserving learned means m,
making VBP compensation exact after pruning.

Training-time only — merge_back() restores standard nn.Linear/nn.Conv2d before
pruning so the TP dependency graph, pruner, and compensation code stay untouched.

Efficiency: uses the effective-bias trick to avoid materializing (x - μ_x):
    z = F.linear(x, V, m - V @ μ_x)
Overhead is O(out × in) per layer per batch for the bias, negligible vs matmul.
"""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("vbp_imagenet")


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


class MeanResidualManager:
    """Lifecycle orchestrator for mean-residual reparameterization.

    Usage:
        mgr = MeanResidualManager(model, target_names, device, lambda_reg=0.01)
        mgr.reparameterize(train_loader)   # calibrate μ_x, replace modules
        ...  # train with mgr.regularization_loss() as aux loss
        mgr.merge_back()                   # restore standard modules before pruning
    """

    def __init__(self, model, target_names, device, lambda_reg=0.01, max_batches=200):
        """
        Args:
            model: The nn.Module to reparameterize.
            target_names: List of dotted module names to reparameterize
                          (e.g. ["vit.encoder.layer.0.intermediate.dense", ...]).
            device: Torch device (all buffers stay on this device).
            lambda_reg: L_{2,1} regularization strength for V.
            max_batches: Max batches for μ_x calibration.
        """
        self.model = model
        self.target_names = target_names
        self.device = device
        self.lambda_reg = lambda_reg
        self.max_batches = max_batches
        self._active = False
        self._reparam_modules = OrderedDict()  # name → MeanResidual* module

    @property
    def is_active(self):
        return self._active

    def reparameterize(self, calibration_loader):
        """Calibrate μ_x and replace target modules with MeanResidual* variants."""
        if self._active:
            raise RuntimeError("Already reparameterized. Call merge_back() first.")

        # 1. Resolve target_names → module objects
        targets = self._resolve_targets()

        # 2. Calibrate μ_x for each target
        mu_dict = self._calibrate(targets, calibration_loader)

        # 3. Replace target modules with MeanResidual* variants
        for name, module in targets.items():
            mu_x = mu_dict[name]
            reparam = self._make_reparam(module, mu_x)
            self._replace_module(name, reparam)
            self._reparam_modules[name] = reparam

        self._active = True
        logger.info(f"MeanResidualManager: reparameterized {len(targets)} modules")

    def merge_back(self):
        """Restore standard nn.Linear/nn.Conv2d from MeanResidual* modules."""
        if not self._active:
            return

        for name, reparam in self._reparam_modules.items():
            weight, bias = reparam.merge_params()
            standard = self._make_standard(reparam, weight, bias)
            self._replace_module(name, standard)

        self._reparam_modules.clear()
        self._active = False
        logger.info("MeanResidualManager: merged back to standard modules")

    def refresh_mu(self, loader):
        """Re-estimate μ_x (function-preserving): adjust m so output is unchanged.

        m_new = m_old + V @ (μ_old - μ_new), then update mu_x buffer.
        """
        if not self._active:
            return

        targets = OrderedDict()
        for name, reparam in self._reparam_modules.items():
            targets[name] = reparam

        mu_new_dict = self._calibrate(targets, loader)

        for name, reparam in self._reparam_modules.items():
            mu_old = reparam.mu_x
            mu_new = mu_new_dict[name]
            delta = mu_old - mu_new
            with torch.no_grad():
                if isinstance(reparam, MeanResidualLinear):
                    reparam.m.add_(reparam.v @ delta)
                elif isinstance(reparam, MeanResidualConv2d):
                    reparam.m.add_(reparam.v.sum(dim=(2, 3)) @ delta)
                reparam.mu_x.copy_(mu_new)

        logger.info("MeanResidualManager: refreshed μ_x (function-preserving)")

    def regularization_loss(self):
        """L_{2,1} regularization on V: λ · Σ_l Σ_k ||v_k^(l)||_2.

        Returns a scalar tensor on device (all GPU, no CPU transfers).
        """
        loss = torch.tensor(0.0, device=self.device)
        for reparam in self._reparam_modules.values():
            v = reparam.v  # [out, in] or [C_out, C_in, kH, kW]
            loss = loss + v.flatten(1).norm(p=2, dim=1).sum()
        return self.lambda_reg * loss

    def reparam_param_ids(self):
        """Return set of id(p) for all m, v parameters (for optimizer grouping)."""
        ids = set()
        for reparam in self._reparam_modules.values():
            ids.add(id(reparam.v))
            ids.add(id(reparam.m))
        return ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_targets(self):
        """Resolve target_names to an OrderedDict of name → module."""
        name_to_module = dict(self.model.named_modules())
        targets = OrderedDict()
        for name in self.target_names:
            if name not in name_to_module:
                logger.warning(f"MeanResidualManager: target '{name}' not found, skipping")
                continue
            module = name_to_module[name]
            if isinstance(module, nn.Linear):
                targets[name] = module
            elif isinstance(module, nn.Conv2d) and module.groups == 1:
                targets[name] = module
            else:
                logger.warning(f"MeanResidualManager: skipping '{name}' "
                               f"({type(module).__name__}, groups={getattr(module, 'groups', 'N/A')})")
        return targets

    def _calibrate(self, targets, loader):
        """Estimate μ_x for each target module via forward hooks.

        Accumulates on GPU (no .cpu() transfers). Returns dict[name → μ_x tensor].
        """
        accumulators = {}  # name → (sum, count)
        hooks = []

        for name, module in targets.items():
            acc = {'sum': None, 'count': 0}
            accumulators[name] = acc

            def make_hook(acc_ref):
                def hook(mod, inp, out):
                    x = inp[0].detach()
                    if x.dim() == 4:
                        # Conv2d input: [B, C, H, W] → mean over B, H, W
                        batch_mean = x.mean(dim=(0, 2, 3))  # [C]
                    elif x.dim() == 3:
                        # Linear (transformer): [B, T, C] → mean over B, T
                        batch_mean = x.mean(dim=(0, 1))  # [C]
                    elif x.dim() == 2:
                        # Linear (classification): [B, C] → mean over B
                        batch_mean = x.mean(dim=0)  # [C]
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

        # Forward pass through loader
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

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute means
        mu_dict = OrderedDict()
        for name, acc in accumulators.items():
            if acc['sum'] is not None and acc['count'] > 0:
                mu_dict[name] = acc['sum'] / acc['count']
            else:
                # Fallback: zero mean
                module = targets[name]
                if isinstance(module, nn.Linear):
                    mu_dict[name] = torch.zeros(module.in_features, device=self.device)
                elif isinstance(module, nn.Conv2d):
                    mu_dict[name] = torch.zeros(module.in_channels, device=self.device)
                elif isinstance(module, (MeanResidualLinear, MeanResidualConv2d)):
                    mu_dict[name] = torch.zeros_like(module.mu_x)
                logger.warning(f"MeanResidualManager: no data for '{name}', using zero μ_x")

        logger.info(f"MeanResidualManager: calibrated μ_x for {len(mu_dict)} modules "
                     f"({sum(a['count'] for a in accumulators.values()) // len(accumulators)} samples avg)")
        return mu_dict

    def _make_reparam(self, module, mu_x):
        """Create MeanResidual* module from standard nn.Linear/Conv2d."""
        if isinstance(module, nn.Linear):
            w = module.weight.data.clone()  # [out, in]
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_features, device=self.device)
            # m = b + w @ μ_x  (so that eff_bias = m - v @ μ_x = b when v = w)
            m = b + w @ mu_x
            reparam = MeanResidualLinear(
                module.in_features, module.out_features,
                v=w, m=m, mu_x=mu_x)
            return reparam.to(self.device)

        elif isinstance(module, nn.Conv2d):
            w = module.weight.data.clone()  # [C_out, C_in, kH, kW]
            b = module.bias.data.clone() if module.bias is not None else torch.zeros(
                module.out_channels, device=self.device)
            # m = b + w.sum(dim=(2,3)) @ μ_x
            m = b + w.sum(dim=(2, 3)) @ mu_x
            reparam = MeanResidualConv2d(
                module.in_channels, module.out_channels,
                kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation,
                groups=module.groups, v=w, m=m, mu_x=mu_x)
            return reparam.to(self.device)

        raise TypeError(f"Unsupported module type: {type(module)}")

    def _make_standard(self, reparam, weight, bias):
        """Create standard nn.Linear/Conv2d from merged params."""
        if isinstance(reparam, MeanResidualLinear):
            linear = nn.Linear(reparam.in_features, reparam.out_features, bias=True)
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            return linear.to(self.device)

        elif isinstance(reparam, MeanResidualConv2d):
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
