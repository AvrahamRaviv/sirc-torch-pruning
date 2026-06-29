"""Measured-variance compensation for pruned nets — the robust, BN-free-friendly dual of --var_comp.

After pruning a producer's output channels, each consumer y = Wx + b sees fewer input channels, so
its output variance drops below the dense value. Analytic --var_comp rescales consumer output rows by
s_j = sqrt(var_full / var_kept) from a PER-LAYER covariance estimate — fragile, because that per-layer
error compounds across depth (the resnet50 / mnv2 cluster crater: var_kept→0 over-amplifies).

This module instead MEASURES both variances by forward passes over calib data:
  - var_full : consumer pre-bias output variance on the DENSE net (pre-prune snapshot).
  - var_kept : consumer pre-bias output variance on the ACTUAL pruned net (post-prune, so the
               compounded distribution shift from all upstream prunes is already baked in).
s_j = clamp(sqrt(var_full / var_kept), 1, s_max), applied to the consumer's output rows; the bias is
shifted by -(s_j-1)*mean_kept so the output MEAN is preserved (bias_comp already restored it to dense).

This is the measure-pass champion (protocol E) generalized to nets with no BatchNorm to recalibrate
(convnext = LayerNorm, deit = LayerNorm). Self-contained: NO base_pruner / normalize_net changes; it
owns its consumer-row rescale (mirrors base_pruner._apply_compensation's variance path). Wired into
normnet_main via --ln_measure_pass: snapshot dense var before _pruner.step(), apply after.
"""
import torch
import torch.nn as nn

_PRUNABLE = (nn.Conv2d, nn.Linear)


def _in_dim(m):
    return m.in_channels if isinstance(m, nn.Conv2d) else m.in_features


@torch.no_grad()
def _collect_output_stats(model, loader, device, max_batches, targets):
    """Per-output-channel pre-bias activation mean/var of each target module over <=max_batches calib
    batches. z = (module output) - bias; reduce over batch (+ spatial for conv) → per-channel mean/var.
    Returns {module: (mean[C], var[C])} (float, on `device`). Two-pass-free: sum / sumsq accumulators."""
    targets = set(targets)
    acc = {}                                            # module -> [n, sum[C], sumsq[C]]

    def _make(mod):
        def hook(m, inp, out):
            o = out
            # Channel axis is module-type-dependent, NOT rank-dependent: Conv2d → [N,C,H,W] (dim 1);
            # Linear → channel LAST (e.g. convnext pwconv is Linear with 4D [N,H,W,C] input/output).
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    o = o - m.bias.view(1, -1, 1, 1)
                x = o.permute(1, 0, 2, 3).reshape(o.shape[1], -1).float()   # [C, N*H*W]
            else:                                       # Linear: channel = last dim, any leading dims
                if m.bias is not None:
                    o = o - m.bias                      # broadcasts over last dim
                x = o.reshape(-1, o.shape[-1]).t().float()                  # [C, prod(rest)]
            su, sq, n = x.sum(1), (x * x).sum(1), x.shape[1]
            s = acc.get(m)
            if s is None:
                acc[m] = [n, su, sq]
            else:
                s[0] += n; s[1] += su; s[2] += sq
        return hook

    handles = [m.register_forward_hook(_make(m)) for m in targets]
    was_training = model.training
    model.eval()
    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(x.to(device, non_blocking=True))
        seen += 1
    for h in handles:
        h.remove()
    if was_training:
        model.train()

    out = {}
    for m, (n, su, sq) in acc.items():
        mean = su / n
        out[m] = (mean, (sq / n - mean * mean).clamp_min(0.0))
    return out


@torch.no_grad()
def collect_dense_output_var(model, loader, device, max_batches):
    """PRE-prune snapshot: {module: (in_dim, var[C])} for every prunable layer, measured on the dense
    net. in_dim is stored so the post-prune step can identify which layers are consumers (in_dim
    shrank). Call BEFORE _pruner.step()."""
    targets = [m for m in model.modules() if isinstance(m, _PRUNABLE)]
    stats = _collect_output_stats(model, loader, device, max_batches, targets)
    return {m: (_in_dim(m), var) for m, (_mean, var) in stats.items()}


@torch.no_grad()
def apply_measured_var_comp(model, dense, loader, device, max_batches, s_max=8.0, log=None):
    """POST-prune: consumers = prunable layers whose in_dim shrank vs the dense snapshot. Measure their
    pruned pre-bias output var, s_j = clamp(sqrt(var_full/var_kept), 1, s_max), scale output rows, shift
    bias by -(s_j-1)*mean_kept to hold the mean. Returns #consumers scaled. Call AFTER _pruner.step()."""
    consumers = [m for m, (d0, _v) in dense.items()
                 if isinstance(m, _PRUNABLE) and _in_dim(m) != d0]
    if not consumers:
        if log:
            log("ln_measure_pass: no consumer changed in-dim → nothing to compensate")
        return 0

    stats = _collect_output_stats(model, loader, device, max_batches, consumers)
    n_scaled = 0
    s_max_seen, s_sum, n_ch, sat = 0.0, 0.0, 0, 0
    for m in consumers:
        if m not in stats:
            continue
        var_full = dense[m][1].to(device)
        mean_k, var_k = (t.to(device) for t in stats[m])
        eps = 1e-8 * var_full.max().clamp_min(1e-12)
        s = (var_full / var_k.clamp_min(eps)).sqrt().clamp_(1.0, s_max)
        if isinstance(m, nn.Conv2d):
            m.weight.data.mul_(s.view(-1, 1, 1, 1))
        else:
            m.weight.data.mul_(s.view(-1, 1))
        if m.bias is not None:
            m.bias.data.add_(-(s - 1.0) * mean_k)       # new out = s·z + b' ; preserves output mean
        n_scaled += 1
        s_max_seen = max(s_max_seen, float(s.max()))
        s_sum += float(s.sum()); n_ch += s.numel()
        sat += int((s >= s_max - 1e-3).sum())

    if log:
        log(f"ln_measure_pass: scaled {n_scaled} consumers | s: max={s_max_seen:.2f} "
            f"mean={s_sum / max(n_ch, 1):.2f} saturated@{s_max}={sat}/{n_ch} "
            f"({100 * sat / max(n_ch, 1):.1f}%)")
    return n_scaled