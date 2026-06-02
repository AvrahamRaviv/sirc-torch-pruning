"""Unified checkpoint save/load for the normnet pipeline (train / normalize / prune / ft / infer).

A checkpoint is a single self-describing dict bundle (torch.save). It carries:
  - "model"          : the deployable nn.Module, pickled IN FULL. Reload is then trivial for
                       ANY structure — pruned (reduced channel dims), merged, or reparam'd
                       (BNResidual/MeanResidual), because all those classes live in this repo.
                       State-dict-only saving can't rebuild a pruned arch; the full object can.
  - "ema_model"      : the EMA shadow (or None). THIS is the model you report/deploy for the
                       v2 recipe — torchvision's 80.86 is the EMA model, not the raw one. EMA
                       is a running average over the whole trajectory and CANNOT be recovered
                       from the raw checkpoint, so it must be saved explicitly.
  - "state_dict" / "ema_state_dict" : portable weights (inspection / cross-tool load).
  - "kind", "arch", "meta" : provenance + metrics.

Load with load_ckpt(path, prefer="ema") to get the deployable model; prefer="raw" for the
training-trajectory endpoint (e.g. to keep training). Works on legacy bare-state_dict and
bare-nn.Module saves too.

Note: full-object load uses weights_only=False (pickle executes). Fine for our own cluster
artifacts; don't load untrusted bundles.
"""
import copy

import torch
import torch.nn as nn

FORMAT = "normnet-ckpt-v1"


def _detached_cpu_copy(model):
    """A CPU deepcopy that never disturbs the live (possibly CUDA / training) model."""
    if model is None:
        return None
    return copy.deepcopy(model).cpu().eval()


def _sd_cpu(model):
    return None if model is None else {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}


def save_ckpt(path, model, *, kind, arch=None, ema_model=None, meta=None):
    """Save a bundle. `model` = deployable net (dense/pruned/merged/reparam'd). `ema_model`
    = the EMA shadow when one was kept (the reported/deployed model). For the switch arm,
    merge the shadow (merge_reparam_modules) BEFORE passing it here so it saves in plain form."""
    bundle = {
        "format": FORMAT, "kind": kind, "arch": arch, "meta": meta or {},
        "model": _detached_cpu_copy(model),
        "ema_model": _detached_cpu_copy(ema_model),
        "state_dict": _sd_cpu(model),
        "ema_state_dict": _sd_cpu(ema_model),
    }
    torch.save(bundle, path)
    return path


def load_ckpt(path, device="cpu", *, prefer="ema"):
    """Return a ready (.eval()) nn.Module. prefer='ema' → the EMA model when present (the
    deployable / reported one), else the raw model. Handles our bundles + legacy saves."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and obj.get("format") == FORMAT:
        m = obj["ema_model"] if (prefer == "ema" and obj.get("ema_model") is not None) else obj["model"]
        if m is None:
            raise ValueError(f"{path}: bundle has no usable model (kind={obj.get('kind')})")
        return m.to(device).eval()
    if isinstance(obj, nn.Module):
        return obj.to(device).eval()
    raise ValueError(f"{path}: legacy state_dict bundle — rebuild the arch and load_state_dict "
                     f"manually, or re-save via save_ckpt.")


def is_bundle(path):
    """True if `path` is a normnet-ckpt bundle (cheap header peek)."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(obj, dict) and obj.get("format") == FORMAT
    except Exception:
        return False


def merge_reparam_modules(model):
    """In-place: replace every BNResidual*/MeanResidual* module with its merged plain
    nn.Conv2d/Linear (function-preserving). Operates on ANY model — used to merge an EMA
    shadow that the reparam manager doesn't track (the manager only merges the raw model).
    No-op if the model has no reparam'd modules. Returns the model."""
    from torch_pruning.utils.reparam import (
        BNResidualLinear, BNResidualConv2d, MeanResidualLinear, MeanResidualConv2d)
    conv_t = (BNResidualConv2d, MeanResidualConv2d)
    lin_t = (BNResidualLinear, MeanResidualLinear)
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, conv_t + lin_t):
            continue
        w, b = mod.merge_params()
        if isinstance(mod, conv_t):
            new = nn.Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride,
                            mod.padding, mod.dilation, mod.groups, bias=True)
        else:
            new = nn.Linear(mod.in_features, mod.out_features, bias=True)
        new.weight.data.copy_(w)
        new.bias.data.copy_(b)
        new = new.to(w.device)
        # install at the dotted name
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attr, new)
    return model
