"""
Tests for normalize-net-by-construction (whole-net VNR transform + save/reload).

Covers the new pieces layered on top of NormalizedResidualManager:
  - build_whole_net_reparam_layers selector
  - channels-last (NHWC Linear) reparam roundtrip — the reparam.py fix
  - attach_biases + _merge_vnr_state_dict save/reload roundtrips (incl. trained bias)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "benchmarks", "vbp"))

import torch
import torch.nn as nn

from torch_pruning.utils.reparam import NormalizedResidualManager
from vbp_common import (
    build_whole_net_reparam_layers, attach_biases, _merge_vnr_state_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(3, 16, 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.project = nn.Conv2d(16, 8, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.bn(self.expand(x)))
        x = self.project(x)
        return self.fc(self.pool(x).flatten(1))


class TinyChannelsLast(nn.Module):
    """Depthwise conv -> NHWC -> Linear/GELU/Linear -> NCHW (ConvNeXt block shape)."""
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)  # depthwise: skipped by manager
        self.pwconv1 = nn.Linear(8, 16)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(16, 8)

    def forward(self, x):
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)          # NCHW -> NHWC
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 3, 1, 2)          # NHWC -> NCHW
        return x


class BiaslessMLP(nn.Module):
    """fc2 has bias=False so merge-back introduces a new (nonzero, once trained) bias."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _vec_loader(dim=16, n=8, bs=2):
    data = torch.randn(n, dim)
    ds = torch.utils.data.TensorDataset(data, torch.zeros(n, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=bs)


def _img_loader(c=3, hw=8, n=8, bs=2):
    data = torch.randn(n, c, hw, hw)
    ds = torch.utils.data.TensorDataset(data, torch.zeros(n, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, batch_size=bs)


CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
class SelectorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)          # groups==1 conv
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8)  # depthwise -> excluded
        self.pw = nn.Conv2d(8, 16, 1)                      # groups==1 conv
        self.fc1 = nn.Linear(16, 32)
        self.head = nn.Linear(32, 10)                      # classifier

    def forward(self, x):
        return x


def test_whole_net_selector():
    model = SelectorNet()

    default = build_whole_net_reparam_layers(model)  # exclude_classifier=True
    assert "dw" not in default, "depthwise conv must be excluded"
    assert "head" not in default, "classifier must be excluded by default"
    assert set(default) == {"stem", "pw", "fc1"}
    # order follows named_modules()
    assert default == ["stem", "pw", "fc1"]

    with_head = build_whole_net_reparam_layers(model, exclude_classifier=False)
    assert "head" in with_head and set(with_head) == {"stem", "pw", "fc1", "head"}

    no_stem = build_whole_net_reparam_layers(model, exclude_stem=True)
    assert "stem" not in no_stem and set(no_stem) == {"pw", "fc1"}

    excl = build_whole_net_reparam_layers(model, exclude=["pw"])
    assert "pw" not in excl


# ---------------------------------------------------------------------------
# Roundtrips: reparameterize -> merge_back, output preserved
# ---------------------------------------------------------------------------
def _reparam_merge_diff(model, layer_names, loader, x):
    model.eval()
    with torch.no_grad():
        y0 = model(x).clone()

    mgr = NormalizedResidualManager(model, layer_names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(loader)
    model.eval()
    with torch.no_grad():
        y1 = model(x).clone()

    mgr.merge_back()
    model.eval()
    with torch.no_grad():
        y2 = model(x).clone()

    return (y1 - y0).abs().max().item(), (y2 - y0).abs().max().item()


def test_roundtrip_linear():
    model = TinyMLP()
    names = build_whole_net_reparam_layers(model, exclude_classifier=False)
    d1, d2 = _reparam_merge_diff(model, names, _vec_loader(16), torch.randn(4, 16))
    assert d1 < 1e-5 and d2 < 1e-5, (d1, d2)


def test_roundtrip_conv():
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model, exclude_classifier=False)
    d1, d2 = _reparam_merge_diff(model, names, _img_loader(3, 8), torch.randn(4, 3, 8, 8))
    assert d1 < 1e-4 and d2 < 1e-4, (d1, d2)


def test_roundtrip_channels_last():
    """ConvNeXt-shaped NHWC Linear input — exercises the channels-last fix."""
    model = TinyChannelsLast()
    d1, d2 = _reparam_merge_diff(model, ["pwconv1", "pwconv2"],
                                 _img_loader(8, 8), torch.randn(4, 8, 8, 8))
    assert d1 < 1e-4 and d2 < 1e-4, (d1, d2)


# ---------------------------------------------------------------------------
# Save / reload roundtrips with a trained (nonzero) bias on a bias=False layer
# ---------------------------------------------------------------------------
def _simulate_training(reparam):
    """Perturb v_tilde and m so the merged bias becomes nonzero."""
    with torch.no_grad():
        reparam.v_tilde.add_(torch.randn_like(reparam.v_tilde) * 0.1)
        reparam.m.add_(torch.randn_like(reparam.m) * 0.1)


def test_trained_bias_roundtrip():
    """VNR-format reload: attach_biases + _merge_vnr_state_dict -> strict load."""
    torch.manual_seed(0)
    model = BiaslessMLP()
    x = torch.randn(4, 16)

    mgr = NormalizedResidualManager(model, ["fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))
    _simulate_training(mgr._reparam_modules["fc2"])

    model.eval()
    with torch.no_grad():
        y_ref = model(x).clone()
    vnr_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    fresh = BiaslessMLP()
    attach_biases(fresh, ["fc2"])
    fresh.load_state_dict(_merge_vnr_state_dict(vnr_state), strict=True)
    fresh.eval()
    with torch.no_grad():
        y_reload = fresh(x)

    assert (y_reload - y_ref).abs().max().item() < 1e-5


def test_merged_biased_roundtrip():
    """merged_biased-format reload: attach_biases -> strict load (no VNR conversion)."""
    torch.manual_seed(1)
    model = BiaslessMLP()
    x = torch.randn(4, 16)

    mgr = NormalizedResidualManager(model, ["fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))
    _simulate_training(mgr._reparam_modules["fc2"])

    model.eval()
    with torch.no_grad():
        y_ref = model(x).clone()

    mgr.merge_back()
    merged_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    fresh = BiaslessMLP()
    attach_biases(fresh, ["fc2"])
    fresh.load_state_dict(merged_state, strict=True)
    fresh.eval()
    with torch.no_grad():
        y_reload = fresh(x)

    assert (y_reload - y_ref).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Real architecture: resnet18 whole-net transform + both reload formats
# ---------------------------------------------------------------------------
def test_real_arch_forward_equality():
    import torchvision.models as tv

    torch.manual_seed(2)
    model = tv.resnet18(weights=None)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y0 = model(x).clone()

    names = build_whole_net_reparam_layers(model, exclude_classifier=True, exclude_stem=True)
    assert "fc" not in names and "conv1" not in names  # classifier + stem dropped
    assert len(names) > 5

    mgr = NormalizedResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=2)
    mgr.reparameterize(_img_loader(3, 64, n=4, bs=2))
    resolved = list(mgr._reparam_modules.keys())
    model.eval()  # swapped-in reparam modules default to train mode; eval freezes BN stats
    with torch.no_grad():
        y1 = model(x)
    vnr_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    mgr.merge_back()
    model.eval()
    with torch.no_grad():
        y2 = model(x)
    merged_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Reload merged_biased
    m_b = tv.resnet18(weights=None)
    attach_biases(m_b, resolved)
    m_b.load_state_dict(merged_state, strict=True)
    m_b.eval()
    with torch.no_grad():
        y3 = m_b(x)

    # Reload vnr
    m_a = tv.resnet18(weights=None)
    attach_biases(m_a, resolved)
    m_a.load_state_dict(_merge_vnr_state_dict(vnr_state), strict=True)
    m_a.eval()
    with torch.no_grad():
        y_vnr = m_a(x)

    for y in (y1, y2, y3, y_vnr):
        assert (y - y0).abs().max().item() < 1e-4


def test_script_end_to_end(tmp_path):
    """Drive normalize_net's real save/reload/verify functions on resnet18."""
    from types import SimpleNamespace
    import torchvision.models as tv
    import normalize_net as nn_mod

    ckpt = str(tmp_path / "r18.pth")
    torch.save(tv.resnet18(weights=None).state_dict(), ckpt)

    args = SimpleNamespace(
        model_type="cnn", model_name="resnet18", cnn_arch="resnet18",
        checkpoint=ckpt, max_batches=2, exclude_layers=None,
        exclude_classifier=True, exclude_stem=True,
        verify_atol=1e-4, validate=False, save_tag="test",
        save_dir=str(tmp_path), rank=0,
    )
    device = torch.device("cpu")
    model = nn_mod.load_model(args, device)
    names = nn_mod.build_whole_net_reparam_layers(
        model, exclude_classifier=True, exclude_stem=True)
    mgr = NormalizedResidualManager(model, names, device, lambda_reg=0.0, max_batches=2)

    ok = nn_mod.verify_roundtrip(model, mgr, _img_loader(3, 64, n=4, bs=2),
                                 args, device, use_ddp=False)
    assert ok
    for fmt in ("vnr", "merged_biased"):
        assert os.path.exists(os.path.join(str(tmp_path), f"test_{fmt}.pth"))
        assert os.path.exists(os.path.join(str(tmp_path), f"test_{fmt}.meta.json"))
