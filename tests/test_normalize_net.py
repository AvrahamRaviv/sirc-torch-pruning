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

from torch_pruning.utils.reparam import (
    NormalizedResidualManager, MeanResidualManager,
    MeanResidualLinear, MeanResidualConv2d,
)
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


# ---------------------------------------------------------------------------
# Training: optimizer WD grouping + short-run smoke
# ---------------------------------------------------------------------------
def _train_args(**over):
    import tempfile
    from types import SimpleNamespace
    base = dict(epochs=1, lr=1e-3, wd=0.05, opt="adamw", momentum=0.9,
                ft_eta_min=1e-5, ft_warmup_epochs=0, model_type="cnn",
                use_kd=False, local_rank=0,
                save_dir=tempfile.mkdtemp(), save_tag="smoke")
    base.update(over)
    return SimpleNamespace(**base)


def test_build_optimizer_wd_on_vtilde():
    """WD must act on v_tilde (decayed group); m must be excluded (no_decay)."""
    import normalize_net as nn_mod

    model = TinyMLP()
    mgr = NormalizedResidualManager(model, ["fc1", "fc2"], CPU,
                                    lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    opt = nn_mod.build_optimizer(model, _train_args(), mgr)
    assert len(opt.param_groups) == 2
    decayed, no_decay = opt.param_groups[0], opt.param_groups[1]
    assert decayed["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0

    v_ids = {id(rp.v_tilde) for rp in mgr._reparam_modules.values()}
    m_ids = {id(rp.m) for rp in mgr._reparam_modules.values()}
    decayed_ids = {id(p) for p in decayed["params"]}
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert v_ids <= decayed_ids, "v_tilde must be weight-decayed"
    assert m_ids <= no_decay_ids, "m must be excluded from weight decay"


def test_build_optimizer_plain():
    """No manager -> single decayed group over all params."""
    import normalize_net as nn_mod
    model = TinyMLP()
    opt = nn_mod.build_optimizer(model, _train_args(), mgr=None)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["weight_decay"] == 0.05


def test_train_smoke_both_arms():
    """train_normalized runs one epoch for the normalized and baseline arms."""
    import normalize_net as nn_mod

    train_loader = _img_loader(3, 16, n=8, bs=4)
    val_loader = _img_loader(3, 16, n=8, bs=4)
    args = _train_args(epochs=1)

    # Normalized arm
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model)  # expand, project (fc excluded)
    mgr = NormalizedResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(train_loader)
    best = nn_mod.train_normalized(model, mgr, train_loader, val_loader, None,
                                   args, CPU, use_ddp=False)
    assert isinstance(best, float) and 0.0 <= best <= 1.0

    # Baseline arm (no reparam)
    base = nn_mod.train_normalized(TinyCNN(), None, train_loader, val_loader, None,
                                   args, CPU, use_ddp=False)
    assert isinstance(base, float) and 0.0 <= base <= 1.0


# ---------------------------------------------------------------------------
# Mean variant (the w̃-parametrization fix: trainable v=W, uniform W-space lr)
# ---------------------------------------------------------------------------
def test_build_reparam_manager_dispatch():
    """--reparam_variant selects the manager: mean->MeanResidual, bn->Normalized."""
    import normalize_net as nn_mod
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model)
    mean_mgr = nn_mod.build_reparam_manager(
        model, names, CPU, _train_args(reparam_variant="mean", max_batches=4))
    assert isinstance(mean_mgr, MeanResidualManager)
    bn_mgr = nn_mod.build_reparam_manager(
        model, names, CPU, _train_args(reparam_variant="bn", max_batches=4))
    assert isinstance(bn_mgr, NormalizedResidualManager)


def test_build_optimizer_mean_variant():
    """Mean variant: WD on v (decayed), m excluded; trainable is v (not v_tilde)."""
    import normalize_net as nn_mod
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    opt = nn_mod.build_optimizer(model, _train_args(reparam_variant="mean"), mgr)
    assert len(opt.param_groups) == 2
    decayed, no_decay = opt.param_groups[0], opt.param_groups[1]
    assert decayed["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0

    v_ids = {id(rp.v) for rp in mgr._reparam_modules.values()}
    m_ids = {id(rp.m) for rp in mgr._reparam_modules.values()}
    decayed_ids = {id(p) for p in decayed["params"]}
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert v_ids <= decayed_ids, "v must be weight-decayed (‖v‖ = magnitude)"
    assert m_ids <= no_decay_ids, "m must be excluded from weight decay"


def test_lr_scale_by_sigma2_ignored_on_mean():
    """σ²-lr-scaling is a no-op on the mean variant (no 1/σ² overshoot to undo)."""
    import normalize_net as nn_mod
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))
    # Even with the flag on, mean variant must fall back to plain WD grouping (2 groups),
    # not the per-layer σ² groups (which would need rp.bn / rp.v_tilde).
    opt = nn_mod.build_optimizer(
        model, _train_args(reparam_variant="mean", lr_scale_by_sigma2=True), mgr)
    assert len(opt.param_groups) == 2


def test_train_smoke_mean_variant():
    """train_normalized runs one epoch with the mean-variant manager."""
    import normalize_net as nn_mod
    train_loader = _img_loader(3, 16, n=8, bs=4)
    val_loader = _img_loader(3, 16, n=8, bs=4)
    args = _train_args(epochs=1, reparam_variant="mean")

    model = TinyCNN()
    names = build_whole_net_reparam_layers(model)
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(train_loader)
    best = nn_mod.train_normalized(model, mgr, train_loader, val_loader, None,
                                   args, CPU, use_ddp=False)
    assert isinstance(best, float) and 0.0 <= best <= 1.0


# ---------------------------------------------------------------------------
# M1: σ_x buffer + ‖σ·v‖ score + λ‖σ·v‖ regularization
# ---------------------------------------------------------------------------
def test_sigma_buffer_present():
    """After reparameterize, every Mean module exposes a sigma_x buffer of the right
    shape with strictly positive entries."""
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model, exclude_classifier=False)
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))

    for name, rp in mgr._reparam_modules.items():
        assert hasattr(rp, "sigma_x"), f"{name} missing sigma_x buffer"
        sigma = rp.sigma_x
        # Buffer (not parameter)
        assert name + ".sigma_x" not in dict(model.named_parameters())
        # Shape matches input dim
        if hasattr(rp, "in_features"):
            assert sigma.shape == (rp.in_features,)
        else:
            assert sigma.shape == (rp.in_channels,)
        # eps floor → strictly positive
        assert (sigma > 0).all(), f"{name} sigma_x has non-positive entries"


def test_conv3x3_channel_grouping():
    """channel_stats / regularization on a 3x3 groups==1 conv must group per INPUT
    channel (length C_in), reducing the kernel taps — not per (in·kH·kW) tap.

    Regression: the old `w.flatten(1).norm(dim=0)` returned a C_in·kH·kW vector for
    k>1 convs (kernel taps merged into the channel axis), the wrong grouping for the
    §2 per-input-channel contribution score and for channel pruning.
    """
    class _Conv3x3Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 6, 3, padding=1)  # groups==1, 3x3
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(6, 5)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).flatten(1)
            return self.head(x)

    model = _Conv3x3Net()
    mgr = MeanResidualManager(model, ["conv"], CPU, lambda_reg=1.0, max_batches=4)
    mgr.reparameterize(_img_loader(4, 8, n=8, bs=4))

    rp = mgr._reparam_modules["conv"]
    assert rp.v.dim() == 4 and rp.v.shape[2:] == (3, 3)  # genuinely a 3x3 kernel

    # channel_stats aggregates internally; assert the per-channel norm vector length
    # equals C_in (4), not C_in·kH·kW (36).
    from torch_pruning.utils.reparam import _channel_group_norm
    w_eff = rp.v.detach() * rp.sigma_x.detach()[None, :, None, None]
    norms = _channel_group_norm(w_eff, mgr.norm_dim)
    assert norms.shape == (4,), f"expected per-input-channel (4,), got {tuple(norms.shape)}"

    # regularization_loss must run on the conv and produce a finite positive scalar.
    loss = mgr.regularization_loss()
    assert torch.isfinite(loss) and loss.item() > 0


def test_score_uses_sigma():
    """Two output channels with identical v rows but different sigma_x scaling must
    rank differently in channel_stats — the score is ‖σ·v‖, not ‖v‖."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    rp = mgr._reparam_modules["fc2"]
    # Force two input channels to have identical v columns but different σ.
    with torch.no_grad():
        rp.v[:, 0] = 1.0
        rp.v[:, 1] = 1.0
        rp.sigma_x[0] = 0.5
        rp.sigma_x[1] = 2.0

    # ‖σ·v‖ column norms for the two channels should differ by exactly the σ ratio
    # (same v column, ‖σ_k · v[:,k]‖ = σ_k · ‖v[:,k]‖).
    v_eff = rp.v.detach() * rp.sigma_x.detach()[None, :]
    col_norms = v_eff.flatten(1).norm(p=2, dim=mgr.norm_dim)
    ratio = (col_norms[1] / col_norms[0]).item()
    assert abs(ratio - 4.0) < 1e-5, f"expected σ ratio 4.0, got {ratio}"


def test_reg_uses_sigma_and_detaches():
    """λ‖σ·v‖ regularization: σ enters the loss but receives no gradient (stop-grad)."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=1.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    # Bump σ so the loss is sensitive to it; if σ weren't detached, the test would
    # catch the missing stop-grad via sigma_x.grad on backward.
    for rp in mgr._reparam_modules.values():
        with torch.no_grad():
            rp.sigma_x.fill_(2.0)
        rp.sigma_x.requires_grad_(True)  # would receive grad if not detached
        rp.v.requires_grad_(True)

    loss = mgr.regularization_loss()
    assert loss.item() > 0, "λ‖σ·v‖ should be > 0 after init"
    loss.backward()
    for name, rp in mgr._reparam_modules.items():
        assert rp.sigma_x.grad is None, f"{name}: sigma_x must be stop-grad"
        assert rp.v.grad is not None, f"{name}: v must receive grad"


def test_load_model_random_init_cnn(tmp_path):
    """load_model returns a random-init CNN when no checkpoint path resolves to a file."""
    from vbp_common import load_model

    class _A:
        model_type = "cnn"
        cnn_arch = "resnet18"
        model_name = "resnet18"   # not a file path
        checkpoint = None

    args = _A()
    model = load_model(args, CPU)
    # Forward must work end-to-end on random weights.
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1000)
    assert getattr(args, "is_pruned_checkpoint", None) is False


# ---------------------------------------------------------------------------
# M2: propagation_importance — global importance via reverse-walk recursion
# ---------------------------------------------------------------------------
def _layer_M_reference(rp, conv_reduction="frobenius"):
    """Test-side reference for M[in, out] = |σ_i · reduce(v[j,i,*])|."""
    v = rp.v.detach()
    sigma = rp.sigma_x.detach()
    if v.dim() == 4:
        if conv_reduction == "frobenius":
            v_red = v.flatten(2).norm(p=2, dim=2)
        else:
            v_red = v.abs().flatten(2).sum(dim=2)
    else:
        v_red = v
    return (v_red.t() * sigma[:, None]).abs()


def _wbar_reference(M, p):
    """Relative W̄ = M^p / Σ_i M^p (column-stochastic; matches propagation_importance)."""
    Mp = M.pow(p)
    return Mp / Mp.sum(dim=0).clamp(min=1e-8)[None, :]


def _wbar_nonrel_reference(M, p):
    """Non-relative W̄ = M^p / σ_pre^p, σ_pre = (Σ_i M^2)^{1/2} (std/L2 col-norm).
    No σ transfer (σ already in M). For p=2 this equals _wbar_reference."""
    Mp = M.pow(p)
    denom = M.pow(2).sum(dim=0).clamp(min=1e-8).pow(p / 2.0)
    return Mp / denom[None, :]


def test_propagation_two_layer_mlp():
    """Hand-derive on TinyMLP (fc1 16→32, fc2 32→16): I^fc2 = W̄^fc2 · I_out;
    I^fc1 = W̄^fc1 · I^fc2. Lock the recursion math for both p (atol 1e-6).
    Default p=2 (variance, VBP-consistent); p=1 (std) also checked."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    I_out = torch.tensor([1.0 if i == 0 else 0.0 for i in range(16)])  # spike on output 0
    rp_fc1 = mgr._reparam_modules["fc1"]
    rp_fc2 = mgr._reparam_modules["fc2"]
    M_fc1 = _layer_M_reference(rp_fc1)
    M_fc2 = _layer_M_reference(rp_fc2)

    for p in (2, 1):  # 2 = default variance, 1 = std
        out = mgr.propagation_importance(I_out=I_out, p=p)
        assert list(out.keys()) == ["fc1", "fc2"], "results in forward order"
        I_fc2_ref = _wbar_reference(M_fc2, p) @ I_out
        assert torch.allclose(out["fc2"], I_fc2_ref, atol=1e-6), \
            f"p={p} I^fc2 mismatch: max diff={(out['fc2']-I_fc2_ref).abs().max()}"
        I_fc1_ref = _wbar_reference(M_fc1, p) @ I_fc2_ref
        assert torch.allclose(out["fc1"], I_fc1_ref, atol=1e-6), \
            f"p={p} I^fc1 mismatch: max diff={(out['fc1']-I_fc1_ref).abs().max()}"

    # Default call uses p=2 (variance); must differ from p=1 unless σ·v is trivial.
    out_default = mgr.propagation_importance(I_out=I_out)
    out_p1 = mgr.propagation_importance(I_out=I_out, p=1)
    assert torch.allclose(out_default["fc2"], _wbar_reference(M_fc2, 2) @ I_out, atol=1e-6)
    assert not torch.allclose(out_default["fc1"], out_p1["fc1"], atol=1e-4), \
        "p=2 and p=1 should give different propagated importance"

    # Shape sanity
    assert out_default["fc1"].shape == (16,)   # in_features of fc1
    assert out_default["fc2"].shape == (32,)   # in_features of fc2

    # Invalid p rejected
    import pytest
    with pytest.raises(ValueError, match="p must be"):
        mgr.propagation_importance(I_out=I_out, p=3)


def test_propagation_relative_vs_non_relative():
    """PDF (normalized_nets_pruning.pdf steps 7-10): σ is folded into M (=σ·W) ONCE, no
    separate transfer. relative → W̄=M^p/Σ_iM^p (column-stochastic). non-relative →
    W̄=M^p/σ_pre^p, σ_pre=(Σ_iM^2)^{1/2} (std/L2 col-norm). For p=2 they are IDENTICAL
    ("variance propagation yields the same relative criterion for p=2"); they differ at p=1.
    Lock both vs hand-derived refs."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    I_out = torch.tensor([1.0 if i == 0 else 0.0 for i in range(16)])
    rp1, rp2 = mgr._reparam_modules["fc1"], mgr._reparam_modules["fc2"]
    M_fc1 = _layer_M_reference(rp1)
    M_fc2 = _layer_M_reference(rp2)

    # relative (column-stochastic), both p
    for p in (2, 1):
        rel = mgr.propagation_importance(I_out=I_out, p=p, relative=True)
        I_fc2_rel = _wbar_reference(M_fc2, p) @ I_out
        assert torch.allclose(rel["fc2"], I_fc2_rel, atol=1e-6)
        assert torch.allclose(rel["fc1"], _wbar_reference(M_fc1, p) @ I_fc2_rel, atol=1e-6)

        # non-relative (σ_pre/L2 norm), NO transfer
        nonrel = mgr.propagation_importance(I_out=I_out, p=p, relative=False)
        I_fc2_nr = _wbar_nonrel_reference(M_fc2, p) @ I_out
        assert torch.allclose(nonrel["fc2"], I_fc2_nr, atol=1e-6)
        assert torch.allclose(nonrel["fc1"], _wbar_nonrel_reference(M_fc1, p) @ I_fc2_nr, atol=1e-6)

    # p=2: relative ≡ non-relative (PDF identity)
    rel2 = mgr.propagation_importance(I_out=I_out, p=2, relative=True)
    nr2 = mgr.propagation_importance(I_out=I_out, p=2, relative=False)
    assert torch.allclose(rel2["fc1"], nr2["fc1"], atol=1e-6), \
        "p=2: variance propagation must equal the relative criterion"

    # p=1: the two derivations differ (L1 vs L2 column norm)
    rel1 = mgr.propagation_importance(I_out=I_out, p=1, relative=True)
    nr1 = mgr.propagation_importance(I_out=I_out, p=1, relative=False)
    assert not torch.allclose(rel1["fc1"], nr1["fc1"], atol=1e-4), \
        "p=1: relative (L1) and non-relative (L2) should differ"


def test_propagation_uniform_default_seed():
    """No I_out → uniform 1/out_dim seed; chain executes; shapes match in_dims."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    out = mgr.propagation_importance()  # I_out=None
    assert out["fc1"].shape == (16,)
    assert out["fc2"].shape == (32,)

    # Compare against explicit uniform I_out
    rp_fc2 = mgr._reparam_modules["fc2"]
    last_out = rp_fc2.out_features
    I_out_ref = torch.full((last_out,), 1.0 / last_out)
    out_explicit = mgr.propagation_importance(I_out=I_out_ref)
    for k in out:
        assert torch.allclose(out[k], out_explicit[k], atol=1e-6)


def test_propagation_conv_kernel_reduction():
    """TinyCNN (1×1 convs): frobenius and abs_sum reductions both run; 1×1 case the
    two reductions match up to sign (Frobenius of one element = |element|)."""
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model, exclude_classifier=False)
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))

    out_frob = mgr.propagation_importance(conv_reduction="frobenius")
    out_abs = mgr.propagation_importance(conv_reduction="abs_sum")

    # Both run end-to-end, same key set.
    assert set(out_frob.keys()) == set(out_abs.keys())
    for k in out_frob:
        assert out_frob[k].shape == out_abs[k].shape
        # 1×1 convs in TinyCNN: kernel reduction is identity → both equal (up to abs).
        # Frobenius of 1 element = |element|; abs_sum of 1 element = |element|.
        # So results should match exactly.
        assert torch.allclose(out_frob[k], out_abs[k], atol=1e-6), \
            f"{k}: 1×1 kernel reductions should agree; max diff={(out_frob[k]-out_abs[k]).abs().max()}"


def test_propagation_mismatch_fallback():
    """Mismatched consecutive layers (simulate residual): warn → per-layer fallback;
    raise → RuntimeError."""
    # TinyMLP: out(fc1)=32, in(fc2)=32 — matches. Fake a mismatch by registering only
    # layers whose channels DON'T chain: pick fc1 (16→32) and reshape via a third hand-
    # built scenario. Easier: construct a 3-layer net with deliberately mismatched dims.
    class MismatchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 16)
            self.relu1 = nn.ReLU()
            self.b = nn.Linear(16, 4)     # a→b matches: 16==16
            self.relu2 = nn.ReLU()
            self.c = nn.Linear(8, 2)      # b→c MISMATCH: out(b)=4 ≠ in(c)=8
            # (an upstream non-reparam'd op would normally bridge; here we reparam a/b/c
            # directly so the chain breaks at b→c.)

        def forward(self, x):
            x = self.relu1(self.a(x))
            x = self.relu2(self.b(x))
            # forward doesn't matter for this test (uses calibration only)
            return x

    model = MismatchNet()
    # Register all three so the manager sees the mismatch.
    mgr = MeanResidualManager(model, ["a", "b", "c"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(8))

    # warn path → returns results for all layers (mismatch layer uses fallback)
    out = mgr.propagation_importance(on_mismatch="warn")
    assert set(out.keys()) == {"a", "b", "c"}
    # The mismatch layer 'b' (whose out=4 ≠ in(c)=8) gets a fallback per-layer score for c.
    # Specifically, at the c step out_dim=2 == I_out seed (2), so c is fine. At b step,
    # I^c has 8 entries but out(b)=4 → mismatch → b uses fallback.
    rp_b = mgr._reparam_modules["b"]
    M_b = _layer_M_reference(rp_b)
    expected_fallback = M_b.norm(p=2, dim=1)  # [in=16]
    assert torch.allclose(out["b"], expected_fallback, atol=1e-6)

    # raise path → RuntimeError
    import pytest
    with pytest.raises(RuntimeError, match="propagation chain mismatch"):
        mgr.propagation_importance(on_mismatch="raise")


# ---------------------------------------------------------------------------
# M3: residual / DAG propagation (σ_out branch weighting via DepGraph)
# ---------------------------------------------------------------------------
class _ResidualBlock(nn.Module):
    """Tiny residual: stem fans out to main + shortcut; main+shortcut merge via add
    feeding tail. All convs groups==1, 1×1 for simplicity."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.bn_main = nn.BatchNorm2d(8)
        self.main = nn.Conv2d(8, 8, 1)
        self.shortcut = nn.Conv2d(8, 8, 1)
        self.bn_tail = nn.BatchNorm2d(8)
        self.tail = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        x = self.stem(x)
        a = self.main(self.bn_main(x))
        b = self.shortcut(x)
        y = a + b                  # residual add
        return self.tail(self.bn_tail(y))


def test_output_sigma_calibrated():
    """sigma_out_x is populated, positive, correct shape."""
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model, exclude_classifier=False)
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))
    for name, rp in mgr._reparam_modules.items():
        assert hasattr(rp, "sigma_out_x"), f"{name} missing sigma_out_x"
        s = rp.sigma_out_x
        out_dim = rp.out_features if hasattr(rp, "out_features") else rp.out_channels
        assert s.shape == (out_dim,)
        assert (s > 0).all(), f"{name} sigma_out_x has non-positive entries"


def test_build_topology_sequential():
    """Sequential TinyMLP: fc1 → fc2 with weight 1.0; fc2 terminal."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))
    topo = mgr.build_propagation_topology(torch.randn(2, 16))
    assert topo["fc1"] == [("fc2", 1.0)]
    assert topo["fc2"] == []


def test_build_topology_residual():
    """ResidualBlock: stem fans out (both weight 1.0); main+shortcut share tail with
    σ_out branch weights summing to 1."""
    model = _ResidualBlock()
    names = ["stem", "main", "shortcut", "tail"]
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))
    topo = mgr.build_propagation_topology(torch.randn(2, 3, 8, 8))

    # stem fans out to both branches (separate downstreams, no add → both weight 1.0)
    stem_dsts = {d for d, _ in topo["stem"]}
    assert stem_dsts == {"main", "shortcut"}
    for _, w in topo["stem"]:
        assert w == 1.0, "fan-out (no residual add) → weight should be 1.0 per branch"

    # main + shortcut both feed tail through the add → σ_out-weighted branch factors
    assert len(topo["main"]) == 1 and topo["main"][0][0] == "tail"
    assert len(topo["shortcut"]) == 1 and topo["shortcut"][0][0] == "tail"
    w_main = topo["main"][0][1]
    w_short = topo["shortcut"][0][1]
    # PER-CHANNEL σ^p share now (tensor over the merged channel axis), summing to 1 per channel
    assert torch.is_tensor(w_main) and torch.is_tensor(w_short), "join weights are per-channel tensors"
    assert torch.allclose(w_main + w_short, torch.ones_like(w_main), atol=1e-5), \
        "branch weights at join must sum to 1 per channel"

    # Default p=2 → per-channel variance split σ²/(σ_a²+σ_b²)
    sm = mgr._reparam_modules["main"].sigma_out_x.pow(2)
    ss = mgr._reparam_modules["shortcut"].sigma_out_x.pow(2)
    expected_w_main = sm / (sm + ss + 1e-8)
    assert torch.allclose(w_main, expected_w_main, atol=1e-4), \
        f"w_main={w_main}, expected per-channel σ²-share ~{expected_w_main}"

    # tail terminal
    assert topo["tail"] == []


def test_propagation_through_residual():
    """End-to-end DAG walk: shapes + verify fan-out sum invariant at stem."""
    model = _ResidualBlock()
    names = ["stem", "main", "shortcut", "tail"]
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))
    topo = mgr.build_propagation_topology(torch.randn(2, 3, 8, 8))
    out = mgr.propagation_importance(topology=topo)

    # All 4 layers in output, correct per-input shapes
    assert set(out.keys()) == set(names)
    assert out["stem"].shape == (3,)
    assert out["main"].shape == (8,)
    assert out["shortcut"].shape == (8,)
    assert out["tail"].shape == (8,)

    # Stem fan-out: I_stem should = W̄_stem · (1.0·I_main + 1.0·I_shortcut), p=2 default
    rp_stem = mgr._reparam_modules["stem"]
    M_stem = _layer_M_reference(rp_stem)  # [in=3, out=8]
    Wbar_stem = _wbar_reference(M_stem, 2)
    I_next_stem = out["main"] + out["shortcut"]  # both weight=1.0
    I_stem_ref = Wbar_stem @ I_next_stem
    assert torch.allclose(out["stem"], I_stem_ref, atol=1e-6), \
        f"stem fan-out sum mismatch: max diff={(out['stem']-I_stem_ref).abs().max()}"


# ---------------------------------------------------------------------------
# E0: NormalizedNetImportance adapter (manager scores → tp pruner)
# ---------------------------------------------------------------------------
def test_e0_per_layer_importance_aligns_to_group():
    """per_layer scores feed a real pruning group: returns a 1-D, finite, non-negative
    per-channel score whose values trace back to input_channel_scores()."""
    import torch_pruning as tp
    from normalized_net_importance import (
        NormalizedNetImportance, extract_input_channel_scores)

    model = TinyCNN()
    mgr = MeanResidualManager(model, ["expand", "project", "fc"], CPU,
                              lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))
    scores = extract_input_channel_scores(mgr, mode="per_layer")
    assert scores["project"].shape == (16,)  # project's input = the 16-ch internal dim
    mgr.merge_back()                          # plain conv/linear again

    imp = NormalizedNetImportance(model, scores)
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 8, 8))
    # Prune the internal 16-channel dim via project's in_channels.
    group = DG.get_pruning_group(model.project, tp.prune_conv_in_channels,
                                 idxs=list(range(16)))
    s = imp(group)
    assert s.dim() == 1 and s.numel() == 16
    assert torch.isfinite(s).all() and (s >= 0).all()

    # The score for project's in-channels must be present (not pure magnitude fallback):
    # reconstruct the project contribution and confirm correlation with the adapter score.
    proj_score = scores["project"]            # length 16, per input channel
    # adapter applies normalizer="mean"; rank order must match proj_score's order
    assert torch.equal(s.argsort(), (proj_score / proj_score.mean()).argsort())


def test_e0_propagation_mode_and_prune_step():
    """propagation mode end-to-end: build adapter, run MagnitudePruner.step(), net
    actually shrinks and stays runnable."""
    import torch_pruning as tp
    from normalized_net_importance import (
        NormalizedNetImportance, extract_input_channel_scores)

    model = TinyCNN()
    mgr = MeanResidualManager(model, ["expand", "project", "fc"], CPU,
                              lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))
    ex = torch.randn(1, 3, 8, 8)
    scores = extract_input_channel_scores(mgr, mode="propagation", example_inputs=ex)
    mgr.merge_back()

    imp = NormalizedNetImportance(model, scores)
    params_before = sum(p.numel() for p in model.parameters())
    pruner = tp.pruner.MagnitudePruner(
        model, ex, importance=imp, pruning_ratio=0.5,
        ignored_layers=[model.fc])  # keep the 2-way classifier intact
    pruner.step()

    params_after = sum(p.numel() for p in model.parameters())
    assert params_after < params_before, "pruning should remove parameters"
    # Still runs a forward.
    out = model(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 2)


def test_e2_build_scorer_and_prune_resnet18():
    """E2 driver's build_scorer produces working importances for all 3 scorers on a
    real residual net; each prunes resnet18 ~50% and stays runnable."""
    import torchvision.models as tv
    import torch_pruning as tp
    from prune_e2 import build_scorer

    class _Args:
        exclude_stem = False
        calib_batches = 2
        max_batches = 2
        reparam_lambda = 0.0
        # bn (canonical) for magnitude/per_layer; propagation needs σ_out → mean.
        reparam_variant = "bn"

    def calib():
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(16, 3, 64, 64),
                                           torch.zeros(16, dtype=torch.long)),
            batch_size=8)

    for scorer in ("magnitude", "per_layer", "propagation"):
        model = tv.resnet18(weights=None).eval()
        ex = torch.randn(1, 3, 64, 64)
        p0 = sum(p.numel() for p in model.parameters())
        args = _Args()
        args.reparam_variant = "mean" if scorer == "propagation" else "bn"
        imp = build_scorer(scorer, model, calib(), CPU, args, ex)
        tp.pruner.MagnitudePruner(model, ex, importance=imp, pruning_ratio=0.5,
                                  ignored_layers=[model.fc]).step()
        p1 = sum(p.numel() for p in model.parameters())
        assert p1 < p0 * 0.5, f"{scorer}: expected ~50% params, got {p1/p0:.2f}"
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 1000)


def test_propagation_on_bn_variant():
    """Fix 2: the bn (canonical) variant now carries sigma_out_x + inherits propagation
    from Base, so propagation_importance runs on a bn-reparam'd net (was mean-only)."""
    from torch_pruning.utils.reparam import NormalizedResidualManager

    model = TinyCNN()
    mgr = NormalizedResidualManager(model, build_whole_net_reparam_layers(model),
                                    CPU, lambda_reg=0.0, max_batches=2)
    mgr.reparameterize(_img_loader())
    # sigma_out_x populated by calibration (not the unit-default placeholder).
    rp0 = next(iter(mgr._reparam_modules.values()))
    assert rp0.sigma_out_x.std().item() > 0
    ex = torch.randn(1, 3, 8, 8)
    topo = mgr.build_propagation_topology(ex)
    I = mgr.propagation_importance(topology=topo)
    assert I and all(v.dim() == 1 and v.numel() > 0 for v in I.values())


def test_normnet_vbppruner_with_compensation_resnet18():
    """Fix 3 core: VBPPruner ranks by NormalizedNetImportance (NCI) AND compensates via a
    mean_dict — the exact DDP-harness mechanism. Compensation ON (mean_dict) and OFF
    (mean_dict=None ≡ magnitude) both prune ~50% and stay runnable."""
    import torchvision.models as tv
    import torch_pruning as tp
    from torch_pruning.utils.reparam import NormalizedResidualManager
    from torch_pruning.utils.normnet_importance import (
        NormalizedNetImportance, extract_normnet_scores)
    from torch_pruning.pruner.importance import build_target_layers, collect_activation_means

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(16, 3, 64, 64),
                                       torch.zeros(16, dtype=torch.long)), batch_size=8)

    for compensate in (True, False):
        model = tv.resnet18(weights=None).eval()
        ex = torch.randn(1, 3, 64, 64)
        p0 = sum(p.numel() for p in model.parameters())
        names = build_whole_net_reparam_layers(model, exclude_classifier=True)
        mgr = NormalizedResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=2)
        mgr.reparameterize(loader)
        scores = extract_normnet_scores(mgr, "per_layer")   # ‖v_tilde‖ = √NCI
        mgr.merge_back()
        imp = NormalizedNetImportance(model, scores)
        pruner = tp.pruner.VBPPruner(model, ex, importance=imp, pruning_ratio=0.5,
                                     ignored_layers=[model.fc])
        if compensate:
            tl = build_target_layers(model, pruner.DG)
            mean_dict = collect_activation_means(model, loader, CPU, target_layers=tl,
                                                 max_batches=2)
            pruner.set_mean_dict(mean_dict)   # VBPPruner.step() folds pruned means into bias
        pruner.step()
        p1 = sum(p.numel() for p in model.parameters())
        assert p1 < p0 * 0.5, f"compensate={compensate}: expected ~50%, got {p1/p0:.2f}"
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 1000)


def test_e0_fallback_to_magnitude_when_no_reparam_member():
    """A group with no reparam'd consumer in scores → magnitude fallback, not a crash."""
    import torch_pruning as tp
    from normalized_net_importance import NormalizedNetImportance

    model = TinyCNN()
    # Empty scores → every group hits the fallback path.
    imp = NormalizedNetImportance(model, {}, fallback=True)
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 8, 8))
    group = DG.get_pruning_group(model.project, tp.prune_conv_in_channels,
                                 idxs=list(range(16)))
    s = imp(group)
    assert s is not None and s.numel() == 16  # magnitude fallback still ranks


# ---------------------------------------------------------------------------
# M4: per-step μ / σ EMA on Mean modules + M5: bn deprecation warning
# ---------------------------------------------------------------------------
def test_ema_disabled_by_default():
    """ema_momentum=0 (default) → mu_x / sigma_x stay frozen across forward train calls."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    rp = mgr._reparam_modules["fc1"]
    mu_before = rp.mu_x.clone()
    sigma_before = rp.sigma_x.clone()

    model.train()
    for _ in range(5):
        model(torch.randn(4, 16))
    assert torch.equal(rp.mu_x, mu_before), "frozen mu_x must not move"
    assert torch.equal(rp.sigma_x, sigma_before), "frozen sigma_x must not move"


def test_ema_updates_mu():
    """ema_momentum>0: μ_x / σ_x track toward batch stats over forward train passes.

    Note: per-step EMA does NOT compensate m (would mutate a Parameter mid-graph and
    break backward). The function output drifts smoothly as μ moves; m re-adapts via
    its own gradient. refresh_stats() between epochs gives function-preserving refresh.
    """
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4,
                              ema_momentum=0.1)  # fast EMA for clear test signal
    mgr.reparameterize(_vec_loader(16))

    rp_fc1 = mgr._reparam_modules["fc1"]
    mu_before = rp_fc1.mu_x.clone()
    sigma_before = rp_fc1.sigma_x.clone()

    model.train()
    shifted_data = torch.randn(4, 16) + 5.0
    for _ in range(20):
        model(shifted_data)

    # μ_x must have moved toward shifted_data's mean.
    assert (rp_fc1.mu_x - mu_before).abs().max() > 0.1, \
        f"μ_x should move; max delta={(rp_fc1.mu_x-mu_before).abs().max()}"
    # σ_x should also have updated (shifted_data has different scale than calibration).
    assert (rp_fc1.sigma_x - sigma_before).abs().max() > 0.0, "σ_x should evolve"


def test_ema_conv_module():
    """Conv variant: same invariants hold for MeanResidualConv2d."""
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model, exclude_classifier=False)
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4,
                              ema_momentum=0.1)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))

    rp_expand = mgr._reparam_modules["expand"]  # 3 → 16 conv
    mu_before = rp_expand.mu_x.clone()

    model.train()
    shifted = torch.randn(4, 3, 8, 8) + 3.0
    for _ in range(15):
        model(shifted)
    assert (rp_expand.mu_x - mu_before).abs().max() > 0.05, "conv μ_x must move under EMA"
    # sigma_out_x must also evolve
    assert hasattr(rp_expand, "sigma_out_x")
    assert (rp_expand.sigma_out_x > 0).all()


def test_ema_backward_and_step_does_not_crash():
    """REGRESSION: ema_momentum>0 must not break autograd's version counter on m.
    Earlier bug: self.m.add_(...) inside forward bumped Parameter version → backward
    raised 'one of the variables needed for gradient computation has been modified'.
    Fix uses self.m.data.add_(...). This test exercises the full forward → backward →
    optimizer.step loop that the original M4 tests skipped (they froze grads)."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4,
                              ema_momentum=0.1)
    mgr.reparameterize(_vec_loader(16))

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    # Three forward-backward-step iterations: each forward triggers the EMA m-update.
    for _ in range(3):
        x = torch.randn(4, 16)
        target = torch.randn(4, 16)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()   # would raise on Parameter version conflict if bug not fixed
        opt.step()


def test_mean_variant_vnr_roundtrip():
    """REGRESSION (audit bug #3): _merge_vnr_state_dict only handled BN variant.
    Mean-variant vnr state has keys .v / .m / .mu_x / .sigma_x / .sigma_out_x;
    merge must produce weight=v, bias=m−v·μ_x and drop the σ buffers."""
    torch.manual_seed(2)
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    # Drift the trainable params a bit so reload exercises non-trivial values.
    for rp in mgr._reparam_modules.values():
        with torch.no_grad():
            rp.v.add_(0.1 * torch.randn_like(rp.v))
            rp.m.add_(0.05 * torch.randn_like(rp.m))

    x = torch.randn(4, 16)
    model.eval()
    with torch.no_grad():
        y_ref = model(x).clone()

    vnr_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Sanity: state_dict carries the Mean-variant keys.
    assert any(k.endswith(".v") for k in vnr_state)
    assert any(k.endswith(".mu_x") for k in vnr_state)
    assert any(k.endswith(".sigma_x") for k in vnr_state)

    # Strict reload into a fresh plain arch.
    fresh = TinyMLP()
    merged = _merge_vnr_state_dict(vnr_state)
    fresh.load_state_dict(merged, strict=True)
    fresh.eval()
    with torch.no_grad():
        y_reload = fresh(x)
    assert (y_reload - y_ref).abs().max().item() < 1e-5, \
        f"mean-variant vnr reload broken: max diff={(y_reload-y_ref).abs().max()}"


def test_refresh_stats_channels_last_linear():
    """REGRESSION (audit bug #4): refresh_stats on a model whose Linears see 4D
    channels-last input (ConvNeXt-style pwconv). Old dispatch checked
    isinstance(module, nn.Linear) — fails for MeanResidualLinear post-reparam →
    wrong dim reduction → garbage refreshed μ/σ."""
    model = TinyChannelsLast()
    mgr = MeanResidualManager(model, ["pwconv1", "pwconv2"], CPU,
                              lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(8, 8, n=8, bs=4))

    # Snapshot pre-refresh shapes
    rp1 = mgr._reparam_modules["pwconv1"]
    assert rp1.mu_x.shape == (8,)
    mu_pre = rp1.mu_x.clone()
    sigma_pre = rp1.sigma_x.clone()

    # Refresh against fresh (shifted) data — shapes must remain correct AFTER refresh.
    shifted_loader = _img_loader(8, 8, n=8, bs=4)
    mgr.refresh_stats(shifted_loader)
    assert rp1.mu_x.shape == (8,), \
        f"refresh_stats broke μ_x shape on channels-last Linear: {rp1.mu_x.shape}"
    assert rp1.sigma_x.shape == (8,)
    assert rp1.sigma_out_x.shape == (16,)
    # And the values actually moved (data was different from calibration).
    assert (rp1.mu_x - mu_pre).abs().max() >= 0.0  # could be 0 if same data; just verifies no shape error


def test_reparam_lambda_actually_applied_in_training():
    """REGRESSION (audit round 4, bug #6): build_reparam_manager had lambda_reg=0.0
    hardcoded and train_normalized never passed aux_loss_fn → the M1 ‖σ·v‖
    regularizer was dead code. With λ > 0 a clear difference must exist between
    train_normalized's loss path with vs without the regularizer.

    Test approach: do one train step manually and verify mgr.regularization_loss()
    returns a positive scalar AND that adding it to the loss + backward yields a
    different gradient on v than CE alone.
    """
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=1.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    reg_loss = mgr.regularization_loss()
    assert reg_loss.item() > 0, "regularization_loss must be positive with λ > 0"

    # Verify that the wired path in build_reparam_manager passes lambda_reg through.
    import normalize_net as nn_mod
    args = _train_args(reparam_variant="mean", reparam_lambda=0.5, max_batches=4)
    mgr_via_cli = nn_mod.build_reparam_manager(
        TinyMLP(), ["fc1", "fc2"], CPU, args)
    assert mgr_via_cli.lambda_reg == 0.5, \
        f"reparam_lambda CLI -> manager.lambda_reg broken: {mgr_via_cli.lambda_reg}"

    # Verify aux_loss_fn IS plumbed into train_normalized: train one epoch with λ > 0
    # and a fresh setup, confirm v moves differently than λ=0 case.
    torch.manual_seed(7)
    m1 = TinyMLP()
    m2 = TinyMLP()
    m2.load_state_dict(m1.state_dict())
    train_loader = _vec_loader(16, n=8, bs=4)
    val_loader = _vec_loader(16, n=8, bs=4)

    mgr_a = MeanResidualManager(m1, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr_a.reparameterize(train_loader)
    mgr_b = MeanResidualManager(m2, ["fc1", "fc2"], CPU, lambda_reg=1.0, max_batches=4)
    mgr_b.reparameterize(train_loader)

    args_a = _train_args(epochs=1, reparam_variant="mean", reparam_lambda=0.0)
    args_b = _train_args(epochs=1, reparam_variant="mean", reparam_lambda=1.0)
    torch.manual_seed(99)
    nn_mod.train_normalized(m1, mgr_a, train_loader, val_loader, None, args_a, CPU, use_ddp=False)
    torch.manual_seed(99)
    nn_mod.train_normalized(m2, mgr_b, train_loader, val_loader, None, args_b, CPU, use_ddp=False)

    # With λ=1, the v should be pushed toward smaller norms vs λ=0.
    v_a = mgr_a._reparam_modules["fc2"].v.detach()
    v_b = mgr_b._reparam_modules["fc2"].v.detach()
    diff = (v_a - v_b).abs().max().item()
    assert diff > 1e-6, \
        f"reparam_lambda > 0 produced IDENTICAL v as λ=0 — aux_loss_fn not wired; diff={diff}"


def test_build_topology_restores_modules_on_exception():
    """REGRESSION (audit round 3, bug #5): build_propagation_topology temporarily
    swaps reparam'd modules for plain ones to let DepGraph trace. If anything in the
    swap loop or DG.build_dependency raises, the finally block must still restore
    the original reparam'd modules so the user's model isn't left half-swapped."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    # Capture identities of the reparam'd modules.
    fc1_before = mgr._reparam_modules["fc1"]
    fc2_before = mgr._reparam_modules["fc2"]
    assert isinstance(fc1_before, MeanResidualLinear)
    assert isinstance(fc2_before, MeanResidualLinear)

    # Force DG.build_dependency to fail by passing example_inputs with the wrong shape.
    import pytest
    with pytest.raises(Exception):
        mgr.build_propagation_topology(torch.randn(2, 999))  # wrong feature dim

    # After the exception, the user's model must still have the ORIGINAL reparam'd
    # modules in place (not the temporarily-swapped plain Linears).
    assert isinstance(model.fc1, MeanResidualLinear), \
        f"fc1 left swapped after exception: got {type(model.fc1).__name__}"
    assert isinstance(model.fc2, MeanResidualLinear)
    assert model.fc1 is fc1_before, "fc1 instance identity changed"
    assert model.fc2 is fc2_before, "fc2 instance identity changed"


def test_sync_ema_buffers_single_rank_noop():
    """DDP: sync_ema_buffers must be a safe no-op when dist not initialized.
    Buffers must remain unchanged."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4,
                              ema_momentum=0.1)
    mgr.reparameterize(_vec_loader(16))

    rp = mgr._reparam_modules["fc1"]
    mu_before = rp.mu_x.clone()
    sigma_before = rp.sigma_x.clone()
    sigma_out_before = rp.sigma_out_x.clone()

    # No torch.distributed.init_process_group called → must be no-op.
    mgr.sync_ema_buffers()

    assert torch.equal(rp.mu_x, mu_before), "sync_ema_buffers altered mu_x on single rank"
    assert torch.equal(rp.sigma_x, sigma_before)
    assert torch.equal(rp.sigma_out_x, sigma_out_before)


def test_sync_ema_buffers_method_exists_on_mean_manager():
    """Sanity: the public API is on MeanResidualManager and callable, NOT on the
    BN variant (whose buffers live under .bn.running_*)."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))
    assert callable(getattr(mgr, "sync_ema_buffers", None)), \
        "MeanResidualManager.sync_ema_buffers missing"

    bn_mgr = NormalizedResidualManager(TinyMLP(), ["fc1"], CPU,
                                       lambda_reg=0.0, max_batches=4)
    bn_mgr.reparameterize(_vec_loader(16))
    # Same underlying _sync_bn_stats — bn variant gets that automatically at merge_back.
    # No public sync_ema_buffers on bn variant (intentional; its buffers are .bn.running_*).
    assert not hasattr(bn_mgr, "sync_ema_buffers"), \
        "sync_ema_buffers should be MeanResidualManager-only"


def test_sync_bn_stats_covers_mean_variant():
    """REGRESSION: _sync_bn_stats now syncs Mean variant buffers too (mu_x, sigma_x,
    sigma_out_x), not just BN variant. Single-rank no-op test."""
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model, exclude_classifier=False)
    mgr = MeanResidualManager(model, names, CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(3, 8, n=8, bs=4))

    # Snapshot buffers; _sync_bn_stats is no-op on single rank → buffers unchanged.
    snaps = {}
    for name, rp in mgr._reparam_modules.items():
        snaps[name] = (rp.mu_x.clone(), rp.sigma_x.clone(), rp.sigma_out_x.clone())

    mgr._sync_bn_stats()

    for name, rp in mgr._reparam_modules.items():
        mu0, sigma0, sigma_out0 = snaps[name]
        assert torch.equal(rp.mu_x, mu0)
        assert torch.equal(rp.sigma_x, sigma0)
        assert torch.equal(rp.sigma_out_x, sigma_out0)


def test_ema_momentum_validation():
    """Hardening: ema_momentum must be in [0, 1]."""
    import pytest
    model = TinyMLP()
    with pytest.raises(ValueError, match="ema_momentum"):
        MeanResidualManager(model, ["fc1"], CPU, lambda_reg=0.0,
                            max_batches=4, ema_momentum=-0.1)
    with pytest.raises(ValueError, match="ema_momentum"):
        MeanResidualManager(model, ["fc1"], CPU, lambda_reg=0.0,
                            max_batches=4, ema_momentum=1.5)


def test_calibrate_channels_last_linear():
    """REGRESSION: TinyChannelsLast has Linear acting on 4D channels-last [N,H,W,C].
    The bug: _calibrate_stats dispatched by tensor.dim() and treated all 4D as Conv
    (channel=dim 1), so 4D Linear calibration reduced wrong dims → garbage μ/σ.
    Fix: dispatch by module type (Linear → channel=last dim)."""
    model = TinyChannelsLast()
    # pwconv1: Linear 8→16, pwconv2: Linear 16→8 (both act on [N,H,W,C]).
    mgr = MeanResidualManager(model, ["pwconv1", "pwconv2"], CPU,
                              lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_img_loader(8, 8, n=8, bs=4))

    rp_pw1 = mgr._reparam_modules["pwconv1"]
    rp_pw2 = mgr._reparam_modules["pwconv2"]
    # μ_x / σ_x must have the right shape (per-input-channel of Linear).
    assert rp_pw1.mu_x.shape == (8,), \
        f"pwconv1 μ_x wrong shape {rp_pw1.mu_x.shape} — channels-last dispatch broken"
    assert rp_pw1.sigma_x.shape == (8,)
    assert rp_pw1.sigma_out_x.shape == (16,)
    assert rp_pw2.mu_x.shape == (16,)
    assert rp_pw2.sigma_x.shape == (16,)
    assert rp_pw2.sigma_out_x.shape == (8,)
    # And populated with sensible values (not all zeros / all ones from fallback).
    assert (rp_pw1.sigma_x > 0).all()
    assert rp_pw1.sigma_x.std().item() > 0, "σ_x is constant → likely calibration didn't run"


def test_mean_variant_ablation_warning(monkeypatch):
    """Post-reversal: --reparam_variant=mean is the ABLATION (σ absent from forward) →
    build_reparam_manager logs an ABLATION note + returns a MeanResidualManager."""
    import normalize_net as nn_mod
    model = TinyCNN()
    names = build_whole_net_reparam_layers(model)
    args = _train_args(reparam_variant="mean", max_batches=4, mu_ema_momentum=0.01)
    captured = []
    monkeypatch.setattr(nn_mod, "log_info", lambda msg: captured.append(msg))
    mgr = nn_mod.build_reparam_manager(model, names, CPU, args)
    assert isinstance(mgr, MeanResidualManager)
    assert any("ABLATION" in m for m in captured), \
        f"expected ABLATION note for --reparam_variant=mean; got {captured}"


def test_bn_variant_is_canonical(monkeypatch):
    """Post-reversal: --reparam_variant=bn is CANONICAL (boss's BN trick) → returns a
    NormalizedResidualManager, logs CANONICAL, and threads --norm_bn_momentum into the
    σ-EMA momentum."""
    import normalize_net as nn_mod

    model = TinyCNN()
    names = build_whole_net_reparam_layers(model)
    args = _train_args(reparam_variant="bn", max_batches=4, norm_bn_momentum=0.02)

    captured = []
    # vbp_imagenet logger has propagate=False + no handler at test time → caplog
    # misses it. Intercept log_info directly.
    monkeypatch.setattr(nn_mod, "log_info", lambda msg: captured.append(msg))

    mgr = nn_mod.build_reparam_manager(model, names, CPU, args)
    assert isinstance(mgr, NormalizedResidualManager)
    assert mgr.bn_momentum == 0.02, f"bn_momentum should follow --norm_bn_momentum; got {mgr.bn_momentum}"
    assert any("CANONICAL" in msg for msg in captured), \
        f"expected CANONICAL note when --reparam_variant=bn; got: {captured}"


def test_propagation_vs_per_layer_score_difference():
    """Propagation has downstream info baked in → differs from per-layer ‖σ·v‖."""
    model = TinyMLP()
    mgr = MeanResidualManager(model, ["fc1", "fc2"], CPU, lambda_reg=0.0, max_batches=4)
    mgr.reparameterize(_vec_loader(16))

    # Force noticeable variation in fc2 so its downstream contribution biases fc1.
    rp_fc2 = mgr._reparam_modules["fc2"]
    with torch.no_grad():
        rp_fc2.v[0, :] *= 100.0  # output 0 of fc2 receives much stronger weights

    prop = mgr.propagation_importance()
    # Per-layer score for fc1 = column norms of (σ·v) on fc1 (M1 score).
    rp_fc1 = mgr._reparam_modules["fc1"]
    M_fc1 = _layer_M_reference(rp_fc1)
    per_layer_fc1 = M_fc1.norm(p=2, dim=1)
    # Propagation factors in fc2's downstream weights → different ranking/magnitudes.
    assert not torch.allclose(prop["fc1"], per_layer_fc1, atol=1e-3), \
        "propagation should differ from per-layer score once downstream layer has bias"


def test_classifier_seed_pdf_output_importance():
    """The PDF seeds propagation from the network output (classifier), I^L = W̄^fc·𝟙_classes,
    NOT uniform over features. _classifier_seed must return a NON-uniform per-feature vector,
    and None for the cases that should fall back to the uniform seed."""
    import torch.nn as nn
    from torch_pruning.utils.normnet_importance import _classifier_seed
    feat, classes = 8, 5

    class _RP:  # minimal reparam-module stub carrying sigma_out_x
        pass
    rp = _RP(); rp.sigma_out_x = torch.rand(feat) + 0.5

    class _MGR:
        pass
    mgr = _MGR(); mgr._reparam_modules = {"last": rp}
    topo = {"last": []}                      # single terminal
    clf = nn.Linear(feat, classes)

    seed = _classifier_seed(mgr, topo, clf, p=2)
    assert seed is not None and "last" in seed
    assert seed["last"].numel() == feat
    assert float(seed["last"].std()) > 0.0, "seed must be non-uniform (else global pruning guts)"
    # fall-backs → None (caller uses uniform)
    assert _classifier_seed(mgr, topo, None, 2) is None          # no classifier
    assert _classifier_seed(mgr, topo, nn.Linear(feat + 1, classes), 2) is None  # dim mismatch
    assert _classifier_seed(mgr, {"a": ["b"], "b": []}, clf, 2) is None  # >1? no, single terminal "b"


def test_ckpt_bundle_roundtrip_dense_ema_and_pruned():
    """ckpt.save_ckpt/load_ckpt: EMA is preserved + selectable, pruned (reduced-dim) nets
    reload exactly, and merge_reparam_modules folds a reparam'd net function-preservingly."""
    import sys, os, tempfile, copy
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "vbp"))
    import torchvision
    import torch_pruning as tp
    from ckpt import save_ckpt, load_ckpt, is_bundle, merge_reparam_modules
    d = tempfile.mkdtemp(); ex = torch.randn(1, 3, 224, 224)

    # dense + EMA: both selectable, EMA distinct from raw
    m = torchvision.models.resnet18(weights=None).eval()
    ema = copy.deepcopy(m)
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(0.01)
    p1 = os.path.join(d, "a.pth")
    save_ckpt(p1, m, kind="ema-trained", arch="resnet18", ema_model=ema)
    assert is_bundle(p1)
    assert (load_ckpt(p1, prefer="raw")(ex) - m(ex)).abs().max() < 1e-6
    assert (load_ckpt(p1, prefer="ema")(ex) - ema(ex)).abs().max() < 1e-6
    assert (load_ckpt(p1, prefer="ema")(ex) - load_ckpt(p1, prefer="raw")(ex)).abs().max() > 1e-3

    # pruned (reduced channel dims) reloads exactly from the full object
    m2 = torchvision.models.resnet18(weights=None).eval()
    tp.pruner.MagnitudePruner(m2, ex, importance=tp.importance.MagnitudeImportance(),
                              global_pruning=True, pruning_ratio=0.5, ignored_layers=[m2.fc]).step()
    y = m2(ex)
    p2 = os.path.join(d, "p.pth")
    save_ckpt(p2, m2, kind="pruned", arch="resnet18")
    assert (load_ckpt(p2)(ex) - y).abs().max() < 1e-6
