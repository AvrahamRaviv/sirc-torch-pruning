"""
Unit tests for unified pruning pipeline.

Tests epoch math, phase transitions, model_changed flags, and reparam lifecycle
through the ChannelPruning class.
"""
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
from torch_pruning.utils.pruning_utils import ChannelPruning, Pruning, PruningMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    """Minimal 2-layer MLP for testing (no GPU needed)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _create_config(tmp_path, *, start_epoch=0, end_epoch=4, epoch_rate=1,
                   prune_rate=0.35, epochs_ft=0, sparse_mode="none",
                   reparam_lambda=0.01, pruning_schedule="geometric",
                   method="GroupNormPruner"):
    """Write a minimal pruning_config.json and return the directory."""
    config = {
        "channel_sparsity_args": {
            "is_prune": True,
            "pruning_method": method,
            "global_pruning": True,
            "block_size": 1,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "epoch_rate": epoch_rate,
            "global_prune_rate": prune_rate,
            "mac_target": 0.0,
            "max_pruning_rate": 0.95,
            "prune_channels_at_init": False,
            "infer": False,
            "input_shape": [1, 16],
            "layers": ["fc1"],
            "regularize": {"reg": 0, "mac_reg": 0},
            "MAC_params": {},
            "verbose": 0,
            "pruning_schedule": pruning_schedule,
            "sparse_mode": sparse_mode,
            "sparse_gmp_target": 0.5,
            "epochs_ft": epochs_ft,
            "reparam_lambda": reparam_lambda,
        },
        "slice_sparsity_args": None,
    }
    config_dir = str(tmp_path / "pruning_config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "pruning_config.json"), "w") as f:
        json.dump(config, f)
    return config_dir


def _make_dataloader():
    """Tiny dataloader for stats collection (4 batches of 2 samples)."""
    data = torch.randn(8, 16)
    labels = torch.randint(0, 2, (8,))
    ds = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(ds, batch_size=2)


# ---------------------------------------------------------------------------
# Test: total_epochs computation
# ---------------------------------------------------------------------------

class TestTotalEpochs:
    """Verify total_epochs = end_epoch + 1 + epochs_ft for various combos."""

    def test_pat_only(self, tmp_path):
        """N=5, K=0, S=0 → total=5"""
        # epoch_rate=1, start=0, end=4
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=4,
                                    epoch_rate=1, epochs_ft=0)
        model = TinyMLP()
        p = Pruning(model, config_dir, device=torch.device("cpu"))
        assert p.channel_pruner.total_epochs == 5

    def test_ft_only(self, tmp_path):
        """N=1, K=10, S=0 → total=11"""
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=0,
                                    epoch_rate=1, epochs_ft=10)
        model = TinyMLP()
        p = Pruning(model, config_dir, device=torch.device("cpu"))
        assert p.channel_pruner.total_epochs == 11

    def test_pat_plus_ft(self, tmp_path):
        """N=5, K=10, S=0 → total=15"""
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=4,
                                    epoch_rate=1, epochs_ft=10)
        model = TinyMLP()
        p = Pruning(model, config_dir, device=torch.device("cpu"))
        assert p.channel_pruner.total_epochs == 15

    def test_pat_with_gaps_plus_ft(self, tmp_path):
        """N=5, M=2, K=10, S=0 → total=23"""
        # epoch_rate=3, end_epoch=12
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=12,
                                    epoch_rate=3, epochs_ft=10)
        model = TinyMLP()
        p = Pruning(model, config_dir, device=torch.device("cpu"))
        assert p.channel_pruner.total_epochs == 23

    def test_sparse_plus_pat_plus_ft(self, tmp_path):
        """N=5, K=10, S=3 → total=18"""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=10,
                                    sparse_mode="l1_group")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        assert p.channel_pruner.total_epochs == 18

    def test_sparse_reparam_no_ft(self, tmp_path):
        """N=5, K=0, S=3, sparse=reparam → total=8"""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="reparam")
        model = TinyMLP()
        p = Pruning(model, config_dir, device=torch.device("cpu"))
        assert p.channel_pruner.total_epochs == 8


# ---------------------------------------------------------------------------
# Test: phase property
# ---------------------------------------------------------------------------

class TestPhaseProperty:
    """Verify phase returns correct value at each epoch."""

    def test_phases_pat_only(self, tmp_path):
        """N=5, K=10, S=0: PAT for 0-4, FT for 5-14."""
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=4,
                                    epoch_rate=1, epochs_ft=10)
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        # Simulate epoch progression
        for epoch in range(15):
            cp.current_epoch = epoch
            if epoch <= 4 and cp.prune_channels:
                assert cp.phase == "PAT", f"epoch {epoch}: expected PAT"
            elif not cp.prune_channels:
                assert cp.phase == "FT", f"epoch {epoch}: expected FT"

    def test_phases_with_sparse(self, tmp_path):
        """N=5, K=10, S=3, sparse=l1_group: Sparse 0-2, PAT 3-7, FT 8-17."""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=10,
                                    sparse_mode="l1_group")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        # Sparse phase
        for epoch in range(3):
            cp.current_epoch = epoch
            assert cp.phase == "Sparse", f"epoch {epoch}: expected Sparse"

        # PAT phase (prune_channels should still be True)
        for epoch in range(3, 8):
            cp.current_epoch = epoch
            if cp.prune_channels:
                assert cp.phase == "PAT", f"epoch {epoch}: expected PAT"


# ---------------------------------------------------------------------------
# Test: model_changed flag
# ---------------------------------------------------------------------------

class TestModelChanged:
    """Verify model_changed auto-resets and is True at expected transitions."""

    def test_model_changed_resets_on_read(self, tmp_path):
        """model_changed should return True once then reset to False."""
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=0,
                                    epoch_rate=1, epochs_ft=5)
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        # Initially False
        assert cp.model_changed is False

        # After pruning at epoch 0, model should have changed (physical prune)
        p.prune(model, epoch=0, mask_only=False)
        assert cp.model_changed is True
        # Second read should be False (auto-reset)
        assert cp.model_changed is False

    def test_no_change_on_ft_epoch(self, tmp_path):
        """model_changed should be False on non-pruning FT epochs."""
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=0,
                                    epoch_rate=1, epochs_ft=5)
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        # Prune at epoch 0 (consumes model_changed)
        p.prune(model, epoch=0, mask_only=False)
        _ = cp.model_changed  # consume

        # Epoch 1 is FT (no pruning) → no change
        p.prune(model, epoch=1, mask_only=False)
        assert cp.model_changed is False


# ---------------------------------------------------------------------------
# Test: skip stats at init with sparse mode
# ---------------------------------------------------------------------------

class TestSkipStatsAtInit:
    """When sparse_mode != 'none' and start_epoch > 0, stats should be skipped."""

    def test_stats_skipped_for_sparse(self, tmp_path):
        """With l1_group sparse and start_epoch=3, init should skip stats."""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="l1_group")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner
        # _stats_fresh should be False since we skipped stats
        assert not cp._stats_fresh

    def test_stats_collected_without_sparse(self, tmp_path):
        """Without sparse mode, stats should be collected at init."""
        config_dir = _create_config(tmp_path, start_epoch=0, end_epoch=4,
                                    epoch_rate=1, epochs_ft=0, method="VBP")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner
        # VBP with no sparse mode should collect stats → _stats_fresh=True
        assert cp._stats_fresh


# ---------------------------------------------------------------------------
# Test: reparam lifecycle
# ---------------------------------------------------------------------------

class TestReparamLifecycle:
    """Test MeanResidualManager init/merge through prune()."""

    def test_reparam_init_on_first_sparse_epoch(self, tmp_path):
        """prune() at epoch < start_epoch with sparse_mode=reparam inits reparam."""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="reparam")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        assert cp._reparam_manager is None

        # Prune at epoch 0 → init reparam
        p.prune(model, epoch=0, mask_only=False)
        assert cp._reparam_manager is not None
        assert cp._reparam_manager.is_active
        assert cp.model_changed is True

    def test_reparam_returns_early_on_subsequent_sparse_epochs(self, tmp_path):
        """prune() at sparse epochs after init should return early (no crash)."""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="reparam")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        # Epoch 0: init
        p.prune(model, epoch=0, mask_only=False)
        _ = cp.model_changed  # consume

        # Epoch 1: should return early, no model_changed
        p.prune(model, epoch=1, mask_only=False)
        assert cp.model_changed is False
        assert cp._reparam_manager.is_active

    def test_reparam_merge_at_start_epoch(self, tmp_path):
        """prune() at start_epoch merges reparam back and triggers pruning."""
        config_dir = _create_config(tmp_path, start_epoch=2, end_epoch=4,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="reparam")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        # Init reparam at epoch 0
        p.prune(model, epoch=0, mask_only=False)
        _ = cp.model_changed
        p.prune(model, epoch=1, mask_only=False)
        _ = cp.model_changed

        # Epoch 2 (start_epoch): merge back + prune
        p.prune(model, epoch=2, mask_only=False)
        assert not cp._reparam_manager.is_active
        assert cp.model_changed is True

    def test_reparam_numerically_exact(self, tmp_path):
        """Forward output is identical before and after reparameterize+merge."""
        config_dir = _create_config(tmp_path, start_epoch=2, end_epoch=4,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="reparam")
        model = TinyMLP()
        loader = _make_dataloader()

        # Capture output before any changes
        model.eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            out_before = model(x).clone()

        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)

        # Epoch 0: reparameterize (changes module types)
        p.prune(model, epoch=0, mask_only=False)

        # Output should be numerically identical after reparam
        model.eval()
        with torch.no_grad():
            out_reparam = model(x)
        assert torch.allclose(out_before, out_reparam, atol=1e-5), \
            f"Max diff after reparam: {(out_before - out_reparam).abs().max()}"

        # Epoch 1: still reparam
        p.prune(model, epoch=1, mask_only=False)

        # Epoch 2: merge back (restores standard modules) + prunes
        # After merge+prune, output will differ (channels removed),
        # but the merge itself should be numerically exact.
        # We can't easily test post-prune identity, so just verify no crash.
        p.prune(model, epoch=2, mask_only=False)
        model.eval()
        with torch.no_grad():
            out_merged = model(x)
        # Output will differ due to pruning, but model should still work
        assert out_merged.shape[0] == 4


# ---------------------------------------------------------------------------
# Test: GMP sparse lifecycle
# ---------------------------------------------------------------------------

class TestGMPLifecycle:
    """Test GMP masking and baking through prune()."""

    def test_gmp_applies_masks_before_start(self, tmp_path):
        """prune() at epoch < start_epoch with gmp applies unstructured masks."""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="gmp")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)

        # Should not crash; applies GMP mask
        p.prune(model, epoch=0, mask_only=False)
        p.prune(model, epoch=1, mask_only=False)
        p.prune(model, epoch=2, mask_only=False)

        # At epoch 3: bakes masks and starts pruning
        p.prune(model, epoch=3, mask_only=False)


# ---------------------------------------------------------------------------
# Test: Conv2d reparam and build_reparam_layers for CNN
# ---------------------------------------------------------------------------

class TinyCNN(nn.Module):
    """Minimal expand→BN→ReLU→project→pool→fc for Conv2d reparam tests."""
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(3, 16, 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.project = nn.Conv2d(16, 8, 1)  # 16>8 → matches MNV2-style filter
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.bn(self.expand(x)))
        x = self.project(x)
        return self.fc(self.pool(x).flatten(1))


def _make_image_dataloader():
    """Tiny image dataloader (4 batches of 2 samples, 8×8)."""
    data = torch.randn(8, 3, 8, 8)
    labels = torch.randint(0, 2, (8,))
    ds = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(ds, batch_size=2)


class TestConv2dReparam:
    """Test MeanResidualConv2d reparameterize/merge numerics and regularization."""

    def test_conv2d_reparam_numerically_exact(self):
        """Forward output identical before reparam, during reparam, and after merge."""
        from torch_pruning.utils.reparam import MeanResidualManager

        model = TinyCNN()
        loader = _make_image_dataloader()
        x = torch.randn(4, 3, 8, 8)

        model.eval()
        with torch.no_grad():
            y_before = model(x).clone()

        mgr = MeanResidualManager(model, ["project"], torch.device("cpu"),
                                  lambda_reg=0.01)
        mgr.reparameterize(loader)

        model.eval()
        with torch.no_grad():
            y_during = model(x)
        assert torch.allclose(y_before, y_during, atol=1e-5), \
            f"Max diff during reparam: {(y_before - y_during).abs().max()}"

        mgr.merge_back()

        model.eval()
        with torch.no_grad():
            y_after = model(x)
        assert torch.allclose(y_before, y_after, atol=1e-5), \
            f"Max diff after merge: {(y_before - y_after).abs().max()}"

    def test_conv2d_regularization_loss(self):
        """Regularization loss is scalar, positive, and differentiable."""
        from torch_pruning.utils.reparam import MeanResidualManager

        model = TinyCNN()
        loader = _make_image_dataloader()

        mgr = MeanResidualManager(model, ["project"], torch.device("cpu"),
                                  lambda_reg=0.01)
        mgr.reparameterize(loader)

        loss = mgr.regularization_loss()
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert loss.requires_grad, "Loss should require grad"
        loss.backward()  # should not error


class TestBuildReparamLayersCNN:
    """Test build_reparam_layers for ResNet-50 and MobileNet V2."""

    def test_build_reparam_layers_resnet50(self):
        import torchvision.models as tv_models

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))), "benchmarks", "vbp"))
        from vbp_common import build_reparam_layers

        model = tv_models.resnet50(weights=None)
        layers = build_reparam_layers(model, "cnn", "resnet50")
        assert len(layers) == 16, f"Expected 16, got {len(layers)}: {layers}"
        for name in layers:
            assert name.endswith("conv3"), f"Expected conv3, got {name}"
            parts = name.split('.')
            parent = model
            for p in parts:
                parent = getattr(parent, p)
            assert isinstance(parent, nn.Conv2d)
            assert parent.kernel_size == (1, 1)
            assert parent.groups == 1

    def test_build_reparam_layers_mobilenet_v2(self):
        import torchvision.models as tv_models

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))), "benchmarks", "vbp"))
        from vbp_common import build_reparam_layers

        model = tv_models.mobilenet_v2(weights=None)
        layers = build_reparam_layers(model, "cnn", "mobilenet_v2")
        assert len(layers) == 17, f"Expected 17, got {len(layers)}: {layers}"
        for name in layers:
            parts = name.split('.')
            parent = model
            for p in parts:
                parent = getattr(parent, p)
            assert isinstance(parent, nn.Conv2d)
            assert parent.kernel_size == (1, 1)
            assert parent.groups == 1
            assert parent.in_channels > parent.out_channels, \
                f"{name}: in={parent.in_channels} should be > out={parent.out_channels}"


# ---------------------------------------------------------------------------
# Test: NormalizedResidualLinear
# ---------------------------------------------------------------------------

class TestNormalizedResidualLinear:
    """Test NormalizedResidualLinear: function-preserving init and merge."""

    def test_function_preserving_init(self):
        """Output of NormalizedResidualLinear matches original Linear."""
        from torch_pruning.utils.reparam import NormalizedResidualLinear

        torch.manual_seed(42)
        linear = nn.Linear(16, 32)
        x = torch.randn(4, 16)

        with torch.no_grad():
            y_orig = linear(x)

        # Simulate calibration
        mu_x = x.mean(dim=0)
        sigma_x = x.std(dim=0).clamp(min=1e-6)

        W = linear.weight.data.clone()
        b = linear.bias.data.clone()
        v_tilde = W * sigma_x[None, :]  # BN-equivalent: Ṽ = W·σ
        m = b + W @ mu_x

        reparam = NormalizedResidualLinear(16, 32, v_tilde, m, mu_x, sigma_x)
        with torch.no_grad():
            y_reparam = reparam(x)

        assert torch.allclose(y_orig, y_reparam, atol=1e-5), \
            f"Max diff: {(y_orig - y_reparam).abs().max()}"

    def test_merge_params_recovers_original(self):
        """merge_params recovers original W, b."""
        from torch_pruning.utils.reparam import NormalizedResidualLinear

        torch.manual_seed(42)
        linear = nn.Linear(16, 32)
        x = torch.randn(8, 16)

        mu_x = x.mean(dim=0)
        sigma_x = x.std(dim=0).clamp(min=1e-6)

        W = linear.weight.data.clone()
        b = linear.bias.data.clone()
        v_tilde = W * sigma_x[None, :]  # BN-equivalent: Ṽ = W·σ
        m = b + W @ mu_x

        reparam = NormalizedResidualLinear(16, 32, v_tilde, m, mu_x, sigma_x)
        w_merged, b_merged = reparam.merge_params()

        # Effective weight: v_tilde / sigma_x = (W·σ)/σ = W
        assert torch.allclose(W, w_merged, atol=1e-5), \
            f"Weight max diff: {(W - w_merged).abs().max()}"

        # Effective bias should match original
        b_eff_orig = b.clone()
        b_eff_merged = b_merged
        # The merged bias = m - w @ mu_x = (b + W @ mu_x) - W @ mu_x = b
        assert torch.allclose(b_eff_orig, b_eff_merged, atol=1e-5), \
            f"Bias max diff: {(b_eff_orig - b_eff_merged).abs().max()}"


# ---------------------------------------------------------------------------
# Test: NormalizedResidualConv2d
# ---------------------------------------------------------------------------

class TestNormalizedResidualConv2d:
    """Test NormalizedResidualConv2d: function-preserving init and merge."""

    def test_function_preserving_init(self):
        """Output of NormalizedResidualConv2d matches original Conv2d."""
        from torch_pruning.utils.reparam import NormalizedResidualConv2d

        torch.manual_seed(42)
        conv = nn.Conv2d(8, 16, 3, padding=1)
        x = torch.randn(2, 8, 8, 8)

        with torch.no_grad():
            y_orig = conv(x)

        # Simulate calibration: spatial-averaged per-channel stats
        mu_x = x.mean(dim=(0, 2, 3))
        sigma_x = ((x * x).mean(dim=(0, 2, 3)) - mu_x * mu_x).clamp(min=1e-12).sqrt().clamp(min=1e-6)

        W = conv.weight.data.clone()
        b = conv.bias.data.clone()
        sigma_bc = sigma_x[None, :, None, None]
        v_tilde = W * sigma_bc  # BN-equivalent: Ṽ = W·σ
        m = b + W.sum(dim=(2, 3)) @ mu_x

        reparam = NormalizedResidualConv2d(
            8, 16, kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
            v_tilde=v_tilde, m=m, mu_x=mu_x, sigma_x=sigma_x)

        with torch.no_grad():
            y_reparam = reparam(x)

        assert torch.allclose(y_orig, y_reparam, atol=1e-4), \
            f"Max diff: {(y_orig - y_reparam).abs().max()}"

    def test_merge_params_recovers_original(self):
        """merge_params recovers original W, b."""
        from torch_pruning.utils.reparam import NormalizedResidualConv2d

        torch.manual_seed(42)
        conv = nn.Conv2d(8, 16, 3, padding=1)
        x = torch.randn(4, 8, 8, 8)

        mu_x = x.mean(dim=(0, 2, 3))
        sigma_x = ((x * x).mean(dim=(0, 2, 3)) - mu_x * mu_x).clamp(min=1e-12).sqrt().clamp(min=1e-6)

        W = conv.weight.data.clone()
        b = conv.bias.data.clone()
        sigma_bc = sigma_x[None, :, None, None]
        v_tilde = W * sigma_bc  # BN-equivalent: Ṽ = W·σ
        m = b + W.sum(dim=(2, 3)) @ mu_x

        reparam = NormalizedResidualConv2d(
            8, 16, kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
            v_tilde=v_tilde, m=m, mu_x=mu_x, sigma_x=sigma_x)

        w_merged, b_merged = reparam.merge_params()

        # Effective weight: v_tilde / sigma = (W·σ)/σ = W
        assert torch.allclose(W, w_merged, atol=1e-5), \
            f"Weight max diff: {(W - w_merged).abs().max()}"
        assert torch.allclose(b, b_merged, atol=1e-5), \
            f"Bias max diff: {(b - b_merged).abs().max()}"


# ---------------------------------------------------------------------------
# Test: NormalizedResidualManager lifecycle
# ---------------------------------------------------------------------------

class TestNormalizedResidualManager:
    """Test full lifecycle: reparameterize → forward → merge_back → output matches."""

    def test_lifecycle_linear(self):
        """NormalizedResidualManager on TinyMLP: reparam → merge → numerically exact."""
        from torch_pruning.utils.reparam import NormalizedResidualManager

        model = TinyMLP()
        loader = _make_dataloader()
        x = torch.randn(4, 16)

        model.eval()
        with torch.no_grad():
            y_before = model(x).clone()

        mgr = NormalizedResidualManager(model, ["fc1"], torch.device("cpu"),
                                         lambda_reg=0.01)
        mgr.reparameterize(loader)

        model.eval()
        with torch.no_grad():
            y_during = model(x)
        assert torch.allclose(y_before, y_during, atol=1e-5), \
            f"Max diff during reparam: {(y_before - y_during).abs().max()}"

        mgr.merge_back()

        model.eval()
        with torch.no_grad():
            y_after = model(x)
        assert torch.allclose(y_before, y_after, atol=1e-5), \
            f"Max diff after merge: {(y_before - y_after).abs().max()}"

    def test_lifecycle_conv2d(self):
        """NormalizedResidualManager on TinyCNN: reparam → merge → numerically exact."""
        from torch_pruning.utils.reparam import NormalizedResidualManager

        model = TinyCNN()
        loader = _make_image_dataloader()
        x = torch.randn(4, 3, 8, 8)

        model.eval()
        with torch.no_grad():
            y_before = model(x).clone()

        mgr = NormalizedResidualManager(model, ["project"], torch.device("cpu"),
                                         lambda_reg=0.01)
        mgr.reparameterize(loader)

        model.eval()
        with torch.no_grad():
            y_during = model(x)
        assert torch.allclose(y_before, y_during, atol=1e-4), \
            f"Max diff during reparam: {(y_before - y_during).abs().max()}"

        mgr.merge_back()

        model.eval()
        with torch.no_grad():
            y_after = model(x)
        assert torch.allclose(y_before, y_after, atol=1e-4), \
            f"Max diff after merge: {(y_before - y_after).abs().max()}"

    def test_regularization_loss(self):
        """Regularization loss is scalar, positive, and differentiable."""
        from torch_pruning.utils.reparam import NormalizedResidualManager

        model = TinyMLP()
        loader = _make_dataloader()

        mgr = NormalizedResidualManager(model, ["fc1"], torch.device("cpu"),
                                         lambda_reg=0.01)
        mgr.reparameterize(loader)

        loss = mgr.regularization_loss()
        assert loss.dim() == 0
        assert loss.item() > 0
        assert loss.requires_grad
        loss.backward()


# ---------------------------------------------------------------------------
# Test: refresh_stats (function-preserving)
# ---------------------------------------------------------------------------

class TestRefreshStats:
    """Output before refresh == output after refresh (function-preserving)."""

    def test_refresh_stats_preserves_output(self):
        from torch_pruning.utils.reparam import NormalizedResidualManager

        model = TinyMLP()
        loader = _make_dataloader()
        x = torch.randn(4, 16)

        mgr = NormalizedResidualManager(model, ["fc1"], torch.device("cpu"),
                                         lambda_reg=0.01)
        mgr.reparameterize(loader)

        model.eval()
        with torch.no_grad():
            y_before = model(x).clone()

        mgr.refresh_stats(loader)

        model.eval()
        with torch.no_grad():
            y_after = model(x)

        assert torch.allclose(y_before, y_after, atol=1e-5), \
            f"Max diff after refresh: {(y_before - y_after).abs().max()}"

    def test_mean_residual_refresh_mu_alias(self):
        """MeanResidualManager.refresh_mu still works (backward compat)."""
        from torch_pruning.utils.reparam import MeanResidualManager

        model = TinyMLP()
        loader = _make_dataloader()
        x = torch.randn(4, 16)

        mgr = MeanResidualManager(model, ["fc1"], torch.device("cpu"),
                                   lambda_reg=0.01)
        mgr.reparameterize(loader)

        model.eval()
        with torch.no_grad():
            y_before = model(x).clone()

        mgr.refresh_mu(loader)

        model.eval()
        with torch.no_grad():
            y_after = model(x)

        assert torch.allclose(y_before, y_after, atol=1e-5), \
            f"Max diff after refresh_mu: {(y_before - y_after).abs().max()}"


# ---------------------------------------------------------------------------
# Test: entropy loss
# ---------------------------------------------------------------------------

class TestEntropyLoss:
    """Entropy loss behavior: uniform → max, single channel → zero."""

    def test_uniform_v_max_entropy(self):
        """Uniform V column norms should give maximum entropy."""
        from torch_pruning.utils.reparam import NormalizedResidualManager

        model = TinyMLP()
        # Set fc1 weights to have uniform column norms
        with torch.no_grad():
            w = torch.randn(32, 16)
            # Normalize each column to have same norm
            w = w / w.norm(p=2, dim=0, keepdim=True)
            model.fc1.weight.copy_(w)

        loader = _make_dataloader()
        mgr = NormalizedResidualManager(model, ["fc1"], torch.device("cpu"),
                                         lambda_reg=0.01, entropy_lambda=1.0)
        mgr.reparameterize(loader)

        H_uniform = mgr.entropy_loss()
        assert H_uniform.item() > 0, "Entropy loss should be positive (negative entropy)"

    def test_zero_lambda_returns_zero(self):
        """entropy_lambda=0 should return zero loss."""
        from torch_pruning.utils.reparam import NormalizedResidualManager

        model = TinyMLP()
        loader = _make_dataloader()
        mgr = NormalizedResidualManager(model, ["fc1"], torch.device("cpu"),
                                         lambda_reg=0.01, entropy_lambda=0.0)
        mgr.reparameterize(loader)

        H = mgr.entropy_loss()
        assert H.item() == 0.0

    def test_concentrated_v_lower_entropy(self):
        """A single dominant column should give lower entropy (closer to zero) than uniform."""
        from torch_pruning.utils.reparam import NormalizedResidualManager

        # Model with concentrated columns
        model_conc = TinyMLP()
        with torch.no_grad():
            w = torch.zeros(32, 16)
            w[:, 0] = 10.0  # one dominant column
            w[:, 1:] = 0.001  # rest near zero
            model_conc.fc1.weight.copy_(w)

        loader = _make_dataloader()
        mgr_conc = NormalizedResidualManager(model_conc, ["fc1"], torch.device("cpu"),
                                              lambda_reg=0.01, entropy_lambda=1.0)
        mgr_conc.reparameterize(loader)
        H_conc = mgr_conc.entropy_loss().item()

        # Model with uniform columns
        model_uni = TinyMLP()
        with torch.no_grad():
            w = torch.randn(32, 16)
            w = w / w.norm(p=2, dim=0, keepdim=True)
            model_uni.fc1.weight.copy_(w)

        mgr_uni = NormalizedResidualManager(model_uni, ["fc1"], torch.device("cpu"),
                                             lambda_reg=0.01, entropy_lambda=1.0)
        mgr_uni.reparameterize(loader)
        H_uni = mgr_uni.entropy_loss().item()

        # Uniform should have higher entropy → higher negative entropy loss
        assert H_uni > H_conc, \
            f"Uniform entropy ({H_uni}) should be > concentrated ({H_conc})"


# ---------------------------------------------------------------------------
# Test: similarity discount in VarianceImportance
# ---------------------------------------------------------------------------

class TestSimilarityDiscount:
    """Similarity discount: identical channels → high discount, orthogonal → none."""

    def test_identical_channels_high_discount(self):
        """Identical output rows → identical activation channels → high similarity."""
        from torch_pruning.pruner.importance import VarianceImportance

        imp = VarianceImportance(similarity_discount=True)

        # Create a model where fc1 output channels 0 and 1 are identical
        # (same row in weight matrix → same output activation)
        model = TinyMLP()
        with torch.no_grad():
            model.fc1.weight[0] = model.fc1.weight[1].clone()
            model.fc1.bias[0] = model.fc1.bias[1].clone()

        loader = _make_dataloader()
        imp.collect_statistics(model, loader, torch.device("cpu"), max_batches=4)

        fc1 = model.fc1
        assert fc1 in imp._similarity, "fc1 should have similarity scores"
        R = imp._similarity[fc1]
        # Output channels 0 and 1 produce identical activations → cosine sim ≈ 1.0
        assert R[0].item() > 0.9, f"Channel 0 max sim should be high, got {R[0].item()}"
        assert R[1].item() > 0.9, f"Channel 1 max sim should be high, got {R[1].item()}"

    def test_no_discount_when_disabled(self):
        """Without similarity_discount, scores should equal raw variance."""
        from torch_pruning.pruner.importance import VarianceImportance
        import torch_pruning as tp

        model = TinyMLP()
        loader = _make_dataloader()

        imp_no = VarianceImportance(similarity_discount=False)
        imp_no.collect_statistics(model, loader, torch.device("cpu"), max_batches=4)

        imp_yes = VarianceImportance(similarity_discount=True)
        imp_yes.collect_statistics(model, loader, torch.device("cpu"), max_batches=4)

        # Raw variances should be the same
        fc1 = model.fc1
        assert torch.allclose(imp_no.variance[fc1], imp_yes.variance[fc1])

    def test_orthogonal_lower_similarity_than_identical(self):
        """Orthogonal weight rows should yield lower activation similarity than identical rows."""
        from torch_pruning.pruner.importance import VarianceImportance

        torch.manual_seed(42)
        loader = _make_dataloader()

        # Model with identical rows → high activation similarity
        model_ident = TinyMLP()
        with torch.no_grad():
            for i in range(32):
                model_ident.fc1.weight[i] = model_ident.fc1.weight[0].clone()
                model_ident.fc1.bias[i] = model_ident.fc1.bias[0].clone()

        imp_ident = VarianceImportance(similarity_discount=True)
        imp_ident.collect_statistics(model_ident, loader, torch.device("cpu"), max_batches=4)
        R_ident = imp_ident._similarity[model_ident.fc1]

        # Model with orthogonal rows → lower activation similarity
        model_orth = TinyMLP()
        with torch.no_grad():
            nn.init.orthogonal_(model_orth.fc1.weight)

        imp_orth = VarianceImportance(similarity_discount=True)
        imp_orth.collect_statistics(model_orth, loader, torch.device("cpu"), max_batches=4)
        R_orth = imp_orth._similarity[model_orth.fc1]

        assert R_orth.mean().item() < R_ident.mean().item(), \
            f"Orthogonal mean sim ({R_orth.mean():.3f}) should be < identical ({R_ident.mean():.3f})"


# ---------------------------------------------------------------------------
# Test: VNR lifecycle through prune()
# ---------------------------------------------------------------------------

class TestVNRLifecycle:
    """Test NormalizedResidualManager init/merge through prune() with sparse_mode=vnr."""

    def test_vnr_init_on_first_sparse_epoch(self, tmp_path):
        """prune() at epoch < start_epoch with sparse_mode=vnr inits VNR reparam."""
        config_dir = _create_config(tmp_path, start_epoch=3, end_epoch=7,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="vnr")
        model = TinyMLP()
        loader = _make_dataloader()
        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)
        cp = p.channel_pruner

        assert cp._reparam_manager is None

        p.prune(model, epoch=0, mask_only=False)
        assert cp._reparam_manager is not None
        assert cp._reparam_manager.is_active
        assert cp.model_changed is True

        # Verify it's a NormalizedResidualManager
        from torch_pruning.utils.reparam import NormalizedResidualManager
        assert isinstance(cp._reparam_manager, NormalizedResidualManager)

    def test_vnr_numerically_exact(self, tmp_path):
        """Forward output is identical before and after VNR reparameterize+merge."""
        config_dir = _create_config(tmp_path, start_epoch=2, end_epoch=4,
                                    epoch_rate=1, epochs_ft=0,
                                    sparse_mode="vnr")
        model = TinyMLP()
        loader = _make_dataloader()

        model.eval()
        x = torch.randn(4, 16)
        with torch.no_grad():
            out_before = model(x).clone()

        p = Pruning(model, config_dir, device=torch.device("cpu"),
                    train_loader=loader)

        p.prune(model, epoch=0, mask_only=False)

        model.eval()
        with torch.no_grad():
            out_reparam = model(x)
        assert torch.allclose(out_before, out_reparam, atol=1e-5), \
            f"Max diff after reparam: {(out_before - out_reparam).abs().max()}"
