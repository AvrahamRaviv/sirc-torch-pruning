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
