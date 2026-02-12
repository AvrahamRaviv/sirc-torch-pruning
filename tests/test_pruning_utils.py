"""
Unit tests for torch_pruning/utils/pruning_utils.py

Covers: _log helper, PruningMethod enum, build_inputs, ignored_layers
accumulation fix, SlicePruning.regularize on CPU.
"""
import sys
import os
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torch_pruning.utils.pruning_utils import (
    _log,
    PruningMethod,
    build_inputs,
)


# ---------------------------------------------------------------------------
# _log helper
# ---------------------------------------------------------------------------

class TestLog:
    def test_log_with_logger(self, caplog):
        """_log dispatches to logger.info when a logger is provided."""
        logger = logging.getLogger("test_pruning_utils")
        with caplog.at_level(logging.INFO, logger="test_pruning_utils"):
            _log(logger, "hello from logger")
        assert "hello from logger" in caplog.text

    def test_log_without_logger(self, capsys):
        """_log falls back to print when log is None."""
        _log(None, "hello from print")
        captured = capsys.readouterr()
        assert "hello from print" in captured.out


# ---------------------------------------------------------------------------
# PruningMethod enum
# ---------------------------------------------------------------------------

class TestPruningMethod:
    def test_values(self):
        assert PruningMethod.BN_SCALE == "BNScalePruner"
        assert PruningMethod.GROUP_NORM == "GroupNormPruner"
        assert PruningMethod.MAC_AWARE == "MACAwareImportance"
        assert PruningMethod.VBP == "VBP"

    def test_construct_from_string(self):
        """Config strings can be converted to enum members."""
        pm = PruningMethod("VBP")
        assert pm is PruningMethod.VBP

    def test_invalid_raises(self):
        """Unknown method string raises ValueError."""
        import pytest
        with pytest.raises(ValueError):
            PruningMethod("NonExistent")

    def test_json_roundtrip(self):
        """Enum value survives JSON serialization/deserialization."""
        d = {"method": PruningMethod.VBP}
        s = json.dumps(d)
        loaded = json.loads(s)
        assert PruningMethod(loaded["method"]) is PruningMethod.VBP

    def test_fstring_rendering(self):
        """f-string renders the raw value, not 'PruningMethod.VBP'."""
        msg = f"Algorithm: {PruningMethod.VBP}"
        assert msg == "Algorithm: VBP"


# ---------------------------------------------------------------------------
# build_inputs
# ---------------------------------------------------------------------------

class TestBuildInputs:
    def test_single_shape(self):
        t = build_inputs({"input_shape": [1, 3, 224, 224]}, device="cpu")
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 3, 224, 224)

    def test_multi_shape_list(self):
        result = build_inputs(
            {"input_shape": [[1, 3, 32, 32], [1, 1, 16, 16]]}, device="cpu"
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (1, 3, 32, 32)
        assert result[1].shape == (1, 1, 16, 16)

    def test_multi_shape_tuple_container(self):
        result = build_inputs(
            {"input_shape": [[1, 3, 32, 32], [1, 1, 16, 16]], "container": "tuple"},
            device="cpu",
        )
        assert isinstance(result, tuple)

    def test_missing_input_shape_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Config must contain"):
            build_inputs({}, device="cpu")

    def test_invalid_shape_raises(self):
        import pytest
        with pytest.raises(ValueError):
            build_inputs({"input_shape": "bad"}, device="cpu")


# ---------------------------------------------------------------------------
# SlicePruning.regularize on CPU (no hardcoded 'cuda')
# ---------------------------------------------------------------------------

class TestSlicePruningRegularizeCPU:
    def test_regularize_returns_zero_tensor_when_disabled(self):
        """SlicePruning.regularize() should return a zero tensor on CPU."""
        from torch_pruning.utils.pruning_utils import SlicePruning

        # Minimal model (SlicePruning.__init__ needs named_modules with Conv2d)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3))
        sp = SlicePruning(None, model)
        # prune_slices is False when config is None
        assert sp.prune_slices is False
        result = sp.regularize(model)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 0.0
        assert result.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# ChannelPruning.ignored_layers accumulation fix
# ---------------------------------------------------------------------------

class TestIgnoredLayersReset:
    def test_set_layers_to_prune_resets_ignored(self):
        """
        Calling set_layers_to_prune multiple times should not accumulate
        ignored_layers — each call starts fresh.
        """
        from torch_pruning.utils.pruning_utils import ChannelPruning

        # Build a minimal ChannelPruning by mocking the heavy __init__
        # We only need to exercise set_layers_to_prune()
        cp = ChannelPruning.__new__(ChannelPruning)
        cp.ignored_layers = []
        cp.layers_to_prune = []  # prune nothing
        cp.global_prune_rate = 0.3
        cp.channels_pruner_args = {"pruning_method": PruningMethod.GROUP_NORM}
        cp.pruning_method = PruningMethod.GROUP_NORM

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.Conv2d(16, 32, 3),
        )

        # Call twice
        cp.set_layers_to_prune(model)
        n_first = len(cp.ignored_layers)
        cp.set_layers_to_prune(model)
        n_second = len(cp.ignored_layers)

        # Without the fix, n_second would be 2 * n_first
        assert n_first == n_second
        assert n_first > 0  # sanity: at least some layers should be ignored


# ---------------------------------------------------------------------------
# _estimate_channel_ratio
# ---------------------------------------------------------------------------

class TestEstimateChannelRatio:
    """Test analytical MAC estimation via _estimate_channel_ratio()."""

    @staticmethod
    def _make_channel_pruning(model, example_inputs, ignored_layers, mac_target,
                              layers_to_prune, max_pruning_rate=0.95):
        """Create a minimal ChannelPruning for testing _estimate_channel_ratio."""
        from torch_pruning.utils.pruning_utils import ChannelPruning
        cp = ChannelPruning.__new__(ChannelPruning)
        cp.example_inputs = example_inputs
        cp.ignored_layers = ignored_layers
        cp.mac_target = mac_target
        cp.channels_pruner_args = {"max_pruning_rate": max_pruning_rate}
        cp.log = None
        cp.layers_to_prune = layers_to_prune
        cp.global_prune_rate = 0.3  # placeholder
        cp.pruning_method = PruningMethod.GROUP_NORM
        return cp

    def test_vit_linear_1dim(self):
        """ViT-like MLP: fc1+fc2 are all 1-dim, ratio = 1 - mac_target."""
        import torch.nn as nn

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 32)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleMLP()
        example_inputs = torch.randn(1, 32)

        # Ignore fc2 → only fc1 is root; both fc1/fc2 are 1-dim
        ignored = [model.fc2]
        layers = ["fc1"]

        cp = self._make_channel_pruning(model, example_inputs, ignored, 0.65, layers)
        ratio = cp._estimate_channel_ratio(model)

        # For pure 1-dim: p = 1 - mac_target
        assert abs(ratio - 0.35) < 0.01, f"Expected ~0.35, got {ratio}"

    def test_convnet_quadratic_2dim(self):
        """Conv chain: middle conv is 2-dim, verifying quadratic formula."""
        import torch.nn as nn

        class SimpleConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 10, 3, padding=1)
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                return self.conv3(x)

        model = SimpleConvNet()
        example_inputs = torch.randn(1, 3, 8, 8)

        # Ignore conv3 → conv2 is 2-dim (in + out), conv1 and conv3 are 1-dim
        ignored = [model.conv3]
        layers = ["conv1", "conv2"]

        cp = self._make_channel_pruning(model, example_inputs, ignored, 0.7, layers)
        ratio = cp._estimate_channel_ratio(model)

        # Should be > 0 and < 1
        assert 0 < ratio < 1, f"ratio out of range: {ratio}"
        # With 2-dim layers, the ratio should be less than 1 - mac_target
        # because quadratic scaling means less pruning needed for same MAC reduction
        assert ratio < 0.3, f"With 2-dim layers, expected ratio < 0.3, got {ratio}"

    def test_mac_target_one_returns_zero(self):
        """mac_target=1.0 means keep all MACs → prune ratio should be ~0."""
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        example_inputs = torch.randn(1, 32)

        cp = self._make_channel_pruning(model, example_inputs, [], 1.0, ["0", "2"])
        ratio = cp._estimate_channel_ratio(model)
        assert ratio < 0.01, f"Expected ~0 for mac_target=1.0, got {ratio}"

    def test_no_prunable_layers_returns_zero(self):
        """All layers ignored → ratio should be 0."""
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        example_inputs = torch.randn(1, 32)
        # Ignore everything
        ignored = [model[0], model[2]]
        cp = self._make_channel_pruning(model, example_inputs, ignored, 0.5, [])
        ratio = cp._estimate_channel_ratio(model)
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# Iterative steps via Pruning class (integration)
# ---------------------------------------------------------------------------

class TestIterativeSteps:
    """Test that the Pruning class correctly handles iterative_steps."""

    @staticmethod
    def _create_pruning_config(config_dir, layers, global_prune_rate=0.3,
                               start_epoch=0, end_epoch=4, epoch_rate=2):
        """Write a minimal pruning_config.json and return the path."""
        config = {
            "channel_sparsity_args": {
                "is_prune": True,
                "pruning_method": "BNScalePruner",
                "global_pruning": True,
                "block_size": 1,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "epoch_rate": epoch_rate,
                "global_prune_rate": global_prune_rate,
                "mac_target": 0.0,
                "max_pruning_rate": 0.95,
                "prune_channels_at_init": False,
                "infer": False,
                "input_shape": [1, 3, 8, 8],
                "layers": layers,
                "regularize": {"reg": 0, "mac_reg": 0},
                "MAC_params": {},
                "verbose": 0,
            },
            "slice_sparsity_args": None,
        }
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "pruning_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return config_dir

    def test_iterative_steps_count(self, tmp_path):
        """Verify iterative_steps matches expected count from epoch schedule."""
        import torch.nn as nn
        from torch_pruning.utils.pruning_utils import Pruning

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        layers = ["0", "3"]
        config_dir = self._create_pruning_config(
            str(tmp_path / "config"), layers,
            start_epoch=0, end_epoch=4, epoch_rate=2)

        pruning = Pruning(model, config_dir, device=torch.device("cpu"))
        cp = pruning.channel_pruner

        # (4 - 0) // 2 + 1 = 3 steps → epochs 0, 2, 4
        assert cp.iterative_steps == 3
        assert cp.pruner.current_step == 0

    def test_step_increments_on_prune_epochs(self, tmp_path):
        """Calling prune() on pruning epochs increments the step counter."""
        import torch.nn as nn
        from torch_pruning.utils.pruning_utils import Pruning

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        layers = ["0", "3"]
        config_dir = self._create_pruning_config(
            str(tmp_path / "config"), layers,
            global_prune_rate=0.3, start_epoch=0, end_epoch=2, epoch_rate=1)

        pruning = Pruning(model, config_dir, device=torch.device("cpu"))
        cp = pruning.channel_pruner

        # iterative_steps = (2-0)//1 + 1 = 3
        assert cp.iterative_steps == 3
        assert cp.prune_channels is True

        # Epoch 0: pruning epoch
        pruning.prune(model, epoch=0, mask_only=True)
        assert cp.pruner.current_step == 1

        # Epoch 1: pruning epoch
        pruning.prune(model, epoch=1, mask_only=True)
        assert cp.pruner.current_step == 2

        # Epoch 2: pruning epoch (last step)
        pruning.prune(model, epoch=2, mask_only=True)
        assert cp.pruner.current_step == 3
        assert cp.prune_channels is False  # all steps done

    def test_non_prune_epoch_skipped(self, tmp_path):
        """Calling prune() on non-pruning epochs does not increment step."""
        import torch.nn as nn
        from torch_pruning.utils.pruning_utils import Pruning

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        layers = ["0", "3"]
        config_dir = self._create_pruning_config(
            str(tmp_path / "config"), layers,
            global_prune_rate=0.2, start_epoch=0, end_epoch=4, epoch_rate=2)

        pruning = Pruning(model, config_dir, device=torch.device("cpu"))
        cp = pruning.channel_pruner

        # Epoch 0: pruning epoch
        pruning.prune(model, epoch=0, mask_only=True)
        assert cp.pruner.current_step == 1

        # Epoch 1: NOT a pruning epoch (1 % 2 != 0)
        pruning.prune(model, epoch=1, mask_only=True)
        assert cp.pruner.current_step == 1  # unchanged


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
