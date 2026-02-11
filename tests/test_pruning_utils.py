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
        cp.current_epoch = 0
        cp.start_epoch = 0
        cp.end_epoch = 5
        cp.epoch_rate = 1
        cp.global_prune_rate = 0.3
        cp.current_pr = 0.0
        cp.current_step = None
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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
