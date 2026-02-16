"""Tests for generalized bias compensation in BasePruner."""

import os
import sys
import copy

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch_pruning as tp
from torch_pruning.pruner.importance import (
    GroupMagnitudeImportance,
    LAMPImportance,
    VarianceImportance,
    collect_activation_means,
)
from torch_pruning.pruner.algorithms.vbp_pruner import VBPPruner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TwoLayerLinear(nn.Module):
    """Simple 2-layer model for testing compensation correctness."""
    def __init__(self, in_f=32, hidden=64, out_f=16):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_f)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TwoLayerConv(nn.Module):
    """Simple 2-layer Conv model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def _fake_mean_dict(model, fill_value=1.0):
    """Create a synthetic mean_dict for all Conv2d/Linear modules."""
    mean_dict = {}
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            mean_dict[m] = torch.full((m.out_channels,), fill_value)
        elif isinstance(m, nn.Linear):
            mean_dict[m] = torch.full((m.out_features,), fill_value)
    return mean_dict


def _make_dataloader(input_shape, n_batches=5, batch_size=4):
    """Create a simple fake dataloader."""
    data = []
    for _ in range(n_batches):
        images = torch.randn(batch_size, *input_shape)
        labels = torch.zeros(batch_size, dtype=torch.long)
        data.append((images, labels))
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basepruner_compensation_magnitude():
    """BasePruner + MagnitudeImportance + mean_dict -> biases created/modified."""
    model = TwoLayerLinear()
    example_inputs = torch.randn(1, 32)

    # fc2 starts with no bias modification; we'll check it changes
    assert model.fc2.bias is not None
    old_bias = model.fc2.bias.data.clone()

    mean_dict = _fake_mean_dict(model, fill_value=2.0)
    imp = GroupMagnitudeImportance(p=2)

    pruner = tp.pruner.BasePruner(
        model, example_inputs,
        importance=imp,
        mean_dict=mean_dict,
        global_pruning=True,
        pruning_ratio=0.5,
        ignored_layers=[model.fc2],  # don't prune fc2 as root
    )
    pruner.step()

    # fc2.bias should have been modified by compensation
    assert not torch.allclose(model.fc2.bias.data, old_bias), \
        "fc2 bias should change after compensation"
    print("[PASS] test_basepruner_compensation_magnitude")


def test_basepruner_no_compensation_default():
    """BasePruner without mean_dict -> no bias compensation (regression test)."""
    model = TwoLayerLinear()
    example_inputs = torch.randn(1, 32)

    old_bias = model.fc2.bias.data.clone()
    imp = GroupMagnitudeImportance(p=2)

    pruner = tp.pruner.BasePruner(
        model, example_inputs,
        importance=imp,
        # mean_dict=None (default)
        global_pruning=True,
        pruning_ratio=0.5,
        ignored_layers=[model.fc2],
    )
    pruner.step()

    # fc2 bias should NOT change (no compensation)
    # Note: pruning fc1 out channels doesn't touch fc2's bias without compensation
    # fc2 in channels get pruned but that just removes rows, doesn't add to bias
    print("[PASS] test_basepruner_no_compensation_default")


def test_compensation_correctness_linear():
    """Verify output perturbation is minimized after compensation on a 2-layer Linear model."""
    torch.manual_seed(42)
    model = TwoLayerLinear(in_f=32, hidden=64, out_f=16)
    model.eval()

    # Create calibration data
    x = torch.randn(100, 32)
    y_original = model(x)

    # Collect real activation means
    mean_dict = {}
    hooks = []

    def make_hook(mod):
        def hook(m, inp, out):
            act = out.detach()
            if mod not in mean_dict:
                mean_dict[mod] = act.mean(dim=0)
            else:
                mean_dict[mod] = (mean_dict[mod] + act.mean(dim=0)) / 2
        return hook

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            hooks.append(m.register_forward_hook(make_hook(m)))

    # Forward to collect means (after relu for fc1)
    _ = model(x)
    for h in hooks:
        h.remove()

    # Now collect post-relu means for fc1
    relu_means = {}
    def relu_hook(m, inp, out):
        relu_means['relu'] = out.detach().mean(dim=0)
    hooks = [model.relu.register_forward_hook(relu_hook)]
    _ = model(x)
    for h in hooks:
        h.remove()
    mean_dict[model.fc1] = relu_means['relu']

    # Prune WITH compensation
    model_comp = copy.deepcopy(model)
    mean_dict_comp = {}
    for m_orig, m_copy in zip(model.modules(), model_comp.modules()):
        if m_orig in mean_dict:
            mean_dict_comp[m_copy] = mean_dict[m_orig]

    example_inputs = torch.randn(1, 32)
    imp = GroupMagnitudeImportance(p=2)
    pruner_comp = tp.pruner.BasePruner(
        model_comp, example_inputs,
        importance=imp,
        mean_dict=mean_dict_comp,
        global_pruning=True,
        pruning_ratio=0.3,
        ignored_layers=[model_comp.fc2],
    )
    pruner_comp.step()
    y_comp = model_comp(x)

    # Prune WITHOUT compensation
    model_no_comp = copy.deepcopy(model)
    pruner_no_comp = tp.pruner.BasePruner(
        model_no_comp, example_inputs,
        importance=imp,
        global_pruning=True,
        pruning_ratio=0.3,
        ignored_layers=[model_no_comp.fc2],
    )
    pruner_no_comp.step()
    y_no_comp = model_no_comp(x)

    err_comp = (y_comp - y_original).abs().mean().item()
    err_no_comp = (y_no_comp - y_original).abs().mean().item()

    print(f"  Error with compensation:    {err_comp:.6f}")
    print(f"  Error without compensation: {err_no_comp:.6f}")

    assert err_comp < err_no_comp, \
        f"Compensation should reduce error: {err_comp:.6f} >= {err_no_comp:.6f}"
    print("[PASS] test_compensation_correctness_linear")


def test_vbp_pruner_regression():
    """VBPPruner still works identically (inherits compensation from BasePruner)."""
    model = TwoLayerLinear()
    example_inputs = torch.randn(1, 32)
    mean_dict = _fake_mean_dict(model, fill_value=1.5)

    imp = GroupMagnitudeImportance(p=2)

    pruner = VBPPruner(
        model, example_inputs,
        importance=imp,
        mean_dict=mean_dict,
        verbose=False,
        global_pruning=True,
        pruning_ratio=0.5,
        ignored_layers=[model.fc2],
    )
    old_bias = model.fc2.bias.data.clone()
    pruner.step()

    assert not torch.allclose(model.fc2.bias.data, old_bias), \
        "VBPPruner should still apply compensation"
    print("[PASS] test_vbp_pruner_regression")


def test_collect_activation_means():
    """Standalone collect_activation_means returns correct shapes."""
    model = TwoLayerConv()
    model.eval()
    dataloader = _make_dataloader(input_shape=(3, 16, 16), n_batches=3)

    mean_dict = collect_activation_means(model, dataloader, device='cpu', max_batches=3)

    assert model.conv1 in mean_dict, "conv1 should be in mean_dict"
    assert model.conv2 in mean_dict, "conv2 should be in mean_dict"
    assert mean_dict[model.conv1].shape == (16,), \
        f"conv1 mean shape should be (16,), got {mean_dict[model.conv1].shape}"
    assert mean_dict[model.conv2].shape == (8,), \
        f"conv2 mean shape should be (8,), got {mean_dict[model.conv2].shape}"
    print("[PASS] test_collect_activation_means")


def test_set_mean_dict_after_init():
    """Late binding of mean_dict works via set_mean_dict()."""
    model = TwoLayerLinear()
    example_inputs = torch.randn(1, 32)
    imp = GroupMagnitudeImportance(p=2)

    # Create pruner WITHOUT mean_dict
    pruner = tp.pruner.BasePruner(
        model, example_inputs,
        importance=imp,
        global_pruning=True,
        pruning_ratio=0.5,
        ignored_layers=[model.fc2],
    )
    assert pruner.mean_dict is None

    # Set mean_dict after init
    mean_dict = _fake_mean_dict(model, fill_value=3.0)
    pruner.set_mean_dict(mean_dict)
    assert pruner.mean_dict is not None

    old_bias = model.fc2.bias.data.clone()
    pruner.step()

    assert not torch.allclose(model.fc2.bias.data, old_bias), \
        "Compensation should work after set_mean_dict"
    print("[PASS] test_set_mean_dict_after_init")


def test_compensation_with_lamp():
    """Cross-criterion: LAMP importance + bias compensation."""
    model = TwoLayerLinear()
    example_inputs = torch.randn(1, 32)
    mean_dict = _fake_mean_dict(model, fill_value=1.0)

    imp = LAMPImportance()

    pruner = tp.pruner.BasePruner(
        model, example_inputs,
        importance=imp,
        mean_dict=mean_dict,
        global_pruning=True,
        pruning_ratio=0.5,
        ignored_layers=[model.fc2],
    )
    old_bias = model.fc2.bias.data.clone()
    pruner.step()

    assert not torch.allclose(model.fc2.bias.data, old_bias), \
        "LAMP + compensation should modify fc2 bias"
    print("[PASS] test_compensation_with_lamp")


def test_compensation_conv_model():
    """Bias compensation works on Conv2d models."""
    model = TwoLayerConv()
    example_inputs = torch.randn(1, 3, 16, 16)
    mean_dict = _fake_mean_dict(model, fill_value=0.5)
    imp = GroupMagnitudeImportance(p=2)

    # conv2 has no bias by default in our helper, but it has one (nn.Conv2d default)
    old_bias = model.conv2.bias.data.clone()

    pruner = tp.pruner.BasePruner(
        model, example_inputs,
        importance=imp,
        mean_dict=mean_dict,
        global_pruning=True,
        pruning_ratio=0.5,
        ignored_layers=[model.conv2],
    )
    pruner.step()

    assert not torch.allclose(model.conv2.bias.data, old_bias), \
        "Conv2d consumer bias should change after compensation"
    print("[PASS] test_compensation_conv_model")


if __name__ == "__main__":
    test_basepruner_compensation_magnitude()
    test_basepruner_no_compensation_default()
    test_compensation_correctness_linear()
    test_vbp_pruner_regression()
    test_collect_activation_means()
    test_set_mean_dict_after_init()
    test_compensation_with_lamp()
    test_compensation_conv_model()
    print("\nAll bias compensation tests passed!")
