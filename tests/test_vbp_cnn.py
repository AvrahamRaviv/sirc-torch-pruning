"""
Unit tests for VBP CNN support.

Covers: build_cnn_target_layers, build_cnn_ignored_layers,
post_act_fn composition, smoke tests for ResNet-50 and MobileNetV2.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning.pruner.importance import (
    VarianceImportance,
    build_cnn_target_layers,
    build_cnn_ignored_layers,
)

import torchvision.models as tv_models
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resnet50_model():
    model = tv_models.resnet50(weights=None)
    model.eval()
    return model


@pytest.fixture(scope="module")
def mobilenetv2_model():
    model = tv_models.mobilenet_v2(weights=None)
    model.eval()
    return model


@pytest.fixture(scope="module")
def resnet50_dg(resnet50_model):
    example = torch.randn(1, 3, 224, 224)
    DG = tp.DependencyGraph().build_dependency(resnet50_model, example_inputs=example)
    return DG


@pytest.fixture(scope="module")
def mobilenetv2_dg(mobilenetv2_model):
    example = torch.randn(1, 3, 224, 224)
    DG = tp.DependencyGraph().build_dependency(mobilenetv2_model, example_inputs=example)
    return DG


# ---------------------------------------------------------------------------
# build_cnn_target_layers
# ---------------------------------------------------------------------------

class TestBuildTargetLayersResNet50:
    def test_detects_conv_layers(self, resnet50_model, resnet50_dg):
        """Auto-detects Conv2d layers with BN+ReLU composition."""
        target_layers = build_cnn_target_layers(resnet50_model, resnet50_dg)
        assert len(target_layers) > 0

        # All entries are (Conv2d, callable_or_None)
        for conv, post_act in target_layers:
            assert isinstance(conv, nn.Conv2d)

    def test_skips_depthwise(self, resnet50_model, resnet50_dg):
        """No depthwise convs in ResNet-50, but verify none sneak in."""
        target_layers = build_cnn_target_layers(resnet50_model, resnet50_dg)
        for conv, _ in target_layers:
            assert not (conv.groups == conv.out_channels and conv.out_channels > 1)

    def test_post_act_fn_not_none(self, resnet50_model, resnet50_dg):
        """ResNet-50 convs should have BN+ReLU detected."""
        target_layers = build_cnn_target_layers(resnet50_model, resnet50_dg)
        # Most convs in ResNet have BN+ReLU; at least some should have post_act
        has_post_act = sum(1 for _, pa in target_layers if pa is not None)
        assert has_post_act > 0, "Expected some layers with detected post_act_fn"


class TestBuildTargetLayersMobileNetV2:
    def test_detects_expand_convs(self, mobilenetv2_model, mobilenetv2_dg):
        """Detects expand convs, skips DW convs."""
        target_layers = build_cnn_target_layers(mobilenetv2_model, mobilenetv2_dg)
        assert len(target_layers) > 0

        # Verify no depthwise convs included
        for conv, _ in target_layers:
            assert not (conv.groups == conv.out_channels and conv.out_channels > 1), \
                f"DW conv should be excluded: groups={conv.groups}, out_ch={conv.out_channels}"


# ---------------------------------------------------------------------------
# post_act_fn correctness
# ---------------------------------------------------------------------------

class TestPostActFnCorrectness:
    def test_composed_fn_matches_manual(self, resnet50_model, resnet50_dg):
        """Composed post_act_fn matches manual BN+ReLU for interior convs."""
        # Use conv1 of layer1[0] — this has a direct BN+ReLU chain
        conv = resnet50_model.layer1[0].conv1
        bn = resnet50_model.layer1[0].bn1

        node = resnet50_dg.module2node[conv]
        from torch_pruning.pruner.importance import _compose_post_act
        post_act = _compose_post_act(node)

        assert post_act is not None, "conv1 of Bottleneck should have post_act_fn"

        x = torch.randn(2, conv.out_channels, 7, 7)
        with torch.no_grad():
            manual_out = torch.relu(bn(x))
            composed_out = post_act(x)

        assert torch.allclose(manual_out, composed_out, atol=1e-6), \
            "Composed post_act_fn doesn't match manual BN+ReLU"

    def test_conv3_gets_bn_only(self, resnet50_model, resnet50_dg):
        """conv3 in Bottleneck has BN -> add -> relu; post_act should be BN only."""
        conv3 = resnet50_model.layer1[0].conv3
        bn3 = resnet50_model.layer1[0].bn3

        node = resnet50_dg.module2node[conv3]
        from torch_pruning.pruner.importance import _compose_post_act
        post_act = _compose_post_act(node)

        # post_act should be the BN module itself (no relu detected after add)
        assert post_act is bn3, "conv3 post_act should be just BN (relu is after add)"


# ---------------------------------------------------------------------------
# build_cnn_ignored_layers
# ---------------------------------------------------------------------------

class TestIgnoredLayersResNet50Interior:
    def test_conv3_in_ignored(self, resnet50_model):
        """Bottleneck conv3 (output to residual) should be ignored."""
        from torchvision.models.resnet import Bottleneck
        ignored = build_cnn_ignored_layers(resnet50_model, "resnet50", interior_only=True)

        for m in resnet50_model.modules():
            if isinstance(m, Bottleneck):
                assert m.conv3 in ignored, "conv3 should be in ignored_layers"

    def test_downsample_in_ignored(self, resnet50_model):
        """Downsample convs should be ignored."""
        from torchvision.models.resnet import Bottleneck
        ignored = build_cnn_ignored_layers(resnet50_model, "resnet50", interior_only=True)

        for m in resnet50_model.modules():
            if isinstance(m, Bottleneck) and m.downsample is not None:
                for sub in m.downsample.modules():
                    if isinstance(sub, nn.Conv2d):
                        assert sub in ignored, "downsample conv should be in ignored_layers"

    def test_conv1_conv2_not_ignored(self, resnet50_model):
        """Bottleneck conv1/conv2 (interior) should NOT be ignored."""
        from torchvision.models.resnet import Bottleneck
        ignored = build_cnn_ignored_layers(resnet50_model, "resnet50", interior_only=True)

        for m in resnet50_model.modules():
            if isinstance(m, Bottleneck):
                assert m.conv1 not in ignored, "conv1 should NOT be in ignored_layers"
                assert m.conv2 not in ignored, "conv2 should NOT be in ignored_layers"

    def test_stem_and_classifier_ignored(self, resnet50_model):
        """Stem conv and classifier should always be ignored."""
        ignored = build_cnn_ignored_layers(resnet50_model, "resnet50", interior_only=True)
        assert resnet50_model.conv1 in ignored, "stem conv should be ignored"
        assert resnet50_model.fc in ignored, "classifier should be ignored"


class TestIgnoredLayersResNet50All:
    def test_only_stem_classifier_downsample(self, resnet50_model):
        """With interior_only=False, only stem/classifier/downsample are ignored."""
        ignored = build_cnn_ignored_layers(resnet50_model, "resnet50", interior_only=False)
        assert resnet50_model.conv1 in ignored
        assert resnet50_model.fc in ignored

        from torchvision.models.resnet import Bottleneck
        for m in resnet50_model.modules():
            if isinstance(m, Bottleneck):
                # conv1/conv2/conv3 should NOT be ignored (except via downsample)
                assert m.conv1 not in ignored
                assert m.conv2 not in ignored
                assert m.conv3 not in ignored


class TestIgnoredLayersMobileNetV2:
    def test_projection_ignored_dw_not(self, mobilenetv2_model):
        """Projection convs should be ignored; DW convs must NOT be (they
        participate in expand groups and would cause group rejection)."""
        ignored = build_cnn_ignored_layers(mobilenetv2_model, "mobilenet_v2", interior_only=True)

        try:
            from torchvision.models.mobilenetv2 import InvertedResidual
        except ImportError:
            from torchvision.models.mobilenet import InvertedResidual

        for m in mobilenetv2_model.modules():
            if isinstance(m, InvertedResidual):
                convs = [sub for sub in m.conv.modules() if isinstance(sub, nn.Conv2d)]
                for conv in convs:
                    if conv.groups == conv.out_channels and conv.out_channels > 1:
                        assert conv not in ignored, "DW conv must NOT be ignored"
                    elif conv.kernel_size == (1, 1) and conv == convs[-1]:
                        assert conv in ignored, "Projection conv should be ignored"


# ---------------------------------------------------------------------------
# Smoke tests: full VBP pipeline
# ---------------------------------------------------------------------------

class TestVBPPruneResNet50Smoke:
    def test_stats_prune_forward(self, resnet50_model):
        """Full pipeline: stats -> prune -> forward, no NaN."""
        model = tv_models.resnet50(weights=None)
        model.eval()
        example = torch.randn(1, 3, 224, 224)

        # Build DG and target layers
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
        target_layers = build_cnn_target_layers(model, DG)

        # Collect stats with small fake data
        imp = VarianceImportance()
        fake_data = [(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
                     for _ in range(3)]
        imp.collect_statistics(model, fake_data, "cpu", target_layers=target_layers, max_batches=3)

        assert len(imp.variance) > 0, "Should have collected stats"
        assert len(imp.means) > 0, "Should have collected means"

        # Create pruner
        ignored = build_cnn_ignored_layers(model, "resnet50", interior_only=True)
        pruner = tp.pruner.VBPPruner(
            model, example,
            importance=imp,
            global_pruning=True,
            pruning_ratio=0.3,
            ignored_layers=ignored,
            output_transform=lambda out: out.sum(),
            mean_dict=imp.means,
            verbose=False,
        )

        # Prune
        pruner.step(interactive=False, enable_compensation=True)

        # Forward pass should work
        model.eval()
        with torch.no_grad():
            out = model(example)
        assert not torch.isnan(out).any(), "Output contains NaN after pruning"
        assert out.shape[1] == 1000, "Output should have 1000 classes"


class TestVBPPruneMobileNetV2Smoke:
    def test_stats_prune_forward(self, mobilenetv2_model):
        """Full pipeline for MobileNetV2."""
        model = tv_models.mobilenet_v2(weights=None)
        model.eval()
        example = torch.randn(1, 3, 224, 224)

        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
        target_layers = build_cnn_target_layers(model, DG)

        imp = VarianceImportance()
        fake_data = [(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
                     for _ in range(3)]
        imp.collect_statistics(model, fake_data, "cpu", target_layers=target_layers, max_batches=3)

        assert len(imp.variance) > 0

        ignored = build_cnn_ignored_layers(model, "mobilenet_v2", interior_only=True)
        pruner = tp.pruner.VBPPruner(
            model, example,
            importance=imp,
            global_pruning=True,
            pruning_ratio=0.3,
            ignored_layers=ignored,
            output_transform=lambda out: out.sum(),
            mean_dict=imp.means,
            verbose=False,
        )

        pruner.step(interactive=False, enable_compensation=True)

        model.eval()
        with torch.no_grad():
            out = model(example)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert out.shape[1] == 1000


# ---------------------------------------------------------------------------
# Compensation adds bias
# ---------------------------------------------------------------------------

class TestCompensationAddsBias:
    def test_consumer_gets_bias(self):
        """After pruning with compensation, consumer conv should have non-None bias."""
        model = tv_models.resnet50(weights=None)
        model.eval()
        example = torch.randn(1, 3, 224, 224)

        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example)
        target_layers = build_cnn_target_layers(model, DG)

        imp = VarianceImportance()
        fake_data = [(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
                     for _ in range(3)]
        imp.collect_statistics(model, fake_data, "cpu", target_layers=target_layers, max_batches=3)

        ignored = build_cnn_ignored_layers(model, "resnet50", interior_only=True)

        # Count convs without bias before pruning
        convs_without_bias_before = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Conv2d) and m not in ignored and m.bias is None
        )

        pruner = tp.pruner.VBPPruner(
            model, example,
            importance=imp,
            global_pruning=True,
            pruning_ratio=0.3,
            ignored_layers=ignored,
            output_transform=lambda out: out.sum(),
            mean_dict=imp.means,
            verbose=False,
        )

        pruner.step(interactive=False, enable_compensation=True)

        # At least some convs should now have bias (compensation adds it)
        convs_without_bias_after = sum(
            1 for m in model.modules()
            if isinstance(m, nn.Conv2d) and m not in ignored and m.bias is None
        )
        assert convs_without_bias_after < convs_without_bias_before, \
            "Compensation should have added bias to at least some consumer convs"
