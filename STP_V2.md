# SIRC Torch-Pruning v2 (STP v2) — Internal Technical Document

> **Audience:** SIRC team
> **Repo:** [`sirc-torch-pruning`](https://github.com/AvrahamRaviv/sirc-torch-pruning.git) on `master`
> **Upstream:** [Torch-Pruning v1.6.1](https://github.com/VainF/Torch-Pruning) by Gongfan Fang
> **Paper:** [DepGraph (2301.12900)](https://arxiv.org/abs/2301.12900), [VBP (2507.12988)](https://arxiv.org/abs/2507.12988)

---

## 1. High-Level Design

### 1.1 The Pruner

The **Pruner** (`BasePruner`) is the central class. Everything flows through it.

```
                                                   Importance Criteria
                                                   (pluggable)
                                                   ~~~~~~~~~~~~~~~~~~~
                                                   | Magnitude       |
                                                   | Taylor          |
     model                                         | Hessian         |
     example_inputs          pruning_ratio         | Variance (VBP)  |
     ignored_layers          global/local          | MACAware        |
         |                   iterative_steps       | LAMP, BNScale...|
         |                       |                  ~~~~~~~~~~~~~~~~~~~
         v                       v                          |
 +=====================================================+    |
 | Pruner (BasePruner / VBPPruner / ...)               |    |
 |                                                     |    |
 |  __init__(model, example_inputs, importance, ...)   |    |
 |     |                                               |    |
 |     +---> DG = DependencyGraph.build(model)         |    |
 |           (traces forward pass, builds dep graph)   |    |
 |                                                     |    |
 |  4 public methods:                                  |    |
 |  ~~~~~~~~~~~~~~~~~                                  |    |
 |                                                     |    |
 |  step()                                             |    |
 |     |                                               |    |
 |     +--> _prune():                                  |    |
 |           for group in DG.get_all_groups():         |    |
 |              |                                      |    |
 |              +--> importance(group) ----plugged in---+----+
 |              |         |
 |              |         v
 |              |    1-D scores per channel
 |              |         |
 |              +--> scope + threshold (local or global)
 |              |         |
 |              |         v
 |              |    indices to prune
 |              |         |
 |              +--> [subclass hook: VBP compensation]
 |              |         |
 |              +--> group.prune()
 |                   (structurally removes channels)
 |                                                     |
 |  estimate_importance(group) --> 1-D tensor          |
 |     (calls importance(group), used for analysis)    |
 |                                                     |
 |  regularize(model, loss) --> reg_loss               |
 |     (sparse training: L2,1 / GMP during training)   |
 |                                                     |
 |  manual_prune_width(layer, fn, ratio_or_idxs)      |
 |     (prune a specific layer directly)               |
 |                                                     |
 +=====================================================+
         |
         v
     Pruned model (fewer channels, smaller weights)
```

**Key abstractions inside the Pruner:**

- **DependencyGraph (DG)** — Traces the model once at init. Yields *groups*: sets of coupled layers that must be pruned together (e.g., Conv -> BN -> next Conv). See `torch_pruning/dependency/graph.py`.
- **Group** — A list of `(Dependency, indices)` pairs. `group.prune()` applies the structural cut atomically across all coupled layers.
- **Importance** — The pluggable criterion. Any callable `(group) -> 1-D tensor`. Swapped at init time. This is where Magnitude vs Taylor vs Variance etc. differ.
- **`_prune()` is a generator** — yields groups one at a time, so subclasses (e.g., `VBPPruner`) can intercept each group to apply corrections *before* `group.prune()`.

### 1.2 PAT (Pruning-Aware Training)

`ChannelPruning` wraps the Pruner and exposes two calls: `regularize()` and `prune()`. The IP's train loop is **unchanged** — just add these two calls, and the wrapper decides internally what to do based on current epoch and config (`start_epoch`, `end_epoch`, `epoch_rate`).

```
 IP's train loop (transparent — same loop for all epochs)
 =========================================================

 pruner = ChannelPruning(config, model, ...)      # init once

 for epoch in range(total_epochs):
     for batch in train_loader:
         loss = model(batch)
         loss.backward()
         pruner.regularize(model)                  # <-- call 1
         optimizer.step()

     pruner.prune(model, epoch)                    # <-- call 2

 =========================================================

 What happens inside (decided by ChannelPruning based on epoch + config):

 epoch < start_epoch          --> regularize: applies reg loss (L2,1 / GMP)
                                  prune: no-op

 epoch in pruning epochs      --> regularize: applies reg loss
 (start..end, every epoch_rate)   prune: collect stats, pruner.step(),
                                         structural channel removal

 epoch > end_epoch            --> regularize: no-op
                                  prune: no-op
                                  (model trains normally = fine-tuning)
```

**Geometric schedule:** each pruning step keeps `per_step_keep = keep_ratio^(1/N)` channels, so after N steps the cumulative ratio equals the target.

**One-shot** = `start_epoch == end_epoch`, single pruning step + remaining epochs as fine-tuning.

---

## 2. STP v2 Improvements

### 2.1 Rebase to TP v1.6.1

Previously on v1.5.1. We rebased onto upstream:

| Commit | What it brings |
|---|---|
| `2924e26` — V1.6.0 | Refactored dependency module: cleaner Node / Group / DG separation, improved isomorphic + MHA pruning support |
| `e80127d` — v1.6.1 | Code style cleanup |
| `a878886` | Internal version bumped to **2.0.0** for our fork |

### 2.2 Pruning Utils Cleanup

10 commits (`21a4bdf` → `cba9bf4`) modernizing `torch_pruning/utils/pruning_utils.py`:

- **PEP 8 class names:** `channel_pruning` → `ChannelPruning`, `slice_pruning` → `SlicePruning`
- **`PruningMethod` enum** replacing magic strings for pruning types
- **`_log()` helper** eliminating ~15 duplicate if-else logging blocks
- **Bug fixes:**
  - `ignored_layers` accumulation bug in `set_layers_to_prune()` (`e3a381f`)
  - Hardcoded `'cuda'` device in `SlicePruning.regularize()` (`1022317`)
- **Unit tests** added (`cba9bf4`)

### 2.3 Transformer Support

The original `pruning_utils.py` only handled Conv2d + BatchNorm. We extended it for transformer architectures, enabling the VBP work on ViTs (section 2.5).

**Done:**
- **Linear layer pruning** — `ChannelPruning` now discovers and prunes `nn.Linear` layers with proper in/out channel handling (`ab3e27f`)
- **LayerNorm pruning** — Coupled with Linear in dependency groups (ViT uses LN before/after attention and MLP)

**TODO:**
- **MHA head pruning** — Prune entire attention heads (Q/K/V/output projection simultaneously). TP v1.6 has `prune_num_heads` support in `BasePruner`; needs integration into `ChannelPruning`
- **Embedding dimension pruning** — Prune the residual stream width (affects all layers uniformly)
- **FFN-only vs full-model pruning modes** — Currently VBP prunes MLP intermediate dim only; extend to joint MLP + attention pruning

### 2.4 New Importance Criteria

#### MACAwareImportance (`6f389ba`)

Wraps any base importance (default: L2 magnitude) and scales scores by per-layer MAC cost, encouraging pruning of computationally expensive layers first.

```
score = alpha * norm(base_imp) + (1 - alpha) * (mac_ratio ^ beta)    # "Sum" mode
score = base_imp * (mac_ratio ^ beta)                                 # "Mul" mode
```

Key parameters: `alpha` (importance vs MAC trade-off), `beta` (MAC exponent).

#### VarianceImportance (`f8526fd`)

Post-activation variance as channel importance (VBP paper, arXiv 2507.12988). Collects exact statistics (no EMA) via forward hooks:

1. Register hooks on target layers → accumulate `sum(x)`, `sum(x^2)`, `count`
2. Compute `variance = E[x^2] - E[x]^2` per channel
3. During pruning: low-variance channels are pruned first

**`target_layers` parameter** enables architecture-specific hooking:
- **ViT:** `[(fc1, gelu_fn), ...]` — post-GELU stats on MLP intermediate layers
- **CNN:** Auto-detected via `build_cnn_target_layers(model, DG)` — walks the DG to find Conv → BN → activation chains and composes `post_act_fn`

#### CNN Auto-Detection Helpers

| Helper | Purpose |
|---|---|
| `build_cnn_target_layers(model, DG)` | Walk DG from each Conv2d → find BN + activation → compose into `post_act_fn` |
| `build_cnn_ignored_layers(model, arch)` | Build ignored layers for ResNet (stem, conv3, downsamples) or MobileNetV2 (stem, classifier, project/DW convs) |

### 2.5 VBP Integration

40+ commits (`f8526fd` → `e25177f`) adding Variance-Based Pruning with full pipeline support.

#### VBPPruner (`torch_pruning/pruner/algorithms/vbp_pruner.py`)

Extends `BasePruner` with three post-prune corrections applied *per group before* `group.prune()`:

1. **Bias compensation** — For each pruned channel, compute `delta_b = W[:, pruned] @ mu[pruned]` and add to consumer's bias. Handles Linear, Conv2d, and depthwise Conv2d consumers.
2. **BN variance update** — Analytically corrects `running_var` of downstream BatchNorm using stored activation variances, avoiding the need for full recalibration.
3. **Mean-check diagnostics** *(optional)* — Forward hooks measure per-channel mean shift before/after compensation for validation.

#### CNN Support

Tested architectures: **ResNet-50**, **MobileNetV2**.

Key challenges solved:
- **BN recalibration:** Must call `reset_running_stats()` before recalibrating — stale stats from the original model otherwise persist → 0% accuracy
- **DW conv group root:** In MobileNetV2, DW conv is the group root but stats are on the expand conv. `_apply_compensation` searches the group for a module with matching-size means
- **DW conv + `ignored_layers`:** DW convs must NOT be in `ignored_layers` — they appear in expand conv groups with `out_channel` pruning, causing group rejection
- **`nn.ReLU6` mapping:** `ReLU6` uses `HardtanhBackward0` internally — mapped `"hardtanh"` → `relu6` in activation detection

#### PAT Pipeline Features

- **DDP support** via `torchrun` with proper stat synchronization across ranks
- **Knowledge distillation** (optional `--use_kd`) with soft cross-entropy loss
- **Variance entropy loss** (optional `--var_loss_weight`) as auxiliary regularization
- **Sparse pre-training** modes: `l1_group` (L2,1), `gmp` (Gradual Magnitude Pruning)
- **Sweep mode**: Collect stats once, then test multiple keep ratios via `deepcopy` + `remap_importance()`

### 2.6 Graph Visualization

5 commits (`c9ce4fd` → `09661c7`) adding Graphviz-based visualization of the DependencyGraph and pruning groups.

`torch_pruning/utils/visualization.py` provides 3 views via `visualize_all_views()`:

- **Computational graph (CG)** — Data flow through the model (gray solid edges)
- **Dependency graph (DG)** — Pruning coupling between layers (green dashed = direct, red dotted = force-shape-match)
- **Combined** — Both overlaid with consistent node layout

Key features:
- **Group cluster boxes** — Nodes belonging to the same pruning group are wrapped in Graphviz `subgraph cluster` containers (`eafd5a7`)
- **Color-coded nodes** — 12+ OPTYPE categories: Conv (blue), Linear (red), BN (green), ElementWise (yellow), etc.
- **Multi-group detection** — Nodes appearing in multiple groups get double borders
- **Consistent layout backbone** — All 3 views share the same computational edge layout (`constraint=true`) so node positions are stable across views
- **ElementWise naming** — Maps `grad_fn` names to readable labels (Add, ReLU, GELU, etc.)
- **Output formats:** PNG, SVG, PDF

Legacy matplotlib heatmap functions (`draw_computational_graph`, `draw_groups`, `draw_dependency_graph`) remain in `torch_pruning/utils/utils.py`.

Demo: `python examples/visualization/demo_improved_viz.py [--resnet] [--format svg]`

---

## 3. Key Source Files

| File | Description |
|---|---|
| `torch_pruning/dependency/graph.py` | DependencyGraph: `build_dependency()`, `get_all_groups()` |
| `torch_pruning/pruner/algorithms/base_pruner.py` | BasePruner: `step()`, `_prune()` generator, scope/threshold logic |
| `torch_pruning/pruner/algorithms/vbp_pruner.py` | VBPPruner: bias compensation, BN variance update |
| `torch_pruning/pruner/importance.py` | All importance criteria + `build_cnn_target_layers/ignored_layers` |
| `torch_pruning/utils/pruning_utils.py` | `ChannelPruning`, `SlicePruning`, `PruningMethod` enum |
| `torch_pruning/utils/visualization.py` | Graphviz-based DG/CG visualization with group clusters |
| `benchmarks/vbp/vbp_imagenet.py` | Full PAT pipeline: sparse → PAT → FT, DDP, KD |
| `benchmarks/vbp/sparse_utils.py` | `l21_regularization()`, `gmp_sparsity_schedule()`, `apply_unstructured_pruning()` |
| `benchmarks/vbp/vbp_imagenet_pat.py` | Thin PAT demo via Pruning class |
| `benchmarks/vbp/plot_results.py` | Log parser, comparison against paper Table 10 |
