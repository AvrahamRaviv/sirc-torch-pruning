# Project Status

Last updated: 2025-02-10

## Completed

### Core Library Extensions
- [x] VarianceImportance with `target_layers` support for post-GELU stats
- [x] VBPPruner with 3D tensor support + bias compensation
- [x] `pruning_utils.py` extended for Linear + LayerNorm (commit ab3e27f)
- [x] Dependency graph visualization with group cluster boxes

### VBP Benchmark (`vbp_imagenet.py`)
- [x] ViT (DeiT-T) + ConvNeXt support
- [x] MLP-only pruning for ViT (ignore attention + residual stream)
- [x] DDP multi-GPU training with proper stats sync
- [x] Knowledge distillation (KD) integration
- [x] Unified `run_pat()` for both one-shot and iterative pruning
- [x] `finetune()` helper — DDP wrap/unwrap, scheduler, best-model saving
- [x] `build_ft_scheduler()` factory with `step_per_batch` flag
- [x] Three-phase pipeline: sparse pre-train → PAT → post-prune FT
- [x] Sparse pre-training (L1-group, GMP modes)
- [x] Variance concentration loss hooks for PAT
- [x] `plot_results.py` — parses logs, compares against paper Table 10
- [x] Train loop: tqdm replaced with periodic `log_info` (~20 per epoch)

### Bug Fixes
- [x] Cosine LR scheduler stepping (per-batch, not per-epoch)
- [x] DDP stats broadcast (use module names, not objects)
- [x] `plot_results.py` — PAT retention format parsing (last occurrence)
- [x] tqdm flooding (refresh=False + mininterval)
- [x] Python 3.9 compat (`__future__` annotations in sparse_utils)

## In Progress

### VBP for Standard CNNs
- [ ] ResNet-50 support (model loading, ignored_layers, target_layers)
- [ ] MobileNetV2 / EfficientNet-B0 support
- [ ] Post-BN+ReLU stats collection (not post-BN alone)
- [ ] Validate bias compensation with BatchNorm layers

### Debug ConvNeXt Paper Gap
- [ ] Paper reports 81.3% ConvNeXt-T, we get 80.9% (−0.4%)
- [ ] Compare fine-tuning configs vs paper code
- [ ] Ablate: is gap from pruning step or fine-tuning step?
- [ ] Check EMA / stochastic depth settings

### Test pruning_utils.py Linear/LayerNorm Refactoring
- [ ] Backward compat: run existing Conv2d config, confirm identical results
- [ ] Run `vbp_imagenet_pat.py --disable_ddp` — verify Linear deps in init log
- [ ] Test mask-only mode: `pruner.prune(model, epoch, mask_only=True)`
- [ ] Verify LayerNorm masking in pruning groups

## Known Issues / Bugs

- `vbp_imagenet_pat.py` has redundant code vs `vbp_imagenet.py` — not yet unified
- `base_pruner.py:412` TODO: no general handling for `torch.unbind` in timm
- `index_mapping.py:167,255` TODO: reshape/view/flatten support is limited
- `graph.py:533` TODO: ViT pruning improvements needed
- `reproduce/main_imagenet.py:235` FIXME: dataset size accounting

## Key Commands

```bash
# One-shot VBP (paper reproduction)
python benchmarks/vbp/vbp_imagenet.py \
  --model_type vit --model_name <path> --data_path <path> \
  --keep_ratio 0.65 --global_pruning --use_kd --epochs_ft 10 --disable_ddp

# PAT (5 steps × 1 epoch) + 5 epoch post-prune FT
python benchmarks/vbp/vbp_imagenet.py \
  --model_type vit --model_name <path> --data_path <path> \
  --keep_ratio 0.65 --global_pruning --use_kd \
  --pat --pat_steps 5 --pat_epochs_per_step 1 --epochs_ft 5 --disable_ddp

# Full 3-phase: 5 sparse + 5×1 PAT + 5 FT
python benchmarks/vbp/vbp_imagenet.py \
  --model_type vit --model_name <path> --data_path <path> \
  --keep_ratio 0.65 --global_pruning --use_kd \
  --sparse_mode l1_group --epochs_sparse 5 \
  --pat --pat_steps 5 --pat_epochs_per_step 1 --epochs_ft 5 --disable_ddp

# Plot results
cd /algo/NetOptimization/outputs/VBP/DeiT_tiny && python plot_results.py
```
