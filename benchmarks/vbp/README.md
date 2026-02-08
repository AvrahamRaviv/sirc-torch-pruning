# VBP (Variance-Based Pruning) ImageNet Benchmark

This directory contains scripts to reproduce results from the VBP paper using the integrated `VarianceImportance` class in Torch-Pruning.

**Reference**: [VBP: Variance-Based Pruning](https://arxiv.org/pdf/2507.12988)

## Overview

VBP is a data-driven pruning method that uses activation variance to determine channel importance:
- **Low variance channels** are considered less important and pruned first
- **Bias compensation** reduces accuracy loss by adjusting successor layer biases

## Quick Start

### Single GPU (Debug Mode)

```bash
python benchmarks/vbp/vbp_imagenet.py \
    --model_type vit \
    --model_name google/vit-base-patch16-224 \
    --data_path /path/to/imagenet \
    --keep_ratio 0.65 \
    --global_pruning \
    --epochs_ft 10 \
    --disable_ddp
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet.py \
    --model_type vit \
    --model_name /path/to/deit_tiny \
    --data_path /path/to/imagenet \
    --keep_ratio 0.65 \
    --global_pruning \
    --epochs_ft 10 \
    --use_kd
```

### ConvNeXt

```bash
torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet.py \
    --model_type convnext \
    --model_name convnext_tiny \
    --convnext_checkpoint /path/to/convnext_tiny_22k_1k_224.pth \
    --data_path /path/to/imagenet \
    --keep_ratio 0.70 \
    --global_pruning \
    --epochs_ft 10
```

## Arguments

### Model
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `vit` | Model type: `vit` or `convnext` |
| `--model_name` | `google/vit-base-patch16-224` | HuggingFace model name/path |
| `--convnext_checkpoint` | None | Path to ConvNeXt weights (.pth) |
| `--bottleneck` | False | Bottleneck mode for ViT |

### Data
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Required | ImageNet root directory |
| `--train_batch_size` | 64 | Per-GPU training batch size |
| `--val_batch_size` | 128 | Validation batch size |
| `--num_workers` | 4 | DataLoader workers |

### Pruning
| Argument | Default | Description |
|----------|---------|-------------|
| `--keep_ratio` | 0.65 | Fraction of channels to keep |
| `--global_pruning` | False | Global vs per-layer pruning |
| `--norm_per_layer` | False | Normalize variance per layer |

### Fine-tuning
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs_ft` | 10 | Fine-tuning epochs |
| `--lr_ft` | 1.5e-5 | AdamW learning rate |

### Knowledge Distillation
| Argument | Default | Description |
|----------|---------|-------------|
| `--use_kd` | False | Enable KD from unpruned teacher |
| `--kd_alpha` | 0.7 | CE loss weight |
| `--kd_beta` | 0.3 | KD loss weight |
| `--kd_T` | 2.0 | Temperature |

### Distributed
| Argument | Default | Description |
|----------|---------|-------------|
| `--disable_ddp` | False | Single-GPU mode |

### Output
| Argument | Default | Description |
|----------|---------|-------------|
| `--save_dir` | `./output/vbp` | Output directory |

## How It Works

### 1. Statistics Collection

```python
imp = tp.importance.VarianceImportance(norm_per_layer=False)
imp.collect_statistics(model, train_loader, device)
```

Collects per-channel activation statistics (mean and variance) by running forward passes on training data.

### 2. VBPPruner Creation

```python
pruner = tp.pruner.VBPPruner(
    model, example_inputs,
    importance=imp,
    pruning_ratio=1.0 - args.keep_ratio,
    global_pruning=args.global_pruning,
    mean_dict=imp.means,
    ...
)
```

### 3. Pruning with Bias Compensation

```python
# Cache consumer inputs for compensation
pruner.enable_meancheck(model)
model.eval()
with torch.no_grad():
    model(example_inputs)

# VBPPruner applies compensation + prune per group
pruner.step(interactive=False, enable_compensation=True)
pruner.disable_meancheck()
```

For each group, VBPPruner compensates the successor layer's bias before pruning:
`b'₂ = b₂ + W₂_pruned @ μ_pruned`, where `μ` is the cached input mean of the consumer.

### 4. DDP Statistics Sync

When using multiple GPUs, statistics are collected on rank 0 and broadcast:

```python
if is_main():
    imp.collect_statistics(model, train_loader, device)

if use_ddp:
    dist.barrier()
    stats_list = [{"variance": imp.variance, "means": imp.means}]
    dist.broadcast_object_list(stats_list, src=0)
```

## Expected Results

With default settings on ImageNet:

| Model | Keep Ratio | Base Acc | Retention | Fine-tuned |
|-------|------------|----------|-----------|------------|
| DeiT-Tiny | 0.65 | 72.2% | ~65% | ~70% |
| ViT-Base | 0.65 | 81.1% | ~75% | ~79% |
| ConvNeXt-Tiny | 0.70 | 82.1% | ~78% | ~81% |

## Reproduce Original VBP Experiments

### DeiT-Tiny (Primary Experiment)

```bash
torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet.py \
    --model_type vit \
    --model_name facebook/deit-tiny-patch16-224 \
    --data_path /path/to/imagenet \
    --keep_ratio 0.65 \
    --global_pruning \
    --epochs_ft 10 \
    --lr_ft 1.5e-5 \
    --use_kd \
    --kd_alpha 0.7 \
    --kd_beta 0.3 \
    --kd_T 2.0 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --save_dir ./output/deit_tiny_vbp
```

**Expected Results:**
- Base accuracy: 72.2%
- Retention (before FT): ~65%
- After fine-tuning: ~70%
- MACs reduction: ~35%

### ConvNeXt-Tiny

```bash
torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet.py \
    --model_type convnext \
    --model_name convnext_tiny \
    --convnext_checkpoint /path/to/convnext_tiny_22k_1k_224.pth \
    --data_path /path/to/imagenet \
    --keep_ratio 0.70 \
    --global_pruning \
    --epochs_ft 10 \
    --lr_ft 1.5e-5 \
    --use_kd \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --save_dir ./output/convnext_tiny_vbp
```

**Expected Results:**
- Base accuracy: 82.1%
- Retention (before FT): ~78%
- After fine-tuning: ~81%
- MACs reduction: ~30%

### ViT-Base (Higher Capacity)

```bash
torchrun --nproc_per_node=4 benchmarks/vbp/vbp_imagenet.py \
    --model_type vit \
    --model_name google/vit-base-patch16-224 \
    --data_path /path/to/imagenet \
    --keep_ratio 0.65 \
    --global_pruning \
    --epochs_ft 10 \
    --use_kd \
    --save_dir ./output/vit_base_vbp
```

### Single-GPU Debug Run

```bash
python benchmarks/vbp/vbp_imagenet.py \
    --model_type vit \
    --model_name facebook/deit-tiny-patch16-224 \
    --data_path /path/to/imagenet \
    --keep_ratio 0.65 \
    --global_pruning \
    --epochs_ft 2 \
    --disable_ddp \
    --train_batch_size 32
```

### Hardware Requirements

- **GPUs**: 4x NVIDIA A100 (40GB) or equivalent
- **Training time**: ~4-6 hours per experiment (DeiT-Tiny)
- **Memory**: ~20GB per GPU during fine-tuning

### Hyperparameter Summary

| Parameter | DeiT-Tiny | ConvNeXt-Tiny | ViT-Base |
|-----------|-----------|---------------|----------|
| `keep_ratio` | 0.65 | 0.70 | 0.65 |
| `global_pruning` | ✓ | ✓ | ✓ |
| `epochs_ft` | 10 | 10 | 10 |
| `lr_ft` | 1.5e-5 | 1.5e-5 | 1.5e-5 |
| `use_kd` | ✓ | ✓ | ✓ |
| `kd_alpha` | 0.7 | 0.7 | 0.7 |
| `kd_beta` | 0.3 | 0.3 | 0.3 |
| `kd_T` | 2.0 | 2.0 | 2.0 |

### Output Files

After each run, find in `--save_dir`:
- `vbp_imagenet.log` - Full training log
- `vbp_best.pth` - Best checkpoint (highest val accuracy)
- `vbp_final.pth` - Final model state

## Files

- `vbp_imagenet.py` - Main reproduction script
- `convnext.py` - Facebook ConvNeXt implementation
- `README.md` - This file

## Troubleshooting

### Out of Memory
- Reduce `--train_batch_size` and `--val_batch_size`
- Statistics collection runs in eval mode with no gradients

### Slow Statistics Collection
- Create cached sample lists:
  ```python
  import pickle
  from torchvision.datasets import ImageFolder
  train_dst = ImageFolder('/path/to/imagenet/train')
  with open('train_samples.pkl', 'wb') as f:
      pickle.dump(train_dst.samples, f)
  ```

### DDP Errors
- Ensure torchrun is used: `torchrun --nproc_per_node=N`
- Check NCCL backend availability
- Use `--disable_ddp` for debugging

## Citation

```bibtex
@article{vbp2024,
  title={Variance-Based Pruning for Efficient Neural Networks},
  author={...},
  journal={arXiv preprint arXiv:2507.12988},
  year={2024}
}
```
