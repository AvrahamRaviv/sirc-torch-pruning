# SIRC Torch-Pruning (SHANO)

Fork of [Torch-Pruning](https://github.com/VainF/Torch-Pruning) extended with VBP (Variance-Based Pruning), PAT (Pruning-Aware Training), bias compensation, and MAC-aware importance.

**Repo:** https://github.com/AvrahamRaviv/sirc-torch-pruning.git
**Branch:** `master` (single branch, linear history)
**Paper reference:** arxiv 2507.12988

## Project Structure

```
torch_pruning/           # Core library
  pruner/
    importance.py        # VarianceImportance (VBP stats collection)
    algorithms/
      vbp_pruner.py      # VBPPruner with 3D tensor + bias compensation
      base_pruner.py     # BasePruner (group filtering, ignored_layers)
  dependency/            # Dependency graph, group formation
  utils/
    pruning_utils.py     # Pruning orchestration (Conv2d + Linear + LayerNorm)

benchmarks/vbp/          # VBP research scripts
  vbp_imagenet.py        # Main benchmark: one-shot + PAT + sparse + FT pipeline
  vbp_imagenet_pat.py    # Standalone PAT via pruning_utils.py
  plot_results.py        # Parse logs, compare against paper Table 10
  backlog.md             # Research roadmap
  convnext.py            # FB ConvNeXt architecture
  sparse_utils.py        # Sparse pre-training utilities

tests/                   # 34 test files
examples/                # ViT, BERT, YOLOv5/7/8, LLMs, timm
reproduce/               # Paper reproduction scripts
```

## Code Guidelines

- **No orphaned code.** If you replace a function, delete the old one. Don't leave commented-out blocks.
- **Reuse existing helpers.** Before writing new code, check if a function already does what you need (e.g., `finetune()`, `build_ft_scheduler()`, `collect_and_sync_stats()`).
- **DRY across modes.** One-shot and PAT share the same `run_pat()` path. Don't duplicate logic — generalize or extract helpers.
- **Verbose logging.** Every training/eval loop must log progress. Use `log_info()` (not print). Train loops: ~20 log lines per epoch via `log_interval`. Eval/stats: tqdm is fine.
- **Commit frequently.** Small, focused commits with clear messages. Don't bundle unrelated changes.

## ML/DL Specifics

- **Scheduler granularity.** `build_ft_scheduler()` returns `(scheduler, step_per_batch)` — always use the flag, never call `scheduler.step()` bare.
- **Stats before pruner.** Always `collect_and_sync_stats()` before creating a pruner. Uniform importance (no stats) → global threshold selects ALL channels → silent failure.
- **ViT MLP-only pruning.** Ignore: classifier, patch_embedding, all attention (Q/K/V/output.dense), fc2. Only fc1 groups get pruned.
- **MPS compatibility.** `pin_memory=False`, `num_workers=0` on Apple Silicon.

## VBP Pipeline (`vbp_imagenet.py`)

Three-phase pipeline, all optional:
1. **Sparse pre-training** (`--sparse_mode`, `--epochs_sparse`)
2. **PAT / One-shot pruning** (`--pat --pat_steps N --pat_epochs_per_step M` or just `--keep_ratio`)
3. **Post-prune fine-tuning** (`--epochs_ft`)

One-shot = `pat_steps=1, pat_epochs_per_step=0, epochs_ft=N`.

## Testing

```bash
# Run core tests
python -m pytest tests/ -x -q

# Quick VBP sanity check (single GPU, no DDP)
python benchmarks/vbp/vbp_imagenet.py --model_type vit \
  --model_name <path> --data_path <path> \
  --keep_ratio 0.9 --disable_ddp --epochs_ft 1
```

## Learnings (CC should update this)

When Claude Code makes a mistake and gets corrected, record the lesson here to avoid repeating it.

- `miniters` alone doesn't throttle tqdm — need `mininterval` too, or switch to manual logging
- `re.search` only finds first match — use `re.finditer` when you need the last occurrence (e.g., PAT retention)
