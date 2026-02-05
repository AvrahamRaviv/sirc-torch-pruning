# SIRC Torch Pruning (PAT)

This project builds on top of [Torch-Pruning](https://github.com/VainF/Torch-Pruning) and adds Pruning-Aware Training (PAT) utilities and workflows.

## What is PAT here?

PAT extends structural pruning with training-time routines for:
- Channel pruning with regularization
- Slice pruning for block-structured sparsity
- Pruned-model loading helpers
- Lightweight graph visualization for dependency graphs

## Installation

```bash
pip install PAT
```

Editable install:
```bash
git clone https://github.com/avrahamraviv/sirc-torch-pruning.git
cd sirc-torch-pruning && pip install -e .
```

## Quickstart (PAT)

```python
import torch
import torch_pruning as tp

model = ...  # your model
config_folder = "path/to/config"

pruner = tp.utils.Pruning(model, config_folder)
for epoch in range(epochs):
    # training step ...
    pruner.channel_regularize(model)
    pruner.slice_regularize(model)
    pruner.prune(model, epoch, mask_only=True)
```

## Acknowledgements

This repo is based on Torch-Pruning. Please cite the original work when using DepGraph or TP utilities.
