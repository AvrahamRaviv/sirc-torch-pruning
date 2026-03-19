#!/usr/bin/env python3
"""
Quick example: Inspect per-channel magnitude vs variance contributions
in combined importance mode during pruning.
"""

import torch
import torch.nn as nn
import torch_pruning as tp


def inspect_combined_decomposition(pruner, model, data_loader, device, alpha=0.5):
    """
    After creating pruner with combined mode, decompose per-channel scores.

    Example:
        >>> pruner = tp.pruner.VBPPruner(...)
        >>> inspect_combined_decomposition(pruner, model, train_loader, device)
    """
    importance = pruner.importance
    if not isinstance(importance, tp.importance.VarianceImportance):
        print("Error: Pruner does not use VarianceImportance")
        return

    if importance.importance_mode != "combined":
        print(f"Error: Expected combined mode, got {importance.importance_mode}")
        return

    # Iterate through groups and decompose
    print(f"\n{'='*70}")
    print(f"COMBINED IMPORTANCE DECOMPOSITION (alpha={alpha})")
    print(f"{'='*70}\n")

    for group_idx, group in enumerate(pruner.DG.get_all_groups()):
        try:
            decomp = importance.decompose_combined(group, verbose=True)
            if decomp is not None:
                print()
        except Exception as e:
            # Skip groups without stats
            pass


# Quick inline inspection (during pruning loop)
def quick_inspect_layer(importance, module, idxs):
    """
    Fast check: What's the mag vs var split for a specific layer?

    Usage:
        >>> for group in pruner.step(interactive=True):
        ...     decomp = quick_inspect_layer(
        ...         pruner.importance, group[0].target.module, group[1]
        ...     )
        ...     if decomp:
        ...         print(f"Mag: {decomp['mag_pct'].mean():.0f}%, "
        ...               f"Var: {decomp['var_pct'].mean():.0f}%")
    """
    if importance.importance_mode != "combined":
        return None

    var = importance.variance.get(module)
    if var is None:
        return None

    idxs = torch.as_tensor(idxs, dtype=torch.long)
    var_scores = var[idxs].clone().clamp(min=0.0).sqrt()
    w = module.weight.view(module.weight.shape[0], -1)
    mag_scores = w.norm(p=2, dim=1)[idxs]

    # Normalize
    mag_scores_norm = mag_scores
    var_scores_norm = var_scores
    if importance.normalize:
        var_mean = var_scores.mean()
        mag_mean = mag_scores.mean()
        if var_mean > importance.eps:
            var_scores_norm = var_scores / var_mean
        if mag_mean > importance.eps:
            mag_scores_norm = mag_scores / mag_mean

    # Contributions
    mag_contrib = importance.alpha * mag_scores_norm
    var_contrib = (1 - importance.alpha) * var_scores_norm
    total = mag_contrib + var_contrib
    mag_pct = (mag_contrib / (total + importance.eps) * 100).cpu().numpy()
    var_pct = (var_contrib / (total + importance.eps) * 100).cpu().numpy()

    return {
        "mag_pct": mag_pct,
        "var_pct": var_pct,
        "mag_mean": mag_scores.mean().item(),
        "var_mean": var_scores.mean().item(),
    }


if __name__ == "__main__":
    print(__doc__)
    print("\nUsage patterns:")
    print("\n1. After building pruner:")
    print("   pruner = tp.pruner.VBPPruner(..., importance_mode='combined')")
    print("   inspect_combined_decomposition(pruner, model, train_loader, device)")
    print("\n2. During pruning step:")
    print("   for group in pruner.step(interactive=True):")
    print("       decomp = quick_inspect_layer(pruner.importance, ...)")
    print("       print(f'Mag: {decomp[\"mag_pct\"].mean():.0f}%')")
    print("\n3. Direct decomposition:")
    print("   decomp = pruner.importance.decompose_combined(group, verbose=True)")
    print("   print(decomp['mag_pct'])  # Per-channel magnitude % contribution")
