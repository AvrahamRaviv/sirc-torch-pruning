#!/usr/bin/env python3
"""
Example: Analyze per-channel contributions of magnitude vs variance
in combined importance mode.
"""

import torch
import torch.nn as nn
import torch_pruning as tp

def analyze_combined_importance(model, train_loader, device, alpha=0.5, normalize=False):
    """
    Collect stats and analyze per-channel importance decomposition.

    Args:
        model: Neural network to analyze
        train_loader: DataLoader for calibration
        device: torch device
        alpha: Magnitude weight in combined mode (0.5 = equal weight)
        normalize: Whether to normalize magnitude/variance per-layer
    """
    # Create importance object
    importance = tp.importance.VarianceImportance(
        importance_mode="combined",
        alpha=alpha,
        normalize=normalize
    )

    # Collect statistics
    print("Collecting activation statistics...")
    target_layers = tp.importance.build_target_layers(model, tp.DependencyGraph())
    importance.collect_statistics(
        model, train_loader, device,
        target_layers=target_layers,
        max_batches=20  # Use more for production
    )

    # Build dependency graph for pruning groups
    DG = tp.DependencyGraph()
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    DG.build_dependency(model, example_inputs=example_inputs)

    # Analyze each pruning group
    print("\n" + "="*70)
    print("PER-LAYER COMBINED IMPORTANCE ANALYSIS")
    print("="*70 + "\n")

    layer_stats = []
    for group_idx, group in enumerate(DG.get_all_groups()):
        dep, idxs = group[0]
        module = dep.target.module

        # Skip if no stats collected
        if module not in importance.variance:
            continue

        if not hasattr(module, '__class__'):
            continue

        print(f"Group {group_idx}: {module.__class__.__name__}")
        print(f"  Channels analyzed: {len(idxs)}")

        # Get decomposition
        decomp = importance.decompose_combined(group, verbose=False)
        if decomp is None:
            continue

        # Store stats
        layer_stats.append({
            "name": f"{module.__class__.__name__}_{group_idx}",
            "mag_mean": decomp["mag"].mean(),
            "var_mean": decomp["var"].mean(),
            "mag_pct_mean": decomp["mag_pct"].mean(),
            "var_pct_mean": decomp["var_pct"].mean(),
        })

        # Print per-channel breakdown
        mag_pct = decomp["mag_pct"]
        var_pct = decomp["var_pct"]
        combined = decomp["combined"]

        print(f"  Magnitude contribution: {mag_pct.mean():.1f}% ± {mag_pct.std():.1f}%")
        print(f"  Variance contribution:  {var_pct.mean():.1f}% ± {var_pct.std():.1f}%")
        print(f"  Combined score range: [{combined.min():.4f}, {combined.max():.4f}]")

        # Show channels where one component dominates
        mag_dominant = (mag_pct > 70).sum()
        var_dominant = (var_pct > 70).sum()
        if mag_dominant > 0:
            print(f"  ⚠ {mag_dominant}/{len(idxs)} channels dominated by magnitude (>70%)")
        if var_dominant > 0:
            print(f"  ⚠ {var_dominant}/{len(idxs)} channels dominated by variance (>70%)")

        print()

    # Summary
    if layer_stats:
        print("\n" + "="*70)
        print("SUMMARY ACROSS ALL LAYERS")
        print("="*70)
        avg_mag_pct = sum(s["mag_pct_mean"] for s in layer_stats) / len(layer_stats)
        avg_var_pct = sum(s["var_pct_mean"] for s in layer_stats) / len(layer_stats)
        print(f"Average magnitude contribution: {avg_mag_pct:.1f}%")
        print(f"Average variance contribution:  {avg_var_pct:.1f}%")
        print(f"Blending is {'balanced' if 40 < avg_mag_pct < 60 else 'biased'}")
        print()


if __name__ == "__main__":
    # Example usage (requires model + data)
    # from torchvision import models
    # model = models.mobilenet_v2(pretrained=True)
    # train_loader = ...  # your dataloader
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # analyze_combined_importance(
    #     model, train_loader, device,
    #     alpha=0.5,
    #     normalize=False
    # )

    print("Usage example:")
    print("  from torchvision import models")
    print("  model = models.mobilenet_v2(pretrained=True)")
    print("  train_loader = ...  # your dataloader")
    print("  analyze_combined_importance(")
    print("    model, train_loader, device='cuda',")
    print("    alpha=0.5, normalize=False")
    print("  )")
