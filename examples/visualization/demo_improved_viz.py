"""Demo: Improved dependency graph visualization.

Features demonstrated:
1. Consistent layout across all 3 views (computational edges as layout backbone)
2. Group boxes (graphviz cluster subgraphs) wrapping pruning groups
3. Differentiated dependency edges (direct=green vs force-shape-match=red)
4. Double-border nodes for multi-group membership (residual connections)
5. Human-readable ElementOps names (Add, ReLU, etc.)

Run:
    python examples/visualization/demo_improved_viz.py

For ResNet18 visualization (requires torchvision):
    python examples/visualization/demo_improved_viz.py --resnet

Output as SVG:
    python examples/visualization/demo_improved_viz.py --format svg
"""
import argparse
import os
import torch
import torch.nn as nn
import torch_pruning as tp


class SimpleResidualBlock(nn.Module):
    """A simple residual block to demonstrate force-shape-match dependencies."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class DemoNet(nn.Module):
    """Demo network with residual blocks."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = SimpleResidualBlock(64)
        self.block2 = SimpleResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_model(use_resnet):
    """Get the model to visualize."""
    if use_resnet:
        try:
            from torchvision.models import resnet18
            print("Loading ResNet18...")
            model = resnet18(weights=None)
            example_inputs = torch.randn(1, 3, 224, 224)
        except ImportError:
            print("torchvision not installed, using DemoNet instead")
            model = DemoNet()
            example_inputs = torch.randn(1, 3, 32, 32)
    else:
        print("Loading DemoNet (simple residual network)...")
        model = DemoNet()
        example_inputs = torch.randn(1, 3, 32, 32)
    return model, example_inputs


def main():
    parser = argparse.ArgumentParser(description="Demo improved visualization")
    parser.add_argument("--resnet", action="store_true", help="Use ResNet18 (requires torchvision)")
    parser.add_argument("--format", default="png", choices=["png", "svg", "pdf"],
                        help="Output format (default: png)")
    args = parser.parse_args()

    model, example_inputs = get_model(args.resnet)

    print("Building dependency graph...")
    DG = tp.DependencyGraph().build_dependency(model, example_inputs)

    output_dir = "./demo_viz"

    print(f"\nGenerating all 3 views ({args.format})...")
    tp.utils.visualize_all_views(
        DG,
        output_dir=output_dir,
        basename="graph",
        format=args.format,
        differentiate_dependencies=True,
    )

    print(f"\nSaved all files to: {output_dir}/")
    print(f"  - graph_computational.{args.format}")
    print(f"  - graph_dependency.{args.format}")
    print(f"  - graph_both.{args.format}")

    print("\n" + "=" * 55)
    print("Legend:")
    print("  Edges:")
    print("    Gray solid arrow: Computational flow (data direction)")
    print("    Green dashed line: Direct pruning coupling")
    print("    Red dotted line: Shape-match coupling (residual)")
    print("  Nodes:")
    print("    Double border: Node belongs to multiple pruning groups")
    print("  Group boxes:")
    print("    Light gray rectangles: Pruning group clusters")
    print("    Visible in dependency and combined views")
    print("    Same cluster = nodes pruned together")
    print("=" * 55)


if __name__ == "__main__":
    main()
