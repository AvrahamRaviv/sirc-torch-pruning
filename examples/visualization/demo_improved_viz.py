"""Demo: Improved dependency graph visualization.

Features demonstrated:
1. Differentiated dependency edges (direct=green vs force-shape-match=red)
2. Bidirectional edges merged into single edge with arrows on both ends
3. Group clustering for pruning groups
4. Human-readable ElementOps names (Add, ReLU, etc.)

Run:
    python examples/visualization/demo_improved_viz.py

For ResNet18 visualization (requires torchvision):
    python examples/visualization/demo_improved_viz.py --resnet
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
    args = parser.parse_args()

    model, example_inputs = get_model(args.resnet)

    print("Building dependency graph...")
    DG = tp.DependencyGraph().build_dependency(model, example_inputs)

    # Output directory
    output_dir = "./demo_viz"
    os.makedirs(output_dir, exist_ok=True)

    # Generate visualizations (both SVG and PNG)
    for fmt in ["svg", "png"]:
        # 1. Dependency graph with groups
        print(f"\nGenerating dependency graph ({fmt})...")
        tp.utils.visualize_graph(
            DG,
            output_path=f"{output_dir}/dependency_graph",
            view="dependency",
            show_groups=True,
            differentiate_dependencies=True,
            format=fmt,
        )

        # 2. Computational graph
        print(f"Generating computational graph ({fmt})...")
        tp.utils.visualize_graph(
            DG,
            output_path=f"{output_dir}/computational_graph",
            view="computational",
            show_groups=True,
            format=fmt,
        )

        # 3. Combined view
        print(f"Generating combined graph ({fmt})...")
        tp.utils.visualize_graph(
            DG,
            output_path=f"{output_dir}/combined_graph",
            view="both",
            show_groups=True,
            differentiate_dependencies=True,
            hide_redundant_computational=True,
            format=fmt,
        )

    print(f"\nSaved all files to: {output_dir}/")
    print("  - dependency_graph.svg/png")
    print("  - computational_graph.svg/png")
    print("  - combined_graph.svg/png")

    # Print legend
    print("\n" + "=" * 50)
    print("Legend:")
    print("  Edges:")
    print("    Green dashed (vee): Direct dependency")
    print("    Red dotted (diamond): Force-shape-match")
    print("    Gray solid: Computational flow")
    print("  Node borders (groups):")
    print("    Colored thick border = nodes in same pruning group")
    print("    Same border color = same group")
    print("=" * 50)


if __name__ == "__main__":
    main()
