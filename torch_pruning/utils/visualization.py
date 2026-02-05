"""Dependency graph visualization using Graphviz.

This module provides intuitive node-edge diagram visualizations for dependency graphs,
replacing the heatmap-based approach with proper graph layouts.
"""
import os.path
from typing import List, Optional, Set, Union
import graphviz
from torch_pruning.ops import OPTYPE


# Color scheme by OPTYPE category
OPTYPE_COLORS = {
    # Convolution (Blue shades)
    OPTYPE.CONV: "#4A90D9",
    OPTYPE.DEPTHWISE_CONV: "#6BA3E0",

    # Linear/Embedding (Red shades)
    OPTYPE.LINEAR: "#E57373",
    OPTYPE.EMBED: "#EF9A9A",
    OPTYPE.LSTM: "#F48FB1",
    OPTYPE.MHA: "#CE93D8",

    # Normalization (Green shades)
    OPTYPE.BN: "#81C784",
    OPTYPE.LN: "#A5D6A7",
    OPTYPE.GN: "#C8E6C9",
    OPTYPE.IN: "#B2DFDB",

    # Structural (Purple shades)
    OPTYPE.CONCAT: "#B39DDB",
    OPTYPE.SPLIT: "#9FA8DA",
    OPTYPE.RESHAPE: "#90CAF9",
    OPTYPE.UNBIND: "#80DEEA",
    OPTYPE.EXPAND: "#80CBC4",
    OPTYPE.SLICE: "#A5D6A7",

    # Other (Yellow/Orange shades)
    OPTYPE.ELEMENTWISE: "#FFD54F",
    OPTYPE.PARAMETER: "#FFB74D",
    OPTYPE.PRELU: "#FF8A65",
    OPTYPE.CUSTOMIZED: "#BCAAA4",
}

# Default color for unknown types
DEFAULT_COLOR = "#BDBDBD"


def _get_node_color(optype: OPTYPE) -> str:
    """Get the color for a given OPTYPE."""
    return OPTYPE_COLORS.get(optype, DEFAULT_COLOR)


def _get_node_label(node, show_channels: bool, DG: "DependencyGraph") -> str:
    """Generate the label for a node.

    Args:
        node: The Node object.
        show_channels: Whether to show channel counts.
        DG: The DependencyGraph for accessing channel info.

    Returns:
        Formatted node label string.
    """
    # Get the layer name (short version)
    name = node._name if node._name else ""

    # Get the module class name
    module_class = node.module_class.__name__

    # Build label
    lines = []
    if name:
        # Truncate long names
        if len(name) > 25:
            name = "..." + name[-22:]
        lines.append(name)
    lines.append(module_class)

    if show_channels:
        in_ch = DG.get_in_channels(node.module)
        out_ch = DG.get_out_channels(node.module)
        if in_ch is not None or out_ch is not None:
            in_str = str(in_ch) if in_ch is not None else "?"
            out_str = str(out_ch) if out_ch is not None else "?"
            lines.append(f"{in_str} → {out_str}")

    return "\\n".join(lines)


def _get_short_fn_name(fn) -> str:
    """Get a short name for a pruning function."""
    if fn is None:
        return "None"
    name = fn.__name__
    # Simplify common names
    if "prune_out" in name:
        return "out"
    if "prune_in" in name:
        return "in"
    return name


def _should_skip_node(node, ignored_types: Optional[Set[OPTYPE]]) -> bool:
    """Check if a node should be skipped based on ignored types."""
    if ignored_types is None:
        return False
    return node.type in ignored_types


def visualize_graph(
    DG: "DependencyGraph",
    output_path: Optional[str] = None,
    view: str = "both",
    format: str = "png",
    show_channels: bool = True,
    show_edge_labels: bool = False,
    rankdir: str = "TB",
    highlight_modules: Optional[List] = None,
    ignored_types: Optional[List[OPTYPE]] = None,
) -> None:
    """Visualize the dependency graph using Graphviz.

    Args:
        DG: The DependencyGraph to visualize.
        output_path: Path to save the output file (without extension).
            If None, returns the graph object without rendering.
        view: Type of visualization:
            - "computational": Show data flow edges (inputs/outputs)
            - "dependency": Show pruning dependency edges
            - "both": Show both types of edges
        format: Output format ("png", "svg", "pdf").
        show_channels: Whether to show input/output channel counts on nodes.
        show_edge_labels: Whether to show trigger/handler names on dependency edges.
        rankdir: Graph direction ("TB" for top-bottom, "LR" for left-right).
        highlight_modules: List of modules to highlight with a thicker border.
        ignored_types: List of OPTYPE values to hide from the graph.

    Returns:
        The graphviz.Digraph object.

    Raises:
        ImportError: If graphviz is not installed.
        ValueError: If an invalid view mode is specified.

    Example:
        >>> import torch_pruning as tp
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> DG = tp.DependencyGraph().build_dependency(model, torch.randn(1,3,224,224))
        >>> DG.visualize("graph.png")
    """

    if view not in ("computational", "dependency", "both"):
        raise ValueError(f"Invalid view mode: {view}. Must be 'computational', 'dependency', or 'both'")

    # Convert ignored_types to a set for faster lookup
    ignored_set = set(ignored_types) if ignored_types else None

    # Convert highlight_modules to a set of modules
    highlight_set = set(highlight_modules) if highlight_modules else set()

    # Create the graph
    dot = graphviz.Digraph(
        name="DependencyGraph",
        format=format,
        graph_attr={
            "rankdir": rankdir,
            "splines": "ortho",
            "nodesep": "0.5",
            "ranksep": "0.5",
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": "Helvetica",
            "fontsize": "10",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "8",
        },
    )

    # Create a mapping from node to unique ID
    node_ids = {}
    for i, node in enumerate(DG.module2node.values()):
        node_ids[node] = f"node_{i}"

    # Add nodes
    for node in DG.module2node.values():
        if _should_skip_node(node, ignored_set):
            continue

        node_id = node_ids[node]
        label = _get_node_label(node, show_channels, DG)
        color = _get_node_color(node.type)

        # Check if this node should be highlighted
        penwidth = "3" if node.module in highlight_set else "1"

        dot.node(
            node_id,
            label=label,
            fillcolor=color,
            penwidth=penwidth,
        )

    # Track edges to avoid duplicates
    added_edges = set()

    # Add computational edges (data flow)
    if view in ("computational", "both"):
        for node in DG.module2node.values():
            if _should_skip_node(node, ignored_set):
                continue

            node_id = node_ids[node]

            # Add edges to output nodes
            for out_node in node.outputs:
                if _should_skip_node(out_node, ignored_set):
                    continue
                out_id = node_ids.get(out_node)
                if out_id is None:
                    continue

                edge_key = (node_id, out_id, "computational")
                if edge_key not in added_edges:
                    added_edges.add(edge_key)
                    dot.edge(
                        node_id,
                        out_id,
                        color="black",
                        style="solid",
                        arrowhead="normal",
                    )

        # Render if output path is provided
        if output_path is not None:
            # Remove extension if provided (graphviz adds it)
            if output_path.endswith(f".{format}"):
                output_path = output_path[:-len(format)-1]
            viz_dir = os.path.join(output_path, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            dot.render(os.path.join(viz_dir, "computational_graph.png"), cleanup=True)

    # Add dependency edges (pruning relationships)
    if view in ("dependency", "both"):
        for node in DG.module2node.values():
            if _should_skip_node(node, ignored_set):
                continue

            for dep in node.dependencies:
                source = dep.source
                target = dep.target

                if _should_skip_node(source, ignored_set) or _should_skip_node(target, ignored_set):
                    continue

                source_id = node_ids.get(source)
                target_id = node_ids.get(target)

                if source_id is None or target_id is None:
                    continue

                edge_key = (source_id, target_id, "dependency")
                if edge_key not in added_edges:
                    added_edges.add(edge_key)

                    edge_attrs = {
                        "color": "#E53935",
                        "style": "dashed",
                        "arrowhead": "vee",
                    }

                    if show_edge_labels:
                        trigger_name = _get_short_fn_name(dep.trigger)
                        handler_name = _get_short_fn_name(dep.handler)
                        edge_attrs["label"] = f"{trigger_name}→{handler_name}"
                        edge_attrs["fontcolor"] = "#E53935"

                    dot.edge(source_id, target_id, **edge_attrs)

        # Render if output path is provided
        if output_path is not None:
            # Remove extension if provided (graphviz adds it)
            if output_path.endswith(f".{format}"):
                output_path = output_path[:-len(format)-1]
            viz_dir = os.path.join(output_path, "viz")
            os.makedirs(viz_dir, exist_ok=True)
            dot.render(os.path.join(viz_dir, "dependency_graph.png"), cleanup=True)
