"""Dependency graph visualization using Graphviz.

This module provides intuitive node-edge diagram visualizations for dependency graphs,
replacing the heatmap-based approach with proper graph layouts.
"""
import os.path
import re
from typing import List, Optional, Set, Dict, Any, TYPE_CHECKING
import graphviz
from torch_pruning.ops import OPTYPE, _ElementWiseOp

if TYPE_CHECKING:
    from torch_pruning.dependency import DependencyGraph, Dependency


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

# Group cluster colors (fill, border)
GROUP_COLORS = [
    ('#BBDEFB', '#1565C0'),  # Blue
    ('#FFE0B2', '#E65100'),  # Orange
    ('#C8E6C9', '#2E7D32'),  # Green
    ('#F8BBD9', '#C2185B'),  # Pink
    ('#E1BEE7', '#7B1FA2'),  # Purple
    ('#B2EBF2', '#00838F'),  # Cyan
    ('#FFF9C4', '#F9A825'),  # Yellow
    ('#D7CCC8', '#5D4037'),  # Brown
]

# Human-readable names for ElementWise operations
ELEMENTWISE_NAMES = {
    'addbackward': 'Add', 'subbackward': 'Sub', 'mulbackward': 'Mul',
    'divbackward': 'Div', 'negbackward': 'Neg',
    'relubackward': 'ReLU', 'leakyrelubackward': 'LeakyReLU',
    'gelubackward': 'GELU', 'silubackward': 'SiLU',
    'sigmoidbackward': 'Sigmoid', 'tanhbackward': 'Tanh',
    'softmaxbackward': 'Softmax', 'elubackward': 'ELU',
    'maxpool2dwithindicesbackward': 'MaxPool2d',
    'avgpool2dbackward': 'AvgPool2d',
    'adaptiveavgpool2dbackward': 'AdaptiveAvgPool2d',
}

# Structural op types (force-shape-match dependencies)
STRUCTURAL_TYPES = {
    OPTYPE.CONCAT, OPTYPE.SPLIT, OPTYPE.RESHAPE,
    OPTYPE.UNBIND, OPTYPE.EXPAND, OPTYPE.SLICE, OPTYPE.ELEMENTWISE,
}


def _get_node_color(optype: OPTYPE) -> str:
    """Get the color for a given OPTYPE."""
    return OPTYPE_COLORS.get(optype, DEFAULT_COLOR)


def _get_elementwise_name(module) -> str:
    """Convert _ElementWiseOp grad_fn to human-readable name."""
    if not isinstance(module, _ElementWiseOp):
        return module.__class__.__name__
    grad_fn_str = str(module._grad_fn).lower()
    for key, name in ELEMENTWISE_NAMES.items():
        if key in grad_fn_str:
            return name
    match = re.search(r'([A-Za-z]+)backward', grad_fn_str, re.IGNORECASE)
    return match.group(1).title() if match else 'Op'


def _get_dependency_type(dep: "Dependency") -> str:
    """Classify dependency as 'direct' or 'force_shape_match'."""
    has_index_mapping = any(m is not None for m in dep.index_mapping)
    involves_structural = (dep.source.type in STRUCTURAL_TYPES or
                          dep.target.type in STRUCTURAL_TYPES)
    if has_index_mapping or involves_structural:
        return "force_shape_match"
    return "direct"


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

    # Get the module class name - use human-readable name for ElementWise ops
    if node.type == OPTYPE.ELEMENTWISE:
        module_class = _get_elementwise_name(node.module)
    else:
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
    show_groups: bool = False,
    differentiate_dependencies: bool = True,
    hide_redundant_computational: bool = True,
) -> graphviz.Digraph:
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
        show_groups: Whether to visually group nodes by pruning groups.
        differentiate_dependencies: Style direct (green) vs force-shape-match (red) differently.
        hide_redundant_computational: Hide computational edges that duplicate dependency edges.

    Returns:
        The graphviz.Digraph object.

    Raises:
        ValueError: If an invalid view mode is specified.

    Example:
        >>> import torch_pruning as tp
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> DG = tp.DependencyGraph().build_dependency(model, torch.randn(1,3,224,224))
        >>> tp.utils.visualize_graph(DG, "./graph", view="dependency", show_groups=True)
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
            "compound": "true",
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
    node_ids: Dict[Any, str] = {}
    for i, node in enumerate(DG.module2node.values()):
        node_ids[node] = f"node_{i}"

    # Collect pruning groups and build node-to-group mapping
    node_to_group: Dict[Any, int] = {}
    if show_groups and view in ("dependency", "both"):
        import torch.nn as nn
        groups_list = list(DG.get_all_groups(root_module_types=(nn.Conv2d, nn.Linear)))
        for group_idx, group in enumerate(groups_list):
            for item in group:
                if not _should_skip_node(item.dep.source, ignored_set):
                    node_to_group[item.dep.source] = group_idx
                if not _should_skip_node(item.dep.target, ignored_set):
                    node_to_group[item.dep.target] = group_idx

    # Collect edges with bidirectional merging
    # edge_registry: canonical (id1, id2) -> {computational: set(directions), dependencies: [(type, dep, direction)]}
    edge_registry: Dict[tuple, Dict] = {}

    def canonical_key(a: str, b: str):
        return (a, b, True) if a <= b else (b, a, False)

    # ALWAYS collect computational edges (for layout structure)
    for node in DG.module2node.values():
        if _should_skip_node(node, ignored_set):
            continue
        node_id = node_ids[node]
        for out_node in node.outputs:
            if _should_skip_node(out_node, ignored_set):
                continue
            out_id = node_ids.get(out_node)
            if out_id is None:
                continue
            key, is_fwd = canonical_key(node_id, out_id)[:2], canonical_key(node_id, out_id)[2]
            if key not in edge_registry:
                edge_registry[key] = {'computational': set(), 'dependencies': []}
            edge_registry[key]['computational'].add(is_fwd)

    # Collect dependency edges
    if view in ("dependency", "both"):
        for node in DG.module2node.values():
            if _should_skip_node(node, ignored_set):
                continue
            for dep in node.dependencies:
                if _should_skip_node(dep.source, ignored_set) or _should_skip_node(dep.target, ignored_set):
                    continue
                src_id, tgt_id = node_ids.get(dep.source), node_ids.get(dep.target)
                if src_id is None or tgt_id is None:
                    continue
                key, is_fwd = canonical_key(src_id, tgt_id)[:2], canonical_key(src_id, tgt_id)[2]
                if key not in edge_registry:
                    edge_registry[key] = {'computational': set(), 'dependencies': []}
                dep_type = _get_dependency_type(dep) if differentiate_dependencies else "default"
                edge_registry[key]['dependencies'].append((dep_type, dep, is_fwd))

    # Add all nodes with group coloring via border (preserves CG layout)
    for node in DG.module2node.values():
        if _should_skip_node(node, ignored_set):
            continue
        nid = node_ids[node]

        # Determine border color based on group membership
        if node in node_to_group:
            group_idx = node_to_group[node]
            _, border_color = GROUP_COLORS[group_idx % len(GROUP_COLORS)]
            border_width = "4"
        elif node.module in highlight_set:
            border_color = "black"
            border_width = "3"
        else:
            border_color = "black"
            border_width = "1"

        dot.node(nid, label=_get_node_label(node, show_channels, DG),
                 fillcolor=_get_node_color(node.type),
                 color=border_color, penwidth=border_width)

    # Render edges
    for (id1, id2), info in edge_registry.items():
        has_deps = len(info['dependencies']) > 0
        has_comp = len(info['computational']) > 0

        # Dependency edges
        if has_deps:
            type_dirs: Dict[str, set] = {}
            type_dep: Dict[str, Any] = {}
            for dt, dep, is_fwd in info['dependencies']:
                if dt not in type_dirs:
                    type_dirs[dt] = set()
                    type_dep[dt] = dep
                type_dirs[dt].add(is_fwd)

            for dt, dirs in type_dirs.items():
                dep = type_dep[dt]
                bidir = len(dirs) == 2
                # constraint=false: dependency edges don't affect layout
                # Only computational edges determine node positions (preserves CG structure)
                if differentiate_dependencies:
                    if dt == "direct":
                        attrs = {"color": "#2E7D32", "style": "dashed", "penwidth": "1.5", "constraint": "false"}
                        arrow = "vee"
                    else:
                        attrs = {"color": "#C62828", "style": "dotted", "penwidth": "2", "constraint": "false"}
                        arrow = "diamond"
                else:
                    attrs = {"color": "#E53935", "style": "dashed", "constraint": "false"}
                    arrow = "vee"

                if bidir:
                    attrs.update({"dir": "both", "arrowhead": arrow, "arrowtail": arrow})
                else:
                    is_fwd = True in dirs
                    if is_fwd:
                        attrs["arrowhead"] = arrow
                    else:
                        attrs.update({"dir": "back", "arrowtail": arrow})

                if show_edge_labels:
                    attrs["xlabel"] = f"{_get_short_fn_name(dep.trigger)}→{_get_short_fn_name(dep.handler)}"
                    attrs["fontcolor"] = attrs["color"]
                dot.edge(id1, id2, **attrs)

        # Computational edges (base structure)
        if has_comp:
            # For "computational" view: always show
            # For "dependency" view: always show as base structure
            # For "both" view: optionally hide if redundant with dependency
            should_show = (view == "computational" or
                          view == "dependency" or
                          (view == "both" and (not has_deps or not hide_redundant_computational)))
            if should_show:
                bidir = len(info['computational']) == 2
                attrs = {"color": "#757575", "style": "solid"}
                if bidir:
                    attrs.update({"dir": "both", "arrowhead": "normal", "arrowtail": "normal"})
                elif True in info['computational']:
                    attrs["arrowhead"] = "normal"
                else:
                    attrs.update({"dir": "back", "arrowtail": "normal"})
                dot.edge(id1, id2, **attrs)

    # Render to file
    if output_path is not None:
        if output_path.endswith(f".{format}"):
            output_path = output_path[:-len(format)-1]
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        dot.render(output_path, cleanup=True)

    return dot
