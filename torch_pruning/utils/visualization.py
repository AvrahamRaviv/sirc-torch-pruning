"""Dependency graph visualization using Graphviz.

This module provides intuitive node-edge diagram visualizations for dependency graphs,
with consistent layout across views and group clustering via Graphviz subgraph clusters.
"""
from __future__ import annotations

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

# Uniform group box styling
GROUP_FILL = "#F5F5F5"
GROUP_BORDER = "#9E9E9E"
GROUP_PENWIDTH = "2"

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
    """Generate the label for a node."""
    name = node._name if node._name else ""
    if node.type == OPTYPE.ELEMENTWISE:
        module_class = _get_elementwise_name(node.module)
    else:
        module_class = node.module_class.__name__

    lines = []
    if name:
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


def _build_group_assignment(
    DG: "DependencyGraph",
    node_ids: Dict[Any, str],
    ignored_set: Optional[Set[OPTYPE]],
    group_root_types=None,
    ignored_layers=None,
) -> Dict[str, int]:
    """Build node_id -> group_idx mapping. Nodes in multiple groups get assigned to their first.

    Returns:
        node_id_to_group: mapping from node_id string to group index
        multi_group_nodes: set of node_ids that appear in more than one group
        num_groups: total number of groups
    """
    import torch.nn as nn
    if group_root_types is None:
        group_root_types = (nn.Conv2d, nn.Linear)

    groups_list = list(DG.get_all_groups(root_module_types=group_root_types, ignored_layers=ignored_layers))
    node_id_to_group: Dict[str, int] = {}
    multi_group_nodes: Set[str] = set()

    for group_idx, group in enumerate(groups_list):
        for item in group:
            for node in (item.dep.source, item.dep.target):
                if _should_skip_node(node, ignored_set):
                    continue
                nid = node_ids.get(node)
                if nid is None:
                    continue
                if nid in node_id_to_group:
                    if node_id_to_group[nid] != group_idx:
                        multi_group_nodes.add(nid)
                    # Keep first assignment
                else:
                    node_id_to_group[nid] = group_idx

    return node_id_to_group, multi_group_nodes, len(groups_list)


def _collect_edges(
    DG: "DependencyGraph",
    node_ids: Dict[Any, str],
    ignored_set: Optional[Set[OPTYPE]],
    collect_deps: bool,
    differentiate_dependencies: bool,
) -> Dict[tuple, Dict]:
    """Collect computational and dependency edges into a unified registry."""

    edge_registry: Dict[tuple, Dict] = {}

    def canonical_key(a: str, b: str):
        return (a, b, True) if a <= b else (b, a, False)

    # Always collect computational edges (layout backbone)
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
            ck = canonical_key(node_id, out_id)
            key, is_fwd = (ck[0], ck[1]), ck[2]
            if key not in edge_registry:
                edge_registry[key] = {'computational': set(), 'dependencies': []}
            edge_registry[key]['computational'].add(is_fwd)

    # Collect dependency edges
    if collect_deps:
        for node in DG.module2node.values():
            if _should_skip_node(node, ignored_set):
                continue
            for dep in node.dependencies:
                if _should_skip_node(dep.source, ignored_set) or _should_skip_node(dep.target, ignored_set):
                    continue
                src_id, tgt_id = node_ids.get(dep.source), node_ids.get(dep.target)
                if src_id is None or tgt_id is None:
                    continue
                ck = canonical_key(src_id, tgt_id)
                key, is_fwd = (ck[0], ck[1]), ck[2]
                if key not in edge_registry:
                    edge_registry[key] = {'computational': set(), 'dependencies': []}
                dep_type = _get_dependency_type(dep) if differentiate_dependencies else "default"
                edge_registry[key]['dependencies'].append((dep_type, dep, is_fwd))

    return edge_registry


def _render_computational_edges(
    dot: graphviz.Digraph,
    edge_registry: Dict[tuple, Dict],
    visible: bool,
) -> None:
    """Render computational edges. Always constraint=true for layout backbone.

    Args:
        dot: The graphviz Digraph to add edges to.
        edge_registry: The unified edge registry.
        visible: If False, edges are rendered invisible (style=invis) but still constrain layout.
    """
    for (id1, id2), info in edge_registry.items():
        if not info['computational']:
            continue
        if visible:
            attrs = {"color": "#757575", "style": "solid", "constraint": "true"}
        else:
            attrs = {"style": "invis", "constraint": "true"}
        # Forward arrow: id1 → id2 if forward direction exists, else reverse
        if True in info['computational']:
            attrs["arrowhead"] = "normal"
            dot.edge(id1, id2, **attrs)
        else:
            attrs["arrowhead"] = "normal"
            dot.edge(id2, id1, **attrs)


def _render_dependency_edges(
    dot: graphviz.Digraph,
    edge_registry: Dict[tuple, Dict],
    differentiate: bool,
    show_labels: bool,
) -> None:
    """Render dependency edges. Always constraint=false to avoid disturbing layout.

    Args:
        dot: The graphviz Digraph to add edges to.
        edge_registry: The unified edge registry.
        differentiate: Style direct (green) vs force-shape-match (red) differently.
        show_labels: Whether to show trigger/handler labels on edges.
    """
    for (id1, id2), info in edge_registry.items():
        if not info['dependencies']:
            continue

        type_dirs: Dict[str, set] = {}
        type_dep: Dict[str, Any] = {}
        for dt, dep, is_fwd in info['dependencies']:
            if dt not in type_dirs:
                type_dirs[dt] = set()
                type_dep[dt] = dep
            type_dirs[dt].add(is_fwd)

        for dt, dirs in type_dirs.items():
            dep = type_dep[dt]
            if differentiate:
                if dt == "direct":
                    attrs = {"color": "#2E7D32", "style": "dashed", "penwidth": "1.5", "constraint": "false"}
                else:
                    attrs = {"color": "#C62828", "style": "dotted", "penwidth": "2", "constraint": "false"}
            else:
                attrs = {"color": "#E53935", "style": "dashed", "constraint": "false"}

            # Undirected: no arrowheads, just a line showing coupling
            attrs.update({"dir": "none", "arrowhead": "none", "arrowtail": "none"})

            if show_labels:
                attrs["xlabel"] = f"{_get_short_fn_name(dep.trigger)}→{_get_short_fn_name(dep.handler)}"
                attrs["fontcolor"] = attrs["color"]
            dot.edge(id1, id2, **attrs)


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
    group_root_types=None,
    ignored_layers: Optional[List] = None,
) -> graphviz.Digraph:
    """Visualize the dependency graph using Graphviz.

    All 3 views share the same layout backbone: computational edges always have
    constraint=true. Views differ only in edge visibility:
    - "computational": comp edges visible, no dep edges
    - "dependency": comp edges invisible (constraint=true), dep edges visible (constraint=false)
    - "both": comp edges visible (lighter), dep edges visible (constraint=false)

    When show_groups=True, nodes are wrapped in graphviz cluster subgraphs.

    Args:
        DG: The DependencyGraph to visualize.
        output_path: Path to save the output file (without extension).
        view: "computational", "dependency", or "both".
        format: Output format ("png", "svg", "pdf").
        show_channels: Whether to show input/output channel counts on nodes.
        show_edge_labels: Whether to show trigger/handler names on dependency edges.
        rankdir: Graph direction ("TB" for top-bottom, "LR" for left-right).
        highlight_modules: List of modules to highlight with a thicker border.
        ignored_types: List of OPTYPE values to hide from the graph.
        show_groups: Whether to wrap nodes in cluster subgraphs by pruning group.
        differentiate_dependencies: Style direct (green) vs force-shape-match (red) differently.
        group_root_types: Module types to use as group roots (default: Conv2d, Linear).

    Returns:
        The graphviz.Digraph object.

    Example:
        >>> import torch_pruning as tp
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> DG = tp.DependencyGraph().build_dependency(model, torch.randn(1,3,224,224))
        >>> tp.utils.visualize_graph(DG, "./graph", view="dependency", show_groups=True)
    """
    if view not in ("computational", "dependency", "both"):
        raise ValueError(f"Invalid view mode: {view}. Must be 'computational', 'dependency', or 'both'")

    ignored_set = set(ignored_types) if ignored_types else None
    highlight_set = set(highlight_modules) if highlight_modules else set()

    dot = graphviz.Digraph(
        name="DependencyGraph",
        format=format,
        graph_attr={
            "rankdir": rankdir,
            "splines": "polyline",
            "nodesep": "0.5",
            "ranksep": "0.5",
            "compound": "true",
            "newrank": "true",
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

    # Build node ID mapping
    node_ids: Dict[Any, str] = {}
    for i, node in enumerate(DG.module2node.values()):
        node_ids[node] = f"node_{i}"

    # Build group assignments
    node_id_to_group: Dict[str, int] = {}
    multi_group_nodes: Set[str] = set()
    num_groups = 0
    if show_groups:
        node_id_to_group, multi_group_nodes, num_groups = _build_group_assignment(
            DG, node_ids, ignored_set, group_root_types, ignored_layers
        )

    # Collect edges
    collect_deps = view in ("dependency", "both")
    edge_registry = _collect_edges(DG, node_ids, ignored_set, collect_deps, differentiate_dependencies)

    # Build per-group node lists and ungrouped list
    group_nodes: Dict[int, list] = {i: [] for i in range(num_groups)}
    ungrouped_nodes: list = []
    for node in DG.module2node.values():
        if _should_skip_node(node, ignored_set):
            continue
        nid = node_ids[node]
        if nid in node_id_to_group:
            group_nodes[node_id_to_group[nid]].append(node)
        else:
            ungrouped_nodes.append(node)

    def _node_attrs(node, nid):
        """Compute node attributes."""
        is_multi = nid in multi_group_nodes
        if node.module in highlight_set:
            border_color = "black"
            border_width = "3"
        else:
            border_color = "black"
            border_width = "1"
        peripheries = "2" if is_multi else "1"
        return {
            "label": _get_node_label(node, show_channels, DG),
            "fillcolor": _get_node_color(node.type),
            "color": border_color,
            "penwidth": border_width,
            "peripheries": peripheries,
        }

    # Add nodes inside cluster subgraphs (for groups) or at top level (ungrouped)
    show_cluster_boxes = view in ("dependency", "both")
    for group_idx in range(num_groups):
        nodes_in_group = group_nodes[group_idx]
        if not nodes_in_group:
            continue
        with dot.subgraph(name=f"cluster_{group_idx}") as sub:
            if show_cluster_boxes:
                sub.attr(
                    style="filled,rounded",
                    fillcolor=GROUP_FILL,
                    color=GROUP_BORDER,
                    penwidth=GROUP_PENWIDTH,
                    label=f"Group {group_idx}",
                    fontname="Helvetica",
                    fontsize="11",
                    fontcolor="#616161",
                )
            else:
                # Computational view: invisible clusters to preserve layout
                sub.attr(
                    style="",
                    color="invis",
                    label="",
                )
            for node in nodes_in_group:
                nid = node_ids[node]
                sub.node(nid, **_node_attrs(node, nid))

    # Ungrouped nodes at top level
    for node in ungrouped_nodes:
        nid = node_ids[node]
        dot.node(nid, **_node_attrs(node, nid))

    # Render edges based on view
    if view == "computational":
        _render_computational_edges(dot, edge_registry, visible=True)
    elif view == "dependency":
        _render_computational_edges(dot, edge_registry, visible=False)
        _render_dependency_edges(dot, edge_registry, differentiate_dependencies, show_edge_labels)
    else:  # "both"
        _render_computational_edges(dot, edge_registry, visible=True)
        _render_dependency_edges(dot, edge_registry, differentiate_dependencies, show_edge_labels)

    # Render to file
    if output_path is not None:
        if output_path.endswith(f".{format}"):
            output_path = output_path[:-len(format)-1]
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        dot.render(output_path, cleanup=True)

    return dot


def visualize_all_views(
    DG: "DependencyGraph",
    output_dir: str,
    basename: str = "graph",
    format: str = "png",
    show_channels: bool = True,
    rankdir: str = "TB",
    ignored_types: Optional[List[OPTYPE]] = None,
    group_root_types=None,
    differentiate_dependencies: bool = True,
    show_edge_labels: bool = False,
    ignored_layers: Optional[List] = None,
) -> Dict[str, graphviz.Digraph]:
    """Generate computational, dependency, and combined views with consistent layout.

    All 3 views share the same layout backbone (computational edges with constraint=true),
    so node positions are identical across views.

    Args:
        DG: The DependencyGraph to visualize.
        output_dir: Directory to save output files.
        basename: Base filename (produces {basename}_computational, etc.).
        format: Output format ("png", "svg", "pdf").
        show_channels: Whether to show input/output channel counts.
        rankdir: Graph direction ("TB" or "LR").
        ignored_types: OPTYPE values to hide.
        group_root_types: Module types for group roots (default: Conv2d, Linear).
        differentiate_dependencies: Style direct vs force-shape-match differently.
        show_edge_labels: Whether to show trigger/handler labels on dependency edges.
        ignored_layers: Layers to exclude from group assignment.

    Returns:
        Dict mapping view name to its graphviz.Digraph object.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for view_name in ("computational", "dependency", "both"):
        output_path = os.path.join(output_dir, f"{basename}_{view_name}")
        results[view_name] = visualize_graph(
            DG,
            output_path=output_path,
            view=view_name,
            format=format,
            show_channels=show_channels,
            show_edge_labels=show_edge_labels,
            rankdir=rankdir,
            ignored_types=ignored_types,
            show_groups=True,
            differentiate_dependencies=differentiate_dependencies,
            group_root_types=group_root_types,
            ignored_layers=ignored_layers,
        )
    return results
