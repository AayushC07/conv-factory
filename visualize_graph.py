"""Visualize the tool graph built during the build stage."""

import json
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def visualize_tool_graph(graph_path: str, output_path: str = None, figsize: tuple = (20, 14)):
    """
    Load and visualize the tool graph.
    
    Args:
        graph_path: Path to tool_graph.json
        output_path: Optional path to save the visualization (PNG/PDF)
        figsize: Figure size (width, height)
    """
    # Load the graph
    with open(graph_path) as f:
        graph_data = json.load(f)
    
    # Reconstruct NetworkX graph from node-link format
    G = nx.node_link_graph(graph_data, directed=True)
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Extract tool names and categories for coloring
    tool_colors = {}
    color_map = plt.cm.Set3(range(20))
    
    unique_tools = set()
    
    for node, attrs in G.nodes(data=True):
        tool = attrs.get('tool_name', 'unknown')
        unique_tools.add(tool)
    
    # Assign colors
    for i, tool in enumerate(sorted(unique_tools)):
        tool_colors[tool] = color_map[i % len(color_map)]
    
    # Color nodes by tool
    node_colors = [tool_colors[G.nodes[node].get('tool_name', 'unknown')] for node in G.nodes()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout with spring algorithm (better for large graphs)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1',
        width=0.5,
        alpha=0.6,
        ax=ax
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.9,
        ax=ax
    )
    
    # Draw labels (simplified: endpoint name only)
    labels = {node: node.split('/')[-1][:12] for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=7,
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title(f"Tool Graph Visualization\n({G.number_of_nodes()} endpoints, {G.number_of_edges()} relationships)", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tool_colors[tool], label=tool) 
                       for tool in sorted(unique_tools)]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def print_graph_stats(graph_path: str, registry_path: str = None):
    """Print statistics about the graph."""
    with open(graph_path) as f:
        graph_data = json.load(f)
    
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    print("\n" + "="*60)
    print("TOOL GRAPH STATISTICS")
    print("="*60)
    
    # Basic stats
    total_edges = len(edges)
    print(f"\n📊 Basic Metrics:")
    print(f"   Nodes (endpoints): {len(nodes)}")
    print(f"   Total edges: {total_edges}")
    print(f"   Avg edges per node: {total_edges / len(nodes):.2f}" if len(nodes) > 0 else "   Avg edges per node: 0")
    
    # Tools and categories
    tools = defaultdict(list)
    categories = defaultdict(list)
    
    for node in nodes:
        if isinstance(node, dict):
            node_id = node.get('id')
            tool = node.get('tool_name')
            category = node.get('category')
        else:
            continue
        
        if tool:
            tools[tool].append(node_id)
        if category:
            categories[category].append(node_id)
    
    print(f"\n🔧 Tools ({len(tools)}):")
    for tool in sorted(tools.keys()):
        print(f"   {tool}: {len(tools[tool])} endpoints")
    
    print(f"\n📂 Categories ({len(categories)}):")
    for category in sorted(categories.keys()):
        print(f"   {category}: {len(categories[category])} endpoints")
    
    # Edge type breakdown
    edge_types = defaultdict(int)
    for edge in edges:
        if isinstance(edge, dict):
            relation = edge.get('relation', 'unknown')
            edge_types[relation] += 1
    
    print(f"\n🔗 Edge Types:")
    for edge_type in sorted(edge_types.keys()):
        print(f"   {edge_type}: {edge_types[edge_type]}")
    
    # Connectivity analysis
    out_degrees = defaultdict(int)
    for edge in edges:
        if isinstance(edge, dict):
            src = edge.get('source')
            if src:
                out_degrees[src] += 1
    
    degrees = list(out_degrees.values())
    print(f"\n📈 Connectivity:")
    print(f"   Max out-degree: {max(degrees) if degrees else 0}")
    print(f"   Min out-degree: {min(degrees) if degrees else 0}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the tool graph")
    parser.add_argument(
        "--graph",
        default="output/tool_graph.json",
        help="Path to tool_graph.json"
    )
    parser.add_argument(
        "--registry",
        default="output/registry.json",
        help="Path to registry.json (optional, for extra info)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for saving visualization (e.g., graph.png)"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[20, 14],
        help="Figure size (width height)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print stats only without visualization"
    )
    
    args = parser.parse_args()
    
    graph_path = Path(args.graph)
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_path}")
        print("   Run: convfactory build --data-dir sample_data --output-dir output")
        exit(1)
    
    # Print stats
    print_graph_stats(str(graph_path), args.registry)
    
    # Visualize
    if not args.stats_only:
        visualize_tool_graph(
            str(graph_path),
            output_path=args.output,
            figsize=tuple(args.figsize)
        )
