"""
Visualize a CESM energy system network from a DuckDB database.

Reads CESM DataFrames from a DuckDB file and produces a PNG network
diagram showing balances, storages, commodities, units, links, and
port connections.

Usage:
    python scripts/visualization/visualize_network.py data/cesm.duckdb output.png
"""

import argparse

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from readers.from_duckdb import dataframes_from_duckdb

# Use non-interactive backend so the script works headless
matplotlib.use("Agg")

# -- visual constants --------------------------------------------------------

NODE_STYLES = {
    "balance": {"color": "#ADD8E6", "shape": "o", "size": 200},
    "storage": {"color": "#90EE90", "shape": "s", "size": 200},
    "commodity": {"color": "#FFFFB0", "shape": "D", "size": 160},
    "unit": {"color": "#FFD8A8", "shape": "p", "size": 120},
}

LINK_STYLE = {"color": "#999999", "style": "dashed", "width": 1.5}
PORT_STYLE = {"color": "#555555", "style": "solid", "width": 1.2}


# -- helpers -----------------------------------------------------------------

def _names_from_df(df):
    """Return entity names from a DataFrame index.

    CESM DataFrames use the index for entity identifiers.  Single-index
    DataFrames have a plain Index (the name column), while relationship
    tables (ports, links) use a MultiIndex where level 0 is the composite
    name and the remaining levels hold the dimension columns (source/sink
    or node_A/node_B).
    """
    if df is None or df.empty:
        return []
    idx = df.index
    if hasattr(idx, "levels"):
        # MultiIndex – level 0 is the composite name
        return idx.get_level_values(0).tolist()
    return idx.tolist()


def _get_index_level(df, level_name):
    """Get values from a named MultiIndex level, or None if not present."""
    if df is None or df.empty:
        return []
    idx = df.index
    if hasattr(idx, "names") and level_name in idx.names:
        return idx.get_level_values(level_name).tolist()
    return []


def _build_graph(dataframes):
    """Build a networkx DiGraph from CESM DataFrames."""
    G = nx.DiGraph()

    # Add balance nodes
    for name in _names_from_df(dataframes.get("balance")):
        G.add_node(name, entity_type="balance")

    # Add storage nodes
    for name in _names_from_df(dataframes.get("storage")):
        G.add_node(name, entity_type="storage")

    # Add commodity nodes
    for name in _names_from_df(dataframes.get("commodity")):
        G.add_node(name, entity_type="commodity")

    # Add unit nodes
    for name in _names_from_df(dataframes.get("unit")):
        G.add_node(name, entity_type="unit")

    # Add port connections: node_to_unit (source=node -> sink=unit)
    ntu = dataframes.get("node_to_unit")
    if ntu is not None and not ntu.empty:
        sources = _get_index_level(ntu, "source")
        sinks = _get_index_level(ntu, "sink")
        for src, snk in zip(sources, sinks):
            if src and snk:
                if src not in G:
                    G.add_node(src, entity_type="commodity")
                if snk not in G:
                    G.add_node(snk, entity_type="unit")
                G.add_edge(src, snk, edge_type="port")

    # Add port connections: unit_to_node (source=unit -> sink=node)
    utn = dataframes.get("unit_to_node")
    if utn is not None and not utn.empty:
        sources = _get_index_level(utn, "source")
        sinks = _get_index_level(utn, "sink")
        for src, snk in zip(sources, sinks):
            if src and snk:
                if src not in G:
                    G.add_node(src, entity_type="unit")
                if snk not in G:
                    G.add_node(snk, entity_type="balance")
                G.add_edge(src, snk, edge_type="port")

    # Add link edges (bidirectional conceptually, drawn as undirected)
    link_df = dataframes.get("link")
    if link_df is not None and not link_df.empty:
        nodes_a = _get_index_level(link_df, "node_A")
        nodes_b = _get_index_level(link_df, "node_B")
        for node_a, node_b in zip(nodes_a, nodes_b):
            if node_a and node_b:
                for n in (node_a, node_b):
                    if n not in G:
                        G.add_node(n, entity_type="balance")
                G.add_edge(node_a, node_b, edge_type="link")

    return G


def _repel_overlaps(pos, min_dist=0.12, iterations=100):
    """Push apart nodes that are closer than *min_dist*.

    This is a simple force-directed post-processing step that only applies
    repulsion (no attraction), so it preserves the overall layout shape while
    eliminating overlaps.
    """
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes], dtype=float)

    for _ in range(iterations):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                delta = coords[i] - coords[j]
                dist = np.linalg.norm(delta)
                if dist < min_dist and dist > 0:
                    # Push both nodes apart along the connecting vector
                    push = (min_dist - dist) / 2.0 * (delta / dist)
                    coords[i] += push
                    coords[j] -= push
                    moved = True
                elif dist == 0:
                    # Identical positions — nudge randomly
                    nudge = np.random.default_rng(42).uniform(-min_dist, min_dist, 2)
                    coords[i] += nudge
                    moved = True
        if not moved:
            break

    return {n: coords[i] for i, n in enumerate(nodes)}


def _draw_graph(G, output_path):
    """Layout and render the graph to a PNG file."""
    if len(G) == 0:
        print("Warning: graph has no nodes, producing empty image.")

    n = max(len(G), 1)

    # Scale figure size with number of nodes
    scale = max(12, min(28, n * 0.15))
    fig, ax = plt.subplots(figsize=(scale, scale * 0.75))
    ax.set_title("CESM Energy System Network", fontsize=14, fontweight="bold", pad=12)

    # --- Layout ---
    # Kamada-Kawai produces more uniform edge lengths than spring layout.
    # For very large graphs (>200 nodes) fall back to spring which is faster.
    if n <= 200:
        pos = nx.kamada_kawai_layout(G, scale=2.0)
    else:
        k = max(1.0, 3.0 / (n ** 0.3))
        pos = nx.spring_layout(G, k=k, iterations=200, seed=42)

    # Post-process: push apart any nodes that are too close.
    # min_dist is relative to the layout scale.
    min_dist = max(0.08, 2.0 / max(n, 1))
    pos = _repel_overlaps(pos, min_dist=min_dist, iterations=150)

    # -- draw edges first (underneath nodes) ---------------------------------

    port_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "port"]
    link_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "link"]

    # Links: dashed gray, no arrows
    nx.draw_networkx_edges(
        G, pos, edgelist=link_edges, ax=ax,
        style="dashed", edge_color=LINK_STYLE["color"],
        width=LINK_STYLE["width"], arrows=False,
    )

    # Port connections: solid arrows with curved paths to reduce overlap
    nx.draw_networkx_edges(
        G, pos, edgelist=port_edges, ax=ax,
        style="solid", edge_color=PORT_STYLE["color"],
        width=PORT_STYLE["width"], arrows=True,
        arrowstyle="-|>", arrowsize=12,
        connectionstyle="arc3,rad=0.12",
    )

    # -- draw nodes per entity type ------------------------------------------

    for etype, style in NODE_STYLES.items():
        nodelist = [nd for nd, d in G.nodes(data=True) if d.get("entity_type") == etype]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist, ax=ax,
            node_color=style["color"],
            node_shape=style["shape"],
            node_size=style["size"],
            edgecolors="#333333",
            linewidths=0.8,
        )

    # -- labels (offset slightly above each node to avoid overlap) -----------

    font_size = max(5, min(9, 250 / max(n, 1)))
    label_pos = {k: (v[0], v[1] + min_dist * 0.35) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, ax=ax, font_size=font_size)

    # -- legend --------------------------------------------------------------

    legend_handles = [
        mpatches.Patch(facecolor=s["color"], edgecolor="#333", label=etype.capitalize())
        for etype, s in NODE_STYLES.items()
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color=LINK_STYLE["color"], linestyle="dashed", linewidth=1.5, label="Link")
    )
    legend_handles.append(
        plt.Line2D([0], [0], color=PORT_STYLE["color"], linestyle="solid", linewidth=1.2, label="Port connection")
    )
    ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.9)

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Network diagram saved to {output_path}")


# -- main --------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize a CESM energy system network from a DuckDB database"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input DuckDB file path",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output PNG file path",
    )
    args = parser.parse_args()

    # Read CESM DataFrames
    print(f"Loading CESM data from {args.input}...")
    dataframes = dataframes_from_duckdb(args.input)

    # Build and draw graph
    print("Building network graph...")
    G = _build_graph(dataframes)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    _draw_graph(G, args.output)


if __name__ == "__main__":
    main()
