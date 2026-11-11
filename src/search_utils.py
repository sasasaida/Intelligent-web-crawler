"""
Shared helper functions for the UCS and A*-inspired crawler modules.

The graph is expected to follow the project interface:
    adjacency, node_ids, node_to_index, in_degree, out_degree, pagerank
"""

import math
import time


def validate_inputs(graph_data, start_node_index, budget):
    """Check the shared graph interface and the requested crawl limits."""
    required = ("adjacency", "node_ids")
    for key in required:
        if key not in graph_data:
            raise KeyError(f"graph_data is missing required key: {key}")

    node_count = len(graph_data["node_ids"])
    if not isinstance(start_node_index, int):
        raise TypeError("start_node_index must be an integer.")
    if not 0 <= start_node_index < node_count:
        raise IndexError("start_node_index is outside the graph.")
    if not isinstance(budget, int):
        raise TypeError("budget must be an integer.")
    if budget < 1:
        raise ValueError("budget must be at least 1 because the start node is visited.")

    adjacency = graph_data["adjacency"]
    if getattr(adjacency, "shape", (node_count, node_count)) != (node_count, node_count):
        raise ValueError("adjacency must have shape (number_of_nodes, number_of_nodes).")


def get_outgoing_neighbors(adjacency, node_index):
    """Return the destination indices of outgoing directed edges."""
    # Efficient path for a SciPy CSR matrix.
    if hasattr(adjacency, "indptr") and hasattr(adjacency, "indices"):
        start = adjacency.indptr[node_index]
        end = adjacency.indptr[node_index + 1]
        return adjacency.indices[start:end]

    # NumPy dense matrix.
    if hasattr(adjacency, "shape"):
        row = adjacency[node_index]
        try:
            import numpy as np
            return np.flatnonzero(row)
        except ImportError:
            return [i for i, value in enumerate(row) if value != 0]

    # Simple adjacency-list fallback for small tests.
    return adjacency[node_index]


def build_results(graph_data, visited, discovered, path_costs, started_at):
    """Build the exact result structure shared by both search algorithms."""
    node_ids = graph_data["node_ids"]
    visited_list = list(visited)
    discovered_list = list(discovered)

    return {
        "visited_node_indices": visited_list,
        "visited_node_ids": [node_ids[index] for index in visited_list],
        "discovered_node_indices": discovered_list,
        "total_cost": sum(path_costs[index] for index in visited_list),
        "execution_time": time.perf_counter() - started_at,
    }


def importance_scores(graph_data):
    """
    Calculate a lightweight, normalized importance score for each node.

    This does NOT calculate PageRank. It uses the scores and degrees supplied
    by Person 1. Log scaling reduces the influence of extreme degree values.
    Missing optional features are treated as zero.
    """
    node_count = len(graph_data["node_ids"])

    def get_feature(name):
        values = graph_data.get(name)
        if values is None:
            return [0.0] * node_count
        if len(values) != node_count:
            raise ValueError(f"{name} must contain one value per node.")
        cleaned = []
        for value in values:
            number = float(value)
            if not math.isfinite(number) or number < 0:
                number = 0.0
            cleaned.append(number)
        return cleaned

    pagerank = get_feature("pagerank")
    in_degree = get_feature("in_degree")
    out_degree = get_feature("out_degree")
    # Balanced connectivity proxy: nodes with both incoming and outgoing
    # links score higher than nodes with only one type of connection.
    connectivity = [math.sqrt(i * o) for i, o in zip(in_degree, out_degree)]

    def normalize(values):
        transformed = [math.log1p(value) for value in values]
        maximum = max(transformed, default=0.0)
        if maximum == 0:
            return [0.0] * node_count
        return [value / maximum for value in transformed]

    pr_norm = normalize(pagerank)
    in_norm = normalize(in_degree)
    out_norm = normalize(out_degree)
    conn_norm = normalize(connectivity)

    return [
        0.40 * pr + 0.20 * indeg + 0.20 * outdeg + 0.20 * conn
        for pr, indeg, outdeg, conn in zip(pr_norm, in_norm, out_norm, conn_norm)
    ]
