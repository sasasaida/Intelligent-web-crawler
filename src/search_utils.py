
"""
Shared helper functions for the UCS and A*-inspired crawler modules.

The graph is expected to follow the project interface:
    adjacency, node_ids, node_to_index, in_degree, out_degree, pagerank
"""

import time
import numpy as np


# Private key used to store precomputed importance scores in graph_data.
IMPORTANCE_CACHE_KEY = "_importance_scores_cache"


def validate_inputs(graph_data, start_node_index, budget):
    """Check the shared graph interface and requested crawl limits."""

    required = ("adjacency", "node_ids")

    for key in required:
        if key not in graph_data:
            raise KeyError(f"graph_data is missing required key: {key}")

    node_count = len(graph_data["node_ids"])

    if not isinstance(start_node_index, (int, np.integer)):
        raise TypeError("start_node_index must be an integer.")

    if not 0 <= start_node_index < node_count:
        raise IndexError("start_node_index is outside the graph.")

    if not isinstance(budget, (int, np.integer)):
        raise TypeError("budget must be an integer.")

    if budget < 1:
        raise ValueError(
            "budget must be at least 1 because the start node is visited."
        )

    adjacency = graph_data["adjacency"]

    if getattr(adjacency, "shape", (node_count, node_count)) != (
        node_count,
        node_count,
    ):
        raise ValueError(
            "adjacency must have shape "
            "(number_of_nodes, number_of_nodes)."
        )


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
        return np.flatnonzero(row)

    # Simple adjacency-list fallback for small tests.
    return adjacency[node_index]


def build_results(graph_data, visited, discovered, path_costs, started_at):
    """Build the shared result structure."""

    node_ids = graph_data["node_ids"]

    visited_list = list(visited)
    discovered_list = list(discovered)

    return {
        "visited_node_indices": visited_list,
        "visited_node_ids": [
            node_ids[index] for index in visited_list
        ],
        "discovered_node_indices": discovered_list,
        "total_cost": sum(
            path_costs[index] for index in visited_list
        ),
        "execution_time": time.perf_counter() - started_at,
    }


def importance_scores(graph_data, use_cache=True):
    """
    Calculate normalized importance scores using NumPy.

    Formula:
        importance =
            0.40 * normalized PageRank
          + 0.20 * normalized in-degree
          + 0.20 * normalized out-degree
          + 0.20 * normalized balanced connectivity

    Each feature uses log1p scaling followed by maximum normalization.

    Results are cached in graph_data so repeated A* runs can reuse them.

    Set use_cache=False to force recalculation.
    """

    node_count = len(graph_data["node_ids"])

    # Reuse previously calculated scores when available.
    if use_cache:
        cached_scores = graph_data.get(IMPORTANCE_CACHE_KEY)

        if (
            isinstance(cached_scores, np.ndarray)
            and cached_scores.shape == (node_count,)
        ):
            return cached_scores

    def get_feature(name):
        """Read, validate, and clean one node feature."""

        values = graph_data.get(name)

        if values is None:
            return np.zeros(node_count, dtype=np.float64)

        array = np.asarray(values, dtype=np.float64)

        if array.ndim != 1 or len(array) != node_count:
            raise ValueError(
                f"{name} must contain one value per node."
            )

        # Replace negative, infinite, and NaN values with zero.
        return np.where(
            np.isfinite(array) & (array >= 0),
            array,
            0.0,
        )

    def normalize(values):
        """Apply log scaling and maximum normalization."""

        transformed = np.log1p(values)

        maximum = np.max(transformed, initial=0.0)

        if maximum == 0:
            return np.zeros(node_count, dtype=np.float64)

        return transformed / maximum

    # Read the four features.
    pagerank = get_feature("pagerank")
    in_degree = get_feature("in_degree")
    out_degree = get_feature("out_degree")

    # Balanced connectivity proxy.
    connectivity = np.sqrt(in_degree * out_degree)

    # Normalize each feature.
    pr_norm = normalize(pagerank)
    in_norm = normalize(in_degree)
    out_norm = normalize(out_degree)
    conn_norm = normalize(connectivity)

    # Combine the normalized features.
    importance = (
        0.40 * pr_norm
        + 0.20 * in_norm
        + 0.20 * out_norm
        + 0.20 * conn_norm
    )

    # Cache the result for future A* runs.
    if use_cache:
        graph_data[IMPORTANCE_CACHE_KEY] = importance

    return importance
