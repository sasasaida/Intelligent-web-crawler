"""Build sparse graph structures and calculate directed graph statistics."""

from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from src.data_loader import load_edge_list
from src.pagerank import calculate_pagerank

PathType = Union[str, Path]
Edge = Tuple[int, int]
GraphStatistics = Dict[str, Union[int, float, np.ndarray]]


def build_adjacency_matrix(
    edges: Sequence[Edge], num_nodes: int
) -> csr_matrix:
    """Build a CSR matrix with source rows and destination columns.

    Edge endpoints are internal node indices. Duplicate edges are collapsed
    and stored as a single 1. Coordinate arrays are created directly from the
    edge sequence to avoid building additional Python lists.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")

    edge_count = len(edges)
    if edge_count == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=np.int32)

    sources = np.fromiter(
        (source for source, _ in edges), dtype=np.int32, count=edge_count
    )
    destinations = np.fromiter(
        (destination for _, destination in edges),
        dtype=np.int32,
        count=edge_count,
    )
    values = np.ones(edge_count, dtype=np.int32)

    adjacency = coo_matrix(
        (values, (sources, destinations)),
        shape=(num_nodes, num_nodes),
        dtype=np.int32,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data.fill(1)
    return adjacency


def calculate_graph_statistics(adjacency: csr_matrix) -> GraphStatistics:
    """Calculate node, edge, and degree statistics for a sparse graph."""
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    num_nodes = adjacency.shape[0]
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    num_edges = int(adjacency.nnz)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_in_degree": (
            float(in_degree.mean()) if num_nodes else 0.0
        ),
        "average_out_degree": (
            float(out_degree.mean()) if num_nodes else 0.0
        ),
        "average_total_degree": (
            float((in_degree.sum() + out_degree.sum()) / num_nodes)
            if num_nodes
            else 0.0
        ),
        "max_in_degree": int(in_degree.max()) if num_nodes else 0,
        "max_out_degree": int(out_degree.max()) if num_nodes else 0,
        "in_degree": in_degree,
        "out_degree": out_degree,
    }


def process_graph(
    dataset_path: PathType,
) -> Tuple[
    csr_matrix,
    np.ndarray,
    Dict[int, int],
    np.ndarray,
    np.ndarray,
    GraphStatistics,
]:
    """Load a dataset and return its sparse graph, IDs, degrees, and stats."""
    edges, original_node_ids, node_to_index = load_edge_list(dataset_path)
    node_ids = np.asarray(original_node_ids)
    adjacency = build_adjacency_matrix(edges, len(node_ids))
    statistics = calculate_graph_statistics(adjacency)

    return (
        adjacency,
        node_ids,
        node_to_index,
        statistics["in_degree"],
        statistics["out_degree"],
        statistics,
    )


def load_graph_data(dataset_path: PathType) -> Dict[str, object]:
    """Load graph data and return aligned IDs, degrees, adjacency, and PageRank.

    The loader's sorted original node IDs define the internal ordering used by
    the adjacency matrix, degree arrays, and PageRank scores.
    """
    edges, original_node_ids, node_to_index = load_edge_list(dataset_path)
    node_ids = np.asarray(original_node_ids)
    adjacency = build_adjacency_matrix(edges, len(node_ids))
    statistics = calculate_graph_statistics(adjacency)
    pagerank_scores = calculate_pagerank(adjacency)

    return {
        "adjacency": adjacency,
        "node_ids": node_ids,
        "node_to_index": node_to_index,
        "in_degree": statistics["in_degree"],
        "out_degree": statistics["out_degree"],
        "pagerank": pagerank_scores,
    }