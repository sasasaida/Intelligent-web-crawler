"""Calculate PageRank scores for a directed graph stored sparsely."""

from numbers import Integral, Real

import numpy as np
from scipy.sparse import diags, issparse


def calculate_pagerank(
    adjacency,
    damping_factor=0.85,
    tolerance=1e-6,
    max_iterations=100,
) -> np.ndarray:
    """Return one normalized PageRank score for each node.

    The adjacency matrix uses rows for source nodes and columns for destination
    nodes. Each iteration follows outgoing links, shares rank from dangling
    (zero-out-degree) nodes evenly, and applies the damping factor. Sparse
    operations are used throughout; the adjacency matrix is never made dense.

    Args:
        adjacency: A square SciPy sparse matrix with non-negative real weights.
        damping_factor: Probability of following a link, in [0, 1).
        tolerance: Stop when the L1 score change is smaller than this value.
        max_iterations: Maximum number of iterations; must be a positive int.

    Returns:
        A float NumPy array with one non-negative score per node. The scores
        sum to approximately one for a non-empty graph. An empty graph returns
        an empty array.

    Raises:
        TypeError: If the adjacency is not sparse or a parameter has an
            unsupported type.
        ValueError: If the adjacency or parameter values are invalid.
    """
    if not issparse(adjacency):
        raise TypeError("adjacency must be a SciPy sparse matrix")
    if len(adjacency.shape) != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    if isinstance(damping_factor, bool) or not isinstance(damping_factor, Real):
        raise TypeError("damping_factor must be a real number")
    if not np.isfinite(damping_factor) or not 0 <= damping_factor < 1:
        raise ValueError("damping_factor must be in the range [0, 1)")

    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number")

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    num_nodes = adjacency.shape[0]
    if num_nodes == 0:
        return np.empty(0, dtype=np.float64)

    if adjacency.dtype.kind not in "biuf":
        raise ValueError("adjacency values must be real numeric values")

    # Check stored values before any format conversion can combine duplicate
    # coordinates and hide a negative entry.
    stored_data = getattr(adjacency, "data", None)
    if stored_data is None or np.asarray(stored_data).dtype.kind not in "biuf":
        stored_data = adjacency.tocoo(copy=False).data
    if not np.isfinite(stored_data).all():
        raise ValueError("adjacency values must be finite")
    if np.any(stored_data < 0):
        raise ValueError("adjacency values must be non-negative")

    # Convert a copy to float64 before CSR conversion or duplicate merging,
    # preventing integer overflow while preserving the caller's matrix.
    graph = adjacency.astype(np.float64, copy=True).tocsr()
    graph.sum_duplicates()
    graph.eliminate_zeros()
    if not np.isfinite(graph.data).all():
        raise ValueError("adjacency values must be finite")

    out_degree = np.asarray(graph.sum(axis=1)).ravel()
    dangling_nodes = out_degree == 0
    inverse_out_degree = np.zeros(num_nodes, dtype=np.float64)
    inverse_out_degree[~dangling_nodes] = 1.0 / out_degree[~dangling_nodes]

    # Normalize each source row, then transpose so multiplying by the score
    # vector accumulates each node's incoming rank.
    transition_transposed = (
        diags(inverse_out_degree) @ graph
    ).transpose().tocsr()

    scores = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    uniform_jump = (1.0 - damping_factor) / num_nodes

    for _ in range(max_iterations):
        dangling_rank = scores[dangling_nodes].sum()
        updated_scores = (
            damping_factor * (transition_transposed @ scores)
            + uniform_jump
            + damping_factor * dangling_rank / num_nodes
        )
        difference = np.abs(updated_scores - scores).sum()
        scores = np.asarray(updated_scores).ravel()
        if difference < tolerance:
            break

    return scores