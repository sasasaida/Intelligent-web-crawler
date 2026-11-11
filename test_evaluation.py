
import numpy as np
from scipy.sparse import csr_matrix

from src.evaluation import (
    evaluate_results,
    evaluate_budget_experiments
)

# Create a small directed graph with 4 nodes
edges = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3)
]

rows, cols = zip(*edges)

adjacency = csr_matrix(
    (np.ones(len(edges)), (rows, cols)),
    shape=(4, 4)
)

# PageRank scores
pagerank = np.array([0.4, 0.3, 0.2, 0.1])

graph_data = {
    "adjacency": adjacency,
    "node_ids": np.array([100, 101, 102, 103]),
    "node_to_index": {
        100: 0,
        101: 1,
        102: 2,
        103: 3
    },
    "in_degree": np.asarray(
        adjacency.sum(axis=0)
    ).ravel(),
    "out_degree": np.asarray(
        adjacency.sum(axis=1)
    ).ravel(),
    "pagerank": pagerank
}

# Simulated UCS results
ucs_results = {
    "visited_node_indices": [1, 2],
    "execution_time": 0.010,
    "peak_memory_mb": 100.0
}

# Simulated A* results
astar_results = {
    "visited_node_indices": [0, 2],
    "execution_time": 0.015,
    "peak_memory_mb": 105.0
}

# Compare both algorithms
comparison = evaluate_results(
    graph_data,
    ucs_results,
    astar_results,
    top_percent=1.0
)

print("===== EVALUATION TEST =====")
print(comparison.to_string(index=False))

# Test multiple budgets
experiment_results = {
    2: {
        "ucs": ucs_results,
        "astar": astar_results
    },
    3: {
        "ucs": {
            "visited_node_indices": [1, 2, 3],
            "execution_time": 0.012,
            "peak_memory_mb": 101.0
        },
        "astar": {
            "visited_node_indices": [0, 1, 2],
            "execution_time": 0.017,
            "peak_memory_mb": 106.0
        }
    }
}

budget_comparison = evaluate_budget_experiments(
    graph_data,
    experiment_results,
    top_percent=1.0
)

print("\n===== BUDGET TEST =====")
print(budget_comparison.to_string(index=False))
