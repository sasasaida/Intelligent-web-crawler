
import numpy as np
from scipy.sparse import isspmatrix_csr

from src.graph_processor import load_graph_data
from src.ucs import run_ucs
from src.astar import run_astar


# STEP 1: Load the processed graph
print("Loading SNAP Web-Google graph...")

graph_data = load_graph_data("data/web-Google.txt")

print("\nGraph loaded successfully!")


# STEP 2: Verify graph structure
required_fields = [
    "adjacency",
    "node_ids",
    "node_to_index",
    "in_degree",
    "out_degree",
    "pagerank"
]

for field in required_fields:
    assert field in graph_data, f"Missing field: {field}"

adjacency = graph_data["adjacency"]
node_ids = graph_data["node_ids"]

assert isspmatrix_csr(adjacency), "Adjacency must be CSR"
assert adjacency.shape == (len(node_ids), len(node_ids))
assert len(graph_data["in_degree"]) == len(node_ids)
assert len(graph_data["out_degree"]) == len(node_ids)
assert len(graph_data["pagerank"]) == len(node_ids)

print("\nGraph structure verified!")
print("Nodes:", len(node_ids))
print("Edges:", adjacency.nnz)
print("PageRank sum:", graph_data["pagerank"].sum())


# STEP 3: Select a starting node
start_node_index = 0
budget = 1000

print("\nStarting node index:", start_node_index)
print("Original SNAP node ID:", node_ids[start_node_index])
print("Visit budget:", budget)


# STEP 4: Run UCS
print("\nRunning UCS...")

ucs_results = run_ucs(
    graph_data,
    start_node_index,
    budget
)


# STEP 5: Run A*
print("\nRunning A*...")

astar_results = run_astar(
    graph_data,
    start_node_index,
    budget
)


# STEP 6: Validate algorithm results
expected_keys = {
    "visited_node_indices",
    "visited_node_ids",
    "discovered_node_indices",
    "total_cost",
    "execution_time"
}

for name, results in [
    ("UCS", ucs_results),
    ("A*", astar_results)
]:
    assert expected_keys.issubset(results.keys())

    visited = results["visited_node_indices"]

    assert len(visited) <= budget
    assert len(visited) == len(set(visited))
    assert visited[0] == start_node_index

    assert len(results["visited_node_ids"]) == len(visited)

    print(f"\n{name} validation passed!")


# STEP 7: Display comparison
print("\n========== RESULTS ==========")

print("\nUCS")
print("Visited nodes:", len(ucs_results["visited_node_indices"]))
print("Discovered nodes:", len(ucs_results["discovered_node_indices"]))
print("Total cost:", ucs_results["total_cost"])
print("Execution time:", ucs_results["execution_time"])

print("\nA*")
print("Visited nodes:", len(astar_results["visited_node_indices"]))
print("Discovered nodes:", len(astar_results["discovered_node_indices"]))
print("Total cost:", astar_results["total_cost"])
print("Execution time:", astar_results["execution_time"])

print("\nAll integration tests passed!")



pagerank = graph_data["pagerank"]

ucs_visited = ucs_results["visited_node_indices"]
astar_visited = astar_results["visited_node_indices"]

ucs_importance = np.sum(pagerank[ucs_visited])
astar_importance = np.sum(pagerank[astar_visited])

print("\nImportant Page Discovery")

print("UCS PageRank importance:", ucs_importance)
print("A* PageRank importance:", astar_importance)