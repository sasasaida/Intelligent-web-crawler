
from src.graph_processor import load_graph_data
from src.ucs import run_ucs
from src.astar import run_astar
from src.evaluation import (
    evaluate_results,
    measure_peak_memory
)

# Experiment settings
START_NODE = 0
BUDGET = 1000

print("Loading Web-Google graph...")

graph_data = load_graph_data("data/web-Google.txt")

print("Graph loaded!")
print("Total nodes:", len(graph_data["node_ids"]))
print("Starting node:", START_NODE)
print("Exploration budget:", BUDGET)

# -----------------------------
# Run UCS
# -----------------------------

print("\nRunning UCS...")

with measure_peak_memory() as ucs_memory:
    ucs_results = run_ucs(
        graph_data,
        START_NODE,
        BUDGET
    )

ucs_results["peak_memory_mb"] = ucs_memory.peak_memory_mb

# -----------------------------
# Run A*
# -----------------------------

print("\nRunning A*...")

with measure_peak_memory() as astar_memory:
    astar_results = run_astar(
        graph_data,
        START_NODE,
        BUDGET
    )

astar_results["peak_memory_mb"] = astar_memory.peak_memory_mb

# -----------------------------
# Evaluate both algorithms
# -----------------------------

print("\nCalculating evaluation metrics...")

comparison = evaluate_results(
    graph_data,
    ucs_results,
    astar_results,
    top_percent=1.0
)

print("\n========== REAL EXPERIMENT RESULTS ==========")

print(comparison.to_string(index=False))

# -----------------------------
# Save results
# -----------------------------

from pathlib import Path

output_path = Path("results") / "real_experiment.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

comparison.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
