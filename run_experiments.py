from pathlib import Path

import numpy as np
import pandas as pd

from src.graph_processor import load_graph_data
from src.ucs import run_ucs
from src.astar import run_astar
from src.evaluation import evaluate_single_result, measure_peak_memory


# Experiment settings
BUDGETS = [100, 500, 1000, 5000, 10000]
REPETITIONS = 3
TOP_PERCENT = 1.0

print("Loading Web-Google graph...")
graph_data = load_graph_data("data/web-Google.txt")

node_count = len(graph_data["node_ids"])
out_degree = np.asarray(graph_data["out_degree"]).reshape(-1)

# Select three starting nodes with different positive out-degree levels.
positive_nodes = np.flatnonzero(out_degree > 0)
selected_nodes = []

for percentile in [25, 50, 90]:
    target = np.percentile(out_degree[positive_nodes], percentile)
    candidate = int(positive_nodes[np.argmin(
        np.abs(out_degree[positive_nodes] - target)
    )])
    if candidate not in selected_nodes:
        selected_nodes.append(candidate)

print("Graph loaded:", node_count, "nodes")
print("Starting nodes:", selected_nodes)
print("Budgets:", BUDGETS)
print("Repetitions:", REPETITIONS)

rows = []

for start_node in selected_nodes:
    for budget in BUDGETS:
        for repetition in range(1, REPETITIONS + 1):

            print(
                f"\nStart: {start_node} | Budget: {budget} "
                f"| Repetition: {repetition}"
            )

            print("Running UCS...")
            with measure_peak_memory() as memory:
                ucs_result = run_ucs(graph_data, start_node, budget)

            ucs_result["peak_memory_mb"] = memory.peak_memory_mb

            ucs_metrics = evaluate_single_result(
                graph_data, ucs_result, "UCS"
            )
            ucs_metrics.update({
                "Start Node": start_node,
                "Budget": budget,
                "Repetition": repetition,
            })
            rows.append(ucs_metrics)

            print("Running A*...")
            # Clear the cache so every A* run includes preprocessing.
            graph_data.pop("_importance_scores_cache", None)

            with measure_peak_memory() as memory:
                astar_result = run_astar(graph_data, start_node, budget)

            astar_result["peak_memory_mb"] = memory.peak_memory_mb

            astar_metrics = evaluate_single_result(
                graph_data, astar_result, "A*"
            )
            astar_metrics.update({
                "Start Node": start_node,
                "Budget": budget,
                "Repetition": repetition,
            })
            rows.append(astar_metrics)


# Save individual experiment results.
results = pd.DataFrame(rows)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

results.to_csv(output_dir / "repeated_experiments.csv", index=False)

# Calculate mean and standard deviation for each budget and algorithm.
metrics = [
    "Visited Webpages",
    "Important Webpages Visited",
    "PageRank Mass Captured",
    "PageRank Mass Captured (%)",
    "Graph Coverage (%)",
    "Execution Time (seconds)",
    "Peak Memory (MB)",
]

summary = results.groupby(
    ["Budget", "Algorithm"]
)[metrics].agg(["mean", "std"])

summary.to_csv(output_dir / "experiment_summary.csv")

print("\n========== EXPERIMENT SUMMARY ==========")
print(summary.round(4).to_string())

print("\nSaved:")
print("results/repeated_experiments.csv")
print("results/experiment_summary.csv")