
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/repeated_experiments.csv")

metrics = {
    "Visited Webpages": "visited_webpages.png",
    "Important Webpages Visited": "important_pages.png",
    "PageRank Mass Captured (%)": "pagerank_captured.png",
    "Graph Coverage (%)": "graph_coverage.png",
    "Execution Time (seconds)": "execution_time.png",
    "Peak Memory (MB)": "memory_consumption.png",
}

output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

for metric, filename in metrics.items():
    summary = df.groupby(
        ["Budget", "Algorithm"]
    )[metric].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(8, 5))

    for algorithm in ["UCS", "A*"]:
        data = summary[summary["Algorithm"] == algorithm]

        plt.errorbar(
            data["Budget"],
            data["mean"],
            yerr=data["std"],
            marker="o",
            capsize=4,
            label=algorithm
        )

    plt.xlabel("Exploration Budget")
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

    print("Saved:", output_dir / filename)

print("\nAll six charts generated successfully.")
