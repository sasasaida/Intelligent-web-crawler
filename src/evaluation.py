"""
Evaluation utilities for the Intelligent Web Crawler Optimization project.

This module evaluates UCS and A* outputs. It does NOT implement search,
crawl, dataset loading, or PageRank calculation.

Expected graph_data keys:
    adjacency, node_ids, node_to_index, in_degree, out_degree, pagerank

Expected algorithm result keys:
    visited_node_indices, visited_node_ids, discovered_node_indices,
    total_cost, execution_time

Memory note:
    Peak memory is process-wide resident set size (RSS), sampled while the
    algorithm runs. It is not isolated to the algorithm and short-lived
    memory spikes may be missed between samples. Wrap the algorithm call
    with measure_peak_memory() and copy measurement.peak_memory_mb into
    its results dictionary.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:  # Allows the rest of evaluation to work without psutil.
    psutil = None


@dataclass
class MemoryMeasurement:
    """Stores the highest process RSS observed during a measurement."""
    peak_memory_mb: float = float("nan")
    start_memory_mb: float = float("nan")


@contextmanager
def measure_peak_memory(interval=0.01):
    """
    Measure process-wide peak resident memory while code inside runs.

    Example:
        with measure_peak_memory() as memory:
            results = run_algorithm(...)
        results["peak_memory_mb"] = memory.peak_memory_mb

    RSS includes Python, imported libraries, and all process activity.
    """
    if psutil is None:
        raise ImportError("psutil is required for memory measurement. Install it with: pip install psutil")

    process = psutil.Process()
    measurement = MemoryMeasurement()
    stop_event = threading.Event()

    def read_rss():
        try:
            return process.memory_info().rss / (1024 ** 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return float("nan")

    measurement.start_memory_mb = read_rss()
    measurement.peak_memory_mb = measurement.start_memory_mb

    def sample_memory():
        while not stop_event.is_set():
            current = read_rss()
            if not np.isnan(current):
                measurement.peak_memory_mb = max(measurement.peak_memory_mb, current)
            stop_event.wait(interval)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        yield measurement
    finally:
        stop_event.set()
        sampler.join()
        final_memory = read_rss()
        if not np.isnan(final_memory):
            measurement.peak_memory_mb = max(measurement.peak_memory_mb, final_memory)


def _unique_valid_indices(indices, node_count):
    """Return unique, in-range integer node indices, preserving order."""
    if indices is None:
        return []
    unique = []
    seen = set()
    for value in indices:
        try:
            index = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= index < node_count and index not in seen:
            unique.append(index)
            seen.add(index)
    return unique


def _get_node_count(graph_data):
    """Find node count without converting or copying the adjacency matrix."""
    if graph_data.get("node_ids") is not None:
        return len(graph_data["node_ids"])
    adjacency = graph_data.get("adjacency")
    if adjacency is not None and hasattr(adjacency, "shape"):
        return int(adjacency.shape[0])
    raise ValueError("graph_data must contain node_ids or a shaped adjacency matrix.")


def _get_memory_value(results):
    """Support a few clear names while keeping the agreed interface intact."""
    for key in ("peak_memory_mb", "memory_peak_mb", "peak_memory"):
        if key in results and results[key] is not None:
            try:
                return float(results[key])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _top_pagerank_indices(pagerank_scores, top_percent=1.0):
    """Return indices belonging to the top X percent of PageRank scores."""
    scores = np.asarray(pagerank_scores, dtype=float).reshape(-1)
    if not 0 < top_percent <= 100:
        raise ValueError("top_percent must be greater than 0 and at most 100.")
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) == 0:
        return set()
    count = max(1, int(np.ceil(len(valid) * top_percent / 100.0)))
    # argpartition avoids sorting the entire score array.
    selected_local = np.argpartition(scores[valid], -count)[-count:]
    return set(valid[selected_local].tolist())


def evaluate_single_result(graph_data, results, algorithm_name, important_nodes=None):
    """
    Calculate metrics for one algorithm's output.

    Important webpages are defined by the supplied important_nodes set,
    normally created from the top 1% of the shared PageRank scores.
    """
    node_count = _get_node_count(graph_data)
    pagerank = np.asarray(graph_data["pagerank"], dtype=float).reshape(-1)

    if len(pagerank) != node_count:
        raise ValueError("pagerank length must match the number of graph nodes.")

    visited = _unique_valid_indices(results.get("visited_node_indices", []), node_count)
    visited_set = set(visited)

    if important_nodes is None:
        important_nodes = _top_pagerank_indices(pagerank, top_percent=1.0)

    important_visited = len(visited_set.intersection(important_nodes))
    captured_mass = float(np.nansum(pagerank[visited])) if visited else 0.0
    total_mass = float(np.nansum(pagerank))
    captured_percent = (captured_mass / total_mass * 100.0) if total_mass > 0 else 0.0

    execution_time = results.get("execution_time", np.nan)
    try:
        execution_time = float(execution_time)
    except (TypeError, ValueError):
        execution_time = np.nan

    return {
        "Algorithm": algorithm_name,
        "Visited Webpages": len(visited),
        "Important Webpages Visited": important_visited,
        "PageRank Mass Captured": captured_mass,
        "PageRank Mass Captured (%)": captured_percent,
        "Graph Coverage (%)": (len(visited) / node_count * 100.0) if node_count else 0.0,
        "Execution Time (seconds)": execution_time,
        "Peak Memory (MB)": _get_memory_value(results),
    }


def evaluate_results(graph_data, ucs_results, astar_results, top_percent=1.0):
    """
    Compare UCS and A* and return a two-row Pandas DataFrame.

    The important-page group is computed once from the shared PageRank
    scores, ensuring both algorithms are evaluated against the same target.
    """
    node_count = _get_node_count(graph_data)
    pagerank = np.asarray(graph_data["pagerank"], dtype=float).reshape(-1)
    if len(pagerank) != node_count:
        raise ValueError("pagerank length must match the number of graph nodes.")

    important_nodes = _top_pagerank_indices(pagerank, top_percent)
    rows = [
        evaluate_single_result(graph_data, ucs_results, "UCS", important_nodes),
        evaluate_single_result(graph_data, astar_results, "A*", important_nodes),
    ]
    return pd.DataFrame(rows)


def evaluate_budget_experiments(graph_data, experiment_results, top_percent=1.0):
    """
    Evaluate multiple budgets.

    experiment_results format:
        {
            100: {"ucs": ucs_result, "astar": astar_result},
            500: {"ucs": ucs_result, "astar": astar_result},
        }

    Returns one row per budget and algorithm.
    """
    rows = []
    for budget, pair in experiment_results.items():
        comparison = evaluate_results(
            graph_data,
            pair.get("ucs", {}),
            pair.get("astar", {}),
            top_percent=top_percent,
        )
        comparison.insert(0, "Budget", budget)
        rows.extend(comparison.to_dict(orient="records"))
    return pd.DataFrame(rows)


def save_metrics_csv(dataframe, output_path="results/metrics.csv"):
    """Create the output directory if needed and save metrics as CSV."""
    from pathlib import Path
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return output


def plot_metric_comparison(dataframe, metric, output_path, title=None, ylabel=None):
    """Create a simple UCS vs A* bar chart and save it."""
    if "Algorithm" not in dataframe.columns or metric not in dataframe.columns:
        raise ValueError("DataFrame must contain 'Algorithm' and the requested metric.")

    import matplotlib.pyplot as plt
    output = __import__("pathlib").Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(dataframe["Algorithm"], dataframe[metric])
    ax.set_title(title or f"UCS vs A*: {metric}")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(ylabel or metric)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_budget_comparison(dataframe, metric, output_path, title=None, ylabel=None):
    """Plot a metric across budgets for UCS and A*."""
    required = {"Budget", "Algorithm", metric}
    if not required.issubset(dataframe.columns):
        raise ValueError(f"DataFrame must contain: {sorted(required)}")

    import matplotlib.pyplot as plt
    output = __import__("pathlib").Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algorithm in ("UCS", "A*"):
        subset = dataframe[dataframe["Algorithm"] == algorithm].sort_values("Budget")
        if not subset.empty:
            ax.plot(subset["Budget"], subset[metric], marker="o", label=algorithm)
    ax.set_title(title or f"UCS vs A*: {metric} by exploration budget")
    ax.set_xlabel("Exploration budget (visited webpages)")
    ax.set_ylabel(ylabel or metric)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output
