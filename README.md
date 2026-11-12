# Intelligent Web Crawler

An experimental directed-graph crawler that compares **Uniform Cost Search
(UCS)** with an **A*-inspired importance-guided strategy**. The project uses
the SNAP Web-Google dataset, sparse graph processing, PageRank, and evaluation
metrics to study how effectively a limited crawl budget discovers important
webpages.

> The A* implementation is intentionally described as **A*-inspired**. It does
> not solve a conventional single-goal shortest-path problem and therefore
> does not have the usual admissibility or optimality guarantees of A*.

## Features

- Loads SNAP-style directed edge lists, including `.gz` files.
- Removes duplicate edges and assigns deterministic internal node indices.
- Builds and caches a SciPy CSR adjacency matrix and graph statistics.
- Calculates PageRank and node importance from PageRank and degree features.
- Runs UCS or importance-guided A* with a configurable visit budget.
- Measures visited pages, PageRank mass captured, graph coverage, execution
  time, and peak process memory.
- Produces CSV experiment results and comparison charts.

## Project layout

```text
.
├── data/
│   └── web-Google.txt          # SNAP Web-Google edge list
├── notebooks/                  # Dataset, algorithm, and visualization notebooks
├── results/                    # Saved CSV results and generated figures
├── src/
│   ├── astar.py                # A*-inspired crawler
│   ├── crawler.py              # Common crawler interface
│   ├── data_loader.py          # Edge-list loading and normalization
│   ├── evaluation.py           # Metrics, memory measurement, and plots
│   ├── graph_processor.py      # Sparse graph construction and caching
│   ├── pagerank.py             # PageRank calculation
│   ├── search_utils.py         # Shared validation and importance scoring
│   └── ucs.py                  # Uniform Cost Search crawler
├── main.py                     # Reserved entrypoint
├── run_experiments.py          # Repeated UCS/A* benchmark
├── generate_charts.py          # Charts from benchmark results
└── test_*.py                  # Unit, integration, and evaluation tests
```

## Requirements

Python 3.9 or newer is recommended. Install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib psutil pytest
```

The repository currently keeps `requirements.txt` as a placeholder. If you
prefer a requirements file, install the packages above and record them in your
environment or add pinned versions appropriate for your deployment.

## Quick start

The integration script loads the dataset, validates the graph, and runs both
algorithms:

```bash
python test_integration.py
```

To run the benchmark configured in `run_experiments.py`:

```bash
python run_experiments.py
```

The first graph load creates a validated cache under `data/processed/`.
Subsequent runs reuse it when the source dataset and PageRank parameters have
not changed.

Generate the six comparison charts after running the benchmark:

```bash
python generate_charts.py
```

Run the automated tests with:

```bash
pytest -q
```

## Using the crawler as a library

```python
from src.astar import run_astar
from src.graph_processor import load_graph_data
from src.ucs import run_ucs

graph_data = load_graph_data("data/web-Google.txt")

# start_node_index is the internal index, not necessarily the original
# dataset node ID.
start_node_index = 0
budget = 1_000

ucs_result = run_ucs(graph_data, start_node_index, budget)
astar_result = run_astar(graph_data, start_node_index, budget)

print(len(ucs_result["visited_node_indices"]))
print(len(astar_result["visited_node_indices"]))
```

Alternatively, use the common interface:

```python
from src.crawler import run_crawler

result = run_crawler(graph_data, 0, 1_000, algorithm="astar")
```

Each result includes visited internal indices, corresponding original node
IDs, discovered nodes, total path cost, and execution time. A crawl always
starts by visiting the selected start node and stops when the budget is
reached or the reachable frontier is exhausted.

## Algorithms

### Uniform Cost Search

All hyperlinks have unit cost, so UCS explores the graph in increasing
shortest-path cost order, equivalent to breadth-first exploration for this
dataset.

### A*-inspired exploration

The priority is:

```text
f(n) = g(n) + h(n)
h(n) = 2 * (1 - importance(n))
```

Importance combines normalized, log-scaled features:

- PageRank: 40%
- in-degree: 20%
- out-degree: 20%
- balanced connectivity (`sqrt(in_degree * out_degree)`): 20%

Importance scores are cached in `graph_data` so repeated A* runs can reuse the
preprocessing work.

## Data and results

`data/web-Google.txt` is a SNAP-format directed edge list. Its header reports
875,713 nodes and 5,105,039 edges. The loader skips comments and blank lines,
reports malformed records, removes duplicate directed edges, and sorts original
node IDs before assigning internal indices.

Benchmark outputs are written to `results/`:

- `repeated_experiments.csv`: individual runs for each start node, budget,
  repetition, and algorithm.
- `experiment_summary.csv`: mean and standard deviation grouped by budget and
  algorithm.
- `figures/*.png`: charts for visits, important-page discovery, PageRank mass,
  graph coverage, execution time, and memory use.

## Notebooks

The notebooks provide an interactive workflow:

1. `01_dataset_exploration.ipynb` explores the input graph.
2. `02_algorithm_testing.ipynb` tests the crawler implementations.
3. `03_results_visualization.ipynb` visualizes experiment output.

## Notes

- `start_node_index` refers to the deterministic internal index. Use
  `graph_data["node_to_index"]` to map an original dataset ID to that index.
- Peak memory is process-wide RSS sampled during execution, so it includes
  imported libraries and other activity in the process.
- The default cache is ignored by Git and can be safely regenerated.
