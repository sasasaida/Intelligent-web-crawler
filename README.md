# Intelligent Web Crawler

An experimental web-graph crawler that compares two budgeted exploration
strategies:

- **Uniform Cost Search (UCS):** explores the directed graph by unit hyperlink
  cost. With every edge costing 1, this behaves like breadth-first exploration.
- **Importance-guided A\*:** prioritizes webpages using path cost and an
  importance score based on PageRank, in-degree, out-degree, and connectivity.

The project evaluates how effectively each strategy discovers webpages and
high-PageRank pages under a fixed exploration budget.

## Project structure

```text
.
├── data/
│   ├── web-Google.txt          # SNAP-style directed edge list
│   └── processed/              # Generated graph and PageRank cache
├── notebooks/                  # Exploration, testing, and visualization notebooks
├── results/                    # Experiment CSV files and generated figures
├── src/
│   ├── astar.py                # Importance-guided A* search
│   ├── crawler.py              # Common crawler interface
│   ├── data_loader.py          # Edge-list loading
│   ├── evaluation.py           # Metrics and plotting helpers
│   ├── graph_processor.py      # Sparse graph construction and caching
│   ├── pagerank.py             # Sparse PageRank implementation
│   └── ucs.py                  # Uniform Cost Search
├── generate_charts.py          # Generate charts from experiment results
├── run_experiments.py          # Repeated UCS/A* experiments
├── test_evaluation.py          # Evaluation utility checks
├── test_integration.py         # End-to-end graph and crawler checks
└── test_real_evaluation.py     # Evaluation on the Web-Google dataset
```

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- pandas
- matplotlib
- psutil

Create and activate a virtual environment from the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib psutil
```

## Dataset

The project uses the [SNAP Web-Google dataset](https://snap.stanford.edu/data/web-Google.html),
stored at `data/web-Google.txt`. Each non-comment line contains a directed
`source destination` edge.

The included dataset contains 875,713 nodes and 5,105,039 unique directed
edges. The loader maps original node IDs to deterministic zero-based indices.

## Usage

Run commands from the repository root.

### Validate the graph and both crawlers

```powershell
python test_integration.py
```

This loads the graph, verifies the sparse representation, and runs UCS and
A* with a visit budget of 1,000.

### Check evaluation utilities

```powershell
python test_evaluation.py
```

### Run a single real-data comparison

```powershell
python test_real_evaluation.py
```

The comparison is saved to `results/real_experiment.csv`.

### Run repeated experiments

```powershell
python run_experiments.py
```

The experiment uses budgets of 100, 500, 1,000, 5,000, and 10,000 webpages,
three starting nodes, and three repetitions per budget. It writes:

- `results/repeated_experiments.csv`
- `results/experiment_summary.csv`

### Generate charts

Run the repeated experiments first, then execute:

```powershell
python generate_charts.py
```

Charts are written to `results/figures/`.

## Python API

```python
from src.astar import run_astar
from src.graph_processor import load_graph_data
from src.ucs import run_ucs

graph_data = load_graph_data("data/web-Google.txt")
start_node_index = 0
budget = 1000

ucs_result = run_ucs(graph_data, start_node_index, budget)
astar_result = run_astar(graph_data, start_node_index, budget)
```

`start_node_index` is an internal zero-based index. Use
`graph_data["node_ids"]` to translate it to the original dataset node ID.

Both search functions return visited and discovered nodes, path-cost
information, and timing data. A* also reports preprocessing and search times.

## Graph cache

`load_graph_data()` stores validated sparse graph arrays and PageRank scores in
`data/processed/`. The cache is automatically rebuilt when the source dataset
changes or the cache is incomplete or invalid.

The generated cache files are ignored by Git and can be safely regenerated.

## Metrics

The evaluation code reports:

- visited webpages
- important webpages visited (top 1% by PageRank by default)
- PageRank mass captured
- graph coverage
- execution time
- peak process memory

The importance-guided search is an exploration heuristic rather than
conventional goal-directed A*, so it does not provide a shortest-path
optimality guarantee.