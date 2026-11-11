"""Small common interface for running either crawler strategy."""

try:
    from .ucs import run_ucs
    from .astar import run_astar
except ImportError:
    from ucs import run_ucs
    from astar import run_astar


def run_crawler(graph_data, start_node_index, budget, algorithm="ucs"):
    """
    Run the selected crawler.

    Parameters
    ----------
    algorithm : str
        "ucs" or "astar" (also accepts "a*").
    """
    algorithm = algorithm.lower()

    if algorithm == "ucs":
        return run_ucs(graph_data, start_node_index, budget)
    if algorithm in ("astar", "a*"):
        return run_astar(graph_data, start_node_index, budget)

    raise ValueError("algorithm must be 'ucs' or 'astar'.")
