
"""
A*-inspired importance-guided exploration for a budgeted web crawler.

This is deliberately NOT described as conventional admissible A*:
there is no single destination node in the budgeted discovery objective.
"""

import heapq
import itertools
import time

try:
    from .search_utils import (
        build_results,
        get_outgoing_neighbors,
        importance_scores,
        validate_inputs,
    )
except ImportError:
    # Also supports running this file directly from src/.
    from search_utils import (
        build_results,
        get_outgoing_neighbors,
        importance_scores,
        validate_inputs,
    )


def run_astar(graph_data, start_node_index, budget):
    """
    Explore up to `budget` unique webpages using f(n) = g(n) + h(n).

    g(n): accumulated unit hyperlink cost from the start node.

    h(n): 2 * (1 - importance(n)).

    Importance is a weighted combination of normalized PageRank,
    in-degree, out-degree, and balanced connectivity.

    Importance scores are cached in graph_data for reuse across runs.

    Timing:
        preprocessing_time: time spent obtaining importance scores.
        execution_time: search time only.
        total_execution_time: preprocessing plus search and result building.

    This is an importance-guided ranking strategy, not conventional
    goal-directed A*. It has no shortest-path optimality guarantee.
    """

    # Validate inputs before starting measurements.
    validate_inputs(graph_data, start_node_index, budget)

    total_started_at = time.perf_counter()

    # -----------------------------------
    # 1. Importance preprocessing
    # -----------------------------------

    preprocessing_started_at = time.perf_counter()

    adjacency = graph_data["adjacency"]
    importance = importance_scores(graph_data)

    preprocessing_time = (
        time.perf_counter() - preprocessing_started_at
    )

    # -----------------------------------
    # 2. Start measuring search time
    # -----------------------------------

    search_started_at = time.perf_counter()

    def heuristic(node):
        return 2.0 * (1.0 - importance[node])

    counter = itertools.count()

    start_g = 0

    frontier = [
        (
            start_g + heuristic(start_node_index),
            start_g,
            next(counter),
            start_node_index,
        )
    ]

    best_cost = {start_node_index: 0}

    discovered = [start_node_index]
    discovered_set = {start_node_index}

    visited = []
    visited_set = set()

    # -----------------------------------
    # 3. A* exploration
    # -----------------------------------

    while frontier and len(visited) < budget:

        _, current_cost, _, current = heapq.heappop(frontier)

        if current in visited_set:
            continue

        if current_cost != best_cost.get(current):
            continue

        visited.append(current)
        visited_set.add(current)

        for neighbor_value in get_outgoing_neighbors(
            adjacency, current
        ):

            neighbor = int(neighbor_value)

            new_cost = current_cost + 1

            # Record every unique discovered node.
            if neighbor not in discovered_set:
                discovered_set.add(neighbor)
                discovered.append(neighbor)

            # Update the best known path cost.
            if (
                neighbor not in visited_set
                and new_cost < best_cost.get(
                    neighbor, float("inf")
                )
            ):

                best_cost[neighbor] = new_cost

                priority = new_cost + heuristic(neighbor)

                heapq.heappush(
                    frontier,
                    (
                        priority,
                        new_cost,
                        next(counter),
                        neighbor,
                    ),
                )

    
    # -----------------------------------
    # 4. Build results
    # -----------------------------------

    search_time = time.perf_counter() - search_started_at

    results = build_results(
        graph_data,
        visited,
        discovered,
        best_cost,
        total_started_at,
    )

    results["preprocessing_time"] = preprocessing_time
    results["search_time"] = search_time

    return results

