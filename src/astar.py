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
except ImportError:  # Also supports running this file directly from src/.
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
    h(n): 2 * (1 - importance(n)), where importance is a [0, 1] weighted
          combination of normalized PageRank, in-degree, out-degree and a
          balanced-connectivity proxy.

    The heuristic is a ranking preference, not an estimate of distance to a
    specified goal. It is not guaranteed admissible or consistent, and this
    budgeted strategy has no conventional A* shortest-path optimality guarantee.
    """
    validate_inputs(graph_data, start_node_index, budget)
    started_at = time.perf_counter()
    adjacency = graph_data["adjacency"]
    importance = importance_scores(graph_data)

    def heuristic(node):
        return 2.0 * (1.0 - importance[node])

    counter = itertools.count()
    start_g = 0
    frontier = [(start_g + heuristic(start_node_index),
                 start_g, next(counter), start_node_index)]
    best_cost = {start_node_index: 0}
    discovered = [start_node_index]
    discovered_set = {start_node_index}
    visited = []
    visited_set = set()

    while frontier and len(visited) < budget:
        _, current_cost, _, current = heapq.heappop(frontier)

        if current in visited_set:
            continue
        if current_cost != best_cost.get(current):
            continue

        visited.append(current)
        visited_set.add(current)

        for neighbor_value in get_outgoing_neighbors(adjacency, current):
            neighbor = int(neighbor_value)
            new_cost = current_cost + 1

            if neighbor not in discovered_set:
                discovered_set.add(neighbor)
                discovered.append(neighbor)

            if neighbor not in visited_set and new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor)
                heapq.heappush(
                    frontier,
                    (priority, new_cost, next(counter), neighbor)
                )

    return build_results(
        graph_data, visited, discovered, best_cost, started_at
    )
