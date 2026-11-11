"""Uniform Cost Search for budgeted directed-web crawling."""

import heapq
import itertools
import time

try:
    from .search_utils import (
        build_results,
        get_outgoing_neighbors,
        validate_inputs,
    )
except ImportError:  # Also supports running this file directly from src/.
    from search_utils import (
        build_results,
        get_outgoing_neighbors,
        validate_inputs,
    )


def run_ucs(graph_data, start_node_index, budget):
    """
    Explore up to `budget` unique webpages using Uniform Cost Search.

    Each hyperlink has cost 1. The priority queue is ordered by accumulated
    path cost g(n), so this is equivalent to breadth-first exploration for
    this project's unit-cost edges.

    discovered_node_indices includes the start node and every unique node
    first encountered through an outgoing edge.
    """
    validate_inputs(graph_data, start_node_index, budget)
    started_at = time.perf_counter()
    adjacency = graph_data["adjacency"]

    counter = itertools.count()
    frontier = [(0, next(counter), start_node_index)]
    best_cost = {start_node_index: 0}
    discovered = [start_node_index]
    discovered_set = {start_node_index}
    visited = []
    visited_set = set()

    while frontier and len(visited) < budget:
        current_cost, _, current = heapq.heappop(frontier)

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
                heapq.heappush(frontier, (new_cost, next(counter), neighbor))

    return build_results(
        graph_data, visited, discovered, best_cost, started_at
    )
