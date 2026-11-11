"""Load SNAP-style edge lists and assign deterministic node indices."""

import gzip
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

PathType = Union[str, Path]
Edge = Tuple[int, int]


def load_edge_list(
    file_path: PathType,
) -> Tuple[List[Edge], List[int], Dict[int, int]]:
    """Load an edge list and return indexed edges, sorted IDs, and their map.

    Blank lines and lines whose first non-whitespace character is ``#`` are
    skipped. Malformed records are skipped and reported with a warning.
    Duplicate directed edges are reported and included only once in the
    returned edge list.

    The returned edges use internal node indices. ``node_ids[index]`` gives
    the original ID for an edge endpoint, and ``node_to_index`` provides the
    reverse mapping. IDs are sorted before indexing so mappings are stable
    across runs. Files ending in ``.gz`` are read directly as gzip streams.
    """
    path = Path(file_path)
    edge_file = (
        gzip.open(path, mode="rt", encoding="utf-8")
        if path.suffix.lower() == ".gz"
        else path.open(mode="rt", encoding="utf-8")
    )

    edges: List[Edge] = []
    node_id_set = set()
    malformed_count = 0
    malformed_lines: List[int] = []

    with edge_file:
        for line_number, line in enumerate(edge_file, start=1):
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue

            columns = stripped_line.split()
            if len(columns) != 2:
                malformed_count += 1
                if len(malformed_lines) < 5:
                    malformed_lines.append(line_number)
                continue

            try:
                source, destination = int(columns[0]), int(columns[1])
            except ValueError:
                malformed_count += 1
                if len(malformed_lines) < 5:
                    malformed_lines.append(line_number)
                continue

            edges.append((source, destination))
            node_id_set.add(source)
            node_id_set.add(destination)

    if malformed_count:
        warnings.warn(
            f"Skipped {malformed_count} malformed record(s) in {path}; "
            f"first line number(s): {', '.join(map(str, malformed_lines))}.",
            RuntimeWarning,
            stacklevel=2,
        )

    edges.sort()
    unique_edge_count = 0
    duplicate_count = 0
    previous_edge = None
    for edge in edges:
        if edge == previous_edge:
            duplicate_count += 1
            continue
        edges[unique_edge_count] = edge
        unique_edge_count += 1
        previous_edge = edge
    del edges[unique_edge_count:]

    if duplicate_count:
        warnings.warn(
            f"Found and removed {duplicate_count} duplicate edge record(s) "
            f"in {path}.",
            RuntimeWarning,
            stacklevel=2,
        )

    node_ids = sorted(node_id_set)
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    for edge_index, (source, destination) in enumerate(edges):
        edges[edge_index] = (
            node_to_index[source],
            node_to_index[destination],
        )

    return edges, node_ids, node_to_index
