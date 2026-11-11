"""Build sparse graph structures and calculate directed graph statistics."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union
from zipfile import BadZipFile

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_csr, load_npz, save_npz

from src.data_loader import load_edge_list
from src.pagerank import calculate_pagerank

PathType = Union[str, Path]
Edge = Tuple[int, int]
GraphStatistics = Dict[str, Union[int, float, np.ndarray]]
CACHE_VERSION = 1
PAGERANK_PARAMETERS = {
    "damping_factor": 0.85,
    "tolerance": 1e-6,
    "max_iterations": 100,
}
CACHE_DATA_FILES = (
    "adjacency.npz",
    "node_ids.npy",
    "in_degree.npy",
    "out_degree.npy",
    "pagerank.npy",
)
CACHE_FILES = (*CACHE_DATA_FILES, "graph_stats.json")


def build_adjacency_matrix(
    edges: Sequence[Edge], num_nodes: int
) -> csr_matrix:
    """Build a CSR matrix with source rows and destination columns.

    Edge endpoints are internal node indices. Duplicate edges are collapsed
    and stored as a single 1. Coordinate arrays are created directly from the
    edge sequence to avoid building additional Python lists.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")

    edge_count = len(edges)
    if edge_count == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=np.int32)

    sources = np.fromiter(
        (source for source, _ in edges), dtype=np.int32, count=edge_count
    )
    destinations = np.fromiter(
        (destination for _, destination in edges),
        dtype=np.int32,
        count=edge_count,
    )
    values = np.ones(edge_count, dtype=np.int32)

    adjacency = coo_matrix(
        (values, (sources, destinations)),
        shape=(num_nodes, num_nodes),
        dtype=np.int32,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data.fill(1)
    return adjacency


def calculate_graph_statistics(adjacency: csr_matrix) -> GraphStatistics:
    """Calculate node, edge, and degree statistics for a sparse graph."""
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    num_nodes = adjacency.shape[0]
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    num_edges = int(adjacency.nnz)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "average_in_degree": (
            float(in_degree.mean()) if num_nodes else 0.0
        ),
        "average_out_degree": (
            float(out_degree.mean()) if num_nodes else 0.0
        ),
        "average_total_degree": (
            float((in_degree.sum() + out_degree.sum()) / num_nodes)
            if num_nodes
            else 0.0
        ),
        "max_in_degree": int(in_degree.max()) if num_nodes else 0,
        "max_out_degree": int(out_degree.max()) if num_nodes else 0,
        "in_degree": in_degree,
        "out_degree": out_degree,
    }


def process_graph(
    dataset_path: PathType,
) -> Tuple[
    csr_matrix,
    np.ndarray,
    Dict[int, int],
    np.ndarray,
    np.ndarray,
    GraphStatistics,
]:
    """Load a dataset and return its sparse graph, IDs, degrees, and stats."""
    edges, original_node_ids, node_to_index = load_edge_list(dataset_path)
    node_ids = np.asarray(original_node_ids)
    adjacency = build_adjacency_matrix(edges, len(node_ids))
    statistics = calculate_graph_statistics(adjacency)

    return (
        adjacency,
        node_ids,
        node_to_index,
        statistics["in_degree"],
        statistics["out_degree"],
        statistics,
    )


def _file_sha256(file_path: Path) -> str:
    """Hash a file in small chunks so large sparse matrices need little memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(dataset_path: Path) -> Dict[str, Union[str, int]]:
    source_stat = dataset_path.stat()
    return {
        "source_path": str(dataset_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
    }


def _load_graph_cache(
    cache_dir: Path,
    source_metadata: Dict[str, Union[str, int]],
) -> Dict[str, object]:
    """Load and validate a complete cache, raising ValueError if invalid."""
    cache_paths = {name: cache_dir / name for name in CACHE_FILES}
    if not all(path.is_file() for path in cache_paths.values()):
        raise ValueError("one or more graph cache files are missing")

    try:
        with cache_paths["graph_stats.json"].open(
            "r", encoding="utf-8"
        ) as stats_file:
            stats = json.load(stats_file)
        if not isinstance(stats, dict) or not isinstance(stats.get("_cache"), dict):
            raise ValueError("graph cache metadata is invalid")
        metadata = stats["_cache"]
        if metadata["cache_version"] != CACHE_VERSION:
            raise ValueError("unsupported graph cache version")
        if metadata["source"] != source_metadata:
            raise ValueError("cache source metadata does not match")
        if metadata["pagerank_parameters"] != PAGERANK_PARAMETERS:
            raise ValueError("cache PageRank parameters do not match")

        checksums = metadata["sha256"]
        if not isinstance(checksums, dict):
            raise ValueError("cache checksums are invalid")
        for name in CACHE_DATA_FILES:
            if checksums[name] != _file_sha256(cache_paths[name]):
                raise ValueError(f"cache checksum mismatch: {name}")

        adjacency = load_npz(cache_paths["adjacency.npz"])
        node_ids = np.load(cache_paths["node_ids.npy"], allow_pickle=False)
        in_degree = np.load(cache_paths["in_degree.npy"], allow_pickle=False)
        out_degree = np.load(cache_paths["out_degree.npy"], allow_pickle=False)
        pagerank_scores = np.load(
            cache_paths["pagerank.npy"], allow_pickle=False
        )
    except (
        OSError,
        EOFError,
        BadZipFile,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("graph cache is corrupted or incomplete") from error

    if not isinstance(stats, dict):
        raise ValueError("cached graph statistics must be an object")
    if not isspmatrix_csr(adjacency) or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("cached adjacency must be a square CSR matrix")
    if adjacency.dtype.kind not in "biuf":
        raise ValueError("cached adjacency must have real numeric values")
    if not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0):
        raise ValueError("cached adjacency values are invalid")

    if (
        node_ids.ndim != 1
        or node_ids.dtype.kind not in "iu"
    ):
        raise ValueError("cached node IDs must be a one-dimensional integer array")
    num_nodes = node_ids.size
    if np.any(node_ids[1:] <= node_ids[:-1]):
        raise ValueError("cached node IDs must be sorted and unique")
    for array_name, array in (
        ("in_degree", in_degree),
        ("out_degree", out_degree),
        ("pagerank", pagerank_scores),
    ):
        if array.ndim != 1 or len(array) != num_nodes:
            raise ValueError(f"cached {array_name} length does not match node IDs")

    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("cached adjacency shape does not match node IDs")
    for array_name, array in (("in_degree", in_degree), ("out_degree", out_degree)):
        if array.dtype.kind not in "iuf":
            raise ValueError(f"cached {array_name} must be real numeric values")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"cached {array_name} values are invalid")
    if pagerank_scores.dtype.kind not in "iuf":
        raise ValueError("cached PageRank scores must be real numeric values")
    if (
        not np.isfinite(pagerank_scores).all()
        or np.any(pagerank_scores < 0)
        or (num_nodes and not np.isclose(pagerank_scores.sum(), 1.0, atol=1e-6))
    ):
        raise ValueError("cached PageRank scores are invalid")

    if stats.get("num_nodes") != num_nodes or stats.get("num_edges") != adjacency.nnz:
        raise ValueError("cached graph statistics do not match adjacency")
    expected_in_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    expected_out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if not np.array_equal(in_degree, expected_in_degree) or not np.array_equal(
        out_degree, expected_out_degree
    ):
        raise ValueError("cached degree arrays do not match adjacency")

    node_to_index = {
        int(node_id): index for index, node_id in enumerate(node_ids)
    }
    return {
        "adjacency": adjacency,
        "node_ids": node_ids,
        "node_to_index": node_to_index,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "pagerank": pagerank_scores,
    }


def _save_graph_cache(
    cache_dir: Path,
    source_metadata: Dict[str, Union[str, int]],
    adjacency: csr_matrix,
    node_ids: np.ndarray,
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    pagerank_scores: np.ndarray,
    statistics: GraphStatistics,
) -> None:
    """Write cache data atomically, publishing metadata only after all data."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".graph-cache-", dir=cache_dir) as temp:
        temp_dir = Path(temp)
        save_npz(temp_dir / "adjacency.npz", adjacency)
        np.save(temp_dir / "node_ids.npy", node_ids, allow_pickle=False)
        np.save(temp_dir / "in_degree.npy", in_degree, allow_pickle=False)
        np.save(temp_dir / "out_degree.npy", out_degree, allow_pickle=False)
        np.save(temp_dir / "pagerank.npy", pagerank_scores, allow_pickle=False)

        checksums = {
            name: _file_sha256(temp_dir / name) for name in CACHE_DATA_FILES
        }
        scalar_statistics = {
            key: value
            for key, value in statistics.items()
            if isinstance(value, (int, float))
        }
        cache_stats = {
            **scalar_statistics,
            "_cache": {
                "cache_version": CACHE_VERSION,
                "source": source_metadata,
                "pagerank_parameters": PAGERANK_PARAMETERS,
                "sha256": checksums,
            },
        }
        with (temp_dir / "graph_stats.json").open(
            "w", encoding="utf-8"
        ) as stats_file:
            json.dump(cache_stats, stats_file, sort_keys=True)

        # Removing the validity marker first ensures an interrupted series of
        # atomic replacements cannot leave a mixed cache accepted as valid.
        stats_path = cache_dir / "graph_stats.json"
        if stats_path.exists():
            stats_path.unlink()
        for name in CACHE_DATA_FILES:
            os.replace(temp_dir / name, cache_dir / name)
        os.replace(temp_dir / "graph_stats.json", stats_path)


def load_graph_data(
    dataset_path: PathType, cache_dir: Union[PathType, None] = None
) -> Dict[str, object]:
    """Load graph data from a validated cache or process and cache the source.

    The loader's sorted original node IDs define the internal ordering used by
    the adjacency matrix, degree arrays, and PageRank scores. By default, the
    cache is stored in the repository's ``data/processed`` directory; tests or
    callers may supply a separate cache directory.
    """
    source_path = Path(dataset_path)
    source_metadata = _source_metadata(source_path)
    if cache_dir is None:
        resolved_cache_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    else:
        resolved_cache_dir = Path(cache_dir)

    try:
        return _load_graph_cache(resolved_cache_dir, source_metadata)
    except (ValueError, OSError):
        pass

    edges, original_node_ids, node_to_index = load_edge_list(source_path)
    node_ids = np.asarray(original_node_ids)
    adjacency = build_adjacency_matrix(edges, len(node_ids))
    statistics = calculate_graph_statistics(adjacency)
    in_degree = statistics["in_degree"]
    out_degree = statistics["out_degree"]
    pagerank_scores = calculate_pagerank(adjacency, **PAGERANK_PARAMETERS)

    if _source_metadata(source_path) != source_metadata:
        raise RuntimeError("source dataset changed while graph cache was built")

    _save_graph_cache(
        resolved_cache_dir,
        source_metadata,
        adjacency,
        node_ids,
        in_degree,
        out_degree,
        pagerank_scores,
        statistics,
    )
    return {
        "adjacency": adjacency,
        "node_ids": node_ids,
        "node_to_index": node_to_index,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "pagerank": pagerank_scores,
    }