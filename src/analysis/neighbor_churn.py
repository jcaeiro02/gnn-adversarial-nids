"""Helpers for measuring structural side effects of feature-space attacks."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from data.graph_builder import FlowGraphBuilder
except ModuleNotFoundError:  # pragma: no cover - fallback for direct test imports
    from src.data.graph_builder import FlowGraphBuilder


def rebuild_knn_graph(
    x: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
    bidirectional: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild a k-NN graph from node features using the existing graph builder."""
    builder = FlowGraphBuilder()
    return builder.build_knn_graph(x, k=k, metric=metric, bidirectional=bidirectional)


def compute_neighbor_churn(
    original_edge_index: np.ndarray,
    attacked_edge_index: np.ndarray,
) -> float:
    """Compute the fraction of changed neighbors between two directed graphs.

    The metric compares the sets of outgoing neighbors for each node in the
    original and attacked graphs and reports the fraction of neighbor slots
    that changed.
    """
    if original_edge_index.size == 0 and attacked_edge_index.size == 0:
        return 0.0

    if original_edge_index.ndim != 2 or original_edge_index.shape[0] != 2:
        raise ValueError(f"original_edge_index must have shape (2, num_edges); got {original_edge_index.shape}")
    if attacked_edge_index.ndim != 2 or attacked_edge_index.shape[0] != 2:
        raise ValueError(f"attacked_edge_index must have shape (2, num_edges); got {attacked_edge_index.shape}")

    num_nodes = max(
        int(original_edge_index.max()) + 1,
        int(attacked_edge_index.max()) + 1,
    )

    original_neighbors = [set() for _ in range(num_nodes)]
    attacked_neighbors = [set() for _ in range(num_nodes)]

    for src, dst in original_edge_index.T:
        original_neighbors[int(src)].add(int(dst))
    for src, dst in attacked_edge_index.T:
        attacked_neighbors[int(src)].add(int(dst))

    changed_count = 0
    total_slots = 0
    for node_idx in range(num_nodes):
        original_slot_count = len(original_neighbors[node_idx])
        attacked_slot_count = len(attacked_neighbors[node_idx])
        total_slots += max(original_slot_count, attacked_slot_count)
        changed_count += len(original_neighbors[node_idx] ^ attacked_neighbors[node_idx])

    if total_slots == 0:
        return 0.0

    return changed_count / total_slots
