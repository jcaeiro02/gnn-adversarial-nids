"""
Flow-centric graph builder for network intrusion detection.

This module constructs graphs where each network flow/sample is a node,
with edges representing relationships between flows (e.g., feature similarity).
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class FlowGraphBuilder:
    """Build flow-centric graphs for GNN-based network analysis.

    In a flow-centric graph:
    - Each network flow/sample is one node
    - Node features are the preprocessed network features
    - Node labels are attack/normal labels
    - Edges represent relationships between flows (e.g., feature similarity)
    """

    def __init__(self):
        """Initialize the graph builder."""
        self.X = None
        self.y = None
        self.graph_data = None
        logger.info("FlowGraphBuilder initialized")

    def _convert_pandas_to_numpy(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pandas DataFrame/Series to numpy arrays if needed.

        Args:
            X: Feature matrix (may be DataFrame or ndarray).
            y: Labels (may be Series or ndarray).

        Returns:
            Tuple of (X_numpy, y_numpy).
        """
        if isinstance(X, pd.DataFrame):
            logger.info("Converting X from pandas DataFrame to numpy array")
            X = X.values
        
        if isinstance(y, pd.Series):
            logger.info("Converting y from pandas Series to numpy array")
            y = y.values
        
        return X, y

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input feature matrix and labels.

        Args:
            X: Feature matrix (N_samples, N_features).
            y: Labels (N_samples,).

        Raises:
            ValueError: If inputs are invalid.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}")

        logger.info(f"Input validation passed: X shape={X.shape}, y shape={y.shape}")

    def build_knn_graph(
        self,
        X: np.ndarray,
        k: int = 5,
        metric: str = "cosine",
        bidirectional: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build k-nearest neighbor graph using similarity metric.

        Args:
            X: Feature matrix (N_samples, N_features).
            k: Number of nearest neighbors.
            metric: Distance metric ('cosine', 'euclidean', etc.).
            bidirectional: If True, add reverse edges (j->i for i->j).

        Returns:
            Tuple of (edge_index, edge_weights).
                edge_index: (2, num_edges) array of node indices.
                edge_weights: (num_edges,) array of edge weights.
        """
        logger.info(
            f"Building kNN graph with k={k}, metric={metric}, bidirectional={bidirectional}..."
        )

        n_samples = X.shape[0]
        if k >= n_samples:
            logger.warning(f"k={k} >= n_samples={n_samples}, reducing to k={n_samples-1}")
            k = max(1, n_samples - 1)
        
        if n_samples < 2:
            logger.warning(f"Cannot build kNN graph with {n_samples} sample(s)")
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)

        # Fit kNN model
        knn_model = NearestNeighbors(n_neighbors=k + 1, metric=metric)  # +1 to include self
        knn_model.fit(X)

        # Get k+1 nearest neighbors (including self)
        distances, indices = knn_model.kneighbors(X)

        # Build edges, excluding self-loops
        edge_list = []
        edge_weights = []

        for i in range(n_samples):
            # indices[i][0] is the sample itself, so skip it
            neighbors = indices[i][1:k+1]  # k nearest neighbors (excluding self)
            neighbor_distances = distances[i][1:k+1]

            for neighbor_idx, dist in zip(neighbors, neighbor_distances):
                # Convert distance to similarity
                if metric == "cosine":
                    # Cosine distance -> similarity
                    similarity = 1.0 - dist
                else:
                    # Euclidean distance -> similarity
                    similarity = 1.0 / (1.0 + dist)

                edge_list.append([i, neighbor_idx])
                edge_weights.append(similarity)

        # Add reverse edges for bidirectional graph (avoid duplicates)
        if bidirectional and len(edge_list) > 0:
            reverse_edges = []
            reverse_weights = []
            edge_set = set(map(tuple, edge_list))
            
            for [src, dst], weight in zip(edge_list, edge_weights):
                reverse_edge = (dst, src)
                if reverse_edge not in edge_set:
                    reverse_edges.append([dst, src])
                    reverse_weights.append(weight)
                    edge_set.add(reverse_edge)
            
            edge_list.extend(reverse_edges)
            edge_weights.extend(reverse_weights)
            logger.info(f"Added {len(reverse_edges)} reverse edges for bidirectional graph")

        if len(edge_list) == 0:
            logger.warning("No edges created in kNN graph")
            edge_index = np.array([[], []], dtype=np.int64)
            edge_weights = np.array([], dtype=np.float32)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_weights = np.array(edge_weights, dtype=np.float32)

        num_edges = edge_index.shape[1]
        avg_degree = num_edges / n_samples if n_samples > 0 else 0
        logger.info(
            f"kNN graph created: {n_samples} nodes, {num_edges} edges, "
            f"avg degree: {avg_degree:.2f} (bidirectional={bidirectional})"
        )

        return edge_index, edge_weights

    def build_similarity_graph(
        self,
        X: np.ndarray,
        threshold: float = 0.8,
        metric: str = "cosine",
        bidirectional: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build threshold-based similarity graph.

        Args:
            X: Feature matrix (N_samples, N_features).
            threshold: Similarity threshold for connecting nodes.
            metric: Distance metric ('cosine', 'euclidean', etc.).
            bidirectional: If True, add reverse edges (automatically done for threshold graph).

        Returns:
            Tuple of (edge_index, edge_weights).
        """
        logger.info(
            f"Building similarity graph with threshold={threshold}, metric={metric}, "
            f"bidirectional={bidirectional}..."
        )

        n_samples = X.shape[0]

        if n_samples < 2:
            logger.warning(f"Cannot build similarity graph with {n_samples} sample(s)")
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)

        if metric == "cosine":
            # Compute cosine similarity
            similarities = cosine_similarity(X)
        else:
            raise NotImplementedError(f"Metric {metric} not implemented for similarity graph")

        # Extract edges above threshold (excluding diagonal)
        edge_list = []
        edge_weights = []

        for i in range(n_samples):
            for j in range(i + 1, n_samples):  # Only upper triangle to avoid duplicates
                sim = similarities[i, j]
                if sim >= threshold:
                    edge_list.append([i, j])
                    edge_weights.append(sim)
                    
                    # Add bidirectional edge
                    if bidirectional:
                        edge_list.append([j, i])
                        edge_weights.append(sim)

        if len(edge_list) == 0:
            logger.warning(f"No edges created with threshold={threshold}")
            edge_index = np.array([[], []], dtype=np.int64)
            edge_weights = np.array([], dtype=np.float32)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_weights = np.array(edge_weights, dtype=np.float32)

        num_edges = edge_index.shape[1]
        avg_degree = num_edges / n_samples if n_samples > 0 else 0
        logger.info(
            f"Similarity graph created: {n_samples} nodes, {num_edges} edges, "
            f"avg degree: {avg_degree:.2f} (bidirectional={bidirectional})"
        )

        return edge_index, edge_weights

    def _log_graph_statistics(
        self,
        num_nodes: int,
        num_edges: int,
        y: np.ndarray,
        is_directed: bool = True,
    ) -> Dict[str, float]:
        """Log and compute graph statistics.

        Args:
            num_nodes: Number of nodes.
            num_edges: Number of edges (directed edge count in PyG).
            y: Node labels.
            is_directed: Whether the graph is directed.

        Returns:
            Dictionary of statistics.
        """
        # Compute graph statistics for directed graphs
        # Directed density = E / (N * (N - 1))
        if is_directed:
            graph_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        else:
            # Undirected density = 2 * E / (N * (N - 1))
            graph_density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
            avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        # Compute class distribution
        unique_labels, label_counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_labels, label_counts))

        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "graph_density": graph_density,
            "avg_degree": avg_degree,
            "is_directed": is_directed,
        }

        logger.info("=" * 60)
        logger.info("GRAPH STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Number of nodes: {num_nodes}")
        logger.info(f"Number of edges: {num_edges} (directed)")
        graph_type = "directed" if is_directed else "undirected"
        logger.info(f"Graph density: {graph_density:.6f} ({graph_type})")
        logger.info(f"Average degree: {avg_degree:.2f}")
        logger.info(f"Class distribution:")
        for label, count in class_distribution.items():
            percentage = 100 * count / num_nodes if num_nodes > 0 else 0
            logger.info(f"  Class {label}: {count} samples ({percentage:.2f}%)")
        logger.info("=" * 60)

        stats["class_distribution"] = class_distribution

        return stats

    def build_graph(
        self,
        X,
        y,
        method: str = "knn",
        bidirectional: bool = True,
        **kwargs
    ) -> Data:
        """Build a PyTorch Geometric graph.

        Args:
            X: Feature matrix (N_samples, N_features) or pandas DataFrame.
            y: Labels (N_samples,) or pandas Series.
            method: Graph construction method ('knn' or 'similarity').
            bidirectional: If True, add reverse edges for bidirectional message passing.
            **kwargs: Additional arguments for the chosen method.
                For 'knn': k (int), metric (str).
                For 'similarity': threshold (float), metric (str).

        Returns:
            PyTorch Geometric Data object with proper dtypes.

        Raises:
            ValueError: If inputs are invalid or method is unknown.
        """
        logger.info(f"Building graph with method={method}, bidirectional={bidirectional}...")

        # Convert pandas to numpy if needed
        X, y = self._convert_pandas_to_numpy(X, y)

        # Validate inputs
        self._validate_inputs(X, y)

        # Store for later reference
        self.X = X
        self.y = y

        # Convert to appropriate dtypes
        X_float32 = X.astype(np.float32)
        y_int64 = y.astype(np.int64)

        # Build edges
        if method == "knn":
            k = kwargs.get("k", 5)
            metric = kwargs.get("metric", "cosine")
            edge_index, edge_weights = self.build_knn_graph(
                X, k=k, metric=metric, bidirectional=bidirectional
            )
        elif method == "similarity":
            threshold = kwargs.get("threshold", 0.8)
            metric = kwargs.get("metric", "cosine")
            edge_index, edge_weights = self.build_similarity_graph(
                X, threshold=threshold, metric=metric, bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Log statistics
        num_edges = edge_index.shape[1]
        self._log_graph_statistics(X.shape[0], num_edges, y_int64, is_directed=True)

        # Create PyTorch Geometric Data object with proper dtypes
        # x: float32, y: long (int64), edge_index: long (int64), edge_attr: float32
        graph_data = Data(
            x=torch.from_numpy(X_float32).float(),  # Ensure float32
            y=torch.from_numpy(y_int64).long(),     # Ensure int64/long
            edge_index=torch.from_numpy(edge_index).long(),  # Ensure int64/long
            edge_attr=(
                torch.from_numpy(edge_weights.reshape(-1, 1)).float()
                if len(edge_weights) > 0
                else None
            ),  # Ensure float32 or None
        )

        self.graph_data = graph_data
        logger.info(
            f"Graph created: nodes={graph_data.x.shape[0]}, edges={graph_data.edge_index.shape[1]}, "
            f"features={graph_data.x.shape[1]}"
        )

        return graph_data

    def build_dataset(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        method: str = "knn",
        bidirectional: bool = True,
        **kwargs
    ) -> List[Data]:
        """Build multiple graphs from a list of feature matrices.

        Args:
            X_list: List of feature matrices.
            y_list: List of label arrays.
            method: Graph construction method ('knn' or 'similarity').
            bidirectional: If True, add reverse edges.
            **kwargs: Additional arguments for graph construction.

        Returns:
            List of PyTorch Geometric Data objects.
        """
        logger.info(f"Building dataset with {len(X_list)} graphs...")

        if len(X_list) != len(y_list):
            raise ValueError(f"X_list and y_list must have same length")

        graphs = []
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            logger.info(f"Building graph {i+1}/{len(X_list)}...")
            graph = self.build_graph(X, y, method=method, bidirectional=bidirectional, **kwargs)
            graphs.append(graph)

        logger.info(f"Dataset created with {len(graphs)} graphs")
        return graphs

    def build_windowed_dataset(
        self,
        X,
        y,
        window_size: int = 1000,
        method: str = "knn",
        bidirectional: bool = True,
        **kwargs
    ) -> List[Data]:
        """Build multiple graphs by splitting a large dataset into sequential windows.

        Each window becomes one flow-centric graph, suitable for temporal or
        sequential analysis of network flows.

        Args:
            X: Large feature matrix (N_samples, N_features) or pandas DataFrame.
            y: Corresponding labels (N_samples,) or pandas Series.
            window_size: Number of samples per window (default: 1000).
            method: Graph construction method ('knn' or 'similarity').
            bidirectional: If True, add reverse edges.
            **kwargs: Additional arguments for graph construction.

        Returns:
            List of PyTorch Geometric Data objects, one per window.

        Raises:
            ValueError: If inputs are invalid.
        """
        logger.info(
            f"Building windowed dataset with window_size={window_size}, "
            f"method={method}, bidirectional={bidirectional}..."
        )

        # Window Size

        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        # Convert pandas to numpy if needed
        X, y = self._convert_pandas_to_numpy(X, y)

        # Validate inputs
        self._validate_inputs(X, y)

        n_samples = X.shape[0]
        num_windows = int(np.ceil(n_samples / window_size))
        logger.info(f"Splitting {n_samples} samples into {num_windows} windows of size {window_size}")

        graphs = []
        for window_idx in range(num_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, n_samples)

            X_window = X[start_idx:end_idx]
            y_window = y[start_idx:end_idx]

            # Skip windows with fewer than 2 samples
            if X_window.shape[0] < 2:
                logger.warning(f"Skipping window {window_idx} with {X_window.shape[0]} sample(s)")
                continue

            logger.info(
                f"Building graph for window {window_idx+1}/{num_windows} "
                f"(samples {start_idx}-{end_idx-1}, size={X_window.shape[0]})"
            )

            graph = self.build_graph(
                X_window, y_window, method=method, bidirectional=bidirectional, **kwargs
            )
            graphs.append(graph)

        logger.info(f"Windowed dataset created with {len(graphs)} graphs")
        return graphs


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    builder = FlowGraphBuilder()

    # Create synthetic data
    n_samples = 100
    n_features = 41
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)

    print("\n" + "="*70)
    print("Example 1: kNN graph with bidirectional edges (default)")
    print("="*70)
    graph = builder.build_graph(X, y, method="knn", k=5, bidirectional=True)
    print(f"Graph: {graph}")
    print(f"  Tensor dtypes: x={graph.x.dtype}, y={graph.y.dtype}, "
          f"edge_index={graph.edge_index.dtype}, edge_attr={graph.edge_attr.dtype if graph.edge_attr is not None else None}")

    print("\n" + "="*70)
    print("Example 2: Similarity graph with bidirectional edges")
    print("="*70)
    graph2 = builder.build_graph(X, y, method="similarity", threshold=0.7, bidirectional=True)
    print(f"Graph: {graph2}")

    print("\n" + "="*70)
    print("Example 3: Windowed dataset (split large dataset into multiple graphs)")
    print("="*70)
    X_large = np.random.randn(5000, n_features).astype(np.float32)
    y_large = np.random.randint(0, 2, 5000)
    graphs = builder.build_windowed_dataset(
        X_large, y_large, window_size=1000, method="knn", k=5, bidirectional=True
    )
    print(f"Created {len(graphs)} graphs from large dataset")
    for i, g in enumerate(graphs):
        print(f"  Graph {i}: {g.x.shape[0]} nodes, {g.edge_index.shape[1]} edges")

    print("\n" + "="*70)
    print("Example 4: Build multiple graphs from list")
    print("="*70)
    X_list = [np.random.randn(50, n_features) for _ in range(3)]
    y_list = [np.random.randint(0, 2, 50) for _ in range(3)]
    graphs_list = builder.build_dataset(X_list, y_list, method="knn", k=5, bidirectional=True)
    print(f"Created {len(graphs_list)} graphs")

