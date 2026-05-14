"""
PyTorch Geometric dataset for network flow graphs.

This module provides dataset management for preprocessed flow-centric graphs.
"""

import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import pickle
import torch
from torch_geometric.data import InMemoryDataset, Data

from .download import DatasetDownloader
from .preprocess import NSLKDDPreprocessor
from .graph_builder import FlowGraphBuilder

logger = logging.getLogger(__name__)


class NetworkFlowDataset(InMemoryDataset):
    """PyTorch Geometric InMemoryDataset for network flow graphs.

    Manages loading, processing, and caching of flow-centric graph datasets.
    """

    def __init__(
        self,
        root: str = "data/graphs",
        name: str = "nsl-kdd",
        split: str = "train",
        transform=None,
        pre_transform=None,
        rebuild: bool = False,
    ):
        """Initialize the dataset.

        Args:
            root: Root directory for saving processed data.
            name: Dataset name ('nsl-kdd' currently supported).
            split: Dataset split ('train' or 'test').
            transform: PyG transform to apply to data.
            pre_transform: PyG pre-transform to apply to data.
            rebuild: Force rebuild of processed data.
        """
        self.name = name
        self.split = split
        self.rebuild = rebuild
        self.data_list = []
        self.statistics = {}

        super().__init__(root, transform, pre_transform)

        logger.info(f"NetworkFlowDataset initialized: name={name}, split={split}")

    @property
    def raw_dir(self) -> str:
        """Return raw data directory path."""
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        """Return processed data directory path."""
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        if self.name == "nsl-kdd":
            if self.split == "train":
                return ["KDDTrain+.txt"]
            else:
                return ["KDDTest+.txt"]
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names."""
        return [
            f"data_{self.split}.pt",
            f"statistics_{self.split}.pkl",
        ]

    def download(self) -> None:
        """Download the dataset if not already present."""
        logger.info(f"Downloading {self.name} dataset...")

        # Create raw directory
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)

        # Download dataset
        downloader = DatasetDownloader(base_dir=self.raw_dir)

        if self.name == "nsl-kdd":
            success = downloader.download_nsl_kdd(force=False)
            if not success:
                raise RuntimeError(f"Failed to download {self.name} dataset")
            logger.info(f"Successfully downloaded {self.name} dataset")
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

    def process(self) -> None:
        """Process raw data into PyG graph format."""
        logger.info(f"Processing {self.name} dataset ({self.split} split)...")

        # Get raw data path
        raw_file = os.path.join(self.raw_dir, self.raw_file_names[0])

        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw data file not found: {raw_file}")

        # Load and preprocess data
        logger.info("Loading and preprocessing raw data...")
        preprocessor = NSLKDDPreprocessor()
        df = preprocessor.load_data(raw_file)
        X, y = preprocessor.preprocess(df, fit=True)

        # Build graph
        logger.info("Building flow-centric graph...")
        builder = FlowGraphBuilder()
        graph = builder.build_graph(X, y, method="knn", k=5, metric="cosine")

        # Save statistics
        self.statistics = {
            "num_nodes": graph.x.shape[0],
            "num_edges": graph.edge_index.shape[1],
            "num_features": graph.x.shape[1],
            "num_classes": len(np.unique(y)),
            "class_distribution": dict(
                zip(*np.unique(y, return_counts=True))
            ),
        }

        logger.info(f"Dataset statistics: {self.statistics}")

        # Save processed data
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        data_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        torch.save(graph, data_path)
        logger.info(f"Saved processed graph to {data_path}")

        # Save statistics
        stats_path = os.path.join(
            self.processed_dir, self.processed_file_names[1]
        )
        with open(stats_path, "wb") as f:
            pickle.dump(self.statistics, f)
        logger.info(f"Saved statistics to {stats_path}")

        # Store in memory
        self.data_list = [graph]

    def len(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            PyG Data object.
        """
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range")
        return self.data_list[idx]

    def load_statistics(self) -> dict:
        """Load dataset statistics.

        Returns:
            Dictionary of statistics.
        """
        stats_path = os.path.join(
            self.processed_dir, self.processed_file_names[1]
        )
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                self.statistics = pickle.load(f)
            logger.info(f"Loaded statistics: {self.statistics}")
            return self.statistics
        return {}

    @staticmethod
    def create_dataset(
        name: str = "nsl-kdd",
        split: str = "train",
        root: str = "data/graphs",
        rebuild: bool = False,
    ) -> "NetworkFlowDataset":
        """Create a network flow dataset.

        Args:
            name: Dataset name.
            split: Dataset split ('train' or 'test').
            root: Root directory for saving processed data.
            rebuild: Force rebuild of processed data.

        Returns:
            NetworkFlowDataset instance.

        Raises:
            ValueError: If split is invalid.
        """
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

        logger.info(f"Creating dataset: name={name}, split={split}, root={root}")

        # Create dataset
        dataset = NetworkFlowDataset(
            root=root,
            name=name,
            split=split,
            rebuild=rebuild,
        )

        # Load statistics
        dataset.load_statistics()

        logger.info(f"Dataset created successfully with {len(dataset)} samples")
        return dataset


def load_split_datasets(
    name: str = "nsl-kdd",
    root: str = "data/graphs",
    rebuild: bool = False,
) -> Tuple[NetworkFlowDataset, NetworkFlowDataset]:
    """Load both train and test datasets.

    Args:
        name: Dataset name.
        root: Root directory.
        rebuild: Force rebuild.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    logger.info(f"Loading train and test datasets for {name}...")

    train_dataset = NetworkFlowDataset.create_dataset(
        name=name, split="train", root=root, rebuild=rebuild
    )
    test_dataset = NetworkFlowDataset.create_dataset(
        name=name, split="test", root=root, rebuild=rebuild
    )

    logger.info(
        f"Datasets loaded: train={len(train_dataset)} samples, "
        f"test={len(test_dataset)} samples"
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    # dataset = NetworkFlowDataset.create_dataset(
    #     name="nsl-kdd",
    #     split="train",
    #     root="data/graphs"
    # )
    # print(f"Dataset: {dataset}")
    # print(f"Sample: {dataset[0]}")
