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

    PREPROCESSOR_STATE_FILE = "preprocessor_state.pkl"

    def __init__(
        self,
        root: str = "data/graphs",
        name: str = "nsl-kdd",
        split: str = "train",
        transform=None,
        pre_transform=None,
        rebuild: bool = False,
        window_size: int = 1000,
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
        # Store provided root path so properties can be used before
        # PyTorch-Geometric's InMemoryDataset.__init__ sets up `self.root`.
        self._root_path = root
        self.window_size = window_size
        self.data_list = []
        self.statistics = {}

        if self.rebuild:
            self._remove_processed_files()

        super().__init__(root, transform, pre_transform)

        # Ensure processed graph list is loaded after initialization
        self._load_processed_graphs()

        logger.info(
            f"NetworkFlowDataset initialized: name={name}, split={split}, "
            f"window_size={window_size}, rebuild={rebuild}"
        )

    @property
    def raw_dir(self) -> str:
        """Return raw data directory path."""
        return os.path.join(self._root_path, "raw")

    @property
    def processed_dir(self) -> str:
        """Return processed data directory path."""
        return os.path.join(self._root_path, "processed")

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
        """Return list of split-specific processed file names."""
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
        if self.split == "train":
            df = preprocessor.load_data(raw_file)
            X, y = preprocessor.preprocess(df, fit=True)
        else:
            preprocessor_state_path = os.path.join(
                self.processed_dir, self.PREPROCESSOR_STATE_FILE
            )
            if not os.path.exists(preprocessor_state_path):
                raise FileNotFoundError(
                    f"Preprocessor state not found: {preprocessor_state_path}. "
                    "Process the train split first so the test split can reuse the fitted preprocessor."
                )
            preprocessor.load_preprocessed(preprocessor_state_path)
            df = preprocessor.load_data(raw_file)
            X, y = preprocessor.preprocess(df, fit=False)

        # Save train preprocessor state
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        if self.split == "train":
            state_path = os.path.join(
                self.processed_dir,
                self.PREPROCESSOR_STATE_FILE,
            )
            state = {
                "scaler": preprocessor.scaler,
                "label_encoders": preprocessor.label_encoders,
                "feature_names": preprocessor.feature_names,
            }
            with open(state_path, "wb") as f:
                pickle.dump(state, f)
            logger.info(f"Saved preprocessor state to {state_path}")

        # Build graphs using windowing
        logger.info("Building flow-centric graphs with windowing...")
        builder = FlowGraphBuilder()
        graphs = builder.build_windowed_dataset(
            X,
            y,
            window_size=self.window_size,
            method="knn",
            bidirectional=True,
            k=5,
        )

        if len(graphs) == 0:
            raise ValueError("No graphs were created from the input data.")

        # Save statistics
        self.statistics = {
            "num_graphs": len(graphs),
            "total_nodes": sum(graph.x.shape[0] for graph in graphs),
            "total_edges": sum(graph.edge_index.shape[1] for graph in graphs),
            "num_features": graphs[0].x.shape[1],
            "class_distribution": self._calculate_class_distribution(graphs),
        }

        logger.info(f"Dataset statistics: {self.statistics}")

        # Save processed data
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)

        data_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        data, slices = self.collate(graphs)
        torch.save((data, slices), data_path)
        logger.info(f"Saved processed graphs to {data_path}")

        # Save statistics
        stats_path = os.path.join(
            self.processed_dir, self.processed_file_names[1]
        )
        with open(stats_path, "wb") as f:
            pickle.dump(self.statistics, f)
        logger.info(f"Saved statistics to {stats_path}")

        # Store in memory
        self.data = data
        self.slices = slices
        self.data_list = graphs

    def len(self) -> int:
        """Return the length of the dataset."""
        if self.data_list:
            return len(self.data_list)
        return super().len()

    def get(self, idx: int) -> Data:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            PyG Data object.
        """
        if self.data_list:
            if idx >= len(self.data_list):
                raise IndexError(f"Index {idx} out of range")
            return self.data_list[idx]
        return super().get(idx)

    def _calculate_class_distribution(self, graphs: List[Data]) -> dict:
        """Calculate class distribution across multiple graphs."""
        counts = {}
        for graph in graphs:
            labels = graph.y.numpy() if isinstance(graph.y, torch.Tensor) else np.array(graph.y)
            unique, freq = np.unique(labels, return_counts=True)
            for label, count in zip(unique, freq):
                counts[int(label)] = counts.get(int(label), 0) + int(count)
        return counts

    def _remove_processed_files(self) -> None:
        """Remove split-specific processed files to force rebuild."""
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        for filename in self.processed_file_names:
            path = os.path.join(self.processed_dir, filename)
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed processed file: {path}")
        if self.split == "train":
            preprocessor_path = os.path.join(self.processed_dir, self.PREPROCESSOR_STATE_FILE)
            if os.path.exists(preprocessor_path):
                os.remove(preprocessor_path)
                logger.info(f"Removed train preprocessor state: {preprocessor_path}")

    def _load_processed_graphs(self) -> None:
        """Load processed graphs into memory from existing processed files."""
        data_path = os.path.join(self.processed_dir, f"data_{self.split}.pt")
        if os.path.exists(data_path):
            data, slices = torch.load(data_path)
            self.data = data
            self.slices = slices
            self.load_statistics()
            num_graphs = super().len()
            self.data_list = [InMemoryDataset.get(self, i) for i in range(num_graphs)]
            logger.info(
                f"Loaded {len(self.data_list)} processed graph(s) for split={self.split}"
            )

    def load_statistics(self) -> dict:
        """Load dataset statistics.

        Returns:
            Dictionary of statistics.
        """
        stats_path = os.path.join(
            self.processed_dir, f"statistics_{self.split}.pkl"
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
        window_size: int = 1000,
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
            window_size=window_size,
        )

        # Load statistics
        dataset.load_statistics()

        logger.info(f"Dataset created successfully with {len(dataset)} samples")
        return dataset


def load_split_datasets(
    name: str = "nsl-kdd",
    root: str = "data/graphs",
    rebuild: bool = False,
    window_size: int = 1000,
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
        name=name,
        split="train",
        root=root,
        rebuild=rebuild,
        window_size=window_size,
    )
    test_dataset = NetworkFlowDataset.create_dataset(
        name=name,
        split="test",
        root=root,
        rebuild=rebuild,
        window_size=window_size,
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
