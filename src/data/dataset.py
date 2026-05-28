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
import torch_geometric.data.data as pyg_data

from .download import DatasetDownloader
from .preprocess import NSLKDDPreprocessor, CICIDS2017Preprocessor
from .graph_builder import FlowGraphBuilder
from .splits import SplitManager

logger = logging.getLogger(__name__)


class NetworkFlowDataset(InMemoryDataset):
    """PyTorch Geometric InMemoryDataset for network flow graphs.

    Manages loading, processing, and caching of flow-centric graph datasets.
    """

    PREPROCESSOR_STATE_FILE = "preprocessor_state.pkl"
    CICIDS2017_SPLIT_STATE_FILE = "cicids2017_split_indices.pkl"

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
            name: Dataset name ('nsl-kdd' or 'cicids2017').
            split: Dataset split ('train', 'validation', or 'test').
            transform: PyG transform to apply to data.
            pre_transform: PyG pre-transform to apply to data.
            rebuild: Force rebuild of processed data.
            window_size: Window size for graph construction.
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
        if self.name == "cicids2017":
            root_path = Path(self._root_path)
            parent_dir = root_path.parent
            return os.path.join(parent_dir, "raw", "cicids2017")
        return os.path.join(self._root_path, "raw")

    @property
    def processed_dir(self) -> str:
        """Return processed data directory path."""
        return os.path.join(self._root_path, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names."""
        if self.name == "nsl-kdd":
            # For NSL-KDD, we always load both train and test files,
            # and use SplitManager to split them
            return ["KDDTrain+.txt", "KDDTest+.txt"]

        raw_dir = Path(self.raw_dir)
        if not raw_dir.exists():
            return []
        return [f.name for f in sorted(raw_dir.glob("*.csv"))]

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

        # Download dataset or validate manual placement
        downloader = DatasetDownloader(base_dir=self.raw_dir)

        if self.name == "nsl-kdd":
            success = downloader.download_nsl_kdd(force=False)
            if not success:
                raise RuntimeError(f"Failed to download {self.name} dataset")
            logger.info(f"Successfully downloaded {self.name} dataset")
        elif self.name == "cicids2017":
            success = downloader.download_cicids2017(base_dir=self.raw_dir)
            if not success:
                raise RuntimeError(
                    f"CICIDS2017 dataset not found in {self.raw_dir}. "
                    "Place CSV files manually under this directory."
                )
            logger.info(f"CICIDS2017 raw files verified at {self.raw_dir}")
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

    def _load_or_create_cicids2017_split_indices(self, n_samples: int) -> dict:
        split_path = os.path.join(self.raw_dir, self.CICIDS2017_SPLIT_STATE_FILE)
        if os.path.exists(split_path):
            with open(split_path, "rb") as f:
                logger.info(f"Loaded existing CICIDS2017 split indices from {split_path}")
                return pickle.load(f)

        indices = np.arange(n_samples)
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        train_size = int(np.ceil(n_samples * 0.8))
        split_indices = {
            "train": indices[:train_size],
            "test": indices[train_size:],
        }

        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        with open(split_path, "wb") as f:
            pickle.dump(split_indices, f)
        logger.info(f"Saved deterministic CICIDS2017 split indices to {split_path}")
        return split_indices

    def process(self) -> None:
        """Process raw data into PyG graph format with formal splits."""
        logger.info(f"Processing {self.name} dataset ({self.split} split)...")

        # Load and preprocess data
        logger.info("Loading and preprocessing raw data...")
        root_path = Path(self._root_path)
        if root_path.name == "graphs" and root_path.parent.name == "data":
            split_dir = str(root_path.parent / "splits")
        else:
            split_dir = str(root_path / "splits")
        
        if self.name == "nsl-kdd":
            # For NSL-KDD, load both train and test files, create formal splits
            train_file = os.path.join(self.raw_dir, "KDDTrain+.txt")
            test_file = os.path.join(self.raw_dir, "KDDTest+.txt")
            
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Raw data file not found: {train_file}")
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Raw data file not found: {test_file}")
            
            preprocessor = NSLKDDPreprocessor()
            
            # Load train and test dataframes
            train_df = preprocessor.load_data(train_file)
            test_df = preprocessor.load_data(test_file)
            
            # Create or load splits using SplitManager
            split_manager = SplitManager(self.name, data_dir=split_dir)
            train_indices, val_indices, test_indices = split_manager.create_or_load_splits(
                dataset_name=self.name,
                train_df=train_df,
                test_df=test_df,
            )
            
            # Get the appropriate split indices
            if self.split == "train":
                indices = train_indices
                full_df = train_df
                fit_preprocessor = True
            elif self.split == "validation":
                indices = val_indices
                full_df = train_df
                fit_preprocessor = False
            elif self.split == "test":
                indices = test_indices
                full_df = test_df
                fit_preprocessor = False
            else:
                raise ValueError(f"Invalid split: {self.split}")
            
            # Apply split indices
            df = full_df.iloc[indices].reset_index(drop=True)
            
            # Load or fit preprocessor
            if fit_preprocessor:
                X, y = preprocessor.preprocess(df, fit=True)
            else:
                preprocessor_state_path = os.path.join(
                    self.processed_dir, self.PREPROCESSOR_STATE_FILE
                )
                if not os.path.exists(preprocessor_state_path):
                    raise FileNotFoundError(
                        f"Preprocessor state not found: {preprocessor_state_path}. "
                        "Process the train split first so validation/test splits can reuse the fitted preprocessor."
                    )
                preprocessor.load_preprocessed(preprocessor_state_path)
                X, y = preprocessor.preprocess(df, fit=False)
                
        elif self.name == "cicids2017":
            # For CICIDS2017, load all CSVs and create formal splits
            preprocessor = CICIDS2017Preprocessor()
            df = preprocessor.load_data(self.raw_dir)
            
            # Create or load splits using SplitManager
            split_manager = SplitManager(self.name, data_dir=split_dir)
            train_indices, val_indices, test_indices = split_manager.create_or_load_splits(
                dataset_name=self.name,
                train_df=df,
            )

            # Persist legacy CICIDS2017 split indices for backward compatibility
            legacy_split_path = os.path.join(self.raw_dir, self.CICIDS2017_SPLIT_STATE_FILE)
            legacy_split_indices = {
                "train": np.concatenate([train_indices, val_indices]),
                "test": test_indices,
            }
            with open(legacy_split_path, "wb") as f:
                pickle.dump(legacy_split_indices, f)
            logger.info(f"Saved legacy CICIDS2017 split indices to {legacy_split_path}")
            
            # Get the appropriate split indices
            if self.split == "train":
                indices = train_indices
                fit_preprocessor = True
            elif self.split == "validation":
                indices = val_indices
                fit_preprocessor = False
            elif self.split == "test":
                indices = test_indices
                fit_preprocessor = False
            else:
                raise ValueError(f"Invalid split: {self.split}")
            
            # Apply split indices
            df_split = df.iloc[indices].reset_index(drop=True)
            
            # Load or fit preprocessor
            if fit_preprocessor:
                X, y = preprocessor.preprocess(df_split, fit=True)
            else:
                preprocessor_state_path = os.path.join(
                    self.processed_dir, self.PREPROCESSOR_STATE_FILE
                )
                if not os.path.exists(preprocessor_state_path):
                    raise FileNotFoundError(
                        f"Preprocessor state not found: {preprocessor_state_path}. "
                        "Process the train split first so validation/test splits can reuse the fitted preprocessor."
                    )
                preprocessor.load_preprocessed(preprocessor_state_path)
                X, y = preprocessor.preprocess(df_split, fit=False)
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

        # Save train preprocessor state (only for train split)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        if self.split == "train":
            state_path = os.path.join(
                self.processed_dir,
                self.PREPROCESSOR_STATE_FILE,
            )
            state = {
                "scaler": preprocessor.scaler,
                "feature_names": preprocessor.feature_names,
            }
            if hasattr(preprocessor, "label_encoders"):
                state["label_encoders"] = preprocessor.label_encoders
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
            try:
                with torch.serialization.safe_globals([
                    pyg_data.DataEdgeAttr,
                    pyg_data.Data,
                ]):
                    data, slices = torch.load(data_path)
            except Exception:
                data, slices = torch.load(data_path, weights_only=False)
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
            split: Dataset split ('train', 'validation', or 'test').
            root: Root directory for saving processed data.
            rebuild: Force rebuild of processed data.
            window_size: Window size for graph construction.

        Returns:
            NetworkFlowDataset instance.

        Raises:
            ValueError: If split is invalid.
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'")

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
) -> Tuple[NetworkFlowDataset, NetworkFlowDataset, NetworkFlowDataset]:
    """Load train, validation, and test datasets with formal split protocol.

    Args:
        name: Dataset name.
        root: Root directory.
        rebuild: Force rebuild.
        window_size: Window size for graph construction.

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset).
    """
    logger.info(f"Loading train, validation, and test datasets for {name}...")

    train_dataset = NetworkFlowDataset.create_dataset(
        name=name,
        split="train",
        root=root,
        rebuild=rebuild,
        window_size=window_size,
    )
    validation_dataset = NetworkFlowDataset.create_dataset(
        name=name,
        split="validation",
        root=root,
        rebuild=False,
        window_size=window_size,
    )
    test_dataset = NetworkFlowDataset.create_dataset(
        name=name,
        split="test",
        root=root,
        rebuild=False,
        window_size=window_size,
    )

    logger.info(
        f"Datasets loaded: train={len(train_dataset)} samples, "
        f"validation={len(validation_dataset)} samples, test={len(test_dataset)} samples"
    )

    return train_dataset, validation_dataset, test_dataset


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
