"""
Formal experimental split protocol for datasets.

This module provides reproducible train/validation/test splits with fixed seed,
persisted indices, and clear metadata.
"""

import logging
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SplitManager:
    """Manages reproducible train/validation/test splits for datasets."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "data/splits",
        random_state: int = 42,
    ):
        """Initialize SplitManager.

        Args:
            dataset_name: Name of the dataset ('nsl-kdd' or 'cicids2017').
            data_dir: Base directory for split storage.
            random_state: Random seed for reproducibility.
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.split_dir = self.data_dir / dataset_name
        
        # Create split directory if it doesn't exist
        self.split_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"SplitManager initialized: dataset={dataset_name}, "
            f"split_dir={self.split_dir}, random_state={random_state}"
        )

    def _get_split_files(self) -> Dict[str, Path]:
        """Get paths to split index files.

        Returns:
            Dictionary mapping split names to file paths.
        """
        return {
            "train": self.split_dir / "train_indices.npy",
            "validation": self.split_dir / "validation_indices.npy",
            "test": self.split_dir / "test_indices.npy",
            "config": self.split_dir / "split_config.json",
        }

    def save_split_indices(
        self,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        test_indices: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save split indices to disk.

        Args:
            train_indices: Training split indices.
            validation_indices: Validation split indices.
            test_indices: Test split indices.
            metadata: Optional metadata dictionary to save.
        """
        files = self._get_split_files()

        # Save indices
        np.save(str(files["train"]), train_indices)
        np.save(str(files["validation"]), validation_indices)
        np.save(str(files["test"]), test_indices)

        logger.info(
            f"Saved split indices: train={len(train_indices)}, "
            f"validation={len(validation_indices)}, test={len(test_indices)}"
        )

        # Save config
        config = {
            "dataset_name": self.dataset_name,
            "random_state": self.random_state,
            "train_size": int(len(train_indices)),
            "validation_size": int(len(validation_indices)),
            "test_size": int(len(test_indices)),
            "total_size": int(len(train_indices) + len(validation_indices) + len(test_indices)),
            "metadata": metadata or {},
        }

        with open(files["config"], "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved split config to {files['config']}")

    def load_split_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load split indices from disk.

        Returns:
            Tuple of (train_indices, validation_indices, test_indices).

        Raises:
            FileNotFoundError: If split files don't exist.
        """
        files = self._get_split_files()

        if not all(f.exists() for f in files.values()):
            raise FileNotFoundError(
                f"Split files not found in {self.split_dir}. "
                "Create splits using create_or_load_splits()."
            )

        train_indices = np.load(str(files["train"]))
        validation_indices = np.load(str(files["validation"]))
        test_indices = np.load(str(files["test"]))

        logger.info(
            f"Loaded split indices: train={len(train_indices)}, "
            f"validation={len(validation_indices)}, test={len(test_indices)}"
        )

        return train_indices, validation_indices, test_indices

    def load_split_config(self) -> Dict:
        """Load split configuration.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        files = self._get_split_files()

        if not files["config"].exists():
            raise FileNotFoundError(
                f"Split config not found in {self.split_dir}"
            )

        with open(files["config"], "r") as f:
            config = json.load(f)

        return config

    def get_split_indices(self, split: str) -> np.ndarray:
        """Get indices for a specific split.

        Args:
            split: Split name ('train', 'validation', or 'test').

        Returns:
            Array of indices for the specified split.

        Raises:
            ValueError: If split name is invalid.
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'")

        train_indices, validation_indices, test_indices = self.load_split_indices()

        split_map = {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        }

        return split_map[split]

    def splits_exist(self) -> bool:
        """Check if split files exist.

        Returns:
            True if all split files exist, False otherwise.
        """
        files = self._get_split_files()
        return all(f.exists() for f in files.values())

    def create_nsl_kdd_splits(
        self,
        train_df: "pd.DataFrame",
        test_df: "pd.DataFrame",
        train_val_split_ratio: float = 0.15,
    ) -> None:
        """Create NSL-KDD splits using official test set.

        NSL-KDD policy:
        - Keep KDDTest+.txt as the final test split.
        - Split KDDTrain+.txt into train and validation using stratified split.
        - All indices are relative to the respective source files.

        Args:
            train_df: Training dataframe (from KDDTrain+.txt).
            test_df: Test dataframe (from KDDTest+.txt).
            train_val_split_ratio: Ratio of training data to use for validation.
        """
        logger.info(
            f"Creating NSL-KDD splits: train_size={len(train_df)}, "
            f"test_size={len(test_df)}, val_ratio={train_val_split_ratio}"
        )

        # Extract binary labels from train_df (normal=0, attack=1)
        # Assume labels are in the last column
        train_labels = train_df.iloc[:, -1].values

        # Stratified split of training data into train and validation
        train_indices, val_indices = train_test_split(
            np.arange(len(train_df)),
            test_size=train_val_split_ratio,
            random_state=self.random_state,
            stratify=train_labels,
        )

        # All test indices (use all of test_df as test set)
        test_indices = np.arange(len(test_df))

        # Store metadata
        metadata = {
            "nsl_kdd_policy": "official_test",
            "train_df_source": "KDDTrain+.txt",
            "test_df_source": "KDDTest+.txt",
            "train_indices_relative_to": "KDDTrain+.txt",
            "validation_indices_relative_to": "KDDTrain+.txt",
            "test_indices_relative_to": "KDDTest+.txt",
        }

        self.save_split_indices(train_indices, val_indices, test_indices, metadata)

    def create_cicids2017_splits(
        self,
        df: "pd.DataFrame",
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        """Create CICIDS2017 splits with stratified sampling.

        CICIDS2017 policy:
        - Load all CICIDS2017 CSV rows.
        - Create deterministic stratified train/validation/test split.
        - All indices relative to the combined loaded dataframe.

        Args:
            df: Combined dataframe from all CICIDS2017 CSV files.
            train_ratio: Fraction for training (default 0.70).
            validation_ratio: Fraction for validation (default 0.15).
            test_ratio: Fraction for testing (default 0.15).

        Raises:
            ValueError: If ratios don't sum to 1.0.
        """
        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + validation_ratio + test_ratio}"
            )

        logger.info(
            f"Creating CICIDS2017 splits: total_size={len(df)}, "
            f"train_ratio={train_ratio}, val_ratio={validation_ratio}, test_ratio={test_ratio}"
        )

        # Extract binary labels (assume last column or 'label' column)
        if "label" in df.columns or "Label" in df.columns:
            label_col = "label" if "label" in df.columns else "Label"
            labels = df[label_col].values
        else:
            labels = df.iloc[:, -1].values

        # Create indices
        indices = np.arange(len(df))

        # First split: train + temp vs test
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=labels,
        )

        # Second split: train vs validation
        train_labels = labels[train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=validation_ratio / (train_ratio + validation_ratio),
            random_state=self.random_state,
            stratify=train_labels,
        )

        # Store metadata
        metadata = {
            "cicids2017_policy": "stratified_3way_split",
            "data_source": "combined_cicids2017_csvs",
            "indices_relative_to": "combined_dataframe",
        }

        self.save_split_indices(train_indices, val_indices, test_indices, metadata)

    def create_or_load_splits(
        self,
        dataset_name: str,
        train_df: Optional["pd.DataFrame"] = None,
        test_df: Optional["pd.DataFrame"] = None,
        cicids2017_ratios: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create or load splits (main entry point).

        Args:
            dataset_name: Dataset name ('nsl-kdd' or 'cicids2017').
            train_df: Training dataframe (required for creation if splits don't exist).
            test_df: Test dataframe (required for NSL-KDD if splits don't exist).
            cicids2017_ratios: Dict with 'train', 'validation', 'test' keys (for CICIDS2017).

        Returns:
            Tuple of (train_indices, validation_indices, test_indices).

        Raises:
            ValueError: If dataset_name is invalid or required arguments missing.
        """
        if dataset_name not in ["nsl-kdd", "cicids2017"]:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Try to load existing splits
        if self.splits_exist():
            logger.info(f"Loading existing splits for {dataset_name}")
            return self.load_split_indices()

        # Create new splits
        logger.info(f"Creating new splits for {dataset_name}")

        if dataset_name == "nsl-kdd":
            if train_df is None or test_df is None:
                raise ValueError(
                    "train_df and test_df required for NSL-KDD split creation"
                )
            self.create_nsl_kdd_splits(train_df, test_df)
        elif dataset_name == "cicids2017":
            if train_df is None:
                raise ValueError("train_df required for CICIDS2017 split creation")
            
            ratios = cicids2017_ratios or {}
            train_ratio = ratios.get("train", 0.70)
            val_ratio = ratios.get("validation", 0.15)
            test_ratio = ratios.get("test", 0.15)
            
            self.create_cicids2017_splits(
                train_df,
                train_ratio=train_ratio,
                validation_ratio=val_ratio,
                test_ratio=test_ratio,
            )

        return self.load_split_indices()
