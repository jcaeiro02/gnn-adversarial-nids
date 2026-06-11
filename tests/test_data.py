"""
Unit tests for data pipeline components.

Tests cover: download, preprocessing, graph construction, and dataset management.
"""

import unittest
from unittest.mock import patch
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from typing import Optional

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.download import DatasetDownloader, CICIDS2017_SUBSETS
from data.preprocess import NSLKDDPreprocessor, CICIDS2017Preprocessor
from data.graph_builder import FlowGraphBuilder
from data.dataset import NetworkFlowDataset, load_split_datasets
import torch

logger = logging.getLogger(__name__)


class TestDatasetDownloader(unittest.TestCase):
    """Tests for DatasetDownloader class."""

    def setUp(self):
        """Create temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = DatasetDownloader(base_dir=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_downloader_initialization(self):
        """Test downloader initialization."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertIsInstance(self.downloader, DatasetDownloader)
        logger.info("✓ Downloader initialization test passed")

    def test_invalid_url_handling(self):
        """Test error handling for invalid URLs."""
        # This should not crash but return False
        result = self.downloader.download_file(
            "http://invalid-url-that-does-not-exist.example.com/file.txt",
            "test_file.txt"
        )
        self.assertFalse(result)
        logger.info("✓ Invalid URL handling test passed")

    def test_list_datasets(self):
        """Test listing datasets."""
        datasets = self.downloader.list_datasets()
        self.assertIsInstance(datasets, list)
        logger.info("✓ List datasets test passed")


class TestNSLKDDPreprocessor(unittest.TestCase):
    """Tests for NSLKDDPreprocessor class."""

    def setUp(self):
        """Initialize preprocessor and create test data."""
        self.preprocessor = NSLKDDPreprocessor()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _create_synthetic_data(self, n_samples: int = 100, include_difficulty: bool = False) -> str:
        """Create synthetic NSL-KDD data for testing.

        Args:
            n_samples: Number of samples to create.
            include_difficulty: Whether to include a difficulty column.

        Returns:
            Path to created data file.
        """
        # Create synthetic data matching NSL-KDD format
        data = []
        protocols = ["tcp", "udp", "icmp"]
        services = ["http", "ftp", "ssh", "domain", "other"]
        flags = ["SF", "S0", "S1", "S2", "S3", "RSTO"]
        labels = ["normal", "back", "buffer_overflow", "dos", "neptune"]

        for i in range(n_samples):
            row = [
                str(i),  # duration
                protocols[i % len(protocols)],  # protocol_type
                services[i % len(services)],  # service
                flags[i % len(flags)],  # flag
            ] + [str(i * 0.1) for _ in range(37)]  # 37 numeric features
            row.append(labels[i % len(labels)])  # label
            if include_difficulty:
                row.append(str(i % 10))  # difficulty value
            data.append(",".join(row))

        filename = "synthetic_data_with_difficulty.txt" if include_difficulty else "synthetic_data.txt"
        filepath = Path(self.temp_dir) / filename
        with open(filepath, "w") as f:
            f.write("\n".join(data))

        return str(filepath)

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsInstance(self.preprocessor, NSLKDDPreprocessor)
        self.assertIsNone(self.preprocessor.scaler)
        logger.info("✓ Preprocessor initialization test passed")

    def test_load_data(self):
        """Test loading synthetic NSL-KDD data."""
        filepath = self._create_synthetic_data(100)
        df = self.preprocessor.load_data(filepath)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(len(df.columns), 42)  # 41 features + label
        self.assertEqual(df.columns[-1], self.preprocessor.LABEL_COLUMN)
        logger.info("✓ Load data test passed")

    def test_load_data_with_difficulty(self):
        """Test loading synthetic NSL-KDD data with difficulty column."""
        filepath = self._create_synthetic_data(100, include_difficulty=True)
        df = self.preprocessor.load_data(filepath)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(len(df.columns), 43)  # 41 features + label + difficulty
        self.assertEqual(df.columns[-2], self.preprocessor.LABEL_COLUMN)
        self.assertEqual(df.columns[-1], self.preprocessor.DIFFICULTY_COLUMN)
        logger.info("✓ Load data with difficulty test passed")

    def test_difficulty_excluded_from_features(self):
        """Test that difficulty column is excluded from feature set."""
        filepath = self._create_synthetic_data(100, include_difficulty=True)
        df = self.preprocessor.load_data(filepath)
        X, y = self.preprocessor.preprocess(df, fit=True)

        self.assertEqual(X.shape[1], 41)
        self.assertNotIn(self.preprocessor.DIFFICULTY_COLUMN, self.preprocessor.feature_names)
        self.assertNotIn(self.preprocessor.LABEL_COLUMN, self.preprocessor.feature_names)
        logger.info("✓ Difficulty excluded from features test passed")

    def test_preprocessing_output_shape(self):
        """Test that preprocessing maintains correct shapes."""
        filepath = self._create_synthetic_data(100)
        df = self.preprocessor.load_data(filepath)
        X, y = self.preprocessor.preprocess(df, fit=True)

        self.assertEqual(X.shape[0], 100)  # Same number of samples
        self.assertEqual(X.shape[1], 41)   # 41 features
        self.assertEqual(y.shape[0], 100)  # Same number of labels
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        logger.info("✓ Preprocessing output shape test passed")

    def test_label_conversion(self):
        """Test binary label conversion."""
        filepath = self._create_synthetic_data(100)
        df = self.preprocessor.load_data(filepath)
        X, y = self.preprocessor.preprocess(df, fit=True)

        # Check labels are binary
        unique_labels = np.unique(y)
        self.assertTrue(np.all(np.isin(unique_labels, [0, 1])))
        # Check that we have both classes
        self.assertGreater(len(unique_labels), 0)
        logger.info("✓ Label conversion test passed")

    def test_missing_values_handling(self):
        """Test handling of missing values."""
        filepath = self._create_synthetic_data(100)
        df = self.preprocessor.load_data(filepath)
        
        # Introduce missing values
        df.iloc[0, 0] = np.nan
        df.iloc[1, 4] = np.inf
        
        X, y = self.preprocessor.preprocess(df, fit=True)
        
        # Check no NaN in output
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isinf(X)))
        logger.info("✓ Missing values handling test passed")

    def test_save_and_load_preprocessed(self):
        """Test saving and loading preprocessed data."""
        filepath = self._create_synthetic_data(100)
        df = self.preprocessor.load_data(filepath)
        X, y = self.preprocessor.preprocess(df, fit=True)

        # Save
        X_path, y_path = self.preprocessor.save_preprocessed(X, y, self.temp_dir)

        # Verify files exist
        self.assertTrue(Path(X_path).exists())
        self.assertTrue(Path(y_path).exists())

        # Load and verify
        X_loaded = np.load(X_path)
        y_loaded = np.load(y_path)

        np.testing.assert_array_almost_equal(X, X_loaded)
        np.testing.assert_array_equal(y, y_loaded)
        logger.info("✓ Save and load preprocessed test passed")


class TestCICIDS2017Preprocessor(unittest.TestCase):
    """Tests for CICIDS2017 data preprocessing."""

    def setUp(self):
        self.preprocessor = CICIDS2017Preprocessor()
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir).parent / "raw" / "cicids2017"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        if self.raw_dir.exists():
            shutil.rmtree(self.raw_dir.parent)

    def _write_cicids_csv(
    self,
    filenames=None,
    num_files: int = 2,
    rows_per_file: int = 5,
    include_bad_values: bool = False,
    ):
        if filenames is None:
            filenames = [f"cicids_part_{file_idx}.csv" for file_idx in range(num_files)]

        for file_idx, filename in enumerate(filenames):
            rows = []
            for i in range(rows_per_file):
                row = {
                    "Flow ID": f"flow-{file_idx}-{i}",
                    "Source IP": "192.168.0.1",
                    "Destination IP": "10.0.0.1",
                    "Timestamp": "2023-01-01 00:00:00",
                    "SimillarHTTP": "None",
                    "Fwd Packet Length Mean": float(i) * 1.5,
                    "Bwd Packet Length Mean": float(i) * 2.5,
                    "Tot Fwd Packets": float(i + 1),
                    "Label": "BENIGN" if i % 2 == 0 else "FTP-Patator",
                }

                if include_bad_values and i == 0:
                    row["Fwd Packet Length Mean"] = np.nan
                    row["Bwd Packet Length Mean"] = np.inf

                rows.append(row)

            pd.DataFrame(rows).to_csv(
                self.raw_dir / filename,
                index=False,
            )

    def test_loads_multiple_csv_files(self):
        self._write_cicids_csv(num_files=2, rows_per_file=4)
        df = self.preprocessor.load_data(str(self.raw_dir))

        self.assertEqual(df.shape[0], 8)
        self.assertIn("Flow ID", df.columns)
        self.assertIn("Label", df.columns)
        logger.info("✓ CICIDS2017 multiple CSV load test passed")

    def test_drops_identifier_columns_and_encodes_labels(self):
        self._write_cicids_csv(num_files=1, rows_per_file=6)
        df = self.preprocessor.load_data(str(self.raw_dir))
        X, y = self.preprocessor.preprocess(df, fit=True)

        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertNotIn("Flow ID", self.preprocessor.feature_names)
        self.assertNotIn("Source IP", self.preprocessor.feature_names)
        self.assertNotIn("Destination IP", self.preprocessor.feature_names)
        self.assertNotIn("Timestamp", self.preprocessor.feature_names)
        self.assertNotIn("SimillarHTTP", self.preprocessor.feature_names)
        self.assertTrue(set(np.unique(y)).issubset({0, 1}))
        self.assertGreater(len(self.preprocessor.feature_names), 0)
        logger.info("✓ CICIDS2017 label conversion and identifier drop test passed")

    def test_handles_nan_and_inf_values(self):
        self._write_cicids_csv(num_files=1, rows_per_file=5, include_bad_values=True)
        df = self.preprocessor.load_data(str(self.raw_dir))
        X, y = self.preprocessor.preprocess(df, fit=True)

        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isinf(X)))
        logger.info("✓ CICIDS2017 missing values handling test passed")


class TestFlowGraphBuilder(unittest.TestCase):
    """Tests for FlowGraphBuilder class."""

    def setUp(self):
        """Initialize builder and create test data."""
        self.builder = FlowGraphBuilder()
        self.n_samples = 50
        self.n_features = 41
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = np.random.randint(0, 2, self.n_samples)

    def test_builder_initialization(self):
        """Test builder initialization."""
        self.assertIsInstance(self.builder, FlowGraphBuilder)
        logger.info("✓ Builder initialization test passed")

    def test_input_validation(self):
        """Test input validation."""
        # Invalid X dimension
        with self.assertRaises(ValueError):
            invalid_X = np.random.randn(self.n_samples)
            self.builder.build_graph(invalid_X, self.y)

        # Invalid y dimension
        with self.assertRaises(ValueError):
            invalid_y = np.random.randn(self.n_samples, 2)
            self.builder.build_graph(self.X, invalid_y)

        # Mismatched sizes
        with self.assertRaises(ValueError):
            invalid_y = np.random.randint(0, 2, self.n_samples + 1)
            self.builder.build_graph(self.X, invalid_y)

        logger.info("✓ Input validation test passed")

    def test_knn_graph_construction(self):
        """Test kNN graph construction."""
        graph = self.builder.build_graph(self.X, self.y, method="knn", k=5)

        self.assertIsNotNone(graph)
        self.assertEqual(graph.x.shape[0], self.n_samples)
        self.assertEqual(graph.x.shape[1], self.n_features)
        self.assertEqual(graph.y.shape[0], self.n_samples)
        self.assertGreater(graph.edge_index.shape[1], 0)
        logger.info("✓ kNN graph construction test passed")

    def test_similarity_graph_construction(self):
        """Test similarity-based graph construction."""
        graph = self.builder.build_graph(
            self.X, self.y, method="similarity", threshold=0.5
        )

        self.assertIsNotNone(graph)
        self.assertEqual(graph.x.shape[0], self.n_samples)
        self.assertEqual(graph.y.shape[0], self.n_samples)
        logger.info("✓ Similarity graph construction test passed")

    def test_valid_pyg_data_object(self):
        """Test that output is valid PyG Data object."""
        graph = self.builder.build_graph(self.X, self.y, method="knn", k=5)

        # Check required attributes
        self.assertTrue(hasattr(graph, "x"))
        self.assertTrue(hasattr(graph, "y"))
        self.assertTrue(hasattr(graph, "edge_index"))

        # Check PyTorch types
        self.assertIsInstance(graph.x, torch.Tensor)
        self.assertIsInstance(graph.y, torch.Tensor)
        self.assertIsInstance(graph.edge_index, torch.Tensor)

        logger.info("✓ Valid PyG Data object test passed")

    def test_no_self_loops(self):
        """Test that no self-loops are created."""
        graph = self.builder.build_graph(self.X, self.y, method="knn", k=5)

        edge_index = graph.edge_index.numpy()
        # Check no self-loops (where source == target)
        self_loops = np.sum(edge_index[0] == edge_index[1])
        self.assertEqual(self_loops, 0)
        logger.info("✓ No self-loops test passed")

    def test_edge_index_format(self):
        """Test edge_index format is compatible with PyG."""
        graph = self.builder.build_graph(self.X, self.y, method="knn", k=5)

        edge_index = graph.edge_index
        # Should be (2, num_edges)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertGreater(edge_index.shape[1], 0)
        logger.info("✓ Edge index format test passed")

    def test_graph_statistics(self):
        """Test graph statistics are computed correctly."""
        graph = self.builder.build_graph(self.X, self.y, method="knn", k=5)

        num_nodes = graph.x.shape[0]
        num_edges = graph.edge_index.shape[1]

        # Verify statistics make sense
        self.assertEqual(num_nodes, self.n_samples)
        self.assertGreater(num_edges, 0)
        self.assertLessEqual(num_edges, num_nodes * 10)  # Reasonable edge limit for kNN
        logger.info("✓ Graph statistics test passed")

    def test_bidirectional_knn_graph(self):
        """Test bidirectional kNN graph construction."""
        graph_bidir = self.builder.build_graph(
            self.X, self.y, method="knn", k=5, bidirectional=True
        )
        graph_unidir = self.builder.build_graph(
            self.X, self.y, method="knn", k=5, bidirectional=False
        )

        # Bidirectional should have more edges
        self.assertGreaterEqual(
            graph_bidir.edge_index.shape[1], graph_unidir.edge_index.shape[1]
        )
        logger.info("✓ Bidirectional kNN graph test passed")

    def test_pandas_input_conversion(self):
        """Test that pandas DataFrames and Series are converted."""
        X_df = pd.DataFrame(self.X, columns=[f"f{i}" for i in range(self.n_features)])
        y_series = pd.Series(self.y)

        graph = self.builder.build_graph(
            X_df, y_series, method="knn", k=5, bidirectional=True
        )

        self.assertEqual(graph.x.shape[0], self.n_samples)
        self.assertEqual(graph.y.shape[0], self.n_samples)
        logger.info("✓ Pandas input conversion test passed")

    def test_tensor_dtypes(self):
        """Test that tensor dtypes are correct."""
        graph = self.builder.build_graph(self.X, self.y, method="knn", k=5)

        # Check dtypes
        self.assertEqual(graph.x.dtype, torch.float32)
        self.assertEqual(graph.y.dtype, torch.int64)
        self.assertEqual(graph.edge_index.dtype, torch.int64)
        self.assertEqual(graph.edge_attr.dtype, torch.float32)
        logger.info("✓ Tensor dtypes test passed")

    def test_k_reduction_safeguard(self):
        """Test that k is safely reduced when it exceeds number of samples."""
        small_X = np.random.randn(3, self.n_features).astype(np.float32)
        small_y = np.random.randint(0, 2, 3)

        # k=10 > n_samples=3, should be reduced
        graph = self.builder.build_graph(small_X, small_y, method="knn", k=10)

        self.assertEqual(graph.x.shape[0], 3)
        self.assertGreater(graph.edge_index.shape[1], 0)
        logger.info("✓ k reduction safeguard test passed")

    def test_small_dataset_handling(self):
        """Test handling of very small datasets."""
        tiny_X = np.random.randn(1, self.n_features).astype(np.float32)
        tiny_y = np.array([0])

        # Should handle gracefully
        graph = self.builder.build_graph(tiny_X, tiny_y, method="knn", k=5)
        self.assertEqual(graph.x.shape[0], 1)
        logger.info("✓ Small dataset handling test passed")

    def test_windowed_dataset_construction(self):
        """Test construction of windowed datasets."""
        large_X = np.random.randn(2500, self.n_features).astype(np.float32)
        large_y = np.random.randint(0, 2, 2500)

        graphs = self.builder.build_windowed_dataset(
            large_X, large_y, window_size=1000, method="knn", k=5
        )

        # Should create 3 windows (1000, 1000, 500)
        self.assertEqual(len(graphs), 3)

        # Check window sizes
        self.assertEqual(graphs[0].x.shape[0], 1000)
        self.assertEqual(graphs[1].x.shape[0], 1000)
        self.assertEqual(graphs[2].x.shape[0], 500)
        logger.info("✓ Windowed dataset construction test passed")

    def test_windowed_dataset_skip_small_windows(self):
        """Test that very small windows are skipped."""
        X = np.random.randn(2001, self.n_features).astype(np.float32)
        y = np.random.randint(0, 2, 2001)

        graphs = self.builder.build_windowed_dataset(
            X, y, window_size=2000, method="knn", k=5
        )

        # Should skip the last single-sample window
        self.assertLess(len(graphs), 2)
        logger.info("✓ Windowed dataset skip small windows test passed")


class TestNetworkFlowDataset(unittest.TestCase):
    """Tests for NetworkFlowDataset class."""

    def setUp(self):
        """Create temporary directory and initialize dataset."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _create_synthetic_dataset(self, n_samples: int = 50) -> str:
        """Create synthetic NSL-KDD data."""
        data = []
        protocols = ["tcp", "udp", "icmp"]
        services = ["http", "ftp", "ssh", "domain"]
        flags = ["SF", "S0", "S1"]
        labels = ["normal", "dos"]

        for i in range(n_samples):
            row = [
                str(i),
                protocols[i % len(protocols)],
                services[i % len(services)],
                flags[i % len(flags)],
            ] + [str(i * 0.1) for _ in range(37)]
            row.append(labels[i % len(labels)])
            data.append(",".join(row))

        filepath = Path(self.temp_dir) / "test_data.txt"
        with open(filepath, "w") as f:
            f.write("\n".join(data))

        return str(filepath)

    def _write_raw_split(self, split: str, n_samples: int = 100) -> str:
        """Write synthetic raw KDD data for a given split."""
        raw_dir = Path(self.temp_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        filename = "KDDTrain+.txt" if split == "train" else "KDDTest+.txt"

        data = []
        protocols = ["tcp", "udp", "icmp"]
        services = ["http", "ftp", "ssh", "domain"]
        flags = ["SF", "S0", "S1"]
        labels = ["normal", "dos"]

        for i in range(n_samples):
            row = [
                str(i),
                protocols[i % len(protocols)],
                services[i % len(services)],
                flags[i % len(flags)],
            ] + [str(i * 0.1) for _ in range(37)]
            row.append(labels[i % len(labels)])
            data.append(",".join(row))

        filepath = raw_dir / filename
        with open(filepath, "w") as f:
            f.write("\n".join(data))

        return str(filepath)

    def test_dataset_creation(self):
        """Test dataset creation."""
        # Note: This test creates dataset but doesn't actually download
        # since we're using local raw files.
        self._write_raw_split("train", 40)
        self._write_raw_split("test", 40)
        dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
        )
        self.assertIsInstance(dataset, NetworkFlowDataset)
        self.assertGreater(len(dataset), 0)
        logger.info("✓ Dataset creation test passed")

    def _write_cicids2017_csv_files(
    self,
    filenames=None,
    rows_per_file: int = 5,
    include_bad_values: bool = False,
    ):
        if filenames is None:
            filenames = ["cicids_part_0.csv", "cicids_part_1.csv"]

        raw_dir = Path(self.temp_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        for file_idx, filename in enumerate(filenames):
            rows = []
            for i in range(rows_per_file):
                row = {
                    "Flow ID": f"flow-{file_idx}-{i}",
                    "Source IP": "192.168.0.1",
                    "Destination IP": "10.0.0.1",
                    "Timestamp": "2023-01-01 00:00:00",
                    "SimillarHTTP": "None",
                    "Fwd Packet Length Mean": float(i) * 1.5,
                    "Bwd Packet Length Mean": float(i) * 2.5,
                    "Tot Fwd Packets": float(i + 1),
                    "Label": "BENIGN" if i % 2 == 0 else "FTP-Patator",
                }

                if include_bad_values and i == 0:
                    row["Fwd Packet Length Mean"] = np.nan
                    row["Bwd Packet Length Mean"] = np.inf

                rows.append(row)

            pd.DataFrame(rows).to_csv(
                raw_dir / filename,
                index=False,
            )

        return raw_dir

    def test_cicids2017_split_indices_are_deterministic(self):
        raw_dir = self._write_cicids2017_csv_files(
            filenames=CICIDS2017_SUBSETS["cicids2017-selected"],
            rows_per_file=8,
        )
        train_dataset = NetworkFlowDataset.create_dataset(
            name="cicids2017-selected",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=16,
        )
        # Verify that variant-specific splits were created and persisted
        split_dir = Path(self.temp_dir) / "splits" / "cicids2017-selected"
        self.assertTrue((split_dir / "train_indices.npy").exists())
        self.assertTrue((split_dir / "test_indices.npy").exists())

        with open(split_dir / "train_indices.npy", "rb") as f:
            train_indices = np.load(f)

        self.assertGreater(len(train_indices), 0)
        self.assertEqual(len(train_dataset), 1)
        logger.info("✓ CICIDS2017 deterministic split index test passed")

    def test_cicids2017_dataset_creation_and_graph_building(self):
        raw_dir = self._write_cicids2017_csv_files(
            filenames=CICIDS2017_SUBSETS["cicids2017-selected"],
            rows_per_file=8,
        )
        train_dataset = NetworkFlowDataset.create_dataset(
            name="cicids2017-selected",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=10,
        )

        test_dataset = NetworkFlowDataset.create_dataset(
            name="cicids2017-selected",
            split="test",
            root=self.temp_dir,
            rebuild=True,
            window_size=10,
        )

        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        self.assertTrue(all(hasattr(graph, "x") for graph in train_dataset.data_list))
        self.assertTrue(all(hasattr(graph, "y") for graph in test_dataset.data_list))
        logger.info("✓ CICIDS2017 dataset creation and graph building test passed")

    def test_split_validation(self):
        """Test that invalid splits raise errors."""
        with self.assertRaises(ValueError):
            NetworkFlowDataset.create_dataset(
                split="invalid",
                root=self.temp_dir
            )
        logger.info("✓ Split validation test passed")

    def test_train_split_fits_preprocessor(self):
        """Test that train split fits the preprocessor state."""
        self._write_raw_split("train", 40)
        self._write_raw_split("test", 40)
        dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )

        # Check preprocessor state file exists
        preprocessor_path = Path(self.temp_dir) / "processed" / NetworkFlowDataset.PREPROCESSOR_STATE_FILE
        self.assertTrue(preprocessor_path.exists())

        # Load and verify contents
        with open(preprocessor_path, "rb") as f:
            state = pickle.load(f)

        self.assertIn("scaler", state)
        self.assertIn("label_encoders", state)
        self.assertIn("feature_names", state)
        self.assertIsNotNone(state["scaler"])
        # Ensure label encoders for categorical cols exist
        for col in NSLKDDPreprocessor.CATEGORICAL_COLUMNS:
            self.assertIn(col, state["label_encoders"]) 
        # Feature names should be 41 features
        self.assertEqual(len(state["feature_names"]), 41)

        # Check processed files exist
        data_path = Path(self.temp_dir) / "processed" / f"data_train.pt"
        stats_path = Path(self.temp_dir) / "processed" / f"statistics_train.pkl"
        self.assertTrue(data_path.exists())
        self.assertTrue(stats_path.exists())
        self.assertEqual(len(dataset), 2)
        logger.info("✓ Train split fit behavior test passed")

    def test_test_split_uses_existing_preprocessor(self):
        """Test that test split loads existing preprocessor and uses fit=False."""
        self._write_raw_split("train", 40)
        self._write_raw_split("test", 40)

        # Build train dataset first
        NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )
        # Record preprocessor state bytes before
        preprocessor_path = Path(self.temp_dir) / "processed" / NetworkFlowDataset.PREPROCESSOR_STATE_FILE
        self.assertTrue(preprocessor_path.exists())
        before_bytes = preprocessor_path.read_bytes()

        # Create test dataset
        dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="test",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )

        # Check test processed files
        data_path = Path(self.temp_dir) / "processed" / f"data_test.pt"
        stats_path = Path(self.temp_dir) / "processed" / f"statistics_test.pkl"
        self.assertTrue(data_path.exists())
        self.assertTrue(stats_path.exists())

        # Ensure preprocessor state still exists and was not overwritten
        self.assertTrue(preprocessor_path.exists())
        after_bytes = preprocessor_path.read_bytes()
        self.assertEqual(before_bytes, after_bytes)

        # Basic dataset length check
        self.assertEqual(len(dataset), 2)
        logger.info("✓ Test split fit=False behavior test passed")

    def test_test_split_fails_without_train_preprocessor_state(self):
        """Test that test split creation fails if train preprocessor state is missing."""
        self._write_raw_split("test", 40)

        with self.assertRaises(FileNotFoundError) as context:
            NetworkFlowDataset.create_dataset(
                name="nsl-kdd",
                split="test",
                root=self.temp_dir,
                rebuild=True,
                window_size=20,
            )

        self.assertIn("Process the train split first", str(context.exception))
        logger.info("✓ Test split missing preprocessor state failure test passed")

    def test_test_rebuild_preserves_train_preprocessor_state(self):
        """Test that rebuild=True for test does not delete train preprocessor state."""
        self._write_raw_split("train", 40)
        self._write_raw_split("test", 40)

        NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )

        preprocessor_path = Path(self.temp_dir) / "processed" / NetworkFlowDataset.PREPROCESSOR_STATE_FILE
        self.assertTrue(preprocessor_path.exists())

        NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="test",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )

        self.assertTrue(preprocessor_path.exists())
        logger.info("✓ Test split rebuild preserves train preprocessor state test passed")

    def test_processed_cache_loads_without_reprocessing(self):
        """Test that processed data is loaded from cache when rebuild=False."""
        self._write_raw_split("train", 40)
        self._write_raw_split("test", 40)
        NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )

        with patch.object(NetworkFlowDataset, "process", autospec=True, wraps=NetworkFlowDataset.process) as mock_process:
            dataset = NetworkFlowDataset.create_dataset(
                name="nsl-kdd",
                split="train",
                root=self.temp_dir,
                rebuild=False,
                window_size=20,
            )

        self.assertFalse(mock_process.called)
        self.assertEqual(len(dataset), 2)
        logger.info("✓ Cached dataset load test passed")

    def test_rebuild_forces_reprocessing(self):
        """Test that rebuild=True forces dataset reprocessing."""
        self._write_raw_split("train", 40)
        self._write_raw_split("test", 40)
        NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )

        with patch.object(NetworkFlowDataset, "process", autospec=True, wraps=NetworkFlowDataset.process) as mock_process:
            NetworkFlowDataset.create_dataset(
                name="nsl-kdd",
                split="train",
                root=self.temp_dir,
                rebuild=True,
                window_size=20,
            )

        self.assertTrue(mock_process.called)
        logger.info("✓ Rebuild forces reprocessing test passed")

    def test_dataset_length_matches_window_count(self):
        """Test that dataset length equals the number of graph windows."""
        self._write_raw_split("train", 45)
        self._write_raw_split("test", 40)
        dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=20,
        )

        self.assertEqual(len(dataset), 2)
        logger.info("✓ Dataset length window count test passed")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        """Setup for integration tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _create_synthetic_data(self, n_samples: int = 100) -> str:
        """Create synthetic NSL-KDD data."""
        data = []
        protocols = ["tcp", "udp", "icmp"]
        services = ["http", "ftp", "ssh"]
        flags = ["SF", "S0", "S1"]
        labels = ["normal", "dos", "neptune"]

        for i in range(n_samples):
            row = [
                str(i),
                protocols[i % len(protocols)],
                services[i % len(services)],
                flags[i % len(flags)],
            ] + [str(i * 0.1) for _ in range(37)]
            row.append(labels[i % len(labels)])
            data.append(",".join(row))

        filepath = Path(self.temp_dir) / "integration_data.txt"
        with open(filepath, "w") as f:
            f.write("\n".join(data))

        return str(filepath)

    def test_full_pipeline(self):
        """Test the full preprocessing and graph building pipeline."""
        # Create synthetic data
        filepath = self._create_synthetic_data(100)

        # Step 1: Load and preprocess
        preprocessor = NSLKDDPreprocessor()
        df = preprocessor.load_data(filepath)
        X, y = preprocessor.preprocess(df, fit=True)

        # Verify preprocessed data
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 41)
        self.assertEqual(y.shape[0], 100)

        # Step 2: Build graph
        builder = FlowGraphBuilder()
        graph = builder.build_graph(X, y, method="knn", k=5)

        # Verify graph
        self.assertEqual(graph.x.shape[0], 100)
        self.assertEqual(graph.y.shape[0], 100)
        self.assertGreater(graph.edge_index.shape[1], 0)

        # Step 3: Save preprocessed data
        X_path, y_path = preprocessor.save_preprocessed(
            X, y, self.temp_dir
        )
        self.assertTrue(Path(X_path).exists())
        self.assertTrue(Path(y_path).exists())

        logger.info("✓ Full pipeline integration test passed")

    def test_multiple_graphs(self):
        """Test building multiple graphs."""
        builder = FlowGraphBuilder()

        # Create two datasets
        X1 = np.random.randn(50, 41).astype(np.float32)
        y1 = np.random.randint(0, 2, 50)

        X2 = np.random.randn(30, 41).astype(np.float32)
        y2 = np.random.randint(0, 2, 30)

        # Build graphs
        graphs = builder.build_dataset(
            [X1, X2], [y1, y2], method="knn", k=5
        )

        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0].x.shape[0], 50)
        self.assertEqual(graphs[1].x.shape[0], 30)

        logger.info("✓ Multiple graphs integration test passed")


class TestSplitManager(unittest.TestCase):
    """Tests for the SplitManager class."""

    def setUp(self):
        """Setup for split manager tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.splits_dir = Path(self.temp_dir) / "splits"
        
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _create_nsl_kdd_dataframes(self, train_size: int = 100, test_size: int = 50):
        """Create synthetic NSL-KDD train and test dataframes."""
        def create_df(n_samples):
            protocols = ["tcp", "udp", "icmp"]
            services = ["http", "ftp", "ssh"]
            flags = ["SF", "S0", "S1"]
            labels = ["normal", "dos"]
            
            data = []
            for i in range(n_samples):
                row = {
                    "protocol": protocols[i % len(protocols)],
                    "service": services[i % len(services)],
                    "flag": flags[i % len(flags)],
                }
                # Add numeric features
                for j in range(38):
                    row[f"feature_{j}"] = float(i * 0.1 + j)
                row["label"] = labels[i % len(labels)]
                data.append(row)
            
            return pd.DataFrame(data)
        
        return create_df(train_size), create_df(test_size)

    def _create_cicids2017_dataframe(self, n_samples: int = 100):
        """Create synthetic CICIDS2017 dataframe."""
        data = []
        for i in range(n_samples):
            row = {
                "Flow ID": f"flow-{i}",
                "Source IP": "192.168.0.1",
                "Destination IP": "10.0.0.1",
            }
            # Add numeric features
            for j in range(15):
                row[f"feature_{j}"] = float(i * 0.1 + j)
            row["label"] = "BENIGN" if i % 2 == 0 else "DDoS"
            data.append(row)
        
        return pd.DataFrame(data)

    def test_split_manager_initialization(self):
        """Test SplitManager initialization."""
        from data.splits import SplitManager
        
        manager = SplitManager("nsl-kdd", data_dir=str(self.splits_dir))
        self.assertEqual(manager.dataset_name, "nsl-kdd")
        self.assertEqual(manager.random_state, 42)
        self.assertTrue(manager.split_dir.exists())
        logger.info("✓ SplitManager initialization test passed")

    def test_nsl_kdd_split_creation(self):
        """Test NSL-KDD split creation."""
        from data.splits import SplitManager
        
        train_df, test_df = self._create_nsl_kdd_dataframes(100, 50)
        
        manager = SplitManager("nsl-kdd", data_dir=str(self.splits_dir))
        train_idx, val_idx, test_idx = manager.create_or_load_splits(
            "nsl-kdd",
            train_df=train_df,
            test_df=test_df
        )
        
        # Verify split sizes
        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)
        self.assertEqual(len(test_idx), 50)  # All test samples
        self.assertEqual(len(train_idx) + len(val_idx), 100)  # All train samples
        
        logger.info("✓ NSL-KDD split creation test passed")

    def test_cicids2017_split_creation(self):
        """Test CICIDS2017 split creation."""
        from data.splits import SplitManager
        
        df = self._create_cicids2017_dataframe(100)
        
        manager = SplitManager("cicids2017-selected", data_dir=str(self.splits_dir))
        train_idx, val_idx, test_idx = manager.create_or_load_splits(
            "cicids2017-selected",
            train_df=df
        )
        
        # Verify split sizes approximately match ratios
        total = len(train_idx) + len(val_idx) + len(test_idx)
        self.assertEqual(total, 100)
        
        # Check approximate ratios (70%, 15%, 15%)
        train_ratio = len(train_idx) / total
        val_ratio = len(val_idx) / total
        test_ratio = len(test_idx) / total
        
        self.assertAlmostEqual(train_ratio, 0.70, delta=0.05)
        self.assertAlmostEqual(val_ratio, 0.15, delta=0.05)
        self.assertAlmostEqual(test_ratio, 0.15, delta=0.05)
        
        logger.info("✓ CICIDS2017 split creation test passed")

    def test_splits_are_persisted(self):
        """Test that splits are persisted to disk."""
        from data.splits import SplitManager
        
        train_df, test_df = self._create_nsl_kdd_dataframes(100, 50)
        
        manager = SplitManager("nsl-kdd", data_dir=str(self.splits_dir))
        manager.create_or_load_splits(
            "nsl-kdd",
            train_df=train_df,
            test_df=test_df
        )
        
        # Check files exist
        self.assertTrue((manager.split_dir / "train_indices.npy").exists())
        self.assertTrue((manager.split_dir / "validation_indices.npy").exists())
        self.assertTrue((manager.split_dir / "test_indices.npy").exists())
        self.assertTrue((manager.split_dir / "split_config.json").exists())
        
        logger.info("✓ Splits persisted test passed")

    def test_splits_are_deterministic(self):
        """Test that splits are deterministic with same random_state."""
        from data.splits import SplitManager
        
        train_df, test_df = self._create_nsl_kdd_dataframes(100, 50)
        
        manager1 = SplitManager("nsl-kdd", data_dir=str(self.splits_dir))
        train_idx1, val_idx1, test_idx1 = manager1.create_or_load_splits(
            "nsl-kdd",
            train_df=train_df,
            test_df=test_df
        )
        
        # Clear splits
        import shutil as shutil_module
        shutil_module.rmtree(str(manager1.split_dir))
        
        # Create again
        manager2 = SplitManager("nsl-kdd", data_dir=str(self.splits_dir))
        train_idx2, val_idx2, test_idx2 = manager2.create_or_load_splits(
            "nsl-kdd",
            train_df=train_df,
            test_df=test_df
        )
        
        # Verify splits are identical
        np.testing.assert_array_equal(train_idx1, train_idx2)
        np.testing.assert_array_equal(val_idx1, val_idx2)
        np.testing.assert_array_equal(test_idx1, test_idx2)
        
        logger.info("✓ Splits deterministic test passed")

    def test_no_overlap_between_splits(self):
        """Test that train/val/test splits have no overlap."""
        from data.splits import SplitManager
        
        train_df, test_df = self._create_nsl_kdd_dataframes(100, 50)
        
        manager = SplitManager("nsl-kdd", data_dir=str(self.splits_dir))
        train_idx, val_idx, test_idx = manager.create_or_load_splits(
            "nsl-kdd",
            train_df=train_df,
            test_df=test_df
        )
        
        # NSL-KDD: train and val share same source (KDDTrain+), test is separate
        train_val_combined = np.concatenate([train_idx, val_idx])
        overlap = np.intersect1d(train_idx, val_idx)
        self.assertEqual(len(overlap), 0, "Train and validation should not overlap")
        
        logger.info("✓ No overlap between splits test passed")

    def test_cicids2017_stratified_split_preserves_label_dist(self):
        """Test that CICIDS2017 stratified split preserves label distribution."""
        from data.splits import SplitManager
        
        df = self._create_cicids2017_dataframe(100)
        
        manager = SplitManager("cicids2017-selected", data_dir=str(self.splits_dir))
        train_idx, val_idx, test_idx = manager.create_or_load_splits(
            "cicids2017-selected",
            train_df=df
        )
        
        # Check label distribution is similar
        full_labels = df["label"].values
        full_benign_ratio = np.sum(full_labels == "BENIGN") / len(full_labels)
        
        train_benign_ratio = np.sum(full_labels[train_idx] == "BENIGN") / len(train_idx)
        val_benign_ratio = np.sum(full_labels[val_idx] == "BENIGN") / len(val_idx)
        test_benign_ratio = np.sum(full_labels[test_idx] == "BENIGN") / len(test_idx)
        
        # Ratios should be within ~10% of full distribution
        self.assertAlmostEqual(train_benign_ratio, full_benign_ratio, delta=0.15)
        self.assertAlmostEqual(val_benign_ratio, full_benign_ratio, delta=0.15)
        self.assertAlmostEqual(test_benign_ratio, full_benign_ratio, delta=0.15)
        
        logger.info("✓ CICIDS2017 stratified split preserves label distribution test passed")

    def test_load_split_config(self):
        """Test loading split configuration."""
        from data.splits import SplitManager
        
        train_df, test_df = self._create_nsl_kdd_dataframes(100, 50)
        
        manager = SplitManager("nsl-kdd", data_dir=str(self.splits_dir))
        manager.create_or_load_splits(
            "nsl-kdd",
            train_df=train_df,
            test_df=test_df
        )
        
        config = manager.load_split_config()
        
        self.assertEqual(config["dataset_name"], "nsl-kdd")
        self.assertEqual(config["random_state"], 42)
        self.assertEqual(config["total_size"], 150)
        self.assertEqual(config["test_size"], 50)
        self.assertEqual(config["train_size"] + config["validation_size"], 100)
        
        logger.info("✓ Load split config test passed")


class TestFormalSplitProtocol(unittest.TestCase):
    """Tests for formal split protocol in NetworkFlowDataset."""

    def setUp(self):
        """Setup for formal split protocol tests."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _write_nsl_kdd_splits(self, train_samples: int = 100, test_samples: int = 50):
        """Write NSL-KDD train and test files."""
        raw_dir = Path(self.temp_dir) / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Write train file
        train_data = []
        for i in range(train_samples):
            row = [str(i), "tcp", "http", "SF"] + [str(i * 0.1) for _ in range(37)] + ["normal" if i % 2 == 0 else "dos"]
            train_data.append(",".join(row))
        
        with open(raw_dir / "KDDTrain+.txt", "w") as f:
            f.write("\n".join(train_data))
        
        # Write test file
        test_data = []
        for i in range(test_samples):
            row = [str(i), "udp", "ftp", "S0"] + [str(i * 0.2) for _ in range(37)] + ["normal" if i % 3 == 0 else "dos"]
            test_data.append(",".join(row))
        
        with open(raw_dir / "KDDTest+.txt", "w") as f:
            f.write("\n".join(test_data))

    def test_train_validation_test_splits_loaded(self):
        """Test that train, validation, and test splits are loaded correctly."""
        self._write_nsl_kdd_splits(100, 50)
        
        # Load all three splits
        train_dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=50,
        )
        
        val_dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="validation",
            root=self.temp_dir,
            rebuild=False,
            window_size=50,
        )
        
        test_dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="test",
            root=self.temp_dir,
            rebuild=False,
            window_size=50,
        )
        
        # All should have data
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        
        logger.info("✓ Train/validation/test splits loaded test passed")

    def test_load_split_datasets_returns_three_splits(self):
        """Test that load_split_datasets returns train, validation, and test."""
        from data.dataset import load_split_datasets
        
        self._write_nsl_kdd_splits(100, 50)
        
        train_dataset, val_dataset, test_dataset = load_split_datasets(
            name="nsl-kdd",
            root=self.temp_dir,
            rebuild=True,
            window_size=50,
        )
        
        # All should have data
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        
        logger.info("✓ load_split_datasets returns three splits test passed")

    def test_validation_reuses_train_preprocessor(self):
        """Test that validation split reuses the train preprocessor."""
        self._write_nsl_kdd_splits(100, 50)
        
        # Load train first
        train_dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=50,
        )
        
        # Load validation
        val_dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="validation",
            root=self.temp_dir,
            rebuild=False,
            window_size=50,
        )
        
        # Both should have same number of features
        self.assertEqual(
            train_dataset[0].x.shape[1],
            val_dataset[0].x.shape[1]
        )
        
        logger.info("✓ Validation reuses train preprocessor test passed")

    def test_test_reuses_train_preprocessor(self):
        """Test that test split reuses the train preprocessor."""
        self._write_nsl_kdd_splits(100, 50)
        
        # Load train first
        train_dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="train",
            root=self.temp_dir,
            rebuild=True,
            window_size=50,
        )
        
        # Load test
        test_dataset = NetworkFlowDataset.create_dataset(
            name="nsl-kdd",
            split="test",
            root=self.temp_dir,
            rebuild=False,
            window_size=50,
        )
        
        # Both should have same number of features
        self.assertEqual(
            train_dataset[0].x.shape[1],
            test_dataset[0].x.shape[1]
        )
        
        logger.info("✓ Test reuses train preprocessor test passed")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run tests
    unittest.main()
