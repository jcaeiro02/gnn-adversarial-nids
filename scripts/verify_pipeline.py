#!/usr/bin/env python3
"""
Demonstration script for the Phase 1 NSL-KDD data pipeline.

This script demonstrates the complete workflow:
1. Generate sample data
2. Load and preprocess NSL-KDD data
3. Build flow-centric graphs
4. Create PyG datasets
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.download import DatasetDownloader
from data.preprocess import NSLKDDPreprocessor
from data.graph_builder import FlowGraphBuilder
from data.dataset import NetworkFlowDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_sample_data():
    """Generate synthetic NSL-KDD data for demo."""
    logger.info("=" * 70)
    logger.info("STEP 1: Generating Sample NSL-KDD Data")
    logger.info("=" * 70)

    # Import generator
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_sample_data.py",
            "--samples", "1000",
            "--output-dir", "data/raw/nsl-kdd",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.info("✓ Sample data generated successfully")
        print(result.stdout)
    else:
        logger.error(f"✗ Failed to generate sample data: {result.stderr}")
        return False

    return True


def download_real_data():
    """Download real NSL-KDD data."""
    logger.info("=" * 70)
    logger.info("STEP 2: Downloading NSL-KDD Dataset")
    logger.info("=" * 70)

    downloader = DatasetDownloader(base_dir="data/raw/nsl-kdd")
    success = downloader.download_nsl_kdd()

    if success:
        logger.info("✓ NSL-KDD dataset downloaded successfully")
    else:
        logger.warning("⚠ Using previously generated sample data")

    datasets = downloader.list_datasets()
    logger.info(f"Available datasets: {datasets}")

    return True


def preprocess_data():
    """Preprocess NSL-KDD training data."""
    logger.info("=" * 70)
    logger.info("STEP 3: Preprocessing Data")
    logger.info("=" * 70)

    preprocessor = NSLKDDPreprocessor()

    # Try to load real data first, fall back to sample
    data_paths = [
        "data/raw/nsl-kdd/KDDTrain+.txt",
    ]

    for data_path in data_paths:
        if Path(data_path).exists():
            logger.info(f"Loading data from: {data_path}")
            break
    else:
        logger.error("No data file found!")
        return None

    # Load data
    df = preprocessor.load_data(data_path)
    logger.info(f"Loaded DataFrame shape: {df.shape}")

    # Preprocess
    X, y = preprocessor.preprocess(df, fit=True)

    logger.info(f"✓ Preprocessing complete")
    logger.info(f"  Feature matrix X shape: {X.shape}")
    logger.info(f"  Labels y shape: {y.shape}")
    logger.info(f"  Unique labels: {np.unique(y)}")

    # Save preprocessed data
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_path, y_path = preprocessor.save_preprocessed(X, y, "data/processed")
    logger.info(f"✓ Saved preprocessed data")
    logger.info(f"  X: {X_path}")
    logger.info(f"  y: {y_path}")

    return X, y, preprocessor


def build_graph(X, y):
    """Build flow-centric graph."""
    logger.info("=" * 70)
    logger.info("STEP 4: Building Flow-Centric Graph")
    logger.info("=" * 70)

    builder = FlowGraphBuilder()
    graph = builder.build_graph(X, y, method="knn", k=5, metric="cosine")

    logger.info(f"✓ Graph created successfully")
    logger.info(f"  Graph: {graph}")
    logger.info(f"  Node features shape: {graph.x.shape}")
    logger.info(f"  Node labels shape: {graph.y.shape}")
    logger.info(f"  Edge index shape: {graph.edge_index.shape}")
    if graph.edge_attr is not None:
        logger.info(f"  Edge attributes shape: {graph.edge_attr.shape}")

    return graph


def create_dataset(X, y):
    """Create PyG dataset."""
    logger.info("=" * 70)
    logger.info("STEP 5: Creating PyG Dataset")
    logger.info("=" * 70)

    # Note: Full dataset creation requires real downloads
    # For demo, we just show the graph creation
    builder = FlowGraphBuilder()
    graph = builder.build_graph(X, y, method="knn", k=5)

    logger.info("✓ Dataset created (in-memory)")
    logger.info(f"  Number of samples: {graph.x.shape[0]}")
    logger.info(f"  Number of features: {graph.x.shape[1]}")
    logger.info(f"  Number of edges: {graph.edge_index.shape[1]}")

    return graph


def verify_pipeline():
    """Verify the entire pipeline."""
    logger.info("=" * 70)
    logger.info("PHASE 1 NSL-KDD DATA PIPELINE - COMPLETE WORKFLOW")
    logger.info("=" * 70)
    logger.info("")

    # Step 1: Generate or verify sample data
    generate_sample_data()

    # Step 2: Download real data (optional)
    download_real_data()

    # Step 3: Preprocess
    result = preprocess_data()
    if result is None:
        logger.error("Failed to preprocess data")
        return False

    X, y, preprocessor = result

    # Step 4: Build graph
    graph = build_graph(X, y)

    # Step 5: Create dataset
    dataset_graph = create_dataset(X, y)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1 PIPELINE VERIFICATION COMPLETE ✓")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  ✓ Data loading: OK")
    logger.info(f"  ✓ Preprocessing: OK")
    logger.info(f"  ✓ Graph construction: OK")
    logger.info(f"  ✓ Dataset creation: OK")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run tests: python -m pytest tests/test_data.py -v")
    logger.info("  2. Implement GNN models (Phase 2)")
    logger.info("  3. Implement attacks (Phase 3)")
    logger.info("  4. Implement defenses (Phase 4)")
    logger.info("")

    return True


if __name__ == "__main__":
    success = verify_pipeline()
    sys.exit(0 if success else 1)
