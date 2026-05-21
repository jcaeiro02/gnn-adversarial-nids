#!/usr/bin/env python3
"""Validate real NSL-KDD files under data/raw/nsl-kdd."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.graph_builder import FlowGraphBuilder
from data.preprocess import NSLKDDPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


REQUIRED_FILES = {
    "train": ["KDDTrain+.txt"],
    "test": ["KDDTest+.txt"],
    "both": ["KDDTrain+.txt", "KDDTest+.txt"],
}


def validate_file_exists(raw_dir: Path, split: str) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"NSL-KDD raw directory not found: {raw_dir}")

    expected_files = REQUIRED_FILES[split]
    paths = []
    missing = []
    for filename in expected_files:
        path = raw_dir / filename
        if path.exists():
            paths.append(path)
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Missing NSL-KDD file(s) in {raw_dir}: {', '.join(missing)}"
        )

    return paths


def sample_dataframe(df: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    if sample_rows <= 0:
        logger.info("Using full dataset with %d rows", len(df))
        return df

    if len(df) <= sample_rows:
        logger.info("Dataset has %d rows; no sampling needed", len(df))
        return df

    logger.info(
        "Sampling %d rows from %d total rows using random_state=42",
        sample_rows,
        len(df),
    )
    return df.sample(n=sample_rows, random_state=42).reset_index(drop=True)


def summarize_raw_data(df: pd.DataFrame, file_path: Path) -> None:
    logger.info("Inspecting NSL-KDD file: %s", file_path.name)
    logger.info("Raw rows: %d", len(df))
    logger.info("Raw columns: %d", len(df.columns))
    logger.info("Difficulty column present: %s", "difficulty" in [c.lower() for c in df.columns])
    logger.info("First 10 columns: %s", list(df.columns[:10]))

    if "label" in df.columns:
        label_preview = df["label"].value_counts().head(10).to_dict()
        logger.info("Raw label distribution preview: %s", label_preview)
    else:
        logger.warning("Label column not found in raw NSL-KDD file: %s", file_path.name)


def preprocess_split(raw_file: Path, sample_rows: int) -> tuple[np.ndarray, np.ndarray, NSLKDDPreprocessor]:
    preprocessor = NSLKDDPreprocessor()
    df = preprocessor.load_data(str(raw_file))
    summarize_raw_data(df, raw_file)
    df = sample_dataframe(df, sample_rows)

    X, y = preprocessor.preprocess(df, fit=True)

    logger.info("Preprocessing succeeded for %s", raw_file.name)
    logger.info("X shape: %s", X.shape)
    logger.info("y shape: %s", y.shape)
    logger.info("Feature count: %d", X.shape[1])

    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    logger.info("Binary label distribution: %s", distribution)

    logger.info(
        "Feature names (%d): %s",
        len(preprocessor.feature_names),
        preprocessor.feature_names[:10],
    )
    if len(preprocessor.feature_names) > 10:
        logger.info("... and %d more feature names", len(preprocessor.feature_names) - 10)

    return X, y, preprocessor


def build_and_report_graphs(X: np.ndarray, y: np.ndarray, window_size: int, k: int) -> None:
    builder = FlowGraphBuilder()
    graphs = builder.build_windowed_dataset(
        X,
        y,
        window_size=window_size,
        method="knn",
        k=k,
        bidirectional=True,
    )

    if not graphs:
        raise RuntimeError("Graph construction failed: no graphs were created.")

    first_graph = graphs[0]
    if hasattr(first_graph.y, "numpy"):
        first_labels = first_graph.y.numpy()
    else:
        first_labels = np.array(first_graph.y)

    unique, counts = np.unique(first_labels, return_counts=True)
    label_distribution = dict(zip(unique.tolist(), counts.tolist()))

    logger.info("Graph construction succeeded")
    logger.info("Created %d graph(s)", len(graphs))
    logger.info(
        "First graph: nodes=%d, edges=%d, features=%d",
        first_graph.x.shape[0],
        first_graph.edge_index.shape[1],
        first_graph.x.shape[1],
    )
    logger.info("First graph label distribution: %s", label_distribution)


def validate_nslkdd(raw_dir: Path, split: str, sample_rows: int, window_size: int, k: int) -> None:
    files = validate_file_exists(raw_dir, split)

    for raw_file in files:
        X, y, _ = preprocess_split(raw_file, sample_rows)
        build_and_report_graphs(X, y, window_size, k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify real NSL-KDD files under data/raw/nsl-kdd"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "nsl-kdd",
        help="Directory containing NSL-KDD raw files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split(s) to validate",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=50000,
        help="Maximum number of rows to validate. Use 0 to load all rows.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Number of flows/nodes per graph window.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbours for kNN graph construction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_nslkdd(
            raw_dir=args.raw_dir,
            split=args.split,
            sample_rows=args.sample_rows,
            window_size=args.window_size,
            k=args.k,
        )
        logger.info("NSL-KDD validation completed successfully")
        return 0
    except FileNotFoundError as exc:
        logger.error("NSL-KDD validation failed: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("NSL-KDD validation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("NSL-KDD validation failed: unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
