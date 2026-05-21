#!/usr/bin/env python3
"""Validate real CICIDS2017 CSV files under data/raw/cicids2017."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.graph_builder import FlowGraphBuilder
from data.preprocess import CICIDS2017Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_csv_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"CICIDS2017 raw directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CICIDS2017 CSV files found in {raw_dir}. "
            "Place one or more CSV files in this directory."
        )

    return csv_files


def inspect_files(csv_files: list[Path]) -> None:
    logger.info("Found %d CICIDS2017 CSV file(s)", len(csv_files))

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, nrows=5)
            logger.info(
                "%s: loaded %d sample rows, %d columns",
                csv_path.name,
                len(df),
                len(df.columns),
            )
            logger.info("First columns: %s", list(df.columns[:10]))
        except Exception as exc:
            raise RuntimeError(f"Failed to read {csv_path}: {exc}") from exc


def sample_dataframe(df: pd.DataFrame, sample_rows: int) -> pd.DataFrame:
    if sample_rows <= 0:
        logger.info("Using full dataset: %d rows", len(df))
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


def validate_dataset(raw_dir: Path, sample_rows: int, window_size: int, k: int) -> None:
    csv_files = find_csv_files(raw_dir)
    inspect_files(csv_files)

    preprocessor = CICIDS2017Preprocessor()
    df = preprocessor.load_data(str(raw_dir))

    logger.info(
        "Loaded combined CICIDS2017 data: %d rows, %d columns",
        len(df),
        len(df.columns),
    )

    df = sample_dataframe(df, sample_rows)

    X, y = preprocessor.preprocess(df, fit=True)

    logger.info("Preprocessing succeeded")
    logger.info("Feature matrix shape: %s", X.shape)
    logger.info("Label vector shape: %s", y.shape)
    logger.info("Feature count: %d", X.shape[1])

    unique, counts = np.unique(y, return_counts=True)
    label_distribution = dict(zip(unique.tolist(), counts.tolist()))
    logger.info("Label distribution: %s", label_distribution)

    logger.info(
        "Feature names (%d): %s",
        len(preprocessor.feature_names),
        preprocessor.feature_names[:10],
    )
    if len(preprocessor.feature_names) > 10:
        logger.info("... and %d more feature names", len(preprocessor.feature_names) - 10)

    logger.info(
        "Building flow-centric graphs with window_size=%d and k=%d",
        window_size,
        k,
    )

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

    logger.info("Graph construction succeeded")
    logger.info("Created %d graph(s)", len(graphs))
    logger.info(
        "First graph: nodes=%d, edges=%d, features=%d",
        graphs[0].x.shape[0],
        graphs[0].edge_index.shape[1],
        graphs[0].x.shape[1],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify real CICIDS2017 CSV files under data/raw/cicids2017"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "cicids2017",
        help="Directory containing CICIDS2017 CSV files",
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
        validate_dataset(
            raw_dir=args.raw_dir,
            sample_rows=args.sample_rows,
            window_size=args.window_size,
            k=args.k,
        )
        logger.info("CICIDS2017 validation completed successfully")
        return 0
    except Exception as exc:
        logger.error("CICIDS2017 validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())