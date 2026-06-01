"""Dataset feature analysis and leakage detection utility.

Produces per-split statistics, correlations, mutual information and a leakage report.

Usage (CLI):
    python scripts/analyze_dataset_features.py --dataset nsl-kdd
    python scripts/analyze_dataset_features.py --dataset cicids2017

The script reuses existing preprocessors and the SplitManager to ensure
the same splits and encodings as the training pipeline.

Feature statistics, correlations and mutual information are computed on
interpretable numeric feature representations derived from the original
DataFrame, while preprocessing is used to ensure consistent feature selection
and categorical encoding.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from data.preprocess import NSLKDDPreprocessor, CICIDS2017Preprocessor
from data.splits import SplitManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _make_output_dir(base: str, dataset: str, timestamp: Optional[str] = None) -> Path:
    ts = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / dataset / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _series_numeric_for_feature(df: pd.DataFrame, feature: str, preprocessor) -> pd.Series:
    """Return a numeric Series for feature, encoding categoricals using preprocessor if available."""
    if feature in df.columns:
        s = df[feature].copy()
    else:
        return pd.Series([], dtype=float)

    if hasattr(preprocessor, "label_encoders") and preprocessor.label_encoders:
        encs = getattr(preprocessor, "label_encoders", {})
        if feature in encs:
            enc = encs[feature]
            vals = s.astype(str).tolist()
            mapping = {lab: idx for idx, lab in enumerate(enc.classes_)}
            numeric = [mapping.get(v, len(enc.classes_)) for v in vals]
            return pd.Series(numeric)

    return pd.to_numeric(s, errors="coerce")


def analyze_from_dataframes(
    dataset: str,
    train_df: pd.DataFrame,
    validation_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    output_base: str = "results/analysis",
    timestamp: Optional[str] = None,
) -> Path:
    """Analyze dataset given loaded DataFrames (useful for tests).

    For NSL-KDD provide `train_df` and `test_df`. For CICIDS2017 provide `train_df`
    containing all rows.
    """
    out_dir = _make_output_dir(output_base, dataset, timestamp)

    warnings = []

    # If explicit validation and test DataFrames are provided, use them and do not
    # consult persisted SplitManager indices. This mode is used by unit tests to
    # stay isolated from repository split files.
    explicit_splits = validation_df is not None and test_df is not None

    if explicit_splits:
        if dataset == "nsl-kdd":
            preprocessor = NSLKDDPreprocessor()
            splits = {
                "train": (train_df.reset_index(drop=True), True),
                "validation": (validation_df.reset_index(drop=True), False),
                "test": (test_df.reset_index(drop=True), False),
            }
        elif dataset == "cicids2017":
            preprocessor = CICIDS2017Preprocessor()
            splits = {
                "train": (train_df.reset_index(drop=True), True),
                "validation": (validation_df.reset_index(drop=True), False),
                "test": (test_df.reset_index(drop=True), False),
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    else:
        # CLI / real-data mode: use SplitManager persisted indices
        split_manager = SplitManager(dataset_name=dataset, data_dir="data/splits")
        if dataset == "nsl-kdd":
            if test_df is None:
                raise ValueError("test_df required for nsl-kdd when not using explicit splits")
            preprocessor = NSLKDDPreprocessor()
            train_indices, val_indices, test_indices = split_manager.create_or_load_splits(
                dataset_name="nsl-kdd", train_df=train_df, test_df=test_df
            )

            splits = {
                "train": (train_df.iloc[train_indices].reset_index(drop=True), True),
                "validation": (train_df.iloc[val_indices].reset_index(drop=True), False),
                "test": (test_df.iloc[test_indices].reset_index(drop=True), False),
            }
        elif dataset == "cicids2017":
            preprocessor = CICIDS2017Preprocessor()
            train_indices, val_indices, test_indices = split_manager.create_or_load_splits(
                dataset_name="cicids2017", train_df=train_df
            )
            full_df = train_df
            splits = {
                "train": (full_df.iloc[train_indices].reset_index(drop=True), True),
                "validation": (full_df.iloc[val_indices].reset_index(drop=True), False),
                "test": (full_df.iloc[test_indices].reset_index(drop=True), False),
            }
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    class_distribution: Dict[str, Dict] = {}

    feature_stats_rows = []
    correlation_rows = []
    mi_rows = []

    trained = False
    feature_names = None

    for split_name, (df_split, fit_flag) in splits.items():
        logger.info(f"Analyzing split: {split_name} (n={len(df_split)})")

        if fit_flag:
            X, y = preprocessor.preprocess(df_split, fit=True)
            trained = True
        else:
            if not trained and hasattr(preprocessor, "load_preprocessed"):
                try:
                    preprocessor.load_preprocessed("data/graphs/processed/preprocessor_state.pkl")
                except Exception:
                    pass
            X, y = preprocessor.preprocess(df_split, fit=False)

        if preprocessor.feature_names is not None:
            feature_names = preprocessor.feature_names

        unique, counts = np.unique(y, return_counts=True)
        class_distribution[split_name] = {int(k): int(v) for k, v in zip(unique, counts)}

        feature_series_map = {}
        for i, feat in enumerate(feature_names):
            ser = _series_numeric_for_feature(df_split, feat, preprocessor)
            missing = int(ser.isna().sum())
            ser = ser.fillna(ser.median() if not ser.dropna().empty else 0)
            feature_series_map[feat] = (ser, missing)

        X_df = pd.DataFrame({f: feature_series_map[f][0].astype(float) for f in feature_names})

        # Sanitize X_df to handle inf/-inf values (common in CICIDS2017 rate features)
        non_finite_count = int((~np.isfinite(X_df.to_numpy(dtype=float))).sum())
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
        
        if non_finite_count > 0:
            warnings.append(
                f"Cleaned {non_finite_count} non-finite values in split {split_name} before analysis."
            )

        # Compute feature statistics
        for f in feature_names:
            ser = X_df[f]
            missing = feature_series_map[f][1]
            mean = float(ser.mean()) if not ser.empty else float("nan")
            std = float(ser.std(ddof=0)) if not ser.empty else float("nan")
            var = float(ser.var(ddof=0)) if not ser.empty else float("nan")
            mn = float(ser.min()) if not ser.empty else float("nan")
            mx = float(ser.max()) if not ser.empty else float("nan")
            is_constant = bool(np.isclose(var, 0.0))
            near_constant = bool(var < 1e-6)

            feature_stats_rows.append({
                "split": split_name,
                "feature": f,
                "mean": mean,
                "std": std,
                "var": var,
                "min": mn,
                "max": mx,
                "missing_count": int(missing),
                "is_constant": is_constant,
                "near_constant": near_constant,
            })

        y_ser = pd.Series(y)
        n_samples = len(y)
        n_unique_labels = len(np.unique(y))

        for f in feature_names:
            feature_values = X_df[f]
            feature_unique = len(feature_values.dropna().unique())
            if n_samples < 2 or n_unique_labels < 2 or feature_unique < 2:
                corr = np.nan
                warnings.append(
                    f"Correlation skipped for split {split_name}, feature {f} due to insufficient data. "
                    f"n_samples={n_samples}, n_unique_labels={n_unique_labels}, feature_unique={feature_unique}"
                )
            else:
                corr = float(feature_values.corr(y_ser))
            correlation_rows.append({"split": split_name, "feature": f, "correlation": corr})
            if not np.isnan(corr) and abs(corr) > 0.95:
                warnings.append(f"High correlation |corr|>{0.95}: {f} in split {split_name} (corr={corr:.3f})")

        # Skip MI computation for very small splits or single-class splits
        if n_samples < 5 or n_unique_labels < 2:
            mi = np.zeros(len(feature_names))
            warnings.append(
                f"Mutual information skipped for split {split_name} due to insufficient samples. n_samples={n_samples}, n_unique_labels={n_unique_labels}"
            )
        else:
            try:
                mi = mutual_info_classif(X_df.values, y, discrete_features=False, random_state=42)
            except Exception:
                mi = np.array([
                    mutual_info_classif(X_df[[col]].values, y, discrete_features=False, random_state=42)[0]
                    for col in X_df.columns
                ])

        for f, v in zip(feature_names, mi.tolist()):
            mi_rows.append({"split": split_name, "feature": f, "mutual_information": float(v)})

        if len(mi) > 0:
            max_mi = float(np.max(mi))
            for f, v in zip(feature_names, mi.tolist()):
                if v >= max(1.0, 0.9 * max_mi):
                    warnings.append(f"High mutual information: {f} in split {split_name} (MI={v:.3f})")

        total = sum(class_distribution[split_name].values())
        if total > 0:
            mins = min(class_distribution[split_name].values())
            if mins / total < 0.05:
                warnings.append(f"Severe class imbalance in split {split_name}: {class_distribution[split_name]}")

        for f in feature_names:
            var = float(X_df[f].var(ddof=0))
            if np.isclose(var, 0.0) or var < 1e-8:
                warnings.append(f"Zero or near-zero variance: {f} in split {split_name} (var={var:.6g})")

    (out_dir / "class_distribution.json").write_text(json.dumps(class_distribution, indent=2))

    stats_df = pd.DataFrame(feature_stats_rows)
    stats_df.to_csv(out_dir / "feature_statistics.csv", index=False)

    corr_df = pd.DataFrame(correlation_rows)
    corr_df.to_csv(out_dir / "feature_label_correlation.csv", index=False)

    mi_df = pd.DataFrame(mi_rows)
    mi_df.to_csv(out_dir / "mutual_information.csv", index=False)

    report_lines = [f"# Leakage report for {dataset}\n\n"]
    report_lines.append("## Warnings\n")
    if warnings:
        for w in warnings:
            report_lines.append(f"- {w}\n")
    else:
        report_lines.append("- No issues detected.\n")

    report_lines.append("\nFeature statistics, correlations and mutual information are based on interpretable numeric feature representations derived from the original dataframe. Preprocessing is used to ensure consistent feature selection and categorical encoding.\n\n")
    report_lines.append("## Summary files\n")
    report_lines.append("- class_distribution.json\n")
    report_lines.append("- feature_statistics.csv\n")
    report_lines.append("- feature_label_correlation.csv\n")
    report_lines.append("- mutual_information.csv\n")

    (out_dir / "leakage_report.md").write_text("".join(report_lines))

    logger.info(f"Analysis saved to {out_dir}")
    return out_dir


def analyze_dataset_cli():
    p = argparse.ArgumentParser(description="Analyze dataset features and leakage.")
    p.add_argument("--dataset", choices=["nsl-kdd", "cicids2017"], required=True)
    p.add_argument("--output", default="results/analysis")
    args = p.parse_args()

    dataset = args.dataset
    out_base = args.output

    if dataset == "nsl-kdd":
        pre = NSLKDDPreprocessor()
        train_file = os.path.join("data/raw/nsl-kdd", "KDDTrain+.txt")
        test_file = os.path.join("data/raw/nsl-kdd", "KDDTest+.txt")
        train_df = pre.load_data(train_file)
        test_df = pre.load_data(test_file)
        analyze_from_dataframes(
            dataset,
            train_df,
            test_df=test_df,
            output_base=out_base,
        )
    else:
        pre = CICIDS2017Preprocessor()
        raw_dir = os.path.join("data/raw/cicids2017")
        df = pre.load_data(raw_dir)
        analyze_from_dataframes(
            dataset,
            df,
            output_base=out_base,
        )


if __name__ == "__main__":
    analyze_dataset_cli()
