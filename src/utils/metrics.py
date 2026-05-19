"""Binary node classification metric utilities."""

from typing import Optional, Sequence, Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def _ensure_numpy(array: Sequence[Any]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array.ravel()
    return np.asarray(array, dtype=np.float64).ravel()


def binary_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    y_true_np = _ensure_numpy(y_true)
    y_pred_np = _ensure_numpy(y_pred)
    if y_true_np.size == 0:
        return 0.0
    return float(accuracy_score(y_true_np, y_pred_np))


def binary_precision(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    y_true_np = _ensure_numpy(y_true)
    y_pred_np = _ensure_numpy(y_pred)
    if y_true_np.size == 0:
        return 0.0
    return float(precision_score(y_true_np, y_pred_np, zero_division=0))


def binary_recall(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    y_true_np = _ensure_numpy(y_true)
    y_pred_np = _ensure_numpy(y_pred)
    if y_true_np.size == 0:
        return 0.0
    return float(recall_score(y_true_np, y_pred_np, zero_division=0))


def binary_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    y_true_np = _ensure_numpy(y_true)
    y_pred_np = _ensure_numpy(y_pred)
    if y_true_np.size == 0:
        return 0.0
    return float(f1_score(y_true_np, y_pred_np, zero_division=0))


def binary_roc_auc(y_true: Sequence[Any], y_scores: Sequence[Any]) -> Optional[float]:
    y_true_np = _ensure_numpy(y_true)
    if y_true_np.size == 0:
        return None
    unique_labels = np.unique(y_true_np)
    if unique_labels.size < 2:
        return None

    y_scores_np = _ensure_numpy(y_scores)
    if y_scores_np.ndim == 2 and y_scores_np.shape[1] == 2:
        y_scores_np = y_scores_np[:, 1]
    if y_scores_np.ndim != 1:
        return None

    try:
        return float(roc_auc_score(y_true_np, y_scores_np))
    except ValueError:
        return None


def binary_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_scores: Optional[Sequence[Any]] = None,
) -> Dict[str, Optional[float]]:
    metrics = {
        "accuracy": binary_accuracy(y_true, y_pred),
        "precision": binary_precision(y_true, y_pred),
        "recall": binary_recall(y_true, y_pred),
        "f1": binary_f1(y_true, y_pred),
        "roc_auc": None,
    }
    if y_scores is not None:
        metrics["roc_auc"] = binary_roc_auc(y_true, y_scores)
    return metrics
