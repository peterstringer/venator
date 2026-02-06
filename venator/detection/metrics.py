"""Evaluation metrics for anomaly detection performance.

AUROC is the primary metric (threshold-independent, standard for anomaly detection).
Also computes precision, recall, F1 at chosen thresholds, and generates threshold
curve data for visualization and analysis.

Methodology note (from CLAUDE.md):
    Report AUROC as primary metric. It's threshold-independent and standard
    for anomaly detection evaluation.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_detector(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics for anomaly detection.

    Args:
        scores: Anomaly scores of shape (n_samples,). Higher = more anomalous.
        labels: Ground truth of shape (n_samples,). 0 = normal, 1 = anomaly.
        threshold: If provided, compute binary classification metrics at this
            threshold (scores > threshold → predicted anomaly).

    Returns:
        Dict with metrics:
            auroc: Area Under ROC Curve (primary metric).
            auprc: Area Under Precision-Recall Curve.
            fpr_at_95_tpr: False positive rate when true positive rate = 95%.
        If threshold provided, also:
            precision, recall, f1, accuracy,
            true_positive_rate, false_positive_rate.

    Raises:
        ValueError: If scores and labels have different lengths, or if
            labels contain only one class (AUROC is undefined).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if len(scores) != len(labels):
        raise ValueError(
            f"Length mismatch: scores ({len(scores)}) != labels ({len(labels)})"
        )

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both classes in labels: got {n_pos} positives, {n_neg} negatives. "
            "AUROC is undefined with a single class."
        )

    # --- AUROC ---
    auroc = float(roc_auc_score(labels, scores))

    # --- AUPRC ---
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    auprc = float(auc(recall_curve, precision_curve))

    # --- FPR at 95% TPR ---
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    # Find the FPR where TPR first reaches 0.95
    idx_95 = np.searchsorted(tpr_arr, 0.95, side="left")
    idx_95 = min(idx_95, len(fpr_arr) - 1)
    fpr_at_95_tpr = float(fpr_arr[idx_95])

    result: dict[str, float] = {
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95_tpr": fpr_at_95_tpr,
    }

    # --- Threshold-dependent metrics ---
    if threshold is not None:
        predicted = (scores > threshold).astype(np.int64)

        tp = int(np.sum((predicted == 1) & (labels == 1)))
        fp = int(np.sum((predicted == 1) & (labels == 0)))
        tn = int(np.sum((predicted == 0) & (labels == 0)))
        fn = int(np.sum((predicted == 0) & (labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(labels)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        result.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
        })

    return result


def compute_threshold_curves(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> dict[str, np.ndarray]:
    """Compute precision, recall, and FPR across a range of thresholds.

    Useful for visualization (threshold selection plots, ROC curves) and
    for understanding the precision-recall trade-off.

    Args:
        scores: Anomaly scores of shape (n_samples,). Higher = more anomalous.
        labels: Ground truth of shape (n_samples,). 0 = normal, 1 = anomaly.
        n_thresholds: Number of evenly-spaced thresholds to evaluate.

    Returns:
        Dict with arrays:
            thresholds: (n_thresholds,) threshold values.
            precision: (n_thresholds,) precision at each threshold.
            recall: (n_thresholds,) recall (TPR) at each threshold.
            fpr: (n_thresholds,) false positive rate at each threshold.
            f1: (n_thresholds,) F1 score at each threshold.
            roc_fpr: FPR values from sklearn roc_curve (variable length).
            roc_tpr: TPR values from sklearn roc_curve (variable length).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    # Evenly spaced thresholds across the score range
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

    precisions = np.zeros(n_thresholds)
    recalls = np.zeros(n_thresholds)
    fprs = np.zeros(n_thresholds)
    f1s = np.zeros(n_thresholds)

    for i, t in enumerate(thresholds):
        predicted = (scores > t).astype(np.int64)

        tp = int(np.sum((predicted == 1) & (labels == 1)))
        fp = int(np.sum((predicted == 1) & (labels == 0)))
        fn = int(np.sum((predicted == 0) & (labels == 1)))
        tn = int(np.sum((predicted == 0) & (labels == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions[i] = prec
        recalls[i] = rec
        fprs[i] = fpr
        f1s[i] = f1

    # sklearn ROC curve (variable-length, more precise)
    roc_fpr_arr, roc_tpr_arr, _ = roc_curve(labels, scores)

    return {
        "thresholds": thresholds,
        "precision": precisions,
        "recall": recalls,
        "fpr": fprs,
        "f1": f1s,
        "roc_fpr": roc_fpr_arr,
        "roc_tpr": roc_tpr_arr,
    }
