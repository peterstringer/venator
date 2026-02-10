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
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Per-detector evaluation metrics and curve data.

    Stores everything needed to render the Results page for one detector:
    metrics, optimal threshold, per-class scores, and curve data.

    Attributes:
        detector_name: Internal name (e.g., "linear_probe").
        display_name: Human-readable name (e.g., "Linear Probe").
        detector_type: One of "supervised", "unsupervised", "ensemble".
        auroc: Area under the ROC curve (primary metric).
        auprc: Area under the precision-recall curve.
        f1: F1 score at optimal threshold.
        precision: Precision at optimal threshold.
        recall: Recall at optimal threshold.
        fpr: False positive rate at optimal threshold.
        threshold: Optimal threshold via Youden's J statistic.
        scores_benign: Raw anomaly scores for benign test samples.
        scores_jailbreak: Raw anomaly scores for jailbreak test samples.
        roc_fpr: FPR values for ROC curve.
        roc_tpr: TPR values for ROC curve.
        pr_precision: Precision values for PR curve.
        pr_recall: Recall values for PR curve.
    """

    detector_name: str
    display_name: str
    detector_type: str
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    fpr: float
    threshold: float
    scores_benign: np.ndarray
    scores_jailbreak: np.ndarray
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    pr_precision: np.ndarray
    pr_recall: np.ndarray


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


def evaluate_all_detectors(
    detectors: dict[str, object],
    X_test_benign: np.ndarray,
    X_test_jailbreak: np.ndarray,
    detector_types: dict[str, str] | None = None,
    display_names: dict[str, str] | None = None,
) -> list[EvaluationResult]:
    """Evaluate every trained detector on the test set.

    Each detector is scored independently. The ensemble (if present) is
    treated as just another detector — no special handling. Results are
    sorted by AUROC descending.

    Threshold selection uses Youden's J statistic (maximizes TPR - FPR)
    for each detector independently.

    Args:
        detectors: Mapping of detector name to object with a score(X) method
            that returns an ndarray of anomaly scores.
        X_test_benign: Benign test activations, shape (n_benign, n_features).
        X_test_jailbreak: Jailbreak test activations, shape (n_jailbreak, n_features).
        detector_types: Optional mapping of name to type
            ("supervised", "unsupervised", "ensemble").
        display_names: Optional mapping of name to human-readable display name.

    Returns:
        List of EvaluationResult sorted by AUROC descending.
    """
    if detector_types is None:
        detector_types = {}
    if display_names is None:
        display_names = {}

    X_test = np.vstack([X_test_benign, X_test_jailbreak])
    labels = np.concatenate([
        np.zeros(len(X_test_benign), dtype=np.int64),
        np.ones(len(X_test_jailbreak), dtype=np.int64),
    ])

    results = []
    for name, detector in detectors.items():
        all_scores = detector.score(X_test)
        scores_benign = all_scores[: len(X_test_benign)]
        scores_jailbreak = all_scores[len(X_test_benign) :]

        # Threshold-independent metrics
        auroc_val = float(roc_auc_score(labels, all_scores))
        pr_prec, pr_rec, _ = precision_recall_curve(labels, all_scores)
        auprc_val = float(auc(pr_rec, pr_prec))

        # ROC curve + optimal threshold via Youden's J
        fpr_arr, tpr_arr, thresholds_arr = roc_curve(labels, all_scores)
        j_stat = tpr_arr - fpr_arr
        best_idx = int(np.argmax(j_stat))
        threshold = float(np.nextafter(thresholds_arr[best_idx], -np.inf))

        # Threshold-dependent metrics
        thresh_metrics = evaluate_detector(
            all_scores, labels, threshold=threshold
        )

        results.append(
            EvaluationResult(
                detector_name=name,
                display_name=display_names.get(name, name),
                detector_type=detector_types.get(name, "unknown"),
                auroc=auroc_val,
                auprc=auprc_val,
                f1=thresh_metrics.get("f1", 0.0),
                precision=thresh_metrics.get("precision", 0.0),
                recall=thresh_metrics.get("recall", 0.0),
                fpr=thresh_metrics.get("false_positive_rate", 0.0),
                threshold=threshold,
                scores_benign=scores_benign,
                scores_jailbreak=scores_jailbreak,
                roc_fpr=fpr_arr,
                roc_tpr=tpr_arr,
                pr_precision=pr_prec,
                pr_recall=pr_rec,
            )
        )

        logger.info(
            "Evaluated %s: AUROC=%.4f, AUPRC=%.4f, F1=%.4f",
            name,
            auroc_val,
            auprc_val,
            thresh_metrics.get("f1", 0.0),
        )

    return sorted(results, key=lambda r: r.auroc, reverse=True)
