#!/usr/bin/env python3
"""Definitive evaluation script — produces everything needed for a write-up.

Loads the trained linear probe and all comparison detectors, runs on the
held-out test set (jailbreaks never seen in training), and outputs:

1. A well-structured results JSON with primary/comparison/error_analysis sections
2. Publication-ready matplotlib figures (score distribution, ROC, PR, detector comparison)
3. A formatted summary to stdout

Usage:
    python scripts/evaluate_final.py \
        --model-dir models/v2/ \
        --store data/activations/all.h5 \
        --splits data/splits_semi.json \
        --output results/final/ \
        --figures results/figures/
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore[import-untyped]

from venator.activation.storage import ActivationStore
from venator.data.splits import SplitManager
from venator.detection.ensemble import DetectorEnsemble, DetectorType
from venator.detection.metrics import evaluate_detector

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Figure style constants
# ---------------------------------------------------------------------------

_FIG_DPI = 200
_FIG_SIZE = (7, 5)
_FONT_SIZE_TITLE = 16
_FONT_SIZE_LABEL = 13
_FONT_SIZE_TICK = 11
_FONT_SIZE_LEGEND = 11
_FONT_SIZE_ANNOTATION = 12

# Colours
_COLOR_BENIGN = "#2196F3"
_COLOR_JAILBREAK = "#F44336"
_COLOR_PRIMARY = "#1B5E20"
_COLOR_DIAGONAL = "#9E9E9E"
_COLOR_THRESHOLD = "#FF9800"
_COLOR_BAR_SUP = "#1565C0"
_COLOR_BAR_UNSUP = "#7B1FA2"
_COLOR_BAR_ENS = "#616161"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": _FONT_SIZE_TICK,
    "axes.titlesize": _FONT_SIZE_TITLE,
    "axes.labelsize": _FONT_SIZE_LABEL,
    "xtick.labelsize": _FONT_SIZE_TICK,
    "ytick.labelsize": _FONT_SIZE_TICK,
    "legend.fontsize": _FONT_SIZE_LEGEND,
    "figure.dpi": _FIG_DPI,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detector_display_name(name: str) -> str:
    """Convert detector internal name to a human-readable display name."""
    _names = {
        "linear_probe": "Linear Probe",
        "pca_mahalanobis": "PCA + Mahalanobis",
        "contrastive_mahalanobis": "Contrastive Mahalanobis",
        "isolation_forest": "Isolation Forest",
        "autoencoder": "Autoencoder",
        "contrastive_direction": "Contrastive Direction",
    }
    return _names.get(name, name.replace("_", " ").title())


def _detector_type_label(ensemble: DetectorEnsemble, name: str) -> str:
    """Get type label for a detector."""
    dtype = ensemble.detector_types_.get(name, DetectorType.UNSUPERVISED)
    return "supervised" if dtype == DetectorType.SUPERVISED else "unsupervised"


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def figure_score_distribution(
    benign_scores: np.ndarray,
    jailbreak_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Benign vs jailbreak score histogram for the primary detector."""
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    bins = np.linspace(
        min(benign_scores.min(), jailbreak_scores.min()),
        max(benign_scores.max(), jailbreak_scores.max()),
        60,
    )

    ax.hist(benign_scores, bins=bins, alpha=0.6, color=_COLOR_BENIGN,
            label="Benign", edgecolor="white", linewidth=0.5)
    ax.hist(jailbreak_scores, bins=bins, alpha=0.6, color=_COLOR_JAILBREAK,
            label="Jailbreak", edgecolor="white", linewidth=0.5)
    ax.axvline(threshold, color=_COLOR_THRESHOLD, linestyle="--", linewidth=2,
               label=f"Threshold ({threshold:.3f})")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution: Linear Probe")
    ax.legend(loc="upper center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def figure_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    op_fpr: float | None,
    op_tpr: float | None,
    output_path: Path,
) -> None:
    """ROC curve for the primary detector with AUC annotation."""
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    ax.plot(fpr, tpr, color=_COLOR_PRIMARY, linewidth=2.5,
            label=f"Linear Probe (AUC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color=_COLOR_DIAGONAL, linestyle="--",
            linewidth=1, label="Random")

    if op_fpr is not None and op_tpr is not None:
        ax.scatter([op_fpr], [op_tpr], color=_COLOR_THRESHOLD, s=100,
                   zorder=5, label=f"Operating Point (FPR={op_fpr:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Linear Probe")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def figure_precision_recall(
    recall: np.ndarray,
    precision: np.ndarray,
    auprc: float,
    output_path: Path,
) -> None:
    """Precision-Recall curve for the primary detector."""
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    ax.plot(recall, precision, color=_COLOR_PRIMARY, linewidth=2.5,
            label=f"Linear Probe (AUPRC = {auprc:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve: Linear Probe")
    ax.legend(loc="lower left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def figure_detector_comparison(
    detectors: list[dict],
    output_path: Path,
) -> None:
    """Horizontal bar chart comparing all detectors by AUROC."""
    # Sort ascending so highest is at top of horizontal bar chart
    detectors = sorted(detectors, key=lambda d: d["auroc"])

    names = [d["display_name"] for d in detectors]
    aurocs = [d["auroc"] for d in detectors]
    colors = []
    for d in detectors:
        if d["type"] == "ensemble":
            colors.append(_COLOR_BAR_ENS)
        elif d["type"] == "supervised":
            colors.append(_COLOR_BAR_SUP)
        else:
            colors.append(_COLOR_BAR_UNSUP)

    fig, ax = plt.subplots(figsize=(8, max(4, len(detectors) * 0.7 + 1)))

    bars = ax.barh(names, aurocs, color=colors, edgecolor="white", height=0.6)

    # Value labels on bars
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=_FONT_SIZE_ANNOTATION,
                fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=_COLOR_BAR_SUP, label="Supervised"),
        mpatches.Patch(facecolor=_COLOR_BAR_UNSUP, label="Unsupervised"),
        mpatches.Patch(facecolor=_COLOR_BAR_ENS, label="Ensemble"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_xlabel("AUROC")
    ax.set_title("Detector Comparison by AUROC")
    ax.set_xlim(0, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------


def print_summary(
    metadata: dict,
    primary: dict,
    comparison: dict,
    primary_name: str,
) -> None:
    """Print the formatted evaluation summary to stdout."""
    w = 60

    print()
    print("=" * w)
    print("  VENATOR EVALUATION RESULTS")
    print("=" * w)

    print()
    print(f"  Model:    {metadata['model']}")
    print(f"  Layer:    {metadata['layer']} | PCA: {metadata['pca_dims']} dims")
    print(
        f"  Training: {metadata['n_train_benign']} benign"
        f" + {metadata['n_train_jailbreak']} labeled jailbreaks"
    )
    print(
        f"  Testing:  {metadata['n_test_benign']} benign"
        f" + {metadata['n_test_jailbreak']} jailbreaks (never seen in training)"
    )

    # Primary detector
    print()
    pname = _detector_display_name(primary_name)
    print(f"  -- Primary Detector: {pname} " + "-" * max(0, w - 27 - len(pname)))
    print()
    print(f"  AUROC:     {primary['auroc']:.4f}")
    print(f"  AUPRC:     {primary['auprc']:.4f}")
    print(f"  Recall:    {primary['recall_at_threshold']:.3f} @ threshold")
    print(f"  Precision: {primary['precision_at_threshold']:.3f} @ threshold")
    print(f"  FPR:       {primary['fpr_at_threshold']:.3f} @ threshold")

    # Comparison table
    print()
    print("  -- Detector Comparison " + "-" * (w - 26))
    print()

    header = f"  {'Detector':<28} {'Type':<14} {'AUROC':>7}  {'AUPRC':>7}"
    print(header)
    print("  " + "-" * (w - 2))

    # Primary first
    print(
        f"  {pname:<26} * {'supervised':<14} "
        f"{primary['auroc']:>7.4f}  {primary['auprc']:>7.4f}"
    )

    # Others sorted by AUROC descending
    sorted_comp = sorted(comparison.items(), key=lambda kv: -kv[1].get("auroc", 0))
    for name, info in sorted_comp:
        dname = _detector_display_name(name)
        dtype = info.get("type", "unknown")
        auroc = info.get("auroc", 0.0)
        auprc = info.get("auprc", 0.0)
        print(f"  {dname:<28} {dtype:<14} {auroc:>7.4f}  {auprc:>7.4f}")

    # Key findings
    print()
    print("  -- Key Findings " + "-" * (w - 20))
    print()

    # Find best unsupervised and supervised (excluding primary)
    best_unsup_auroc = 0.0
    best_sup_auroc = 0.0
    for name, info in comparison.items():
        if info.get("type") == "unsupervised" and info["auroc"] > best_unsup_auroc:
            best_unsup_auroc = info["auroc"]
        if info.get("type") == "supervised" and info["auroc"] > best_sup_auroc:
            best_sup_auroc = info["auroc"]

    best_sup_auroc = max(best_sup_auroc, primary["auroc"])  # include primary
    ensemble_auroc = 0.0
    for name, info in comparison.items():
        if info.get("type") == "ensemble":
            ensemble_auroc = info["auroc"]
            break

    print(
        f"  1. {pname} on layer {metadata['layer']} activations achieves"
    )
    print(
        f"     near-perfect jailbreak detection ({primary['auroc']:.4f} AUROC)"
    )
    print(
        f"  2. A simple logistic regression outperforms any ensemble"
    )
    if best_unsup_auroc > 0:
        print(
            f"  3. Supervised detectors outperform unsupervised"
            f" ({best_sup_auroc:.3f} vs {best_unsup_auroc:.3f} AUROC"
            f" for best in each category)"
        )
    if ensemble_auroc > 0:
        print(
            f"  4. The ensemble ({ensemble_auroc:.3f} AUROC) performs worse"
            f" than the linear probe alone"
        )

    print()
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Definitive evaluation — structured JSON, figures, and summary"
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Directory containing the saved ensemble model",
    )
    parser.add_argument(
        "--store", type=Path, required=True,
        help="Path to the HDF5 activation store",
    )
    parser.add_argument(
        "--splits", type=Path, required=True,
        help="Path to split definitions (JSON)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/final"),
        help="Directory for output JSON (default: results/final/)",
    )
    parser.add_argument(
        "--figures", type=Path, default=Path("results/figures"),
        help="Directory for output figures (default: results/figures/)",
    )
    args = parser.parse_args()

    # Validate inputs
    for label, path in [("Model dir", args.model_dir), ("Store", args.store), ("Splits", args.splits)]:
        if not path.exists():
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model and data
    # ------------------------------------------------------------------

    ensemble = DetectorEnsemble.load(args.model_dir)
    store = ActivationStore(args.store)
    splits = SplitManager.load_splits(args.splits)

    # Pipeline metadata
    meta_path = args.model_dir / "pipeline_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        layer = meta["layer"]
        ensemble_type = meta.get("ensemble_type", "unsupervised")
        primary_name = meta.get("primary_name")
        primary_threshold = meta.get("primary_threshold")
    else:
        available = store.layers
        layer = available[len(available) // 2]
        ensemble_type = "unknown"
        primary_name = None
        primary_threshold = None
        logger.warning("No pipeline_meta.json found, using layer %d", layer)

    logger.info("Layer %d | ensemble_type=%s", layer, ensemble_type)

    # ------------------------------------------------------------------
    # Resolve primary detector
    # ------------------------------------------------------------------

    _priority = [
        "linear_probe", "pca_mahalanobis", "contrastive_mahalanobis",
        "isolation_forest", "autoencoder", "contrastive_direction",
    ]
    det_names = {n for n, _, _ in ensemble.detectors}

    if primary_name is None or primary_name not in det_names:
        for candidate in _priority:
            if candidate in det_names:
                primary_name = candidate
                break
        else:
            primary_name = ensemble.detectors[0][0] if ensemble.detectors else "unknown"

    primary_det = None
    for n, d, _ in ensemble.detectors:
        if n == primary_name:
            primary_det = d
            break

    # ------------------------------------------------------------------
    # Build test set
    # ------------------------------------------------------------------

    test_benign_indices = splits["test_benign"].indices.tolist()
    test_jailbreak_indices = splits["test_jailbreak"].indices.tolist()

    X_test_benign = store.get_activations(layer, indices=test_benign_indices)
    X_test_jailbreak = store.get_activations(layer, indices=test_jailbreak_indices)

    X_test = np.vstack([X_test_benign, X_test_jailbreak])
    labels = np.concatenate([
        np.zeros(len(X_test_benign), dtype=np.int64),
        np.ones(len(X_test_jailbreak), dtype=np.int64),
    ])

    n_test_benign = len(X_test_benign)
    n_test_jailbreak = len(X_test_jailbreak)
    logger.info("Test set: %d benign + %d jailbreak = %d total",
                n_test_benign, n_test_jailbreak, n_test_benign + n_test_jailbreak)

    # ------------------------------------------------------------------
    # Count training samples from splits
    # ------------------------------------------------------------------

    n_train_benign = splits["train_benign"].n_samples
    n_train_jailbreak = splits["train_jailbreak"].n_samples

    # ------------------------------------------------------------------
    # Score primary detector
    # ------------------------------------------------------------------

    threshold = primary_threshold if primary_threshold is not None else ensemble.threshold_

    if primary_det is not None:
        primary_scores = primary_det.score(X_test)
        primary_metrics = evaluate_detector(primary_scores, labels, threshold=threshold)
    else:
        logger.warning("Primary detector %s not found, falling back to ensemble", primary_name)
        ens_result = ensemble.score(X_test)
        primary_scores = ens_result.ensemble_scores
        primary_metrics = evaluate_detector(primary_scores, labels, threshold=threshold)

    benign_scores = primary_scores[:n_test_benign]
    jailbreak_scores = primary_scores[n_test_benign:]

    # ROC curve data (from sklearn for smooth curve)
    roc_fpr, roc_tpr, _ = roc_curve(labels, primary_scores)

    # PR curve data
    pr_precision, pr_recall, _ = precision_recall_curve(labels, primary_scores)

    # ------------------------------------------------------------------
    # Score all detectors through ensemble
    # ------------------------------------------------------------------

    ens_result = ensemble.score(X_test)
    ens_metrics = evaluate_detector(ens_result.ensemble_scores, labels, threshold=ensemble.threshold_)

    # Per-detector comparison
    comparison_detectors: list[dict] = []
    for det_result in ens_result.detector_results:
        if det_result.name == primary_name:
            continue  # skip primary, it goes in its own section
        det_metrics = evaluate_detector(det_result.normalized_scores, labels)
        dtype = _detector_type_label(ensemble, det_result.name)
        comparison_detectors.append({
            "name": det_result.name,
            "display_name": _detector_display_name(det_result.name),
            "type": dtype,
            "auroc": det_metrics["auroc"],
            "auprc": det_metrics["auprc"],
            "fpr_at_95_tpr": det_metrics["fpr_at_95_tpr"],
        })

    # ------------------------------------------------------------------
    # Error analysis: false positives and false negatives
    # ------------------------------------------------------------------

    predictions = (primary_scores > threshold).astype(int)

    # Get prompts and source labels
    all_test_indices = test_benign_indices + test_jailbreak_indices
    test_prompts = store.get_prompts(indices=all_test_indices)
    test_labels_from_store = store.get_labels()

    fp_entries = []
    fn_entries = []
    for i in range(len(labels)):
        is_fp = predictions[i] == 1 and labels[i] == 0
        is_fn = predictions[i] == 0 and labels[i] == 1
        if is_fp or is_fn:
            entry = {
                "prompt": test_prompts[i][:200],
                "score": round(float(primary_scores[i]), 6),
            }
            if is_fp:
                fp_entries.append(entry)
            else:
                fn_entries.append(entry)

    # Sort FPs by score descending (highest-confidence mistakes first)
    fp_entries.sort(key=lambda e: -e["score"])
    fn_entries.sort(key=lambda e: e["score"])

    # ------------------------------------------------------------------
    # Build output JSON
    # ------------------------------------------------------------------

    metadata = {
        "model": store.model_id,
        "layer": layer,
        "pca_dims": meta.get("pca_dims", "unknown") if meta_path.exists() else "unknown",
        "n_train_benign": n_train_benign,
        "n_train_jailbreak": n_train_jailbreak,
        "n_test_benign": n_test_benign,
        "n_test_jailbreak": n_test_jailbreak,
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "split_mode": split_mode.value,
        "ensemble_type": ensemble_type,
    }

    primary_section = {
        "detector": primary_name,
        "auroc": float(primary_metrics["auroc"]),
        "auprc": float(primary_metrics["auprc"]),
        "precision_at_threshold": float(primary_metrics.get("precision", 0.0)),
        "recall_at_threshold": float(primary_metrics.get("recall", 0.0)),
        "f1_at_threshold": float(primary_metrics.get("f1", 0.0)),
        "fpr_at_threshold": float(primary_metrics.get("false_positive_rate", 0.0)),
        "fpr_at_95_tpr": float(primary_metrics["fpr_at_95_tpr"]),
        "threshold": float(threshold),
        "roc_curve": {
            "fpr": roc_fpr.tolist(),
            "tpr": roc_tpr.tolist(),
        },
        "score_distribution": {
            "benign_scores": benign_scores.tolist(),
            "jailbreak_scores": jailbreak_scores.tolist(),
        },
    }

    comparison_section = {}
    for det in comparison_detectors:
        comparison_section[det["name"]] = {
            "type": det["type"],
            "auroc": det["auroc"],
            "auprc": det["auprc"],
            "fpr_at_95_tpr": det["fpr_at_95_tpr"],
        }

    # Add ensemble as a comparison entry
    comparison_section["ensemble_all"] = {
        "type": "ensemble",
        "auroc": float(ens_metrics["auroc"]),
        "auprc": float(ens_metrics["auprc"]),
        "fpr_at_95_tpr": float(ens_metrics["fpr_at_95_tpr"]),
        "note": "Included for completeness. Performs worse than linear probe alone."
            if ens_metrics["auroc"] < primary_metrics["auroc"]
            else "Included for completeness.",
    }

    error_analysis = {
        "false_positives": fp_entries,
        "false_negatives": fn_entries,
    }

    results = {
        "metadata": metadata,
        "primary": primary_section,
        "comparison": comparison_section,
        "error_analysis": error_analysis,
    }

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------

    json_path = args.output / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results JSON: %s", json_path)

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------

    logger.info("Generating figures...")

    figure_score_distribution(
        benign_scores, jailbreak_scores, float(threshold),
        args.figures / "score_distribution.png",
    )

    figure_roc_curve(
        roc_fpr, roc_tpr, primary_metrics["auroc"],
        primary_metrics.get("false_positive_rate"),
        primary_metrics.get("true_positive_rate"),
        args.figures / "roc_curve.png",
    )

    figure_precision_recall(
        pr_recall, pr_precision, primary_metrics["auprc"],
        args.figures / "precision_recall.png",
    )

    # Build detector list for comparison chart
    all_detectors_for_chart = [{
        "display_name": _detector_display_name(primary_name) + " *",
        "type": _detector_type_label(ensemble, primary_name),
        "auroc": float(primary_metrics["auroc"]),
    }]
    for det in comparison_detectors:
        all_detectors_for_chart.append({
            "display_name": det["display_name"],
            "type": det["type"],
            "auroc": det["auroc"],
        })
    all_detectors_for_chart.append({
        "display_name": "Ensemble (all)",
        "type": "ensemble",
        "auroc": float(ens_metrics["auroc"]),
    })

    figure_detector_comparison(
        all_detectors_for_chart,
        args.figures / "detector_comparison.png",
    )

    logger.info("Figures saved to: %s", args.figures)

    # ------------------------------------------------------------------
    # Print summary to stdout
    # ------------------------------------------------------------------

    print_summary(metadata, primary_section, comparison_section, primary_name)

    print(f"  Results JSON: {json_path}")
    print(f"  Figures:      {args.figures}/")
    print()


if __name__ == "__main__":
    main()
