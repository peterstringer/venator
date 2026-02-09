#!/usr/bin/env python3
"""Full evaluation of trained detectors on test data (benign + jailbreak).

Loads a trained ensemble, scores the test set, computes metrics (AUROC, AUPRC,
precision, recall, F1), and reports a per-detector comparison table showing
each detector's individual contribution.

Usage:
    python scripts/evaluate.py \
        --model-dir models/detector_v1/ \
        --store data/activations/all.h5 \
        --splits data/splits.json

    python scripts/evaluate.py \
        --model-dir models/detector_v1/ \
        --store data/activations/all.h5 \
        --splits data/splits.json \
        --output results/eval_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from venator.activation.storage import ActivationStore
from venator.data.splits import SplitManager
from venator.detection.ensemble import DetectorEnsemble, DetectorType
from venator.detection.metrics import evaluate_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _detector_type_label(ensemble: DetectorEnsemble, name: str) -> str:
    """Get a human-readable type label for a detector."""
    dtype = ensemble.detector_types_.get(name, DetectorType.UNSUPERVISED)
    return "sup" if dtype == DetectorType.SUPERVISED else "unsup"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained detectors on test data (benign + jailbreak)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing the saved ensemble model",
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Path to the HDF5 activation store",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="Path to split definitions (JSON)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional: save full results to a JSON file",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        logger.error("Model directory not found: %s", args.model_dir)
        sys.exit(1)
    if not args.store.exists():
        logger.error("Store not found: %s", args.store)
        sys.exit(1)
    if not args.splits.exists():
        logger.error("Splits file not found: %s", args.splits)
        sys.exit(1)

    # Load model and data
    ensemble = DetectorEnsemble.load(args.model_dir)
    store = ActivationStore(args.store)
    splits = SplitManager.load_splits(args.splits)

    # Read layer and ensemble type from pipeline metadata
    meta_path = args.model_dir / "pipeline_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        layer = meta["layer"]
        ensemble_type = meta.get("ensemble_type", "unsupervised")
    else:
        # Fallback to middle of available layers
        available = store.layers
        layer = available[len(available) // 2]
        ensemble_type = "unknown"
        logger.warning("No pipeline_meta.json found, using layer %d", layer)

    logger.info("Evaluating on layer %d (ensemble_type=%s)", layer, ensemble_type)

    # Get test data
    X_test_benign = store.get_activations(
        layer, indices=splits["test_benign"].indices.tolist()
    )
    X_test_jailbreak = store.get_activations(
        layer, indices=splits["test_jailbreak"].indices.tolist()
    )

    X_test = np.vstack([X_test_benign, X_test_jailbreak])
    labels = np.concatenate([
        np.zeros(len(X_test_benign), dtype=np.int64),
        np.ones(len(X_test_jailbreak), dtype=np.int64),
    ])

    logger.info(
        "Test set: %d benign + %d jailbreak = %d total",
        len(X_test_benign), len(X_test_jailbreak), len(X_test),
    )

    # --- Determine primary detector ---
    # Priority: linear_probe > pca_mahalanobis > first available
    primary_name = meta.get("primary_name") if meta_path.exists() else None
    primary_threshold = meta.get("primary_threshold") if meta_path.exists() else None

    _priority = ["linear_probe", "pca_mahalanobis", "contrastive_mahalanobis",
                 "isolation_forest", "autoencoder", "contrastive_direction"]
    det_names_in_ensemble = {n for n, _, _ in ensemble.detectors}
    if primary_name is None or primary_name not in det_names_in_ensemble:
        for candidate in _priority:
            if candidate in det_names_in_ensemble:
                primary_name = candidate
                break
        else:
            primary_name = ensemble.detectors[0][0] if ensemble.detectors else "unknown"

    # Score through ensemble (runs all detectors)
    result = ensemble.score(X_test)

    # --- Primary detector metrics ---
    primary_det = None
    for name, det, _ in ensemble.detectors:
        if name == primary_name:
            primary_det = det
            break

    if primary_det is not None:
        primary_scores = primary_det.score(X_test)
        if primary_threshold is not None:
            primary_metrics = evaluate_detector(primary_scores, labels, threshold=primary_threshold)
        else:
            primary_metrics = evaluate_detector(primary_scores, labels, threshold=ensemble.threshold_)
            primary_threshold = ensemble.threshold_
    else:
        primary_metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=ensemble.threshold_
        )

    # Ensemble-level metrics (for retired comparison)
    ensemble_metrics = evaluate_detector(
        result.ensemble_scores, labels, threshold=ensemble.threshold_
    )

    # Per-detector metrics
    per_detector: list[dict[str, object]] = []
    for det_result in result.detector_results:
        det_metrics = evaluate_detector(
            det_result.normalized_scores, labels, threshold=ensemble.threshold_
        )
        type_label = _detector_type_label(ensemble, det_result.name)
        per_detector.append({
            "name": det_result.name,
            "type": type_label,
            "weight": det_result.weight,
            "auroc": det_metrics["auroc"],
            "auprc": det_metrics["auprc"],
            "f1": det_metrics.get("f1", 0.0),
            "fpr_at_95_tpr": det_metrics["fpr_at_95_tpr"],
        })

    # --- Print structured comparison ---
    width = 75
    print(f"\nDETECTOR COMPARISON (test set, {len(X_test_jailbreak)} jailbreaks never seen in training)")
    print("=" * width)

    # PRIMARY section
    print(f"\n  PRIMARY:")
    p_label = f"  {primary_name:<28}"
    print(
        f"  {p_label} AUROC: {primary_metrics['auroc']:.3f}   "
        f"AUPRC: {primary_metrics['auprc']:.3f}"
    )

    # BASELINES section
    print(f"\n  BASELINES:")
    baselines = [d for d in per_detector if d["name"] != primary_name]
    baselines.sort(key=lambda d: d["auroc"], reverse=True)
    for det in baselines:
        label = f"{det['name']}"
        print(
            f"    {label:<26} AUROC: {det['auroc']:>5.3f}   "
            f"AUPRC: {det['auprc']:>5.3f}  ({det['type']})"
        )

    # RETIRED section
    ens_note = "worse than primary" if ensemble_metrics["auroc"] < primary_metrics["auroc"] else "comparable"
    print(f"\n  RETIRED:")
    print(
        f"    {'Ensemble (all detectors)':<26} AUROC: {ensemble_metrics['auroc']:>5.3f}   "
        f"<- {ens_note}"
    )

    print(f"\n{'=' * width}")

    # Additional primary metrics
    print(f"\n  {'Metric':<30} | {'Value':>10}")
    print("  " + "-" * 43)
    extra_keys = ["precision", "recall", "accuracy", "true_positive_rate", "false_positive_rate"]
    for key in extra_keys:
        if key in primary_metrics:
            print(f"  {key:<30} | {primary_metrics[key]:>10.4f}")
    print()

    # Save JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        json_result = {
            "layer": layer,
            "ensemble_type": ensemble_type,
            "n_test_benign": len(X_test_benign),
            "n_test_jailbreak": len(X_test_jailbreak),
            # Primary detector at top level
            "primary": {
                "detector": primary_name,
                "threshold": float(primary_threshold),
                **{k: float(v) for k, v in primary_metrics.items()},
            },
            "baselines": [
                {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in det.items()}
                for det in baselines
            ],
            "retired": [{
                "detector": "ensemble",
                "threshold": float(ensemble.threshold_),
                **{k: float(v) for k, v in ensemble_metrics.items()},
                "note": ens_note,
            }],
            # Top-level convenience keys
            "auroc": float(primary_metrics["auroc"]),
            "auprc": float(primary_metrics["auprc"]),
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2)
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
