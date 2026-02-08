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

    # Score through ensemble
    result = ensemble.score(X_test)

    # Ensemble-level metrics
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

    # --- Print comparison table ---
    print(f"\n{'=' * 75}")
    print(f"Evaluation Results — {ensemble_type} ensemble, layer {layer}")
    print(f"Test: {len(X_test_benign)} benign + {len(X_test_jailbreak)} jailbreak")
    print(f"Threshold: {ensemble.threshold_:.4f}")
    print(f"{'=' * 75}")

    # Detector comparison table
    header = f"  {'Detector':<30} | {'AUROC':>7} | {'AUPRC':>7} | {'F1':>7} | {'FPR@95':>7}"
    separator = "  " + "-" * 71
    print(header)
    print(separator)

    for det in per_detector:
        label = f"{det['name']} ({det['type']})"
        print(
            f"  {label:<30} | {det['auroc']:>7.4f} | {det['auprc']:>7.4f} | "
            f"{det['f1']:>7.4f} | {det['fpr_at_95_tpr']:>7.4f}"
        )

    print(separator)
    print(
        f"  {'Ensemble':<30} | {ensemble_metrics['auroc']:>7.4f} | "
        f"{ensemble_metrics['auprc']:>7.4f} | "
        f"{ensemble_metrics.get('f1', 0.0):>7.4f} | "
        f"{ensemble_metrics['fpr_at_95_tpr']:>7.4f}"
    )
    print(f"{'=' * 75}")

    # Additional ensemble metrics
    print(f"\n  {'Metric':<30} | {'Value':>10}")
    print("  " + "-" * 43)
    extra_keys = ["precision", "recall", "accuracy", "true_positive_rate", "false_positive_rate"]
    for key in extra_keys:
        if key in ensemble_metrics:
            print(f"  {key:<30} | {ensemble_metrics[key]:>10.4f}")
    print()

    # Save JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        json_result = {
            "layer": layer,
            "ensemble_type": ensemble_type,
            "n_test_benign": len(X_test_benign),
            "n_test_jailbreak": len(X_test_jailbreak),
            "threshold": float(ensemble.threshold_),
            "ensemble_metrics": {k: float(v) for k, v in ensemble_metrics.items()},
            "per_detector": [
                {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in det.items()}
                for det in per_detector
            ],
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2)
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
