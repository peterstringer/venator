#!/usr/bin/env python3
"""Full evaluation of trained detectors on test data (benign + jailbreak).

Loads a trained ensemble, scores the test set, computes metrics (AUROC, AUPRC,
precision, recall, F1), and reports per-detector breakdowns.

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
from venator.detection.ensemble import DetectorEnsemble
from venator.detection.metrics import evaluate_detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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

    # Read layer from pipeline metadata
    meta_path = args.model_dir / "pipeline_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        layer = meta["layer"]
    else:
        # Fallback to middle of available layers
        available = store.layers
        layer = available[len(available) // 2]
        logger.warning("No pipeline_meta.json found, using layer %d", layer)

    logger.info("Evaluating on layer %d", layer)

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

    # Score and evaluate
    result = ensemble.score(X_test)
    metrics = evaluate_detector(
        result.ensemble_scores, labels, threshold=ensemble.threshold_
    )

    # Per-detector AUROC
    for det_result in result.detector_results:
        det_metrics = evaluate_detector(det_result.normalized_scores, labels)
        metrics[f"auroc_{det_result.name}"] = det_metrics["auroc"]

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Layer:               {layer}")
    print(f"  Test benign:         {len(X_test_benign)}")
    print(f"  Test jailbreak:      {len(X_test_jailbreak)}")
    print(f"  Threshold:           {ensemble.threshold_:.4f}")
    print()

    # Primary metrics
    print(f"{'Metric':<30} | {'Value':>10}")
    print("-" * 43)
    primary_keys = ["auroc", "auprc", "fpr_at_95_tpr", "precision", "recall", "f1", "accuracy"]
    for key in primary_keys:
        if key in metrics:
            print(f"  {key:<28} | {metrics[key]:>10.4f}")

    # Per-detector AUROC
    print()
    print("Per-Detector AUROC:")
    print("-" * 43)
    for key in sorted(metrics.keys()):
        if key.startswith("auroc_"):
            det_name = key.replace("auroc_", "")
            print(f"  {det_name:<28} | {metrics[key]:>10.4f}")
    print("=" * 60)

    # Save JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Convert any numpy values to Python types for JSON serialization
        json_metrics = {k: float(v) for k, v in metrics.items()}
        json_metrics["layer"] = layer
        json_metrics["n_test_benign"] = len(X_test_benign)
        json_metrics["n_test_jailbreak"] = len(X_test_jailbreak)
        json_metrics["threshold"] = float(ensemble.threshold_)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
