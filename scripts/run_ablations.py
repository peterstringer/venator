#!/usr/bin/env python3
"""Ablation studies: compare layers, PCA dimensions, and detector variants.

Runs systematic comparisons to identify the best configuration for jailbreak
detection. Tests individual detectors vs the full ensemble, different PCA
dimensionalities, and different transformer layers.

Usage:
    python scripts/run_ablations.py \
        --store data/activations/all.h5 \
        --splits data/splits.json \
        --output results/ablations/

    python scripts/run_ablations.py \
        --store data/activations/all.h5 \
        --splits data/splits.json \
        --ablate layers pca_dims detectors \
        --output results/ablations/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from venator.activation.storage import ActivationStore
from venator.config import VenatorConfig
from venator.data.splits import SplitManager
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.ensemble import DetectorEnsemble, create_default_ensemble
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.metrics import evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ablation configurations
ABLATION_LAYERS = [8, 10, 12, 14, 16, 18, 20, 22, 24]
ABLATION_PCA_DIMS = [10, 20, 30, 50, 75, 100]


def _get_test_data(
    store: ActivationStore,
    splits: dict,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get stacked test data and labels for a given layer."""
    X_benign = store.get_activations(
        layer, indices=splits["test_benign"].indices.tolist()
    )
    X_jailbreak = store.get_activations(
        layer, indices=splits["test_jailbreak"].indices.tolist()
    )
    X_test = np.vstack([X_benign, X_jailbreak])
    labels = np.concatenate([
        np.zeros(len(X_benign), dtype=np.int64),
        np.ones(len(X_jailbreak), dtype=np.int64),
    ])
    return X_test, labels


def ablate_layers(
    store: ActivationStore,
    splits: dict,
    config: VenatorConfig,
) -> list[dict]:
    """Test which transformer layers produce the best detection AUROC.

    Uses PCA+Mahalanobis only (fastest detector) to isolate the effect of
    layer choice without confounding from ensemble dynamics.
    """
    available_layers = set(store.layers)
    test_layers = [layer for layer in ABLATION_LAYERS if layer in available_layers]

    if not test_layers:
        logger.warning(
            "None of the ablation layers %s are in the store (available: %s)",
            ABLATION_LAYERS,
            sorted(available_layers),
        )
        return []

    results = []
    for layer in test_layers:
        logger.info("Layer ablation: testing layer %d...", layer)
        t0 = time.perf_counter()

        X_train = store.get_activations(layer, indices=splits["train"].indices.tolist())
        X_val = store.get_activations(layer, indices=splits["val"].indices.tolist())
        X_test, labels = _get_test_data(store, splits, layer)

        ensemble = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        ensemble.add_detector(
            "pca_mahalanobis",
            PCAMahalanobisDetector(n_components=config.pca_dims),
            weight=1.0,
        )
        ensemble.fit(X_train, X_val)

        result = ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=ensemble.threshold_
        )

        elapsed = time.perf_counter() - t0
        results.append({"layer": layer, "time_s": round(elapsed, 2), **metrics})

    return results


def ablate_pca_dims(
    store: ActivationStore,
    splits: dict,
    config: VenatorConfig,
    layer: int,
) -> list[dict]:
    """Test different PCA dimensionalities on a fixed layer."""
    X_train = store.get_activations(layer, indices=splits["train"].indices.tolist())
    X_val = store.get_activations(layer, indices=splits["val"].indices.tolist())
    X_test, labels = _get_test_data(store, splits, layer)

    results = []
    for n_dims in ABLATION_PCA_DIMS:
        # Skip if more components than training samples
        if n_dims >= X_train.shape[0]:
            logger.info("Skipping PCA dims=%d (n_train=%d)", n_dims, X_train.shape[0])
            continue

        logger.info("PCA ablation: testing %d dimensions...", n_dims)
        t0 = time.perf_counter()

        ensemble = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        ensemble.add_detector(
            "pca_mahalanobis",
            PCAMahalanobisDetector(n_components=n_dims),
            weight=1.0,
        )
        ensemble.fit(X_train, X_val)

        result = ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=ensemble.threshold_
        )

        elapsed = time.perf_counter() - t0
        results.append({"pca_dims": n_dims, "time_s": round(elapsed, 2), **metrics})

    return results


def ablate_detectors(
    store: ActivationStore,
    splits: dict,
    config: VenatorConfig,
    layer: int,
) -> list[dict]:
    """Test individual detectors vs the full ensemble."""
    X_train = store.get_activations(layer, indices=splits["train"].indices.tolist())
    X_val = store.get_activations(layer, indices=splits["val"].indices.tolist())
    X_test, labels = _get_test_data(store, splits, layer)

    detector_configs = [
        ("pca_mahalanobis", PCAMahalanobisDetector(n_components=config.pca_dims)),
        ("isolation_forest", IsolationForestDetector(n_components=config.pca_dims)),
        ("autoencoder", AutoencoderDetector(n_components=config.pca_dims)),
    ]

    results = []

    # Individual detectors
    for name, detector in detector_configs:
        logger.info("Detector ablation: testing %s...", name)
        t0 = time.perf_counter()

        ensemble = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        ensemble.add_detector(name, detector, weight=1.0)
        ensemble.fit(X_train, X_val)

        result = ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=ensemble.threshold_
        )

        elapsed = time.perf_counter() - t0
        results.append({"detector": name, "time_s": round(elapsed, 2), **metrics})

    # Full ensemble
    logger.info("Detector ablation: testing full ensemble...")
    t0 = time.perf_counter()

    full_ensemble = create_default_ensemble(
        threshold_percentile=config.anomaly_threshold_percentile,
    )
    full_ensemble.fit(X_train, X_val)

    result = full_ensemble.score(X_test)
    metrics = evaluate_detector(
        result.ensemble_scores, labels, threshold=full_ensemble.threshold_
    )

    elapsed = time.perf_counter() - t0
    results.append({"detector": "ensemble", "time_s": round(elapsed, 2), **metrics})

    return results


def _print_table(title: str, results: list[dict], key_col: str) -> None:
    """Print a formatted comparison table."""
    if not results:
        print(f"\n{title}: No results")
        return

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    print(
        f"  {key_col:<20} | {'AUROC':>8} | {'AUPRC':>8} | {'FPR@95':>8} | {'F1':>8} | {'Time':>6}"
    )
    print("  " + "-" * 66)

    for row in results:
        key_val = str(row.get(key_col, "?"))
        print(
            f"  {key_val:<20} | {row.get('auroc', 0):.4f}   | "
            f"{row.get('auprc', 0):.4f}   | {row.get('fpr_at_95_tpr', 0):.4f}   | "
            f"{row.get('f1', 0):.4f}   | {row.get('time_s', 0):>5.1f}s"
        )

    # Highlight best AUROC
    best = max(results, key=lambda r: r.get("auroc", 0))
    print(f"\n  Best: {key_col}={best.get(key_col)} (AUROC={best.get('auroc', 0):.4f})")
    print("=" * 70)


def main() -> None:
    config = VenatorConfig()
    default_layer = config.extraction_layers[len(config.extraction_layers) // 2]

    parser = argparse.ArgumentParser(
        description="Run ablation studies on detector configurations"
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
        "--ablate",
        nargs="+",
        choices=["layers", "pca_dims", "detectors"],
        default=["layers", "pca_dims", "detectors"],
        help="Which ablations to run (default: all)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=default_layer,
        help=f"Base layer for PCA/detector ablations (default: {default_layer})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for JSON results",
    )
    args = parser.parse_args()

    if not args.store.exists():
        logger.error("Store not found: %s", args.store)
        sys.exit(1)
    if not args.splits.exists():
        logger.error("Splits file not found: %s", args.splits)
        sys.exit(1)

    store = ActivationStore(args.store)
    splits = SplitManager.load_splits(args.splits)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Store: %s", store)
    logger.info("Ablations to run: %s", args.ablate)

    all_results = {}

    # --- Layer ablation ---
    if "layers" in args.ablate:
        logger.info("Running layer ablation...")
        layer_results = ablate_layers(store, splits, config)
        all_results["layers"] = layer_results

        with open(output_dir / "ablation_layers.json", "w", encoding="utf-8") as f:
            json.dump(layer_results, f, indent=2)

        _print_table("Layer Ablation", layer_results, "layer")

    # --- PCA dimension ablation ---
    if "pca_dims" in args.ablate:
        logger.info("Running PCA dimension ablation on layer %d...", args.layer)
        pca_results = ablate_pca_dims(store, splits, config, args.layer)
        all_results["pca_dims"] = pca_results

        with open(output_dir / "ablation_pca_dims.json", "w", encoding="utf-8") as f:
            json.dump(pca_results, f, indent=2)

        _print_table(f"PCA Dimension Ablation (layer={args.layer})", pca_results, "pca_dims")

    # --- Detector ablation ---
    if "detectors" in args.ablate:
        logger.info("Running detector ablation on layer %d...", args.layer)
        detector_results = ablate_detectors(store, splits, config, args.layer)
        all_results["detectors"] = detector_results

        with open(output_dir / "ablation_detectors.json", "w", encoding="utf-8") as f:
            json.dump(detector_results, f, indent=2)

        _print_table(
            f"Detector Ablation (layer={args.layer})",
            detector_results,
            "detector",
        )

    # Save combined results
    with open(output_dir / "ablation_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
