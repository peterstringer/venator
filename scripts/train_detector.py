#!/usr/bin/env python3
"""Train anomaly detectors on activation data.

Supports unsupervised, supervised, and hybrid ensemble configurations.
Unsupervised ensembles train on benign-only data; supervised and hybrid
ensembles additionally use labeled jailbreak data for contrastive detectors
and supervised threshold calibration.

Usage:
    # Unsupervised (original, benign-only)
    python scripts/train_detector.py \
        --store data/activations/all.h5 \
        --splits data/splits.json \
        --output models/detector_v1/

    # Hybrid (supervised + unsupervised detectors)
    python scripts/train_detector.py \
        --store data/activations/all.h5 \
        --splits data/splits_semi.json \
        --ensemble-type hybrid \
        --layer 16 \
        --output models/hybrid_v1/

    # Supervised-only
    python scripts/train_detector.py \
        --store data/activations/all.h5 \
        --splits data/splits_semi.json \
        --ensemble-type supervised \
        --output models/supervised_v1/
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
from venator.detection.ensemble import (
    DetectorEnsemble,
    create_default_ensemble,
    create_hybrid_ensemble,
    create_supervised_ensemble,
    create_unsupervised_ensemble,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_ENSEMBLE_FACTORIES = {
    "unsupervised": create_unsupervised_ensemble,
    "supervised": create_supervised_ensemble,
    "hybrid": create_hybrid_ensemble,
}


def main() -> None:
    config = VenatorConfig()

    parser = argparse.ArgumentParser(
        description="Train anomaly detectors on activation data"
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
        help="Path to split definitions (JSON from create_splits.py)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=config.extraction_layers[len(config.extraction_layers) // 2],
        help="Transformer layer index to use for detection "
        f"(default: {config.extraction_layers[len(config.extraction_layers) // 2]})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--ensemble-type",
        choices=["unsupervised", "supervised", "hybrid"],
        default="unsupervised",
        help="Ensemble configuration: unsupervised (benign-only), supervised "
        "(labeled data), or hybrid (supervised + unsupervised detectors). "
        "Default: unsupervised",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=config.anomaly_threshold_percentile,
        help=f"Percentile for threshold calibration (default: {config.anomaly_threshold_percentile})",
    )
    args = parser.parse_args()

    if not args.store.exists():
        logger.error("Store not found: %s", args.store)
        sys.exit(1)
    if not args.splits.exists():
        logger.error("Splits file not found: %s", args.splits)
        sys.exit(1)

    # Load data
    store = ActivationStore(args.store)
    splits = SplitManager.load_splits(args.splits)
    split_mode = SplitManager.load_mode(args.splits)
    logger.info("Loaded store: %s", store)
    logger.info("Loaded splits (%s): %s", split_mode.value, {
        name: s.n_samples for name, s in splits.items()
    })

    # Validate ensemble type vs split mode
    if args.ensemble_type in ("supervised", "hybrid") and "train_benign" not in splits:
        logger.error(
            "ensemble-type '%s' requires semi-supervised splits "
            "(with train_benign/train_jailbreak keys). "
            "Use --split-mode semi_supervised in create_splits.py first.",
            args.ensemble_type,
        )
        sys.exit(1)

    # Get training and validation data
    if "train" in splits:
        X_train = store.get_activations(args.layer, indices=splits["train"].indices.tolist())
        X_val = store.get_activations(args.layer, indices=splits["val"].indices.tolist())
        X_train_jailbreak = None
        X_val_jailbreak = None
    else:
        X_train = store.get_activations(args.layer, indices=splits["train_benign"].indices.tolist())
        X_val = store.get_activations(args.layer, indices=splits["val_benign"].indices.tolist())
        if args.ensemble_type in ("supervised", "hybrid"):
            X_train_jailbreak = store.get_activations(
                args.layer, indices=splits["train_jailbreak"].indices.tolist()
            )
            X_val_jailbreak = store.get_activations(
                args.layer, indices=splits["val_jailbreak"].indices.tolist()
            )
        else:
            X_train_jailbreak = None
            X_val_jailbreak = None

    logger.info(
        "Training data: layer=%d, n_train=%d, n_val=%d, n_features=%d%s",
        args.layer, len(X_train), len(X_val), X_train.shape[1],
        f", n_train_jailbreak={len(X_train_jailbreak)}, "
        f"n_val_jailbreak={len(X_val_jailbreak)}"
        if X_train_jailbreak is not None else "",
    )

    # Create and train ensemble
    factory = _ENSEMBLE_FACTORIES[args.ensemble_type]
    ensemble = factory(threshold_percentile=args.threshold_percentile)

    t0 = time.perf_counter()
    ensemble.fit(
        X_train, X_val,
        X_train_jailbreak=X_train_jailbreak,
        X_val_jailbreak=X_val_jailbreak,
    )
    elapsed = time.perf_counter() - t0

    # Save ensemble
    output = Path(args.output)
    ensemble.save(output)

    # Also save pipeline metadata for evaluate.py to read
    meta = {
        "layer": args.layer,
        "model_id": store.model_id,
        "extraction_layers": store.layers,
        "ensemble_type": args.ensemble_type,
    }
    with open(output / "pipeline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Compute validation FPR
    val_result = ensemble.score(X_val)
    val_fpr = float(np.mean(val_result.is_anomaly))

    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"  Store:              {args.store}")
    print(f"  Ensemble type:      {args.ensemble_type}")
    print(f"  Detectors:          {', '.join(n for n, _, _ in ensemble.detectors)}")
    print(f"  Layer:              {args.layer}")
    print(f"  Training samples:   {len(X_train)} benign" + (
        f" + {len(X_train_jailbreak)} jailbreak" if X_train_jailbreak is not None else ""
    ))
    print(f"  Validation samples: {len(X_val)} benign" + (
        f" + {len(X_val_jailbreak)} jailbreak" if X_val_jailbreak is not None else ""
    ))
    print(f"  Threshold pctile:   {args.threshold_percentile}")
    print(f"  Threshold value:    {ensemble.threshold_:.4f}")
    print(f"  Val FPR:            {val_fpr:.4f} ({val_fpr * 100:.1f}%)")
    print(f"  Training time:      {elapsed:.1f}s")
    print(f"  Saved to:           {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
