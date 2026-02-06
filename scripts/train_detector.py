#!/usr/bin/env python3
"""Train anomaly detectors on benign-only activation data.

Creates a default ensemble (PCA+Mahalanobis, Isolation Forest, Autoencoder),
trains on benign-only training activations, calibrates the threshold on benign
validation activations, and saves the trained model.

Usage:
    python scripts/train_detector.py \
        --store data/activations/all.h5 \
        --splits data/splits.json \
        --output models/detector_v1/

    python scripts/train_detector.py \
        --store data/activations/all.h5 \
        --splits data/splits.json \
        --layer 16 \
        --output models/detector_v1/
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
from venator.detection.ensemble import DetectorEnsemble, create_default_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = VenatorConfig()

    parser = argparse.ArgumentParser(
        description="Train anomaly detectors on benign-only activation data"
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
    logger.info("Loaded store: %s", store)
    logger.info("Loaded splits: %s", {name: s.n_samples for name, s in splits.items()})

    # Get training and validation data
    X_train = store.get_activations(args.layer, indices=splits["train"].indices.tolist())
    X_val = store.get_activations(args.layer, indices=splits["val"].indices.tolist())
    logger.info(
        "Training data: layer=%d, n_train=%d, n_val=%d, n_features=%d",
        args.layer, len(X_train), len(X_val), X_train.shape[1],
    )

    # Create and train ensemble
    ensemble = create_default_ensemble(threshold_percentile=args.threshold_percentile)

    t0 = time.perf_counter()
    ensemble.fit(X_train, X_val)
    elapsed = time.perf_counter() - t0

    # Save ensemble
    output = Path(args.output)
    ensemble.save(output)

    # Also save pipeline metadata for evaluate.py to read
    meta = {
        "layer": args.layer,
        "model_id": store.model_id,
        "extraction_layers": store.layers,
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
    print(f"  Layer:              {args.layer}")
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Threshold pctile:   {args.threshold_percentile}")
    print(f"  Threshold value:    {ensemble.threshold_:.4f}")
    print(f"  Val FPR:            {val_fpr:.4f} ({val_fpr * 100:.1f}%)")
    print(f"  Training time:      {elapsed:.1f}s")
    print(f"  Saved to:           {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
