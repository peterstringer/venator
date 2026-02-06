#!/usr/bin/env python3
"""Split activation data into train/val/test with strict methodology constraints.

Enforces: training and validation contain ONLY benign prompts.
Jailbreak prompts appear ONLY in the test set.

Usage:
    python scripts/create_splits.py \
        --store data/activations/all.h5 \
        --output data/splits.json

    python scripts/create_splits.py \
        --store data/activations/all.h5 \
        --output data/splits.json \
        --train-frac 0.70 --val-frac 0.15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from venator.activation.storage import ActivationStore
from venator.data.splits import SplitManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from an activation store"
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Path to the HDF5 activation store",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for split definitions (JSON)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of benign prompts for training (default: 0.70)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of benign prompts for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not args.store.exists():
        logger.error("Store not found: %s", args.store)
        sys.exit(1)

    # Load store and create splits
    store = ActivationStore(args.store)
    logger.info("Loaded store: %s", store)

    manager = SplitManager(seed=args.seed)
    splits = manager.create_splits(
        store,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    # Validate (also done inside create_splits, but explicit for user confidence)
    manager.validate_splits(splits, store)
    logger.info("Validation passed: no jailbreaks in train/val")

    # Save
    manager.save_splits(splits, args.output)

    # Summary table
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    print(f"{'Split':<20} | {'N Samples':>10} | {'% Jailbreak':>12}")
    print("-" * 60)

    total = 0
    for name in ("train", "val", "test_benign", "test_jailbreak"):
        split = splits[name]
        pct = 100.0 if split.contains_jailbreaks else 0.0
        print(f"{name:<20} | {split.n_samples:>10} | {pct:>11.1f}%")
        total += split.n_samples

    print("-" * 60)
    print(f"{'Total':<20} | {total:>10} |")
    print("=" * 60)

    print(f"\nSaved to: {args.output}")
    print("No jailbreaks in train or val splits (methodology verified)")


if __name__ == "__main__":
    main()
