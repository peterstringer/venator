#!/usr/bin/env python3
"""Split activation data into train/val/test with strict methodology constraints.

Supports two modes:
- UNSUPERVISED: training and validation contain ONLY benign prompts.
- SEMI_SUPERVISED: a small labeled jailbreak fraction in train/val for
  supervised/hybrid detectors.

Usage:
    # Unsupervised (default)
    python scripts/create_splits.py \
        --store data/activations/all.h5 \
        --output data/splits.json

    # Semi-supervised (for supervised/hybrid detectors)
    python scripts/create_splits.py \
        --store data/activations/all.h5 \
        --output data/splits_semi.json \
        --split-mode semi_supervised
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from venator.activation.storage import ActivationStore
from venator.data.splits import SplitManager, SplitMode

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
        "--split-mode",
        choices=["unsupervised", "semi_supervised"],
        default="unsupervised",
        help="Split mode: unsupervised (jailbreaks test-only) or "
        "semi_supervised (small jailbreak fraction in train/val). "
        "Default: unsupervised",
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

    mode = SplitMode(args.split_mode)

    # Load store and create splits
    store = ActivationStore(args.store)
    logger.info("Loaded store: %s", store)

    manager = SplitManager(seed=args.seed)
    splits = manager.create_splits(
        store,
        mode=mode,
        benign_train_frac=args.train_frac,
        benign_val_frac=args.val_frac,
    )

    # Validate (also done inside create_splits, but explicit for user confidence)
    manager.validate_splits(splits, store, mode=mode)
    if mode == SplitMode.UNSUPERVISED:
        logger.info("Validation passed: no jailbreaks in train/val")
    else:
        logger.info("Validation passed: label integrity verified for all splits")

    # Save
    manager.save_splits(splits, args.output, mode=mode)

    # Summary table
    print("\n" + "=" * 60)
    print(f"Split Summary ({mode.value})")
    print("=" * 60)
    print(f"{'Split':<20} | {'N Samples':>10} | {'% Jailbreak':>12}")
    print("-" * 60)

    total = 0
    for name, split in splits.items():
        pct = 100.0 if split.contains_jailbreaks else 0.0
        print(f"{name:<20} | {split.n_samples:>10} | {pct:>11.1f}%")
        total += split.n_samples

    print("-" * 60)
    print(f"{'Total':<20} | {total:>10} |")
    print("=" * 60)

    print(f"\nSaved to: {args.output}")
    if mode == SplitMode.UNSUPERVISED:
        print("No jailbreaks in train or val splits (methodology verified)")
    else:
        print("Semi-supervised: small jailbreak fraction in train/val, "
              "majority reserved for testing")


if __name__ == "__main__":
    main()
