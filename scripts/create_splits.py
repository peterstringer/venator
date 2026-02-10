#!/usr/bin/env python3
"""Create unified train/val/test splits from an activation store.

Produces a single split that serves all detector types:
    Unsupervised: fit(train_benign) → calibrate(val_benign) → test(test_*)
    Supervised:   fit(train_benign + train_jailbreak) → calibrate(val_*) → test(test_*)

Usage:
    python scripts/create_splits.py \
        --store data/activations/all.h5 \
        --output data/splits.json

    python scripts/create_splits.py \
        --store data/activations/all.h5 \
        --output data/splits.json \
        --jailbreak-train-frac 0.10 \
        --jailbreak-val-frac 0.10
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
        description="Create unified train/val/test splits from an activation store"
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
        "--jailbreak-train-frac",
        type=float,
        default=0.15,
        help="Fraction of jailbreaks for training (default: 0.15)",
    )
    parser.add_argument(
        "--jailbreak-val-frac",
        type=float,
        default=0.15,
        help="Fraction of jailbreaks for validation (default: 0.15)",
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
        benign_train_frac=args.train_frac,
        benign_val_frac=args.val_frac,
        jailbreak_train_frac=args.jailbreak_train_frac,
        jailbreak_val_frac=args.jailbreak_val_frac,
    )

    # Validate (also done inside create_splits, but explicit for user confidence)
    manager.validate_splits(splits, store)
    logger.info("Validation passed: label integrity verified for all splits")

    # Save
    manager.save_splits(splits, args.output)

    # Summary table
    n_benign = sum(
        s.n_samples for name, s in splits.items() if not s.contains_jailbreaks
    )
    n_jailbreak = sum(
        s.n_samples for name, s in splits.items() if s.contains_jailbreaks
    )

    print("\n" + "=" * 65)
    print("Unified Split Summary")
    print("=" * 65)

    print(f"\n  YOUR DATA: {n_benign} benign + {n_jailbreak} jailbreak prompts\n")

    print(f"  {'Split':<20} | {'N Samples':>10} | {'Role':>25}")
    print("  " + "-" * 60)
    print(f"  {'train_benign':<20} | {splits['train_benign'].n_samples:>10} | {'unsupervised training':>25}")
    print(f"  {'train_jailbreak':<20} | {splits['train_jailbreak'].n_samples:>10} | {'supervised training':>25}")
    print(f"  {'val_benign':<20} | {splits['val_benign'].n_samples:>10} | {'all detectors':>25}")
    print(f"  {'val_jailbreak':<20} | {splits['val_jailbreak'].n_samples:>10} | {'all detectors':>25}")
    print(f"  {'test_benign':<20} | {splits['test_benign'].n_samples:>10} | {'final evaluation':>25}")
    print(f"  {'test_jailbreak':<20} | {splits['test_jailbreak'].n_samples:>10} | {'final evaluation':>25}")
    print("  " + "-" * 60)

    total = sum(s.n_samples for s in splits.values())
    print(f"  {'Total':<20} | {total:>10} |")

    print(f"\n  Split once, train anything.")
    print(f"  Saved to: {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
