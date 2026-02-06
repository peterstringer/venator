#!/usr/bin/env python3
"""Gather benign prompts (Alpaca, MMLU) and jailbreak datasets (JailbreakBench, AdvBench).

Collects prompts from public HuggingFace datasets and hand-curated sources,
then saves them as JSONL files for the Venator pipeline.

Usage:
    python scripts/collect_prompts.py --output data/prompts/
    python scripts/collect_prompts.py --n-benign 500 --n-jailbreak 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from venator.data.prompts import (
    PromptDataset,
    collect_benign_prompts,
    collect_jailbreak_prompts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect benign and jailbreak prompts for Venator"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prompts"),
        help="Output directory for JSONL files (default: data/prompts/)",
    )
    parser.add_argument(
        "--n-benign",
        type=int,
        default=500,
        help="Number of benign prompts to collect (default: 500)",
    )
    parser.add_argument(
        "--n-jailbreak",
        type=int,
        default=200,
        help="Number of jailbreak prompts to collect (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect benign prompts ---
    logger.info("Collecting %d benign prompts...", args.n_benign)
    benign_items = collect_benign_prompts(n=args.n_benign, seed=args.seed)
    benign_ds = PromptDataset(
        prompts=[p for p, _ in benign_items],
        labels=[0] * len(benign_items),
        sources=[s for _, s in benign_items],
    )
    benign_path = output_dir / "benign.jsonl"
    benign_ds.save(benign_path)

    # --- Collect jailbreak prompts ---
    logger.info("Collecting %d jailbreak prompts...", args.n_jailbreak)
    jailbreak_items = collect_jailbreak_prompts(n=args.n_jailbreak, seed=args.seed)
    jailbreak_ds = PromptDataset(
        prompts=[p for p, _ in jailbreak_items],
        labels=[1] * len(jailbreak_items),
        sources=[s for _, s in jailbreak_items],
    )
    jailbreak_path = output_dir / "jailbreaks.jsonl"
    jailbreak_ds.save(jailbreak_path)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Collection Summary")
    print("=" * 60)
    print(f"\nBenign prompts:    {len(benign_ds):>5}")
    for source, count in sorted(benign_ds.source_counts().items()):
        print(f"  {source:<20} {count:>5}")
    print(f"  Saved to: {benign_path}")

    print(f"\nJailbreak prompts: {len(jailbreak_ds):>5}")
    for source, count in sorted(jailbreak_ds.source_counts().items()):
        print(f"  {source:<20} {count:>5}")
    print(f"  Saved to: {jailbreak_path}")

    total = len(benign_ds) + len(jailbreak_ds)
    print(f"\nTotal prompts:     {total:>5}")
    print("=" * 60)


if __name__ == "__main__":
    main()
