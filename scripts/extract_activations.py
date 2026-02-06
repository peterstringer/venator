#!/usr/bin/env python3
"""Batch extraction of hidden state activations from an LLM.

Extracts activations from specified transformer layers for each prompt in a
JSONL file, storing the results in an HDF5 activation store for downstream
anomaly detection.

Usage:
    python scripts/extract_activations.py \
        --prompts data/prompts/benign.jsonl \
        --output data/activations/all.h5

    python scripts/extract_activations.py \
        --prompts data/prompts/benign.jsonl \
        --output data/activations/all.h5 \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --layers 12 14 16 18 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path so we can import venator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # type: ignore[import-untyped]

from venator.activation.extractor import ActivationExtractor
from venator.activation.storage import ActivationStore
from venator.config import VenatorConfig
from venator.data.prompts import PromptDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    config = VenatorConfig()

    parser = argparse.ArgumentParser(
        description="Extract hidden state activations from an LLM for each prompt"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Path to a JSONL file containing prompts (from collect_prompts.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the HDF5 activation store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.model_id,
        help=f"HuggingFace model ID (default: {config.model_id})",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=config.extraction_layers,
        help=f"Transformer layer indices to extract (default: {config.extraction_layers})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.extraction_batch_size,
        help=f"Batch size for extraction (default: {config.extraction_batch_size})",
    )
    args = parser.parse_args()

    if not args.prompts.exists():
        logger.error("Prompts file not found: %s", args.prompts)
        sys.exit(1)

    if args.output.exists():
        logger.error("Output file already exists: %s (delete it first)", args.output)
        sys.exit(1)

    # Load prompts
    logger.info("Loading prompts from %s", args.prompts)
    dataset = PromptDataset.load(args.prompts)
    logger.info("Loaded %d prompts (%d benign, %d jailbreak)", len(dataset), dataset.n_benign, dataset.n_jailbreaks)

    # Create extractor
    logger.info("Initializing extractor (model=%s, layers=%s)", args.model, args.layers)
    extractor = ActivationExtractor(model_id=args.model, layers=args.layers)

    # Get model metadata (triggers model load)
    hidden_dim = extractor.hidden_dim
    layers = sorted(extractor._target_layers)

    # Create output store
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store = ActivationStore.create(args.output, args.model, layers, hidden_dim)

    # Extract one-by-one with progress bar
    t0 = time.perf_counter()

    for i in tqdm(range(len(dataset)), desc="Extracting activations"):
        prompt = dataset.prompts[i]
        label = dataset.labels[i]
        activations = extractor.extract_single(prompt)
        store.append(prompt, activations, label=label)

    elapsed = time.perf_counter() - t0

    # Summary
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  Layers:        {layers}")
    print(f"  Hidden dim:    {hidden_dim}")
    print(f"  Prompts:       {len(dataset)}")
    print(f"    Benign:      {dataset.n_benign}")
    print(f"    Jailbreak:   {dataset.n_jailbreaks}")
    print(f"  Output:        {args.output}")
    print(f"  Time:          {elapsed:.1f}s ({elapsed / len(dataset):.2f}s/prompt)")
    print("=" * 60)


if __name__ == "__main__":
    main()
