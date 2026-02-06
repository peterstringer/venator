"""Split prompt data into train/val/test with strict methodology constraints.

Enforces: training and validation contain ONLY benign prompts.
Jailbreak prompts appear ONLY in the test set.

Usage:
    python scripts/create_splits.py \
        --prompts data/prompts/ \
        --output data/prompts/ \
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""
