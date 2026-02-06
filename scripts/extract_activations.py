"""Batch extraction of hidden state activations from an LLM.

Usage:
    python scripts/extract_activations.py \
        --prompts data/prompts/benign_train.jsonl \
        --output data/activations/benign_train.h5 \
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
        --layers 12 14 16 18 20 \
        --batch-size 1
"""
