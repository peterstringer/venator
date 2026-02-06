"""Ablation studies: compare layers, PCA dimensions, and detector variants.

Usage:
    python scripts/run_ablations.py \
        --train data/activations/benign_train.h5 \
        --test-benign data/activations/benign_test.h5 \
        --test-jailbreak data/activations/jailbreak_test.h5 \
        --ablate layers pca_dims detectors \
        --output results/ablations/
"""
