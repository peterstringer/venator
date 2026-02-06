"""Full evaluation of trained detectors on test data (benign + jailbreak).

Usage:
    python scripts/evaluate.py \
        --model-dir models/detector_v1/ \
        --test-benign data/activations/benign_test.h5 \
        --test-jailbreak data/activations/jailbreak_test.h5 \
        --output results/eval_v1.json
"""
