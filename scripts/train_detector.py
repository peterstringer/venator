"""Train anomaly detectors on benign-only activation data.

Usage:
    python scripts/train_detector.py \
        --train-activations data/activations/benign_train.h5 \
        --val-activations data/activations/benign_val.h5 \
        --detector pca_mahalanobis \
        --pca-dims 50 \
        --output models/detector_v1/
"""
