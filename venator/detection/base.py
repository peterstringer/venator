"""Abstract base class for anomaly detectors.

All detectors implement a common interface: fit on benign-only training data,
score new samples (higher = more anomalous), and save/load model state.
"""
