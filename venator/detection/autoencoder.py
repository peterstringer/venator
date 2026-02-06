"""Autoencoder-based anomaly detector.

Learns a compressed representation of benign activations via reconstruction.
Anomaly score is the reconstruction error — jailbreak activations that deviate
from the learned manifold produce higher reconstruction loss. Uses PyTorch (CPU).
"""
