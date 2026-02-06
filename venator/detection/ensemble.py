"""Weighted ensemble of anomaly detectors with score normalization.

Combines multiple detectors with decorrelated errors for robust detection
(per ELK paper: "useful even when no more accurate, as long as errors are
decorrelated"). Normalizes scores to common scale before weighted combination.
"""
