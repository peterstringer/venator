"""Isolation Forest anomaly detector.

Tree-based detector with no distributional assumptions — provides complementary
bias to the Gaussian-assuming PCA+Mahalanobis detector. Decorrelated errors
strengthen the ensemble per ELK paper findings.
"""
