"""PCA + Mahalanobis distance anomaly detector.

Primary detector validated by ELK literature (0.94+ AUROC). Reduces activation
dimensionality via PCA, then computes Mahalanobis distance from the learned
distribution of benign activations. Assumes approximate Gaussianity in PCA space.
"""
