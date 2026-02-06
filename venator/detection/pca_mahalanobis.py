"""PCA + Mahalanobis distance anomaly detector.

Primary detector validated by ELK literature (0.94+ AUROC). Reduces activation
dimensionality via PCA, then computes Mahalanobis distance from the learned
distribution of benign activations. Assumes approximate Gaussianity in PCA space.

Pipeline:
    1. PCA reduces ~4096-dim activations to n_components (default 50).
    2. Fit Gaussian (mean + covariance) on PCA-reduced training data.
    3. Score = Mahalanobis distance from the fitted Gaussian.

Why PCA first:
    - Raw 4096 dims with ~350 training samples = badly estimated covariance.
    - Need n_samples >> n_features for stable Mahalanobis.
    - PCA preserves directions of maximum variance (likely the useful signal).
    - 50 dims with 350 samples = healthy 7:1 ratio.

Why Mahalanobis:
    - Accounts for feature correlations (unlike Euclidean).
    - Has a natural probabilistic interpretation (chi-squared under Gaussian).
    - Simple, fast, interpretable.
    - Validated in the ELK literature.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from venator.detection.base import AnomalyDetector

logger = logging.getLogger(__name__)


class PCAMahalanobisDetector(AnomalyDetector):
    """PCA dimensionality reduction + Mahalanobis distance anomaly detection.

    Attributes:
        n_components: Target number of PCA dimensions.
        regularization: Ridge added to covariance diagonal for numerical stability.
        pca_: Fitted PCA model (set after fit).
        mean_: Mean of training data in PCA space (set after fit).
        cov_inv_: Inverse covariance in PCA space (set after fit).
    """

    def __init__(
        self,
        n_components: int = 50,
        regularization: float = 1e-6,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization}")

        self.n_components = n_components
        self.regularization = regularization

        # Set after fit()
        self.pca_: PCA | None = None
        self.mean_: np.ndarray | None = None
        self.cov_inv_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> PCAMahalanobisDetector:
        """Fit PCA and Gaussian on normal activation vectors.

        Args:
            X: Training data of shape (n_samples, n_features).
               Must contain ONLY benign/normal activations.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If X has fewer than 2 samples or 1 feature.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 training samples, got {n_samples}"
            )

        # Adjust PCA dims if needed — can't exceed min(n_samples-1, n_features)
        effective_components = min(self.n_components, n_samples - 1, n_features)
        if effective_components < self.n_components:
            warnings.warn(
                f"Reduced PCA components from {self.n_components} to "
                f"{effective_components} (n_samples={n_samples}, "
                f"n_features={n_features})"
            )

        # 1. PCA dimensionality reduction
        self.pca_ = PCA(n_components=effective_components)
        X_reduced = self.pca_.fit_transform(X)

        # 2. Gaussian parameters in PCA space
        self.mean_ = X_reduced.mean(axis=0)
        cov = np.cov(X_reduced, rowvar=False)

        # np.cov returns a scalar when n_components=1 — normalize to 2D
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        elif cov.ndim == 1:
            cov = cov.reshape(1, 1)

        # Regularize for numerical stability
        cov += self.regularization * np.eye(cov.shape[0])
        self.cov_inv_ = np.linalg.inv(cov)

        logger.info(
            "Fitted PCA+Mahalanobis: %d components, explained_variance=%.3f, "
            "n_train=%d",
            effective_components,
            float(self.pca_.explained_variance_ratio_.sum()),
            n_samples,
        )

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance in PCA space for each sample.

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            Mahalanobis distances of shape (n_samples,).
            Higher = more anomalous.

        Raises:
            RuntimeError: If called before fit().
        """
        if self.pca_ is None or self.mean_ is None or self.cov_inv_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        X_reduced = self.pca_.transform(X)
        diff = X_reduced - self.mean_

        # Mahalanobis: sqrt( (x-mu)^T @ Sigma^{-1} @ (x-mu) )
        left = diff @ self.cov_inv_
        mahal_sq = np.sum(left * diff, axis=1)

        # Clamp to avoid sqrt of negative due to floating point
        distances = np.sqrt(np.maximum(mahal_sq, 0.0))

        return distances

    def save(self, path: Path | str) -> None:
        """Save PCA model and Gaussian parameters to a directory.

        Args:
            path: Directory to save model files into.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pca_, path / "pca.joblib")
        np.savez(
            path / "gaussian.npz",
            mean=self.mean_,
            cov_inv=self.cov_inv_,
            n_components=np.array(self.n_components),
            regularization=np.array(self.regularization),
        )
        logger.info("Saved PCAMahalanobisDetector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> PCAMahalanobisDetector:
        """Load a saved PCAMahalanobisDetector from disk.

        Args:
            path: Directory containing pca.joblib and gaussian.npz.

        Returns:
            A fitted PCAMahalanobisDetector.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        data = np.load(path / "gaussian.npz")

        detector = cls(
            n_components=int(data["n_components"]),
            regularization=float(data["regularization"]),
        )
        detector.pca_ = joblib.load(path / "pca.joblib")
        detector.mean_ = data["mean"]
        detector.cov_inv_ = data["cov_inv"]

        logger.info("Loaded PCAMahalanobisDetector from %s", path)
        return detector
