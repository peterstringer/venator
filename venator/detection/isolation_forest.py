"""Isolation Forest anomaly detector.

Tree-based detector with no distributional assumptions — provides complementary
bias to the Gaussian-assuming PCA+Mahalanobis detector. Decorrelated errors
strengthen the ensemble per ELK paper findings.

Why Isolation Forest complements Mahalanobis:
    - No Gaussian assumption — detects non-ellipsoidal anomalies.
    - Works well in moderate dimensions (post-PCA).
    - Robust to irrelevant features.
    - Different failure modes → decorrelated errors in the ensemble.

Also applies PCA first for consistency with the Mahalanobis detector and
to keep the ensemble operating in the same reduced feature space.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]

from venator.detection.base import AnomalyDetector

logger = logging.getLogger(__name__)


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detection on PCA-reduced activations.

    Attributes:
        n_components: Target number of PCA dimensions.
        n_estimators: Number of trees in the forest.
        contamination: Expected proportion of outliers in training data.
            Since we train on benign-only data, keep this very low.
        random_state: Random seed for reproducibility.
        pca_: Fitted PCA model (set after fit).
        model_: Fitted IsolationForest model (set after fit).
    """

    def __init__(
        self,
        n_components: int = 50,
        n_estimators: int = 200,
        contamination: float = 0.01,
        random_state: int = 42,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")

        self.n_components = n_components
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

        # Set after fit()
        self.pca_: PCA | None = None
        self.model_: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        """Fit PCA and Isolation Forest on normal activation vectors.

        Args:
            X: Training data of shape (n_samples, n_features).
               Must contain ONLY benign/normal activations.

        Returns:
            self, for method chaining.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 training samples, got {n_samples}"
            )

        # Adjust PCA dims if needed
        effective_components = min(self.n_components, n_samples - 1, n_features)
        if effective_components < self.n_components:
            warnings.warn(
                f"Reduced PCA components from {self.n_components} to "
                f"{effective_components} (n_samples={n_samples}, "
                f"n_features={n_features})"
            )

        # 1. PCA
        self.pca_ = PCA(n_components=effective_components)
        X_reduced = self.pca_.fit_transform(X)

        # 2. Isolation Forest
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model_.fit(X_reduced)

        logger.info(
            "Fitted IsolationForest: %d components, %d estimators, n_train=%d",
            effective_components,
            self.n_estimators,
            n_samples,
        )

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores via Isolation Forest.

        sklearn's score_samples returns negative scores where lower = more
        anomalous. We negate so higher = more anomalous per our convention.

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,). Higher = more anomalous.
        """
        if self.pca_ is None or self.model_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        X_reduced = self.pca_.transform(X)

        # sklearn score_samples: lower = more anomalous → negate
        raw_scores = self.model_.score_samples(X_reduced)
        return -raw_scores

    def save(self, path: Path | str) -> None:
        """Save PCA and Isolation Forest models to a directory.

        Args:
            path: Directory to save model files into.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pca_, path / "pca.joblib")
        joblib.dump(self.model_, path / "iforest.joblib")
        np.savez(
            path / "config.npz",
            n_components=np.array(self.n_components),
            n_estimators=np.array(self.n_estimators),
            contamination=np.array(self.contamination),
            random_state=np.array(self.random_state),
        )
        logger.info("Saved IsolationForestDetector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> IsolationForestDetector:
        """Load a saved IsolationForestDetector from disk.

        Args:
            path: Directory containing saved model files.

        Returns:
            A fitted IsolationForestDetector.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        data = np.load(path / "config.npz")

        detector = cls(
            n_components=int(data["n_components"]),
            n_estimators=int(data["n_estimators"]),
            contamination=float(data["contamination"]),
            random_state=int(data["random_state"]),
        )
        detector.pca_ = joblib.load(path / "pca.joblib")
        detector.model_ = joblib.load(path / "iforest.joblib")

        logger.info("Loaded IsolationForestDetector from %s", path)
        return detector
