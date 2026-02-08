"""Contrastive direction and class-conditional Mahalanobis detectors.

These supervised detectors exploit the difference between benign and jailbreak
activation distributions directly, requiring very few labeled examples.

From the ELK paper (Mallen et al., 2024): the diff-in-means direction is
"more causally implicated" than logistic regression probes, meaning it captures
the actual computation difference rather than just a correlate.

Two variants are provided:

    ContrastiveDirectionDetector:
        Simplest possible supervised detector — projects activations onto the
        direction where benign and jailbreak means differ most. Equivalent to
        Fisher's Linear Discriminant with equal covariance assumption. Works
        with as few as 5-10 labeled jailbreaks.

    ContrastiveMahalanobisDetector:
        Class-conditional Mahalanobis distance. Fits separate Gaussian
        distributions for benign and jailbreak classes, then scores as
        distance-to-benign minus distance-to-jailbreak. Strictly better than
        the unsupervised Mahalanobis because it knows what jailbreaks look
        like, not just what benign looks like. Uses shared PCA across classes.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA  # type: ignore[import-untyped]

from venator.detection.base import AnomalyDetector

logger = logging.getLogger(__name__)


class ContrastiveDirectionDetector(AnomalyDetector):
    """Contrastive direction detector using difference-in-means.

    A SUPERVISED detector that finds the direction in activation space where
    benign and jailbreak prompts differ most, then projects new activations
    onto that direction.

    Method:
        1. Compute mean activation for benign prompts: mu_benign
        2. Compute mean activation for jailbreak prompts: mu_jailbreak
        3. Contrastive direction: d = mu_jailbreak - mu_benign (normalized)
        4. Score = projection of each activation onto d, z-scored relative
           to the benign training distribution.

    This is equivalent to Fisher's Linear Discriminant with equal covariance
    assumption, and requires very few labeled examples to estimate reliably.

    Why this is interesting:
        - Almost zero parameters to estimate (just two means)
        - Works even with 5-10 labeled jailbreaks
        - The direction itself is interpretable
        - Can be computed in the original activation space (no PCA needed)

    Attributes:
        normalize: Whether to normalize the contrastive direction to unit length.
        direction_: The learned contrastive direction (set after fit).
        benign_mean_proj_: Mean projection of benign training data (set after fit).
        benign_std_proj_: Std of benign training projections (set after fit).
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

        # Set after fit()
        self.direction_: np.ndarray | None = None
        self.benign_mean_proj_: float | None = None
        self.benign_std_proj_: float | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> ContrastiveDirectionDetector:
        """Compute contrastive direction from labeled data.

        Args:
            X: Training data of shape (n_samples, hidden_dim).
            y: Binary labels of shape (n_samples,). 0=benign, 1=jailbreak.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If y is None, X is wrong shape, or labels invalid.
        """
        if y is None:
            raise ValueError(
                "ContrastiveDirectionDetector is supervised and requires labels. "
                "Pass y with 0=benign, 1=jailbreak."
            )
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 training samples, got {n_samples}"
            )

        y = np.asarray(y, dtype=np.int64)
        if len(y) != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {len(y)} labels"
            )
        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Labels must be 0 or 1, got unique values: {unique_labels}"
            )
        if len(unique_labels) < 2:
            raise ValueError(
                "Need both benign (0) and jailbreak (1) labels for supervised "
                f"training. Got only label(s): {unique_labels}"
            )

        benign_mask = y == 0
        jailbreak_mask = y == 1

        mu_benign = X[benign_mask].mean(axis=0)
        mu_jailbreak = X[jailbreak_mask].mean(axis=0)

        self.direction_ = mu_jailbreak - mu_benign
        if self.normalize:
            norm = np.linalg.norm(self.direction_)
            if norm > 1e-10:
                self.direction_ = self.direction_ / norm

        # Score benign training data to establish baseline for z-scoring
        benign_projections = X[benign_mask] @ self.direction_
        self.benign_mean_proj_ = float(benign_projections.mean())
        self.benign_std_proj_ = float(benign_projections.std() + 1e-10)

        n_benign = int(benign_mask.sum())
        n_jailbreak = int(jailbreak_mask.sum())
        logger.info(
            "Fitted ContrastiveDirection: %d samples (%d benign, %d jailbreak), "
            "hidden_dim=%d, normalize=%s",
            n_samples,
            n_benign,
            n_jailbreak,
            X.shape[1],
            self.normalize,
        )

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Project onto contrastive direction. Higher = more jailbreak-like.

        Scores are z-scored relative to the benign training distribution
        so they're roughly centered at 0 for benign inputs.

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            Z-scored projections of shape (n_samples,).
            Higher = more anomalous (jailbreak-like).
        """
        if self.direction_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        projections = X @ self.direction_
        return (projections - self.benign_mean_proj_) / self.benign_std_proj_

    @property
    def contrastive_direction(self) -> np.ndarray:
        """The learned jailbreak direction in activation space."""
        if self.direction_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        return self.direction_.copy()

    def save(self, path: Path | str) -> None:
        """Save contrastive direction and normalization parameters.

        Args:
            path: Directory to save model files into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez(
            path / "contrastive_direction.npz",
            direction=self.direction_,
            benign_mean_proj=np.array(self.benign_mean_proj_),
            benign_std_proj=np.array(self.benign_std_proj_),
            normalize=np.array(self.normalize),
        )
        logger.info("Saved ContrastiveDirectionDetector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> ContrastiveDirectionDetector:
        """Load a saved ContrastiveDirectionDetector from disk.

        Args:
            path: Directory containing saved model files.

        Returns:
            A fitted ContrastiveDirectionDetector.
        """
        path = Path(path)
        data = np.load(path / "contrastive_direction.npz")

        detector = cls(normalize=bool(data["normalize"]))
        detector.direction_ = data["direction"]
        detector.benign_mean_proj_ = float(data["benign_mean_proj"])
        detector.benign_std_proj_ = float(data["benign_std_proj"])

        logger.info("Loaded ContrastiveDirectionDetector from %s", path)
        return detector


# ---------------------------------------------------------------------------
# Class-conditional Mahalanobis
# ---------------------------------------------------------------------------


class ContrastiveMahalanobisDetector(AnomalyDetector):
    """Class-conditional Mahalanobis distance for jailbreak detection.

    A SUPERVISED detector that fits separate Gaussian distributions for
    benign and jailbreak classes in PCA space, then scores each sample as:

        score = mahal_distance_to_benign - mahal_distance_to_jailbreak

    Higher score = closer to jailbreak cluster, further from benign cluster.

    This is strictly better than the unsupervised PCAMahalanobisDetector
    because it knows what jailbreaks look like, not just what benign looks
    like. It uses shared PCA across both classes for dimensionality reduction.

    Attributes:
        n_components: Target number of PCA dimensions.
        regularization: Ridge added to covariance diagonal for stability.
        pca_: Fitted PCA model (set after fit).
        mean_benign_: Benign mean in PCA space (set after fit).
        cov_inv_benign_: Inverse benign covariance (set after fit).
        mean_jailbreak_: Jailbreak mean in PCA space (set after fit).
        cov_inv_jailbreak_: Inverse jailbreak covariance (set after fit).
    """

    def __init__(
        self,
        n_components: int = 50,
        regularization: float = 1e-5,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if regularization < 0:
            raise ValueError(
                f"regularization must be >= 0, got {regularization}"
            )

        self.n_components = n_components
        self.regularization = regularization

        # Set after fit()
        self.pca_: PCA | None = None
        self.mean_benign_: np.ndarray | None = None
        self.cov_inv_benign_: np.ndarray | None = None
        self.mean_jailbreak_: np.ndarray | None = None
        self.cov_inv_jailbreak_: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> ContrastiveMahalanobisDetector:
        """Fit PCA and class-conditional Gaussians on labeled data.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,). 0=benign, 1=jailbreak.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If y is None, X is wrong shape, or labels invalid.
        """
        if y is None:
            raise ValueError(
                "ContrastiveMahalanobisDetector is supervised and requires "
                "labels. Pass y with 0=benign, 1=jailbreak."
            )
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 training samples, got {n_samples}"
            )

        y = np.asarray(y, dtype=np.int64)
        if len(y) != n_samples:
            raise ValueError(
                f"X has {n_samples} samples but y has {len(y)} labels"
            )
        unique_labels = set(np.unique(y))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"Labels must be 0 or 1, got unique values: {unique_labels}"
            )
        if len(unique_labels) < 2:
            raise ValueError(
                "Need both benign (0) and jailbreak (1) labels for supervised "
                f"training. Got only label(s): {unique_labels}"
            )

        benign_mask = y == 0
        jailbreak_mask = y == 1
        n_benign = int(benign_mask.sum())
        n_jailbreak = int(jailbreak_mask.sum())

        # --- Shared PCA on all data ---
        effective_components = min(self.n_components, n_samples - 1, n_features)
        if effective_components < self.n_components:
            warnings.warn(
                f"Reduced PCA components from {self.n_components} to "
                f"{effective_components} (n_samples={n_samples}, "
                f"n_features={n_features})"
            )

        self.pca_ = PCA(n_components=effective_components)
        X_reduced = self.pca_.fit_transform(X)

        # --- Benign Gaussian ---
        X_benign = X_reduced[benign_mask]
        self.mean_benign_ = X_benign.mean(axis=0)
        cov_benign = np.cov(X_benign, rowvar=False)
        cov_benign = self._regularize_cov(cov_benign, effective_components)
        self.cov_inv_benign_ = np.linalg.inv(cov_benign)

        # --- Jailbreak Gaussian ---
        X_jailbreak = X_reduced[jailbreak_mask]
        self.mean_jailbreak_ = X_jailbreak.mean(axis=0)
        cov_jailbreak = np.cov(X_jailbreak, rowvar=False)
        cov_jailbreak = self._regularize_cov(cov_jailbreak, effective_components)
        self.cov_inv_jailbreak_ = np.linalg.inv(cov_jailbreak)

        logger.info(
            "Fitted ContrastiveMahalanobis: %d components, "
            "n_benign=%d, n_jailbreak=%d, explained_var=%.3f",
            effective_components,
            n_benign,
            n_jailbreak,
            float(self.pca_.explained_variance_ratio_.sum()),
        )

        return self

    def _regularize_cov(
        self, cov: np.ndarray, n_components: int
    ) -> np.ndarray:
        """Ensure covariance is a well-conditioned 2D matrix."""
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        elif cov.ndim == 1:
            cov = cov.reshape(1, 1)
        cov += self.regularization * np.eye(n_components)
        return cov

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute class-conditional Mahalanobis score.

        score = mahal_to_benign - mahal_to_jailbreak

        Higher = closer to jailbreak cluster, further from benign cluster.

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            Scores of shape (n_samples,). Higher = more anomalous.
        """
        if self.pca_ is None or self.mean_benign_ is None:
            raise RuntimeError("Detector has not been fitted. Call fit() first.")
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        X_reduced = self.pca_.transform(X)

        # Mahalanobis distance to benign class
        diff_b = X_reduced - self.mean_benign_
        mahal_sq_b = np.sum((diff_b @ self.cov_inv_benign_) * diff_b, axis=1)
        mahal_b = np.sqrt(np.maximum(mahal_sq_b, 0.0))

        # Mahalanobis distance to jailbreak class
        diff_j = X_reduced - self.mean_jailbreak_
        mahal_sq_j = np.sum(
            (diff_j @ self.cov_inv_jailbreak_) * diff_j, axis=1
        )
        mahal_j = np.sqrt(np.maximum(mahal_sq_j, 0.0))

        # Higher = more jailbreak-like
        return mahal_b - mahal_j

    def save(self, path: Path | str) -> None:
        """Save PCA model and class-conditional Gaussian parameters.

        Args:
            path: Directory to save model files into.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pca_, path / "pca.joblib")
        np.savez(
            path / "class_gaussians.npz",
            mean_benign=self.mean_benign_,
            cov_inv_benign=self.cov_inv_benign_,
            mean_jailbreak=self.mean_jailbreak_,
            cov_inv_jailbreak=self.cov_inv_jailbreak_,
            n_components=np.array(self.n_components),
            regularization=np.array(self.regularization),
        )
        logger.info("Saved ContrastiveMahalanobisDetector to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> ContrastiveMahalanobisDetector:
        """Load a saved ContrastiveMahalanobisDetector from disk.

        Args:
            path: Directory containing saved model files.

        Returns:
            A fitted ContrastiveMahalanobisDetector.
        """
        import joblib  # type: ignore[import-untyped]

        path = Path(path)
        data = np.load(path / "class_gaussians.npz")

        detector = cls(
            n_components=int(data["n_components"]),
            regularization=float(data["regularization"]),
        )
        detector.pca_ = joblib.load(path / "pca.joblib")
        detector.mean_benign_ = data["mean_benign"]
        detector.cov_inv_benign_ = data["cov_inv_benign"]
        detector.mean_jailbreak_ = data["mean_jailbreak"]
        detector.cov_inv_jailbreak_ = data["cov_inv_jailbreak"]

        logger.info("Loaded ContrastiveMahalanobisDetector from %s", path)
        return detector
