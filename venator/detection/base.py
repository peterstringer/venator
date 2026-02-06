"""Abstract base class for anomaly detectors.

All detectors implement a common interface: fit on benign-only training data,
score new samples (higher = more anomalous), and save/load model state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AnomalyDetector(ABC):
    """Base class for anomaly detectors.

    Contract:
    - fit() on reference (normal) distribution ONLY.
    - score() returns anomaly scores where HIGHER = MORE ANOMALOUS.
    - Scores are raw (not normalized) — the ensemble handles normalization.

    Subclasses must implement fit, score, save, and load.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> AnomalyDetector:
        """Fit on normal data only.

        Args:
            X: Training data of shape (n_samples, n_features).
               Must contain ONLY benign/normal samples.

        Returns:
            self, for method chaining.
        """
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for each sample.

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            Scores of shape (n_samples,). Higher = more anomalous.
        """
        ...

    def score_single(self, x: np.ndarray) -> float:
        """Score a single sample.

        Args:
            x: Single sample of shape (n_features,).

        Returns:
            Anomaly score (scalar). Higher = more anomalous.
        """
        return float(self.score(x.reshape(1, -1))[0])

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model state to disk.

        Args:
            path: Directory to save model files into.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AnomalyDetector:
        """Load a saved model from disk.

        Args:
            path: Directory containing saved model files.

        Returns:
            A fitted detector instance.
        """
        ...
