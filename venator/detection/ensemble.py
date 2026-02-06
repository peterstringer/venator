"""Weighted ensemble of anomaly detectors with score normalization.

Combines multiple detectors with decorrelated errors for robust detection
(per ELK paper: "useful even when no more accurate, as long as errors are
decorrelated"). Normalizes scores to common scale before weighted combination.

Score combination pipeline:
    1. Each detector produces raw anomaly scores.
    2. Normalize each to [0, 1] via percentile rank against training scores.
    3. Weighted average of normalized scores.
    4. Threshold calibrated from validation set at a chosen percentile.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.base import AnomalyDetector
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

logger = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    """Result from a single detector in the ensemble."""

    name: str
    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    weight: float


@dataclass
class EnsembleResult:
    """Complete result from the ensemble scoring pipeline."""

    ensemble_scores: np.ndarray
    is_anomaly: np.ndarray
    threshold: float
    detector_results: list[DetectorResult]


class DetectorEnsemble:
    """Weighted ensemble of anomaly detectors with decorrelated errors.

    From the ELK paper: "An ELK method can be useful even when it is no more
    accurate than other sources of information, as long as its errors are
    decorrelated."

    The ensemble normalizes each detector's scores to a common [0, 1] scale
    via percentile rank against training scores, then takes a weighted average.
    The threshold is calibrated on a validation set (benign-only) at a chosen
    percentile.

    Attributes:
        threshold_percentile: Percentile of validation ensemble scores to use
            as the anomaly threshold. E.g. 95.0 means ~5% of benign validation
            samples will be flagged as anomalous (false positive rate control).
        detectors: List of (name, detector, weight) tuples.
        train_scores_: Raw training scores per detector, for normalization.
        threshold_: Calibrated decision threshold (set after fit).
    """

    def __init__(self, threshold_percentile: float = 95.0) -> None:
        if not (0 < threshold_percentile < 100):
            raise ValueError(
                f"threshold_percentile must be in (0, 100), got {threshold_percentile}"
            )
        self.threshold_percentile = threshold_percentile
        self.detectors: list[tuple[str, AnomalyDetector, float]] = []
        self.train_scores_: dict[str, np.ndarray] | None = None
        self.threshold_: float | None = None

    def add_detector(
        self, name: str, detector: AnomalyDetector, weight: float = 1.0
    ) -> None:
        """Add a detector to the ensemble.

        Args:
            name: Unique identifier for this detector.
            detector: An AnomalyDetector instance (unfitted is fine).
            weight: Relative weight in the ensemble combination.
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        existing_names = {n for n, _, _ in self.detectors}
        if name in existing_names:
            raise ValueError(f"Detector name '{name}' already exists in ensemble")
        self.detectors.append((name, detector, weight))

    def fit(
        self, X_train: np.ndarray, X_val: np.ndarray
    ) -> DetectorEnsemble:
        """Fit all detectors and calibrate the ensemble threshold.

        1. Fits each detector on X_train (benign-only).
        2. Scores X_train with each detector and stores for normalization.
        3. Scores X_val through the full ensemble pipeline.
        4. Sets the threshold at the configured percentile of val scores.

        Args:
            X_train: Normal/benign activations for training, (n_train, n_features).
            X_val: Normal/benign activations for threshold calibration.

        Returns:
            self, for method chaining.
        """
        if len(self.detectors) == 0:
            raise RuntimeError("No detectors added. Call add_detector() first.")

        if X_train.ndim != 2 or X_val.ndim != 2:
            raise ValueError("X_train and X_val must be 2D arrays")

        self.train_scores_ = {}

        # 1. Fit each detector and store training scores for normalization
        for name, detector, weight in self.detectors:
            logger.info("Fitting detector '%s' (weight=%.1f)...", name, weight)
            detector.fit(X_train)

            raw_train = detector.score(X_train)
            self.train_scores_[name] = np.sort(raw_train)

            logger.info(
                "  %s train scores: mean=%.4f, std=%.4f, max=%.4f",
                name,
                raw_train.mean(),
                raw_train.std(),
                raw_train.max(),
            )

        # 2. Score validation set through the full pipeline
        val_result = self.score(X_val)
        val_scores = val_result.ensemble_scores

        # 3. Set threshold at configured percentile
        self.threshold_ = float(np.percentile(val_scores, self.threshold_percentile))

        logger.info(
            "Ensemble threshold set at %.4f (%.1f%% percentile of val scores, "
            "val_mean=%.4f, val_max=%.4f)",
            self.threshold_,
            self.threshold_percentile,
            val_scores.mean(),
            val_scores.max(),
        )

        return self

    def score(self, X: np.ndarray) -> EnsembleResult:
        """Score data through the full ensemble pipeline.

        Args:
            X: Data to score, shape (n_samples, n_features).

        Returns:
            EnsembleResult with combined scores, anomaly flags, and per-detector
            results.
        """
        if self.train_scores_ is None:
            raise RuntimeError("Ensemble has not been fitted. Call fit() first.")

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        detector_results = []
        total_weight = sum(w for _, _, w in self.detectors)

        weighted_sum = np.zeros(X.shape[0], dtype=np.float64)

        for name, detector, weight in self.detectors:
            raw = detector.score(X)
            normalized = self._normalize_scores(raw, name)

            detector_results.append(
                DetectorResult(
                    name=name,
                    raw_scores=raw,
                    normalized_scores=normalized,
                    weight=weight,
                )
            )

            weighted_sum += normalized * weight

        ensemble_scores = weighted_sum / total_weight

        # Apply threshold (use stored threshold if available, else no thresholding)
        threshold = self.threshold_ if self.threshold_ is not None else float("inf")
        is_anomaly = ensemble_scores > threshold

        return EnsembleResult(
            ensemble_scores=ensemble_scores.astype(np.float64),
            is_anomaly=is_anomaly,
            threshold=threshold,
            detector_results=detector_results,
        )

    def _normalize_scores(
        self, raw_scores: np.ndarray, detector_name: str
    ) -> np.ndarray:
        """Normalize scores to [0, 1] via percentile rank against training.

        This is critical — raw Mahalanobis distances and Isolation Forest scores
        are on completely different scales. Percentile normalization puts them
        on a common scale before weighted combination.

        Args:
            raw_scores: Raw detector scores to normalize.
            detector_name: Name of the detector (to look up training scores).

        Returns:
            Normalized scores in [0, 1].
        """
        sorted_train = self.train_scores_[detector_name]
        n_train = len(sorted_train)

        # searchsorted on sorted array gives the rank
        ranks = np.searchsorted(sorted_train, raw_scores, side="right")
        return ranks.astype(np.float64) / n_train

    def save(self, path: Path | str) -> None:
        """Save ensemble state: all detectors, training scores, threshold.

        Args:
            path: Directory to save all ensemble files into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each detector in its own subdirectory
        detector_meta = []
        for name, detector, weight in self.detectors:
            det_dir = path / name
            detector.save(det_dir)
            detector_meta.append({
                "name": name,
                "weight": weight,
                "type": type(detector).__name__,
            })

        # Save training scores for normalization
        if self.train_scores_ is not None:
            np.savez(
                path / "train_scores.npz",
                **{name: scores for name, scores in self.train_scores_.items()},
            )

        # Save config
        config = {
            "threshold_percentile": self.threshold_percentile,
            "threshold": self.threshold_,
            "detectors": detector_meta,
        }
        with open(path / "ensemble_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved DetectorEnsemble to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> DetectorEnsemble:
        """Load a saved DetectorEnsemble from disk.

        Args:
            path: Directory containing saved ensemble files.

        Returns:
            A fitted DetectorEnsemble.
        """
        path = Path(path)

        with open(path / "ensemble_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        ensemble = cls(threshold_percentile=config["threshold_percentile"])
        ensemble.threshold_ = config["threshold"]

        # Map type names to classes
        _type_map: dict[str, type[AnomalyDetector]] = {
            "PCAMahalanobisDetector": PCAMahalanobisDetector,
            "IsolationForestDetector": IsolationForestDetector,
            "AutoencoderDetector": AutoencoderDetector,
        }

        for det_info in config["detectors"]:
            name = det_info["name"]
            weight = det_info["weight"]
            det_type = det_info["type"]

            det_cls = _type_map.get(det_type)
            if det_cls is None:
                raise ValueError(f"Unknown detector type: {det_type}")

            detector = det_cls.load(path / name)
            ensemble.detectors.append((name, detector, weight))

        # Load training scores
        train_scores_path = path / "train_scores.npz"
        if train_scores_path.exists():
            data = np.load(train_scores_path)
            ensemble.train_scores_ = {name: data[name] for name in data.files}

        logger.info("Loaded DetectorEnsemble from %s", path)
        return ensemble


def create_default_ensemble(
    threshold_percentile: float = 95.0,
) -> DetectorEnsemble:
    """Create the recommended ensemble configuration.

    Weights from ELK paper findings:
    - PCA+Mahalanobis: 2.0 (primary detector, 0.94+ AUROC)
    - Isolation Forest: 1.5 (complementary tree-based bias)
    - Autoencoder: 1.0 (reconstruction-based diversity)

    Args:
        threshold_percentile: Percentile for threshold calibration.

    Returns:
        An unfitted DetectorEnsemble with the default detector configuration.
    """
    ensemble = DetectorEnsemble(threshold_percentile=threshold_percentile)
    ensemble.add_detector("pca_mahalanobis", PCAMahalanobisDetector(), weight=2.0)
    ensemble.add_detector("isolation_forest", IsolationForestDetector(), weight=1.5)
    ensemble.add_detector("autoencoder", AutoencoderDetector(), weight=1.0)
    return ensemble
