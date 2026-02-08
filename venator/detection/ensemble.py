"""Weighted ensemble of anomaly detectors with score normalization.

Combines multiple detectors with decorrelated errors for robust detection
(per ELK paper: "useful even when no more accurate, as long as errors are
decorrelated"). Normalizes scores to common scale before weighted combination.

Supports both unsupervised detectors (fit on benign data only) and supervised
detectors (fit on labeled benign + jailbreak data). When labeled validation
data is available, uses Youden's J statistic for optimal threshold selection
instead of the simpler percentile-based calibration.

Score combination pipeline:
    1. Each detector produces raw anomaly scores.
    2. Normalize each to [0, 1] via percentile rank against benign training scores.
    3. Weighted average of normalized scores.
    4. Threshold calibrated from validation set (percentile or supervised method).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve  # type: ignore[import-untyped]

from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.base import AnomalyDetector
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

logger = logging.getLogger(__name__)


class DetectorType(str, Enum):
    """Whether a detector requires labeled data for training."""

    UNSUPERVISED = "unsupervised"  # fit(X) — benign data only
    SUPERVISED = "supervised"  # fit(X, y) — labeled data


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
    via percentile rank against benign training scores, then takes a weighted
    average. The threshold is calibrated on a validation set — either at a
    chosen percentile (unsupervised) or via Youden's J / F1 (supervised).

    Supports mixed ensembles with both unsupervised and supervised detectors.
    Unsupervised detectors are fitted on benign data only; supervised detectors
    are fitted on combined benign + jailbreak data with labels.

    Attributes:
        threshold_percentile: Percentile of validation ensemble scores to use
            as the anomaly threshold (used when no labeled val data available).
        detectors: List of (name, detector, weight) tuples.
        detector_types_: Mapping from detector name to DetectorType.
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
        self.detector_types_: dict[str, DetectorType] = {}
        self.train_scores_: dict[str, np.ndarray] | None = None
        self.threshold_: float | None = None

    def add_detector(
        self,
        name: str,
        detector: AnomalyDetector,
        weight: float = 1.0,
        detector_type: DetectorType = DetectorType.UNSUPERVISED,
    ) -> None:
        """Add a detector to the ensemble.

        Args:
            name: Unique identifier for this detector.
            detector: An AnomalyDetector instance (unfitted is fine).
            weight: Relative weight in the ensemble combination.
            detector_type: Whether the detector is unsupervised or supervised.
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        existing_names = {n for n, _, _ in self.detectors}
        if name in existing_names:
            raise ValueError(f"Detector name '{name}' already exists in ensemble")
        self.detectors.append((name, detector, weight))
        self.detector_types_[name] = detector_type

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_train_jailbreak: np.ndarray | None = None,
        X_val_jailbreak: np.ndarray | None = None,
        threshold_method: str | None = None,
    ) -> DetectorEnsemble:
        """Fit all detectors and calibrate the ensemble threshold.

        Unsupervised detectors are fitted on X_train (benign-only).
        Supervised detectors are fitted on X_train + X_train_jailbreak with labels.
        All detectors are normalized against benign training scores.

        Threshold calibration:
            - If X_val_jailbreak is provided (and threshold_method != "percentile"):
              uses supervised calibration (Youden's J by default).
            - Otherwise: uses percentile of benign validation ensemble scores.

        Args:
            X_train: Benign activations for training, (n_train, n_features).
            X_val: Benign activations for threshold calibration.
            X_train_jailbreak: Jailbreak activations for supervised detectors.
                Required if any detector has detector_type=SUPERVISED.
            X_val_jailbreak: Jailbreak activations for supervised threshold
                calibration. When provided, enables Youden's J / F1 thresholding.
            threshold_method: Override threshold calibration method.
                None = auto ("youden" if labeled val data, else "percentile").
                "percentile" = percentile on benign val scores.
                "youden" = maximize TPR - FPR (Youden's J statistic).
                "f1" = maximize F1 score.
                "fpr_target" = find threshold for target FPR derived from
                    threshold_percentile (e.g. 95% → 5% FPR).

        Returns:
            self, for method chaining.
        """
        if len(self.detectors) == 0:
            raise RuntimeError("No detectors added. Call add_detector() first.")

        if X_train.ndim != 2 or X_val.ndim != 2:
            raise ValueError("X_train and X_val must be 2D arrays")

        self.train_scores_ = {}

        # 1. Fit each detector based on its type
        for name, detector, weight in self.detectors:
            dtype = self.detector_types_.get(name, DetectorType.UNSUPERVISED)
            logger.info(
                "Fitting detector '%s' (weight=%.1f, type=%s)...",
                name,
                weight,
                dtype.value,
            )

            if dtype == DetectorType.UNSUPERVISED:
                detector.fit(X_train)
            elif dtype == DetectorType.SUPERVISED:
                if X_train_jailbreak is None:
                    raise ValueError(
                        f"Detector '{name}' is supervised but no jailbreak "
                        f"training data was provided (X_train_jailbreak=None)."
                    )
                X_combined = np.vstack([X_train, X_train_jailbreak])
                y_combined = np.concatenate([
                    np.zeros(len(X_train), dtype=np.int64),
                    np.ones(len(X_train_jailbreak), dtype=np.int64),
                ])
                detector.fit(X_combined, y_combined)

            # Normalize against benign training scores for all detectors.
            # This ensures consistent interpretation: "what fraction of
            # normal training scores does this score exceed?"
            raw_train = detector.score(X_train)
            self.train_scores_[name] = np.sort(raw_train)

            logger.info(
                "  %s train scores: mean=%.4f, std=%.4f, max=%.4f",
                name,
                raw_train.mean(),
                raw_train.std(),
                raw_train.max(),
            )

        # 2. Threshold calibration
        use_supervised = (
            X_val_jailbreak is not None
            and threshold_method != "percentile"
        )

        if use_supervised:
            method = threshold_method or "youden"
            val_benign_result = self.score(X_val)
            val_jailbreak_result = self.score(X_val_jailbreak)

            self.threshold_ = self._calibrate_threshold_supervised(
                val_benign_result.ensemble_scores,
                val_jailbreak_result.ensemble_scores,
                method=method,
            )

            logger.info(
                "Ensemble threshold set at %.4f (method=%s, "
                "val_benign_mean=%.4f, val_jailbreak_mean=%.4f)",
                self.threshold_,
                method,
                val_benign_result.ensemble_scores.mean(),
                val_jailbreak_result.ensemble_scores.mean(),
            )
        else:
            # Original unsupervised: percentile on benign val scores
            val_result = self.score(X_val)
            val_scores = val_result.ensemble_scores
            self.threshold_ = float(
                np.percentile(val_scores, self.threshold_percentile)
            )

            logger.info(
                "Ensemble threshold set at %.4f (%.1f%% percentile of val "
                "scores, val_mean=%.4f, val_max=%.4f)",
                self.threshold_,
                self.threshold_percentile,
                val_scores.mean(),
                val_scores.max(),
            )

        return self

    def _calibrate_threshold_supervised(
        self,
        scores_benign: np.ndarray,
        scores_jailbreak: np.ndarray,
        method: str = "youden",
    ) -> float:
        """Find optimal threshold using labeled validation data.

        Args:
            scores_benign: Ensemble scores for benign validation samples.
            scores_jailbreak: Ensemble scores for jailbreak validation samples.
            method: Calibration method — "youden", "f1", or "fpr_target".

        Returns:
            Optimal threshold value.
        """
        scores = np.concatenate([scores_benign, scores_jailbreak])
        labels = np.concatenate([
            np.zeros(len(scores_benign), dtype=np.int64),
            np.ones(len(scores_jailbreak), dtype=np.int64),
        ])

        fpr, tpr, thresholds = roc_curve(labels, scores)

        if method == "youden":
            # Maximize TPR - FPR (Youden's J statistic)
            j_stat = tpr - fpr
            best_idx = int(np.argmax(j_stat))
            threshold = float(thresholds[best_idx])

        elif method == "f1":
            # Maximize F1 score
            n_pos = len(scores_jailbreak)
            n_neg = len(scores_benign)
            precision = tpr * n_pos / np.maximum(
                tpr * n_pos + fpr * n_neg, 1e-10
            )
            recall = tpr
            f1 = 2 * precision * recall / np.maximum(
                precision + recall, 1e-10
            )
            best_idx = int(np.argmax(f1))
            threshold = float(thresholds[best_idx])

        elif method == "fpr_target":
            # Find threshold for target FPR (derived from threshold_percentile)
            target_fpr = 1.0 - self.threshold_percentile / 100.0
            valid = np.where(fpr <= target_fpr)[0]
            if len(valid) == 0:
                threshold = float(thresholds[0])
            else:
                best_idx = int(valid[-1])
                threshold = float(thresholds[best_idx])

        else:
            raise ValueError(
                f"Unknown threshold method: {method!r}. "
                f"Use 'youden', 'f1', or 'fpr_target'."
            )

        # sklearn's roc_curve assumes >= for classification, but our score()
        # method uses > for thresholding. Nudge the threshold down by the
        # smallest representable amount so that samples at the exact boundary
        # are correctly classified as anomalous.
        return float(np.nextafter(threshold, -np.inf))

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

        All detectors are normalized against BENIGN training scores, ensuring
        consistent interpretation regardless of detector type.

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
                "detector_type": self.detector_types_.get(
                    name, DetectorType.UNSUPERVISED
                ).value,
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

        # Map type names to classes — lazy imports for supervised detectors
        # to avoid pulling in PyTorch/sklearn unless needed.
        _type_map: dict[str, type[AnomalyDetector]] = {
            "PCAMahalanobisDetector": PCAMahalanobisDetector,
            "IsolationForestDetector": IsolationForestDetector,
            "AutoencoderDetector": AutoencoderDetector,
        }

        # Check if we need supervised detector classes
        supervised_types = {
            d["type"]
            for d in config["detectors"]
            if d["type"]
            not in ("PCAMahalanobisDetector", "IsolationForestDetector", "AutoencoderDetector")
        }
        if supervised_types:
            from venator.detection.contrastive import (
                ContrastiveDirectionDetector,
                ContrastiveMahalanobisDetector,
            )
            from venator.detection.linear_probe import (
                LinearProbeDetector,
                MLPProbeDetector,
            )

            _type_map.update({
                "LinearProbeDetector": LinearProbeDetector,
                "MLPProbeDetector": MLPProbeDetector,
                "ContrastiveDirectionDetector": ContrastiveDirectionDetector,
                "ContrastiveMahalanobisDetector": ContrastiveMahalanobisDetector,
            })

        for det_info in config["detectors"]:
            name = det_info["name"]
            weight = det_info["weight"]
            det_type = det_info["type"]

            det_cls = _type_map.get(det_type)
            if det_cls is None:
                raise ValueError(f"Unknown detector type: {det_type}")

            detector = det_cls.load(path / name)
            ensemble.detectors.append((name, detector, weight))

            # Restore detector type (default to UNSUPERVISED for legacy saves)
            dtype_str = det_info.get("detector_type", "unsupervised")
            ensemble.detector_types_[name] = DetectorType(dtype_str)

        # Load training scores
        train_scores_path = path / "train_scores.npz"
        if train_scores_path.exists():
            data = np.load(train_scores_path)
            ensemble.train_scores_ = {name: data[name] for name in data.files}

        logger.info("Loaded DetectorEnsemble from %s", path)
        return ensemble


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_default_ensemble(
    threshold_percentile: float = 95.0,
) -> DetectorEnsemble:
    """Create the recommended unsupervised ensemble configuration.

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
    ensemble.add_detector(
        "pca_mahalanobis",
        PCAMahalanobisDetector(),
        weight=2.0,
        detector_type=DetectorType.UNSUPERVISED,
    )
    ensemble.add_detector(
        "isolation_forest",
        IsolationForestDetector(),
        weight=1.5,
        detector_type=DetectorType.UNSUPERVISED,
    )
    ensemble.add_detector(
        "autoencoder",
        AutoencoderDetector(),
        weight=1.0,
        detector_type=DetectorType.UNSUPERVISED,
    )
    return ensemble


def create_unsupervised_ensemble(
    threshold_percentile: float = 95.0,
) -> DetectorEnsemble:
    """Create an unsupervised-only ensemble (for baseline comparison).

    Same as create_default_ensemble — provided for API symmetry with
    create_supervised_ensemble and create_hybrid_ensemble.
    """
    return create_default_ensemble(threshold_percentile=threshold_percentile)


def create_supervised_ensemble(
    threshold_percentile: float = 95.0,
) -> DetectorEnsemble:
    """Create a supervised ensemble using labeled jailbreak data.

    Uses detectors from Anthropic's "Cheap Monitors" paper and the
    ELK paper's diff-in-means approach. Requires labeled jailbreak data
    for training.

    Args:
        threshold_percentile: Fallback percentile if no labeled val data.

    Returns:
        An unfitted DetectorEnsemble with supervised detectors.
    """
    from venator.detection.contrastive import (
        ContrastiveDirectionDetector,
        ContrastiveMahalanobisDetector,
    )
    from venator.detection.linear_probe import LinearProbeDetector

    ensemble = DetectorEnsemble(threshold_percentile=threshold_percentile)
    ensemble.add_detector(
        "linear_probe",
        LinearProbeDetector(),
        weight=2.5,
        detector_type=DetectorType.SUPERVISED,
    )
    ensemble.add_detector(
        "contrastive_direction",
        ContrastiveDirectionDetector(),
        weight=2.0,
        detector_type=DetectorType.SUPERVISED,
    )
    ensemble.add_detector(
        "contrastive_mahalanobis",
        ContrastiveMahalanobisDetector(),
        weight=1.5,
        detector_type=DetectorType.SUPERVISED,
    )
    return ensemble


def create_hybrid_ensemble(
    threshold_percentile: float = 95.0,
) -> DetectorEnsemble:
    """Create a hybrid ensemble: supervised detectors + unsupervised autoencoder.

    Best of both: supervised detectors (for accuracy on known attack patterns)
    plus the unsupervised autoencoder (for catching novel attacks not in
    the training jailbreaks). Requires labeled jailbreak data for the
    supervised detectors.

    Args:
        threshold_percentile: Fallback percentile if no labeled val data.

    Returns:
        An unfitted DetectorEnsemble with mixed detector types.
    """
    from venator.detection.contrastive import ContrastiveDirectionDetector
    from venator.detection.linear_probe import LinearProbeDetector

    ensemble = DetectorEnsemble(threshold_percentile=threshold_percentile)
    ensemble.add_detector(
        "linear_probe",
        LinearProbeDetector(),
        weight=2.5,
        detector_type=DetectorType.SUPERVISED,
    )
    ensemble.add_detector(
        "contrastive_direction",
        ContrastiveDirectionDetector(),
        weight=2.0,
        detector_type=DetectorType.SUPERVISED,
    )
    ensemble.add_detector(
        "autoencoder",
        AutoencoderDetector(),
        weight=1.0,
        detector_type=DetectorType.UNSUPERVISED,
    )
    return ensemble
