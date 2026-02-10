"""End-to-end Venator pipeline: extract activations, reduce, detect, evaluate.

Orchestrates the full workflow from raw prompts to detection results.
Called by both CLI scripts and the Streamlit dashboard.

The pipeline uses a **primary detector** pattern: after training, the single
best detector (linear probe for semi-supervised, PCA+Mahalanobis for
unsupervised) is used for all detection. The ensemble is retained only as
a container for training all detectors and for comparison evaluation.

Usage::

    # Full pipeline (with MLX model)
    pipeline = VenatorPipeline.from_config()
    store = pipeline.extract_and_store(prompts, "data/activations/all.h5")
    splits = SplitManager().create_splits(store)
    pipeline.train(store, splits)
    metrics = pipeline.evaluate(store, splits)
    result = pipeline.detect("Is this a jailbreak?")

    # Training-only (no model needed)
    pipeline = VenatorPipeline(ensemble=create_default_ensemble(), layer=16)
    pipeline.train(store, splits)
    pipeline.save("models/v1/")

    # Inference from saved model
    pipeline = VenatorPipeline.load("models/v1/")
    result = pipeline.detect("test prompt")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from sklearn.metrics import roc_curve  # type: ignore[import-untyped]

from venator.activation.storage import ActivationStore
from venator.config import VenatorConfig, config as default_config
from venator.data.splits import DataSplit
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.base import AnomalyDetector
from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorType,
    EnsembleResult,
    create_default_ensemble,
)
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.metrics import evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

if TYPE_CHECKING:
    from venator.activation.extractor import ActivationExtractor

logger = logging.getLogger(__name__)

# Priority order for selecting primary detector.
# First match found in the ensemble becomes the primary.
_PRIMARY_DETECTOR_PRIORITY = [
    "linear_probe",
    "pca_mahalanobis",
    "contrastive_mahalanobis",
    "isolation_forest",
    "autoencoder",
    "contrastive_direction",
]


class VenatorPipeline:
    """End-to-end jailbreak detection pipeline.

    Orchestrates: extract activations -> store -> train detectors -> evaluate -> detect.

    After training, a single **primary detector** is selected for production
    detection (``detect()``). The full ensemble of detectors is still available
    for research comparison via ``evaluate()``.

    The ``extractor`` is optional — methods that don't require the MLX model
    (``train``, ``evaluate``, ``save``) work without one. Methods that do require
    it (``extract_and_store``, ``detect``) raise ``RuntimeError`` if no extractor
    is available.

    Attributes:
        extractor: Activation extractor (MLX-based), or None for training-only use.
        ensemble: Detector ensemble (fitted or unfitted).
        layer: Primary transformer layer index used for detection.
        primary_name: Name of the primary detector (set after train or load).
        primary_threshold: Calibrated threshold for the primary detector.
    """

    def __init__(
        self,
        ensemble: DetectorEnsemble,
        layer: int = 16,
        extractor: ActivationExtractor | None = None,
    ) -> None:
        self.extractor = extractor
        self.ensemble = ensemble
        self.layer = layer
        self.primary_name: str | None = None
        self.primary_threshold: float | None = None
        self._primary_detector: AnomalyDetector | None = None

    @classmethod
    def from_config(cls, config: VenatorConfig | None = None) -> VenatorPipeline:
        """Create a pipeline from configuration.

        Builds an ``ActivationExtractor`` and a ``DetectorEnsemble`` using the
        config's model_id, extraction_layers, pca_dims, and ensemble weights.

        Args:
            config: Configuration to use. Defaults to the global singleton.

        Returns:
            An unfitted VenatorPipeline ready for training.
        """
        from venator.activation.extractor import ActivationExtractor

        config = config or default_config

        extractor = ActivationExtractor(
            model_id=config.model_id,
            layers=config.extraction_layers,
            config=config,
        )

        # Build ensemble with config weights and PCA dims
        ensemble = DetectorEnsemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        weights = config.ensemble_weights
        ensemble.add_detector(
            "pca_mahalanobis",
            PCAMahalanobisDetector(n_components=config.pca_dims),
            weight=weights["pca_mahalanobis"],
        )
        ensemble.add_detector(
            "isolation_forest",
            IsolationForestDetector(n_components=config.pca_dims),
            weight=weights["isolation_forest"],
        )
        ensemble.add_detector(
            "autoencoder",
            AutoencoderDetector(n_components=config.pca_dims),
            weight=weights["autoencoder"],
        )

        # Pick the middle extraction layer as default
        layer = config.extraction_layers[len(config.extraction_layers) // 2]

        return cls(extractor=extractor, ensemble=ensemble, layer=layer)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_and_store(
        self,
        prompts: list[str],
        output_path: Path | str,
        labels: list[int] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> ActivationStore:
        """Extract activations for all prompts and save to HDF5.

        Processes prompts one at a time to keep memory usage bounded.

        Args:
            prompts: Prompt texts to extract activations for.
            output_path: Path for the output HDF5 file.
            labels: Per-prompt labels (0=benign, 1=jailbreak). Defaults to all 0.
            on_progress: Optional callback ``(current, total)`` for progress tracking.

        Returns:
            The populated ActivationStore.

        Raises:
            RuntimeError: If no extractor is available.
        """
        if self.extractor is None:
            raise RuntimeError(
                "No extractor available. Use from_config() or pass an extractor "
                "to the constructor."
            )

        if labels is not None and len(labels) != len(prompts):
            raise ValueError(
                f"labels length ({len(labels)}) must match prompts length ({len(prompts)})"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get model metadata (triggers lazy model load)
        hidden_dim = self.extractor.hidden_dim
        layers = sorted(self.extractor._target_layers)

        store = ActivationStore.create(
            output_path,
            model_id=self.extractor._model_id,
            layers=layers,
            hidden_dim=hidden_dim,
        )

        total = len(prompts)
        for i, prompt in enumerate(prompts):
            activations = self.extractor.extract_single(prompt)
            label = labels[i] if labels is not None else 0
            store.append(prompt, activations, label=label)

            if on_progress is not None:
                on_progress(i + 1, total)

        logger.info(
            "Extracted and stored %d prompts to %s (layers=%s, dim=%d)",
            total,
            output_path,
            layers,
            hidden_dim,
        )
        return store

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        store: ActivationStore,
        splits: dict[str, DataSplit],
        ensemble_type: str = "auto",
    ) -> dict[str, float]:
        """Train the ensemble and calibrate the threshold.

        Automatically detects the split mode from the split keys and routes
        data to the appropriate detectors. The ``ensemble_type`` parameter
        provides explicit control over whether labeled jailbreak data is passed
        to the ensemble:

        - ``"auto"`` (default): pass jailbreak data if semi-supervised splits
          are detected (has ``"train_benign"`` key).
        - ``"unsupervised"``: ignore jailbreak splits even if present — passes
          only benign data (useful for baseline comparison).
        - ``"supervised"`` or ``"hybrid"``: require semi-supervised splits and
          pass jailbreak data to the ensemble's ``fit()`` method. The ensemble
          itself routes data based on each detector's ``DetectorType``.

        Args:
            store: ActivationStore with extracted activations.
            splits: Unified split dict from SplitManager with keys
                ``"train_benign"``/``"val_benign"``/``"train_jailbreak"``/
                ``"val_jailbreak"``/``"test_benign"``/``"test_jailbreak"``.
            ensemble_type: One of ``"auto"``, ``"unsupervised"``,
                ``"supervised"``, ``"hybrid"``. Controls data routing.

        Returns:
            Dict with ``"val_false_positive_rate"`` — the fraction of benign
            validation samples flagged as anomalous by the calibrated threshold.
        """
        valid_types = {"auto", "unsupervised", "supervised", "hybrid"}
        if ensemble_type not in valid_types:
            raise ValueError(
                f"ensemble_type must be one of {valid_types}, got {ensemble_type!r}"
            )

        # --- Extract training/validation data from splits ---
        # Unified split format: always has train_benign, val_benign, etc.
        if "train_benign" not in splits:
            raise ValueError(
                f"Unrecognized split keys: {set(splits.keys())}. "
                "Expected unified format with 'train_benign', 'val_benign', etc."
            )

        X_train = store.get_activations(
            self.layer, indices=splits["train_benign"].indices.tolist()
        )
        X_val = store.get_activations(
            self.layer, indices=splits["val_benign"].indices.tolist()
        )

        # Pass jailbreak data unless explicitly unsupervised
        if ensemble_type == "unsupervised":
            X_train_jailbreak = None
            X_val_jailbreak = None
        else:
            X_train_jailbreak = store.get_activations(
                self.layer,
                indices=splits["train_jailbreak"].indices.tolist(),
            )
            X_val_jailbreak = store.get_activations(
                self.layer,
                indices=splits["val_jailbreak"].indices.tolist(),
            )

        logger.info(
            "Training ensemble (type=%s): layer=%d, n_train=%d, n_val=%d%s",
            ensemble_type,
            self.layer,
            len(X_train),
            len(X_val),
            f", n_train_jailbreak={len(X_train_jailbreak)}, "
            f"n_val_jailbreak={len(X_val_jailbreak)}"
            if X_train_jailbreak is not None
            else "",
        )

        self.ensemble.fit(
            X_train, X_val,
            X_train_jailbreak=X_train_jailbreak,
            X_val_jailbreak=X_val_jailbreak,
        )

        # --- Select and calibrate primary detector ---
        self._select_primary_detector()
        self._calibrate_primary_threshold(
            X_val, X_val_jailbreak=X_val_jailbreak,
        )

        # Compute validation false positive rate (benign val only, primary detector)
        primary_val_scores = self._primary_detector.score(X_val)
        val_fpr = float(np.mean(primary_val_scores > self.primary_threshold))

        logger.info(
            "Primary detector: %s (threshold=%.4f, val_fpr=%.4f)",
            self.primary_name, self.primary_threshold, val_fpr,
        )
        return {"val_false_positive_rate": val_fpr}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        store: ActivationStore,
        splits: dict[str, DataSplit],
    ) -> dict[str, Any]:
        """Evaluate detectors on the test set (benign + jailbreak).

        Returns structured output with the **primary detector** metrics at
        the top level, plus a comparison of all individual detectors and the
        (now retired) ensemble.

        The return dict has:
            - Top-level metrics (``auroc``, ``auprc``, ``precision``, etc.)
              from the primary detector.
            - ``"primary"``: dict with primary detector name and metrics.
            - ``"baselines"``: list of dicts for other individual detectors.
            - ``"retired"``: list with the ensemble result (for comparison).
            - ``"per_detector"``: flat dict mapping ``auroc_<name>`` etc.
              for backwards compatibility.

        Args:
            store: ActivationStore with extracted activations.
            splits: Must contain ``"test_benign"`` and ``"test_jailbreak"`` keys.

        Returns:
            Structured evaluation dict.
        """
        X_test_benign = store.get_activations(
            self.layer, indices=splits["test_benign"].indices.tolist()
        )
        X_test_jailbreak = store.get_activations(
            self.layer, indices=splits["test_jailbreak"].indices.tolist()
        )

        X_test = np.vstack([X_test_benign, X_test_jailbreak])
        labels = np.concatenate([
            np.zeros(len(X_test_benign), dtype=np.int64),
            np.ones(len(X_test_jailbreak), dtype=np.int64),
        ])

        # Score through ensemble (runs all detectors)
        result = self.ensemble.score(X_test)

        # --- Primary detector metrics (top-level) ---
        primary_scores = self._primary_detector.score(X_test)
        primary_metrics = evaluate_detector(
            primary_scores, labels, threshold=self.primary_threshold,
        )

        # --- Per-detector metrics ---
        baselines = []
        per_detector_flat: dict[str, float] = {}
        for det_result in result.detector_results:
            det_metrics = evaluate_detector(
                det_result.normalized_scores, labels,
                threshold=self.ensemble.threshold_,
            )
            det_type = self.ensemble.detector_types_.get(
                det_result.name, DetectorType.UNSUPERVISED
            )
            type_label = "sup" if det_type == DetectorType.SUPERVISED else "unsup"

            entry = {
                "detector": det_result.name,
                "type": type_label,
                "auroc": det_metrics["auroc"],
                "auprc": det_metrics["auprc"],
                "f1": det_metrics.get("f1", 0.0),
                "fpr_at_95_tpr": det_metrics["fpr_at_95_tpr"],
            }

            # Skip the primary — it's reported separately
            if det_result.name != self.primary_name:
                baselines.append(entry)

            # Flat keys for backwards compat
            per_detector_flat[f"auroc_{det_result.name}"] = det_metrics["auroc"]
            per_detector_flat[f"auprc_{det_result.name}"] = det_metrics["auprc"]
            if "f1" in det_metrics:
                per_detector_flat[f"f1_{det_result.name}"] = det_metrics["f1"]

        # Sort baselines by AUROC descending
        baselines.sort(key=lambda d: d["auroc"], reverse=True)

        # --- Ensemble as retired ---
        ensemble_metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=self.ensemble.threshold_
        )
        retired = [{
            "detector": "ensemble",
            "auroc": ensemble_metrics["auroc"],
            "auprc": ensemble_metrics["auprc"],
            "f1": ensemble_metrics.get("f1", 0.0),
            "fpr_at_95_tpr": ensemble_metrics["fpr_at_95_tpr"],
            "note": "worse than primary" if ensemble_metrics["auroc"] < primary_metrics["auroc"] else "comparable to primary",
        }]

        # Build the structured result
        metrics: dict[str, Any] = {
            # Top-level = primary detector metrics (backwards compat)
            **{k: v for k, v in primary_metrics.items()},
            # Structured comparison
            "primary": {
                "detector": self.primary_name,
                "threshold": self.primary_threshold,
                **{k: v for k, v in primary_metrics.items()},
            },
            "baselines": baselines,
            "retired": retired,
            # Flat per-detector keys for backwards compat
            **per_detector_flat,
        }

        logger.info(
            "Evaluation (primary=%s): AUROC=%.4f, AUPRC=%.4f, FPR@95TPR=%.4f",
            self.primary_name,
            primary_metrics["auroc"],
            primary_metrics["auprc"],
            primary_metrics["fpr_at_95_tpr"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, prompt: str) -> dict[str, Any]:
        """Score a single prompt for jailbreak detection.

        Uses only the primary detector (not the full ensemble) for fast,
        accurate inference. The score is the primary detector's raw output
        (e.g., P(jailbreak) for the linear probe).

        Args:
            prompt: The text to analyze.

        Returns:
            Dict with keys: ``prompt``, ``score``, ``is_jailbreak``, ``threshold``.

        Raises:
            RuntimeError: If no extractor is available, or pipeline is not fitted.
        """
        if self.extractor is None:
            raise RuntimeError(
                "No extractor available. Use from_config() or pass an extractor "
                "to the constructor."
            )

        if self._primary_detector is None or self.primary_threshold is None:
            raise RuntimeError(
                "Pipeline has not been trained. Call train() first."
            )

        activations = self.extractor.extract_single(prompt)
        X = activations[self.layer].reshape(1, -1)
        score = float(self._primary_detector.score(X)[0])

        return {
            "prompt": prompt,
            "score": score,
            "is_jailbreak": bool(score > self.primary_threshold),
            "threshold": float(self.primary_threshold),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save the trained ensemble and pipeline metadata.

        Writes the ensemble files and a ``pipeline_meta.json`` containing the
        layer index, primary detector info, and model information.

        Args:
            path: Directory to save into.
        """
        path = Path(path)
        self.ensemble.save(path)

        # Save pipeline-specific metadata
        meta: dict[str, Any] = {
            "layer": self.layer,
        }
        if self.primary_name is not None:
            meta["primary_name"] = self.primary_name
        if self.primary_threshold is not None:
            meta["primary_threshold"] = self.primary_threshold
        if self.extractor is not None:
            meta["model_id"] = self.extractor._model_id
            meta["extraction_layers"] = sorted(self.extractor._target_layers)

        with open(path / "pipeline_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved VenatorPipeline to %s", path)

    @classmethod
    def load(
        cls, path: Path | str, config: VenatorConfig | None = None
    ) -> VenatorPipeline:
        """Load a saved pipeline from disk.

        Restores the ensemble, primary detector metadata, and creates an
        extractor from the saved model_id (or from the provided config).
        The MLX model is NOT loaded until the first call to ``detect()``
        or ``extract_and_store()``.

        Args:
            path: Directory containing the saved pipeline files.
            config: Optional config override. If None, uses defaults.

        Returns:
            A fitted VenatorPipeline ready for inference or evaluation.
        """
        path = Path(path)
        config = config or default_config

        ensemble = DetectorEnsemble.load(path)

        # Load pipeline metadata
        meta_path = path / "pipeline_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            layer = meta.get("layer", config.extraction_layers[len(config.extraction_layers) // 2])
            model_id = meta.get("model_id", config.model_id)
            extraction_layers = meta.get("extraction_layers", config.extraction_layers)
        else:
            meta = {}
            layer = config.extraction_layers[len(config.extraction_layers) // 2]
            model_id = config.model_id
            extraction_layers = config.extraction_layers

        # Create extractor (lazy — won't load the model until needed)
        from venator.activation.extractor import ActivationExtractor

        extractor = ActivationExtractor(
            model_id=model_id,
            layers=extraction_layers,
            config=config,
        )

        pipeline = cls(extractor=extractor, ensemble=ensemble, layer=layer)

        # Restore primary detector from metadata
        primary_name = meta.get("primary_name")
        primary_threshold = meta.get("primary_threshold")

        if primary_name is not None and primary_threshold is not None:
            # Locate the detector in the ensemble by name
            for name, detector, _ in ensemble.detectors:
                if name == primary_name:
                    pipeline.primary_name = primary_name
                    pipeline.primary_threshold = primary_threshold
                    pipeline._primary_detector = detector
                    break
            else:
                logger.warning(
                    "Primary detector '%s' not found in ensemble, "
                    "falling back to auto-selection",
                    primary_name,
                )
                pipeline._select_primary_detector()
                # Use ensemble threshold as fallback
                pipeline.primary_threshold = ensemble.threshold_
        else:
            # Legacy model without primary metadata — select automatically
            # and use ensemble threshold as a fallback
            pipeline._select_primary_detector()
            pipeline.primary_threshold = ensemble.threshold_

        logger.info(
            "Loaded VenatorPipeline from %s (layer=%d, primary=%s)",
            path, layer, pipeline.primary_name,
        )
        return pipeline

    # ------------------------------------------------------------------
    # Primary detector helpers
    # ------------------------------------------------------------------

    def _select_primary_detector(self) -> None:
        """Select the primary detector from the ensemble by priority.

        Sets ``primary_name`` and ``_primary_detector``. Does NOT set
        ``primary_threshold`` — call ``_calibrate_primary_threshold``
        separately.
        """
        detector_names = {name for name, _, _ in self.ensemble.detectors}

        for candidate in _PRIMARY_DETECTOR_PRIORITY:
            if candidate in detector_names:
                for name, detector, _ in self.ensemble.detectors:
                    if name == candidate:
                        self.primary_name = name
                        self._primary_detector = detector
                        return

        # Fallback: use the first detector
        if self.ensemble.detectors:
            name, detector, _ = self.ensemble.detectors[0]
            self.primary_name = name
            self._primary_detector = detector
        else:
            raise RuntimeError("Cannot select primary detector: ensemble has no detectors.")

    def _calibrate_primary_threshold(
        self,
        X_val: np.ndarray,
        X_val_jailbreak: np.ndarray | None = None,
    ) -> None:
        """Calibrate the primary detector's threshold.

        With labeled validation data: uses Youden's J on raw primary scores.
        Without labels: uses percentile on benign validation scores.

        Args:
            X_val: Benign validation activations.
            X_val_jailbreak: Jailbreak validation activations (optional).
        """
        val_scores = self._primary_detector.score(X_val)

        if X_val_jailbreak is not None and len(X_val_jailbreak) > 0:
            # Supervised calibration via Youden's J on raw primary scores
            jb_scores = self._primary_detector.score(X_val_jailbreak)
            all_scores = np.concatenate([val_scores, jb_scores])
            all_labels = np.concatenate([
                np.zeros(len(val_scores), dtype=np.int64),
                np.ones(len(jb_scores), dtype=np.int64),
            ])

            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            j_stat = tpr - fpr
            best_idx = int(np.argmax(j_stat))
            self.primary_threshold = float(np.nextafter(thresholds[best_idx], -np.inf))

            logger.info(
                "Primary threshold (Youden's J): %.4f "
                "(val_benign_mean=%.4f, val_jb_mean=%.4f)",
                self.primary_threshold,
                val_scores.mean(),
                jb_scores.mean(),
            )
        else:
            # Unsupervised: percentile on benign validation scores
            self.primary_threshold = float(
                np.percentile(val_scores, self.ensemble.threshold_percentile)
            )
            logger.info(
                "Primary threshold (%.1f%% percentile): %.4f",
                self.ensemble.threshold_percentile,
                self.primary_threshold,
            )
