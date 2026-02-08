"""End-to-end Venator pipeline: extract activations, reduce, detect, evaluate.

Orchestrates the full workflow from raw prompts to detection results.
Called by both CLI scripts and the Streamlit dashboard.

Usage::

    # Full pipeline (with MLX model)
    pipeline = VenatorPipeline.from_config()
    store = pipeline.extract_and_store(prompts, "data/activations/all.h5")
    splits = SplitManager().create_splits(store, mode=SplitMode.UNSUPERVISED)
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

from venator.activation.storage import ActivationStore
from venator.config import VenatorConfig, config as default_config
from venator.data.splits import DataSplit
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.ensemble import (
    DetectorEnsemble,
    EnsembleResult,
    create_default_ensemble,
)
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.metrics import evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

if TYPE_CHECKING:
    from venator.activation.extractor import ActivationExtractor

logger = logging.getLogger(__name__)


class VenatorPipeline:
    """End-to-end jailbreak detection pipeline.

    Orchestrates: extract activations -> store -> train detectors -> evaluate -> detect.

    The ``extractor`` is optional — methods that don't require the MLX model
    (``train``, ``evaluate``, ``save``) work without one. Methods that do require
    it (``extract_and_store``, ``detect``) raise ``RuntimeError`` if no extractor
    is available.

    Attributes:
        extractor: Activation extractor (MLX-based), or None for training-only use.
        ensemble: Detector ensemble (fitted or unfitted).
        layer: Primary transformer layer index used for detection.
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
            splits: Split dict from SplitManager. Either unsupervised
                (``"train"``/``"val"``) or semi-supervised
                (``"train_benign"``/``"val_benign"``/``"train_jailbreak"``/
                ``"val_jailbreak"``).
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
        if "train" in splits:
            # Unsupervised splits
            X_train = store.get_activations(
                self.layer, indices=splits["train"].indices.tolist()
            )
            X_val = store.get_activations(
                self.layer, indices=splits["val"].indices.tolist()
            )
            X_train_jailbreak = None
            X_val_jailbreak = None

            if ensemble_type in ("supervised", "hybrid"):
                raise ValueError(
                    f"ensemble_type={ensemble_type!r} requires semi-supervised "
                    "splits (keys: 'train_benign', 'train_jailbreak', ...). "
                    "Got unsupervised splits. Use SplitMode.SEMI_SUPERVISED "
                    "when creating splits."
                )

        elif "train_benign" in splits:
            # Semi-supervised splits
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
        else:
            raise ValueError(
                f"Unrecognized split keys: {set(splits.keys())}. "
                "Expected 'train'/'val' (unsupervised) or "
                "'train_benign'/'val_benign' (semi-supervised)."
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

        # Compute validation false positive rate (benign val only)
        val_result = self.ensemble.score(X_val)
        val_fpr = float(np.mean(val_result.is_anomaly))

        logger.info("Validation FPR: %.4f (threshold=%.4f)", val_fpr, self.ensemble.threshold_)
        return {"val_false_positive_rate": val_fpr}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        store: ActivationStore,
        splits: dict[str, DataSplit],
    ) -> dict[str, float]:
        """Evaluate the trained ensemble on the test set (benign + jailbreak).

        Returns ensemble-level metrics plus a full per-detector breakdown
        (AUROC, AUPRC, F1@threshold) for comparing individual detectors
        and identifying which contribute most to ensemble performance.

        Args:
            store: ActivationStore with extracted activations.
            splits: Must contain ``"test_benign"`` and ``"test_jailbreak"`` keys.

        Returns:
            Dict with ensemble AUROC, AUPRC, precision, recall, F1,
            per-detector metrics (``auroc_<name>``, ``auprc_<name>``,
            ``f1_<name>``), and other metrics from ``evaluate_detector``.
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

        result = self.ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=self.ensemble.threshold_
        )

        # Per-detector metrics: AUROC, AUPRC, and F1@threshold
        for det_result in result.detector_results:
            det_metrics = evaluate_detector(
                det_result.normalized_scores, labels,
                threshold=self.ensemble.threshold_,
            )
            metrics[f"auroc_{det_result.name}"] = det_metrics["auroc"]
            metrics[f"auprc_{det_result.name}"] = det_metrics["auprc"]
            if "f1" in det_metrics:
                metrics[f"f1_{det_result.name}"] = det_metrics["f1"]

        logger.info(
            "Evaluation: AUROC=%.4f, AUPRC=%.4f, FPR@95TPR=%.4f",
            metrics["auroc"],
            metrics["auprc"],
            metrics["fpr_at_95_tpr"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, prompt: str) -> dict[str, Any]:
        """Score a single prompt for jailbreak detection.

        Args:
            prompt: The text to analyze.

        Returns:
            Dict with keys: ``prompt``, ``ensemble_score``, ``is_anomaly``,
            ``threshold``, ``detector_scores``.

        Raises:
            RuntimeError: If no extractor is available, or ensemble is not fitted.
        """
        if self.extractor is None:
            raise RuntimeError(
                "No extractor available. Use from_config() or pass an extractor "
                "to the constructor."
            )

        activations = self.extractor.extract_single(prompt)
        X = activations[self.layer].reshape(1, -1)
        result = self.ensemble.score(X)

        return {
            "prompt": prompt,
            "ensemble_score": float(result.ensemble_scores[0]),
            "is_anomaly": bool(result.is_anomaly[0]),
            "threshold": result.threshold,
            "detector_scores": {
                det.name: float(det.normalized_scores[0])
                for det in result.detector_results
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save the trained ensemble and pipeline metadata.

        Writes the ensemble files and a ``pipeline_meta.json`` containing the
        layer index and model information needed to reconstruct the pipeline.

        Args:
            path: Directory to save into.
        """
        path = Path(path)
        self.ensemble.save(path)

        # Save pipeline-specific metadata
        meta: dict[str, Any] = {
            "layer": self.layer,
        }
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

        Restores the ensemble and creates an extractor from the saved model_id
        (or from the provided config). The MLX model is NOT loaded until the
        first call to ``detect()`` or ``extract_and_store()``.

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

        logger.info("Loaded VenatorPipeline from %s (layer=%d)", path, layer)
        return cls(extractor=extractor, ensemble=ensemble, layer=layer)
