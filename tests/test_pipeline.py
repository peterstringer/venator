"""Integration tests for the full Venator pipeline.

Tests VenatorPipeline with a mock extractor (no MLX dependency) and synthetic
Gaussian data. Verifies the full workflow: extract → train → evaluate → detect,
plus save/load roundtrip and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from venator.activation.storage import ActivationStore
from venator.config import VenatorConfig
from venator.data.splits import DataSplit, SplitManager, SplitMode
from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorType,
    create_default_ensemble,
    create_hybrid_ensemble,
)
from venator.pipeline import VenatorPipeline

SEED = 42
HIDDEN_DIM = 50
LAYERS = [12, 16, 20]
MODEL_ID = "test-model/mock-7b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def mock_extractor():
    """Mock ActivationExtractor that returns synthetic Gaussian activations."""
    extractor = MagicMock()
    extractor._model_id = MODEL_ID
    extractor._target_layers = set(LAYERS)
    extractor.hidden_dim = HIDDEN_DIM

    def _extract_single(prompt: str) -> dict[int, np.ndarray]:
        # Deterministic per-prompt via hash
        seed = abs(hash(prompt)) % (2**31)
        local_rng = np.random.default_rng(seed)
        return {
            layer: local_rng.standard_normal(HIDDEN_DIM).astype(np.float32)
            for layer in LAYERS
        }

    extractor.extract_single = _extract_single
    return extractor


@pytest.fixture
def populated_store(tmp_path, rng):
    """Store with 250 benign N(0,I) + 50 jailbreak N(5,I) prompts."""
    store_path = tmp_path / "test_activations.h5"
    store = ActivationStore.create(store_path, MODEL_ID, LAYERS, HIDDEN_DIM)

    n_benign = 250
    n_jailbreak = 50

    # Benign: N(0, I)
    benign_acts = {
        layer: rng.standard_normal((n_benign, HIDDEN_DIM)).astype(np.float32)
        for layer in LAYERS
    }
    store.append_batch(
        prompts=[f"benign prompt {i}" for i in range(n_benign)],
        activations=benign_acts,
        labels=[0] * n_benign,
    )

    # Jailbreak: N(5, I) — clearly anomalous
    jailbreak_acts = {
        layer: (rng.standard_normal((n_jailbreak, HIDDEN_DIM)) + 5.0).astype(np.float32)
        for layer in LAYERS
    }
    store.append_batch(
        prompts=[f"jailbreak prompt {i}" for i in range(n_jailbreak)],
        activations=jailbreak_acts,
        labels=[1] * n_jailbreak,
    )

    return store


@pytest.fixture
def splits(populated_store):
    """Create splits from the populated store."""
    manager = SplitManager(seed=SEED)
    return manager.create_splits(populated_store, mode=SplitMode.UNSUPERVISED)


@pytest.fixture
def trained_pipeline(mock_extractor, populated_store, splits):
    """A pipeline that has been trained on the populated store."""
    ensemble = create_default_ensemble(threshold_percentile=95.0)
    pipeline = VenatorPipeline(
        ensemble=ensemble, layer=16, extractor=mock_extractor
    )
    pipeline.train(populated_store, splits)
    return pipeline


# ===========================================================================
# TestFromConfig
# ===========================================================================


class TestFromConfig:
    """Test from_config — patches ActivationExtractor to avoid MLX import."""

    _PATCH_TARGET = "venator.activation.extractor.ActivationExtractor"

    def test_creates_pipeline_with_three_detectors(self):
        with patch(self._PATCH_TARGET) as MockExtractor:
            pipeline = VenatorPipeline.from_config()
            assert len(pipeline.ensemble.detectors) == 3

    def test_uses_config_weights(self):
        config = VenatorConfig(
            weight_pca_mahalanobis=3.0,
            weight_isolation_forest=2.0,
            weight_autoencoder=0.5,
        )
        with patch(self._PATCH_TARGET) as MockExtractor:
            pipeline = VenatorPipeline.from_config(config)
            weights = {name: w for name, _, w in pipeline.ensemble.detectors}
            assert weights["pca_mahalanobis"] == 3.0
            assert weights["isolation_forest"] == 2.0
            assert weights["autoencoder"] == 0.5

    def test_uses_config_pca_dims(self):
        config = VenatorConfig(pca_dims=30)
        with patch(self._PATCH_TARGET) as MockExtractor:
            pipeline = VenatorPipeline.from_config(config)
            for _, detector, _ in pipeline.ensemble.detectors:
                assert detector.n_components == 30

    def test_picks_middle_layer(self):
        config = VenatorConfig(extraction_layers=[8, 12, 16, 20, 24])
        with patch(self._PATCH_TARGET) as MockExtractor:
            pipeline = VenatorPipeline.from_config(config)
            assert pipeline.layer == 16

    def test_detector_names_correct(self):
        with patch(self._PATCH_TARGET) as MockExtractor:
            pipeline = VenatorPipeline.from_config()
            names = [name for name, _, _ in pipeline.ensemble.detectors]
            assert names == ["pca_mahalanobis", "isolation_forest", "autoencoder"]


# ===========================================================================
# TestExtractAndStore
# ===========================================================================


class TestExtractAndStore:
    def test_produces_valid_store(self, mock_extractor, tmp_path):
        pipeline = VenatorPipeline(
            ensemble=DetectorEnsemble(), layer=16, extractor=mock_extractor
        )
        prompts = ["hello world", "test prompt", "another one"]
        output = tmp_path / "extracted.h5"
        store = pipeline.extract_and_store(prompts, output)

        assert store.n_prompts == 3
        assert store.layers == sorted(LAYERS)
        assert store.hidden_dim == HIDDEN_DIM

    def test_labels_default_to_benign(self, mock_extractor, tmp_path):
        pipeline = VenatorPipeline(
            ensemble=DetectorEnsemble(), layer=16, extractor=mock_extractor
        )
        store = pipeline.extract_and_store(
            ["prompt 1", "prompt 2"], tmp_path / "out.h5"
        )
        labels = store.get_labels()
        assert list(labels) == [0, 0]

    def test_labels_preserved(self, mock_extractor, tmp_path):
        pipeline = VenatorPipeline(
            ensemble=DetectorEnsemble(), layer=16, extractor=mock_extractor
        )
        store = pipeline.extract_and_store(
            ["benign", "jailbreak", "benign"],
            tmp_path / "out.h5",
            labels=[0, 1, 0],
        )
        labels = store.get_labels()
        assert list(labels) == [0, 1, 0]

    def test_progress_callback(self, mock_extractor, tmp_path):
        pipeline = VenatorPipeline(
            ensemble=DetectorEnsemble(), layer=16, extractor=mock_extractor
        )
        calls = []
        pipeline.extract_and_store(
            ["a", "b", "c"],
            tmp_path / "out.h5",
            on_progress=lambda cur, tot: calls.append((cur, tot)),
        )
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_without_extractor_raises(self, tmp_path):
        pipeline = VenatorPipeline(ensemble=DetectorEnsemble(), layer=16)
        with pytest.raises(RuntimeError, match="No extractor"):
            pipeline.extract_and_store(["test"], tmp_path / "out.h5")

    def test_labels_length_mismatch_raises(self, mock_extractor, tmp_path):
        pipeline = VenatorPipeline(
            ensemble=DetectorEnsemble(), layer=16, extractor=mock_extractor
        )
        with pytest.raises(ValueError, match="labels length"):
            pipeline.extract_and_store(
                ["a", "b"], tmp_path / "out.h5", labels=[0]
            )


# ===========================================================================
# TestTrain
# ===========================================================================


class TestTrain:
    def test_returns_val_fpr(self, mock_extractor, populated_store, splits):
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(
            ensemble=ensemble, layer=16, extractor=mock_extractor
        )
        metrics = pipeline.train(populated_store, splits)
        assert "val_false_positive_rate" in metrics
        assert 0.0 <= metrics["val_false_positive_rate"] <= 1.0

    def test_sets_threshold(self, mock_extractor, populated_store, splits):
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(
            ensemble=ensemble, layer=16, extractor=mock_extractor
        )
        pipeline.train(populated_store, splits)
        assert pipeline.ensemble.threshold_ is not None

    def test_works_without_extractor(self, populated_store, splits):
        """Train and evaluate don't need an extractor."""
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        metrics = pipeline.train(populated_store, splits)
        assert "val_false_positive_rate" in metrics


# ===========================================================================
# TestEvaluate
# ===========================================================================


class TestEvaluate:
    def test_returns_auroc(self, trained_pipeline, populated_store, splits):
        metrics = trained_pipeline.evaluate(populated_store, splits)
        assert "auroc" in metrics
        # Well-separated synthetic data should give high AUROC
        assert metrics["auroc"] > 0.8

    def test_returns_auprc(self, trained_pipeline, populated_store, splits):
        metrics = trained_pipeline.evaluate(populated_store, splits)
        assert "auprc" in metrics
        assert 0.0 <= metrics["auprc"] <= 1.0

    def test_returns_per_detector_auroc(self, trained_pipeline, populated_store, splits):
        metrics = trained_pipeline.evaluate(populated_store, splits)
        assert "auroc_pca_mahalanobis" in metrics
        assert "auroc_isolation_forest" in metrics
        assert "auroc_autoencoder" in metrics

    def test_returns_threshold_metrics(self, trained_pipeline, populated_store, splits):
        metrics = trained_pipeline.evaluate(populated_store, splits)
        # evaluate_detector with a threshold should include these
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_works_without_extractor(self, populated_store, splits):
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        pipeline.train(populated_store, splits)
        metrics = pipeline.evaluate(populated_store, splits)
        assert "auroc" in metrics


# ===========================================================================
# TestDetect
# ===========================================================================


class TestDetect:
    def test_returns_expected_keys(self, trained_pipeline):
        result = trained_pipeline.detect("test prompt")
        expected_keys = {"prompt", "ensemble_score", "is_anomaly", "threshold", "detector_scores"}
        assert set(result.keys()) == expected_keys

    def test_is_anomaly_is_python_bool(self, trained_pipeline):
        result = trained_pipeline.detect("test prompt")
        assert isinstance(result["is_anomaly"], bool)

    def test_ensemble_score_is_python_float(self, trained_pipeline):
        result = trained_pipeline.detect("test prompt")
        assert isinstance(result["ensemble_score"], float)

    def test_prompt_echoed_back(self, trained_pipeline):
        result = trained_pipeline.detect("my test prompt")
        assert result["prompt"] == "my test prompt"

    def test_detector_scores_has_all_detectors(self, trained_pipeline):
        result = trained_pipeline.detect("test prompt")
        assert "pca_mahalanobis" in result["detector_scores"]
        assert "isolation_forest" in result["detector_scores"]
        assert "autoencoder" in result["detector_scores"]

    def test_without_extractor_raises(self, populated_store, splits):
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        pipeline.train(populated_store, splits)
        with pytest.raises(RuntimeError, match="No extractor"):
            pipeline.detect("test")


# ===========================================================================
# TestSaveLoad
# ===========================================================================


class TestSaveLoad:
    def test_roundtrip_scores_match(self, trained_pipeline, populated_store, splits, tmp_path):
        save_dir = tmp_path / "saved_model"
        trained_pipeline.save(save_dir)

        loaded = VenatorPipeline.load(save_dir)

        # Compare scores on test data
        X_test = populated_store.get_activations(
            16, indices=splits["test_benign"].indices.tolist()
        )
        original = trained_pipeline.ensemble.score(X_test).ensemble_scores
        loaded_scores = loaded.ensemble.score(X_test).ensemble_scores
        np.testing.assert_array_almost_equal(original, loaded_scores)

    def test_saves_pipeline_meta(self, trained_pipeline, tmp_path):
        import json

        save_dir = tmp_path / "saved_model"
        trained_pipeline.save(save_dir)

        meta_path = save_dir / "pipeline_meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["layer"] == 16
        assert meta["model_id"] == MODEL_ID

    def test_loaded_layer_matches(self, trained_pipeline, tmp_path):
        save_dir = tmp_path / "saved_model"
        trained_pipeline.save(save_dir)

        loaded = VenatorPipeline.load(save_dir)
        assert loaded.layer == trained_pipeline.layer

    def test_loaded_threshold_matches(self, trained_pipeline, tmp_path):
        save_dir = tmp_path / "saved_model"
        trained_pipeline.save(save_dir)

        loaded = VenatorPipeline.load(save_dir)
        assert loaded.ensemble.threshold_ == pytest.approx(
            trained_pipeline.ensemble.threshold_
        )

    def test_save_without_extractor(self, populated_store, splits, tmp_path):
        """Saving without extractor should still save layer."""
        import json

        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        pipeline.train(populated_store, splits)

        save_dir = tmp_path / "model"
        pipeline.save(save_dir)

        with open(save_dir / "pipeline_meta.json") as f:
            meta = json.load(f)
        assert meta["layer"] == 16
        # model_id should not be present without extractor
        assert "model_id" not in meta


# ===========================================================================
# TestEndToEnd
# ===========================================================================


class TestEndToEnd:
    def test_extract_train_evaluate_detect(self, mock_extractor, tmp_path):
        """Full pipeline: extract → split → train → evaluate → detect."""
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(
            ensemble=ensemble, layer=16, extractor=mock_extractor
        )

        # Extract benign prompts
        benign_prompts = [f"benign prompt {i}" for i in range(100)]
        store_path = tmp_path / "activations.h5"
        store = pipeline.extract_and_store(benign_prompts, store_path, labels=[0] * 100)

        # Also add some "jailbreak" prompts
        jailbreak_prompts = [f"jailbreak prompt {i}" for i in range(30)]
        # Can't append to existing store through pipeline, so use store directly
        # Create jailbreak activations with shift
        rng = np.random.default_rng(99)
        jailbreak_acts = {
            layer: (rng.standard_normal((30, HIDDEN_DIM)) + 5.0).astype(np.float32)
            for layer in LAYERS
        }
        store.append_batch(jailbreak_prompts, jailbreak_acts, labels=[1] * 30)

        # Create splits
        manager = SplitManager(seed=SEED)
        splits = manager.create_splits(store, mode=SplitMode.UNSUPERVISED)

        # Train
        train_metrics = pipeline.train(store, splits)
        assert "val_false_positive_rate" in train_metrics

        # Evaluate
        eval_metrics = pipeline.evaluate(store, splits)
        assert "auroc" in eval_metrics

        # Detect
        result = pipeline.detect("some prompt")
        assert "is_anomaly" in result
        assert isinstance(result["is_anomaly"], bool)


# ===========================================================================
# TestSemiSupervisedTraining
# ===========================================================================


class TestSemiSupervisedTraining:
    """Test pipeline.train() with semi-supervised splits."""

    @pytest.fixture
    def semi_splits(self, populated_store):
        """Create semi-supervised splits from the populated store."""
        manager = SplitManager(seed=SEED)
        return manager.create_splits(populated_store, mode=SplitMode.SEMI_SUPERVISED)

    def test_train_with_semi_supervised_splits(self, populated_store, semi_splits):
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        metrics = pipeline.train(populated_store, semi_splits)
        assert "val_false_positive_rate" in metrics
        assert 0.0 <= metrics["val_false_positive_rate"] <= 1.0

    def test_train_hybrid_with_semi_supervised(self, populated_store, semi_splits):
        ensemble = create_hybrid_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        metrics = pipeline.train(populated_store, semi_splits)
        assert "val_false_positive_rate" in metrics

    def test_evaluate_after_semi_supervised_train(self, populated_store, semi_splits):
        ensemble = create_default_ensemble(threshold_percentile=95.0)
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        pipeline.train(populated_store, semi_splits)
        metrics = pipeline.evaluate(populated_store, semi_splits)
        assert "auroc" in metrics
        assert metrics["auroc"] > 0.8

    def test_unrecognized_split_keys_raises(self, populated_store):
        bad_splits = {"foo": DataSplit("foo", np.array([0, 1]), 2, False)}
        ensemble = create_default_ensemble()
        pipeline = VenatorPipeline(ensemble=ensemble, layer=16)
        with pytest.raises(ValueError, match="Unrecognized split keys"):
            pipeline.train(populated_store, bad_splits)
