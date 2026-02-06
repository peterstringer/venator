"""Tests for the detector ensemble and evaluation metrics.

Tests the DetectorEnsemble (normalization, weighted combination, threshold
calibration, save/load) and the metrics module (AUROC, AUPRC, precision,
recall, threshold curves) on synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorResult,
    EnsembleResult,
    create_default_ensemble,
)
from venator.detection.metrics import compute_threshold_curves, evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ensemble_data(
    n_train: int = 200,
    n_val: int = 50,
    n_test_normal: int = 50,
    n_test_outlier: int = 50,
    n_features: int = 20,
    outlier_shift: float = 8.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val/test data for ensemble testing.

    Returns: (X_train, X_val, X_test_normal, X_test_outlier)
    """
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    X_val = rng.standard_normal((n_val, n_features)).astype(np.float32)
    X_test_normal = rng.standard_normal((n_test_normal, n_features)).astype(np.float32)
    X_test_outlier = (
        rng.standard_normal((n_test_outlier, n_features)).astype(np.float32)
        + outlier_shift
    )
    return X_train, X_val, X_test_normal, X_test_outlier


def _make_simple_ensemble(n_components: int = 10) -> DetectorEnsemble:
    """Create a lightweight ensemble with a single Mahalanobis detector for fast tests."""
    ensemble = DetectorEnsemble(threshold_percentile=95.0)
    ensemble.add_detector(
        "mahal", PCAMahalanobisDetector(n_components=n_components), weight=1.0
    )
    return ensemble


def _make_dual_ensemble(n_components: int = 10) -> DetectorEnsemble:
    """Create an ensemble with two Mahalanobis detectors at different weights."""
    ensemble = DetectorEnsemble(threshold_percentile=95.0)
    ensemble.add_detector(
        "mahal_a",
        PCAMahalanobisDetector(n_components=n_components, regularization=1e-6),
        weight=2.0,
    )
    ensemble.add_detector(
        "mahal_b",
        PCAMahalanobisDetector(n_components=n_components, regularization=1e-4),
        weight=1.0,
    )
    return ensemble


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ensemble_data():
    return _make_ensemble_data()


@pytest.fixture
def fitted_single_ensemble(ensemble_data):
    X_train, X_val, _, _ = ensemble_data
    ensemble = _make_simple_ensemble()
    ensemble.fit(X_train, X_val)
    return ensemble


# ---------------------------------------------------------------------------
# Ensemble: add_detector validation
# ---------------------------------------------------------------------------


class TestAddDetector:
    def test_add_one(self):
        e = DetectorEnsemble()
        e.add_detector("test", PCAMahalanobisDetector(n_components=5), weight=1.0)
        assert len(e.detectors) == 1

    def test_duplicate_name_raises(self):
        e = DetectorEnsemble()
        e.add_detector("test", PCAMahalanobisDetector(n_components=5))
        with pytest.raises(ValueError, match="already exists"):
            e.add_detector("test", PCAMahalanobisDetector(n_components=5))

    def test_zero_weight_raises(self):
        e = DetectorEnsemble()
        with pytest.raises(ValueError, match="positive"):
            e.add_detector("test", PCAMahalanobisDetector(n_components=5), weight=0.0)

    def test_negative_weight_raises(self):
        e = DetectorEnsemble()
        with pytest.raises(ValueError, match="positive"):
            e.add_detector("test", PCAMahalanobisDetector(n_components=5), weight=-1.0)


# ---------------------------------------------------------------------------
# Ensemble: threshold percentile validation
# ---------------------------------------------------------------------------


class TestThresholdPercentile:
    def test_zero_raises(self):
        with pytest.raises(ValueError, match="threshold_percentile"):
            DetectorEnsemble(threshold_percentile=0.0)

    def test_hundred_raises(self):
        with pytest.raises(ValueError, match="threshold_percentile"):
            DetectorEnsemble(threshold_percentile=100.0)

    def test_valid(self):
        e = DetectorEnsemble(threshold_percentile=90.0)
        assert e.threshold_percentile == 90.0


# ---------------------------------------------------------------------------
# Ensemble: fit and score
# ---------------------------------------------------------------------------


class TestEnsembleFitScore:
    def test_fit_returns_self(self, ensemble_data):
        X_train, X_val, _, _ = ensemble_data
        ensemble = _make_simple_ensemble()
        result = ensemble.fit(X_train, X_val)
        assert result is ensemble

    def test_fit_sets_threshold(self, fitted_single_ensemble):
        assert fitted_single_ensemble.threshold_ is not None
        assert isinstance(fitted_single_ensemble.threshold_, float)

    def test_fit_stores_train_scores(self, fitted_single_ensemble):
        assert fitted_single_ensemble.train_scores_ is not None
        assert "mahal" in fitted_single_ensemble.train_scores_
        assert len(fitted_single_ensemble.train_scores_["mahal"]) == 200

    def test_score_returns_ensemble_result(self, fitted_single_ensemble, ensemble_data):
        _, _, X_test_normal, _ = ensemble_data
        result = fitted_single_ensemble.score(X_test_normal)
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_scores.shape == (50,)
        assert result.is_anomaly.shape == (50,)
        assert len(result.detector_results) == 1

    def test_detector_result_has_all_fields(self, fitted_single_ensemble, ensemble_data):
        _, _, X_test_normal, _ = ensemble_data
        result = fitted_single_ensemble.score(X_test_normal)
        dr = result.detector_results[0]
        assert isinstance(dr, DetectorResult)
        assert dr.name == "mahal"
        assert dr.weight == 1.0
        assert dr.raw_scores.shape == (50,)
        assert dr.normalized_scores.shape == (50,)

    def test_fit_no_detectors_raises(self, ensemble_data):
        X_train, X_val, _, _ = ensemble_data
        ensemble = DetectorEnsemble()
        with pytest.raises(RuntimeError, match="No detectors"):
            ensemble.fit(X_train, X_val)

    def test_score_before_fit_raises(self, ensemble_data):
        _, _, X_test_normal, _ = ensemble_data
        ensemble = _make_simple_ensemble()
        with pytest.raises(RuntimeError, match="not been fitted"):
            ensemble.score(X_test_normal)


# ---------------------------------------------------------------------------
# Ensemble: normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalized_scores_in_0_1(self, fitted_single_ensemble, ensemble_data):
        _, _, X_test_normal, _ = ensemble_data
        result = fitted_single_ensemble.score(X_test_normal)
        normed = result.detector_results[0].normalized_scores
        # In-distribution scores should mostly be in [0, 1]
        # (a few might be slightly above 1 if they exceed all training scores)
        assert normed.min() >= 0.0

    def test_outliers_normalize_higher(self, fitted_single_ensemble, ensemble_data):
        _, _, X_test_normal, X_test_outlier = ensemble_data
        normal_result = fitted_single_ensemble.score(X_test_normal)
        outlier_result = fitted_single_ensemble.score(X_test_outlier)

        normal_normed = normal_result.detector_results[0].normalized_scores
        outlier_normed = outlier_result.detector_results[0].normalized_scores

        assert outlier_normed.mean() > normal_normed.mean()

    def test_training_data_normalizes_to_uniform(self, fitted_single_ensemble, ensemble_data):
        """Training data percentile ranks should be roughly uniform on [0, 1]."""
        X_train, _, _, _ = ensemble_data
        result = fitted_single_ensemble.score(X_train)
        normed = result.detector_results[0].normalized_scores

        # Mean of uniform [0,1] is 0.5 — training should be near this
        assert 0.3 < normed.mean() < 0.7


# ---------------------------------------------------------------------------
# Ensemble: weighted combination
# ---------------------------------------------------------------------------


class TestWeightedCombination:
    def test_dual_detector_weights(self, ensemble_data):
        X_train, X_val, X_test_normal, _ = ensemble_data
        ensemble = _make_dual_ensemble()
        ensemble.fit(X_train, X_val)

        result = ensemble.score(X_test_normal)
        assert len(result.detector_results) == 2
        assert result.detector_results[0].weight == 2.0
        assert result.detector_results[1].weight == 1.0

    def test_ensemble_scores_finite(self, ensemble_data):
        X_train, X_val, X_test_normal, X_test_outlier = ensemble_data
        ensemble = _make_dual_ensemble()
        ensemble.fit(X_train, X_val)

        for X in (X_test_normal, X_test_outlier):
            result = ensemble.score(X)
            assert not np.any(np.isnan(result.ensemble_scores))
            assert not np.any(np.isinf(result.ensemble_scores))

    def test_ensemble_separates_normal_from_outlier(self, ensemble_data):
        X_train, X_val, X_test_normal, X_test_outlier = ensemble_data
        ensemble = _make_dual_ensemble()
        ensemble.fit(X_train, X_val)

        normal_result = ensemble.score(X_test_normal)
        outlier_result = ensemble.score(X_test_outlier)

        assert outlier_result.ensemble_scores.mean() > normal_result.ensemble_scores.mean()


# ---------------------------------------------------------------------------
# Ensemble: threshold behavior
# ---------------------------------------------------------------------------


class TestThresholdBehavior:
    def test_most_normal_not_flagged(self, ensemble_data):
        """At 95th percentile threshold, most normal test data shouldn't be flagged."""
        X_train, X_val, X_test_normal, _ = ensemble_data
        ensemble = _make_simple_ensemble()
        ensemble.fit(X_train, X_val)

        result = ensemble.score(X_test_normal)
        flagged_rate = result.is_anomaly.mean()
        # Should be low — not necessarily exactly 5%, but well under 50%
        assert flagged_rate < 0.30

    def test_most_outliers_flagged(self, ensemble_data):
        """Outliers shifted by 8 sigma should mostly be flagged."""
        X_train, X_val, _, X_test_outlier = ensemble_data
        ensemble = _make_simple_ensemble()
        ensemble.fit(X_train, X_val)

        result = ensemble.score(X_test_outlier)
        flagged_rate = result.is_anomaly.mean()
        assert flagged_rate > 0.80

    def test_lower_percentile_flags_more(self, ensemble_data):
        """Lower threshold percentile → more samples flagged."""
        X_train, X_val, X_test_normal, _ = ensemble_data

        e_strict = DetectorEnsemble(threshold_percentile=99.0)
        e_strict.add_detector("m", PCAMahalanobisDetector(n_components=10))
        e_strict.fit(X_train, X_val)

        e_loose = DetectorEnsemble(threshold_percentile=50.0)
        e_loose.add_detector("m", PCAMahalanobisDetector(n_components=10))
        e_loose.fit(X_train, X_val)

        strict_flags = e_strict.score(X_test_normal).is_anomaly.sum()
        loose_flags = e_loose.score(X_test_normal).is_anomaly.sum()

        assert loose_flags >= strict_flags


# ---------------------------------------------------------------------------
# Ensemble: save / load
# ---------------------------------------------------------------------------


class TestEnsemblePersistence:
    def test_save_load_roundtrip(self, tmp_path, ensemble_data):
        X_train, X_val, X_test_normal, X_test_outlier = ensemble_data
        ensemble = _make_dual_ensemble()
        ensemble.fit(X_train, X_val)

        original = ensemble.score(X_test_normal)

        ensemble.save(tmp_path / "ensemble")
        loaded = DetectorEnsemble.load(tmp_path / "ensemble")

        loaded_result = loaded.score(X_test_normal)

        np.testing.assert_array_almost_equal(
            loaded_result.ensemble_scores, original.ensemble_scores, decimal=5
        )

    def test_loaded_threshold_matches(self, tmp_path, ensemble_data):
        X_train, X_val, _, _ = ensemble_data
        ensemble = _make_simple_ensemble()
        ensemble.fit(X_train, X_val)

        ensemble.save(tmp_path / "ens")
        loaded = DetectorEnsemble.load(tmp_path / "ens")

        assert loaded.threshold_ == pytest.approx(ensemble.threshold_)
        assert loaded.threshold_percentile == ensemble.threshold_percentile

    def test_loaded_config_matches(self, tmp_path, ensemble_data):
        X_train, X_val, _, _ = ensemble_data
        ensemble = _make_dual_ensemble()
        ensemble.fit(X_train, X_val)

        ensemble.save(tmp_path / "ens")
        loaded = DetectorEnsemble.load(tmp_path / "ens")

        assert len(loaded.detectors) == 2
        names_weights = [(n, w) for n, _, w in loaded.detectors]
        assert ("mahal_a", 2.0) in names_weights
        assert ("mahal_b", 1.0) in names_weights

    def test_loaded_still_separates(self, tmp_path, ensemble_data):
        X_train, X_val, X_test_normal, X_test_outlier = ensemble_data
        ensemble = _make_simple_ensemble()
        ensemble.fit(X_train, X_val)

        ensemble.save(tmp_path / "ens")
        loaded = DetectorEnsemble.load(tmp_path / "ens")

        normal_result = loaded.score(X_test_normal)
        outlier_result = loaded.score(X_test_outlier)

        assert outlier_result.ensemble_scores.mean() > normal_result.ensemble_scores.mean()

    def test_save_creates_dirs(self, tmp_path, ensemble_data):
        X_train, X_val, _, _ = ensemble_data
        ensemble = _make_simple_ensemble()
        ensemble.fit(X_train, X_val)

        deep = tmp_path / "a" / "b"
        ensemble.save(deep)
        assert (deep / "ensemble_config.json").exists()
        assert (deep / "train_scores.npz").exists()


# ---------------------------------------------------------------------------
# create_default_ensemble
# ---------------------------------------------------------------------------


class TestCreateDefault:
    def test_has_three_detectors(self):
        e = create_default_ensemble()
        assert len(e.detectors) == 3

    def test_correct_names_and_weights(self):
        e = create_default_ensemble()
        names_weights = {n: w for n, _, w in e.detectors}
        assert names_weights["pca_mahalanobis"] == 2.0
        assert names_weights["isolation_forest"] == 1.5
        assert names_weights["autoencoder"] == 1.0

    def test_custom_percentile(self):
        e = create_default_ensemble(threshold_percentile=99.0)
        assert e.threshold_percentile == 99.0


# ===========================================================================
# Metrics: evaluate_detector
# ===========================================================================


class TestEvaluateDetectorPerfect:
    """Perfect separation should yield AUROC = 1.0."""

    def test_perfect_auroc(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = evaluate_detector(scores, labels)
        assert result["auroc"] == pytest.approx(1.0)

    def test_perfect_auprc(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = evaluate_detector(scores, labels)
        assert result["auprc"] == pytest.approx(1.0)

    def test_perfect_fpr_at_95_tpr(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = evaluate_detector(scores, labels)
        assert result["fpr_at_95_tpr"] == pytest.approx(0.0, abs=1e-6)


class TestEvaluateDetectorRandom:
    """Random scores should yield AUROC ~ 0.5."""

    def test_random_auroc_near_half(self):
        rng = np.random.default_rng(SEED)
        scores = rng.uniform(0, 1, 1000)
        labels = np.array([0] * 500 + [1] * 500)

        result = evaluate_detector(scores, labels)
        # With 1000 samples, random AUROC should be within ~0.1 of 0.5
        assert 0.4 < result["auroc"] < 0.6


class TestEvaluateDetectorThreshold:
    """Threshold-dependent metrics."""

    def test_threshold_metrics_present(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = evaluate_detector(scores, labels, threshold=0.5)

        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "accuracy" in result
        assert "true_positive_rate" in result
        assert "false_positive_rate" in result

    def test_perfect_threshold(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = evaluate_detector(scores, labels, threshold=0.5)

        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["true_positive_rate"] == pytest.approx(1.0)
        assert result["false_positive_rate"] == pytest.approx(0.0)

    def test_threshold_too_high(self):
        """Threshold above all scores → no predictions → recall = 0."""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = evaluate_detector(scores, labels, threshold=1.0)

        assert result["recall"] == pytest.approx(0.0)
        assert result["false_positive_rate"] == pytest.approx(0.0)

    def test_threshold_too_low(self):
        """Threshold below all scores → everything flagged → recall = 1."""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = evaluate_detector(scores, labels, threshold=0.0)

        assert result["recall"] == pytest.approx(1.0)
        assert result["true_positive_rate"] == pytest.approx(1.0)

    def test_no_threshold_excludes_binary_metrics(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        result = evaluate_detector(scores, labels)

        assert "precision" not in result
        assert "recall" not in result


class TestEvaluateDetectorValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_detector(np.array([1.0, 2.0]), np.array([0]))

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="both classes"):
            evaluate_detector(np.array([1.0, 2.0]), np.array([0, 0]))

    def test_only_positives_raises(self):
        with pytest.raises(ValueError, match="both classes"):
            evaluate_detector(np.array([1.0, 2.0]), np.array([1, 1]))


# ===========================================================================
# Metrics: compute_threshold_curves
# ===========================================================================


class TestThresholdCurves:
    def test_returns_expected_keys(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        curves = compute_threshold_curves(scores, labels, n_thresholds=10)

        expected_keys = {"thresholds", "precision", "recall", "fpr", "f1", "roc_fpr", "roc_tpr"}
        assert set(curves.keys()) == expected_keys

    def test_threshold_count(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        curves = compute_threshold_curves(scores, labels, n_thresholds=50)

        assert len(curves["thresholds"]) == 50
        assert len(curves["precision"]) == 50
        assert len(curves["recall"]) == 50
        assert len(curves["fpr"]) == 50
        assert len(curves["f1"]) == 50

    def test_roc_curve_starts_at_origin(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        curves = compute_threshold_curves(scores, labels)

        assert curves["roc_fpr"][0] == pytest.approx(0.0)
        assert curves["roc_tpr"][0] == pytest.approx(0.0)

    def test_roc_curve_ends_at_1_1(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        curves = compute_threshold_curves(scores, labels)

        assert curves["roc_fpr"][-1] == pytest.approx(1.0)
        assert curves["roc_tpr"][-1] == pytest.approx(1.0)

    def test_recall_decreases_as_threshold_increases(self):
        """Higher threshold → fewer positives → recall decreases (or stays)."""
        rng = np.random.default_rng(SEED)
        n = 200
        scores = np.concatenate([rng.uniform(0, 0.5, n), rng.uniform(0.5, 1.0, n)])
        labels = np.array([0] * n + [1] * n)

        curves = compute_threshold_curves(scores, labels, n_thresholds=100)

        # Recall should be monotonically non-increasing
        recall = curves["recall"]
        diffs = np.diff(recall)
        assert np.all(diffs <= 1e-10), "Recall should not increase as threshold increases"

    def test_values_in_valid_range(self):
        rng = np.random.default_rng(SEED)
        scores = rng.uniform(0, 1, 100)
        labels = np.array([0] * 50 + [1] * 50)
        curves = compute_threshold_curves(scores, labels)

        assert np.all(curves["precision"] >= 0) and np.all(curves["precision"] <= 1)
        assert np.all(curves["recall"] >= 0) and np.all(curves["recall"] <= 1)
        assert np.all(curves["fpr"] >= 0) and np.all(curves["fpr"] <= 1)
        assert np.all(curves["f1"] >= 0) and np.all(curves["f1"] <= 1)


# ===========================================================================
# Integration: ensemble → metrics
# ===========================================================================


class TestEnsembleMetricsIntegration:
    """End-to-end: fit ensemble, score test data, compute metrics."""

    def test_ensemble_auroc_high_on_synthetic(self, ensemble_data):
        X_train, X_val, X_test_normal, X_test_outlier = ensemble_data

        ensemble = _make_simple_ensemble()
        ensemble.fit(X_train, X_val)

        # Score combined test set
        X_test = np.vstack([X_test_normal, X_test_outlier])
        labels = np.array([0] * len(X_test_normal) + [1] * len(X_test_outlier))

        result = ensemble.score(X_test)
        metrics = evaluate_detector(result.ensemble_scores, labels, threshold=result.threshold)

        # With 8-sigma shift, AUROC should be very high
        assert metrics["auroc"] > 0.95

    def test_ensemble_threshold_from_val_works_on_test(self, ensemble_data):
        X_train, X_val, X_test_normal, X_test_outlier = ensemble_data

        ensemble = _make_simple_ensemble()
        ensemble.fit(X_train, X_val)

        X_test = np.vstack([X_test_normal, X_test_outlier])
        labels = np.array([0] * len(X_test_normal) + [1] * len(X_test_outlier))

        result = ensemble.score(X_test)
        metrics = evaluate_detector(result.ensemble_scores, labels, threshold=result.threshold)

        # With a good threshold, recall should be high
        assert metrics["recall"] > 0.80
