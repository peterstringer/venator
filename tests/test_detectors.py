"""Tests for anomaly detectors on synthetic Gaussian data.

Tests the AnomalyDetector base class contract and the PCAMahalanobisDetector
on controlled synthetic distributions where we know what "normal" and
"anomalous" look like.
"""

from __future__ import annotations

import numpy as np
import pytest

from venator.detection.base import AnomalyDetector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_data(
    n_normal: int = 300,
    n_outlier: int = 50,
    n_features: int = 20,
    outlier_shift: float = 8.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic normal + outlier data from two Gaussians.

    Normal data: N(0, I)
    Outliers: N(outlier_shift, I) — shifted far from the training distribution

    Returns:
        (X_train, X_normal_test, X_outlier) — all float32
    """
    rng = np.random.default_rng(seed)

    X_all_normal = rng.standard_normal((n_normal + n_normal // 3, n_features)).astype(
        np.float32
    )
    X_train = X_all_normal[:n_normal]
    X_normal_test = X_all_normal[n_normal:]

    X_outlier = (
        rng.standard_normal((n_outlier, n_features)).astype(np.float32) + outlier_shift
    )

    return X_train, X_normal_test, X_outlier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_data():
    """Standard 20-dim Gaussian: 300 train, 100 normal test, 50 outliers."""
    return _make_gaussian_data()


@pytest.fixture
def fitted_detector(gaussian_data):
    """A PCAMahalanobisDetector already fitted on the normal training set."""
    X_train, _, _ = gaussian_data
    detector = PCAMahalanobisDetector(n_components=10)
    detector.fit(X_train)
    return detector


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------


class TestBaseClass:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            AnomalyDetector()

    def test_subclass_must_implement_fit(self):
        class Incomplete(AnomalyDetector):
            def score(self, X):
                return np.zeros(len(X))

            def save(self, path):
                pass

            @classmethod
            def load(cls, path):
                return cls()

        with pytest.raises(TypeError):
            Incomplete()

    def test_pca_mahalanobis_is_subclass(self):
        assert issubclass(PCAMahalanobisDetector, AnomalyDetector)


# ---------------------------------------------------------------------------
# Anomaly separation — the core contract
# ---------------------------------------------------------------------------


class TestAnomalySeparation:
    """Outliers must score higher than in-distribution points."""

    def test_outliers_score_higher_than_normal(self, gaussian_data):
        X_train, X_normal_test, X_outlier = gaussian_data

        detector = PCAMahalanobisDetector(n_components=10)
        detector.fit(X_train)

        normal_scores = detector.score(X_normal_test)
        outlier_scores = detector.score(X_outlier)

        # Every outlier should score higher than the median normal score
        median_normal = np.median(normal_scores)
        assert np.all(outlier_scores > median_normal), (
            f"Some outliers scored below median normal. "
            f"Outlier min={outlier_scores.min():.2f}, "
            f"normal median={median_normal:.2f}"
        )

    def test_mean_outlier_score_much_higher(self, gaussian_data):
        X_train, X_normal_test, X_outlier = gaussian_data

        detector = PCAMahalanobisDetector(n_components=10)
        detector.fit(X_train)

        normal_scores = detector.score(X_normal_test)
        outlier_scores = detector.score(X_outlier)

        # Outlier mean should be at least 2x the normal mean
        assert outlier_scores.mean() > 2 * normal_scores.mean()

    def test_score_single_matches_batch(self, fitted_detector, gaussian_data):
        _, X_normal_test, _ = gaussian_data

        batch_scores = fitted_detector.score(X_normal_test)

        for i in range(min(5, len(X_normal_test))):
            single_score = fitted_detector.score_single(X_normal_test[i])
            np.testing.assert_almost_equal(single_score, batch_scores[i], decimal=5)

    def test_training_data_scores_low(self, fitted_detector, gaussian_data):
        """Training data itself should have relatively low scores."""
        X_train, _, X_outlier = gaussian_data

        train_scores = fitted_detector.score(X_train)
        outlier_scores = fitted_detector.score(X_outlier)

        assert train_scores.mean() < outlier_scores.mean()


# ---------------------------------------------------------------------------
# PCA reduction
# ---------------------------------------------------------------------------


class TestPCAReduction:
    def test_correct_output_components(self, fitted_detector):
        """PCA should reduce to n_components dimensions internally."""
        assert fitted_detector.pca_.n_components_ == 10

    def test_explained_variance_is_positive(self, fitted_detector):
        assert np.all(fitted_detector.pca_.explained_variance_ratio_ > 0)

    def test_explained_variance_sums_less_than_one(self, fitted_detector):
        # We're keeping 10 of 20 components — not 100% of variance
        total = fitted_detector.pca_.explained_variance_ratio_.sum()
        assert 0 < total <= 1.0

    def test_auto_reduce_components_when_few_samples(self):
        """If n_samples < n_components, PCA should auto-reduce."""
        rng = np.random.default_rng(SEED)
        X = rng.standard_normal((5, 100)).astype(np.float32)

        detector = PCAMahalanobisDetector(n_components=50)
        with pytest.warns(UserWarning, match="Reduced PCA components"):
            detector.fit(X)

        # Should reduce to min(n_samples-1, n_features) = 4
        assert detector.pca_.n_components_ == 4

    def test_auto_reduce_when_features_less_than_components(self):
        """If n_features < n_components, PCA should auto-reduce."""
        rng = np.random.default_rng(SEED)
        X = rng.standard_normal((100, 3)).astype(np.float32)

        detector = PCAMahalanobisDetector(n_components=50)
        with pytest.warns(UserWarning, match="Reduced PCA components"):
            detector.fit(X)

        assert detector.pca_.n_components_ == 3

    def test_full_components_no_warning(self, gaussian_data):
        """No warning when n_components <= min(n_samples-1, n_features)."""
        import warnings

        X_train, _, _ = gaussian_data  # 300 samples, 20 features

        detector = PCAMahalanobisDetector(n_components=10)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            detector.fit(X_train)

        our_warnings = [w for w in record if "Reduced PCA" in str(w.message)]
        assert len(our_warnings) == 0


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_no_nan_in_scores(self, fitted_detector, gaussian_data):
        _, X_normal_test, X_outlier = gaussian_data

        normal_scores = fitted_detector.score(X_normal_test)
        outlier_scores = fitted_detector.score(X_outlier)

        assert not np.any(np.isnan(normal_scores))
        assert not np.any(np.isnan(outlier_scores))

    def test_no_inf_in_scores(self, fitted_detector, gaussian_data):
        _, X_normal_test, X_outlier = gaussian_data

        normal_scores = fitted_detector.score(X_normal_test)
        outlier_scores = fitted_detector.score(X_outlier)

        assert not np.any(np.isinf(normal_scores))
        assert not np.any(np.isinf(outlier_scores))

    def test_scores_are_non_negative(self, fitted_detector, gaussian_data):
        """Mahalanobis distance is always >= 0."""
        _, X_normal_test, X_outlier = gaussian_data

        normal_scores = fitted_detector.score(X_normal_test)
        outlier_scores = fitted_detector.score(X_outlier)

        assert np.all(normal_scores >= 0)
        assert np.all(outlier_scores >= 0)

    def test_near_singular_covariance(self):
        """With near-singular covariance, regularization keeps things stable."""
        rng = np.random.default_rng(SEED)
        n_samples = 50

        # Create data with one near-constant feature (near-singular cov)
        X = rng.standard_normal((n_samples, 5)).astype(np.float32)
        X[:, 0] = 1.0 + 1e-10 * rng.standard_normal(n_samples)

        detector = PCAMahalanobisDetector(n_components=4, regularization=1e-4)
        detector.fit(X)

        scores = detector.score(X)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_high_regularization(self):
        """High regularization pushes scores toward Euclidean-like behavior."""
        rng = np.random.default_rng(SEED)
        X_train = rng.standard_normal((100, 10)).astype(np.float32)
        X_test = rng.standard_normal((20, 10)).astype(np.float32)

        # High regularization: cov ≈ reg * I, so Mahalanobis ≈ Euclidean / sqrt(reg)
        detector = PCAMahalanobisDetector(n_components=5, regularization=100.0)
        detector.fit(X_train)

        scores = detector.score(X_test)
        assert not np.any(np.isnan(scores))
        assert np.all(scores >= 0)

    def test_single_pca_component(self):
        """Edge case: reduce to just 1 PCA dimension."""
        rng = np.random.default_rng(SEED)
        X_train = rng.standard_normal((50, 10)).astype(np.float32)
        X_outlier = rng.standard_normal((10, 10)).astype(np.float32) + 5.0

        detector = PCAMahalanobisDetector(n_components=1)
        detector.fit(X_train)

        normal_scores = detector.score(X_train[:10])
        outlier_scores = detector.score(X_outlier)

        assert not np.any(np.isnan(normal_scores))
        assert not np.any(np.isnan(outlier_scores))
        # Outliers should still score higher on average
        assert outlier_scores.mean() > normal_scores.mean()


# ---------------------------------------------------------------------------
# Save / Load roundtrip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path, gaussian_data):
        X_train, X_normal_test, X_outlier = gaussian_data

        detector = PCAMahalanobisDetector(n_components=10, regularization=1e-5)
        detector.fit(X_train)

        original_normal_scores = detector.score(X_normal_test)
        original_outlier_scores = detector.score(X_outlier)

        # Save
        model_dir = tmp_path / "detector"
        detector.save(model_dir)

        # Load
        loaded = PCAMahalanobisDetector.load(model_dir)

        # Scores should be identical
        loaded_normal_scores = loaded.score(X_normal_test)
        loaded_outlier_scores = loaded.score(X_outlier)

        np.testing.assert_array_almost_equal(
            loaded_normal_scores, original_normal_scores
        )
        np.testing.assert_array_almost_equal(
            loaded_outlier_scores, original_outlier_scores
        )

    def test_save_preserves_config(self, tmp_path, gaussian_data):
        X_train, _, _ = gaussian_data

        detector = PCAMahalanobisDetector(n_components=8, regularization=1e-3)
        detector.fit(X_train)

        model_dir = tmp_path / "detector"
        detector.save(model_dir)

        loaded = PCAMahalanobisDetector.load(model_dir)
        assert loaded.n_components == 8
        assert loaded.regularization == pytest.approx(1e-3)

    def test_save_creates_parent_dirs(self, tmp_path, gaussian_data):
        X_train, _, _ = gaussian_data

        detector = PCAMahalanobisDetector(n_components=5)
        detector.fit(X_train)

        deep_path = tmp_path / "a" / "b" / "c"
        detector.save(deep_path)
        assert (deep_path / "pca.joblib").exists()
        assert (deep_path / "gaussian.npz").exists()

    def test_loaded_detector_still_separates(self, tmp_path, gaussian_data):
        """Loaded detector should still cleanly separate normal from outlier."""
        X_train, X_normal_test, X_outlier = gaussian_data

        detector = PCAMahalanobisDetector(n_components=10)
        detector.fit(X_train)
        detector.save(tmp_path / "model")

        loaded = PCAMahalanobisDetector.load(tmp_path / "model")
        normal_scores = loaded.score(X_normal_test)
        outlier_scores = loaded.score(X_outlier)

        assert outlier_scores.mean() > 2 * normal_scores.mean()


# ---------------------------------------------------------------------------
# Validation and error handling
# ---------------------------------------------------------------------------


class TestValidation:
    def test_score_before_fit_raises(self):
        detector = PCAMahalanobisDetector(n_components=10)
        X = np.random.default_rng(SEED).standard_normal((5, 10)).astype(np.float32)

        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(X)

    def test_fit_1d_array_raises(self):
        detector = PCAMahalanobisDetector(n_components=5)
        X = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            detector.fit(X)

    def test_score_1d_array_raises(self, fitted_detector):
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2D"):
            fitted_detector.score(x)

    def test_fit_single_sample_raises(self):
        detector = PCAMahalanobisDetector(n_components=5)
        X = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="at least 2"):
            detector.fit(X)

    def test_negative_components_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            PCAMahalanobisDetector(n_components=0)

    def test_negative_regularization_raises(self):
        with pytest.raises(ValueError, match="regularization"):
            PCAMahalanobisDetector(regularization=-1.0)


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------


class TestMethodChaining:
    def test_fit_returns_self(self, gaussian_data):
        X_train, _, _ = gaussian_data
        detector = PCAMahalanobisDetector(n_components=10)
        result = detector.fit(X_train)
        assert result is detector


# ---------------------------------------------------------------------------
# High-dimensional (realistic scenario)
# ---------------------------------------------------------------------------


class TestHighDimensional:
    def test_4096_dim_activations(self):
        """Simulate realistic 4096-dim LLM activations."""
        rng = np.random.default_rng(SEED)
        n_train, n_test, n_outlier = 200, 50, 30
        dim = 4096

        X_train = rng.standard_normal((n_train, dim)).astype(np.float32)
        X_test = rng.standard_normal((n_test, dim)).astype(np.float32)
        X_outlier = (
            rng.standard_normal((n_outlier, dim)).astype(np.float32) + 3.0
        )

        detector = PCAMahalanobisDetector(n_components=50)
        detector.fit(X_train)

        normal_scores = detector.score(X_test)
        outlier_scores = detector.score(X_outlier)

        assert not np.any(np.isnan(normal_scores))
        assert not np.any(np.isnan(outlier_scores))
        assert outlier_scores.mean() > normal_scores.mean()

    def test_small_shift_still_detectable(self):
        """Even a moderate shift should produce higher scores."""
        rng = np.random.default_rng(SEED)
        dim = 100

        X_train = rng.standard_normal((200, dim)).astype(np.float32)
        X_normal = rng.standard_normal((50, dim)).astype(np.float32)
        # Moderate shift: 3 sigma in each dim
        X_shifted = rng.standard_normal((50, dim)).astype(np.float32) + 3.0

        detector = PCAMahalanobisDetector(n_components=20)
        detector.fit(X_train)

        normal_scores = detector.score(X_normal)
        shifted_scores = detector.score(X_shifted)

        assert shifted_scores.mean() > normal_scores.mean()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_data_same_scores(self, gaussian_data):
        X_train, X_test, _ = gaussian_data

        d1 = PCAMahalanobisDetector(n_components=10)
        d1.fit(X_train)
        s1 = d1.score(X_test)

        d2 = PCAMahalanobisDetector(n_components=10)
        d2.fit(X_train)
        s2 = d2.score(X_test)

        np.testing.assert_array_almost_equal(s1, s2)
