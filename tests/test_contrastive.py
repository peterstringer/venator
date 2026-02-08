"""Tests for contrastive direction and class-conditional Mahalanobis detectors.

Tests verify:
- Supervised detectors refuse to train without labels
- Contrastive direction separates synthetic clusters
- Score is higher for jailbreak-like points
- Works with as few as 5 labeled jailbreaks
- Class-conditional Mahalanobis outperforms unsupervised Mahalanobis
- Save/load roundtrip
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from venator.detection.base import AnomalyDetector
from venator.detection.contrastive import (
    ContrastiveDirectionDetector,
    ContrastiveMahalanobisDetector,
)

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labeled_data(
    n_benign: int = 200,
    n_jailbreak: int = 50,
    n_features: int = 20,
    shift: float = 5.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create labeled synthetic data from two Gaussians.

    Benign: N(0, I), Jailbreak: N(shift, I).

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    rng = np.random.default_rng(seed)

    X_benign = rng.standard_normal((n_benign, n_features)).astype(np.float32)
    X_jailbreak = (
        rng.standard_normal((n_jailbreak, n_features)).astype(np.float32) + shift
    )

    X_all = np.vstack([X_benign, X_jailbreak])
    y_all = np.concatenate([
        np.zeros(n_benign, dtype=np.int64),
        np.ones(n_jailbreak, dtype=np.int64),
    ])

    # 80/20 train/test split
    indices = rng.permutation(len(X_all))
    n_train = int(len(X_all) * 0.8)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X_all[train_idx], y_all[train_idx], X_all[test_idx], y_all[test_idx]


def _make_few_shot_data(
    n_benign_train: int = 50,
    n_jailbreak_train: int = 5,
    n_benign_test: int = 100,
    n_jailbreak_test: int = 50,
    n_features: int = 20,
    shift: float = 5.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create data with very few labeled jailbreaks for training."""
    rng = np.random.default_rng(seed)

    X_train_benign = rng.standard_normal((n_benign_train, n_features)).astype(
        np.float32
    )
    X_train_jailbreak = (
        rng.standard_normal((n_jailbreak_train, n_features)).astype(np.float32)
        + shift
    )
    X_train = np.vstack([X_train_benign, X_train_jailbreak])
    y_train = np.concatenate([
        np.zeros(n_benign_train, dtype=np.int64),
        np.ones(n_jailbreak_train, dtype=np.int64),
    ])

    X_test_benign = rng.standard_normal((n_benign_test, n_features)).astype(
        np.float32
    )
    X_test_jailbreak = (
        rng.standard_normal((n_jailbreak_test, n_features)).astype(np.float32)
        + shift
    )
    X_test = np.vstack([X_test_benign, X_test_jailbreak])
    y_test = np.concatenate([
        np.zeros(n_benign_test, dtype=np.int64),
        np.ones(n_jailbreak_test, dtype=np.int64),
    ])

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def labeled_data():
    """Standard labeled data: 200 benign + 50 jailbreak, 20 features."""
    return _make_labeled_data()


@pytest.fixture
def few_shot_data():
    """Few-shot data: 50 benign + 5 jailbreak for training."""
    return _make_few_shot_data()


@pytest.fixture
def fitted_contrastive(labeled_data):
    X_train, y_train, _, _ = labeled_data
    detector = ContrastiveDirectionDetector()
    detector.fit(X_train, y_train)
    return detector


@pytest.fixture
def fitted_mahalanobis(labeled_data):
    X_train, y_train, _, _ = labeled_data
    detector = ContrastiveMahalanobisDetector(n_components=10)
    detector.fit(X_train, y_train)
    return detector


# ===========================================================================
# ContrastiveDirectionDetector — Subclass
# ===========================================================================


class TestContrastiveDirectionSubclass:
    def test_is_anomaly_detector(self):
        assert issubclass(ContrastiveDirectionDetector, AnomalyDetector)

    def test_instance_is_anomaly_detector(self):
        detector = ContrastiveDirectionDetector()
        assert isinstance(detector, AnomalyDetector)


# ===========================================================================
# ContrastiveDirectionDetector — Requires Labels
# ===========================================================================


class TestContrastiveDirectionRequiresLabels:
    def test_fit_without_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        with pytest.raises(ValueError, match="supervised and requires labels"):
            detector.fit(X_train)

    def test_fit_with_none_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        with pytest.raises(ValueError, match="supervised and requires labels"):
            detector.fit(X_train, y=None)

    def test_fit_single_class_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        y_all_benign = np.zeros(len(X_train), dtype=np.int64)
        with pytest.raises(ValueError, match="Need both"):
            detector.fit(X_train, y_all_benign)

    def test_fit_invalid_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        y_bad = np.full(len(X_train), 2, dtype=np.int64)
        with pytest.raises(ValueError, match="Labels must be 0 or 1"):
            detector.fit(X_train, y_bad)

    def test_fit_length_mismatch_raises(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        with pytest.raises(ValueError, match="samples but y has"):
            detector.fit(X_train, y_train[:5])


# ===========================================================================
# ContrastiveDirectionDetector — Separation
# ===========================================================================


class TestContrastiveDirectionSeparation:
    def test_jailbreaks_score_higher(self, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        benign_scores = scores[y_test == 0]
        jailbreak_scores = scores[y_test == 1]

        assert np.mean(jailbreak_scores) > np.mean(benign_scores)

    def test_high_auroc_on_synthetic(self, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        auroc = roc_auc_score(y_test, scores)
        assert auroc > 0.9, f"AUROC={auroc:.3f} too low for well-separated data"

    def test_benign_scores_centered_near_zero(self, labeled_data):
        """Benign scores should be roughly centered at 0 (z-scored)."""
        X_train, y_train, X_test, y_test = labeled_data
        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        benign_mean = np.mean(scores[y_test == 0])
        # Should be within ~2 std of zero (generous for test)
        assert abs(benign_mean) < 3.0

    def test_score_single_matches_batch(self, fitted_contrastive, labeled_data):
        _, _, X_test, _ = labeled_data
        batch_scores = fitted_contrastive.score(X_test)
        single_score = fitted_contrastive.score_single(X_test[0])
        assert abs(batch_scores[0] - single_score) < 1e-4

    def test_fit_returns_self(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        result = detector.fit(X_train, y_train)
        assert result is detector


# ===========================================================================
# ContrastiveDirectionDetector — Few-shot
# ===========================================================================


class TestContrastiveDirectionFewShot:
    def test_works_with_5_jailbreaks(self, few_shot_data):
        """Should work even with just 5 labeled jailbreaks."""
        X_train, y_train, X_test, y_test = few_shot_data
        assert np.sum(y_train == 1) == 5  # Confirm few-shot

        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        auroc = roc_auc_score(y_test, scores)
        assert auroc > 0.8, f"AUROC={auroc:.3f} too low even with 5 jailbreaks"

    def test_jailbreaks_still_score_higher_few_shot(self, few_shot_data):
        X_train, y_train, X_test, y_test = few_shot_data
        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        assert np.mean(scores[y_test == 1]) > np.mean(scores[y_test == 0])

    def test_works_with_3_jailbreaks(self):
        """Extreme few-shot: just 3 jailbreaks."""
        X_train, y_train, X_test, y_test = _make_few_shot_data(
            n_benign_train=50, n_jailbreak_train=3, shift=5.0, seed=99
        )
        assert np.sum(y_train == 1) == 3

        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        auroc = roc_auc_score(y_test, scores)
        assert auroc > 0.7, f"AUROC={auroc:.3f} even 3 jailbreaks should help"


# ===========================================================================
# ContrastiveDirectionDetector — Direction Properties
# ===========================================================================


class TestContrastiveDirection:
    def test_direction_is_unit_vector(self, fitted_contrastive):
        direction = fitted_contrastive.contrastive_direction
        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-5

    def test_direction_shape_matches_input(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)
        assert detector.contrastive_direction.shape == (X_train.shape[1],)

    def test_direction_points_toward_jailbreaks(self, labeled_data):
        """The direction should have positive projection on jailbreak mean."""
        X_train, y_train, _, _ = labeled_data
        detector = ContrastiveDirectionDetector()
        detector.fit(X_train, y_train)

        mu_benign = X_train[y_train == 0].mean(axis=0)
        mu_jailbreak = X_train[y_train == 1].mean(axis=0)
        diff = mu_jailbreak - mu_benign

        # Direction should be aligned with the true diff-in-means
        cos_sim = np.dot(detector.contrastive_direction, diff) / (
            np.linalg.norm(diff) + 1e-10
        )
        assert cos_sim > 0.9  # Should be nearly aligned

    def test_unnormalized_direction(self, labeled_data):
        """With normalize=False, direction magnitude = distance between means."""
        X_train, y_train, _, _ = labeled_data
        detector = ContrastiveDirectionDetector(normalize=False)
        detector.fit(X_train, y_train)

        mu_benign = X_train[y_train == 0].mean(axis=0)
        mu_jailbreak = X_train[y_train == 1].mean(axis=0)
        expected = mu_jailbreak - mu_benign

        np.testing.assert_array_almost_equal(
            detector.contrastive_direction, expected
        )

    def test_direction_before_fit_raises(self):
        detector = ContrastiveDirectionDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = detector.contrastive_direction

    def test_direction_is_copy(self, fitted_contrastive):
        """Contrastive direction property should return a copy."""
        d1 = fitted_contrastive.contrastive_direction
        d2 = fitted_contrastive.contrastive_direction
        assert d1 is not d2
        np.testing.assert_array_equal(d1, d2)


# ===========================================================================
# ContrastiveDirectionDetector — Validation
# ===========================================================================


class TestContrastiveDirectionValidation:
    def test_score_before_fit_raises(self):
        detector = ContrastiveDirectionDetector()
        X = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(X)

    def test_fit_1d_raises(self):
        detector = ContrastiveDirectionDetector()
        with pytest.raises(ValueError, match="2D array"):
            detector.fit(np.array([1.0, 2.0]), np.array([0, 1]))

    def test_fit_single_sample_raises(self):
        detector = ContrastiveDirectionDetector()
        X = np.random.randn(1, 10).astype(np.float32)
        with pytest.raises(ValueError, match="at least 2"):
            detector.fit(X, np.array([0]))

    def test_score_1d_raises(self, fitted_contrastive):
        with pytest.raises(ValueError, match="2D array"):
            fitted_contrastive.score(np.array([1.0, 2.0]))


# ===========================================================================
# ContrastiveDirectionDetector — Persistence
# ===========================================================================


class TestContrastiveDirectionPersistence:
    def test_save_load_roundtrip(self, tmp_path, fitted_contrastive, labeled_data):
        _, _, X_test, _ = labeled_data
        save_dir = tmp_path / "contrastive"
        fitted_contrastive.save(save_dir)

        loaded = ContrastiveDirectionDetector.load(save_dir)
        original_scores = fitted_contrastive.score(X_test)
        loaded_scores = loaded.score(X_test)
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_save_preserves_config(self, tmp_path, fitted_contrastive):
        save_dir = tmp_path / "contrastive"
        fitted_contrastive.save(save_dir)
        loaded = ContrastiveDirectionDetector.load(save_dir)
        assert loaded.normalize == fitted_contrastive.normalize

    def test_save_preserves_direction(self, tmp_path, fitted_contrastive):
        save_dir = tmp_path / "contrastive"
        fitted_contrastive.save(save_dir)
        loaded = ContrastiveDirectionDetector.load(save_dir)
        np.testing.assert_array_almost_equal(
            fitted_contrastive.contrastive_direction,
            loaded.contrastive_direction,
        )

    def test_save_creates_parent_dirs(self, tmp_path, fitted_contrastive):
        save_dir = tmp_path / "sub" / "dir" / "contrastive"
        fitted_contrastive.save(save_dir)
        assert save_dir.exists()


# ===========================================================================
# ContrastiveMahalanobisDetector — Subclass
# ===========================================================================


class TestContrastiveMahalanobisSubclass:
    def test_is_anomaly_detector(self):
        assert issubclass(ContrastiveMahalanobisDetector, AnomalyDetector)

    def test_instance_is_anomaly_detector(self):
        detector = ContrastiveMahalanobisDetector()
        assert isinstance(detector, AnomalyDetector)


# ===========================================================================
# ContrastiveMahalanobisDetector — Requires Labels
# ===========================================================================


class TestContrastiveMahalanobisRequiresLabels:
    def test_fit_without_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = ContrastiveMahalanobisDetector()
        with pytest.raises(ValueError, match="supervised and requires labels"):
            detector.fit(X_train)

    def test_fit_single_class_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = ContrastiveMahalanobisDetector()
        y_all_benign = np.zeros(len(X_train), dtype=np.int64)
        with pytest.raises(ValueError, match="Need both"):
            detector.fit(X_train, y_all_benign)

    def test_fit_invalid_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = ContrastiveMahalanobisDetector()
        y_bad = np.full(len(X_train), 3, dtype=np.int64)
        with pytest.raises(ValueError, match="Labels must be 0 or 1"):
            detector.fit(X_train, y_bad)

    def test_fit_length_mismatch_raises(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = ContrastiveMahalanobisDetector()
        with pytest.raises(ValueError, match="samples but y has"):
            detector.fit(X_train, y_train[:5])


# ===========================================================================
# ContrastiveMahalanobisDetector — Separation
# ===========================================================================


class TestContrastiveMahalanobisSeparation:
    def test_jailbreaks_score_higher(self, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = ContrastiveMahalanobisDetector(n_components=10)
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        benign_scores = scores[y_test == 0]
        jailbreak_scores = scores[y_test == 1]

        assert np.mean(jailbreak_scores) > np.mean(benign_scores)

    def test_high_auroc_on_synthetic(self, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = ContrastiveMahalanobisDetector(n_components=10)
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        auroc = roc_auc_score(y_test, scores)
        assert auroc > 0.9, f"AUROC={auroc:.3f} too low for well-separated data"

    def test_score_single_matches_batch(self, fitted_mahalanobis, labeled_data):
        _, _, X_test, _ = labeled_data
        batch_scores = fitted_mahalanobis.score(X_test)
        single_score = fitted_mahalanobis.score_single(X_test[0])
        assert abs(batch_scores[0] - single_score) < 1e-5

    def test_fit_returns_self(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = ContrastiveMahalanobisDetector(n_components=10)
        result = detector.fit(X_train, y_train)
        assert result is detector

    def test_no_nan_in_scores(self, fitted_mahalanobis, labeled_data):
        _, _, X_test, _ = labeled_data
        scores = fitted_mahalanobis.score(X_test)
        assert not np.any(np.isnan(scores))

    def test_no_inf_in_scores(self, fitted_mahalanobis, labeled_data):
        _, _, X_test, _ = labeled_data
        scores = fitted_mahalanobis.score(X_test)
        assert not np.any(np.isinf(scores))


# ===========================================================================
# ContrastiveMahalanobisDetector — Outperforms Unsupervised
# ===========================================================================


class TestContrastiveMahalanobisVsUnsupervised:
    def test_beats_unsupervised_mahalanobis(self, labeled_data):
        """Class-conditional Mahalanobis should beat unsupervised version."""
        from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

        X_train, y_train, X_test, y_test = labeled_data

        # Supervised: fit on all labeled data
        supervised = ContrastiveMahalanobisDetector(n_components=10)
        supervised.fit(X_train, y_train)
        sup_scores = supervised.score(X_test)
        sup_auroc = roc_auc_score(y_test, sup_scores)

        # Unsupervised: fit on benign only
        unsupervised = PCAMahalanobisDetector(n_components=10)
        unsupervised.fit(X_train[y_train == 0])
        unsup_scores = unsupervised.score(X_test)
        unsup_auroc = roc_auc_score(y_test, unsup_scores)

        assert sup_auroc >= unsup_auroc - 0.02, (
            f"Supervised AUROC={sup_auroc:.3f} should be >= "
            f"unsupervised AUROC={unsup_auroc:.3f} (within margin)"
        )


# ===========================================================================
# ContrastiveMahalanobisDetector — Few-shot
# ===========================================================================


class TestContrastiveMahalanobisFewShot:
    def test_works_with_few_jailbreaks(self):
        """Should work with limited jailbreak examples for the jailbreak Gaussian."""
        X_train, y_train, X_test, y_test = _make_few_shot_data(
            n_benign_train=50,
            n_jailbreak_train=10,
            n_features=20,
            shift=5.0,
        )
        # Need n_components < n_jailbreak for well-conditioned jailbreak cov
        detector = ContrastiveMahalanobisDetector(
            n_components=5, regularization=1e-3
        )
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        auroc = roc_auc_score(y_test, scores)
        assert auroc > 0.8, f"AUROC={auroc:.3f} too low with few-shot jailbreaks"


# ===========================================================================
# ContrastiveMahalanobisDetector — Validation
# ===========================================================================


class TestContrastiveMahalanobisValidation:
    def test_score_before_fit_raises(self):
        detector = ContrastiveMahalanobisDetector()
        X = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(X)

    def test_fit_1d_raises(self):
        detector = ContrastiveMahalanobisDetector()
        with pytest.raises(ValueError, match="2D array"):
            detector.fit(np.array([1.0, 2.0]), np.array([0, 1]))

    def test_fit_single_sample_raises(self):
        detector = ContrastiveMahalanobisDetector()
        X = np.random.randn(1, 10).astype(np.float32)
        with pytest.raises(ValueError, match="at least 2"):
            detector.fit(X, np.array([0]))

    def test_negative_components_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            ContrastiveMahalanobisDetector(n_components=-1)

    def test_negative_regularization_raises(self):
        with pytest.raises(ValueError, match="regularization"):
            ContrastiveMahalanobisDetector(regularization=-0.01)

    def test_score_1d_raises(self, fitted_mahalanobis):
        with pytest.raises(ValueError, match="2D array"):
            fitted_mahalanobis.score(np.array([1.0, 2.0]))


# ===========================================================================
# ContrastiveMahalanobisDetector — Persistence
# ===========================================================================


class TestContrastiveMahalanobisPersistence:
    def test_save_load_roundtrip(self, tmp_path, fitted_mahalanobis, labeled_data):
        _, _, X_test, _ = labeled_data
        save_dir = tmp_path / "contrastive_mahal"
        fitted_mahalanobis.save(save_dir)

        loaded = ContrastiveMahalanobisDetector.load(save_dir)
        original_scores = fitted_mahalanobis.score(X_test)
        loaded_scores = loaded.score(X_test)
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_save_preserves_config(self, tmp_path, fitted_mahalanobis):
        save_dir = tmp_path / "contrastive_mahal"
        fitted_mahalanobis.save(save_dir)
        loaded = ContrastiveMahalanobisDetector.load(save_dir)
        assert loaded.n_components == fitted_mahalanobis.n_components
        assert loaded.regularization == fitted_mahalanobis.regularization

    def test_save_creates_parent_dirs(self, tmp_path, fitted_mahalanobis):
        save_dir = tmp_path / "sub" / "dir" / "mahal"
        fitted_mahalanobis.save(save_dir)
        assert save_dir.exists()

    def test_loaded_still_separates(self, tmp_path, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = ContrastiveMahalanobisDetector(n_components=10)
        detector.fit(X_train, y_train)

        save_dir = tmp_path / "contrastive_mahal"
        detector.save(save_dir)
        loaded = ContrastiveMahalanobisDetector.load(save_dir)

        scores = loaded.score(X_test)
        benign_mean = np.mean(scores[y_test == 0])
        jailbreak_mean = np.mean(scores[y_test == 1])
        assert jailbreak_mean > benign_mean


# ===========================================================================
# Comparison — Contrastive Direction vs Class-Conditional Mahalanobis
# ===========================================================================


class TestContrastiveComparison:
    def test_both_separate_synthetic_data(self, labeled_data):
        """Both contrastive detectors should achieve good separation."""
        X_train, y_train, X_test, y_test = labeled_data

        direction = ContrastiveDirectionDetector()
        direction.fit(X_train, y_train)
        dir_scores = direction.score(X_test)
        dir_auroc = roc_auc_score(y_test, dir_scores)

        mahal = ContrastiveMahalanobisDetector(n_components=10)
        mahal.fit(X_train, y_train)
        mahal_scores = mahal.score(X_test)
        mahal_auroc = roc_auc_score(y_test, mahal_scores)

        assert dir_auroc > 0.85, f"Direction AUROC={dir_auroc:.3f}"
        assert mahal_auroc > 0.85, f"Mahalanobis AUROC={mahal_auroc:.3f}"

    def test_both_output_correct_shape(self, labeled_data):
        X_train, y_train, X_test, _ = labeled_data

        direction = ContrastiveDirectionDetector()
        direction.fit(X_train, y_train)

        mahal = ContrastiveMahalanobisDetector(n_components=10)
        mahal.fit(X_train, y_train)

        assert direction.score(X_test).shape == (len(X_test),)
        assert mahal.score(X_test).shape == (len(X_test),)
