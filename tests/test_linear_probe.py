"""Tests for supervised probe detectors (LinearProbe and MLPProbe).

Tests verify:
- Supervised detectors refuse to train without labels
- Separation on synthetic Gaussian data
- Scores are in [0, 1] range (probabilities)
- Handling of imbalanced data (few jailbreaks, many benign)
- Save/load roundtrip
- Comparison between linear probe and MLP probe
"""

from __future__ import annotations

import numpy as np
import pytest

from venator.detection.base import AnomalyDetector
from venator.detection.linear_probe import LinearProbeDetector, MLPProbeDetector

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


def _make_imbalanced_data(
    n_benign: int = 300,
    n_jailbreak: int = 10,
    n_features: int = 20,
    shift: float = 5.0,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create heavily imbalanced labeled data."""
    return _make_labeled_data(
        n_benign=n_benign,
        n_jailbreak=n_jailbreak,
        n_features=n_features,
        shift=shift,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def labeled_data():
    """Standard labeled data: 200 benign + 50 jailbreak, 20 features."""
    return _make_labeled_data()


@pytest.fixture
def imbalanced_data():
    """Imbalanced data: 300 benign + 10 jailbreak, 20 features."""
    return _make_imbalanced_data()


@pytest.fixture
def fitted_linear_probe(labeled_data):
    X_train, y_train, _, _ = labeled_data
    detector = LinearProbeDetector(n_components=10)
    detector.fit(X_train, y_train)
    return detector


@pytest.fixture
def fitted_mlp_probe(labeled_data):
    X_train, y_train, _, _ = labeled_data
    detector = MLPProbeDetector(
        n_components=10, hidden1=32, hidden2=16,
        epochs=50, early_stopping_patience=10,
    )
    detector.fit(X_train, y_train)
    return detector


# ===========================================================================
# LinearProbeDetector — Subclass
# ===========================================================================


class TestLinearProbeSubclass:
    def test_is_anomaly_detector(self):
        assert issubclass(LinearProbeDetector, AnomalyDetector)

    def test_instance_is_anomaly_detector(self):
        detector = LinearProbeDetector()
        assert isinstance(detector, AnomalyDetector)


# ===========================================================================
# LinearProbeDetector — Requires Labels
# ===========================================================================


class TestLinearProbeRequiresLabels:
    def test_fit_without_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = LinearProbeDetector()
        with pytest.raises(ValueError, match="supervised and requires labels"):
            detector.fit(X_train)

    def test_fit_with_none_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = LinearProbeDetector()
        with pytest.raises(ValueError, match="supervised and requires labels"):
            detector.fit(X_train, y=None)

    def test_fit_single_class_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = LinearProbeDetector()
        y_all_benign = np.zeros(len(X_train), dtype=np.int64)
        with pytest.raises(ValueError, match="Need both"):
            detector.fit(X_train, y_all_benign)

    def test_fit_invalid_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = LinearProbeDetector()
        y_bad = np.full(len(X_train), 2, dtype=np.int64)
        with pytest.raises(ValueError, match="Labels must be 0 or 1"):
            detector.fit(X_train, y_bad)

    def test_fit_length_mismatch_raises(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = LinearProbeDetector()
        with pytest.raises(ValueError, match="samples but y has"):
            detector.fit(X_train, y_train[:5])


# ===========================================================================
# LinearProbeDetector — Separation
# ===========================================================================


class TestLinearProbeSeparation:
    def test_jailbreaks_score_higher(self, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = LinearProbeDetector(n_components=10)
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        benign_scores = scores[y_test == 0]
        jailbreak_scores = scores[y_test == 1]

        assert np.mean(jailbreak_scores) > np.mean(benign_scores)

    def test_high_auroc_on_synthetic(self, labeled_data):
        from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

        X_train, y_train, X_test, y_test = labeled_data
        detector = LinearProbeDetector(n_components=10)
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        auroc = roc_auc_score(y_test, scores)
        assert auroc > 0.9, f"AUROC={auroc:.3f} too low for well-separated data"

    def test_scores_in_0_1_range(self, labeled_data):
        X_train, y_train, X_test, _ = labeled_data
        detector = LinearProbeDetector(n_components=10)
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_score_single_matches_batch(self, fitted_linear_probe, labeled_data):
        _, _, X_test, _ = labeled_data
        batch_scores = fitted_linear_probe.score(X_test)
        single_score = fitted_linear_probe.score_single(X_test[0])
        assert abs(batch_scores[0] - single_score) < 1e-6

    def test_fit_returns_self(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = LinearProbeDetector(n_components=10)
        result = detector.fit(X_train, y_train)
        assert result is detector


# ===========================================================================
# LinearProbeDetector — No PCA
# ===========================================================================


class TestLinearProbeNoPCA:
    def test_works_without_pca(self, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = LinearProbeDetector(n_components=None)
        detector.fit(X_train, y_train)
        scores = detector.score(X_test)
        assert scores.shape == (len(X_test),)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_pca_none_no_pca_fitted(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = LinearProbeDetector(n_components=None)
        detector.fit(X_train, y_train)
        assert detector.pca_ is None


# ===========================================================================
# LinearProbeDetector — Probe Direction
# ===========================================================================


class TestProbeDirection:
    def test_direction_is_unit_vector(self, fitted_linear_probe):
        direction = fitted_linear_probe.get_probe_direction()
        norm = np.linalg.norm(direction)
        assert abs(norm - 1.0) < 1e-5

    def test_direction_shape_with_pca(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        n_features = X_train.shape[1]
        detector = LinearProbeDetector(n_components=10)
        detector.fit(X_train, y_train)
        direction = detector.get_probe_direction()
        assert direction.shape == (n_features,)

    def test_direction_shape_without_pca(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        n_features = X_train.shape[1]
        detector = LinearProbeDetector(n_components=None)
        detector.fit(X_train, y_train)
        direction = detector.get_probe_direction()
        assert direction.shape == (n_features,)

    def test_direction_before_fit_raises(self):
        detector = LinearProbeDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.get_probe_direction()


# ===========================================================================
# LinearProbeDetector — Imbalanced Data
# ===========================================================================


class TestLinearProbeImbalanced:
    def test_handles_imbalanced_data(self, imbalanced_data):
        X_train, y_train, X_test, y_test = imbalanced_data
        detector = LinearProbeDetector(n_components=10, class_weight="balanced")
        detector.fit(X_train, y_train)
        scores = detector.score(X_test)

        # Should still separate despite imbalance
        benign_scores = scores[y_test == 0]
        jailbreak_scores = scores[y_test == 1]
        if len(jailbreak_scores) > 0:
            assert np.mean(jailbreak_scores) > np.mean(benign_scores)


# ===========================================================================
# LinearProbeDetector — Validation
# ===========================================================================


class TestLinearProbeValidation:
    def test_score_before_fit_raises(self):
        detector = LinearProbeDetector()
        X = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(X)

    def test_fit_1d_raises(self):
        detector = LinearProbeDetector()
        with pytest.raises(ValueError, match="2D array"):
            detector.fit(np.array([1.0, 2.0]), np.array([0, 1]))

    def test_fit_single_sample_raises(self):
        detector = LinearProbeDetector()
        X = np.random.randn(1, 10).astype(np.float32)
        with pytest.raises(ValueError, match="at least 2"):
            detector.fit(X, np.array([0]))

    def test_negative_components_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            LinearProbeDetector(n_components=-1)

    def test_negative_C_raises(self):
        with pytest.raises(ValueError, match="C must be"):
            LinearProbeDetector(C=-1.0)


# ===========================================================================
# LinearProbeDetector — Persistence
# ===========================================================================


class TestLinearProbePersistence:
    def test_save_load_roundtrip(self, tmp_path, fitted_linear_probe, labeled_data):
        _, _, X_test, _ = labeled_data
        save_dir = tmp_path / "linear_probe"
        fitted_linear_probe.save(save_dir)

        loaded = LinearProbeDetector.load(save_dir)
        original_scores = fitted_linear_probe.score(X_test)
        loaded_scores = loaded.score(X_test)
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_save_preserves_config(self, tmp_path, fitted_linear_probe):
        save_dir = tmp_path / "linear_probe"
        fitted_linear_probe.save(save_dir)
        loaded = LinearProbeDetector.load(save_dir)
        assert loaded.C == fitted_linear_probe.C
        assert loaded.max_iter == fitted_linear_probe.max_iter

    def test_save_creates_parent_dirs(self, tmp_path, fitted_linear_probe):
        save_dir = tmp_path / "sub" / "dir" / "probe"
        fitted_linear_probe.save(save_dir)
        assert save_dir.exists()

    def test_save_load_no_pca(self, tmp_path, labeled_data):
        X_train, y_train, X_test, _ = labeled_data
        detector = LinearProbeDetector(n_components=None)
        detector.fit(X_train, y_train)

        save_dir = tmp_path / "no_pca_probe"
        detector.save(save_dir)
        loaded = LinearProbeDetector.load(save_dir)

        assert loaded.n_components is None
        np.testing.assert_array_almost_equal(
            detector.score(X_test), loaded.score(X_test)
        )


# ===========================================================================
# MLPProbeDetector — Subclass
# ===========================================================================


class TestMLPProbeSubclass:
    def test_is_anomaly_detector(self):
        assert issubclass(MLPProbeDetector, AnomalyDetector)


# ===========================================================================
# MLPProbeDetector — Requires Labels
# ===========================================================================


class TestMLPProbeRequiresLabels:
    def test_fit_without_labels_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = MLPProbeDetector()
        with pytest.raises(ValueError, match="supervised and requires labels"):
            detector.fit(X_train)

    def test_fit_single_class_raises(self, labeled_data):
        X_train, _, _, _ = labeled_data
        detector = MLPProbeDetector()
        y_all_benign = np.zeros(len(X_train), dtype=np.int64)
        with pytest.raises(ValueError, match="Need both"):
            detector.fit(X_train, y_all_benign)


# ===========================================================================
# MLPProbeDetector — Separation
# ===========================================================================


class TestMLPProbeSeparation:
    def test_jailbreaks_score_higher(self, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = MLPProbeDetector(
            n_components=10, hidden1=32, hidden2=16,
            epochs=100, early_stopping_patience=15,
        )
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        benign_scores = scores[y_test == 0]
        jailbreak_scores = scores[y_test == 1]
        assert np.mean(jailbreak_scores) > np.mean(benign_scores)

    def test_scores_in_0_1_range(self, labeled_data):
        X_train, y_train, X_test, _ = labeled_data
        detector = MLPProbeDetector(
            n_components=10, hidden1=32, hidden2=16,
            epochs=50, early_stopping_patience=10,
        )
        detector.fit(X_train, y_train)

        scores = detector.score(X_test)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_no_nan_in_scores(self, fitted_mlp_probe, labeled_data):
        _, _, X_test, _ = labeled_data
        scores = fitted_mlp_probe.score(X_test)
        assert not np.any(np.isnan(scores))

    def test_no_inf_in_scores(self, fitted_mlp_probe, labeled_data):
        _, _, X_test, _ = labeled_data
        scores = fitted_mlp_probe.score(X_test)
        assert not np.any(np.isinf(scores))

    def test_score_single_matches_batch(self, fitted_mlp_probe, labeled_data):
        _, _, X_test, _ = labeled_data
        batch_scores = fitted_mlp_probe.score(X_test)
        single_score = fitted_mlp_probe.score_single(X_test[0])
        assert abs(batch_scores[0] - single_score) < 1e-5

    def test_fit_returns_self(self, labeled_data):
        X_train, y_train, _, _ = labeled_data
        detector = MLPProbeDetector(
            n_components=10, hidden1=16, hidden2=8,
            epochs=10, early_stopping_patience=5,
        )
        result = detector.fit(X_train, y_train)
        assert result is detector


# ===========================================================================
# MLPProbeDetector — Imbalanced Data
# ===========================================================================


class TestMLPProbeImbalanced:
    def test_handles_imbalanced_data(self, imbalanced_data):
        X_train, y_train, X_test, y_test = imbalanced_data
        detector = MLPProbeDetector(
            n_components=10, hidden1=32, hidden2=16,
            epochs=100, early_stopping_patience=15,
        )
        detector.fit(X_train, y_train)
        scores = detector.score(X_test)

        # Should still produce valid probabilities
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# ===========================================================================
# MLPProbeDetector — Validation
# ===========================================================================


class TestMLPProbeValidation:
    def test_score_before_fit_raises(self):
        detector = MLPProbeDetector()
        X = np.random.randn(5, 10).astype(np.float32)
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(X)

    def test_fit_1d_raises(self):
        detector = MLPProbeDetector()
        with pytest.raises(ValueError, match="2D array"):
            detector.fit(np.array([1.0, 2.0]), np.array([0, 1]))

    def test_negative_components_raises(self):
        with pytest.raises(ValueError, match="n_components"):
            MLPProbeDetector(n_components=-1)

    def test_negative_hidden1_raises(self):
        with pytest.raises(ValueError, match="hidden1"):
            MLPProbeDetector(hidden1=-1)

    def test_negative_hidden2_raises(self):
        with pytest.raises(ValueError, match="hidden2"):
            MLPProbeDetector(hidden2=-1)


# ===========================================================================
# MLPProbeDetector — Persistence
# ===========================================================================


class TestMLPProbePersistence:
    def test_save_load_roundtrip(self, tmp_path, fitted_mlp_probe, labeled_data):
        _, _, X_test, _ = labeled_data
        save_dir = tmp_path / "mlp_probe"
        fitted_mlp_probe.save(save_dir)

        loaded = MLPProbeDetector.load(save_dir)
        original_scores = fitted_mlp_probe.score(X_test)
        loaded_scores = loaded.score(X_test)
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_save_preserves_config(self, tmp_path, fitted_mlp_probe):
        save_dir = tmp_path / "mlp_probe"
        fitted_mlp_probe.save(save_dir)
        loaded = MLPProbeDetector.load(save_dir)
        assert loaded.hidden1 == fitted_mlp_probe.hidden1
        assert loaded.hidden2 == fitted_mlp_probe.hidden2

    def test_save_creates_parent_dirs(self, tmp_path, fitted_mlp_probe):
        save_dir = tmp_path / "sub" / "dir" / "probe"
        fitted_mlp_probe.save(save_dir)
        assert save_dir.exists()

    def test_loaded_still_separates(self, tmp_path, labeled_data):
        X_train, y_train, X_test, y_test = labeled_data
        detector = MLPProbeDetector(
            n_components=10, hidden1=32, hidden2=16,
            epochs=50, early_stopping_patience=10,
        )
        detector.fit(X_train, y_train)

        save_dir = tmp_path / "mlp_probe"
        detector.save(save_dir)
        loaded = MLPProbeDetector.load(save_dir)

        scores = loaded.score(X_test)
        benign_mean = np.mean(scores[y_test == 0])
        jailbreak_mean = np.mean(scores[y_test == 1])
        assert jailbreak_mean > benign_mean


# ===========================================================================
# Comparison — Linear vs MLP
# ===========================================================================


class TestProbeComparison:
    def test_both_separate_synthetic_data(self, labeled_data):
        """Both probes should achieve reasonable separation on easy data."""
        from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

        X_train, y_train, X_test, y_test = labeled_data

        linear = LinearProbeDetector(n_components=10)
        linear.fit(X_train, y_train)
        linear_scores = linear.score(X_test)

        mlp = MLPProbeDetector(
            n_components=10, hidden1=32, hidden2=16,
            epochs=100, early_stopping_patience=15,
        )
        mlp.fit(X_train, y_train)
        mlp_scores = mlp.score(X_test)

        linear_auroc = roc_auc_score(y_test, linear_scores)
        mlp_auroc = roc_auc_score(y_test, mlp_scores)

        # Both should be good on this easy dataset
        assert linear_auroc > 0.85, f"Linear AUROC={linear_auroc:.3f}"
        assert mlp_auroc > 0.85, f"MLP AUROC={mlp_auroc:.3f}"

    def test_both_output_same_shape(self, labeled_data):
        X_train, y_train, X_test, _ = labeled_data

        linear = LinearProbeDetector(n_components=10)
        linear.fit(X_train, y_train)

        mlp = MLPProbeDetector(
            n_components=10, hidden1=16, hidden2=8,
            epochs=10, early_stopping_patience=5,
        )
        mlp.fit(X_train, y_train)

        assert linear.score(X_test).shape == mlp.score(X_test).shape
