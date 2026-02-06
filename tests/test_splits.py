"""Tests for data splitting methodology — verifies no jailbreak leakage into training.

Core invariant under test: jailbreak prompts must NEVER appear in the
training or validation sets. Every test that creates splits checks this.
"""

from __future__ import annotations

import numpy as np
import pytest

from venator.activation.storage import ActivationStore
from venator.data.splits import DataSplit, SplitManager

LAYERS = [12, 14]
HIDDEN_DIM = 32
MODEL_ID = "test-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_store(
    tmp_path,
    n_benign: int,
    n_jailbreak: int,
    name: str = "store.h5",
) -> ActivationStore:
    """Create a store with n_benign benign and n_jailbreak jailbreak prompts."""
    rng = np.random.default_rng(123)
    path = str(tmp_path / name)
    store = ActivationStore.create(path, MODEL_ID, LAYERS, HIDDEN_DIM)

    total = n_benign + n_jailbreak
    if total == 0:
        return store

    prompts = [f"prompt_{i}" for i in range(total)]
    labels = [0] * n_benign + [1] * n_jailbreak
    acts = {l: rng.standard_normal((total, HIDDEN_DIM)).astype(np.float32) for l in LAYERS}
    store.append_batch(prompts, acts, labels)
    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store_500_200(tmp_path) -> ActivationStore:
    """Realistic store: 500 benign + 200 jailbreak prompts."""
    return _create_store(tmp_path, 500, 200)


@pytest.fixture
def store_100_50(tmp_path) -> ActivationStore:
    """Moderate store: 100 benign + 50 jailbreak."""
    return _create_store(tmp_path, 100, 50)


@pytest.fixture
def manager() -> SplitManager:
    return SplitManager(seed=42)


# ---------------------------------------------------------------------------
# No label leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    """The most critical tests — verify jailbreaks never leak into train/val."""

    def test_train_has_zero_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        labels = store_500_200.get_labels()
        train_labels = labels[splits["train"].indices]
        assert np.all(train_labels == 0), "Train set contains jailbreak prompts!"

    def test_val_has_zero_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        labels = store_500_200.get_labels()
        val_labels = labels[splits["val"].indices]
        assert np.all(val_labels == 0), "Val set contains jailbreak prompts!"

    def test_test_jailbreak_has_only_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        labels = store_500_200.get_labels()
        jb_labels = labels[splits["test_jailbreak"].indices]
        assert np.all(jb_labels == 1), "test_jailbreak contains benign prompts!"

    def test_test_benign_has_only_benign(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        labels = store_500_200.get_labels()
        tb_labels = labels[splits["test_benign"].indices]
        assert np.all(tb_labels == 0), "test_benign contains jailbreak prompts!"

    def test_all_jailbreaks_in_test(self, store_500_200, manager):
        """Every jailbreak prompt in the store appears in test_jailbreak."""
        splits = manager.create_splits(store_500_200)
        _, jb_idx = store_500_200.split_by_labels()
        np.testing.assert_array_equal(
            np.sort(splits["test_jailbreak"].indices),
            np.sort(jb_idx),
        )


# ---------------------------------------------------------------------------
# Correct proportions
# ---------------------------------------------------------------------------


class TestProportions:
    def test_default_fractions(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)

        n_benign = 500
        assert splits["train"].n_samples == round(n_benign * 0.70)
        assert splits["val"].n_samples == round(n_benign * 0.15)
        # test_benign gets remainder
        expected_test = n_benign - splits["train"].n_samples - splits["val"].n_samples
        assert splits["test_benign"].n_samples == expected_test
        assert splits["test_jailbreak"].n_samples == 200

    def test_custom_fractions(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, train_frac=0.60, val_frac=0.20)

        n_benign = 100
        assert splits["train"].n_samples == round(n_benign * 0.60)
        assert splits["val"].n_samples == round(n_benign * 0.20)
        expected_test = n_benign - splits["train"].n_samples - splits["val"].n_samples
        assert splits["test_benign"].n_samples == expected_test
        assert splits["test_jailbreak"].n_samples == 50

    def test_total_coverage(self, store_500_200, manager):
        """All prompts are assigned to exactly one split."""
        splits = manager.create_splits(store_500_200)
        total = sum(s.n_samples for s in splits.values())
        assert total == store_500_200.n_prompts

    def test_no_index_overlap(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        all_idx = np.concatenate([s.indices for s in splits.values()])
        assert len(all_idx) == len(np.unique(all_idx))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_splits(self, store_100_50):
        s1 = SplitManager(seed=42).create_splits(store_100_50)
        s2 = SplitManager(seed=42).create_splits(store_100_50)

        for name in s1:
            np.testing.assert_array_equal(s1[name].indices, s2[name].indices)

    def test_different_seed_different_splits(self, store_100_50):
        s1 = SplitManager(seed=42).create_splits(store_100_50)
        s2 = SplitManager(seed=99).create_splits(store_100_50)

        # Train indices should differ (different shuffle)
        assert not np.array_equal(s1["train"].indices, s2["train"].indices)

    def test_different_seed_still_no_leakage(self, store_100_50):
        """Regardless of seed, methodology holds."""
        labels = store_100_50.get_labels()
        for seed in (0, 1, 42, 99, 12345):
            splits = SplitManager(seed=seed).create_splits(store_100_50)
            assert np.all(labels[splits["train"].indices] == 0)
            assert np.all(labels[splits["val"].indices] == 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_minimum_benign_samples(self, tmp_path):
        """3 benign + some jailbreaks: each benign split gets at least 1."""
        store = _create_store(tmp_path, 3, 5)
        splits = SplitManager(seed=42).create_splits(store)

        assert splits["train"].n_samples >= 1
        assert splits["val"].n_samples >= 1
        assert splits["test_benign"].n_samples >= 1
        assert splits["test_jailbreak"].n_samples == 5

        # Still no leakage
        labels = store.get_labels()
        assert np.all(labels[splits["train"].indices] == 0)
        assert np.all(labels[splits["val"].indices] == 0)

    def test_too_few_benign_raises(self, tmp_path):
        store = _create_store(tmp_path, 2, 5)
        with pytest.raises(ValueError, match="Too few benign"):
            SplitManager(seed=42).create_splits(store)

    def test_no_benign_raises(self, tmp_path):
        store = _create_store(tmp_path, 0, 10)
        with pytest.raises(ValueError, match="no benign prompts"):
            SplitManager(seed=42).create_splits(store)

    def test_no_jailbreaks(self, tmp_path):
        """All benign — test_jailbreak split should be empty."""
        store = _create_store(tmp_path, 100, 0)
        splits = SplitManager(seed=42).create_splits(store)

        assert splits["test_jailbreak"].n_samples == 0
        total = sum(s.n_samples for s in splits.values())
        assert total == 100

    def test_many_jailbreaks_few_benign(self, tmp_path):
        """10 benign + 500 jailbreaks — extreme ratio."""
        store = _create_store(tmp_path, 10, 500)
        splits = SplitManager(seed=42).create_splits(store)

        labels = store.get_labels()
        assert np.all(labels[splits["train"].indices] == 0)
        assert np.all(labels[splits["val"].indices] == 0)
        assert splits["test_jailbreak"].n_samples == 500


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_passes_for_correct_splits(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Should not raise
        manager.validate_splits(splits, store_100_50)

    def test_validate_catches_jailbreak_in_train(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Corrupt: swap a train index with a jailbreak index
        jb_idx = splits["test_jailbreak"].indices[0]
        splits["train"].indices[0] = jb_idx
        splits["test_jailbreak"] = DataSplit(
            name="test_jailbreak",
            indices=splits["test_jailbreak"].indices[1:],
            n_samples=splits["test_jailbreak"].n_samples - 1,
            contains_jailbreaks=True,
        )

        with pytest.raises(ValueError, match="METHODOLOGY VIOLATION"):
            manager.validate_splits(splits, store_100_50)

    def test_validate_catches_index_overlap(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Corrupt: duplicate an index into val
        splits["val"].indices = np.append(
            splits["val"].indices, splits["train"].indices[0]
        )
        splits["val"].n_samples += 1

        with pytest.raises(ValueError, match="overlap"):
            manager.validate_splits(splits, store_100_50)

    def test_validate_catches_missing_coverage(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Corrupt: remove an index from train
        splits["train"].indices = splits["train"].indices[1:]
        splits["train"].n_samples -= 1

        with pytest.raises(ValueError, match="cover"):
            manager.validate_splits(splits, store_100_50)


# ---------------------------------------------------------------------------
# Invalid fractions
# ---------------------------------------------------------------------------


class TestInvalidFractions:
    def test_fractions_exceed_one(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(store_100_50, train_frac=0.80, val_frac=0.30)

    def test_negative_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(store_100_50, train_frac=-0.1, val_frac=0.5)

    def test_zero_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(store_100_50, train_frac=0.0, val_frac=0.5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50)

        path = tmp_path / "splits.json"
        manager.save_splits(splits, path)

        loaded = SplitManager.load_splits(path)

        assert set(loaded.keys()) == set(splits.keys())
        for name in splits:
            np.testing.assert_array_equal(loaded[name].indices, splits[name].indices)
            assert loaded[name].n_samples == splits[name].n_samples
            assert loaded[name].contains_jailbreaks == splits[name].contains_jailbreaks
            assert loaded[name].name == splits[name].name

    def test_loaded_splits_validate(self, tmp_path, store_100_50, manager):
        """Loaded splits pass validation against the same store."""
        splits = manager.create_splits(store_100_50)
        path = tmp_path / "splits.json"
        manager.save_splits(splits, path)

        loaded = SplitManager.load_splits(path)
        # Should not raise
        manager.validate_splits(loaded, store_100_50)

    def test_save_creates_parent_dirs(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        path = tmp_path / "sub" / "dir" / "splits.json"
        manager.save_splits(splits, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# DataSplit dataclass
# ---------------------------------------------------------------------------


class TestDataSplit:
    def test_post_init_converts_to_int64(self):
        split = DataSplit(
            name="test",
            indices=[1, 2, 3],
            n_samples=3,
            contains_jailbreaks=False,
        )
        assert split.indices.dtype == np.int64

    def test_post_init_updates_n_samples(self):
        split = DataSplit(
            name="test",
            indices=[1, 2, 3, 4, 5],
            n_samples=0,  # Wrong — __post_init__ fixes it
            contains_jailbreaks=False,
        )
        assert split.n_samples == 5
