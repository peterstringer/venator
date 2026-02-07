"""Tests for data splitting methodology — verifies no jailbreak leakage into training.

Core invariant under test: jailbreak prompts must NEVER appear in the
training or validation sets in UNSUPERVISED mode. Every test that creates
splits checks this.

Also tests the SEMI_SUPERVISED mode where a small labeled jailbreak fraction
is included in train/val, with the majority reserved for testing.
"""

from __future__ import annotations

import numpy as np
import pytest

from venator.activation.storage import ActivationStore
from venator.data.splits import DataSplit, SplitManager, SplitMode

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


# ===========================================================================
# UNSUPERVISED MODE TESTS (original methodology)
# ===========================================================================


class TestNoLeakage:
    """The most critical tests — verify jailbreaks never leak into train/val."""

    def test_train_has_zero_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)
        labels = store_500_200.get_labels()
        train_labels = labels[splits["train"].indices]
        assert np.all(train_labels == 0), "Train set contains jailbreak prompts!"

    def test_val_has_zero_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)
        labels = store_500_200.get_labels()
        val_labels = labels[splits["val"].indices]
        assert np.all(val_labels == 0), "Val set contains jailbreak prompts!"

    def test_test_jailbreak_has_only_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)
        labels = store_500_200.get_labels()
        jb_labels = labels[splits["test_jailbreak"].indices]
        assert np.all(jb_labels == 1), "test_jailbreak contains benign prompts!"

    def test_test_benign_has_only_benign(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)
        labels = store_500_200.get_labels()
        tb_labels = labels[splits["test_benign"].indices]
        assert np.all(tb_labels == 0), "test_benign contains jailbreak prompts!"

    def test_all_jailbreaks_in_test(self, store_500_200, manager):
        """Every jailbreak prompt in the store appears in test_jailbreak."""
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)
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
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)

        n_benign = 500
        assert splits["train"].n_samples == round(n_benign * 0.70)
        assert splits["val"].n_samples == round(n_benign * 0.15)
        # test_benign gets remainder
        expected_test = n_benign - splits["train"].n_samples - splits["val"].n_samples
        assert splits["test_benign"].n_samples == expected_test
        assert splits["test_jailbreak"].n_samples == 200

    def test_custom_fractions(self, store_100_50, manager):
        splits = manager.create_splits(
            store_100_50,
            mode=SplitMode.UNSUPERVISED,
            benign_train_frac=0.60,
            benign_val_frac=0.20,
        )

        n_benign = 100
        assert splits["train"].n_samples == round(n_benign * 0.60)
        assert splits["val"].n_samples == round(n_benign * 0.20)
        expected_test = n_benign - splits["train"].n_samples - splits["val"].n_samples
        assert splits["test_benign"].n_samples == expected_test
        assert splits["test_jailbreak"].n_samples == 50

    def test_total_coverage(self, store_500_200, manager):
        """All prompts are assigned to exactly one split."""
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)
        total = sum(s.n_samples for s in splits.values())
        assert total == store_500_200.n_prompts

    def test_no_index_overlap(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.UNSUPERVISED)
        all_idx = np.concatenate([s.indices for s in splits.values()])
        assert len(all_idx) == len(np.unique(all_idx))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_splits(self, store_100_50):
        s1 = SplitManager(seed=42).create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        s2 = SplitManager(seed=42).create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)

        for name in s1:
            np.testing.assert_array_equal(s1[name].indices, s2[name].indices)

    def test_different_seed_different_splits(self, store_100_50):
        s1 = SplitManager(seed=42).create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        s2 = SplitManager(seed=99).create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)

        # Train indices should differ (different shuffle)
        assert not np.array_equal(s1["train"].indices, s2["train"].indices)

    def test_different_seed_still_no_leakage(self, store_100_50):
        """Regardless of seed, methodology holds."""
        labels = store_100_50.get_labels()
        for seed in (0, 1, 42, 99, 12345):
            splits = SplitManager(seed=seed).create_splits(
                store_100_50, mode=SplitMode.UNSUPERVISED
            )
            assert np.all(labels[splits["train"].indices] == 0)
            assert np.all(labels[splits["val"].indices] == 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_minimum_benign_samples(self, tmp_path):
        """3 benign + some jailbreaks: each benign split gets at least 1."""
        store = _create_store(tmp_path, 3, 5)
        splits = SplitManager(seed=42).create_splits(store, mode=SplitMode.UNSUPERVISED)

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
            SplitManager(seed=42).create_splits(store, mode=SplitMode.UNSUPERVISED)

    def test_no_benign_raises(self, tmp_path):
        store = _create_store(tmp_path, 0, 10)
        with pytest.raises(ValueError, match="no benign prompts"):
            SplitManager(seed=42).create_splits(store, mode=SplitMode.UNSUPERVISED)

    def test_no_jailbreaks(self, tmp_path):
        """All benign — test_jailbreak split should be empty."""
        store = _create_store(tmp_path, 100, 0)
        splits = SplitManager(seed=42).create_splits(store, mode=SplitMode.UNSUPERVISED)

        assert splits["test_jailbreak"].n_samples == 0
        total = sum(s.n_samples for s in splits.values())
        assert total == 100

    def test_many_jailbreaks_few_benign(self, tmp_path):
        """10 benign + 500 jailbreaks — extreme ratio."""
        store = _create_store(tmp_path, 10, 500)
        splits = SplitManager(seed=42).create_splits(store, mode=SplitMode.UNSUPERVISED)

        labels = store.get_labels()
        assert np.all(labels[splits["train"].indices] == 0)
        assert np.all(labels[splits["val"].indices] == 0)
        assert splits["test_jailbreak"].n_samples == 500


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_passes_for_correct_splits(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        # Should not raise
        manager.validate_splits(splits, store_100_50, mode=SplitMode.UNSUPERVISED)

    def test_validate_catches_jailbreak_in_train(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
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
            manager.validate_splits(splits, store_100_50, mode=SplitMode.UNSUPERVISED)

    def test_validate_catches_index_overlap(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        # Corrupt: duplicate an index into val
        splits["val"].indices = np.append(
            splits["val"].indices, splits["train"].indices[0]
        )
        splits["val"].n_samples += 1

        with pytest.raises(ValueError, match="overlap"):
            manager.validate_splits(splits, store_100_50, mode=SplitMode.UNSUPERVISED)

    def test_validate_catches_missing_coverage(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        # Corrupt: remove an index from train
        splits["train"].indices = splits["train"].indices[1:]
        splits["train"].n_samples -= 1

        with pytest.raises(ValueError, match="cover"):
            manager.validate_splits(splits, store_100_50, mode=SplitMode.UNSUPERVISED)


# ---------------------------------------------------------------------------
# Invalid fractions
# ---------------------------------------------------------------------------


class TestInvalidFractions:
    def test_fractions_exceed_one(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(
                store_100_50,
                mode=SplitMode.UNSUPERVISED,
                benign_train_frac=0.80,
                benign_val_frac=0.30,
            )

    def test_negative_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(
                store_100_50,
                mode=SplitMode.UNSUPERVISED,
                benign_train_frac=-0.1,
                benign_val_frac=0.5,
            )

    def test_zero_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(
                store_100_50,
                mode=SplitMode.UNSUPERVISED,
                benign_train_frac=0.0,
                benign_val_frac=0.5,
            )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)

        path = tmp_path / "splits.json"
        manager.save_splits(splits, path, mode=SplitMode.UNSUPERVISED)

        loaded = SplitManager.load_splits(path)

        assert set(loaded.keys()) == set(splits.keys())
        for name in splits:
            np.testing.assert_array_equal(loaded[name].indices, splits[name].indices)
            assert loaded[name].n_samples == splits[name].n_samples
            assert loaded[name].contains_jailbreaks == splits[name].contains_jailbreaks
            assert loaded[name].name == splits[name].name

    def test_loaded_splits_validate(self, tmp_path, store_100_50, manager):
        """Loaded splits pass validation against the same store."""
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        path = tmp_path / "splits.json"
        manager.save_splits(splits, path, mode=SplitMode.UNSUPERVISED)

        loaded = SplitManager.load_splits(path)
        # Should not raise
        manager.validate_splits(loaded, store_100_50, mode=SplitMode.UNSUPERVISED)

    def test_save_creates_parent_dirs(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        path = tmp_path / "sub" / "dir" / "splits.json"
        manager.save_splits(splits, path, mode=SplitMode.UNSUPERVISED)
        assert path.exists()

    def test_load_mode_unsupervised(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        path = tmp_path / "splits.json"
        manager.save_splits(splits, path, mode=SplitMode.UNSUPERVISED)

        loaded_mode = SplitManager.load_mode(path)
        assert loaded_mode == SplitMode.UNSUPERVISED

    def test_load_mode_defaults_for_legacy_files(self, tmp_path, store_100_50, manager):
        """Files saved without a mode field default to UNSUPERVISED."""
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        path = tmp_path / "splits.json"
        manager.save_splits(splits, path, mode=SplitMode.UNSUPERVISED)

        # Manually remove the mode key to simulate legacy format
        import json
        with open(path, "r") as f:
            data = json.load(f)
        del data["mode"]
        with open(path, "w") as f:
            json.dump(data, f)

        loaded_mode = SplitManager.load_mode(path)
        assert loaded_mode == SplitMode.UNSUPERVISED


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


# ===========================================================================
# SEMI-SUPERVISED MODE TESTS
# ===========================================================================


class TestSemiSupervisedNoLeakage:
    """Verify label integrity in semi-supervised splits."""

    def test_benign_splits_have_only_benign(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)
        labels = store_500_200.get_labels()

        for split_name in ("train_benign", "val_benign", "test_benign"):
            split_labels = labels[splits[split_name].indices]
            assert np.all(split_labels == 0), (
                f"{split_name} contains jailbreak prompts!"
            )

    def test_jailbreak_splits_have_only_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)
        labels = store_500_200.get_labels()

        for split_name in ("train_jailbreak", "val_jailbreak", "test_jailbreak"):
            split_labels = labels[splits[split_name].indices]
            assert np.all(split_labels == 1), (
                f"{split_name} contains benign prompts!"
            )

    def test_test_jailbreaks_never_in_train(self, store_500_200, manager):
        """No jailbreak index appears in both train and test."""
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)

        train_jb = set(splits["train_jailbreak"].indices.tolist())
        test_jb = set(splits["test_jailbreak"].indices.tolist())
        assert train_jb.isdisjoint(test_jb), (
            "Jailbreak indices overlap between train and test!"
        )

    def test_six_splits_present(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)
        expected_keys = {
            "train_benign", "train_jailbreak",
            "val_benign", "val_jailbreak",
            "test_benign", "test_jailbreak",
        }
        assert set(splits.keys()) == expected_keys


class TestSemiSupervisedProportions:
    def test_default_fractions(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)

        n_benign, n_jailbreak = 500, 200

        # Benign splits
        assert splits["train_benign"].n_samples == round(n_benign * 0.70)
        assert splits["val_benign"].n_samples == round(n_benign * 0.15)
        expected_test_b = (
            n_benign
            - splits["train_benign"].n_samples
            - splits["val_benign"].n_samples
        )
        assert splits["test_benign"].n_samples == expected_test_b

        # Jailbreak splits
        assert splits["train_jailbreak"].n_samples == round(n_jailbreak * 0.15)
        assert splits["val_jailbreak"].n_samples == round(n_jailbreak * 0.15)
        expected_test_j = (
            n_jailbreak
            - splits["train_jailbreak"].n_samples
            - splits["val_jailbreak"].n_samples
        )
        assert splits["test_jailbreak"].n_samples == expected_test_j

    def test_majority_jailbreaks_reserved_for_test(self, store_500_200, manager):
        """At least 70% of jailbreaks end up in test_jailbreak with defaults."""
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)
        _, jb_idx = store_500_200.split_by_labels()
        n_jailbreak = len(jb_idx)
        assert splits["test_jailbreak"].n_samples >= n_jailbreak * 0.60

    def test_custom_jailbreak_fractions(self, store_500_200, manager):
        splits = manager.create_splits(
            store_500_200,
            mode=SplitMode.SEMI_SUPERVISED,
            jailbreak_train_frac=0.10,
            jailbreak_val_frac=0.10,
        )
        n_jailbreak = 200
        assert splits["train_jailbreak"].n_samples == round(n_jailbreak * 0.10)
        assert splits["val_jailbreak"].n_samples == round(n_jailbreak * 0.10)
        # 80% reserved for test
        expected_test = (
            n_jailbreak
            - splits["train_jailbreak"].n_samples
            - splits["val_jailbreak"].n_samples
        )
        assert splits["test_jailbreak"].n_samples == expected_test

    def test_total_coverage(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)
        total = sum(s.n_samples for s in splits.values())
        assert total == store_500_200.n_prompts

    def test_no_index_overlap(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200, mode=SplitMode.SEMI_SUPERVISED)
        all_idx = np.concatenate([s.indices for s in splits.values()])
        assert len(all_idx) == len(np.unique(all_idx))


class TestSemiSupervisedReproducibility:
    def test_same_seed_same_splits(self, store_100_50):
        s1 = SplitManager(seed=42).create_splits(
            store_100_50, mode=SplitMode.SEMI_SUPERVISED
        )
        s2 = SplitManager(seed=42).create_splits(
            store_100_50, mode=SplitMode.SEMI_SUPERVISED
        )
        for name in s1:
            np.testing.assert_array_equal(s1[name].indices, s2[name].indices)

    def test_different_seed_different_splits(self, store_100_50):
        s1 = SplitManager(seed=42).create_splits(
            store_100_50, mode=SplitMode.SEMI_SUPERVISED
        )
        s2 = SplitManager(seed=99).create_splits(
            store_100_50, mode=SplitMode.SEMI_SUPERVISED
        )
        assert not np.array_equal(
            s1["train_benign"].indices, s2["train_benign"].indices
        )


class TestSemiSupervisedEdgeCases:
    def test_no_jailbreaks_raises(self, tmp_path):
        """Semi-supervised mode requires jailbreak prompts."""
        store = _create_store(tmp_path, 100, 0)
        with pytest.raises(ValueError, match="no jailbreak prompts"):
            SplitManager(seed=42).create_splits(store, mode=SplitMode.SEMI_SUPERVISED)

    def test_too_much_jailbreak_in_train_val_raises(self, store_100_50, manager):
        """Jailbreak test fraction must be >= 50%."""
        with pytest.raises(ValueError, match="50%"):
            manager.create_splits(
                store_100_50,
                mode=SplitMode.SEMI_SUPERVISED,
                jailbreak_train_frac=0.30,
                jailbreak_val_frac=0.25,
            )

    def test_few_jailbreaks(self, tmp_path):
        """With only 3 jailbreaks and default 15/15/70, we should still work."""
        store = _create_store(tmp_path, 50, 3)
        splits = SplitManager(seed=42).create_splits(
            store, mode=SplitMode.SEMI_SUPERVISED
        )

        # At least 1 jailbreak in test
        assert splits["test_jailbreak"].n_samples >= 1

        # Total coverage
        total = sum(s.n_samples for s in splits.values())
        assert total == store.n_prompts

    def test_minimum_benign_semi_supervised(self, tmp_path):
        """3 benign + 10 jailbreaks — each benign split gets at least 1."""
        store = _create_store(tmp_path, 3, 10)
        splits = SplitManager(seed=42).create_splits(
            store, mode=SplitMode.SEMI_SUPERVISED
        )

        assert splits["train_benign"].n_samples >= 1
        assert splits["val_benign"].n_samples >= 1
        assert splits["test_benign"].n_samples >= 1


class TestSemiSupervisedValidation:
    def test_validate_passes_for_correct_splits(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.SEMI_SUPERVISED)
        manager.validate_splits(
            splits, store_100_50, mode=SplitMode.SEMI_SUPERVISED
        )

    def test_validate_catches_benign_in_jailbreak_split(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.SEMI_SUPERVISED)
        # Corrupt: put a benign index into train_jailbreak
        benign_idx = splits["train_benign"].indices[0]
        splits["train_jailbreak"].indices = np.append(
            splits["train_jailbreak"].indices, benign_idx
        )
        splits["train_jailbreak"].n_samples += 1
        splits["train_benign"].indices = splits["train_benign"].indices[1:]
        splits["train_benign"].n_samples -= 1

        with pytest.raises(ValueError, match="LABEL MISMATCH"):
            manager.validate_splits(
                splits, store_100_50, mode=SplitMode.SEMI_SUPERVISED
            )

    def test_validate_catches_overlap(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.SEMI_SUPERVISED)
        # Corrupt: duplicate an index
        splits["val_benign"].indices = np.append(
            splits["val_benign"].indices, splits["train_benign"].indices[0]
        )
        splits["val_benign"].n_samples += 1

        with pytest.raises(ValueError, match="overlap"):
            manager.validate_splits(
                splits, store_100_50, mode=SplitMode.SEMI_SUPERVISED
            )


class TestSemiSupervisedInvalidFractions:
    def test_benign_fractions_exceed_one(self, store_100_50, manager):
        with pytest.raises(ValueError, match="Invalid benign fractions"):
            manager.create_splits(
                store_100_50,
                mode=SplitMode.SEMI_SUPERVISED,
                benign_train_frac=0.80,
                benign_val_frac=0.30,
            )

    def test_jailbreak_fractions_exceed_one(self, store_100_50, manager):
        with pytest.raises(ValueError, match="Invalid jailbreak fractions"):
            manager.create_splits(
                store_100_50,
                mode=SplitMode.SEMI_SUPERVISED,
                jailbreak_train_frac=0.60,
                jailbreak_val_frac=0.50,
            )

    def test_negative_jailbreak_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(
                store_100_50,
                mode=SplitMode.SEMI_SUPERVISED,
                jailbreak_train_frac=-0.1,
                jailbreak_val_frac=0.15,
            )


class TestSemiSupervisedPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.SEMI_SUPERVISED)

        path = tmp_path / "splits_semi.json"
        manager.save_splits(splits, path, mode=SplitMode.SEMI_SUPERVISED)

        loaded = SplitManager.load_splits(path)

        assert set(loaded.keys()) == set(splits.keys())
        for name in splits:
            np.testing.assert_array_equal(loaded[name].indices, splits[name].indices)
            assert loaded[name].n_samples == splits[name].n_samples
            assert loaded[name].contains_jailbreaks == splits[name].contains_jailbreaks
            assert loaded[name].name == splits[name].name

    def test_load_mode_semi_supervised(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.SEMI_SUPERVISED)
        path = tmp_path / "splits_semi.json"
        manager.save_splits(splits, path, mode=SplitMode.SEMI_SUPERVISED)

        loaded_mode = SplitManager.load_mode(path)
        assert loaded_mode == SplitMode.SEMI_SUPERVISED

    def test_loaded_splits_validate(self, tmp_path, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.SEMI_SUPERVISED)
        path = tmp_path / "splits_semi.json"
        manager.save_splits(splits, path, mode=SplitMode.SEMI_SUPERVISED)

        loaded = SplitManager.load_splits(path)
        manager.validate_splits(loaded, store_100_50, mode=SplitMode.SEMI_SUPERVISED)


# ===========================================================================
# SplitMode enum
# ===========================================================================


class TestSplitMode:
    def test_values(self):
        assert SplitMode.UNSUPERVISED.value == "unsupervised"
        assert SplitMode.SEMI_SUPERVISED.value == "semi_supervised"

    def test_is_str_enum(self):
        assert isinstance(SplitMode.UNSUPERVISED, str)
        assert SplitMode.SEMI_SUPERVISED == "semi_supervised"

    def test_from_string(self):
        assert SplitMode("unsupervised") == SplitMode.UNSUPERVISED
        assert SplitMode("semi_supervised") == SplitMode.SEMI_SUPERVISED
