"""Tests for unified data splitting — verifies label integrity and methodology.

Core invariant: benign splits contain ONLY benign prompts, jailbreak splits
contain ONLY jailbreak prompts. The unified split always produces 6 keys:
train_benign, train_jailbreak, val_benign, val_jailbreak, test_benign,
test_jailbreak.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from venator.activation.storage import ActivationStore
from venator.data.splits import (
    DataSplit,
    SplitManager,
    SplitMode,
    UnifiedSplit,
    create_unified_split,
)

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
# LABEL INTEGRITY — the most critical tests
# ===========================================================================


class TestLabelIntegrity:
    """Verify benign splits are benign-only and jailbreak splits are jailbreak-only."""

    def test_benign_splits_have_only_benign(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        labels = store_500_200.get_labels()

        for split_name in ("train_benign", "val_benign", "test_benign"):
            split_labels = labels[splits[split_name].indices]
            assert np.all(split_labels == 0), (
                f"{split_name} contains jailbreak prompts!"
            )

    def test_jailbreak_splits_have_only_jailbreaks(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        labels = store_500_200.get_labels()

        for split_name in ("train_jailbreak", "val_jailbreak", "test_jailbreak"):
            split_labels = labels[splits[split_name].indices]
            assert np.all(split_labels == 1), (
                f"{split_name} contains benign prompts!"
            )

    def test_test_jailbreaks_never_in_train(self, store_500_200, manager):
        """No jailbreak index appears in both train and test."""
        splits = manager.create_splits(store_500_200)

        train_jb = set(splits["train_jailbreak"].indices.tolist())
        test_jb = set(splits["test_jailbreak"].indices.tolist())
        assert train_jb.isdisjoint(test_jb), (
            "Jailbreak indices overlap between train and test!"
        )

    def test_six_splits_present(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)
        expected_keys = {
            "train_benign", "train_jailbreak",
            "val_benign", "val_jailbreak",
            "test_benign", "test_jailbreak",
        }
        assert set(splits.keys()) == expected_keys

    def test_label_integrity_across_seeds(self, store_100_50):
        """Regardless of seed, methodology holds."""
        labels = store_100_50.get_labels()
        for seed in (0, 1, 42, 99, 12345):
            splits = SplitManager(seed=seed).create_splits(store_100_50)
            assert np.all(labels[splits["train_benign"].indices] == 0)
            assert np.all(labels[splits["val_benign"].indices] == 0)
            for name in ("train_jailbreak", "val_jailbreak", "test_jailbreak"):
                if splits[name].n_samples > 0:
                    assert np.all(labels[splits[name].indices] == 1)


# ---------------------------------------------------------------------------
# Correct proportions
# ---------------------------------------------------------------------------


class TestProportions:
    def test_default_fractions(self, store_500_200, manager):
        splits = manager.create_splits(store_500_200)

        n_benign, n_jailbreak = 500, 200

        # Benign splits (70/15/15)
        assert splits["train_benign"].n_samples == round(n_benign * 0.70)
        assert splits["val_benign"].n_samples == round(n_benign * 0.15)
        expected_test_b = (
            n_benign
            - splits["train_benign"].n_samples
            - splits["val_benign"].n_samples
        )
        assert splits["test_benign"].n_samples == expected_test_b

        # Jailbreak splits (15/15/70)
        assert splits["train_jailbreak"].n_samples == round(n_jailbreak * 0.15)
        assert splits["val_jailbreak"].n_samples == round(n_jailbreak * 0.15)
        expected_test_j = (
            n_jailbreak
            - splits["train_jailbreak"].n_samples
            - splits["val_jailbreak"].n_samples
        )
        assert splits["test_jailbreak"].n_samples == expected_test_j

    def test_custom_benign_fractions(self, store_100_50, manager):
        splits = manager.create_splits(
            store_100_50,
            benign_train_frac=0.60,
            benign_val_frac=0.20,
        )

        n_benign = 100
        assert splits["train_benign"].n_samples == round(n_benign * 0.60)
        assert splits["val_benign"].n_samples == round(n_benign * 0.20)
        expected_test = (
            n_benign
            - splits["train_benign"].n_samples
            - splits["val_benign"].n_samples
        )
        assert splits["test_benign"].n_samples == expected_test

    def test_custom_jailbreak_fractions(self, store_500_200, manager):
        splits = manager.create_splits(
            store_500_200,
            jailbreak_train_frac=0.10,
            jailbreak_val_frac=0.10,
        )
        n_jailbreak = 200
        assert splits["train_jailbreak"].n_samples == round(n_jailbreak * 0.10)
        assert splits["val_jailbreak"].n_samples == round(n_jailbreak * 0.10)
        expected_test = (
            n_jailbreak
            - splits["train_jailbreak"].n_samples
            - splits["val_jailbreak"].n_samples
        )
        assert splits["test_jailbreak"].n_samples == expected_test

    def test_majority_jailbreaks_reserved_for_test(self, store_500_200, manager):
        """At least 60% of jailbreaks end up in test_jailbreak with defaults."""
        splits = manager.create_splits(store_500_200)
        _, jb_idx = store_500_200.split_by_labels()
        n_jailbreak = len(jb_idx)
        assert splits["test_jailbreak"].n_samples >= n_jailbreak * 0.60

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
        assert not np.array_equal(
            s1["train_benign"].indices, s2["train_benign"].indices
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_minimum_benign_samples(self, tmp_path):
        """3 benign + some jailbreaks: each benign split gets at least 1."""
        store = _create_store(tmp_path, 3, 5)
        splits = SplitManager(seed=42).create_splits(store)

        assert splits["train_benign"].n_samples >= 1
        assert splits["val_benign"].n_samples >= 1
        assert splits["test_benign"].n_samples >= 1

        # Still no leakage
        labels = store.get_labels()
        assert np.all(labels[splits["train_benign"].indices] == 0)
        assert np.all(labels[splits["val_benign"].indices] == 0)

    def test_too_few_benign_raises(self, tmp_path):
        store = _create_store(tmp_path, 2, 5)
        with pytest.raises(ValueError, match="Too few benign"):
            SplitManager(seed=42).create_splits(store)

    def test_no_benign_raises(self, tmp_path):
        store = _create_store(tmp_path, 0, 10)
        with pytest.raises(ValueError, match="[Nn]o benign prompts"):
            SplitManager(seed=42).create_splits(store)

    def test_no_jailbreaks(self, tmp_path):
        """All benign — jailbreak splits should be empty."""
        store = _create_store(tmp_path, 100, 0)
        splits = SplitManager(seed=42).create_splits(store)

        assert splits["test_jailbreak"].n_samples == 0
        assert splits["train_jailbreak"].n_samples == 0
        assert splits["val_jailbreak"].n_samples == 0
        total = sum(s.n_samples for s in splits.values())
        assert total == 100

    def test_many_jailbreaks_few_benign(self, tmp_path):
        """10 benign + 500 jailbreaks — extreme ratio."""
        store = _create_store(tmp_path, 10, 500)
        splits = SplitManager(seed=42).create_splits(store)

        labels = store.get_labels()
        assert np.all(labels[splits["train_benign"].indices] == 0)
        assert np.all(labels[splits["val_benign"].indices] == 0)

    def test_few_jailbreaks(self, tmp_path):
        """With only 3 jailbreaks and default 15/15/70, we should still work."""
        store = _create_store(tmp_path, 50, 3)
        splits = SplitManager(seed=42).create_splits(store)

        # At least 1 jailbreak in test
        assert splits["test_jailbreak"].n_samples >= 1

        # Total coverage
        total = sum(s.n_samples for s in splits.values())
        assert total == store.n_prompts

    def test_minimum_benign_with_jailbreaks(self, tmp_path):
        """3 benign + 10 jailbreaks — each benign split gets at least 1."""
        store = _create_store(tmp_path, 3, 10)
        splits = SplitManager(seed=42).create_splits(store)

        assert splits["train_benign"].n_samples >= 1
        assert splits["val_benign"].n_samples >= 1
        assert splits["test_benign"].n_samples >= 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_passes_for_correct_splits(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Should not raise
        manager.validate_splits(splits, store_100_50)

    def test_validate_catches_jailbreak_in_benign_split(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Corrupt: swap a train_benign index with a jailbreak index
        jb_idx = splits["test_jailbreak"].indices[0]
        splits["train_benign"].indices[0] = jb_idx
        splits["test_jailbreak"] = DataSplit(
            name="test_jailbreak",
            indices=splits["test_jailbreak"].indices[1:],
            n_samples=splits["test_jailbreak"].n_samples - 1,
            contains_jailbreaks=True,
        )

        with pytest.raises(ValueError, match="LABEL MISMATCH"):
            manager.validate_splits(splits, store_100_50)

    def test_validate_catches_benign_in_jailbreak_split(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Corrupt: put a benign index into train_jailbreak
        benign_idx = splits["train_benign"].indices[0]
        splits["train_jailbreak"].indices = np.append(
            splits["train_jailbreak"].indices, benign_idx
        )
        splits["train_jailbreak"].n_samples += 1
        splits["train_benign"].indices = splits["train_benign"].indices[1:]
        splits["train_benign"].n_samples -= 1

        with pytest.raises(ValueError, match="LABEL MISMATCH"):
            manager.validate_splits(splits, store_100_50)

    def test_validate_catches_index_overlap(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Corrupt: duplicate an index into val_benign
        splits["val_benign"].indices = np.append(
            splits["val_benign"].indices, splits["train_benign"].indices[0]
        )
        splits["val_benign"].n_samples += 1

        with pytest.raises(ValueError, match="overlap"):
            manager.validate_splits(splits, store_100_50)

    def test_validate_catches_missing_coverage(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50)
        # Corrupt: remove an index from train_benign
        splits["train_benign"].indices = splits["train_benign"].indices[1:]
        splits["train_benign"].n_samples -= 1

        with pytest.raises(ValueError, match="cover"):
            manager.validate_splits(splits, store_100_50)


# ---------------------------------------------------------------------------
# Invalid fractions
# ---------------------------------------------------------------------------


class TestInvalidFractions:
    def test_benign_fractions_exceed_one(self, store_100_50, manager):
        with pytest.raises(ValueError, match="Invalid benign fractions"):
            manager.create_splits(
                store_100_50,
                benign_train_frac=0.80,
                benign_val_frac=0.30,
            )

    def test_negative_benign_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(
                store_100_50,
                benign_train_frac=-0.1,
                benign_val_frac=0.5,
            )

    def test_zero_benign_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(
                store_100_50,
                benign_train_frac=0.0,
                benign_val_frac=0.5,
            )

    def test_jailbreak_fractions_exceed_one(self, store_100_50, manager):
        with pytest.raises(ValueError, match="Invalid jailbreak fractions"):
            manager.create_splits(
                store_100_50,
                jailbreak_train_frac=0.60,
                jailbreak_val_frac=0.50,
            )

    def test_negative_jailbreak_fraction(self, store_100_50, manager):
        with pytest.raises(ValueError):
            manager.create_splits(
                store_100_50,
                jailbreak_train_frac=-0.1,
                jailbreak_val_frac=0.15,
            )

    def test_too_much_jailbreak_in_train_val_raises(self, store_100_50, manager):
        """Jailbreak test fraction must be >= 50%."""
        with pytest.raises(ValueError, match="50%"):
            manager.create_splits(
                store_100_50,
                jailbreak_train_frac=0.30,
                jailbreak_val_frac=0.25,
            )


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

    def test_load_mode_unified(self, tmp_path, store_100_50, manager):
        """New saves always use UNIFIED mode."""
        splits = manager.create_splits(store_100_50)
        path = tmp_path / "splits.json"
        manager.save_splits(splits, path)

        loaded_mode = SplitManager.load_mode(path)
        assert loaded_mode == SplitMode.UNIFIED

    def test_load_mode_defaults_for_legacy_files(self, tmp_path, store_100_50, manager):
        """Files saved without a mode field default to UNSUPERVISED."""
        splits = manager.create_splits(store_100_50)
        path = tmp_path / "splits.json"
        manager.save_splits(splits, path)

        # Manually remove the mode key to simulate legacy format
        with open(path, "r") as f:
            data = json.load(f)
        del data["mode"]
        with open(path, "w") as f:
            json.dump(data, f)

        loaded_mode = SplitManager.load_mode(path)
        assert loaded_mode == SplitMode.UNSUPERVISED


# ---------------------------------------------------------------------------
# Legacy format migration
# ---------------------------------------------------------------------------


class TestLegacyMigration:
    """Verify old 4-key unsupervised format is automatically migrated to 6-key."""

    def test_old_4key_format_migrated(self, tmp_path, store_100_50, manager):
        """Simulate a legacy unsupervised split file and verify migration."""
        splits = manager.create_splits(store_100_50)

        # Save in old 4-key format
        path = tmp_path / "legacy_splits.json"
        old_data = {
            "seed": 42,
            "mode": "unsupervised",
            "splits": {
                "train": {
                    "name": "train",
                    "indices": splits["train_benign"].indices.tolist(),
                    "n_samples": splits["train_benign"].n_samples,
                    "contains_jailbreaks": False,
                },
                "val": {
                    "name": "val",
                    "indices": splits["val_benign"].indices.tolist(),
                    "n_samples": splits["val_benign"].n_samples,
                    "contains_jailbreaks": False,
                },
                "test_benign": {
                    "name": "test_benign",
                    "indices": splits["test_benign"].indices.tolist(),
                    "n_samples": splits["test_benign"].n_samples,
                    "contains_jailbreaks": False,
                },
                "test_jailbreak": {
                    "name": "test_jailbreak",
                    "indices": splits["test_jailbreak"].indices.tolist(),
                    "n_samples": splits["test_jailbreak"].n_samples,
                    "contains_jailbreaks": True,
                },
            },
        }
        with open(path, "w") as f:
            json.dump(old_data, f)

        loaded = SplitManager.load_splits(path)

        # Should have 6 keys after migration
        assert set(loaded.keys()) == {
            "train_benign", "train_jailbreak",
            "val_benign", "val_jailbreak",
            "test_benign", "test_jailbreak",
        }

        # Migrated keys should match original data
        np.testing.assert_array_equal(
            loaded["train_benign"].indices, splits["train_benign"].indices
        )
        np.testing.assert_array_equal(
            loaded["val_benign"].indices, splits["val_benign"].indices
        )

        # New jailbreak splits should be empty
        assert loaded["train_jailbreak"].n_samples == 0
        assert loaded["val_jailbreak"].n_samples == 0


# ---------------------------------------------------------------------------
# Backward compatibility — mode parameter is accepted but ignored
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Verify that passing mode= still works (backward compat)."""

    def test_mode_unsupervised_still_produces_six_keys(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.UNSUPERVISED)
        assert set(splits.keys()) == {
            "train_benign", "train_jailbreak",
            "val_benign", "val_jailbreak",
            "test_benign", "test_jailbreak",
        }

    def test_mode_semi_supervised_still_produces_six_keys(self, store_100_50, manager):
        splits = manager.create_splits(store_100_50, mode=SplitMode.SEMI_SUPERVISED)
        assert set(splits.keys()) == {
            "train_benign", "train_jailbreak",
            "val_benign", "val_jailbreak",
            "test_benign", "test_jailbreak",
        }

    def test_mode_parameter_does_not_affect_output(self, store_100_50):
        """Same seed + same store produces identical splits regardless of mode."""
        s1 = SplitManager(seed=42).create_splits(
            store_100_50, mode=SplitMode.UNSUPERVISED
        )
        s2 = SplitManager(seed=42).create_splits(
            store_100_50, mode=SplitMode.SEMI_SUPERVISED
        )
        s3 = SplitManager(seed=42).create_splits(
            store_100_50, mode=SplitMode.UNIFIED
        )

        for name in s1:
            np.testing.assert_array_equal(s1[name].indices, s2[name].indices)
            np.testing.assert_array_equal(s1[name].indices, s3[name].indices)

    def test_save_with_mode_param_still_works(self, tmp_path, store_100_50, manager):
        """save_splits accepts mode= for backward compat but always saves unified."""
        splits = manager.create_splits(store_100_50)
        path = tmp_path / "splits.json"
        manager.save_splits(splits, path, mode=SplitMode.UNSUPERVISED)

        loaded_mode = SplitManager.load_mode(path)
        assert loaded_mode == SplitMode.UNIFIED  # Always unified regardless of param

    def test_validate_with_mode_param_still_works(self, store_100_50, manager):
        """validate_splits accepts mode= for backward compat."""
        splits = manager.create_splits(store_100_50)
        # Should not raise
        manager.validate_splits(splits, store_100_50, mode=SplitMode.UNSUPERVISED)
        manager.validate_splits(splits, store_100_50, mode=SplitMode.SEMI_SUPERVISED)


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
# UnifiedSplit dataclass and create_unified_split()
# ===========================================================================


class TestUnifiedSplit:
    def test_to_split_dict_produces_six_keys(self):
        split = UnifiedSplit(
            train_benign_indices=np.array([0, 1, 2]),
            train_jailbreak_indices=np.array([10, 11]),
            val_benign_indices=np.array([3, 4]),
            val_jailbreak_indices=np.array([12]),
            test_benign_indices=np.array([5, 6, 7]),
            test_jailbreak_indices=np.array([13, 14, 15]),
        )
        d = split.to_split_dict()
        assert set(d.keys()) == {
            "train_benign", "train_jailbreak",
            "val_benign", "val_jailbreak",
            "test_benign", "test_jailbreak",
        }

    def test_to_split_dict_preserves_indices(self):
        indices = np.array([10, 20, 30])
        split = UnifiedSplit(
            train_benign_indices=indices,
            train_jailbreak_indices=np.array([]),
            val_benign_indices=np.array([40]),
            val_jailbreak_indices=np.array([]),
            test_benign_indices=np.array([50]),
            test_jailbreak_indices=np.array([]),
        )
        d = split.to_split_dict()
        np.testing.assert_array_equal(d["train_benign"].indices, indices)
        assert d["train_benign"].n_samples == 3
        assert d["train_benign"].contains_jailbreaks is False
        assert d["train_jailbreak"].contains_jailbreaks is True

    def test_post_init_converts_to_int64(self):
        split = UnifiedSplit(
            train_benign_indices=[0, 1],
            train_jailbreak_indices=[10],
            val_benign_indices=[2],
            val_jailbreak_indices=[11],
            test_benign_indices=[3],
            test_jailbreak_indices=[12],
        )
        assert split.train_benign_indices.dtype == np.int64
        assert split.train_jailbreak_indices.dtype == np.int64


class TestCreateUnifiedSplit:
    def test_basic_split(self):
        benign = np.arange(100)
        jailbreak = np.arange(100, 150)
        split = create_unified_split(benign, jailbreak)

        d = split.to_split_dict()
        total = sum(s.n_samples for s in d.values())
        assert total == 150

    def test_no_jailbreaks(self):
        benign = np.arange(100)
        jailbreak = np.array([], dtype=np.int64)
        split = create_unified_split(benign, jailbreak)

        assert len(split.train_jailbreak_indices) == 0
        assert len(split.val_jailbreak_indices) == 0
        assert len(split.test_jailbreak_indices) == 0

    def test_custom_fractions(self):
        benign = np.arange(100)
        jailbreak = np.arange(100, 200)
        split = create_unified_split(
            benign, jailbreak,
            benign_train_frac=0.60,
            benign_val_frac=0.20,
            jailbreak_train_frac=0.10,
            jailbreak_val_frac=0.10,
        )
        assert len(split.train_benign_indices) == 60
        assert len(split.val_benign_indices) == 20
        assert len(split.train_jailbreak_indices) == 10
        assert len(split.val_jailbreak_indices) == 10

    def test_no_benign_raises(self):
        with pytest.raises(ValueError, match="[Nn]o benign"):
            create_unified_split(
                np.array([], dtype=np.int64),
                np.array([1, 2, 3]),
            )

    def test_invalid_benign_fractions_raises(self):
        with pytest.raises(ValueError, match="Invalid benign fractions"):
            create_unified_split(
                np.arange(100),
                np.arange(100, 150),
                benign_train_frac=0.80,
                benign_val_frac=0.30,
            )

    def test_jailbreak_test_too_small_raises(self):
        with pytest.raises(ValueError, match="50%"):
            create_unified_split(
                np.arange(100),
                np.arange(100, 200),
                jailbreak_train_frac=0.30,
                jailbreak_val_frac=0.25,
            )

    def test_reproducible_with_same_seed(self):
        benign = np.arange(100)
        jailbreak = np.arange(100, 150)
        s1 = create_unified_split(benign, jailbreak, seed=42)
        s2 = create_unified_split(benign, jailbreak, seed=42)
        np.testing.assert_array_equal(
            s1.train_benign_indices, s2.train_benign_indices
        )

    def test_different_seed_different_result(self):
        benign = np.arange(100)
        jailbreak = np.arange(100, 150)
        s1 = create_unified_split(benign, jailbreak, seed=42)
        s2 = create_unified_split(benign, jailbreak, seed=99)
        assert not np.array_equal(
            s1.train_benign_indices, s2.train_benign_indices
        )


# ===========================================================================
# SplitMode enum
# ===========================================================================


class TestSplitMode:
    def test_values(self):
        assert SplitMode.UNSUPERVISED.value == "unsupervised"
        assert SplitMode.SEMI_SUPERVISED.value == "semi_supervised"
        assert SplitMode.UNIFIED.value == "unified"

    def test_is_str_enum(self):
        assert isinstance(SplitMode.UNSUPERVISED, str)
        assert SplitMode.SEMI_SUPERVISED == "semi_supervised"
        assert SplitMode.UNIFIED == "unified"

    def test_from_string(self):
        assert SplitMode("unsupervised") == SplitMode.UNSUPERVISED
        assert SplitMode("semi_supervised") == SplitMode.SEMI_SUPERVISED
        assert SplitMode("unified") == SplitMode.UNIFIED
