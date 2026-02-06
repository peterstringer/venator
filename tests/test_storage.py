"""Tests for HDF5 activation storage.

Verifies create/append/read roundtrip, batch vs sequential consistency,
metadata preservation, label splitting, edge cases, and performance with
larger datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from venator.activation.storage import ActivationStore

LAYERS = [12, 14, 16]
HIDDEN_DIM = 64
MODEL_ID = "test-model/mock-7b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def store_path(tmp_path) -> str:
    """Return a path (not yet existing) for a new store."""
    return str(tmp_path / "test_activations.h5")


@pytest.fixture
def empty_store(store_path) -> ActivationStore:
    """Create a fresh empty store."""
    return ActivationStore.create(store_path, MODEL_ID, LAYERS, HIDDEN_DIM)


def _make_activations(
    rng: np.random.Generator, n: int, layers: list[int] = LAYERS, dim: int = HIDDEN_DIM
) -> dict[int, np.ndarray]:
    """Generate random activations shaped (n, dim) for each layer."""
    return {l: rng.standard_normal((n, dim)).astype(np.float32) for l in layers}


def _make_single_activations(
    rng: np.random.Generator, layers: list[int] = LAYERS, dim: int = HIDDEN_DIM
) -> dict[int, np.ndarray]:
    """Generate random 1-D activations for a single prompt."""
    return {l: rng.standard_normal(dim).astype(np.float32) for l in layers}


# ---------------------------------------------------------------------------
# Creation tests
# ---------------------------------------------------------------------------


class TestCreate:
    def test_creates_file(self, store_path):
        store = ActivationStore.create(store_path, MODEL_ID, LAYERS, HIDDEN_DIM)
        assert store.path.exists()

    def test_initial_metadata(self, empty_store):
        assert empty_store.n_prompts == 0
        assert empty_store.layers == sorted(LAYERS)
        assert empty_store.hidden_dim == HIDDEN_DIM
        assert empty_store.model_id == MODEL_ID

    def test_layers_are_sorted(self, store_path):
        store = ActivationStore.create(store_path, MODEL_ID, [20, 12, 16], HIDDEN_DIM)
        assert store.layers == [12, 16, 20]

    def test_duplicate_path_raises(self, empty_store):
        with pytest.raises(FileExistsError):
            ActivationStore.create(empty_store.path, MODEL_ID, LAYERS, HIDDEN_DIM)

    def test_empty_layers_raises(self, store_path):
        with pytest.raises(ValueError, match="non-empty"):
            ActivationStore.create(store_path, MODEL_ID, [], HIDDEN_DIM)

    def test_zero_hidden_dim_raises(self, store_path):
        with pytest.raises(ValueError, match="hidden_dim"):
            ActivationStore.create(store_path, MODEL_ID, LAYERS, 0)


# ---------------------------------------------------------------------------
# Single append + read roundtrip
# ---------------------------------------------------------------------------


class TestAppendSingle:
    def test_append_one_prompt(self, empty_store, rng):
        acts = _make_single_activations(rng)
        empty_store.append("hello world", acts, label=0)

        assert empty_store.n_prompts == 1
        assert empty_store.get_prompts() == ["hello world"]
        np.testing.assert_array_equal(empty_store.get_labels(), [0])

    def test_activation_roundtrip(self, empty_store, rng):
        acts = _make_single_activations(rng)
        empty_store.append("test prompt", acts)

        for layer_idx in LAYERS:
            stored = empty_store.get_activations(layer_idx)
            assert stored.shape == (1, HIDDEN_DIM)
            np.testing.assert_allclose(stored[0], acts[layer_idx], atol=1e-7)

    def test_append_multiple_singles(self, empty_store, rng):
        for i in range(5):
            acts = _make_single_activations(rng)
            empty_store.append(f"prompt {i}", acts, label=i % 2)

        assert empty_store.n_prompts == 5
        assert len(empty_store.get_prompts()) == 5
        labels = empty_store.get_labels()
        assert labels.shape == (5,)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1, 0])

    def test_append_jailbreak_label(self, empty_store, rng):
        acts = _make_single_activations(rng)
        empty_store.append("jailbreak attempt", acts, label=1)
        np.testing.assert_array_equal(empty_store.get_labels(), [1])


# ---------------------------------------------------------------------------
# Batch append + read roundtrip
# ---------------------------------------------------------------------------


class TestAppendBatch:
    def test_batch_append(self, empty_store, rng):
        n = 10
        acts = _make_activations(rng, n)
        prompts = [f"prompt {i}" for i in range(n)]
        labels = [0] * 7 + [1] * 3

        empty_store.append_batch(prompts, acts, labels)

        assert empty_store.n_prompts == n
        assert empty_store.get_prompts() == prompts
        np.testing.assert_array_equal(empty_store.get_labels(), labels)

    def test_batch_activation_roundtrip(self, empty_store, rng):
        n = 10
        acts = _make_activations(rng, n)
        prompts = [f"p{i}" for i in range(n)]

        empty_store.append_batch(prompts, acts)

        for layer_idx in LAYERS:
            stored = empty_store.get_activations(layer_idx)
            assert stored.shape == (n, HIDDEN_DIM)
            np.testing.assert_allclose(stored, acts[layer_idx], atol=1e-7)

    def test_default_labels_are_benign(self, empty_store, rng):
        acts = _make_activations(rng, 5)
        prompts = [f"p{i}" for i in range(5)]
        empty_store.append_batch(prompts, acts)

        np.testing.assert_array_equal(empty_store.get_labels(), [0, 0, 0, 0, 0])

    def test_multiple_batches(self, empty_store, rng):
        """Appending two batches produces correct concatenated results."""
        n1, n2 = 4, 6
        acts1 = _make_activations(rng, n1)
        acts2 = _make_activations(rng, n2)

        empty_store.append_batch([f"a{i}" for i in range(n1)], acts1, [0] * n1)
        empty_store.append_batch([f"b{i}" for i in range(n2)], acts2, [1] * n2)

        assert empty_store.n_prompts == n1 + n2

        for layer_idx in LAYERS:
            stored = empty_store.get_activations(layer_idx)
            assert stored.shape == (n1 + n2, HIDDEN_DIM)
            np.testing.assert_allclose(stored[:n1], acts1[layer_idx], atol=1e-7)
            np.testing.assert_allclose(stored[n1:], acts2[layer_idx], atol=1e-7)

    def test_empty_batch_is_noop(self, empty_store):
        empty_store.append_batch([], {})
        assert empty_store.n_prompts == 0


# ---------------------------------------------------------------------------
# Batch vs sequential consistency
# ---------------------------------------------------------------------------


class TestBatchVsSequential:
    def test_batch_matches_sequential(self, tmp_path, rng):
        """Batch append produces identical results to sequential single appends."""
        n = 8
        prompts = [f"prompt {i}" for i in range(n)]
        labels = [i % 2 for i in range(n)]
        # Generate all single activations first, then combine into batch
        single_acts_list = [_make_single_activations(rng) for _ in range(n)]
        batch_acts = {
            l: np.stack([sa[l] for sa in single_acts_list], axis=0) for l in LAYERS
        }

        # Sequential store
        seq_path = str(tmp_path / "seq.h5")
        seq_store = ActivationStore.create(seq_path, MODEL_ID, LAYERS, HIDDEN_DIM)
        for i in range(n):
            seq_store.append(prompts[i], single_acts_list[i], labels[i])

        # Batch store
        batch_path = str(tmp_path / "batch.h5")
        batch_store = ActivationStore.create(batch_path, MODEL_ID, LAYERS, HIDDEN_DIM)
        batch_store.append_batch(prompts, batch_acts, labels)

        # Compare
        assert seq_store.n_prompts == batch_store.n_prompts
        assert seq_store.get_prompts() == batch_store.get_prompts()
        np.testing.assert_array_equal(seq_store.get_labels(), batch_store.get_labels())

        for layer_idx in LAYERS:
            np.testing.assert_allclose(
                seq_store.get_activations(layer_idx),
                batch_store.get_activations(layer_idx),
                atol=1e-7,
            )


# ---------------------------------------------------------------------------
# Read with indexing
# ---------------------------------------------------------------------------


class TestIndexing:
    @pytest.fixture
    def populated_store(self, empty_store, rng):
        n = 20
        acts = _make_activations(rng, n)
        prompts = [f"prompt {i}" for i in range(n)]
        labels = [0] * 15 + [1] * 5
        empty_store.append_batch(prompts, acts, labels)
        return empty_store, acts

    def test_get_activations_by_indices(self, populated_store):
        store, acts = populated_store
        indices = [0, 5, 10]
        for layer_idx in LAYERS:
            result = store.get_activations(layer_idx, indices=indices)
            assert result.shape == (3, HIDDEN_DIM)
            np.testing.assert_allclose(result, acts[layer_idx][indices], atol=1e-7)

    def test_get_activations_by_slice(self, populated_store):
        store, acts = populated_store
        for layer_idx in LAYERS:
            result = store.get_activations(layer_idx, indices=slice(2, 7))
            assert result.shape == (5, HIDDEN_DIM)
            np.testing.assert_allclose(result, acts[layer_idx][2:7], atol=1e-7)

    def test_get_prompts_by_indices(self, populated_store):
        store, _ = populated_store
        result = store.get_prompts(indices=[0, 1, 19])
        assert result == ["prompt 0", "prompt 1", "prompt 19"]

    def test_get_prompts_by_slice(self, populated_store):
        store, _ = populated_store
        result = store.get_prompts(indices=slice(0, 3))
        assert result == ["prompt 0", "prompt 1", "prompt 2"]

    def test_missing_layer_raises(self, populated_store):
        store, _ = populated_store
        with pytest.raises(KeyError):
            store.get_activations(999)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_persists_across_reopen(self, store_path, rng):
        store = ActivationStore.create(store_path, MODEL_ID, LAYERS, HIDDEN_DIM)
        acts = _make_activations(rng, 3)
        store.append_batch(["a", "b", "c"], acts)

        # Re-open from path
        reopened = ActivationStore(store_path)
        assert reopened.n_prompts == 3
        assert reopened.layers == sorted(LAYERS)
        assert reopened.hidden_dim == HIDDEN_DIM
        assert reopened.model_id == MODEL_ID

    def test_n_prompts_updates_after_appends(self, empty_store, rng):
        assert empty_store.n_prompts == 0

        acts = _make_single_activations(rng)
        empty_store.append("p1", acts)
        assert empty_store.n_prompts == 1

        acts = _make_activations(rng, 3)
        empty_store.append_batch(["p2", "p3", "p4"], acts)
        assert empty_store.n_prompts == 4


# ---------------------------------------------------------------------------
# split_by_labels
# ---------------------------------------------------------------------------


class TestSplitByLabels:
    def test_split(self, empty_store, rng):
        n = 10
        acts = _make_activations(rng, n)
        labels = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        empty_store.append_batch([f"p{i}" for i in range(n)], acts, labels)

        benign, jailbreak = empty_store.split_by_labels()
        np.testing.assert_array_equal(benign, [0, 1, 3, 6, 7, 9])
        np.testing.assert_array_equal(jailbreak, [2, 4, 5, 8])

    def test_split_all_benign(self, empty_store, rng):
        acts = _make_activations(rng, 5)
        empty_store.append_batch([f"p{i}" for i in range(5)], acts)

        benign, jailbreak = empty_store.split_by_labels()
        assert len(benign) == 5
        assert len(jailbreak) == 0

    def test_split_empty_store(self, empty_store):
        benign, jailbreak = empty_store.split_by_labels()
        assert len(benign) == 0
        assert len(jailbreak) == 0


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_layer_in_append_raises(self, empty_store, rng):
        # Only provide 2 of 3 required layers
        acts = {l: rng.standard_normal((1, HIDDEN_DIM)).astype(np.float32) for l in LAYERS[:2]}
        with pytest.raises(KeyError, match="Missing activations"):
            empty_store.append_batch(["p"], acts)

    def test_wrong_shape_raises(self, empty_store, rng):
        acts = _make_activations(rng, 3)
        # Corrupt one layer's shape
        acts[LAYERS[0]] = rng.standard_normal((2, HIDDEN_DIM)).astype(np.float32)
        with pytest.raises(ValueError, match="expected shape"):
            empty_store.append_batch(["a", "b", "c"], acts)

    def test_nan_activations_raise(self, empty_store):
        acts = {l: np.full((1, HIDDEN_DIM), np.nan, dtype=np.float32) for l in LAYERS}
        with pytest.raises(ValueError, match="NaN or Inf"):
            empty_store.append_batch(["bad"], acts)

    def test_inf_activations_raise(self, empty_store):
        acts = {l: np.full((1, HIDDEN_DIM), np.inf, dtype=np.float32) for l in LAYERS}
        with pytest.raises(ValueError, match="NaN or Inf"):
            empty_store.append_batch(["bad"], acts)

    def test_labels_length_mismatch_raises(self, empty_store, rng):
        acts = _make_activations(rng, 3)
        with pytest.raises(ValueError, match="labels length"):
            empty_store.append_batch(["a", "b", "c"], acts, labels=[0, 1])


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------


class TestDunder:
    def test_len(self, empty_store, rng):
        assert len(empty_store) == 0
        acts = _make_activations(rng, 5)
        empty_store.append_batch([f"p{i}" for i in range(5)], acts)
        assert len(empty_store) == 5

    def test_repr_empty(self, empty_store):
        r = repr(empty_store)
        assert "n=0" in r
        assert "dim=64" in r

    def test_repr_not_created(self, tmp_path):
        store = ActivationStore(tmp_path / "nonexistent.h5")
        assert "not yet created" in repr(store)


# ---------------------------------------------------------------------------
# Performance: larger dataset
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_1000_entries(self, tmp_path, rng):
        """Verify correctness with 1000+ entries across multiple batch appends."""
        path = str(tmp_path / "large.h5")
        store = ActivationStore.create(path, MODEL_ID, LAYERS, HIDDEN_DIM)

        total = 0
        all_acts: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}

        # Append in batches of varying sizes
        for batch_size in [100, 250, 150, 300, 200]:
            acts = _make_activations(rng, batch_size)
            prompts = [f"p{total + i}" for i in range(batch_size)]
            labels = [0 if (total + i) % 3 != 0 else 1 for i in range(batch_size)]
            store.append_batch(prompts, acts, labels)

            for l in LAYERS:
                all_acts[l].append(acts[l])

            total += batch_size

        assert store.n_prompts == 1000
        assert len(store) == 1000

        # Verify all data is correctly stored
        for layer_idx in LAYERS:
            expected = np.concatenate(all_acts[layer_idx], axis=0)
            stored = store.get_activations(layer_idx)
            assert stored.shape == (1000, HIDDEN_DIM)
            np.testing.assert_allclose(stored, expected, atol=1e-7)

        # Verify prompts
        prompts = store.get_prompts()
        assert len(prompts) == 1000
        assert prompts[0] == "p0"
        assert prompts[999] == "p999"

        # Verify labels
        labels = store.get_labels()
        assert labels.shape == (1000,)
        benign, jailbreak = store.split_by_labels()
        assert len(benign) + len(jailbreak) == 1000

    def test_indexed_read_on_large_store(self, tmp_path, rng):
        """Indexed reads return correct subsets from a large store."""
        path = str(tmp_path / "indexed.h5")
        store = ActivationStore.create(path, MODEL_ID, LAYERS, HIDDEN_DIM)

        n = 500
        acts = _make_activations(rng, n)
        prompts = [f"prompt_{i}" for i in range(n)]
        store.append_batch(prompts, acts)

        # Read specific indices
        indices = [0, 99, 250, 499]
        for layer_idx in LAYERS:
            result = store.get_activations(layer_idx, indices=indices)
            assert result.shape == (4, HIDDEN_DIM)
            for j, idx in enumerate(indices):
                np.testing.assert_allclose(result[j], acts[layer_idx][idx], atol=1e-7)

        # Read a slice
        for layer_idx in LAYERS:
            result = store.get_activations(layer_idx, indices=slice(100, 200))
            assert result.shape == (100, HIDDEN_DIM)
            np.testing.assert_allclose(result, acts[layer_idx][100:200], atol=1e-7)
