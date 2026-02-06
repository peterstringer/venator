"""Tests for prompt dataset management.

Tests the PromptDataset class (construction, filtering, JSONL persistence,
merge) and the collection helper functions (deduplication, shuffling,
built-in prompt lists). Does NOT test live HuggingFace downloads — those
are integration tests that require network access.
"""

from __future__ import annotations

import json

import pytest

from venator.data.prompts import (
    PromptDataset,
    _deduplicate_and_shuffle,
    _get_dan_style_jailbreaks,
    _get_diverse_prompts,
    collect_benign_prompts,
    collect_jailbreak_prompts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> PromptDataset:
    """A small mixed dataset for testing."""
    return PromptDataset(
        prompts=["hello", "attack prompt", "goodbye", "another attack"],
        labels=[0, 1, 0, 1],
        sources=["diverse", "dan_style", "alpaca", "advbench"],
    )


@pytest.fixture
def benign_dataset() -> PromptDataset:
    """A pure benign dataset."""
    return PromptDataset(
        prompts=["prompt a", "prompt b", "prompt c"],
        labels=[0, 0, 0],
        sources=["alpaca", "mmlu", "diverse"],
    )


# ---------------------------------------------------------------------------
# PromptDataset construction
# ---------------------------------------------------------------------------


class TestPromptDatasetInit:
    def test_basic_construction(self):
        ds = PromptDataset(
            prompts=["a", "b"],
            labels=[0, 1],
            sources=["s1", "s2"],
        )
        assert len(ds) == 2
        assert ds.prompts == ["a", "b"]
        assert ds.labels == [0, 1]
        assert ds.sources == ["s1", "s2"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            PromptDataset(
                prompts=["a", "b"],
                labels=[0],
                sources=["s1", "s2"],
            )

    def test_empty_dataset(self):
        ds = PromptDataset(prompts=[], labels=[], sources=[])
        assert len(ds) == 0
        assert ds.n_benign == 0
        assert ds.n_jailbreaks == 0


# ---------------------------------------------------------------------------
# Filtered views
# ---------------------------------------------------------------------------


class TestFilteredViews:
    def test_benign(self, sample_dataset):
        assert sample_dataset.benign == ["hello", "goodbye"]

    def test_jailbreaks(self, sample_dataset):
        assert sample_dataset.jailbreaks == ["attack prompt", "another attack"]

    def test_n_benign(self, sample_dataset):
        assert sample_dataset.n_benign == 2

    def test_n_jailbreaks(self, sample_dataset):
        assert sample_dataset.n_jailbreaks == 2

    def test_all_benign(self, benign_dataset):
        assert benign_dataset.n_benign == 3
        assert benign_dataset.n_jailbreaks == 0
        assert benign_dataset.jailbreaks == []

    def test_source_counts(self, sample_dataset):
        counts = sample_dataset.source_counts()
        assert counts == {
            "diverse": 1,
            "dan_style": 1,
            "alpaca": 1,
            "advbench": 1,
        }


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


class TestJsonlPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, sample_dataset):
        path = tmp_path / "test.jsonl"
        sample_dataset.save(path)

        loaded = PromptDataset.load(path)
        assert loaded.prompts == sample_dataset.prompts
        assert loaded.labels == sample_dataset.labels
        assert loaded.sources == sample_dataset.sources

    def test_save_creates_parent_dirs(self, tmp_path, sample_dataset):
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        sample_dataset.save(path)
        assert path.exists()

    def test_jsonl_format(self, tmp_path, sample_dataset):
        """Each line is valid JSON with the expected keys."""
        path = tmp_path / "test.jsonl"
        sample_dataset.save(path)

        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == len(sample_dataset)
        for line in lines:
            record = json.loads(line)
            assert "prompt" in record
            assert "label" in record
            assert "source" in record
            assert isinstance(record["label"], int)

    def test_unicode_roundtrip(self, tmp_path):
        ds = PromptDataset(
            prompts=["Héllo wörld", "日本語テスト", "emoji 🎉"],
            labels=[0, 0, 0],
            sources=["test", "test", "test"],
        )
        path = tmp_path / "unicode.jsonl"
        ds.save(path)

        loaded = PromptDataset.load(path)
        assert loaded.prompts == ds.prompts

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptDataset.load(tmp_path / "nonexistent.jsonl")

    def test_load_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        with open(path, "w") as f:
            f.write('{"prompt": "a", "label": 0, "source": "s"}\n')
            f.write("\n")
            f.write('{"prompt": "b", "label": 1, "source": "s"}\n')
            f.write("   \n")

        loaded = PromptDataset.load(path)
        assert len(loaded) == 2

    def test_empty_dataset_roundtrip(self, tmp_path):
        ds = PromptDataset(prompts=[], labels=[], sources=[])
        path = tmp_path / "empty.jsonl"
        ds.save(path)

        loaded = PromptDataset.load(path)
        assert len(loaded) == 0


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_two(self, benign_dataset, sample_dataset):
        merged = PromptDataset.merge(benign_dataset, sample_dataset)
        assert len(merged) == len(benign_dataset) + len(sample_dataset)
        assert merged.prompts[:3] == benign_dataset.prompts
        assert merged.prompts[3:] == sample_dataset.prompts

    def test_merge_preserves_labels(self, benign_dataset, sample_dataset):
        merged = PromptDataset.merge(benign_dataset, sample_dataset)
        assert merged.labels == benign_dataset.labels + sample_dataset.labels

    def test_merge_empty(self):
        ds = PromptDataset(prompts=["a"], labels=[0], sources=["s"])
        empty = PromptDataset(prompts=[], labels=[], sources=[])
        merged = PromptDataset.merge(ds, empty)
        assert len(merged) == 1

    def test_merge_multiple(self):
        d1 = PromptDataset(["a"], [0], ["s1"])
        d2 = PromptDataset(["b"], [1], ["s2"])
        d3 = PromptDataset(["c"], [0], ["s3"])
        merged = PromptDataset.merge(d1, d2, d3)
        assert merged.prompts == ["a", "b", "c"]
        assert merged.labels == [0, 1, 0]


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------


class TestDunder:
    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 4

    def test_repr(self, sample_dataset):
        r = repr(sample_dataset)
        assert "n=4" in r
        assert "benign=2" in r
        assert "jailbreak=2" in r


# ---------------------------------------------------------------------------
# Deduplication and shuffling
# ---------------------------------------------------------------------------


class TestDeduplicateAndShuffle:
    def test_removes_duplicates(self):
        items = [("hello", "s1"), ("hello", "s2"), ("world", "s1")]
        result = _deduplicate_and_shuffle(items, seed=42)
        prompts = [p for p, _ in result]
        assert len(result) == 2
        assert "hello" in prompts
        assert "world" in prompts

    def test_removes_empty_and_whitespace(self):
        items = [("hello", "s1"), ("", "s2"), ("   ", "s3"), ("world", "s1")]
        result = _deduplicate_and_shuffle(items, seed=42)
        assert len(result) == 2

    def test_strips_whitespace(self):
        items = [("  hello  ", "s1"), ("hello", "s2")]
        result = _deduplicate_and_shuffle(items, seed=42)
        assert len(result) == 1
        assert result[0][0] == "hello"

    def test_deterministic_with_same_seed(self):
        items = [(f"prompt {i}", "s") for i in range(100)]
        r1 = _deduplicate_and_shuffle(items, seed=42)
        r2 = _deduplicate_and_shuffle(items, seed=42)
        assert r1 == r2

    def test_different_seed_gives_different_order(self):
        items = [(f"prompt {i}", "s") for i in range(100)]
        r1 = _deduplicate_and_shuffle(items, seed=42)
        r2 = _deduplicate_and_shuffle(items, seed=99)
        # Different order (could theoretically be the same, but extremely unlikely with 100 items)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Built-in prompt collections
# ---------------------------------------------------------------------------


class TestBuiltinPrompts:
    def test_diverse_prompts_are_nonempty(self):
        prompts = _get_diverse_prompts()
        assert len(prompts) >= 50  # We have ~75 hand-crafted prompts

    def test_diverse_prompts_are_unique(self):
        prompts = _get_diverse_prompts()
        assert len(prompts) == len(set(prompts))

    def test_diverse_prompts_are_strings(self):
        for p in _get_diverse_prompts():
            assert isinstance(p, str)
            assert len(p.strip()) > 0

    def test_dan_jailbreaks_are_nonempty(self):
        prompts = _get_dan_style_jailbreaks()
        assert len(prompts) >= 15  # We have 20 curated jailbreaks

    def test_dan_jailbreaks_are_unique(self):
        prompts = _get_dan_style_jailbreaks()
        assert len(prompts) == len(set(prompts))

    def test_dan_jailbreaks_are_strings(self):
        for p in _get_dan_style_jailbreaks():
            assert isinstance(p, str)
            assert len(p.strip()) > 0


# ---------------------------------------------------------------------------
# Collection functions (offline mode — no HuggingFace downloads)
# ---------------------------------------------------------------------------


class TestCollectBenignOffline:
    """Test benign collection with only the built-in diverse prompts.

    When HuggingFace datasets fail to load (e.g. no network), the collector
    falls back to hand-crafted prompts only.
    """

    def test_returns_tuples(self):
        # With n small enough to be satisfied by built-in prompts alone
        result = collect_benign_prompts(n=10, seed=42)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_respects_n_limit(self):
        result = collect_benign_prompts(n=10, seed=42)
        assert len(result) <= 10

    def test_deterministic(self):
        r1 = collect_benign_prompts(n=20, seed=42)
        r2 = collect_benign_prompts(n=20, seed=42)
        assert r1 == r2

    def test_at_least_diverse_prompts_available(self):
        """Even without network, we should get the hand-crafted prompts."""
        result = collect_benign_prompts(n=50, seed=42)
        sources = {s for _, s in result}
        assert "diverse" in sources


class TestCollectJailbreakOffline:
    """Test jailbreak collection with only the built-in Dan-style prompts."""

    def test_returns_tuples(self):
        result = collect_jailbreak_prompts(n=10, seed=42)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_respects_n_limit(self):
        result = collect_jailbreak_prompts(n=5, seed=42)
        assert len(result) <= 5

    def test_deterministic(self):
        r1 = collect_jailbreak_prompts(n=15, seed=42)
        r2 = collect_jailbreak_prompts(n=15, seed=42)
        assert r1 == r2

    def test_at_least_dan_prompts_available(self):
        """Even without network, we should get the Dan-style jailbreaks."""
        result = collect_jailbreak_prompts(n=15, seed=42)
        sources = {s for _, s in result}
        assert "dan_style" in sources


# ---------------------------------------------------------------------------
# Large roundtrip
# ---------------------------------------------------------------------------


class TestLargeRoundtrip:
    def test_500_prompts_roundtrip(self, tmp_path):
        """Verify JSONL roundtrip with a realistically sized dataset."""
        n = 500
        prompts = [f"prompt number {i}" for i in range(n)]
        labels = [0 if i < 400 else 1 for i in range(n)]
        sources = [f"source_{i % 5}" for i in range(n)]

        ds = PromptDataset(prompts, labels, sources)
        path = tmp_path / "large.jsonl"
        ds.save(path)

        loaded = PromptDataset.load(path)
        assert len(loaded) == n
        assert loaded.prompts == ds.prompts
        assert loaded.labels == ds.labels
        assert loaded.sources == ds.sources
        assert loaded.n_benign == 400
        assert loaded.n_jailbreaks == 100
