"""Unified data splitting for all detector types.

Creates a single split that serves every detector:

    Unsupervised: fit(train_benign) → calibrate(val_benign) → test(test_*)
    Supervised:   fit(train_benign + train_jailbreak) → calibrate(val_*) → test(test_*)
    Ensemble:     each component uses its own path above → combine → test(test_*)

The split always produces 6 pieces:
    train_benign       — For unsupervised detector training (learn "normal")
    train_jailbreak    — For supervised detector training (labeled examples)
    val_benign         — For threshold calibration (both detector types)
    val_jailbreak      — For threshold calibration (supervised + ensemble)
    test_benign        — For final evaluation (NEVER used in training)
    test_jailbreak     — For final evaluation (NEVER used in training)

Default allocations:
    Benign:    70% train / 15% val / 15% test
    Jailbreak: 15% train / 15% val / 70% test

Split once, train anything. The pipeline routes data to the right detector.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from venator.activation.storage import ActivationStore

logger = logging.getLogger(__name__)


class SplitMode(str, Enum):
    """Split methodology mode.

    UNSUPERVISED: Legacy mode — train/val contain only benign prompts.
        Kept for backward compatibility with old split files.
    SEMI_SUPERVISED: Legacy name for the unified split format.
        All new splits use this format.
    UNIFIED: The current standard — always produces 6 pieces.
        Equivalent to SEMI_SUPERVISED in output format.
    """

    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    UNIFIED = "unified"


@dataclass
class DataSplit:
    """A named data split with indices into an ActivationStore.

    Attributes:
        name: Split identifier (e.g. "train_benign", "test_jailbreak").
        indices: Row indices into the activation store (sorted).
        n_samples: Number of samples in this split.
        contains_jailbreaks: Whether this split includes jailbreak prompts.
    """

    name: str
    indices: np.ndarray
    n_samples: int
    contains_jailbreaks: bool

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.n_samples = len(self.indices)


@dataclass
class UnifiedSplit:
    """One split that serves every detector type.

    Pieces:
        train_benign       — For unsupervised detector training (learn "normal")
        train_jailbreak    — For supervised detector training (labeled examples)
        val_benign         — For threshold calibration (both detector types)
        val_jailbreak      — For threshold calibration (supervised + ensemble)
        test_benign        — For final evaluation (NEVER used in training)
        test_jailbreak     — For final evaluation (NEVER used in training)

    How each detector type uses the split:
        Unsupervised: fit(train_benign) → calibrate(val_benign) → test(test_*)
        Supervised:   fit(train_benign + train_jailbreak) → calibrate(val_*) → test(test_*)
        Ensemble:     each component uses its own path above → combine → test(test_*)
    """

    train_benign_indices: np.ndarray
    train_jailbreak_indices: np.ndarray
    val_benign_indices: np.ndarray
    val_jailbreak_indices: np.ndarray
    test_benign_indices: np.ndarray
    test_jailbreak_indices: np.ndarray

    def __post_init__(self) -> None:
        self.train_benign_indices = np.asarray(self.train_benign_indices, dtype=np.int64)
        self.train_jailbreak_indices = np.asarray(self.train_jailbreak_indices, dtype=np.int64)
        self.val_benign_indices = np.asarray(self.val_benign_indices, dtype=np.int64)
        self.val_jailbreak_indices = np.asarray(self.val_jailbreak_indices, dtype=np.int64)
        self.test_benign_indices = np.asarray(self.test_benign_indices, dtype=np.int64)
        self.test_jailbreak_indices = np.asarray(self.test_jailbreak_indices, dtype=np.int64)

    def to_split_dict(self) -> dict[str, DataSplit]:
        """Convert to the dict[str, DataSplit] format used by consumers."""
        return {
            "train_benign": DataSplit(
                name="train_benign",
                indices=self.train_benign_indices,
                n_samples=len(self.train_benign_indices),
                contains_jailbreaks=False,
            ),
            "train_jailbreak": DataSplit(
                name="train_jailbreak",
                indices=self.train_jailbreak_indices,
                n_samples=len(self.train_jailbreak_indices),
                contains_jailbreaks=True,
            ),
            "val_benign": DataSplit(
                name="val_benign",
                indices=self.val_benign_indices,
                n_samples=len(self.val_benign_indices),
                contains_jailbreaks=False,
            ),
            "val_jailbreak": DataSplit(
                name="val_jailbreak",
                indices=self.val_jailbreak_indices,
                n_samples=len(self.val_jailbreak_indices),
                contains_jailbreaks=True,
            ),
            "test_benign": DataSplit(
                name="test_benign",
                indices=self.test_benign_indices,
                n_samples=len(self.test_benign_indices),
                contains_jailbreaks=False,
            ),
            "test_jailbreak": DataSplit(
                name="test_jailbreak",
                indices=self.test_jailbreak_indices,
                n_samples=len(self.test_jailbreak_indices),
                contains_jailbreaks=True,
            ),
        }


def create_unified_split(
    benign_indices: np.ndarray,
    jailbreak_indices: np.ndarray,
    benign_train_frac: float = 0.70,
    benign_val_frac: float = 0.15,
    jailbreak_train_frac: float = 0.15,
    jailbreak_val_frac: float = 0.15,
    seed: int = 42,
) -> UnifiedSplit:
    """Create one split that all detectors can use.

    Default allocations:
        Benign:    70% train / 15% val / 15% test
        Jailbreak: 15% train / 15% val / 70% test

    Args:
        benign_indices: Store indices of benign prompts.
        jailbreak_indices: Store indices of jailbreak prompts.
        benign_train_frac: Fraction of benign prompts for training.
        benign_val_frac: Fraction of benign prompts for validation.
        jailbreak_train_frac: Fraction of jailbreaks for training.
        jailbreak_val_frac: Fraction of jailbreaks for validation.
        seed: Random seed for reproducible shuffling.

    Returns:
        A UnifiedSplit with all 6 index arrays.

    Raises:
        ValueError: If fractions are invalid or insufficient data.
    """
    benign_indices = np.asarray(benign_indices, dtype=np.int64).copy()
    jailbreak_indices = np.asarray(jailbreak_indices, dtype=np.int64).copy()

    n_benign = len(benign_indices)
    n_jailbreak = len(jailbreak_indices)

    if n_benign == 0:
        raise ValueError("No benign prompts — cannot create splits")

    # Validate benign fractions
    benign_test_frac = 1.0 - benign_train_frac - benign_val_frac
    if not (0 < benign_train_frac < 1 and 0 < benign_val_frac < 1 and benign_test_frac > 0):
        raise ValueError(
            f"Invalid benign fractions: train={benign_train_frac}, "
            f"val={benign_val_frac}, test={benign_test_frac:.4f}. "
            "All must be positive and sum to 1."
        )

    # Validate jailbreak fractions (only if jailbreaks exist)
    if n_jailbreak > 0:
        jailbreak_test_frac = 1.0 - jailbreak_train_frac - jailbreak_val_frac
        if not (0 <= jailbreak_train_frac < 1 and 0 <= jailbreak_val_frac < 1 and jailbreak_test_frac > 0):
            raise ValueError(
                f"Invalid jailbreak fractions: train={jailbreak_train_frac}, "
                f"val={jailbreak_val_frac}, test={jailbreak_test_frac:.4f}. "
                "Train/val must be non-negative, test must be positive, "
                "and all must sum to 1."
            )
        if jailbreak_test_frac < 0.5:
            raise ValueError(
                f"Jailbreak test fraction is {jailbreak_test_frac:.2f} — at least "
                "50% of jailbreaks must be reserved for uncontaminated testing."
            )

    # Shuffle deterministically
    rng = np.random.default_rng(seed)
    rng.shuffle(benign_indices)
    if n_jailbreak > 0:
        rng.shuffle(jailbreak_indices)

    # --- Benign splits ---
    n_b_train = max(1, int(round(n_benign * benign_train_frac)))
    n_b_val = max(1, int(round(n_benign * benign_val_frac)))
    n_b_test = n_benign - n_b_train - n_b_val

    if n_b_test < 1:
        n_b_test = 1
        n_b_train = n_benign - n_b_val - n_b_test
        if n_b_train < 1:
            n_b_train = 1
            n_b_val = n_benign - n_b_train - n_b_test
            if n_b_val < 1:
                raise ValueError(
                    f"Too few benign samples ({n_benign}) to create 3 splits. "
                    "Need at least 3 benign prompts."
                )

    # --- Jailbreak splits ---
    if n_jailbreak > 0:
        n_j_train = max(0, int(round(n_jailbreak * jailbreak_train_frac)))
        n_j_val = max(0, int(round(n_jailbreak * jailbreak_val_frac)))
        n_j_test = n_jailbreak - n_j_train - n_j_val

        if n_j_test < 1:
            raise ValueError(
                f"Too few jailbreak samples ({n_jailbreak}) to reserve any for testing "
                f"with train_frac={jailbreak_train_frac}, val_frac={jailbreak_val_frac}. "
                "Reduce jailbreak train/val fractions or add more jailbreak prompts."
            )
    else:
        n_j_train = 0
        n_j_val = 0

    split = UnifiedSplit(
        train_benign_indices=np.sort(benign_indices[:n_b_train]),
        train_jailbreak_indices=np.sort(jailbreak_indices[:n_j_train]),
        val_benign_indices=np.sort(benign_indices[n_b_train : n_b_train + n_b_val]),
        val_jailbreak_indices=np.sort(jailbreak_indices[n_j_train : n_j_train + n_j_val]),
        test_benign_indices=np.sort(benign_indices[n_b_train + n_b_val :]),
        test_jailbreak_indices=np.sort(jailbreak_indices[n_j_train + n_j_val :]),
    )

    logger.info(
        "Created unified split: "
        "train_benign=%d, train_jailbreak=%d, "
        "val_benign=%d, val_jailbreak=%d, "
        "test_benign=%d, test_jailbreak=%d",
        len(split.train_benign_indices),
        len(split.train_jailbreak_indices),
        len(split.val_benign_indices),
        len(split.val_jailbreak_indices),
        len(split.test_benign_indices),
        len(split.test_jailbreak_indices),
    )

    return split


class SplitManager:
    """Creates and manages train/val/test splits.

    The primary API is ``create_splits()`` which always produces a unified
    split with 6 keys (train_benign, train_jailbreak, val_benign,
    val_jailbreak, test_benign, test_jailbreak).

    Each detector type picks the pieces it needs:
    - Unsupervised: train_benign only
    - Supervised: train_benign + train_jailbreak
    - Threshold calibration: val_benign + val_jailbreak

    Args:
        seed: Random seed for reproducible shuffling.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def create_splits(
        self,
        store: ActivationStore,
        benign_train_frac: float = 0.70,
        benign_val_frac: float = 0.15,
        jailbreak_train_frac: float = 0.15,
        jailbreak_val_frac: float = 0.15,
        # Backward compat: mode is accepted but ignored (always produces unified)
        mode: SplitMode | None = None,
    ) -> dict[str, DataSplit]:
        """Create splits from an activation store.

        Always produces the unified 6-key format regardless of mode parameter.
        The ``mode`` parameter is accepted for backward compatibility but is
        ignored — all splits are unified.

        Args:
            store: ActivationStore containing prompts with labels.
            benign_train_frac: Fraction of benign prompts for training.
            benign_val_frac: Fraction of benign prompts for validation.
            jailbreak_train_frac: Fraction of jailbreaks for training.
            jailbreak_val_frac: Fraction of jailbreaks for validation.
            mode: Ignored. Kept for backward compatibility.

        Returns:
            Dict mapping split name -> DataSplit (always 6 keys).

        Raises:
            ValueError: If fractions are invalid or store is empty.
        """
        benign_idx, jailbreak_idx = store.split_by_labels()

        unified = create_unified_split(
            benign_indices=benign_idx,
            jailbreak_indices=jailbreak_idx,
            benign_train_frac=benign_train_frac,
            benign_val_frac=benign_val_frac,
            jailbreak_train_frac=jailbreak_train_frac,
            jailbreak_val_frac=jailbreak_val_frac,
            seed=self.seed,
        )

        splits = unified.to_split_dict()

        # Validate immediately — fail fast if something is wrong
        self.validate_splits(splits, store)

        return splits

    def validate_splits(
        self,
        splits: dict[str, DataSplit],
        store: ActivationStore,
        mode: SplitMode | None = None,
    ) -> None:
        """Verify split integrity against the activation store.

        Raises ValueError if any methodology constraint is violated.

        Checks:
            1. Benign splits contain only benign labels.
            2. Jailbreak splits contain only jailbreak labels.
            3. No index appears in multiple splits.
            4. All store indices are covered exactly once.

        Args:
            splits: Dict of split name -> DataSplit.
            store: The ActivationStore these splits reference.
            mode: Ignored. Kept for backward compatibility.
        """
        labels = store.get_labels()

        # Benign splits must contain only benign labels
        for split_name in ("train_benign", "val_benign", "test_benign"):
            if split_name not in splits:
                continue
            split = splits[split_name]
            if split.n_samples == 0:
                continue
            split_labels = labels[split.indices]
            n_jailbreak = int(np.sum(split_labels == 1))
            if n_jailbreak > 0:
                raise ValueError(
                    f"LABEL MISMATCH: {split_name} split contains "
                    f"{n_jailbreak} jailbreak prompts but should contain "
                    "only benign prompts."
                )

        # Jailbreak splits must contain only jailbreak labels
        for split_name in ("train_jailbreak", "val_jailbreak", "test_jailbreak"):
            if split_name not in splits:
                continue
            split = splits[split_name]
            if split.n_samples == 0:
                continue
            split_labels = labels[split.indices]
            n_benign = int(np.sum(split_labels == 0))
            if n_benign > 0:
                raise ValueError(
                    f"LABEL MISMATCH: {split_name} split contains "
                    f"{n_benign} benign prompts but should contain "
                    "only jailbreak prompts."
                )

        # Also validate legacy unsupervised format if present
        for split_name in ("train", "val"):
            if split_name not in splits:
                continue
            split = splits[split_name]
            if split.n_samples == 0:
                continue
            split_labels = labels[split.indices]
            n_jailbreak = int(np.sum(split_labels == 1))
            if n_jailbreak > 0:
                raise ValueError(
                    f"METHODOLOGY VIOLATION: {split_name} split contains "
                    f"{n_jailbreak} jailbreak prompts. Training and validation "
                    "must contain ONLY benign prompts."
                )

        # --- No index overlap ---
        all_indices: list[np.ndarray] = []
        for split in splits.values():
            all_indices.append(split.indices)

        combined = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int64)
        if len(combined) != len(np.unique(combined)):
            raise ValueError(
                "Index overlap detected between splits. Each prompt must "
                "appear in exactly one split."
            )

        # --- Full coverage ---
        n_total = store.n_prompts
        if len(combined) != n_total:
            raise ValueError(
                f"Splits cover {len(combined)} indices but store has "
                f"{n_total} prompts. All prompts must be assigned to a split."
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_splits(
        self,
        splits: dict[str, DataSplit],
        path: Path | str,
        mode: SplitMode | None = None,
    ) -> None:
        """Save split definitions as JSON.

        Stores only indices and metadata — not the activation data itself.

        Args:
            splits: Dict of split name -> DataSplit.
            path: Output JSON file path.
            mode: Ignored. All new saves use "unified" mode.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "seed": self.seed,
            "mode": SplitMode.UNIFIED.value,
            "splits": {
                name: {
                    "name": split.name,
                    "indices": split.indices.tolist(),
                    "n_samples": split.n_samples,
                    "contains_jailbreaks": split.contains_jailbreaks,
                }
                for name, split in splits.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved unified split definitions to %s", path)

    @classmethod
    def load_splits(cls, path: Path | str) -> dict[str, DataSplit]:
        """Load split definitions from JSON.

        Handles both old (unsupervised 4-key) and new (unified 6-key) formats.
        Old 4-key format is automatically converted to the 6-key format by
        mapping "train" -> "train_benign" and "val" -> "val_benign", with
        empty train_jailbreak and val_jailbreak arrays.

        Args:
            path: Path to the JSON file saved by save_splits.

        Returns:
            Dict mapping split name -> DataSplit (always 6 keys).
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        splits = {}
        for name, info in data["splits"].items():
            splits[name] = DataSplit(
                name=info["name"],
                indices=np.array(info["indices"], dtype=np.int64),
                n_samples=info["n_samples"],
                contains_jailbreaks=info["contains_jailbreaks"],
            )

        # Migrate old unsupervised format (4 keys) to unified format (6 keys)
        if "train" in splits and "train_benign" not in splits:
            logger.info("Migrating old unsupervised split format to unified format")
            old_splits = splits
            splits = {
                "train_benign": DataSplit(
                    name="train_benign",
                    indices=old_splits["train"].indices,
                    n_samples=old_splits["train"].n_samples,
                    contains_jailbreaks=False,
                ),
                "train_jailbreak": DataSplit(
                    name="train_jailbreak",
                    indices=np.array([], dtype=np.int64),
                    n_samples=0,
                    contains_jailbreaks=True,
                ),
                "val_benign": DataSplit(
                    name="val_benign",
                    indices=old_splits["val"].indices,
                    n_samples=old_splits["val"].n_samples,
                    contains_jailbreaks=False,
                ),
                "val_jailbreak": DataSplit(
                    name="val_jailbreak",
                    indices=np.array([], dtype=np.int64),
                    n_samples=0,
                    contains_jailbreaks=True,
                ),
                "test_benign": DataSplit(
                    name="test_benign",
                    indices=old_splits["test_benign"].indices,
                    n_samples=old_splits["test_benign"].n_samples,
                    contains_jailbreaks=False,
                ),
                "test_jailbreak": DataSplit(
                    name="test_jailbreak",
                    indices=old_splits["test_jailbreak"].indices,
                    n_samples=old_splits["test_jailbreak"].n_samples,
                    contains_jailbreaks=True,
                ),
            }

        logger.info("Loaded split definitions from %s", path)
        return splits

    @classmethod
    def load_mode(cls, path: Path | str) -> SplitMode:
        """Load the SplitMode from a saved splits JSON file.

        Returns SplitMode.UNSUPERVISED for legacy files without a mode field.

        Args:
            path: Path to the JSON file saved by save_splits.

        Returns:
            The SplitMode used when the splits were created.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mode_str = data.get("mode", "unsupervised")
        return SplitMode(mode_str)
