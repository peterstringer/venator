"""Train/validation/test splitting with strict methodology constraints.

Supports two split modes:

**UNSUPERVISED** (original methodology):
- Training set: benign prompts ONLY (detector learns "normal")
- Validation set: benign prompts ONLY (for threshold tuning)
- Test set: held-out benign + ALL jailbreak prompts
- Jailbreak prompts NEVER appear in training or validation data.

Split strategy (benign indices only):
    train:          70% of benign prompts
    val:            15% of benign prompts
    test_benign:    15% of benign prompts
    test_jailbreak: ALL jailbreak prompts (100%)

**SEMI_SUPERVISED** (few-shot labeled jailbreaks):
- Training and validation sets include a small labeled jailbreak subset.
- 70% of jailbreaks are reserved for uncontaminated testing.
- Enables contrastive/few-shot detection methods.

Split strategy:
    train_benign:    70% of benign prompts
    train_jailbreak: 15% of jailbreak prompts
    val_benign:      15% of benign prompts
    val_jailbreak:   15% of jailbreak prompts
    test_benign:     15% of benign prompts (remainder)
    test_jailbreak:  70% of jailbreak prompts (remainder)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from venator.activation.storage import ActivationStore

logger = logging.getLogger(__name__)


class SplitMode(str, Enum):
    """Split methodology mode.

    UNSUPERVISED: Train/val contain only benign prompts; all jailbreaks
        are reserved for testing. This is the standard unsupervised
        anomaly detection setup.
    SEMI_SUPERVISED: A small fraction of labeled jailbreaks is included
        in train and val splits, with the majority reserved for testing.
        Enables contrastive or few-shot detection methods.
    """

    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"


@dataclass
class DataSplit:
    """A named data split with indices into an ActivationStore.

    Attributes:
        name: Split identifier ("train", "val", "test_benign", "test_jailbreak").
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


class SplitManager:
    """Creates and manages train/val/test splits with strict methodology.

    Supports two modes:
    - UNSUPERVISED: Jailbreaks appear ONLY in the test set.
    - SEMI_SUPERVISED: A small labeled jailbreak fraction is included in
      train/val, with the majority reserved for testing.

    Args:
        seed: Random seed for reproducible shuffling.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def create_splits(
        self,
        store: ActivationStore,
        mode: SplitMode = SplitMode.SEMI_SUPERVISED,
        benign_train_frac: float = 0.70,
        benign_val_frac: float = 0.15,
        jailbreak_train_frac: float = 0.15,
        jailbreak_val_frac: float = 0.15,
    ) -> dict[str, DataSplit]:
        """Create splits from an activation store.

        Args:
            store: ActivationStore containing prompts with labels.
            mode: Split methodology — UNSUPERVISED or SEMI_SUPERVISED.
            benign_train_frac: Fraction of benign prompts for training.
            benign_val_frac: Fraction of benign prompts for validation.
            jailbreak_train_frac: Fraction of jailbreaks for training
                (SEMI_SUPERVISED only, ignored in UNSUPERVISED).
            jailbreak_val_frac: Fraction of jailbreaks for validation
                (SEMI_SUPERVISED only, ignored in UNSUPERVISED).

        Returns:
            Dict mapping split name -> DataSplit.

        Raises:
            ValueError: If fractions are invalid or store is empty.
        """
        if mode == SplitMode.UNSUPERVISED:
            return self._create_unsupervised_splits(
                store, benign_train_frac, benign_val_frac,
            )
        else:
            return self._create_semi_supervised_splits(
                store,
                benign_train_frac,
                benign_val_frac,
                jailbreak_train_frac,
                jailbreak_val_frac,
            )

    def _create_unsupervised_splits(
        self,
        store: ActivationStore,
        train_frac: float,
        val_frac: float,
    ) -> dict[str, DataSplit]:
        """Create unsupervised splits — jailbreaks only in test."""
        test_frac = 1.0 - train_frac - val_frac

        if not (0 < train_frac < 1 and 0 < val_frac < 1 and test_frac > 0):
            raise ValueError(
                f"Invalid fractions: train={train_frac}, val={val_frac}, "
                f"test={test_frac:.4f}. All must be positive and sum to 1."
            )
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
            raise ValueError(
                f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac:.6f}"
            )

        benign_idx, jailbreak_idx = store.split_by_labels()

        if len(benign_idx) == 0:
            raise ValueError("Store contains no benign prompts — cannot create splits")

        # Shuffle benign indices deterministically
        rng = np.random.default_rng(self.seed)
        rng.shuffle(benign_idx)

        # Compute split boundaries
        n_benign = len(benign_idx)
        n_train = max(1, int(round(n_benign * train_frac)))
        n_val = max(1, int(round(n_benign * val_frac)))
        # test_benign gets the remainder to avoid off-by-one from rounding
        n_test_benign = n_benign - n_train - n_val

        if n_test_benign < 1:
            # With very few samples, ensure at least 1 in each split
            n_test_benign = 1
            n_train = n_benign - n_val - n_test_benign
            if n_train < 1:
                n_train = 1
                n_val = n_benign - n_train - n_test_benign
                if n_val < 1:
                    raise ValueError(
                        f"Too few benign samples ({n_benign}) to create 3 splits. "
                        "Need at least 3 benign prompts."
                    )

        train_indices = np.sort(benign_idx[:n_train])
        val_indices = np.sort(benign_idx[n_train : n_train + n_val])
        test_benign_indices = np.sort(benign_idx[n_train + n_val :])

        splits = {
            "train": DataSplit(
                name="train",
                indices=train_indices,
                n_samples=len(train_indices),
                contains_jailbreaks=False,
            ),
            "val": DataSplit(
                name="val",
                indices=val_indices,
                n_samples=len(val_indices),
                contains_jailbreaks=False,
            ),
            "test_benign": DataSplit(
                name="test_benign",
                indices=test_benign_indices,
                n_samples=len(test_benign_indices),
                contains_jailbreaks=False,
            ),
            "test_jailbreak": DataSplit(
                name="test_jailbreak",
                indices=np.sort(jailbreak_idx),
                n_samples=len(jailbreak_idx),
                contains_jailbreaks=True,
            ),
        }

        # Validate immediately — fail fast if something is wrong
        self.validate_splits(splits, store, mode=SplitMode.UNSUPERVISED)

        logger.info(
            "Created UNSUPERVISED splits: train=%d, val=%d, "
            "test_benign=%d, test_jailbreak=%d",
            splits["train"].n_samples,
            splits["val"].n_samples,
            splits["test_benign"].n_samples,
            splits["test_jailbreak"].n_samples,
        )

        return splits

    def _create_semi_supervised_splits(
        self,
        store: ActivationStore,
        benign_train_frac: float,
        benign_val_frac: float,
        jailbreak_train_frac: float,
        jailbreak_val_frac: float,
    ) -> dict[str, DataSplit]:
        """Create semi-supervised splits — small jailbreak fraction in train/val."""
        # Validate benign fractions
        benign_test_frac = 1.0 - benign_train_frac - benign_val_frac
        if not (0 < benign_train_frac < 1 and 0 < benign_val_frac < 1 and benign_test_frac > 0):
            raise ValueError(
                f"Invalid benign fractions: train={benign_train_frac}, "
                f"val={benign_val_frac}, test={benign_test_frac:.4f}. "
                "All must be positive and sum to 1."
            )

        # Validate jailbreak fractions
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

        benign_idx, jailbreak_idx = store.split_by_labels()

        if len(benign_idx) == 0:
            raise ValueError("Store contains no benign prompts — cannot create splits")
        if len(jailbreak_idx) == 0:
            raise ValueError(
                "Store contains no jailbreak prompts — cannot create "
                "semi-supervised splits. Use UNSUPERVISED mode instead."
            )

        rng = np.random.default_rng(self.seed)
        rng.shuffle(benign_idx)
        rng.shuffle(jailbreak_idx)

        # --- Benign splits ---
        n_benign = len(benign_idx)
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
        n_jailbreak = len(jailbreak_idx)
        n_j_train = max(0, int(round(n_jailbreak * jailbreak_train_frac)))
        n_j_val = max(0, int(round(n_jailbreak * jailbreak_val_frac)))
        n_j_test = n_jailbreak - n_j_train - n_j_val

        if n_j_test < 1:
            raise ValueError(
                f"Too few jailbreak samples ({n_jailbreak}) to reserve any for testing "
                f"with train_frac={jailbreak_train_frac}, val_frac={jailbreak_val_frac}. "
                "Reduce jailbreak train/val fractions or add more jailbreak prompts."
            )

        # Build splits
        splits = {
            "train_benign": DataSplit(
                name="train_benign",
                indices=np.sort(benign_idx[:n_b_train]),
                n_samples=n_b_train,
                contains_jailbreaks=False,
            ),
            "train_jailbreak": DataSplit(
                name="train_jailbreak",
                indices=np.sort(jailbreak_idx[:n_j_train]),
                n_samples=n_j_train,
                contains_jailbreaks=True,
            ),
            "val_benign": DataSplit(
                name="val_benign",
                indices=np.sort(benign_idx[n_b_train : n_b_train + n_b_val]),
                n_samples=n_b_val,
                contains_jailbreaks=False,
            ),
            "val_jailbreak": DataSplit(
                name="val_jailbreak",
                indices=np.sort(jailbreak_idx[n_j_train : n_j_train + n_j_val]),
                n_samples=n_j_val,
                contains_jailbreaks=True,
            ),
            "test_benign": DataSplit(
                name="test_benign",
                indices=np.sort(benign_idx[n_b_train + n_b_val :]),
                n_samples=n_b_test,
                contains_jailbreaks=False,
            ),
            "test_jailbreak": DataSplit(
                name="test_jailbreak",
                indices=np.sort(jailbreak_idx[n_j_train + n_j_val :]),
                n_samples=n_j_test,
                contains_jailbreaks=True,
            ),
        }

        self.validate_splits(splits, store, mode=SplitMode.SEMI_SUPERVISED)

        logger.info(
            "Created SEMI_SUPERVISED splits: "
            "train_benign=%d, train_jailbreak=%d, "
            "val_benign=%d, val_jailbreak=%d, "
            "test_benign=%d, test_jailbreak=%d",
            splits["train_benign"].n_samples,
            splits["train_jailbreak"].n_samples,
            splits["val_benign"].n_samples,
            splits["val_jailbreak"].n_samples,
            splits["test_benign"].n_samples,
            splits["test_jailbreak"].n_samples,
        )

        return splits

    def validate_splits(
        self,
        splits: dict[str, DataSplit],
        store: ActivationStore,
        mode: SplitMode = SplitMode.UNSUPERVISED,
    ) -> None:
        """Verify split integrity against the activation store.

        Raises ValueError if any methodology constraint is violated.

        Checks (UNSUPERVISED):
            1. Train labels are all 0 (benign).
            2. Val labels are all 0 (benign).
            3. Test_jailbreak labels are all 1 (jailbreak).
            4. No index appears in multiple splits.
            5. All store indices are covered exactly once.

        Checks (SEMI_SUPERVISED):
            1. Benign splits contain only benign labels.
            2. Jailbreak splits contain only jailbreak labels.
            3. No index appears in multiple splits.
            4. All store indices are covered exactly once.
        """
        labels = store.get_labels()

        if mode == SplitMode.UNSUPERVISED:
            # --- Check 1 & 2: No jailbreaks in train or val ---
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

            # --- Check 3: Test_jailbreak contains only jailbreaks ---
            if "test_jailbreak" in splits:
                split = splits["test_jailbreak"]
                if split.n_samples > 0:
                    split_labels = labels[split.indices]
                    n_benign = int(np.sum(split_labels == 0))
                    if n_benign > 0:
                        raise ValueError(
                            f"test_jailbreak split contains {n_benign} benign prompts. "
                            "It should contain ONLY jailbreak prompts."
                        )

        else:  # SEMI_SUPERVISED
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

        # --- No index overlap (both modes) ---
        all_indices: list[np.ndarray] = []
        for split in splits.values():
            all_indices.append(split.indices)

        combined = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int64)
        if len(combined) != len(np.unique(combined)):
            raise ValueError(
                "Index overlap detected between splits. Each prompt must "
                "appear in exactly one split."
            )

        # --- Full coverage (both modes) ---
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
        mode: SplitMode = SplitMode.UNSUPERVISED,
    ) -> None:
        """Save split definitions as JSON.

        Stores only indices and metadata — not the activation data itself.
        The splits can be reloaded and applied to the same ActivationStore.

        Args:
            splits: Dict of split name -> DataSplit.
            path: Output JSON file path.
            mode: The SplitMode used to create these splits.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "seed": self.seed,
            "mode": mode.value,
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

        logger.info("Saved %s split definitions to %s", mode.value, path)

    @classmethod
    def load_splits(cls, path: Path | str) -> dict[str, DataSplit]:
        """Load split definitions from JSON.

        Args:
            path: Path to the JSON file saved by save_splits.

        Returns:
            Dict mapping split name -> DataSplit.
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

        logger.info("Loaded split definitions from %s", path)
        return splits

    @classmethod
    def load_mode(cls, path: Path | str) -> SplitMode:
        """Load the SplitMode from a saved splits JSON file.

        Returns SplitMode.UNSUPERVISED for files saved before mode was added.

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
