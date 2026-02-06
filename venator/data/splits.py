"""Train/validation/test splitting with strict methodology constraints.

Enforces the unsupervised anomaly detection methodology:
- Training set: benign prompts ONLY (detector learns "normal")
- Validation set: benign prompts ONLY (for threshold tuning)
- Test set: held-out benign + ALL jailbreak prompts

Jailbreak prompts must NEVER appear in training or validation data.
Violating this invalidates all results.

Split strategy (applied to benign indices only):
    train:          70% of benign prompts
    val:            15% of benign prompts
    test_benign:    15% of benign prompts
    test_jailbreak: ALL jailbreak prompts (100%)

The evaluation test set is test_benign + test_jailbreak.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from venator.activation.storage import ActivationStore

logger = logging.getLogger(__name__)


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

    CRITICAL CONSTRAINT: Jailbreaks appear ONLY in the test set.
    This class enforces this invariant at creation and provides validation
    to verify it after loading.

    Args:
        seed: Random seed for reproducible shuffling of benign indices.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def create_splits(
        self,
        store: ActivationStore,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
    ) -> dict[str, DataSplit]:
        """Create splits from an activation store.

        Benign prompts are shuffled and split into train / val / test_benign.
        All jailbreak prompts go into test_jailbreak. Fractions apply to the
        benign subset only.

        Args:
            store: ActivationStore containing prompts with labels.
            train_frac: Fraction of benign prompts for training.
            val_frac: Fraction of benign prompts for validation.

        Returns:
            Dict mapping split name -> DataSplit.

        Raises:
            ValueError: If fractions are invalid or store is empty.
        """
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
        self.validate_splits(splits, store)

        logger.info(
            "Created splits: train=%d, val=%d, test_benign=%d, test_jailbreak=%d",
            splits["train"].n_samples,
            splits["val"].n_samples,
            splits["test_benign"].n_samples,
            splits["test_jailbreak"].n_samples,
        )

        return splits

    def validate_splits(
        self,
        splits: dict[str, DataSplit],
        store: ActivationStore,
    ) -> None:
        """Verify split integrity against the activation store.

        Raises ValueError if any methodology constraint is violated.

        Checks:
            1. Train labels are all 0 (benign).
            2. Val labels are all 0 (benign).
            3. Test_jailbreak labels are all 1 (jailbreak).
            4. No index appears in multiple splits.
            5. All store indices are covered exactly once.
        """
        labels = store.get_labels()

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

        # --- Check 4: No index overlap ---
        all_indices: list[np.ndarray] = []
        for split in splits.values():
            all_indices.append(split.indices)

        combined = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int64)
        if len(combined) != len(np.unique(combined)):
            raise ValueError(
                "Index overlap detected between splits. Each prompt must "
                "appear in exactly one split."
            )

        # --- Check 5: Full coverage ---
        n_total = store.n_prompts
        if len(combined) != n_total:
            raise ValueError(
                f"Splits cover {len(combined)} indices but store has "
                f"{n_total} prompts. All prompts must be assigned to a split."
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_splits(self, splits: dict[str, DataSplit], path: Path | str) -> None:
        """Save split definitions as JSON.

        Stores only indices and metadata — not the activation data itself.
        The splits can be reloaded and applied to the same ActivationStore.

        Args:
            path: Output JSON file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "seed": self.seed,
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

        logger.info("Saved split definitions to %s", path)

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
