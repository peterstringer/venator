"""HDF5-based storage for activation matrices.

Provides efficient read/write for large activation arrays extracted from LLM
hidden states. Uses h5py for chunked, compressed storage suitable for the
high-dimensional activation data (n_samples x n_layers x hidden_dim).

HDF5 file layout:
    /metadata                (attrs: model_id, layers, hidden_dim, n_prompts, created_at)
    /prompts                 (dataset: variable-length UTF-8 strings, resizable)
    /labels                  (dataset: int8 array, 0=benign 1=jailbreak, resizable)
    /activations/
        /layer_{idx}         (dataset: float32 (n_prompts, hidden_dim), resizable)

Datasets are created with maxshape=(None, ...) so rows can be appended
incrementally without knowing the final count upfront.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class ActivationStore:
    """Efficient storage and retrieval of activation vectors using HDF5.

    Supports incremental appends (single or batch), random-access reads by
    index or slice, and metadata preservation. Datasets are chunked and
    gzip-compressed for compact storage of high-dimensional float32 data.

    Args:
        path: Filesystem path to the HDF5 file (existing or to-be-created).
    """

    # Chunk row count — balances compression ratio vs random-access granularity.
    # 64 rows × 4096 dims × 4 bytes ≈ 1 MB per chunk.
    _CHUNK_ROWS = 64

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: str | Path,
        model_id: str,
        layers: list[int],
        hidden_dim: int,
    ) -> ActivationStore:
        """Create a new HDF5 store with metadata and empty resizable datasets.

        Args:
            path: Where to write the HDF5 file. Parent directory must exist.
            model_id: Identifier of the model that produced the activations.
            layers: Transformer layer indices stored in this file.
            hidden_dim: Dimensionality of each activation vector.

        Returns:
            A new ActivationStore instance ready for appends.

        Raises:
            FileExistsError: If the file already exists.
            ValueError: If layers is empty or hidden_dim < 1.
        """
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Store already exists: {path}")
        if not layers:
            raise ValueError("layers must be a non-empty list of layer indices")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        sorted_layers = sorted(layers)
        chunk_rows = cls._CHUNK_ROWS

        with h5py.File(path, "w") as f:
            # -- Metadata --
            f.attrs["model_id"] = model_id
            f.attrs["layers"] = sorted_layers
            f.attrs["hidden_dim"] = hidden_dim
            f.attrs["n_prompts"] = 0
            f.attrs["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # -- Prompts (variable-length UTF-8 strings) --
            str_dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset(
                "prompts",
                shape=(0,),
                maxshape=(None,),
                dtype=str_dt,
            )

            # -- Labels (0=benign, 1=jailbreak) --
            f.create_dataset(
                "labels",
                shape=(0,),
                maxshape=(None,),
                dtype="int8",
            )

            # -- Activation datasets, one per layer --
            grp = f.create_group("activations")
            for layer_idx in sorted_layers:
                grp.create_dataset(
                    f"layer_{layer_idx}",
                    shape=(0, hidden_dim),
                    maxshape=(None, hidden_dim),
                    dtype="float32",
                    chunks=(min(chunk_rows, 1), hidden_dim),  # min 1 for initial append
                )

        logger.info(
            "Created activation store %s (model=%s, layers=%s, dim=%d)",
            path,
            model_id,
            sorted_layers,
            hidden_dim,
        )
        return cls(path)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        prompt: str,
        activations: dict[int, np.ndarray],
        label: int = 0,
    ) -> None:
        """Append a single prompt's activations to the store.

        Args:
            prompt: The source prompt text.
            activations: Mapping of layer_index -> 1-D activation vector (hidden_dim,).
            label: 0 for benign, 1 for jailbreak.

        Raises:
            ValueError: If activation shapes don't match the store's hidden_dim.
            KeyError: If activations are missing for any stored layer.
        """
        # Reshape single vectors to (1, hidden_dim) batch for shared logic
        batch_acts = {}
        for layer_idx, vec in activations.items():
            if vec.ndim == 1:
                batch_acts[layer_idx] = vec[np.newaxis, :]
            else:
                batch_acts[layer_idx] = vec

        self.append_batch(
            prompts=[prompt],
            activations=batch_acts,
            labels=[label],
        )

    def append_batch(
        self,
        prompts: list[str],
        activations: dict[int, np.ndarray],
        labels: list[int] | None = None,
    ) -> None:
        """Append a batch of activations to the store.

        Args:
            prompts: List of source prompt texts.
            activations: Mapping of layer_index -> (n_prompts, hidden_dim) array.
            labels: Per-prompt labels (0=benign, 1=jailbreak). Defaults to all 0.

        Raises:
            ValueError: On shape mismatches or inconsistent batch sizes.
            KeyError: If activations are missing for any stored layer.
        """
        n = len(prompts)
        if n == 0:
            return

        if labels is None:
            labels = [0] * n
        if len(labels) != n:
            raise ValueError(
                f"labels length ({len(labels)}) != prompts length ({n})"
            )

        with h5py.File(self.path, "a") as f:
            stored_layers = list(f.attrs["layers"])
            hidden_dim = int(f.attrs["hidden_dim"])
            old_n = int(f.attrs["n_prompts"])
            new_n = old_n + n

            # Validate all required layers are present
            missing = [l for l in stored_layers if l not in activations]
            if missing:
                raise KeyError(
                    f"Missing activations for layers {missing}. "
                    f"Store expects layers {stored_layers}."
                )

            # Validate shapes
            for layer_idx in stored_layers:
                act = activations[layer_idx]
                if act.shape != (n, hidden_dim):
                    raise ValueError(
                        f"Layer {layer_idx}: expected shape ({n}, {hidden_dim}), "
                        f"got {act.shape}"
                    )
                if not np.all(np.isfinite(act)):
                    raise ValueError(
                        f"Layer {layer_idx}: activations contain NaN or Inf values"
                    )

            # -- Resize and write prompts --
            ds_prompts = f["prompts"]
            ds_prompts.resize(new_n, axis=0)
            ds_prompts[old_n:new_n] = prompts

            # -- Resize and write labels --
            ds_labels = f["labels"]
            ds_labels.resize(new_n, axis=0)
            ds_labels[old_n:new_n] = np.array(labels, dtype="int8")

            # -- Resize and write activations --
            for layer_idx in stored_layers:
                ds = f[f"activations/layer_{layer_idx}"]
                ds.resize(new_n, axis=0)
                ds[old_n:new_n] = activations[layer_idx].astype(np.float32)

            # -- Update count --
            f.attrs["n_prompts"] = new_n

        logger.debug("Appended %d prompts (total: %d)", n, new_n)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_activations(
        self,
        layer: int,
        indices: list[int] | slice | None = None,
    ) -> np.ndarray:
        """Get activation matrix for a layer.

        Args:
            layer: Transformer layer index.
            indices: Row indices or slice. None returns all rows.

        Returns:
            np.ndarray of shape (n_selected, hidden_dim), dtype float32.

        Raises:
            KeyError: If the layer doesn't exist in the store.
        """
        with h5py.File(self.path, "r") as f:
            key = f"activations/layer_{layer}"
            if key not in f:
                raise KeyError(
                    f"Layer {layer} not in store. Available: {self.layers}"
                )
            ds = f[key]
            if indices is None:
                return ds[()].astype(np.float32)
            return ds[indices].astype(np.float32)

    def get_labels(self) -> np.ndarray:
        """Get the full label array.

        Returns:
            np.ndarray of shape (n_prompts,), dtype int8. 0=benign, 1=jailbreak.
        """
        with h5py.File(self.path, "r") as f:
            return f["labels"][()].astype(np.int8)

    def get_prompts(self, indices: list[int] | slice | None = None) -> list[str]:
        """Get prompt strings.

        Args:
            indices: Row indices or slice. None returns all prompts.

        Returns:
            List of prompt strings.
        """
        with h5py.File(self.path, "r") as f:
            ds = f["prompts"]
            if indices is None:
                raw = ds[()]
            else:
                raw = ds[indices]
            # h5py returns bytes or str depending on version — normalize
            return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_prompts(self) -> int:
        """Number of prompts currently stored."""
        with h5py.File(self.path, "r") as f:
            return int(f.attrs["n_prompts"])

    @property
    def layers(self) -> list[int]:
        """Sorted list of layer indices stored in this file."""
        with h5py.File(self.path, "r") as f:
            return list(f.attrs["layers"])

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of each activation vector."""
        with h5py.File(self.path, "r") as f:
            return int(f.attrs["hidden_dim"])

    @property
    def model_id(self) -> str:
        """Model identifier that produced these activations."""
        with h5py.File(self.path, "r") as f:
            return str(f.attrs["model_id"])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def split_by_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Return arrays of indices split by label.

        Returns:
            (benign_indices, jailbreak_indices) — each a 1-D int array.
        """
        labels = self.get_labels()
        benign = np.where(labels == 0)[0]
        jailbreak = np.where(labels == 1)[0]
        return benign, jailbreak

    def __len__(self) -> int:
        return self.n_prompts

    def __repr__(self) -> str:
        if not self.path.exists():
            return f"ActivationStore({self.path!s}, not yet created)"
        return (
            f"ActivationStore({self.path!s}, "
            f"n={self.n_prompts}, layers={self.layers}, dim={self.hidden_dim})"
        )
