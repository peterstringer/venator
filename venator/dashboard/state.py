"""Session state management for the Streamlit dashboard.

Tracks pipeline progress (which stages are complete), loaded data references,
and carries state between pages via st.session_state.

The PipelineState class wraps st.session_state to provide typed access to
pipeline artifacts and progress tracking. It also detects data on disk from
prior CLI runs so the dashboard picks up where the user left off.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from venator.config import VenatorConfig

logger = logging.getLogger(__name__)

# Pipeline stage definitions: (number, key, display_name)
STAGES = [
    (1, "model_ready", "Pipeline"),
    (2, "evaluation_ready", "Results"),
    (3, "evaluation_ready", "Explore"),
    (4, "model_ready", "Live Detection"),
    (5, "splits_ready", "Ablations"),
]

# Prerequisites: stage_number -> list of state keys that must be True
PREREQUISITES: dict[int, list[str]] = {
    1: [],
    2: ["model_ready"],
    3: ["model_ready", "evaluation_ready"],
    4: ["model_ready"],
    5: ["activations_ready", "splits_ready"],
}


class PipelineState:
    """Typed wrapper around st.session_state for pipeline progress.

    Provides clean property access to pipeline artifacts and progress flags.
    All state is stored in st.session_state so it persists across Streamlit
    reruns within a session.
    """

    # Keys and their default values
    _DEFAULTS: dict[str, object] = {
        # Progress flags
        "prompts_ready": False,
        "activations_ready": False,
        "splits_ready": False,
        "model_ready": False,
        "evaluation_ready": False,
        # Artifact paths
        "benign_path": None,
        "jailbreak_path": None,
        "store_path": None,
        "splits_path": None,
        "model_path": None,
        # Results
        "eval_results": None,
        "train_metrics": None,
        # Config
        "config": None,
        # Initialized flag
        "_pipeline_state_initialized": False,
    }

    def __init__(self, config: VenatorConfig | None = None) -> None:
        """Initialize pipeline state, setting defaults for any missing keys."""
        for key, default in self._DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = default

        if config is not None and st.session_state["config"] is None:
            st.session_state["config"] = config

        # Auto-detect existing artifacts on first init
        if not st.session_state["_pipeline_state_initialized"]:
            self._detect_existing_artifacts()
            st.session_state["_pipeline_state_initialized"] = True

    # ------------------------------------------------------------------
    # Progress flags (read/write)
    # ------------------------------------------------------------------

    @property
    def prompts_ready(self) -> bool:
        return bool(st.session_state["prompts_ready"])

    @prompts_ready.setter
    def prompts_ready(self, value: bool) -> None:
        st.session_state["prompts_ready"] = value

    @property
    def activations_ready(self) -> bool:
        return bool(st.session_state["activations_ready"])

    @activations_ready.setter
    def activations_ready(self, value: bool) -> None:
        st.session_state["activations_ready"] = value

    @property
    def splits_ready(self) -> bool:
        return bool(st.session_state["splits_ready"])

    @splits_ready.setter
    def splits_ready(self, value: bool) -> None:
        st.session_state["splits_ready"] = value

    @property
    def model_ready(self) -> bool:
        return bool(st.session_state["model_ready"])

    @model_ready.setter
    def model_ready(self, value: bool) -> None:
        st.session_state["model_ready"] = value

    @property
    def evaluation_ready(self) -> bool:
        return bool(st.session_state["evaluation_ready"])

    @evaluation_ready.setter
    def evaluation_ready(self, value: bool) -> None:
        st.session_state["evaluation_ready"] = value

    # ------------------------------------------------------------------
    # Artifact paths (read/write)
    # ------------------------------------------------------------------

    @property
    def benign_path(self) -> Path | None:
        v = st.session_state["benign_path"]
        return Path(v) if v else None

    @benign_path.setter
    def benign_path(self, value: Path | str | None) -> None:
        st.session_state["benign_path"] = str(value) if value else None

    @property
    def jailbreak_path(self) -> Path | None:
        v = st.session_state["jailbreak_path"]
        return Path(v) if v else None

    @jailbreak_path.setter
    def jailbreak_path(self, value: Path | str | None) -> None:
        st.session_state["jailbreak_path"] = str(value) if value else None

    @property
    def store_path(self) -> Path | None:
        v = st.session_state["store_path"]
        return Path(v) if v else None

    @store_path.setter
    def store_path(self, value: Path | str | None) -> None:
        st.session_state["store_path"] = str(value) if value else None

    @property
    def splits_path(self) -> Path | None:
        v = st.session_state["splits_path"]
        return Path(v) if v else None

    @splits_path.setter
    def splits_path(self, value: Path | str | None) -> None:
        st.session_state["splits_path"] = str(value) if value else None

    @property
    def model_path(self) -> Path | None:
        v = st.session_state["model_path"]
        return Path(v) if v else None

    @model_path.setter
    def model_path(self, value: Path | str | None) -> None:
        st.session_state["model_path"] = str(value) if value else None

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def eval_results(self) -> dict | None:
        return st.session_state["eval_results"]

    @eval_results.setter
    def eval_results(self, value: dict | None) -> None:
        st.session_state["eval_results"] = value

    @property
    def train_metrics(self) -> dict | None:
        return st.session_state["train_metrics"]

    @train_metrics.setter
    def train_metrics(self, value: dict | None) -> None:
        st.session_state["train_metrics"] = value

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def config(self) -> VenatorConfig:
        cfg = st.session_state["config"]
        if cfg is None:
            cfg = VenatorConfig()
            st.session_state["config"] = cfg
        return cfg

    # ------------------------------------------------------------------
    # Stage availability
    # ------------------------------------------------------------------

    def is_stage_available(self, stage: int) -> bool:
        """Check if a pipeline stage is available (prerequisites met).

        Args:
            stage: Stage number (1-5).

        Returns:
            True if all prerequisite flags are True.
        """
        prereqs = PREREQUISITES.get(stage, [])
        return all(bool(st.session_state.get(key, False)) for key in prereqs)

    def get_progress(self) -> list[tuple[str, bool]]:
        """Get pipeline sub-step progress for sidebar display.

        Returns:
            List of (step_name, is_complete) tuples for the 4 pipeline steps.
        """
        return [
            ("Data", self.prompts_ready),
            ("Extract", self.activations_ready),
            ("Split", self.splits_ready),
            ("Train", self.model_ready),
        ]

    def reset_from(self, stage: int) -> None:
        """Invalidate all stages from the given stage onward.

        When a user re-runs an earlier stage, downstream results are stale.
        This clears the progress flags and artifacts for those stages.

        Args:
            stage: The stage being re-run (1-indexed). All stages >= this
                   stage are reset.
        """
        # Map stage numbers to what they invalidate
        resets: dict[int, list[str]] = {
            1: [
                "prompts_ready", "activations_ready", "splits_ready",
                "model_ready", "evaluation_ready",
                "benign_path", "jailbreak_path", "store_path",
                "splits_path", "model_path",
                "eval_results", "train_metrics",
            ],
            2: [
                "activations_ready", "splits_ready",
                "model_ready", "evaluation_ready",
                "store_path", "splits_path", "model_path",
                "eval_results", "train_metrics",
            ],
            3: [
                "splits_ready", "model_ready", "evaluation_ready",
                "splits_path", "model_path",
                "eval_results", "train_metrics",
            ],
            4: [
                "model_ready", "evaluation_ready",
                "model_path", "eval_results", "train_metrics",
            ],
            5: [
                "evaluation_ready", "eval_results",
            ],
        }

        keys_to_reset = resets.get(stage, [])
        for key in keys_to_reset:
            if key in self._DEFAULTS:
                st.session_state[key] = self._DEFAULTS[key]

    # ------------------------------------------------------------------
    # Auto-detection of existing artifacts
    # ------------------------------------------------------------------

    def _detect_existing_artifacts(self) -> None:
        """Scan the default data/models directories for existing artifacts.

        This lets the dashboard pick up where CLI scripts left off — if the
        user already extracted activations via the CLI, the dashboard will
        detect the HDF5 file and mark that stage as complete.
        """
        cfg = self.config

        # Check for prompt files
        benign_path = cfg.prompts_dir / "benign.jsonl"
        jailbreak_path = cfg.prompts_dir / "jailbreaks.jsonl"
        if benign_path.exists() and jailbreak_path.exists():
            self.benign_path = benign_path
            self.jailbreak_path = jailbreak_path
            self.prompts_ready = True
            logger.info("Detected existing prompts: %s, %s", benign_path, jailbreak_path)

        # Check for activation store
        store_candidates = list(cfg.activations_dir.glob("*.h5"))
        if store_candidates:
            self.store_path = store_candidates[0]
            self.activations_ready = True
            logger.info("Detected existing activation store: %s", self.store_path)

        # Check for splits
        splits_path = cfg.data_dir / "splits.json"
        if splits_path.exists():
            self.splits_path = splits_path
            self.splits_ready = True
            logger.info("Detected existing splits: %s", splits_path)

        # Check for trained model
        model_candidates = list(cfg.models_dir.glob("*/ensemble_config.json"))
        if model_candidates:
            model_dir = model_candidates[0].parent
            self.model_path = model_dir
            self.model_ready = True
            logger.info("Detected existing model: %s", model_dir)
