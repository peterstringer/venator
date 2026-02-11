"""Sidebar component with reset functionality.

Adds a reset button below the navigation. Page navigation itself is
handled by st.navigation() in app.py.
"""

from __future__ import annotations

import logging
import shutil

import streamlit as st

from venator.dashboard.state import PipelineState

logger = logging.getLogger(__name__)


def render_sidebar(state: PipelineState) -> None:
    """Render the reset button in the sidebar.

    Args:
        state: Current pipeline state.
    """
    with st.sidebar:
        with st.popover("Reset Everything", use_container_width=True):
            st.markdown(
                "**This will delete all data, models, and cached results** "
                "so you can start completely from scratch."
            )
            if st.button("Yes, reset everything", type="primary", key="_sidebar_confirm_reset"):
                _reset_everything(state)
                st.rerun()


def _reset_everything(state: PipelineState) -> None:
    """Delete all on-disk artifacts and clear session state."""
    cfg = state.config

    # Delete on-disk artifacts
    for dir_path in [cfg.prompts_dir, cfg.activations_dir, cfg.models_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info("Deleted %s", dir_path)

    # Delete splits files
    if cfg.data_dir.exists():
        for f in cfg.data_dir.glob("splits*.json"):
            f.unlink()
            logger.info("Deleted %s", f)

    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
