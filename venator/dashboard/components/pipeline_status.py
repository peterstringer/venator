"""Sidebar branding and stats component.

Renders the VENATOR header and optional pipeline stats in the sidebar.
Navigation with status icons is handled by the page titles in app.py —
this module provides only the supplementary sidebar content.
"""

from __future__ import annotations

import json
import logging

import streamlit as st

from venator.dashboard.state import PipelineState

logger = logging.getLogger(__name__)


def render_sidebar_header(state: PipelineState) -> None:
    """Render the VENATOR header and pipeline stats in the sidebar.

    The sidebar contains:
    1. VENATOR branding header
    2. Progress summary (N/4 pipeline steps complete)
    3. Model stats when a trained model is available

    Args:
        state: Current pipeline state.
    """
    with st.sidebar:
        # Header
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <h2 style="margin: 0; letter-spacing: 0.15em; font-weight: 700;">
                    \U0001f3af VENATOR
                </h2>
                <p style="margin: 0; font-size: 0.75rem; color: #888; letter-spacing: 0.05em;">
                    Jailbreak Detection Pipeline
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Progress summary
        progress = state.get_progress()
        n_complete = sum(1 for _, done in progress if done)
        st.caption(f"{n_complete}/4 pipeline steps complete")

        # Model stats (from pipeline_meta.json if available)
        if state.model_ready and state.model_path:
            meta_path = state.model_path / "pipeline_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    model_id = meta.get("model_id", "")
                    model_short = model_id.split("/")[-1] if "/" in model_id else model_id
                    layer = meta.get("layer", "?")
                    st.caption(f"Model: {model_short} | Layer: {layer}")
                except Exception:
                    logger.debug("Could not read pipeline_meta.json for sidebar stats")
