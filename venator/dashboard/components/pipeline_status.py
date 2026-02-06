"""Sidebar pipeline progress tracker component.

Renders a vertical step indicator in the Streamlit sidebar showing which
pipeline stages are complete, which is current, and which are locked.
"""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState


# Stage display config: (number, name, page_key)
_STAGES = [
    (1, "Data", "data"),
    (2, "Extract", "extract"),
    (3, "Split", "split"),
    (4, "Train", "train"),
    (5, "Evaluate", "evaluate"),
    (6, "Detect", "detect"),
    (7, "Ablations", "ablations"),
]


def render_pipeline_sidebar(state: PipelineState, current_page: str = "") -> None:
    """Render the pipeline progress tracker in the sidebar.

    Shows a vertical list of pipeline stages with status indicators:
    - Complete stages are marked with a checkmark
    - The current stage is highlighted
    - Locked stages (prerequisites not met) are grayed out

    Args:
        state: Current pipeline state.
        current_page: Key of the currently active page (for highlighting).
    """
    with st.sidebar:
        # Header
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <h2 style="margin: 0; letter-spacing: 0.15em; font-weight: 700;">
                    VENATOR
                </h2>
                <p style="margin: 0; font-size: 0.75rem; color: #888; letter-spacing: 0.05em;">
                    Jailbreak Detection Pipeline
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Pipeline stages
        progress = state.get_progress()
        progress_map = {name: done for name, done in progress}

        for stage_num, stage_name, page_key in _STAGES:
            available = state.is_stage_available(stage_num)
            # Map stage names to progress (Detect and Ablations don't have
            # their own progress flag — they use prerequisites)
            is_complete = progress_map.get(stage_name, False)
            is_current = current_page == page_key

            if is_complete:
                icon = ":white_check_mark:"
                label = f"{icon} **{stage_num}. {stage_name}**"
                st.markdown(label)
            elif is_current:
                label = f":arrow_right: **{stage_num}. {stage_name}**"
                st.markdown(label)
            elif available:
                label = f":radio_button: {stage_num}. {stage_name}"
                st.markdown(label)
            else:
                label = f":lock: {stage_num}. {stage_name}"
                st.markdown(
                    f"<span style='color: #666;'>{label}</span>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # Summary stats
        n_complete = sum(1 for _, done in progress if done)
        st.caption(f"{n_complete}/5 stages complete")
