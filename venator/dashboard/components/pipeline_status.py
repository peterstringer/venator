"""Sidebar pipeline progress tracker component.

Renders a vertical step indicator in the Streamlit sidebar showing which
pipeline pages are available, which is current, and which are locked.
"""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState


# Page display config: (number, name, page_key)
_STAGES = [
    (1, "Pipeline", "pipeline"),
    (2, "Results", "results"),
    (3, "Explore", "explore"),
    (4, "Live Detection", "detect"),
    (5, "Ablations", "ablations"),
]


def render_pipeline_sidebar(state: PipelineState, current_page: str = "") -> None:
    """Render the pipeline progress tracker in the sidebar.

    Shows a vertical list of pages with status indicators:
    - The Pipeline page shows sub-step progress (Data/Extract/Split/Train)
    - Locked pages (prerequisites not met) are grayed out
    - The current page is highlighted

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

        # Pipeline pages
        for stage_num, stage_name, page_key in _STAGES:
            available = state.is_stage_available(stage_num)
            is_current = current_page == page_key

            if is_current:
                label = f":arrow_right: **{stage_name}**"
                st.markdown(label)
            elif available:
                label = f":radio_button: {stage_name}"
                st.markdown(label)
            else:
                label = f":lock: {stage_name}"
                st.markdown(
                    f"<span style='color: #666;'>{label}</span>",
                    unsafe_allow_html=True,
                )

            # Show sub-steps for the Pipeline page
            if stage_name == "Pipeline":
                substeps = state.get_progress()
                for sub_name, sub_done in substeps:
                    sub_icon = ":white_check_mark:" if sub_done else ":radio_button:"
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{sub_icon} {sub_name}")

        st.divider()

        # Summary stats — count pipeline sub-steps
        progress = state.get_progress()
        n_complete = sum(1 for _, done in progress if done)
        st.caption(f"{n_complete}/4 pipeline steps complete")
