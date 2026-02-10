"""Streamlit dashboard main entry point.

Launch with: streamlit run venator/dashboard/app.py

Provides a 5-page dashboard UI with status-embedded navigation.
Navigation items show completion status directly in their titles:
  ✅ = stage complete
  →  = available (prerequisites met, not yet done)
  🔒 = locked (prerequisites not met)
"""

from __future__ import annotations

import streamlit as st

from venator.dashboard.components.pipeline_status import render_sidebar_header
from venator.dashboard.state import PipelineState

# Must be the first Streamlit call
st.set_page_config(page_title="Venator", page_icon="\U0001f3af", layout="wide")

# Initialize pipeline state (auto-detects existing artifacts on first load)
state = PipelineState()


def _status_icon(page_key: str) -> str:
    """Get status icon for a page based on pipeline state."""
    # Completion: has this page's work been done?
    completion: dict[str, bool] = {
        "pipeline": state.model_ready,
        "results": state.evaluation_ready,
        "explore": state.evaluation_ready,
        "detect": state.model_ready,
        "ablations": False,  # No tracked completion state
    }
    # Availability: are prerequisites met?
    stage_num = {"pipeline": 1, "results": 2, "explore": 3, "detect": 4, "ablations": 5}

    if completion.get(page_key, False):
        return "\u2705"
    elif state.is_stage_available(stage_num[page_key]):
        return "\u2192"
    else:
        return "\U0001f512"


# Build pages with dynamic status titles
pages = [
    st.Page("pages/1_pipeline.py", title=f"{_status_icon('pipeline')} Pipeline", url_path="pipeline"),
    st.Page("pages/2_results.py", title=f"{_status_icon('results')} Results", url_path="results"),
    st.Page("pages/3_explore.py", title=f"{_status_icon('explore')} Explore", url_path="explore"),
    st.Page("pages/4_detect.py", title=f"{_status_icon('detect')} Live Detection", url_path="detect"),
    st.Page("pages/5_ablations.py", title=f"{_status_icon('ablations')} Ablations", url_path="ablations"),
]

pg = st.navigation(pages)

# Render minimal sidebar (header + stats only — no duplicate page list)
render_sidebar_header(state)

pg.run()
