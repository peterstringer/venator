"""Streamlit dashboard main entry point.

Launch with: streamlit run venator/dashboard/app.py

Provides a 5-page dashboard UI with persistent sidebar progress tracking.
"""

from __future__ import annotations

import streamlit as st

from venator.dashboard.components.pipeline_status import render_pipeline_sidebar
from venator.dashboard.state import PipelineState

# Must be the first Streamlit call
st.set_page_config(page_title="Venator", page_icon="\U0001f3af", layout="wide")

# Initialize pipeline state (auto-detects existing artifacts on first load)
state = PipelineState()

# Define pages
pages = [
    st.Page("pages/1_pipeline.py", title="Pipeline", url_path="pipeline"),
    st.Page("pages/2_results.py", title="Results", url_path="results"),
    st.Page("pages/3_explore.py", title="Explore", url_path="explore"),
    st.Page("pages/4_detect.py", title="Live Detection", url_path="detect"),
    st.Page("pages/5_ablations.py", title="Ablations", url_path="ablations"),
]

pg = st.navigation(pages)

# Render sidebar on every page
render_pipeline_sidebar(state, current_page=pg.url_path)

pg.run()
