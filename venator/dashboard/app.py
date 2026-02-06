"""Streamlit dashboard main entry point.

Launch with: streamlit run venator/dashboard/app.py

Provides a 7-step guided pipeline UI with persistent sidebar progress tracking.
"""

from __future__ import annotations

import streamlit as st

from venator.dashboard.components.pipeline_status import render_pipeline_sidebar
from venator.dashboard.state import PipelineState

# Must be the first Streamlit call
st.set_page_config(page_title="Venator", page_icon="\U0001f3af", layout="wide")

# Initialize pipeline state (auto-detects existing artifacts on first load)
state = PipelineState()

# Define pages — url_path matches pipeline_status.py page keys
pages = [
    st.Page("pages/1_data.py", title="1. Data", url_path="data"),
    st.Page("pages/2_extract.py", title="2. Extract", url_path="extract"),
    st.Page("pages/3_split.py", title="3. Split", url_path="split"),
    st.Page("pages/4_train.py", title="4. Train", url_path="train"),
    st.Page("pages/5_evaluate.py", title="5. Evaluate", url_path="evaluate"),
    st.Page("pages/6_detect.py", title="6. Detect", url_path="detect"),
    st.Page("pages/7_ablations.py", title="7. Ablations", url_path="ablations"),
]

pg = st.navigation(pages)

# Render sidebar on every page
render_pipeline_sidebar(state, current_page=pg.url_path)

pg.run()
