"""Streamlit dashboard main entry point.

Launch with: streamlit run venator/dashboard/app.py

6-page dashboard grouped into three sections. Quick Run is the default
landing page. Pages that require prerequisites show a message when
clicked before their dependencies are complete.
"""

from __future__ import annotations

import streamlit as st

from venator.dashboard.components.pipeline_status import render_sidebar
from venator.dashboard.state import PipelineState

st.set_page_config(page_title="Venator", layout="wide")

state = PipelineState()

# Inject title above navigation via CSS pseudo-elements on the sidebar header.
# st.sidebar content always renders BELOW st.navigation(), so we target the
# header container ([data-testid="stSidebarHeader"]) which sits above the nav.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Offside&display=swap');

    [data-testid="stSidebarHeader"]::before {
        content: "Venator";
        font-family: 'Offside', cursive;
        font-size: 1.8rem;
        display: block;
        padding: 0.5rem 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page groups separated by dividers (empty keys = no visible header)
pages = {
    "": [
        st.Page("pages/quick_run.py", title="Quick Run", url_path="quick-run", default=True),
    ],
    "Pipeline": [
        st.Page("pages/1_pipeline.py", title="Run Pipeline", url_path="pipeline"),
        st.Page("pages/2_results.py", title="Results", url_path="results"),
        st.Page("pages/5_ablations.py", title="Ablations", url_path="ablations"),
    ],
    "Prompts": [
        st.Page("pages/3_explore.py", title="Explore Prompts", url_path="explore"),
        st.Page("pages/4_detect.py", title="Test Prompts", url_path="detect"),
    ],
}

pg = st.navigation(pages)
render_sidebar(state)
pg.run()
