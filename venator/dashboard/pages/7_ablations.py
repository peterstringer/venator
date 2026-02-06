"""Ablation studies page — layer/PCA/detector comparison."""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("7. Ablation Studies")
st.markdown(
    "Compare detector performance across different layers, PCA dimensions, "
    "and detector configurations."
)

if not state.is_stage_available(7):
    st.warning(
        "Complete the **Extract** and **Split** stages first to unlock ablations."
    )
else:
    st.info(
        "Select an ablation type to run. Or run `scripts/run_ablations.py` "
        "from the CLI."
    )
    ablation_type = st.selectbox(
        "Ablation type",
        ["Layer selection", "PCA dimensions", "Detector comparison"],
    )
    if st.button("Run Ablation"):
        st.warning(f"Running {ablation_type} ablation... Coming in Phase 6.2.")

st.divider()
st.caption("Full implementation coming in Phase 6.2.")
