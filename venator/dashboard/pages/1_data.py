"""Data collection page — collect/upload prompts and preview datasets."""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("1. Data Collection")
st.markdown(
    "Collect or upload benign and jailbreak prompt datasets for the pipeline."
)

if state.prompts_ready:
    st.success("Prompt datasets loaded.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Benign prompts file", str(state.benign_path) if state.benign_path else "—")
    with col2:
        st.metric("Jailbreak prompts file", str(state.jailbreak_path) if state.jailbreak_path else "—")
else:
    st.info(
        "No prompt datasets detected. Upload JSONL files or run "
        "`scripts/collect_prompts.py` from the CLI."
    )

st.divider()
st.caption("Full implementation coming in Phase 6.2.")
