"""Live detection page — single-prompt jailbreak detection."""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("6. Detect")
st.markdown(
    "Run live single-prompt jailbreak detection using the trained model."
)

if not state.is_stage_available(6):
    st.warning("Complete the **Train** stage first to unlock detection.")
else:
    st.info(
        "Enter a prompt below to check whether it would be flagged as a "
        "jailbreak attempt."
    )
    prompt = st.text_area("Prompt", height=150, placeholder="Type a prompt here...")
    if st.button("Detect", disabled=not prompt):
        st.warning("Live detection requires a loaded model. Coming in Phase 6.2.")

st.divider()
st.caption("Full implementation coming in Phase 6.2.")
