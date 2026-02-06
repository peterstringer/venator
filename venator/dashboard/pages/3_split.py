"""Data splitting page — create splits and verify methodology constraints."""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("3. Split Data")
st.markdown(
    "Create train/validation/test splits with proper methodology constraints: "
    "jailbreaks are reserved for the test set only."
)

if not state.is_stage_available(3):
    st.warning("Complete the **Extract** stage first to unlock splitting.")
elif state.splits_ready:
    st.success("Splits created.")
    st.metric("Splits file", str(state.splits_path) if state.splits_path else "—")
else:
    st.info(
        "Ready to create splits. Or run `scripts/create_splits.py` from the CLI."
    )

st.divider()
st.caption("Full implementation coming in Phase 6.2.")
