"""Activation extraction page — run extraction with progress tracking."""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("2. Extract Activations")
st.markdown(
    "Extract hidden state activations from an LLM for each prompt in the dataset."
)

if not state.is_stage_available(2):
    st.warning("Complete the **Data** stage first to unlock extraction.")
elif state.activations_ready:
    st.success("Activations extracted.")
    st.metric("Activation store", str(state.store_path) if state.store_path else "—")
else:
    st.info(
        "Ready to extract. Configure model and layers, then run extraction. "
        "Or run `scripts/extract_activations.py` from the CLI."
    )

st.divider()
st.caption("Full implementation coming in Phase 6.2.")
