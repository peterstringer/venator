"""Evaluation page — test metrics, ROC curves, score distributions."""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("5. Evaluate")
st.markdown(
    "Evaluate the trained detector ensemble on held-out test data "
    "(benign + jailbreak) and view metrics."
)

if not state.is_stage_available(5):
    st.warning("Complete the **Train** stage first to unlock evaluation.")
elif state.evaluation_ready and state.eval_results:
    st.success("Evaluation complete.")
    st.subheader("Results")
    primary_keys = ["auroc", "auprc", "precision", "recall", "f1", "accuracy"]
    cols = st.columns(3)
    for i, key in enumerate(primary_keys):
        if key in state.eval_results:
            with cols[i % 3]:
                st.metric(key.upper(), f"{state.eval_results[key]:.4f}")
else:
    st.info(
        "Ready to evaluate. Or run `scripts/evaluate.py` from the CLI."
    )

st.divider()
st.caption("Full implementation coming in Phase 6.2.")
