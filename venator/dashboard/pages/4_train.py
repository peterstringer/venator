"""Detector training page — train detectors and show convergence."""

from __future__ import annotations

import streamlit as st

from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("4. Train Detectors")
st.markdown(
    "Train the anomaly detection ensemble (PCA+Mahalanobis, Isolation Forest, "
    "Autoencoder) on benign-only training data."
)

if not state.is_stage_available(4):
    st.warning("Complete the **Split** stage first to unlock training.")
elif state.model_ready:
    st.success("Model trained.")
    st.metric("Model directory", str(state.model_path) if state.model_path else "—")
    if state.train_metrics:
        st.subheader("Training Metrics")
        for key, value in state.train_metrics.items():
            st.metric(key, f"{value:.4f}" if isinstance(value, float) else str(value))
else:
    st.info(
        "Ready to train. Or run `scripts/train_detector.py` from the CLI."
    )

st.divider()
st.caption("Full implementation coming in Phase 6.2.")
