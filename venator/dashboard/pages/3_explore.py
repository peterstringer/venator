"""Explore page — prompt-level deep dive with FP/FN analysis."""

from __future__ import annotations

import numpy as np
import streamlit as st

from venator.dashboard.components.prompt_table import (
    render_error_analysis,
    render_prompt_explorer,
)
from venator.dashboard.state import PipelineState

state = PipelineState()

st.header("Explore")
st.markdown("Prompt-level analysis of test set predictions.")

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(3):
    st.warning("Complete **Results** evaluation first to unlock the prompt explorer.")
    st.stop()

# ------------------------------------------------------------------
# Load evaluation data from session state
# ------------------------------------------------------------------

eval_data = st.session_state.get("_eval_display_data")

if eval_data is None:
    st.info(
        "Evaluation display data not available. "
        "Go to the **Results** page and run evaluation first."
    )
    st.stop()

# Extract arrays
prompts = eval_data["prompts"]
labels = np.array(eval_data["labels"])
scores = np.array(eval_data["scores"])
predictions = np.array(eval_data["predictions"])
threshold = eval_data["threshold"]

# Summary stats
n_total = len(labels)
n_jailbreak = int(labels.sum())
n_benign = n_total - n_jailbreak
n_correct = int((predictions == labels).sum())

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
with stat_col1:
    st.metric("Total Prompts", n_total)
with stat_col2:
    st.metric("Benign", n_benign)
with stat_col3:
    st.metric("Jailbreak", n_jailbreak)
with stat_col4:
    st.metric("Accuracy", f"{n_correct / n_total:.1%}" if n_total > 0 else "N/A")

# ------------------------------------------------------------------
# Prompt Explorer
# ------------------------------------------------------------------

st.subheader("Prompt Explorer")
render_prompt_explorer(prompts, labels, scores, predictions, threshold)

# ------------------------------------------------------------------
# Error Analysis
# ------------------------------------------------------------------

st.subheader("Error Analysis")
render_error_analysis(prompts, labels, scores, predictions)
