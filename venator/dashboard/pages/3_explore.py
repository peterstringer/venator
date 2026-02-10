"""Explore page — prompt-level deep dive with FP/FN analysis.

Respects the detector selector: FP/FN analysis uses whichever detector
the user picks from the dropdown (defaults to the top-ranked detector).
"""

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

if eval_data is None or "detector_results" not in eval_data:
    st.info(
        "Evaluation display data not available. "
        "Go to the **Results** page and run evaluation first."
    )
    st.stop()

# ------------------------------------------------------------------
# Detector selector
# ------------------------------------------------------------------

detector_results = eval_data["detector_results"]
prompts = eval_data["prompts"]
labels = np.array(eval_data["labels"])

detector_options = {r["display_name"]: r for r in detector_results}

selected_name = st.selectbox(
    "Detector",
    options=list(detector_options.keys()),
    index=0,
    key="explore_detector_select",
)

selected = detector_options[selected_name]

# Derive scores and predictions from selected detector
# Labels are ordered: benign first, jailbreak second — match the score order
scores = np.array(selected["scores_benign"] + selected["scores_jailbreak"])
threshold = selected["threshold"]
predictions = (scores > threshold).astype(int)

# ------------------------------------------------------------------
# Summary stats
# ------------------------------------------------------------------

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
