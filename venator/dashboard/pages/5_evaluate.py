"""Evaluation page — test metrics, ROC curves, score distributions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore[import-untyped]

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import (
    detector_comparison_chart,
    precision_recall_chart,
    roc_curve_chart,
    score_distribution_chart,
)
from venator.dashboard.components.prompt_table import (
    render_error_analysis,
    render_prompt_explorer,
)
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager
from venator.detection.ensemble import DetectorEnsemble
from venator.detection.metrics import compute_threshold_curves, evaluate_detector

state = PipelineState()
config = state.config

st.header("5. Evaluate")
st.markdown(
    "Evaluate the trained detector ensemble on held-out test data "
    "(benign + jailbreak) and view comprehensive metrics."
)

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(5):
    st.warning("Complete the **Train** stage first to unlock evaluation.")
    st.stop()


# ------------------------------------------------------------------
# Display function — renders the full results view from cached data
# ------------------------------------------------------------------


def _show_results(eval_data: dict) -> None:
    """Render the full evaluation results from cached display data."""
    metrics = eval_data["metrics"]
    curves = eval_data["curves"]
    benign_scores = np.array(eval_data["benign_scores"])
    jailbreak_scores = np.array(eval_data["jailbreak_scores"])
    threshold = eval_data["threshold"]
    detector_aurocs = eval_data["detector_aurocs"]
    prompts = eval_data["prompts"]
    labels = np.array(eval_data["labels"])
    scores = np.array(eval_data["scores"])
    predictions = np.array(eval_data["predictions"])

    # --- Headline Metrics ---
    st.subheader("Headline Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("AUROC", f"{metrics['auroc']:.4f}")
    with m_col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with m_col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    with m_col4:
        st.metric("FPR@threshold", f"{metrics.get('false_positive_rate', 0):.4f}")

    # --- Visualizations (2x2 grid) ---
    st.subheader("Visualizations")

    viz_top1, viz_top2 = st.columns(2)
    with viz_top1:
        fig = score_distribution_chart(benign_scores, jailbreak_scores, threshold)
        st.plotly_chart(fig, width="stretch")

    with viz_top2:
        roc_fpr = np.array(curves["roc_fpr"])
        roc_tpr = np.array(curves["roc_tpr"])
        # Find operating point on ROC
        op_fpr = metrics.get("false_positive_rate")
        op_tpr = metrics.get("true_positive_rate")
        fig = roc_curve_chart(roc_fpr, roc_tpr, metrics["auroc"], op_fpr, op_tpr)
        st.plotly_chart(fig, width="stretch")

    viz_bot1, viz_bot2 = st.columns(2)
    with viz_bot1:
        fig = detector_comparison_chart(detector_aurocs)
        st.plotly_chart(fig, width="stretch")

    with viz_bot2:
        # Compute PR curve from sklearn for smooth curves
        pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
        fig = precision_recall_chart(pr_recall, pr_precision, metrics["auprc"])
        st.plotly_chart(fig, width="stretch")

    # --- Detailed Results ---
    st.subheader("Detailed Results")

    with st.expander("Prompt Explorer", expanded=False):
        render_prompt_explorer(prompts, labels, scores, predictions, threshold)

    render_error_analysis(prompts, labels, scores, predictions)


# ------------------------------------------------------------------
# Already complete — show cached results
# ------------------------------------------------------------------

if state.evaluation_ready and state.eval_results:
    st.success("Evaluation complete.")

    eval_data = st.session_state.get("_eval_display_data")
    if eval_data is not None:
        _show_results(eval_data)
    else:
        # Cached display data lost (e.g., server restart) — show basic metrics
        st.subheader("Results (summary)")
        metrics = state.eval_results
        primary_keys = ["auroc", "auprc", "precision", "recall", "f1", "accuracy"]
        cols = st.columns(3)
        for i, key in enumerate(primary_keys):
            if key in metrics:
                with cols[i % 3]:
                    st.metric(key.upper(), f"{metrics[key]:.4f}")
        st.info("Re-run evaluation to see full visualizations and prompt explorer.")

    col_re, col_cont = st.columns(2)
    with col_re:
        if st.button("Re-evaluate"):
            state.evaluation_ready = False
            state.eval_results = None
            if "_eval_display_data" in st.session_state:
                del st.session_state["_eval_display_data"]
            st.rerun()
    with col_cont:
        if st.button("Continue  \u2192", type="primary"):
            st.switch_page("pages/6_detect.py")
    st.stop()

# ------------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------------

st.divider()

if st.button("Run Evaluation", type="primary"):
    # Load components
    store = ActivationStore(state.store_path)
    splits = SplitManager.load_splits(state.splits_path)
    ensemble = DetectorEnsemble.load(state.model_path)

    # Read layer from pipeline metadata
    meta_path = Path(state.model_path) / "pipeline_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        layer = meta["layer"]
    else:
        layer = store.layers[len(store.layers) // 2]

    with st.status("Running evaluation...", expanded=True) as status:
        st.write(f"Scoring test set on layer {layer}...")

        # Get test data
        test_benign_indices = splits["test_benign"].indices.tolist()
        test_jailbreak_indices = splits["test_jailbreak"].indices.tolist()

        X_test_benign = store.get_activations(layer, indices=test_benign_indices)
        X_test_jailbreak = store.get_activations(layer, indices=test_jailbreak_indices)

        X_test = np.vstack([X_test_benign, X_test_jailbreak])
        labels = np.concatenate([
            np.zeros(len(X_test_benign), dtype=np.int64),
            np.ones(len(X_test_jailbreak), dtype=np.int64),
        ])

        st.write(
            f"Test set: {len(X_test_benign)} benign + "
            f"{len(X_test_jailbreak)} jailbreak = {len(X_test)} total"
        )

        # Score
        result = ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=ensemble.threshold_
        )

        # Per-detector AUROC
        detector_aurocs = {"Ensemble": metrics["auroc"]}
        for det_result in result.detector_results:
            det_metrics = evaluate_detector(det_result.normalized_scores, labels)
            metrics[f"auroc_{det_result.name}"] = det_metrics["auroc"]
            detector_aurocs[det_result.name] = det_metrics["auroc"]

        # Threshold curves for ROC/PR plotting
        curves = compute_threshold_curves(result.ensemble_scores, labels)

        # Get test prompts for explorer
        all_test_indices = test_benign_indices + test_jailbreak_indices
        test_prompts = store.get_prompts(indices=all_test_indices)

        status.update(label="Evaluation complete!", state="complete")

    # Cache all display data in session state
    eval_data = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "curves": {k: v.tolist() for k, v in curves.items()},
        "benign_scores": result.ensemble_scores[:len(X_test_benign)].tolist(),
        "jailbreak_scores": result.ensemble_scores[len(X_test_benign):].tolist(),
        "threshold": float(ensemble.threshold_),
        "detector_aurocs": detector_aurocs,
        "prompts": test_prompts,
        "labels": labels.tolist(),
        "scores": result.ensemble_scores.tolist(),
        "predictions": result.is_anomaly.astype(int).tolist(),
    }
    st.session_state["_eval_display_data"] = eval_data

    # Update state
    state.eval_results = eval_data["metrics"]
    state.evaluation_ready = True

    # Show results
    _show_results(eval_data)

    if st.button("Continue to Detect  \u2192", type="primary", key="post_eval_continue"):
        st.switch_page("pages/6_detect.py")
