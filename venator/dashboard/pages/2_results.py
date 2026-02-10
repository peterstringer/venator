"""Results page — demo-quality evaluation with headline metrics and charts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore[import-untyped]

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import (
    detector_comparison_chart,
    detector_comparison_grouped_bar,
    precision_recall_chart,
    roc_curve_chart,
    score_distribution_chart,
)
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager
from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorType,
)
from venator.detection.metrics import compute_threshold_curves, evaluate_detector

state = PipelineState()
config = state.config

st.header("Results")
st.markdown(
    "Evaluate the trained detector on held-out test data "
    "(benign + jailbreak) and view comprehensive metrics."
)

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(2):
    st.warning("Complete the **Pipeline** (train) stage first to unlock results.")
    st.stop()


# ------------------------------------------------------------------
# Display function — renders the full results view from cached data
# ------------------------------------------------------------------


def _show_results(eval_data: dict) -> None:
    """Render the full evaluation results — flat layout, no tabs."""
    metrics = eval_data["metrics"]
    curves = eval_data["curves"]
    benign_scores = np.array(eval_data["benign_scores"])
    jailbreak_scores = np.array(eval_data["jailbreak_scores"])
    threshold = eval_data["threshold"]
    detector_aurocs = eval_data["detector_aurocs"]
    per_detector_info = eval_data.get("per_detector_info", [])

    # Primary detector info
    primary_info = eval_data.get("primary_detector")
    ensemble_type = eval_data.get("ensemble_type", "unsupervised")
    if primary_info:
        st.caption(
            f"Primary detector: **{primary_info}** | Ensemble type: **{ensemble_type}**"
        )

    # --- Headline Metrics ---
    st.subheader("Headline Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("AUROC", f"{metrics['auroc']:.4f}")
    with m_col2:
        st.metric("AUPRC", f"{metrics.get('auprc', 0):.4f}")
    with m_col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    with m_col4:
        st.metric("FPR@threshold", f"{metrics.get('false_positive_rate', 0):.4f}")

    # --- Visualizations (2x2 grid) ---
    viz_top1, viz_top2 = st.columns(2)
    with viz_top1:
        fig = score_distribution_chart(benign_scores, jailbreak_scores, threshold)
        st.plotly_chart(fig, width="stretch")

    with viz_top2:
        roc_fpr = np.array(curves["roc_fpr"])
        roc_tpr = np.array(curves["roc_tpr"])
        op_fpr = metrics.get("false_positive_rate")
        op_tpr = metrics.get("true_positive_rate")
        fig = roc_curve_chart(roc_fpr, roc_tpr, metrics["auroc"], op_fpr, op_tpr)
        st.plotly_chart(fig, width="stretch")

    viz_bot1, viz_bot2 = st.columns(2)
    with viz_bot1:
        if per_detector_info:
            unsup_aurocs = {
                d["name"]: d["auroc"] for d in per_detector_info
                if d.get("type") == "unsup"
            }
            sup_aurocs = {
                d["name"]: d["auroc"] for d in per_detector_info
                if d.get("type") == "sup"
            }
            if unsup_aurocs and sup_aurocs:
                fig = detector_comparison_grouped_bar(unsup_aurocs, sup_aurocs)
                st.plotly_chart(fig, width="stretch")
            else:
                fig = detector_comparison_chart(detector_aurocs)
                st.plotly_chart(fig, width="stretch")
        else:
            fig = detector_comparison_chart(detector_aurocs)
            st.plotly_chart(fig, width="stretch")

    with viz_bot2:
        labels = np.array(eval_data["labels"])
        scores = np.array(eval_data["scores"])
        pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
        fig = precision_recall_chart(pr_recall, pr_precision, metrics.get("auprc", 0))
        st.plotly_chart(fig, width="stretch")

    # --- Per-detector breakdown table ---
    if per_detector_info:
        st.subheader("Detector Comparison")
        df = pd.DataFrame(per_detector_info)
        display_cols = [c for c in ["name", "tag", "type", "auroc", "auprc", "f1", "fpr_at_95_tpr"]
                       if c in df.columns]
        st.dataframe(df[display_cols].round(4), use_container_width=True, hide_index=True)


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
        st.info("Re-run evaluation to see full visualizations.")

    # Download results
    st.divider()
    _export_eval_data = eval_data if eval_data is not None else {"metrics": state.eval_results}
    _export_safe = {
        k: v for k, v in _export_eval_data.items()
        if k not in ("prompts",)
    }
    json_str = json.dumps(_export_safe, indent=2, default=str)
    st.download_button(
        "Download Evaluation Results (JSON)",
        data=json_str,
        file_name="evaluation_results.json",
        mime="application/json",
    )

    col_re, _ = st.columns(2)
    with col_re:
        if st.button("Re-evaluate"):
            state.evaluation_ready = False
            state.eval_results = None
            if "_eval_display_data" in st.session_state:
                del st.session_state["_eval_display_data"]
            st.rerun()
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

    # Read layer and ensemble_type from pipeline metadata
    meta_path = Path(state.model_path) / "pipeline_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        layer = meta["layer"]
        ensemble_type = meta.get("ensemble_type", "unsupervised")
    else:
        layer = store.layers[len(store.layers) // 2]
        ensemble_type = "unsupervised"

    with st.status("Running evaluation...", expanded=True) as status:
        st.write(f"Scoring test set on layer {layer} ({ensemble_type} ensemble)...")

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

        # Determine primary detector
        primary_name = meta.get("primary_name") if meta_path.exists() else None
        primary_threshold = meta.get("primary_threshold") if meta_path.exists() else None

        _priority = ["linear_probe", "pca_mahalanobis", "contrastive_mahalanobis",
                     "isolation_forest", "autoencoder", "contrastive_direction"]
        det_names_in_ensemble = {n for n, _, _ in ensemble.detectors}
        if primary_name is None or primary_name not in det_names_in_ensemble:
            for candidate in _priority:
                if candidate in det_names_in_ensemble:
                    primary_name = candidate
                    break

        primary_det = None
        for n, d, _ in ensemble.detectors:
            if n == primary_name:
                primary_det = d
                break

        # Score primary detector
        if primary_det is not None:
            primary_scores = primary_det.score(X_test)
            if primary_threshold is not None:
                metrics = evaluate_detector(primary_scores, labels, threshold=primary_threshold)
            else:
                metrics = evaluate_detector(primary_scores, labels, threshold=ensemble.threshold_)
        else:
            primary_scores = None
            metrics = evaluate_detector(
                ensemble.score(X_test).ensemble_scores, labels, threshold=ensemble.threshold_
            )

        # Score through ensemble (for comparison)
        result = ensemble.score(X_test)

        # Per-detector metrics
        detector_aurocs = {}
        if primary_name:
            detector_aurocs[f"{primary_name} (PRIMARY)"] = metrics["auroc"]
        per_detector_info = []
        for det_result in result.detector_results:
            det_metrics = evaluate_detector(
                det_result.normalized_scores, labels, threshold=ensemble.threshold_
            )
            metrics[f"auroc_{det_result.name}"] = det_metrics["auroc"]
            metrics[f"auprc_{det_result.name}"] = det_metrics["auprc"]
            detector_aurocs[det_result.name] = det_metrics["auroc"]

            det_type_label = "sup" if ensemble.detector_types_.get(
                det_result.name, DetectorType.UNSUPERVISED
            ) == DetectorType.SUPERVISED else "unsup"

            tag = "PRIMARY" if det_result.name == primary_name else "BASELINE"
            per_detector_info.append({
                "name": det_result.name,
                "type": det_type_label,
                "tag": tag,
                "weight": det_result.weight,
                "auroc": det_metrics["auroc"],
                "auprc": det_metrics["auprc"],
                "f1": det_metrics.get("f1", 0.0),
                "fpr_at_95_tpr": det_metrics["fpr_at_95_tpr"],
            })

        # Add ensemble as RETIRED entry
        ensemble_eval = evaluate_detector(
            result.ensemble_scores, labels, threshold=ensemble.threshold_
        )
        per_detector_info.append({
            "name": "ensemble",
            "type": "ensemble",
            "tag": "RETIRED",
            "weight": 0.0,
            "auroc": ensemble_eval["auroc"],
            "auprc": ensemble_eval["auprc"],
            "f1": ensemble_eval.get("f1", 0.0),
            "fpr_at_95_tpr": ensemble_eval["fpr_at_95_tpr"],
        })

        # Sort: PRIMARY first, then BASELINE by AUROC desc, then RETIRED
        _tag_order = {"PRIMARY": 0, "BASELINE": 1, "RETIRED": 2}
        per_detector_info.sort(key=lambda d: (_tag_order.get(d["tag"], 9), -d["auroc"]))

        # Use primary scores for ROC/PR curves
        scores_for_curves = primary_scores if primary_scores is not None else result.ensemble_scores
        curves = compute_threshold_curves(scores_for_curves, labels)

        # Get test prompts for explorer (used by Explore page)
        all_test_indices = test_benign_indices + test_jailbreak_indices
        test_prompts = store.get_prompts(indices=all_test_indices)

        status.update(label="Evaluation complete!", state="complete")

    # Use primary scores for display
    display_scores = scores_for_curves
    display_threshold = primary_threshold if primary_threshold is not None else ensemble.threshold_
    display_predictions = (display_scores > display_threshold).astype(int)

    # Cache all display data in session state
    eval_data = {
        "metrics": {k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
                    for k, v in metrics.items()
                    if not isinstance(v, (list, dict))},
        "curves": {k: v.tolist() for k, v in curves.items()},
        "benign_scores": display_scores[:len(X_test_benign)].tolist(),
        "jailbreak_scores": display_scores[len(X_test_benign):].tolist(),
        "threshold": float(display_threshold),
        "detector_aurocs": detector_aurocs,
        "per_detector_info": per_detector_info,
        "ensemble_type": ensemble_type,
        "primary_detector": primary_name,
        "prompts": test_prompts,
        "labels": labels.tolist(),
        "scores": display_scores.tolist(),
        "predictions": display_predictions.tolist(),
    }

    st.session_state["_eval_display_data"] = eval_data

    # Update state
    state.eval_results = eval_data["metrics"]
    state.evaluation_ready = True

    # Show results
    _show_results(eval_data)

    # Download
    st.divider()
    _export_safe = {
        k: v for k, v in eval_data.items()
        if k not in ("prompts",)
    }
    json_str = json.dumps(_export_safe, indent=2, default=str)
    st.download_button(
        "Download Evaluation Results (JSON)",
        data=json_str,
        file_name="evaluation_results.json",
        mime="application/json",
        key="post_eval_download",
    )
