"""Results page — per-detector evaluation with ranked comparison table.

Every trained detector (including ensembles) appears as its own row in the
results, ranked by AUROC descending. The top-ranked detector gets headline
metrics; a dropdown lets the user switch visualizations to any detector.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import (
    ensemble_roc_comparison,
    precision_recall_chart,
    roc_curve_chart,
    score_distribution_chart,
)
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager
from venator.detection.ensemble import DetectorEnsemble, DetectorType
from venator.detection.metrics import evaluate_all_detectors

state = PipelineState()
config = state.config

st.header("Results")
st.markdown(
    "Evaluate all trained detectors on held-out test data "
    "and compare their performance."
)

# Display name mapping
_DISPLAY_NAMES: dict[str, str] = {
    "linear_probe": "Linear Probe",
    "contrastive_mahalanobis": "Contrastive Mahalanobis",
    "contrastive_direction": "Contrastive Direction",
    "mlp_probe": "MLP Probe",
    "autoencoder": "Autoencoder",
    "pca_mahalanobis": "PCA + Mahalanobis",
    "isolation_forest": "Isolation Forest",
    "ensemble": "Custom Ensemble",
}


# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(2):
    st.warning("Complete the **Pipeline** (train) stage first to unlock results.")
    st.stop()


# ------------------------------------------------------------------
# Ensemble scorer adapter
# ------------------------------------------------------------------


class _EnsembleScorer:
    """Adapter so DetectorEnsemble can be passed to evaluate_all_detectors."""

    def __init__(self, ensemble: DetectorEnsemble) -> None:
        self._ensemble = ensemble

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._ensemble.score(X).ensemble_scores


# ------------------------------------------------------------------
# Serialization helpers
# ------------------------------------------------------------------


def _result_to_dict(r: object) -> dict:
    """Convert an EvaluationResult to a JSON-serializable dict."""
    return {
        "name": r.detector_name,
        "display_name": r.display_name,
        "type": r.detector_type,
        "auroc": float(r.auroc),
        "auprc": float(r.auprc),
        "f1": float(r.f1),
        "precision": float(r.precision),
        "recall": float(r.recall),
        "fpr": float(r.fpr),
        "threshold": float(r.threshold),
        "scores_benign": r.scores_benign.tolist(),
        "scores_jailbreak": r.scores_jailbreak.tolist(),
        "roc_fpr": r.roc_fpr.tolist(),
        "roc_tpr": r.roc_tpr.tolist(),
        "pr_precision": r.pr_precision.tolist(),
        "pr_recall": r.pr_recall.tolist(),
    }


# ------------------------------------------------------------------
# Display function
# ------------------------------------------------------------------


def _color_metric(val: object) -> str:
    """Color code: green >= 0.95, amber >= 0.85, red < 0.85."""
    if isinstance(val, (float, int)):
        if val >= 0.95:
            return "color: #2ca065"
        if val >= 0.85:
            return "color: #d4a017"
        return "color: #db4052"
    return ""


def _show_results(eval_data: dict) -> None:
    """Render the full evaluation results from cached data."""
    detector_results = eval_data["detector_results"]
    n_benign = eval_data["n_benign"]
    n_jailbreak = eval_data["n_jailbreak"]

    if not detector_results:
        st.warning("No detector results found.")
        return

    st.caption(
        f"Test set: **{n_benign}** benign + **{n_jailbreak}** jailbreak "
        f"(never seen in training)"
    )

    # --- Headline Metrics (top-ranked detector) ---
    top = detector_results[0]
    st.subheader("Headline Metrics")
    st.caption(f"Top-ranked detector: **{top['display_name']}**")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("AUROC", f"{top['auroc']:.4f}")
    with m_col2:
        st.metric("AUPRC", f"{top['auprc']:.4f}")
    with m_col3:
        st.metric("Recall", f"{top['recall']:.4f}")
    with m_col4:
        st.metric("FPR", f"{top['fpr']:.4f}")

    # --- Ranked Detector Table ---
    st.subheader("Detector Comparison")

    table_rows = []
    for i, r in enumerate(detector_results):
        rank_str = f"\u2605 {i + 1}" if i == 0 else str(i + 1)
        table_rows.append({
            "Rank": rank_str,
            "Detector": r["display_name"],
            "Type": r["type"],
            "AUROC": r["auroc"],
            "AUPRC": r["auprc"],
            "F1": r["f1"],
            "FPR": r["fpr"],
        })

    df = pd.DataFrame(table_rows)

    styled = df.style.map(
        _color_metric, subset=["AUROC", "AUPRC"]
    ).format({
        "AUROC": "{:.4f}",
        "AUPRC": "{:.4f}",
        "F1": "{:.4f}",
        "FPR": "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Visualizations with detector selector ---
    st.subheader("Visualizations")

    detector_options = [r["display_name"] for r in detector_results]

    sel_col, toggle_col = st.columns([3, 1])
    with sel_col:
        selected_display = st.selectbox(
            "Showing results for",
            options=detector_options,
            index=0,
            key="viz_detector_select",
        )
    with toggle_col:
        compare_all = st.checkbox("Compare All (ROC)", key="compare_all_roc")

    selected = next(
        r for r in detector_results if r["display_name"] == selected_display
    )

    # Charts — top row: score distribution + ROC
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        benign_scores = np.array(selected["scores_benign"])
        jailbreak_scores = np.array(selected["scores_jailbreak"])
        fig = score_distribution_chart(
            benign_scores, jailbreak_scores, selected["threshold"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        if compare_all and len(detector_results) > 1:
            roc_data = {}
            for r in detector_results:
                roc_data[r["display_name"]] = (
                    np.array(r["roc_fpr"]),
                    np.array(r["roc_tpr"]),
                    r["auroc"],
                )
            fig = ensemble_roc_comparison(roc_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fpr = np.array(selected["roc_fpr"])
            tpr = np.array(selected["roc_tpr"])
            fig = roc_curve_chart(fpr, tpr, selected["auroc"])
            st.plotly_chart(fig, use_container_width=True)

    # Bottom row: precision-recall
    pr_col, _ = st.columns(2)
    with pr_col:
        pr_prec = np.array(selected["pr_precision"])
        pr_rec = np.array(selected["pr_recall"])
        fig = precision_recall_chart(pr_rec, pr_prec, selected["auprc"])
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Already complete — show cached results
# ------------------------------------------------------------------

if state.evaluation_ready and state.eval_results:
    st.success("Evaluation complete.")

    eval_data = st.session_state.get("_eval_display_data")
    if eval_data is not None and "detector_results" in eval_data:
        _show_results(eval_data)
    else:
        # Cached display data lost (e.g., server restart) — show basic metrics
        st.subheader("Results (summary)")
        metrics = state.eval_results
        primary_keys = ["auroc", "auprc", "precision", "recall", "f1"]
        cols = st.columns(3)
        for i, key in enumerate(primary_keys):
            if key in metrics:
                with cols[i % 3]:
                    st.metric(key.upper(), f"{metrics[key]:.4f}")
        if "top_detector" in metrics:
            st.caption(f"Top-ranked: **{metrics['top_detector']}**")
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

    # Read layer from pipeline metadata
    meta_path = Path(state.model_path) / "pipeline_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        layer = meta["layer"]
    else:
        layer = store.layers[len(store.layers) // 2]

    with st.status("Running evaluation...", expanded=True) as status:
        # Get test data
        test_benign_indices = splits["test_benign"].indices.tolist()
        test_jailbreak_indices = splits["test_jailbreak"].indices.tolist()

        X_test_benign = store.get_activations(layer, indices=test_benign_indices)
        X_test_jailbreak = store.get_activations(
            layer, indices=test_jailbreak_indices
        )

        st.write(
            f"Test set: {len(X_test_benign)} benign + "
            f"{len(X_test_jailbreak)} jailbreak = "
            f"{len(X_test_benign) + len(X_test_jailbreak)} total"
        )

        # Build detectors dict: individual detectors + ensemble if > 1
        detectors: dict[str, object] = {}
        detector_types: dict[str, str] = {}
        display_names: dict[str, str] = {}

        for name, det, _weight in ensemble.detectors:
            detectors[name] = det
            dtype = ensemble.detector_types_.get(name, DetectorType.UNSUPERVISED)
            detector_types[name] = dtype.value
            display_names[name] = _DISPLAY_NAMES.get(name, name)

        # Add ensemble as another detector if there are multiple components
        if len(ensemble.detectors) > 1:
            detectors["ensemble"] = _EnsembleScorer(ensemble)
            detector_types["ensemble"] = "ensemble"
            display_names["ensemble"] = "Custom Ensemble"

        st.write(f"Evaluating {len(detectors)} detector(s)...")

        # Evaluate all detectors
        eval_results = evaluate_all_detectors(
            detectors,
            X_test_benign,
            X_test_jailbreak,
            detector_types=detector_types,
            display_names=display_names,
        )

        # Get test prompts for explorer (used by Explore page)
        all_test_indices = test_benign_indices + test_jailbreak_indices
        test_prompts = store.get_prompts(indices=all_test_indices)

        status.update(label="Evaluation complete!", state="complete")

    # Serialize results for session state
    serialized_results = [_result_to_dict(r) for r in eval_results]

    labels = np.concatenate([
        np.zeros(len(X_test_benign), dtype=np.int64),
        np.ones(len(X_test_jailbreak), dtype=np.int64),
    ])

    eval_data = {
        "detector_results": serialized_results,
        "n_benign": len(X_test_benign),
        "n_jailbreak": len(X_test_jailbreak),
        "prompts": test_prompts,
        "labels": labels.tolist(),
    }

    st.session_state["_eval_display_data"] = eval_data

    # Update state with top-ranked detector's metrics
    top = eval_results[0]
    state.eval_results = {
        "auroc": float(top.auroc),
        "auprc": float(top.auprc),
        "f1": float(top.f1),
        "precision": float(top.precision),
        "recall": float(top.recall),
        "fpr": float(top.fpr),
        "top_detector": top.display_name,
        "n_detectors": len(eval_results),
    }
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
