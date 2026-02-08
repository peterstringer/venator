"""Evaluation page — test metrics, ROC curves, score distributions, ensemble comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore[import-untyped]

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import (
    detector_comparison_chart,
    detector_comparison_grouped_bar,
    ensemble_roc_comparison,
    precision_recall_chart,
    roc_curve_chart,
    score_distribution_chart,
)
from venator.dashboard.components.prompt_table import (
    render_error_analysis,
    render_prompt_explorer,
)
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager, SplitMode
from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorType,
    create_hybrid_ensemble,
    create_supervised_ensemble,
    create_unsupervised_ensemble,
)
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

    # Determine if we have per-detector type info
    per_detector_info = eval_data.get("per_detector_info", [])
    ensemble_type = eval_data.get("ensemble_type", "unsupervised")

    # --- Tab view ---
    tab_names = ["Individual Detectors"]
    if eval_data.get("ensemble_comparison"):
        tab_names.append("Ensemble Comparison")
    if eval_data.get("generalization"):
        tab_names.append("Generalization Analysis")

    tabs = st.tabs(tab_names) if len(tab_names) > 1 else [st.container()]
    tab_idx = 0

    # --- Tab 1: Individual Detectors ---
    with tabs[tab_idx]:
        # Headline Metrics
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

        # Visualizations (2x2 grid)
        st.subheader("Visualizations")

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
            # Grouped bar if we have type information
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
            pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
            fig = precision_recall_chart(pr_recall, pr_precision, metrics["auprc"])
            st.plotly_chart(fig, width="stretch")

        # Per-detector breakdown table
        if per_detector_info:
            st.subheader("Per-Detector Breakdown")
            import pandas as pd
            df = pd.DataFrame(per_detector_info)
            display_cols = [c for c in ["name", "type", "weight", "auroc", "auprc", "f1", "fpr_at_95_tpr"]
                           if c in df.columns]
            st.dataframe(df[display_cols].round(4), use_container_width=True, hide_index=True)

        # Detailed Results
        st.subheader("Detailed Results")

        with st.expander("Prompt Explorer", expanded=False):
            render_prompt_explorer(prompts, labels, scores, predictions, threshold)

        render_error_analysis(prompts, labels, scores, predictions)

    tab_idx += 1

    # --- Tab 2: Ensemble Comparison ---
    if eval_data.get("ensemble_comparison"):
        with tabs[tab_idx]:
            comparison = eval_data["ensemble_comparison"]
            st.subheader("Ensemble Type Comparison")

            # Side-by-side metrics
            ens_names = list(comparison.keys())
            cols = st.columns(len(ens_names))
            for i, ens_name in enumerate(ens_names):
                ens_data = comparison[ens_name]
                with cols[i]:
                    st.markdown(f"**{ens_name.capitalize()}**")
                    st.metric("AUROC", f"{ens_data['auroc']:.4f}")
                    st.metric("AUPRC", f"{ens_data['auprc']:.4f}")
                    st.metric("F1", f"{ens_data.get('f1', 0):.4f}")
                    st.metric("FPR@95TPR", f"{ens_data['fpr_at_95_tpr']:.4f}")

            # Overlaid ROC curves
            roc_data = {}
            for ens_name, ens_data in comparison.items():
                if "roc_fpr" in ens_data and "roc_tpr" in ens_data:
                    roc_data[ens_name.capitalize()] = (
                        np.array(ens_data["roc_fpr"]),
                        np.array(ens_data["roc_tpr"]),
                        ens_data["auroc"],
                    )
            if roc_data:
                fig = ensemble_roc_comparison(roc_data)
                st.plotly_chart(fig, width="stretch")

            # Score distribution overlays
            for ens_name, ens_data in comparison.items():
                if "benign_scores" in ens_data and "jailbreak_scores" in ens_data:
                    with st.expander(f"Score Distribution: {ens_name.capitalize()}", expanded=False):
                        fig = score_distribution_chart(
                            np.array(ens_data["benign_scores"]),
                            np.array(ens_data["jailbreak_scores"]),
                            ens_data.get("threshold", threshold),
                        )
                        st.plotly_chart(fig, width="stretch")

        tab_idx += 1

    # --- Tab 3: Generalization Analysis ---
    if eval_data.get("generalization"):
        with tabs[tab_idx]:
            gen_data = eval_data["generalization"]
            st.subheader("Generalization Analysis")

            if "label_efficiency" in gen_data:
                st.markdown("### How many labeled examples are needed?")
                from venator.dashboard.components.charts import labeled_data_efficiency_chart
                eff = gen_data["label_efficiency"]
                fig = labeled_data_efficiency_chart(
                    eff["n_labeled"],
                    eff["aurocs_by_detector"],
                    eff.get("unsupervised_baselines", {}),
                )
                st.plotly_chart(fig, width="stretch")

            if "cross_source" in gen_data:
                st.markdown("### Cross-Source Generalization")
                from venator.dashboard.components.charts import generalization_heatmap
                cs = gen_data["cross_source"]
                fig = generalization_heatmap(
                    cs["sources"],
                    np.array(cs["auroc_matrix"]),
                )
                st.plotly_chart(fig, width="stretch")
                st.info(
                    "Diagonal values show in-distribution performance. "
                    "Off-diagonal values show generalization to unseen attack types. "
                    "High off-diagonal AUROC indicates the detector learns general "
                    "jailbreak-ness rather than memorizing specific attack patterns."
                )


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

    # --- Download results ---
    st.divider()
    _export_eval_data = eval_data if eval_data is not None else {"metrics": state.eval_results}
    _export_safe = {
        k: v for k, v in _export_eval_data.items()
        if k not in ("prompts",)  # exclude large prompt lists
    }
    json_str = json.dumps(_export_safe, indent=2, default=str)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download Evaluation Results (JSON)",
            data=json_str,
            file_name="evaluation_results.json",
            mime="application/json",
        )
    with dl_col2:
        st.info(
            "Use the camera icon in the top-right corner of each chart to "
            "download individual figures as PNG."
        )

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

# Detect split mode for ensemble comparison option
split_mode = SplitManager.load_mode(state.splits_path)
is_semi = split_mode == SplitMode.SEMI_SUPERVISED

run_comparison = False
if is_semi:
    run_comparison = st.checkbox(
        "Run ensemble comparison (unsupervised vs supervised vs hybrid)",
        value=False,
        help="Train and evaluate all three ensemble types for side-by-side comparison. Takes longer.",
    )

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

        # Score through primary ensemble
        result = ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=ensemble.threshold_
        )

        # Per-detector metrics
        detector_aurocs = {"Ensemble": metrics["auroc"]}
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

            per_detector_info.append({
                "name": det_result.name,
                "type": det_type_label,
                "weight": det_result.weight,
                "auroc": det_metrics["auroc"],
                "auprc": det_metrics["auprc"],
                "f1": det_metrics.get("f1", 0.0),
                "fpr_at_95_tpr": det_metrics["fpr_at_95_tpr"],
            })

        # Threshold curves for ROC/PR plotting
        curves = compute_threshold_curves(result.ensemble_scores, labels)

        # Get test prompts for explorer
        all_test_indices = test_benign_indices + test_jailbreak_indices
        test_prompts = store.get_prompts(indices=all_test_indices)

        # --- Ensemble comparison (if requested) ---
        ensemble_comparison = {}
        if run_comparison and is_semi:
            st.write("Running ensemble comparison...")

            X_train = store.get_activations(
                layer, indices=splits["train_benign"].indices.tolist()
            )
            X_val = store.get_activations(
                layer, indices=splits["val_benign"].indices.tolist()
            )
            X_train_jb = store.get_activations(
                layer, indices=splits["train_jailbreak"].indices.tolist()
            )
            X_val_jb = store.get_activations(
                layer, indices=splits["val_jailbreak"].indices.tolist()
            )

            factories = {
                "unsupervised": (create_unsupervised_ensemble, False),
                "supervised": (create_supervised_ensemble, True),
                "hybrid": (create_hybrid_ensemble, True),
            }

            for ens_name, (factory, needs_jb) in factories.items():
                st.write(f"  Training {ens_name} ensemble...")
                ens = factory()
                if needs_jb:
                    ens.fit(X_train, X_val, X_train_jailbreak=X_train_jb, X_val_jailbreak=X_val_jb)
                else:
                    ens.fit(X_train, X_val)

                ens_result = ens.score(X_test)
                ens_metrics = evaluate_detector(
                    ens_result.ensemble_scores, labels, threshold=ens.threshold_
                )

                fpr_arr, tpr_arr, _ = roc_curve(labels, ens_result.ensemble_scores)

                ensemble_comparison[ens_name] = {
                    **{k: float(v) for k, v in ens_metrics.items()},
                    "roc_fpr": fpr_arr.tolist(),
                    "roc_tpr": tpr_arr.tolist(),
                    "threshold": float(ens.threshold_),
                    "benign_scores": ens_result.ensemble_scores[:len(X_test_benign)].tolist(),
                    "jailbreak_scores": ens_result.ensemble_scores[len(X_test_benign):].tolist(),
                }

        status.update(label="Evaluation complete!", state="complete")

    # Cache all display data in session state
    eval_data = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "curves": {k: v.tolist() for k, v in curves.items()},
        "benign_scores": result.ensemble_scores[:len(X_test_benign)].tolist(),
        "jailbreak_scores": result.ensemble_scores[len(X_test_benign):].tolist(),
        "threshold": float(ensemble.threshold_),
        "detector_aurocs": detector_aurocs,
        "per_detector_info": per_detector_info,
        "ensemble_type": ensemble_type,
        "prompts": test_prompts,
        "labels": labels.tolist(),
        "scores": result.ensemble_scores.tolist(),
        "predictions": result.is_anomaly.astype(int).tolist(),
    }
    if ensemble_comparison:
        eval_data["ensemble_comparison"] = ensemble_comparison

    st.session_state["_eval_display_data"] = eval_data

    # Update state
    state.eval_results = eval_data["metrics"]
    state.evaluation_ready = True

    # Show results
    _show_results(eval_data)

    # Download results
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

    if st.button("Continue to Detect  \u2192", type="primary", key="post_eval_continue"):
        st.switch_page("pages/6_detect.py")
