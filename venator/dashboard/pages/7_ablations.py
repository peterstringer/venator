"""Ablation studies page — layer/PCA/detector comparison."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import ablation_line_chart, correlation_heatmap
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.ensemble import DetectorEnsemble, create_default_ensemble
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.metrics import evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

state = PipelineState()
config = state.config

_ABLATION_LAYERS = [8, 10, 12, 14, 16, 18, 20, 22, 24]
_ABLATION_PCA_DIMS = [10, 20, 30, 50, 75, 100]

st.header("7. Ablation Studies")
st.markdown(
    "Compare detector performance across different layers, PCA dimensions, "
    "and detector configurations. Results reference findings from the ELK "
    "literature on optimal layer selection and ensemble decorrelation."
)

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(7):
    st.warning(
        "Complete the **Extract** and **Split** stages first to unlock ablations."
    )
    st.stop()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_test_data(
    store: ActivationStore,
    splits: dict,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get stacked test data (benign + jailbreak) and labels for a layer."""
    X_benign = store.get_activations(
        layer, indices=splits["test_benign"].indices.tolist()
    )
    X_jailbreak = store.get_activations(
        layer, indices=splits["test_jailbreak"].indices.tolist()
    )
    X_test = np.vstack([X_benign, X_jailbreak])
    labels = np.concatenate([
        np.zeros(len(X_benign), dtype=np.int64),
        np.ones(len(X_jailbreak), dtype=np.int64),
    ])
    return X_test, labels


# ------------------------------------------------------------------
# Results display
# ------------------------------------------------------------------


def _show_ablation_results(results: dict) -> None:
    """Render all ablation results in tabs."""
    tab_names = []
    if "layers" in results:
        tab_names.append("Layer Comparison")
    if "pca_dims" in results:
        tab_names.append("PCA Dimensions")
    if "detectors" in results:
        tab_names.append("Detector Comparison")

    if not tab_names:
        st.warning("No ablation results to display.")
        return

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # --- Layer Comparison ---
    if "layers" in results:
        with tabs[tab_idx]:
            layer_data = results["layers"]
            layers = [r["layer"] for r in layer_data]
            aurocs = [r["auroc"] for r in layer_data]

            fig = ablation_line_chart(
                layers, aurocs, "Layer", "AUROC",
                title="AUROC by Transformer Layer",
            )
            st.plotly_chart(fig, use_container_width=True)

            best = max(layer_data, key=lambda r: r["auroc"])
            st.info(
                f"**Best layer: {best['layer']}** (AUROC = {best['auroc']:.4f}). "
                "Middle layers typically generalize best for anomaly detection, "
                "consistent with the Earliest Informative Layer criterion from "
                "the ELK literature (Mallen et al., 2024)."
            )

            df = pd.DataFrame(layer_data)
            display_cols = ["layer", "auroc", "auprc", "precision", "recall", "f1", "time_s"]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].round(4),
                use_container_width=True,
                hide_index=True,
            )
        tab_idx += 1

    # --- PCA Dimensions ---
    if "pca_dims" in results:
        with tabs[tab_idx]:
            pca_data = results["pca_dims"]
            dims = [r["pca_dims"] for r in pca_data]
            aurocs = [r["auroc"] for r in pca_data]

            fig = ablation_line_chart(
                dims, aurocs, "PCA Dimensions", "AUROC",
                title="AUROC by PCA Dimensionality",
            )
            st.plotly_chart(fig, use_container_width=True)

            best = max(pca_data, key=lambda r: r["auroc"])
            st.info(
                f"**Best PCA dimensions: {best['pca_dims']}** "
                f"(AUROC = {best['auroc']:.4f}). "
                "Higher dimensions capture more variance but may include noise. "
                "The optimal point balances the sample-to-feature ratio with "
                "signal retention."
            )

            df = pd.DataFrame(pca_data)
            display_cols = ["pca_dims", "auroc", "auprc", "precision", "recall", "f1", "time_s"]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].round(4),
                use_container_width=True,
                hide_index=True,
            )
        tab_idx += 1

    # --- Detector Comparison ---
    if "detectors" in results:
        with tabs[tab_idx]:
            det_data = results["detectors"]
            det_names = [r["detector"] for r in det_data]
            aurocs = [r["auroc"] for r in det_data]
            auprcs = [r.get("auprc", 0) for r in det_data]

            # Grouped bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="AUROC",
                x=det_names,
                y=aurocs,
                marker_color="rgba(55, 128, 191, 0.8)",
                text=[f"{v:.3f}" for v in aurocs],
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                name="AUPRC",
                x=det_names,
                y=auprcs,
                marker_color="rgba(44, 160, 101, 0.8)",
                text=[f"{v:.3f}" for v in auprcs],
                textposition="outside",
            ))
            fig.update_layout(
                barmode="group",
                title="Detector Performance Comparison",
                yaxis_title="Score",
                height=400,
                margin=dict(t=40, b=40, l=40, r=20),
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap
            if "detector_sample_scores" in results:
                sample_scores = {
                    k: np.array(v)
                    for k, v in results["detector_sample_scores"].items()
                }
                fig = correlation_heatmap(
                    sample_scores,
                    title="Score Correlation Between Detectors",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Low pairwise correlation between detectors confirms decorrelated "
                "errors, which is the key advantage of ensemble detection per the "
                "ELK paper: *\"An ELK method can be useful even when no more "
                "accurate, as long as its errors are decorrelated.\"*"
            )

            df = pd.DataFrame(det_data)
            display_cols = [
                "detector", "auroc", "auprc", "precision", "recall", "f1", "time_s",
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].round(4),
                use_container_width=True,
                hide_index=True,
            )

    # --- Export ---
    st.divider()
    st.subheader("Export")

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        # Exclude large sample_scores arrays from JSON export
        export_data = {
            k: v for k, v in results.items() if k != "detector_sample_scores"
        }
        json_str = json.dumps(export_data, indent=2, default=str)
        st.download_button(
            "Download JSON Results",
            data=json_str,
            file_name="ablation_results.json",
            mime="application/json",
        )
    with export_col2:
        st.info(
            "Use the camera icon in the top-right corner of each chart to "
            "download individual figures as PNG."
        )


# ------------------------------------------------------------------
# Already-run results
# ------------------------------------------------------------------

cached = st.session_state.get("_ablation_results")
if cached:
    st.success("Ablation results available.")
    _show_ablation_results(cached)
    if st.button("Re-run Ablations"):
        del st.session_state["_ablation_results"]
        st.rerun()
    st.stop()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

st.subheader("Configuration")

store = ActivationStore(state.store_path)
available_layers = store.layers

abl_col1, abl_col2, abl_col3 = st.columns(3)
with abl_col1:
    run_layers = st.checkbox("Layer comparison", value=True)
with abl_col2:
    run_pca = st.checkbox("PCA dimensions", value=True)
with abl_col3:
    run_detectors = st.checkbox("Detector comparison", value=True)

default_layer = available_layers[len(available_layers) // 2]
base_layer = st.selectbox(
    "Base layer (for PCA & detector ablations)",
    options=available_layers,
    index=available_layers.index(default_layer),
)

n_selected = sum([run_layers, run_pca, run_detectors])
if n_selected == 0:
    st.error("Select at least one ablation type.")
    st.stop()

# ------------------------------------------------------------------
# Run ablations
# ------------------------------------------------------------------

st.divider()

if st.button("Run Ablations", type="primary"):
    splits = SplitManager.load_splits(state.splits_path)

    # Count total configurations for progress
    test_layers = [l for l in _ABLATION_LAYERS if l in set(available_layers)]
    total_configs = 0
    if run_layers:
        total_configs += len(test_layers)
    if run_pca:
        total_configs += len(_ABLATION_PCA_DIMS)
    if run_detectors:
        total_configs += 4  # 3 individual + 1 ensemble

    current = 0
    results: dict = {}
    progress = st.progress(0.0, text="Starting ablations...")

    # --- Layer ablation ---
    if run_layers:
        layer_results = []
        for layer in test_layers:
            progress.progress(
                current / total_configs,
                text=f"Layer ablation: layer {layer}...",
            )

            t0 = time.perf_counter()
            X_train = store.get_activations(
                layer, indices=splits["train"].indices.tolist()
            )
            X_val = store.get_activations(
                layer, indices=splits["val"].indices.tolist()
            )
            X_test, labels = _get_test_data(store, splits, layer)

            ensemble = DetectorEnsemble(
                threshold_percentile=config.anomaly_threshold_percentile,
            )
            ensemble.add_detector(
                "pca_mahalanobis",
                PCAMahalanobisDetector(n_components=config.pca_dims),
                weight=1.0,
            )
            ensemble.fit(X_train, X_val)

            result = ensemble.score(X_test)
            metrics = evaluate_detector(
                result.ensemble_scores, labels, threshold=ensemble.threshold_
            )

            elapsed = time.perf_counter() - t0
            layer_results.append({
                "layer": layer, "time_s": round(elapsed, 2), **metrics,
            })
            current += 1

        results["layers"] = layer_results

    # --- PCA dimension ablation ---
    if run_pca:
        X_train = store.get_activations(
            base_layer, indices=splits["train"].indices.tolist()
        )
        X_val = store.get_activations(
            base_layer, indices=splits["val"].indices.tolist()
        )
        X_test, labels = _get_test_data(store, splits, base_layer)

        pca_results = []
        for n_dims in _ABLATION_PCA_DIMS:
            if n_dims >= X_train.shape[0]:
                current += 1
                continue

            progress.progress(
                current / total_configs,
                text=f"PCA ablation: {n_dims} dimensions...",
            )

            t0 = time.perf_counter()
            ensemble = DetectorEnsemble(
                threshold_percentile=config.anomaly_threshold_percentile,
            )
            ensemble.add_detector(
                "pca_mahalanobis",
                PCAMahalanobisDetector(n_components=n_dims),
                weight=1.0,
            )
            ensemble.fit(X_train, X_val)

            result = ensemble.score(X_test)
            metrics = evaluate_detector(
                result.ensemble_scores, labels, threshold=ensemble.threshold_
            )

            elapsed = time.perf_counter() - t0
            pca_results.append({
                "pca_dims": n_dims, "time_s": round(elapsed, 2), **metrics,
            })
            current += 1

        results["pca_dims"] = pca_results

    # --- Detector ablation ---
    if run_detectors:
        X_train = store.get_activations(
            base_layer, indices=splits["train"].indices.tolist()
        )
        X_val = store.get_activations(
            base_layer, indices=splits["val"].indices.tolist()
        )
        X_test, labels = _get_test_data(store, splits, base_layer)

        detector_configs = [
            ("pca_mahalanobis", PCAMahalanobisDetector(n_components=config.pca_dims)),
            ("isolation_forest", IsolationForestDetector(n_components=config.pca_dims)),
            ("autoencoder", AutoencoderDetector(n_components=config.pca_dims)),
        ]

        detector_results = []
        detector_sample_scores: dict[str, list] = {}

        for name, detector in detector_configs:
            progress.progress(
                current / total_configs,
                text=f"Detector ablation: {name}...",
            )

            t0 = time.perf_counter()
            ens = DetectorEnsemble(
                threshold_percentile=config.anomaly_threshold_percentile,
            )
            ens.add_detector(name, detector, weight=1.0)
            ens.fit(X_train, X_val)

            result = ens.score(X_test)
            metrics = evaluate_detector(
                result.ensemble_scores, labels, threshold=ens.threshold_
            )

            elapsed = time.perf_counter() - t0
            detector_results.append({
                "detector": name, "time_s": round(elapsed, 2), **metrics,
            })
            detector_sample_scores[name] = result.ensemble_scores.tolist()
            current += 1

        # Full ensemble
        progress.progress(
            current / total_configs,
            text="Detector ablation: full ensemble...",
        )

        t0 = time.perf_counter()
        full_ensemble = create_default_ensemble(
            threshold_percentile=config.anomaly_threshold_percentile,
        )
        full_ensemble.fit(X_train, X_val)

        result = full_ensemble.score(X_test)
        metrics = evaluate_detector(
            result.ensemble_scores, labels, threshold=full_ensemble.threshold_
        )

        elapsed = time.perf_counter() - t0
        detector_results.append({
            "detector": "ensemble", "time_s": round(elapsed, 2), **metrics,
        })
        detector_sample_scores["ensemble"] = result.ensemble_scores.tolist()
        current += 1

        results["detectors"] = detector_results
        results["detector_sample_scores"] = detector_sample_scores

    progress.progress(1.0, text="Ablations complete!")

    # Cache results and display
    st.session_state["_ablation_results"] = results
    _show_ablation_results(results)
