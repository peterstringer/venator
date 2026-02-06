"""Detector training page — train detectors and show convergence."""

from __future__ import annotations

import json
import time

import numpy as np
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.ensemble import DetectorEnsemble
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

state = PipelineState()
config = state.config

st.header("4. Train Detectors")
st.markdown(
    "Train the anomaly detection ensemble (PCA+Mahalanobis, Isolation Forest, "
    "Autoencoder) on benign-only training data."
)

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(4):
    st.warning("Complete the **Split** stage first to unlock training.")
    st.stop()

# ------------------------------------------------------------------
# Already complete — show results
# ------------------------------------------------------------------

if state.model_ready and state.model_path:
    st.success("Model trained.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model directory", str(state.model_path))
    with col2:
        if state.train_metrics:
            val_fpr = state.train_metrics.get("val_false_positive_rate")
            if val_fpr is not None:
                st.metric("Validation FPR", f"{val_fpr:.4f} ({val_fpr * 100:.1f}%)")

    if state.train_metrics:
        for key, value in state.train_metrics.items():
            if key != "val_false_positive_rate":
                st.metric(key, f"{value:.4f}" if isinstance(value, float) else str(value))

    col_re, col_cont = st.columns(2)
    with col_re:
        if st.button("Re-train"):
            state.reset_from(4)
            st.rerun()
    with col_cont:
        if st.button("Continue  \u2192", type="primary"):
            st.switch_page("pages/5_evaluate.py")
    st.stop()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

st.subheader("Configuration")

store = ActivationStore(state.store_path)
available_layers = store.layers

cfg_col1, cfg_col2 = st.columns(2)
with cfg_col1:
    # Default to middle layer
    default_layer = available_layers[len(available_layers) // 2]
    layer = st.selectbox(
        "Layer for detection",
        options=available_layers,
        index=available_layers.index(default_layer),
    )
    pca_dims = st.slider(
        "PCA dimensions",
        min_value=10,
        max_value=100,
        value=config.pca_dims,
        step=5,
    )

with cfg_col2:
    threshold_pctile = st.slider(
        "Threshold percentile",
        min_value=90.0,
        max_value=99.0,
        value=config.anomaly_threshold_percentile,
        step=0.5,
    )

st.subheader("Detector Selection")
det_col1, det_col2, det_col3 = st.columns(3)
with det_col1:
    use_pca_maha = st.checkbox("PCA + Mahalanobis", value=True)
    weight_pca_maha = st.number_input(
        "Weight", value=config.weight_pca_mahalanobis, min_value=0.1, step=0.5,
        key="w_pca",
    ) if use_pca_maha else 0.0
with det_col2:
    use_iforest = st.checkbox("Isolation Forest", value=True)
    weight_iforest = st.number_input(
        "Weight", value=config.weight_isolation_forest, min_value=0.1, step=0.5,
        key="w_if",
    ) if use_iforest else 0.0
with det_col3:
    use_autoencoder = st.checkbox("Autoencoder", value=True)
    weight_ae = st.number_input(
        "Weight", value=config.weight_autoencoder, min_value=0.1, step=0.5,
        key="w_ae",
    ) if use_autoencoder else 0.0

n_detectors = sum([use_pca_maha, use_iforest, use_autoencoder])
if n_detectors == 0:
    st.error("Select at least one detector.")
    st.stop()

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

st.divider()

if st.button("Train Ensemble", type="primary"):
    splits = SplitManager.load_splits(state.splits_path)

    X_train = store.get_activations(layer, indices=splits["train"].indices.tolist())
    X_val = store.get_activations(layer, indices=splits["val"].indices.tolist())

    st.info(
        f"Training data: **{len(X_train)}** samples, "
        f"validation: **{len(X_val)}** samples, "
        f"features: **{X_train.shape[1]}** dims, "
        f"layer: **{layer}**"
    )

    # Build ensemble
    ensemble = DetectorEnsemble(threshold_percentile=threshold_pctile)
    if use_pca_maha:
        ensemble.add_detector(
            "pca_mahalanobis",
            PCAMahalanobisDetector(n_components=pca_dims),
            weight=weight_pca_maha,
        )
    if use_iforest:
        ensemble.add_detector(
            "isolation_forest",
            IsolationForestDetector(n_components=pca_dims),
            weight=weight_iforest,
        )
    if use_autoencoder:
        ensemble.add_detector(
            "autoencoder",
            AutoencoderDetector(n_components=pca_dims),
            weight=weight_ae,
        )

    with st.status("Training ensemble...", expanded=True) as status:
        t0 = time.perf_counter()
        st.write(f"Fitting {n_detectors} detector(s)...")
        ensemble.fit(X_train, X_val)
        elapsed = time.perf_counter() - t0
        st.write(f"Training completed in {elapsed:.1f}s")
        status.update(label="Training complete!", state="complete")

    # Compute validation FPR
    val_result = ensemble.score(X_val)
    val_fpr = float(np.mean(val_result.is_anomaly))

    # Save model
    output_dir = config.models_dir / "detector_v1"
    ensemble.save(output_dir)

    # Save pipeline metadata
    meta = {
        "layer": layer,
        "model_id": store.model_id,
        "extraction_layers": store.layers,
    }
    with open(output_dir / "pipeline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Update state
    train_metrics = {
        "val_false_positive_rate": val_fpr,
        "threshold": float(ensemble.threshold_),
        "layer": layer,
        "pca_dims": pca_dims,
        "n_detectors": n_detectors,
        "training_time_s": round(elapsed, 1),
    }

    state.reset_from(4)
    state.model_path = output_dir
    state.train_metrics = train_metrics
    state.model_ready = True

    # Show results
    st.success(f"Trained in {elapsed:.1f}s. Model saved to `{output_dir}`.")

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Validation FPR", f"{val_fpr:.4f} ({val_fpr * 100:.1f}%)")
    with res_col2:
        st.metric("Threshold", f"{ensemble.threshold_:.4f}")
    with res_col3:
        st.metric("Training time", f"{elapsed:.1f}s")

    if st.button("Continue to Evaluate  \u2192", type="primary", key="post_train_continue"):
        st.switch_page("pages/5_evaluate.py")
