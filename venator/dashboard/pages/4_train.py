"""Detector training page — train detectors and show convergence."""

from __future__ import annotations

import json
import time

import numpy as np
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager, SplitMode
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorType,
    create_hybrid_ensemble,
    create_supervised_ensemble,
    create_unsupervised_ensemble,
)
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

state = PipelineState()
config = state.config

st.header("4. Train Detectors")
st.markdown(
    "Train anomaly detection ensemble on activation data. Supports unsupervised "
    "(benign-only), supervised (labeled data), and hybrid configurations."
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
# Detect split mode
# ------------------------------------------------------------------

splits = SplitManager.load_splits(state.splits_path)
split_mode = SplitManager.load_mode(state.splits_path)
is_semi = split_mode == SplitMode.SEMI_SUPERVISED

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

st.subheader("Configuration")

store = ActivationStore(state.store_path)
available_layers = store.layers

# --- Ensemble type selector ---
if is_semi:
    ensemble_type = st.radio(
        "Ensemble Type",
        options=["unsupervised", "supervised", "hybrid"],
        format_func=lambda x: x.capitalize(),
        horizontal=True,
        index=2,  # Default to hybrid for semi-supervised
        help=(
            "**Unsupervised**: PCA+Mahalanobis, Isolation Forest, Autoencoder (benign-only). "
            "**Supervised**: Linear Probe, Contrastive Direction, Contrastive Mahalanobis (labeled data). "
            "**Hybrid**: Linear Probe, Contrastive Direction + Autoencoder (best of both)."
        ),
    )
else:
    ensemble_type = "unsupervised"
    st.info(
        "Using **unsupervised** ensemble (splits are unsupervised). "
        "To use supervised/hybrid detectors, re-create splits in Semi-Supervised mode."
    )

# Show detector composition for selected ensemble type
_ENSEMBLE_DETECTORS = {
    "unsupervised": [
        ("PCA + Mahalanobis", "unsup", 2.0),
        ("Isolation Forest", "unsup", 1.5),
        ("Autoencoder", "unsup", 1.0),
    ],
    "supervised": [
        ("Linear Probe", "sup", 2.5),
        ("Contrastive Direction", "sup", 2.0),
        ("Contrastive Mahalanobis", "sup", 1.5),
    ],
    "hybrid": [
        ("Linear Probe", "sup", 2.5),
        ("Contrastive Direction", "sup", 2.0),
        ("Autoencoder", "unsup", 1.0),
    ],
}

st.markdown("**Detectors in this ensemble:**")
for det_name, det_type, det_weight in _ENSEMBLE_DETECTORS[ensemble_type]:
    badge = ":green[supervised]" if det_type == "sup" else ":blue[unsupervised]"
    st.markdown(f"- {det_name} ({badge}, weight={det_weight})")

# --- Auto-tune toggle ---
auto_tune = st.toggle(
    "Auto-tune PCA dimensions and threshold",
    value=False,
    help=(
        "Automatically select the best PCA dimensions by maximizing explained "
        "variance while maintaining a healthy sample-to-feature ratio, and "
        "optimize the threshold percentile using validation score distribution."
    ),
    disabled=(ensemble_type != "unsupervised"),
)
if ensemble_type != "unsupervised" and auto_tune:
    auto_tune = False

# --- Layer and PCA/threshold controls ---
cfg_col1, cfg_col2 = st.columns(2)
with cfg_col1:
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
        disabled=auto_tune,
        help="Disabled when auto-tune is on." if auto_tune else None,
    )

with cfg_col2:
    threshold_pctile = st.slider(
        "Threshold percentile",
        min_value=90.0,
        max_value=99.0,
        value=config.anomaly_threshold_percentile,
        step=0.5,
        disabled=auto_tune,
        help="Disabled when auto-tune is on." if auto_tune else None,
    )

# --- Manual detector selection (unsupervised only) ---
if ensemble_type == "unsupervised":
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
    # Get training data based on split mode
    if "train" in splits:
        X_train = store.get_activations(layer, indices=splits["train"].indices.tolist())
        X_val = store.get_activations(layer, indices=splits["val"].indices.tolist())
        X_train_jailbreak = None
        X_val_jailbreak = None
    else:
        X_train = store.get_activations(layer, indices=splits["train_benign"].indices.tolist())
        X_val = store.get_activations(layer, indices=splits["val_benign"].indices.tolist())
        if ensemble_type in ("supervised", "hybrid"):
            X_train_jailbreak = store.get_activations(
                layer, indices=splits["train_jailbreak"].indices.tolist()
            )
            X_val_jailbreak = store.get_activations(
                layer, indices=splits["val_jailbreak"].indices.tolist()
            )
        else:
            X_train_jailbreak = None
            X_val_jailbreak = None

    # --- Auto-tune PCA dims and threshold if enabled ---
    if auto_tune:
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]

        with st.status("Auto-tuning hyperparameters...", expanded=True) as tune_status:
            st.write("Testing PCA dimensions...")
            candidates = [d for d in [10, 20, 30, 40, 50, 75, 100]
                          if d < X_train.shape[0] // 5]
            if not candidates:
                candidates = [max(5, X_train.shape[0] // 10)]

            best_dims = candidates[0]
            best_score = -1.0
            tune_results = []
            for d in candidates:
                pca = PCA(n_components=d)
                pca.fit(X_train)
                explained = float(np.sum(pca.explained_variance_ratio_))
                ratio = X_train.shape[0] / d
                penalty = min(1.0, ratio / 7.0)
                score = explained * penalty
                tune_results.append((d, explained, ratio, score))
                if score > best_score:
                    best_score = score
                    best_dims = d
            pca_dims = best_dims

            results_text = " | ".join(
                f"d={d}: var={v:.2%}, ratio={r:.1f}x" for d, v, r, _ in tune_results
            )
            st.write(f"Candidates: {results_text}")
            st.write(f"Selected PCA dims: **{pca_dims}** (score={best_score:.3f})")

            st.write("Optimizing threshold percentile...")
            probe = PCAMahalanobisDetector(n_components=pca_dims)
            probe.fit(X_train)
            val_scores = probe.score(X_val)
            train_scores = probe.score(X_train)

            best_pctile = 95.0
            best_gap = 0.0
            median_train = float(np.median(train_scores))
            for pctile in [90.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0]:
                thresh = float(np.percentile(val_scores, pctile))
                fpr = float(np.mean(val_scores > thresh))
                gap = thresh - median_train
                fpr_penalty = abs(fpr - 0.05) * 10
                adjusted = gap - fpr_penalty
                if adjusted > best_gap:
                    best_gap = adjusted
                    best_pctile = pctile

            threshold_pctile = best_pctile
            st.write(f"Selected threshold percentile: **{threshold_pctile}%**")
            tune_status.update(
                label=f"Auto-tune complete: PCA={pca_dims}, threshold={threshold_pctile}%",
                state="complete",
            )

    # Training info
    info_parts = [
        f"Training data: **{len(X_train)}** benign samples",
    ]
    if X_train_jailbreak is not None:
        info_parts[0] += f" + **{len(X_train_jailbreak)}** jailbreak samples"
    info_parts.append(f"Validation: **{len(X_val)}** benign samples")
    if X_val_jailbreak is not None:
        info_parts[-1] += f" + **{len(X_val_jailbreak)}** jailbreak samples"
    info_parts.extend([
        f"Features: **{X_train.shape[1]}** dims",
        f"Layer: **{layer}**",
        f"PCA dims: **{pca_dims}**",
        f"Threshold: **{threshold_pctile}%**",
    ])
    st.info(", ".join(info_parts))

    # Build ensemble
    if ensemble_type == "unsupervised":
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
        n_detectors = sum([use_pca_maha, use_iforest, use_autoencoder])
    elif ensemble_type == "supervised":
        ensemble = create_supervised_ensemble(threshold_percentile=threshold_pctile)
        n_detectors = len(ensemble.detectors)
    else:  # hybrid
        ensemble = create_hybrid_ensemble(threshold_percentile=threshold_pctile)
        n_detectors = len(ensemble.detectors)

    with st.status("Training ensemble...", expanded=True) as status:
        t0 = time.perf_counter()

        for name, det, weight in ensemble.detectors:
            det_type = ensemble.detector_types_.get(name, DetectorType.UNSUPERVISED)
            type_label = "supervised" if det_type == DetectorType.SUPERVISED else "unsupervised"
            st.write(f"Fitting **{name}** ({type_label}, weight={weight})...")

        ensemble.fit(
            X_train, X_val,
            X_train_jailbreak=X_train_jailbreak,
            X_val_jailbreak=X_val_jailbreak,
        )
        elapsed = time.perf_counter() - t0

        for name, det, weight in ensemble.detectors:
            det_type = ensemble.detector_types_.get(name, DetectorType.UNSUPERVISED)
            if det_type == DetectorType.SUPERVISED:
                st.write(
                    f"  {name}: Fitted on {len(X_train)} benign + "
                    f"{len(X_train_jailbreak) if X_train_jailbreak is not None else 0} jailbreak"
                )
            else:
                st.write(f"  {name}: Fitted on {len(X_train)} benign samples")

        st.write(f"Training completed in {elapsed:.1f}s")
        status.update(label="Training complete!", state="complete")

    # Compute validation FPR (benign-only)
    val_result = ensemble.score(X_val)
    val_fpr = float(np.mean(val_result.is_anomaly))

    # Save model
    output_dir = config.models_dir / f"detector_{ensemble_type}_v1"
    ensemble.save(output_dir)

    # Save pipeline metadata
    meta = {
        "layer": int(layer),
        "model_id": store.model_id,
        "extraction_layers": [int(l) for l in store.layers],
        "ensemble_type": ensemble_type,
    }
    with open(output_dir / "pipeline_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Update state
    train_metrics = {
        "val_false_positive_rate": val_fpr,
        "threshold": float(ensemble.threshold_),
        "layer": int(layer),
        "pca_dims": int(pca_dims),
        "n_detectors": int(n_detectors),
        "ensemble_type": ensemble_type,
        "training_time_s": round(elapsed, 1),
    }

    state.reset_from(4)
    state.model_path = output_dir
    state.train_metrics = train_metrics
    state.model_ready = True

    # Show results
    st.success(f"Trained {ensemble_type} ensemble in {elapsed:.1f}s. Model saved to `{output_dir}`.")

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Validation FPR", f"{val_fpr:.4f} ({val_fpr * 100:.1f}%)")
    with res_col2:
        st.metric("Threshold", f"{ensemble.threshold_:.4f}")
    with res_col3:
        st.metric("Training time", f"{elapsed:.1f}s")

    if st.button("Continue to Evaluate  \u2192", type="primary", key="post_train_continue"):
        st.switch_page("pages/5_evaluate.py")
