"""Quick Run page — automated full pipeline from prompts to optimized detector.

One-click workflow: collect data → extract activations → split → auto-optimize
→ evaluate. Populates PipelineState so all other pages (Results, Explore,
Detect, Ablations) work with the quick run data.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import (
    roc_curve_chart,
    score_distribution_chart,
)
from venator.dashboard.state import PipelineState
from venator.data.prompts import (
    PromptDataset,
    collect_benign_prompts,
    collect_jailbreak_prompts,
)
from venator.data.splits import SplitManager
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.contrastive import ContrastiveMahalanobisDetector
from venator.detection.ensemble import DetectorEnsemble, DetectorType
from venator.detection.linear_probe import LinearProbeDetector
from venator.detection.metrics import evaluate_all_detectors

state = PipelineState()
config = state.config

st.header("Quick Run")
st.markdown(
    "Go from raw prompts to a fully optimized detector in a few clicks. "
    "Automates the full pipeline: collect, extract, split, auto-optimize, evaluate."
)


# ==================================================================
# Detector options
# ==================================================================

_DETECTOR_OPTIONS = {
    "linear_probe": {
        "display": "Linear Probe (recommended)",
        "factory": lambda pca: LinearProbeDetector(n_components=pca),
        "type": "supervised",
    },
    "contrastive_mahalanobis": {
        "display": "Contrastive Mahalanobis",
        "factory": lambda pca: ContrastiveMahalanobisDetector(n_components=pca),
        "type": "supervised",
    },
    "autoencoder": {
        "display": "Autoencoder",
        "factory": lambda pca: AutoencoderDetector(n_components=pca, epochs=200),
        "type": "unsupervised",
    },
}

# Display name mapping for evaluation
_DISPLAY_NAMES = {
    "linear_probe": "Linear Probe",
    "contrastive_mahalanobis": "Contrastive Mahalanobis",
    "autoencoder": "Autoencoder",
    "ensemble": "Custom Ensemble",
}


# ==================================================================
# Helpers
# ==================================================================


def _compute_val_auroc(
    det: object,
    X_val: np.ndarray,
    X_val_jailbreak: np.ndarray | None,
) -> float | None:
    """Compute AUROC for a detector on validation data."""
    if X_val_jailbreak is None or len(X_val_jailbreak) == 0:
        return None
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

    scores_b = det.score(X_val)
    scores_j = det.score(X_val_jailbreak)
    all_scores = np.concatenate([scores_b, scores_j])
    all_labels = np.concatenate([
        np.zeros(len(scores_b), dtype=np.int64),
        np.ones(len(scores_j), dtype=np.int64),
    ])
    try:
        return float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        return None


def _calibrate_threshold(
    det: object,
    X_val: np.ndarray,
    X_val_jailbreak: np.ndarray | None,
    threshold_pctile: float = 95.0,
) -> tuple[float, float]:
    """Calibrate threshold. Returns (threshold, val_fpr)."""
    from sklearn.metrics import roc_curve as _roc_curve  # type: ignore[import-untyped]

    val_scores = det.score(X_val)
    if X_val_jailbreak is not None and len(X_val_jailbreak) > 0:
        jb_scores = det.score(X_val_jailbreak)
        all_scores = np.concatenate([val_scores, jb_scores])
        all_labels = np.concatenate([
            np.zeros(len(val_scores), dtype=np.int64),
            np.ones(len(jb_scores), dtype=np.int64),
        ])
        fpr_arr, tpr_arr, thresholds_arr = _roc_curve(all_labels, all_scores)
        j_stat = tpr_arr - fpr_arr
        best_idx = int(np.argmax(j_stat))
        threshold = float(np.nextafter(thresholds_arr[best_idx], -np.inf))
    else:
        threshold = float(np.percentile(val_scores, threshold_pctile))
    val_fpr = float(np.mean(val_scores > threshold))
    return threshold, val_fpr


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


# ==================================================================
# Check if a quick run already completed
# ==================================================================

if st.session_state.get("_quick_run_complete"):
    qr = st.session_state["_quick_run_results"]

    st.success("Quick Run Complete")

    # Headline metrics
    best = qr["best"]
    st.subheader(f"Best: {best['display']} — Layer {best['layer']}, PCA {best['pca']}")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("AUROC", f"{best['test_auroc']:.4f}")
    with m2:
        st.metric("AUPRC", f"{best['test_auprc']:.4f}")
    with m3:
        st.metric("Recall", f"{best['test_recall']:.4f}")
    with m4:
        st.metric("FPR", f"{best['test_fpr']:.4f}")

    # Charts
    if "test_scores_benign" in best:
        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            fig = score_distribution_chart(
                np.array(best["test_scores_benign"]),
                np.array(best["test_scores_jailbreak"]),
                best["threshold"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with chart_c2:
            fig = roc_curve_chart(
                np.array(best["test_roc_fpr"]),
                np.array(best["test_roc_tpr"]),
                best["test_auroc"],
            )
            st.plotly_chart(fig, use_container_width=True)

    # Per-detector comparison (if "All" was selected)
    if "all_detector_bests" in qr:
        st.subheader("Detector Comparison")
        for i, det_best in enumerate(qr["all_detector_bests"]):
            rank = i + 1
            st.markdown(
                f"{rank}. **{det_best['display']}** — "
                f"Layer {det_best['layer']}, PCA {det_best['pca']} "
                f"→ Test AUROC: {det_best['test_auroc']:.4f}"
            )

    # Action buttons
    st.divider()
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        if st.button("Save Model", type="primary", key="qr_save"):
            model_path = qr.get("model_path")
            if model_path:
                st.success(f"Model already saved to `{model_path}`")
            else:
                st.info("Model was saved during the run.")
    with btn2:
        if st.button("View Full Results", key="qr_results"):
            st.switch_page("pages/2_results.py")
    with btn3:
        if st.button("Run Again", key="qr_rerun"):
            st.session_state["_quick_run_complete"] = False
            if "_quick_run_results" in st.session_state:
                del st.session_state["_quick_run_results"]
            st.rerun()
    st.stop()


# ==================================================================
# SECTION 1: Data Source
# ==================================================================

st.subheader("1. Data Source")

data_mode = st.radio(
    "Choose data source",
    options=["Use existing data", "Start fresh"],
    horizontal=True,
    key="qr_data_mode",
)

existing_store_path: Path | None = None
existing_splits_path: Path | None = None
fresh_n_benign = 500
fresh_n_jailbreak = 200

if data_mode == "Use existing data":
    # Scan for existing files
    prompt_files = sorted(config.prompts_dir.glob("*.jsonl")) if config.prompts_dir.exists() else []
    h5_files = sorted(config.activations_dir.glob("*.h5")) if config.activations_dir.exists() else []
    split_files = sorted(config.data_dir.glob("splits*.json")) if config.data_dir.exists() else []

    if not h5_files:
        st.warning("No activation stores found. Use 'Start fresh' or run the Pipeline first.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        store_options = [str(p) for p in h5_files]
        selected_store = st.selectbox(
            "Activation store (.h5)",
            options=store_options,
            key="qr_store_select",
        )
        existing_store_path = Path(selected_store) if selected_store else None

    with c2:
        if split_files:
            split_options = [str(p) for p in split_files]
            selected_splits = st.selectbox(
                "Splits file (.json)",
                options=split_options,
                key="qr_splits_select",
            )
            existing_splits_path = Path(selected_splits) if selected_splits else None
        else:
            st.info("No splits file found — will create one automatically.")

    if existing_store_path and existing_store_path.exists():
        store = ActivationStore(existing_store_path)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Prompts", store.n_prompts)
        with s2:
            st.metric("Layers", len(store.layers))
        with s3:
            n_jb = int((store.get_labels() == 1).sum())
            st.metric("Jailbreaks", n_jb)

else:
    # Start fresh configuration
    fc1, fc2 = st.columns(2)
    with fc1:
        fresh_n_benign = st.slider(
            "Benign prompts", min_value=100, max_value=1000, value=500, step=50,
            key="qr_n_benign",
        )
    with fc2:
        fresh_n_jailbreak = st.slider(
            "Jailbreak prompts", min_value=50, max_value=500, value=200, step=50,
            key="qr_n_jailbreak",
        )


# ==================================================================
# SECTION 2: Extraction Config (fresh only)
# ==================================================================

extract_model_id = config.model_id
extract_layers: list[int] = []

if data_mode == "Start fresh":
    st.subheader("2. Extraction Config")

    extract_model_id = st.text_input(
        "Model", value=config.model_id, key="qr_model_id",
    )

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        layer_start = st.selectbox("Layer start", options=list(range(0, 32)), index=4, key="qr_layer_start")
    with lc2:
        layer_end = st.selectbox("Layer end", options=list(range(0, 32)), index=28, key="qr_layer_end")
    with lc3:
        layer_step = st.selectbox("Layer step", options=[1, 2, 4], index=1, key="qr_layer_step")

    extract_layers = list(range(layer_start, layer_end + 1, layer_step))
    n_layers = len(extract_layers)
    n_prompts_total = fresh_n_benign + fresh_n_jailbreak
    est_minutes = n_prompts_total * n_layers * 0.15 / 60  # rough estimate

    st.info(
        f"Will extract from **{n_layers} layers** ({extract_layers[0]}–{extract_layers[-1]}). "
        f"Estimated time: ~{est_minutes:.0f} minutes for {n_prompts_total} prompts."
    )


# ==================================================================
# SECTION 3: Detector Selection
# ==================================================================

st.subheader("3. Detector Selection" if data_mode == "Start fresh" else "2. Detector Selection")

det_options = list(_DETECTOR_OPTIONS.keys()) + ["all"]
det_labels = [_DETECTOR_OPTIONS[k]["display"] for k in _DETECTOR_OPTIONS] + ["All of the above (compare and pick best)"]

selected_det = st.radio(
    "Which detector to optimize",
    options=det_options,
    format_func=lambda k: dict(zip(det_options, det_labels))[k],
    index=0,
    key="qr_detector",
)


# ==================================================================
# SECTION 4: Auto-Optimization Settings
# ==================================================================

section_num = "4" if data_mode == "Start fresh" else "3"
st.subheader(f"{section_num}. Auto-Optimization Settings")

opt_c1, opt_c2, opt_c3 = st.columns(3)
with opt_c1:
    opt_layers = st.checkbox("Layers (test all extracted layers)", value=True, key="qr_opt_layers")
with opt_c2:
    opt_pca = st.checkbox("PCA dimensions", value=True, key="qr_opt_pca")
with opt_c3:
    opt_metric = st.selectbox("Optimization metric", options=["AUROC", "F1", "AUPRC"], key="qr_opt_metric")

pca_grid = [None, 10, 20, 30, 50, 75, 100]

# Estimate configurations
if data_mode == "Use existing data" and existing_store_path and existing_store_path.exists():
    n_test_layers = len(ActivationStore(existing_store_path).layers) if opt_layers else 1
else:
    n_test_layers = len(extract_layers) if opt_layers else 1

n_pca_options = len(pca_grid) if opt_pca else 1
n_detectors = len(_DETECTOR_OPTIONS) if selected_det == "all" else 1
n_configs = n_test_layers * n_pca_options * n_detectors

st.info(
    f"Configurations to test: **{n_configs}** "
    f"({n_test_layers} layers x {n_pca_options} PCA dims x {n_detectors} detectors)"
)


# ==================================================================
# RUN BUTTON
# ==================================================================

st.divider()

if st.button("Run Full Pipeline", type="primary", use_container_width=True, key="qr_run"):
    run_t0 = time.perf_counter()

    # Determine which detectors to run
    if selected_det == "all":
        det_keys = list(_DETECTOR_OPTIONS.keys())
    else:
        det_keys = [selected_det]

    # ---- STEP 1: Collect prompts (if fresh) ----
    step_status = st.empty()
    progress_area = st.container()

    if data_mode == "Start fresh":
        with progress_area:
            with st.status("Step 1/5: Collecting prompts...", expanded=True) as s1_status:
                t0 = time.perf_counter()
                st.write(f"Collecting {fresh_n_benign} benign prompts...")
                raw_benign = collect_benign_prompts(n=fresh_n_benign, seed=config.random_seed)
                benign_prompts = [p for p, _ in raw_benign]
                benign_sources = [s for _, s in raw_benign]
                benign_ds = PromptDataset(benign_prompts, [0] * len(benign_prompts), benign_sources)

                st.write(f"Collecting {fresh_n_jailbreak} jailbreak prompts...")
                raw_jailbreak = collect_jailbreak_prompts(n=fresh_n_jailbreak, seed=config.random_seed)
                jailbreak_prompts = [p for p, _ in raw_jailbreak]
                jailbreak_sources = [s for _, s in raw_jailbreak]
                jailbreak_ds = PromptDataset(jailbreak_prompts, [1] * len(jailbreak_prompts), jailbreak_sources)

                # Save prompts
                config.prompts_dir.mkdir(parents=True, exist_ok=True)
                benign_path = config.prompts_dir / "benign.jsonl"
                jailbreak_path = config.prompts_dir / "jailbreaks.jsonl"
                benign_ds.save(benign_path)
                jailbreak_ds.save(jailbreak_path)

                elapsed_s1 = time.perf_counter() - t0
                s1_status.update(
                    label=f"Step 1/5: Collected {len(benign_ds)} benign + {len(jailbreak_ds)} jailbreak prompts ({elapsed_s1:.0f}s)",
                    state="complete",
                )

        # Update state for prompts
        state.reset_from(1)
        state.benign_path = benign_path
        state.jailbreak_path = jailbreak_path
        state.prompts_ready = True

        # ---- STEP 2: Extract activations ----
        with progress_area:
            with st.status("Step 2/5: Extracting activations...", expanded=True) as s2_status:
                from venator.activation.extractor import ActivationExtractor

                combined = PromptDataset.merge(benign_ds, jailbreak_ds)
                n_total_prompts = len(combined)

                st.write("Loading model... (this may take a minute on first run)")
                extractor = ActivationExtractor(
                    model_id=extract_model_id,
                    layers=extract_layers,
                    config=config,
                )

                hidden_dim = extractor.hidden_dim
                layers_sorted = sorted(extractor._target_layers)
                st.write(f"Model loaded. Hidden dim: {hidden_dim}, {len(layers_sorted)} layers")

                output_path = config.activations_dir / "all.h5"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    output_path.unlink()

                store = ActivationStore.create(
                    output_path,
                    model_id=extract_model_id,
                    layers=layers_sorted,
                    hidden_dim=hidden_dim,
                )

                extract_progress = st.progress(0)
                extract_text = st.empty()
                t0 = time.perf_counter()

                for i in range(n_total_prompts):
                    prompt_text = combined.prompts[i]
                    label = combined.labels[i]
                    extract_text.text(
                        f"Prompt {i + 1}/{n_total_prompts}: {prompt_text[:60]}..."
                    )
                    activations = extractor.extract_single(prompt_text)
                    store.append(prompt_text, activations, label=label)
                    extract_progress.progress((i + 1) / n_total_prompts)

                elapsed_s2 = time.perf_counter() - t0
                s2_status.update(
                    label=f"Step 2/5: Extracted {n_total_prompts} prompts ({elapsed_s2:.0f}s)",
                    state="complete",
                )

        # Update state
        state.reset_from(2)
        state.store_path = output_path
        state.activations_ready = True
        store_path = output_path
    else:
        # Using existing data
        store_path = existing_store_path
        store = ActivationStore(store_path)

        # Update state
        state.store_path = store_path
        state.activations_ready = True

        # Check for prompt files and update state
        benign_path = config.prompts_dir / "benign.jsonl"
        jailbreak_path = config.prompts_dir / "jailbreaks.jsonl"
        if benign_path.exists() and jailbreak_path.exists():
            state.benign_path = benign_path
            state.jailbreak_path = jailbreak_path
            state.prompts_ready = True

    # ---- STEP 3: Split data ----
    with progress_area:
        with st.status("Step 3/5: Splitting data...", expanded=True) as s3_status:
            t0 = time.perf_counter()
            store = ActivationStore(store_path)

            if data_mode == "Use existing data" and existing_splits_path and existing_splits_path.exists():
                splits = SplitManager.load_splits(existing_splits_path)
                splits_path = existing_splits_path
                st.write(f"Loaded existing splits from {existing_splits_path}")
            else:
                manager = SplitManager(seed=config.random_seed)
                splits = manager.create_splits(
                    store,
                    benign_train_frac=0.70,
                    benign_val_frac=0.15,
                    jailbreak_train_frac=0.15,
                    jailbreak_val_frac=0.15,
                )
                splits_path = config.data_dir / "splits.json"
                manager.save_splits(splits, splits_path)
                st.write(
                    f"Train: {splits['train_benign'].n_samples} benign + "
                    f"{splits['train_jailbreak'].n_samples} jailbreak | "
                    f"Val: {splits['val_benign'].n_samples} + {splits['val_jailbreak'].n_samples} | "
                    f"Test: {splits['test_benign'].n_samples} + {splits['test_jailbreak'].n_samples}"
                )

            elapsed_s3 = time.perf_counter() - t0
            s3_status.update(
                label=f"Step 3/5: Data split complete ({elapsed_s3:.1f}s)",
                state="complete",
            )

    state.splits_path = splits_path
    state.splits_ready = True

    has_labeled_jailbreaks = splits["train_jailbreak"].n_samples > 0

    # ---- STEP 4: Auto-optimize ----
    available_layers = sorted(store.layers)
    test_layers = available_layers if opt_layers else [available_layers[len(available_layers) // 2]]
    test_pca_dims = pca_grid if opt_pca else [50]

    # Choose metric function
    metric_key = opt_metric.lower()  # "auroc", "f1", "auprc"

    # Track best per detector
    all_detector_bests: list[dict] = []

    with progress_area:
        with st.status("Step 4/5: Auto-optimizing...", expanded=True) as s4_status:
            t0 = time.perf_counter()
            total_configs = len(test_layers) * len(test_pca_dims) * len(det_keys)
            opt_progress = st.progress(0)
            opt_text = st.empty()
            leaderboard = st.empty()

            config_step = 0

            for det_key in det_keys:
                det_info = _DETECTOR_OPTIONS[det_key]
                best_val_score = -1.0
                best_config: dict = {}
                is_supervised = det_info["type"] == "supervised"

                for try_layer in test_layers:
                    # Load data for this layer
                    X_tr = store.get_activations(
                        try_layer, indices=splits["train_benign"].indices.tolist()
                    )
                    X_va = store.get_activations(
                        try_layer, indices=splits["val_benign"].indices.tolist()
                    )
                    X_tr_jb = (
                        store.get_activations(
                            try_layer, indices=splits["train_jailbreak"].indices.tolist()
                        )
                        if has_labeled_jailbreaks else None
                    )
                    X_va_jb = (
                        store.get_activations(
                            try_layer, indices=splits["val_jailbreak"].indices.tolist()
                        )
                        if has_labeled_jailbreaks else None
                    )

                    for try_pca in test_pca_dims:
                        config_step += 1
                        pca_label = "raw" if try_pca is None else str(try_pca)
                        opt_text.text(
                            f"[{det_info['display'].split(' (')[0]}] "
                            f"Config {config_step}/{total_configs}: "
                            f"Layer {try_layer}, PCA {pca_label}"
                        )
                        opt_progress.progress(config_step / total_configs)

                        try:
                            effective_pca = try_pca
                            det = det_info["factory"](effective_pca)

                            if is_supervised:
                                if X_tr_jb is None:
                                    continue
                                X_combined = np.vstack([X_tr, X_tr_jb])
                                y_combined = np.concatenate([
                                    np.zeros(len(X_tr), dtype=np.int64),
                                    np.ones(len(X_tr_jb), dtype=np.int64),
                                ])
                                det.fit(X_combined, y_combined)
                            else:
                                det.fit(X_tr)

                            # Evaluate on validation set
                            auroc = _compute_val_auroc(det, X_va, X_va_jb)
                            if auroc is None:
                                continue

                            # Get the requested metric
                            val_score = auroc  # default
                            if metric_key == "f1" or metric_key == "auprc":
                                # Need full metrics for F1/AUPRC
                                if X_va_jb is not None and len(X_va_jb) > 0:
                                    from venator.detection.metrics import evaluate_detector as _eval_det
                                    from sklearn.metrics import (  # type: ignore[import-untyped]
                                        precision_recall_curve as _prc,
                                        auc as _auc,
                                    )

                                    all_scores = np.concatenate([det.score(X_va), det.score(X_va_jb)])
                                    all_labels = np.concatenate([
                                        np.zeros(len(X_va), dtype=np.int64),
                                        np.ones(len(X_va_jb), dtype=np.int64),
                                    ])
                                    if metric_key == "f1":
                                        metrics = _eval_det(all_scores, all_labels)
                                        val_score = metrics.get("f1", 0.0)
                                    else:  # auprc
                                        prec, rec, _ = _prc(all_labels, all_scores)
                                        val_score = float(_auc(rec, prec))

                            if val_score > best_val_score:
                                best_val_score = val_score
                                best_config = {
                                    "layer": try_layer,
                                    "pca": try_pca,
                                    "val_score": val_score,
                                    "det_key": det_key,
                                    "display": det_info["display"].split(" (")[0],
                                }

                                # Update leaderboard
                                leaderboard.markdown(
                                    f"**Current best:** "
                                    f"{best_config['display']} — "
                                    f"Layer {try_layer}, PCA {pca_label} "
                                    f"→ Val {opt_metric}: {val_score:.4f}"
                                )
                        except Exception:
                            continue

                if best_config:
                    all_detector_bests.append(best_config)

            elapsed_s4 = time.perf_counter() - t0
            s4_status.update(
                label=f"Step 4/5: Auto-optimization complete ({elapsed_s4:.1f}s)",
                state="complete",
            )

    if not all_detector_bests:
        st.error("No valid configurations found. Check that your data has labeled jailbreaks for supervised detectors.")
        st.stop()

    # Sort by val score and pick overall best
    all_detector_bests.sort(key=lambda x: x["val_score"], reverse=True)
    overall_best = all_detector_bests[0]

    # ---- STEP 5: Final evaluation on TEST set ----
    with progress_area:
        with st.status("Step 5/5: Evaluating on test set...", expanded=True) as s5_status:
            t0 = time.perf_counter()

            best_layer = overall_best["layer"]
            best_pca = overall_best["pca"]
            best_det_key = overall_best["det_key"]
            best_det_info = _DETECTOR_OPTIONS[best_det_key]

            # Retrain best detector on best config
            X_train_best = store.get_activations(
                best_layer, indices=splits["train_benign"].indices.tolist()
            )
            X_val_best = store.get_activations(
                best_layer, indices=splits["val_benign"].indices.tolist()
            )
            X_train_jb_best = (
                store.get_activations(
                    best_layer, indices=splits["train_jailbreak"].indices.tolist()
                )
                if has_labeled_jailbreaks else None
            )
            X_val_jb_best = (
                store.get_activations(
                    best_layer, indices=splits["val_jailbreak"].indices.tolist()
                )
                if has_labeled_jailbreaks else None
            )

            # Train the final best detector
            final_det = best_det_info["factory"](best_pca)
            if best_det_info["type"] == "supervised":
                X_combined = np.vstack([X_train_best, X_train_jb_best])
                y_combined = np.concatenate([
                    np.zeros(len(X_train_best), dtype=np.int64),
                    np.ones(len(X_train_jb_best), dtype=np.int64),
                ])
                final_det.fit(X_combined, y_combined)
            else:
                final_det.fit(X_train_best)

            # Calibrate threshold on validation set
            threshold, val_fpr = _calibrate_threshold(
                final_det, X_val_best, X_val_jb_best,
            )

            # Build ensemble for saving (wraps the single best detector)
            ensemble = DetectorEnsemble(threshold_percentile=config.anomaly_threshold_percentile)
            det_type_enum = (
                DetectorType.SUPERVISED
                if best_det_info["type"] == "supervised"
                else DetectorType.UNSUPERVISED
            )
            ensemble.add_detector(best_det_key, final_det, weight=1.0, detector_type=det_type_enum)

            # If "all" was selected, also train and add other best detectors
            if selected_det == "all":
                for other_best in all_detector_bests[1:]:
                    other_key = other_best["det_key"]
                    other_info = _DETECTOR_OPTIONS[other_key]
                    other_layer = other_best["layer"]
                    other_pca = other_best["pca"]

                    X_tr_o = store.get_activations(
                        other_layer, indices=splits["train_benign"].indices.tolist()
                    )
                    other_det = other_info["factory"](other_pca)

                    other_type = (
                        DetectorType.SUPERVISED
                        if other_info["type"] == "supervised"
                        else DetectorType.UNSUPERVISED
                    )

                    if other_info["type"] == "supervised" and has_labeled_jailbreaks:
                        X_tr_jb_o = store.get_activations(
                            other_layer, indices=splits["train_jailbreak"].indices.tolist()
                        )
                        X_comb_o = np.vstack([X_tr_o, X_tr_jb_o])
                        y_comb_o = np.concatenate([
                            np.zeros(len(X_tr_o), dtype=np.int64),
                            np.ones(len(X_tr_jb_o), dtype=np.int64),
                        ])
                        other_det.fit(X_comb_o, y_comb_o)
                    else:
                        other_det.fit(X_tr_o)

                    ensemble.add_detector(
                        other_key, other_det, weight=0.5,
                        detector_type=other_type,
                    )

            # Train ensemble (fit normalization)
            ensemble.fit(
                X_train_best, X_val_best,
                X_train_jailbreak=X_train_jb_best,
                X_val_jailbreak=X_val_jb_best,
            )

            # Save model
            ensemble_type = "supervised" if best_det_info["type"] == "supervised" else "unsupervised"
            output_dir = config.models_dir / f"detector_{ensemble_type}_v1"
            ensemble.save(output_dir)

            meta = {
                "layer": int(best_layer),
                "model_id": store.model_id,
                "extraction_layers": [int(l) for l in store.layers],
                "ensemble_type": ensemble_type,
                "primary_name": best_det_key,
                "primary_threshold": threshold,
            }
            with open(output_dir / "pipeline_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            st.write(f"Model saved to `{output_dir}`")

            # Evaluate on test set (ONLY happens here, never during optimization)
            test_benign_indices = splits["test_benign"].indices.tolist()
            test_jailbreak_indices = splits["test_jailbreak"].indices.tolist()
            X_test_benign = store.get_activations(best_layer, indices=test_benign_indices)
            X_test_jailbreak = store.get_activations(best_layer, indices=test_jailbreak_indices)

            # Build detectors dict for evaluate_all_detectors
            eval_detectors: dict[str, object] = {}
            eval_detector_types: dict[str, str] = {}
            eval_display_names: dict[str, str] = {}

            for name, det, _weight in ensemble.detectors:
                eval_detectors[name] = det
                dtype = ensemble.detector_types_.get(name, DetectorType.UNSUPERVISED)
                eval_detector_types[name] = dtype.value
                eval_display_names[name] = _DISPLAY_NAMES.get(name, name)

            st.write(f"Evaluating {len(eval_detectors)} detector(s) on test set...")

            eval_results = evaluate_all_detectors(
                eval_detectors,
                X_test_benign,
                X_test_jailbreak,
                detector_types=eval_detector_types,
                display_names=eval_display_names,
            )

            elapsed_s5 = time.perf_counter() - t0
            s5_status.update(
                label=f"Step 5/5: Evaluation complete ({elapsed_s5:.1f}s)",
                state="complete",
            )

    # ---- Populate PipelineState for other pages ----
    state.model_path = output_dir
    state.model_ready = True

    # Build train_metrics
    train_metrics = {
        "val_false_positive_rate": val_fpr,
        "threshold": threshold,
        "ensemble_threshold": float(ensemble.threshold_),
        "layer": int(best_layer),
        "pca_dims": int(best_pca) if best_pca is not None else 0,
        "n_detectors": len(det_keys),
        "ensemble_type": ensemble_type,
        "primary_detector": best_det_key,
        "training_time_s": round(time.perf_counter() - run_t0, 1),
    }
    state.train_metrics = train_metrics

    # Build trained_detectors list
    trained_detectors = []
    for name, det, weight in ensemble.detectors:
        auroc = _compute_val_auroc(det, X_val_best, X_val_jb_best)
        trained_detectors.append({
            "name": name,
            "display": _DISPLAY_NAMES.get(name, name),
            "val_auroc": auroc,
            "time_s": 0.0,
            "is_primary": name == best_det_key,
        })
    state.trained_detectors = trained_detectors

    # Populate eval results for Results/Explore pages
    serialized_results = [_result_to_dict(r) for r in eval_results]
    all_test_indices = test_benign_indices + test_jailbreak_indices
    test_prompts = store.get_prompts(indices=all_test_indices)
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

    # Build quick run results for the completion view
    best_eval = next(
        (r for r in eval_results if r.detector_name == best_det_key),
        eval_results[0],
    )

    qr_results: dict = {
        "best": {
            "det_key": best_det_key,
            "display": _DISPLAY_NAMES.get(best_det_key, best_det_key),
            "layer": best_layer,
            "pca": best_pca if best_pca is not None else "raw",
            "threshold": threshold,
            "test_auroc": float(best_eval.auroc),
            "test_auprc": float(best_eval.auprc),
            "test_recall": float(best_eval.recall),
            "test_fpr": float(best_eval.fpr),
            "test_scores_benign": best_eval.scores_benign.tolist(),
            "test_scores_jailbreak": best_eval.scores_jailbreak.tolist(),
            "test_roc_fpr": best_eval.roc_fpr.tolist(),
            "test_roc_tpr": best_eval.roc_tpr.tolist(),
        },
        "model_path": str(output_dir),
        "total_time_s": round(time.perf_counter() - run_t0, 1),
    }

    # If "all" was selected, add per-detector comparison
    if selected_det == "all":
        per_det = []
        for det_best_cfg in all_detector_bests:
            dk = det_best_cfg["det_key"]
            matched_eval = next(
                (r for r in eval_results if r.detector_name == dk), None
            )
            test_auroc = float(matched_eval.auroc) if matched_eval else det_best_cfg["val_score"]
            per_det.append({
                "det_key": dk,
                "display": det_best_cfg["display"],
                "layer": det_best_cfg["layer"],
                "pca": det_best_cfg["pca"] if det_best_cfg["pca"] is not None else "raw",
                "test_auroc": test_auroc,
            })
        qr_results["all_detector_bests"] = per_det

    st.session_state["_quick_run_complete"] = True
    st.session_state["_quick_run_results"] = qr_results

    total_elapsed = time.perf_counter() - run_t0
    st.success(f"Quick Run complete in {total_elapsed:.0f}s!")
    st.rerun()
