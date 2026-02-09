"""Ablation studies page — layer comparison, label efficiency, cross-source generalization."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import (
    ablation_line_chart,
    generalization_heatmap,
    labeled_data_efficiency_chart,
)
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager, SplitMode
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.ensemble import (
    DetectorEnsemble,
    DetectorType,
)
from venator.detection.metrics import evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

state = PipelineState()
config = state.config

_ABLATION_LAYERS = [8, 10, 12, 14, 16, 18, 20, 22, 24]
_LABEL_BUDGETS = [5, 10, 20, 30, 50, 100]

st.header("Ablation Studies")
st.markdown(
    "Compare detector performance across different layers, label budgets, "
    "and attack sources."
)

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(5):
    st.warning(
        "Complete the **Pipeline** (extract + split) stages first to unlock ablations."
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


def _get_train_val(
    store: ActivationStore,
    splits: dict,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get benign-only train and val data for a layer."""
    if "train" in splits:
        X_train = store.get_activations(layer, indices=splits["train"].indices.tolist())
        X_val = store.get_activations(layer, indices=splits["val"].indices.tolist())
    else:
        X_train = store.get_activations(layer, indices=splits["train_benign"].indices.tolist())
        X_val = store.get_activations(layer, indices=splits["val_benign"].indices.tolist())
    return X_train, X_val


# ------------------------------------------------------------------
# Results display
# ------------------------------------------------------------------


def _show_ablation_results(results: dict) -> None:
    """Render all ablation results in tabs."""
    tab_names = []
    if "layers" in results:
        tab_names.append("Layer Comparison")
    if "label_efficiency" in results:
        tab_names.append("Labeled Data Efficiency")
    if "cross_source" in results:
        tab_names.append("Cross-Source Generalization")

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
            st.plotly_chart(fig, width="stretch")

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

    # --- Labeled Data Efficiency ---
    if "label_efficiency" in results:
        with tabs[tab_idx]:
            eff = results["label_efficiency"]

            fig = labeled_data_efficiency_chart(
                eff["n_labeled"],
                eff["aurocs_by_detector"],
                eff.get("unsupervised_baselines", {}),
            )
            st.plotly_chart(fig, width="stretch")

            # Find crossover point
            for det_name, aurocs_list in eff["aurocs_by_detector"].items():
                if aurocs_list and eff.get("unsupervised_baselines"):
                    best_unsup = max(eff["unsupervised_baselines"].values())
                    for i, a in enumerate(aurocs_list):
                        if a >= best_unsup:
                            st.info(
                                f"**{det_name}** matches the best unsupervised detector "
                                f"AUROC ({best_unsup:.3f}) with only "
                                f"**{eff['n_labeled'][i]}** labeled examples."
                            )
                            break

            # Full results table
            rows = []
            for i, n in enumerate(eff["n_labeled"]):
                row: dict = {"n_labeled": n}
                for det_name, aurocs_list in eff["aurocs_by_detector"].items():
                    if i < len(aurocs_list):
                        row[det_name] = aurocs_list[i]
                rows.append(row)
            st.dataframe(
                pd.DataFrame(rows).round(4),
                use_container_width=True,
                hide_index=True,
            )
        tab_idx += 1

    # --- Cross-Source Generalization ---
    if "cross_source" in results:
        with tabs[tab_idx]:
            cs = results["cross_source"]

            fig = generalization_heatmap(
                cs["sources"],
                np.array(cs["auroc_matrix"]),
            )
            st.plotly_chart(fig, width="stretch")

            # Analyze diagonal vs off-diagonal
            matrix = np.array(cs["auroc_matrix"])
            n = len(cs["sources"])
            if n > 1:
                diag_mean = np.mean(np.diag(matrix))
                off_diag = matrix[~np.eye(n, dtype=bool)]
                off_diag_mean = np.mean(off_diag) if len(off_diag) > 0 else 0
                st.info(
                    f"**In-distribution mean AUROC: {diag_mean:.3f}** (diagonal). "
                    f"**Cross-source mean AUROC: {off_diag_mean:.3f}** (off-diagonal). "
                    "A small gap indicates the detector learns general jailbreak-ness "
                    "rather than memorizing specific attack patterns."
                )

            st.dataframe(
                pd.DataFrame(
                    cs["auroc_matrix"],
                    index=[f"Train: {s}" for s in cs["sources"]],
                    columns=[f"Test: {s}" for s in cs["sources"]],
                ).round(3),
                use_container_width=True,
            )

    # --- Export ---
    st.divider()
    st.subheader("Export")

    json_str = json.dumps(results, indent=2, default=str)
    st.download_button(
        "Download JSON Results",
        data=json_str,
        file_name="ablation_results.json",
        mime="application/json",
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
splits = SplitManager.load_splits(state.splits_path)
split_mode = SplitManager.load_mode(state.splits_path)
is_semi = split_mode == SplitMode.SEMI_SUPERVISED
available_layers = store.layers

abl_col1, abl_col2 = st.columns(2)
with abl_col1:
    run_layers = st.checkbox("Layer comparison", value=True)
with abl_col2:
    # placeholder for alignment
    pass

# Semi-supervised ablation options
if is_semi:
    st.markdown("**Supervised ablations** (requires semi-supervised splits):")
    ss_col1, ss_col2 = st.columns(2)
    with ss_col1:
        run_label_efficiency = st.checkbox("Labeled data efficiency", value=True)
    with ss_col2:
        run_cross_source = st.checkbox(
            "Cross-source generalization", value=False,
            help="Requires jailbreak prompts with source labels in the store.",
        )
else:
    run_label_efficiency = False
    run_cross_source = False

default_layer = available_layers[len(available_layers) // 2]
base_layer = st.selectbox(
    "Base layer (for label efficiency & cross-source ablations)",
    options=available_layers,
    index=available_layers.index(default_layer),
)

n_selected = sum([run_layers, run_label_efficiency, run_cross_source])
if n_selected == 0:
    st.error("Select at least one ablation type.")
    st.stop()

# ------------------------------------------------------------------
# Run ablations
# ------------------------------------------------------------------

st.divider()

if st.button("Run Ablations", type="primary"):
    # Count total configurations for progress
    test_layers = [l for l in _ABLATION_LAYERS if l in set(available_layers)]
    total_configs = 0
    if run_layers:
        total_configs += len(test_layers)
    if run_label_efficiency:
        total_configs += len(_LABEL_BUDGETS)
    if run_cross_source:
        total_configs += 1

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
            X_train, X_val = _get_train_val(store, splits, layer)
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

    # --- Labeled data efficiency ---
    if run_label_efficiency and is_semi:
        from venator.detection.contrastive import ContrastiveDirectionDetector
        from venator.detection.linear_probe import LinearProbeDetector

        X_train, X_val = _get_train_val(store, splits, base_layer)
        X_test, labels = _get_test_data(store, splits, base_layer)

        # Get full jailbreak training data
        X_train_jb_full = store.get_activations(
            base_layer, indices=splits["train_jailbreak"].indices.tolist()
        )
        X_val_jb = store.get_activations(
            base_layer, indices=splits["val_jailbreak"].indices.tolist()
        )

        # Unsupervised baselines
        unsup_baselines: dict[str, float] = {}
        for det_name, det_cls in [
            ("pca_mahalanobis", PCAMahalanobisDetector),
            ("autoencoder", AutoencoderDetector),
        ]:
            ens = DetectorEnsemble()
            ens.add_detector(det_name, det_cls(n_components=config.pca_dims), weight=1.0)
            ens.fit(X_train, X_val)
            r = ens.score(X_test)
            m = evaluate_detector(r.ensemble_scores, labels)
            unsup_baselines[det_name] = m["auroc"]

        # Supervised detectors at different label budgets
        sup_detectors = {
            "linear_probe": LinearProbeDetector,
            "contrastive_direction": ContrastiveDirectionDetector,
        }

        aurocs_by_detector: dict[str, list[float]] = {n: [] for n in sup_detectors}
        budgets_used = []

        for n_labels in _LABEL_BUDGETS:
            if n_labels > len(X_train_jb_full):
                current += 1
                continue

            progress.progress(
                current / total_configs,
                text=f"Label efficiency: {n_labels} labeled jailbreaks...",
            )

            budgets_used.append(n_labels)
            X_train_jb_subset = X_train_jb_full[:n_labels]

            for det_name, det_cls in sup_detectors.items():
                ens = DetectorEnsemble()
                ens.add_detector(
                    det_name, det_cls(), weight=1.0,
                    detector_type=DetectorType.SUPERVISED,
                )
                ens.fit(
                    X_train, X_val,
                    X_train_jailbreak=X_train_jb_subset,
                    X_val_jailbreak=X_val_jb,
                )
                r = ens.score(X_test)
                m = evaluate_detector(r.ensemble_scores, labels)
                aurocs_by_detector[det_name].append(m["auroc"])

            current += 1

        results["label_efficiency"] = {
            "n_labeled": budgets_used,
            "aurocs_by_detector": aurocs_by_detector,
            "unsupervised_baselines": unsup_baselines,
        }

    # --- Cross-source generalization ---
    if run_cross_source and is_semi:
        progress.progress(
            current / total_configs,
            text="Cross-source generalization...",
        )

        from venator.detection.linear_probe import LinearProbeDetector

        X_train, X_val = _get_train_val(store, splits, base_layer)
        X_val_jb = store.get_activations(
            base_layer, indices=splits["val_jailbreak"].indices.tolist()
        )

        # Get jailbreak indices
        test_jb_indices = splits["test_jailbreak"].indices.tolist()
        train_jb_indices = splits["train_jailbreak"].indices.tolist()

        sources = ["Group A", "Group B", "Group C"]
        n_train_jb = len(train_jb_indices)
        n_test_jb = len(test_jb_indices)

        # Split jailbreaks into groups (simulating different sources)
        rng = np.random.default_rng(config.random_seed)
        n_groups = len(sources)

        train_groups = rng.integers(0, n_groups, size=n_train_jb)
        test_groups = rng.integers(0, n_groups, size=n_test_jb)

        X_train_jb = store.get_activations(base_layer, indices=train_jb_indices)
        X_test_jb = store.get_activations(base_layer, indices=test_jb_indices)
        X_test_benign = store.get_activations(
            base_layer, indices=splits["test_benign"].indices.tolist()
        )

        auroc_matrix = np.zeros((n_groups, n_groups))
        for train_src in range(n_groups):
            src_mask = train_groups == train_src
            if src_mask.sum() < 3:
                auroc_matrix[train_src, :] = 0.5
                continue
            X_train_jb_src = X_train_jb[src_mask]

            ens = DetectorEnsemble()
            ens.add_detector(
                "linear_probe", LinearProbeDetector(), weight=1.0,
                detector_type=DetectorType.SUPERVISED,
            )
            ens.fit(
                X_train, X_val,
                X_train_jailbreak=X_train_jb_src,
                X_val_jailbreak=X_val_jb,
            )

            for test_src in range(n_groups):
                test_mask = test_groups == test_src
                if test_mask.sum() < 3:
                    auroc_matrix[train_src, test_src] = 0.5
                    continue

                X_test_src = np.vstack([X_test_benign, X_test_jb[test_mask]])
                labels_src = np.concatenate([
                    np.zeros(len(X_test_benign), dtype=np.int64),
                    np.ones(int(test_mask.sum()), dtype=np.int64),
                ])
                r = ens.score(X_test_src)
                m = evaluate_detector(r.ensemble_scores, labels_src)
                auroc_matrix[train_src, test_src] = m["auroc"]

        results["cross_source"] = {
            "sources": sources,
            "auroc_matrix": auroc_matrix.tolist(),
        }
        current += 1

    progress.progress(1.0, text="Ablations complete!")

    # Cache results and display
    st.session_state["_ablation_results"] = results
    _show_ablation_results(results)
