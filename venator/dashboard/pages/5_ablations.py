"""Ablation studies page — per-detector ablation with layer, PCA, label efficiency, and cross-source tabs.

The user selects a detector class from the dropdown, configures which
ablations to run, and results are cached per-detector so switching
detectors shows previous results instantly.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import (
    ablation_line_chart,
    ablation_multi_line_chart,
    generalization_heatmap,
    labeled_data_efficiency_chart,
)
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager
from venator.detection.autoencoder import AutoencoderDetector
from venator.detection.contrastive import (
    ContrastiveDirectionDetector,
    ContrastiveMahalanobisDetector,
)
from venator.detection.ensemble import DetectorEnsemble, DetectorType
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.linear_probe import LinearProbeDetector, MLPProbeDetector
from venator.detection.metrics import evaluate_detector
from venator.detection.pca_mahalanobis import PCAMahalanobisDetector

state = PipelineState()
config = state.config

# Detector registry (mirrors 1_pipeline.py)
_DETECTOR_REGISTRY: dict[str, dict] = {
    "linear_probe": {
        "display": "Linear Probe",
        "type": "supervised",
        "uses_pca": True,
        "factory": lambda pca: LinearProbeDetector(n_components=pca),
    },
    "contrastive_mahalanobis": {
        "display": "Contrastive Mahalanobis",
        "type": "supervised",
        "uses_pca": True,
        "factory": lambda pca: ContrastiveMahalanobisDetector(n_components=pca),
    },
    "contrastive_direction": {
        "display": "Contrastive Direction",
        "type": "supervised",
        "uses_pca": False,
        "factory": lambda _pca: ContrastiveDirectionDetector(),
    },
    "mlp_probe": {
        "display": "MLP Probe",
        "type": "supervised",
        "uses_pca": True,
        "factory": lambda pca: MLPProbeDetector(n_components=pca),
    },
    "pca_mahalanobis": {
        "display": "PCA + Mahalanobis",
        "type": "unsupervised",
        "uses_pca": True,
        "factory": lambda pca: PCAMahalanobisDetector(n_components=pca),
    },
    "isolation_forest": {
        "display": "Isolation Forest",
        "type": "unsupervised",
        "uses_pca": True,
        "factory": lambda pca: IsolationForestDetector(n_components=pca),
    },
    "autoencoder": {
        "display": "Autoencoder",
        "type": "unsupervised",
        "uses_pca": True,
        "factory": lambda pca: AutoencoderDetector(n_components=pca),
    },
}

_LABEL_BUDGETS = [5, 10, 15, 20, 30, 50, 75, 100, 150]
_PCA_DIMS = [None, 10, 20, 30, 50, 75, 100, 150, 200]

st.header("Ablation Studies")
st.markdown(
    "Investigate how detector performance varies across layers, PCA dimensions, "
    "label budgets, and attack sources."
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
# Load data references
# ------------------------------------------------------------------

store = ActivationStore(state.store_path)
splits = SplitManager.load_splits(state.splits_path)
available_layers = sorted(store.layers)
has_labeled_jailbreaks = splits["train_jailbreak"].n_samples > 0


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
    X_train = store.get_activations(layer, indices=splits["train_benign"].indices.tolist())
    X_val = store.get_activations(layer, indices=splits["val_benign"].indices.tolist())
    return X_train, X_val


def _train_and_evaluate(
    det_key: str,
    pca_dims: int | None,
    store: ActivationStore,
    splits: dict,
    layer: int,
) -> dict[str, float]:
    """Train a fresh detector instance and evaluate on test data.

    Returns dict with auroc, auprc, precision, recall, f1, fpr_at_95_tpr.
    """
    info = _DETECTOR_REGISTRY[det_key]
    X_train, X_val = _get_train_val(store, splits, layer)
    X_test, labels = _get_test_data(store, splits, layer)

    effective_pca = pca_dims if (pca_dims is not None and info["uses_pca"]) else 50
    detector = info["factory"](effective_pca)

    if info["type"] == "supervised":
        X_train_jb = store.get_activations(
            layer, indices=splits["train_jailbreak"].indices.tolist()
        )
        X_combined = np.vstack([X_train, X_train_jb])
        y_combined = np.concatenate([
            np.zeros(len(X_train), dtype=np.int64),
            np.ones(len(X_train_jb), dtype=np.int64),
        ])
        detector.fit(X_combined, y_combined)
    else:
        detector.fit(X_train)

    scores = detector.score(X_test)
    return evaluate_detector(scores, labels)


def _get_ablation_cache() -> dict[str, dict]:
    """Get the per-detector ablation results cache from session state."""
    if "_ablation_cache" not in st.session_state:
        st.session_state["_ablation_cache"] = {}
    return st.session_state["_ablation_cache"]


# ------------------------------------------------------------------
# Detector and Ablation Selection
# ------------------------------------------------------------------

st.subheader("Configuration")

det_col, abl_col = st.columns([1, 2])

with det_col:
    detector_options = {info["display"]: key for key, info in _DETECTOR_REGISTRY.items()}
    selected_display = st.selectbox(
        "Run ablations for",
        options=list(detector_options.keys()),
        index=0,
        key="ablation_detector_select",
    )
    selected_det_key = detector_options[selected_display]
    selected_info = _DETECTOR_REGISTRY[selected_det_key]
    is_supervised = selected_info["type"] == "supervised"
    uses_pca = selected_info["uses_pca"]

with abl_col:
    st.markdown("**Available ablations:**")
    abl_checks = st.columns(2)
    with abl_checks[0]:
        run_layers = st.checkbox("Layer comparison", value=True, key="abl_layers")
        run_pca = st.checkbox(
            "PCA dimensions",
            value=uses_pca,
            disabled=not uses_pca,
            key="abl_pca",
            help="Not applicable — this detector doesn't use PCA." if not uses_pca else None,
        )
    with abl_checks[1]:
        run_label_eff = st.checkbox(
            "Labeled data efficiency",
            value=is_supervised and has_labeled_jailbreaks,
            disabled=not (is_supervised and has_labeled_jailbreaks),
            key="abl_label_eff",
            help=(
                "Only available for supervised detectors with labeled jailbreaks."
                if not (is_supervised and has_labeled_jailbreaks)
                else None
            ),
        )
        run_cross_source = st.checkbox(
            "Cross-source generalization",
            value=False,
            disabled=not (is_supervised and has_labeled_jailbreaks),
            key="abl_cross_source",
            help=(
                "Requires a supervised detector and labeled jailbreaks."
                if not (is_supervised and has_labeled_jailbreaks)
                else None
            ),
        )

# ------------------------------------------------------------------
# Layer range configuration
# ------------------------------------------------------------------

st.markdown("**Layer range:**")

if len(available_layers) <= 1:
    st.warning("Only one layer available. Extract more layers for meaningful ablation.")
    layer_start = available_layers[0] if available_layers else 0
    layer_end = layer_start
    layer_step = 1
else:
    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        layer_start = st.selectbox(
            "Start", options=available_layers,
            index=0, key="abl_layer_start",
        )
    with l_col2:
        layer_end = st.selectbox(
            "End", options=available_layers,
            index=len(available_layers) - 1, key="abl_layer_end",
        )
    with l_col3:
        layer_step = st.selectbox(
            "Step", options=[1, 2, 4],
            index=0, key="abl_layer_step",
        )

    if layer_start > layer_end:
        layer_start, layer_end = layer_end, layer_start

test_layers = [
    l for l in available_layers
    if layer_start <= l <= layer_end and (l - layer_start) % layer_step == 0
]

if test_layers:
    st.caption(f"Testing layers: {', '.join(str(l) for l in test_layers)}")

# Suggest wider extraction if layers are limited
_WIDE_RANGE = set(range(4, 26, 2))
missing_from_wide = _WIDE_RANGE - set(available_layers)
if missing_from_wide and run_layers:
    st.info(
        f"Only layers **{available_layers}** are available in the HDF5 store. "
        "Go to **Pipeline > Extraction** to extract additional layers for a wider ablation range."
    )

# Base layer for non-layer ablations
default_layer = available_layers[len(available_layers) // 2]
base_layer = st.selectbox(
    "Base layer (for PCA, label efficiency & cross-source ablations)",
    options=available_layers,
    index=available_layers.index(default_layer),
    key="abl_base_layer",
)

n_selected = sum([run_layers, run_pca, run_label_eff, run_cross_source])
if n_selected == 0:
    st.error("Select at least one ablation type.")
    st.stop()

# ------------------------------------------------------------------
# Cached results — show if available for selected detector
# ------------------------------------------------------------------

cache = _get_ablation_cache()
cached_results = cache.get(selected_det_key)


def _show_results(det_key: str, results: dict) -> None:
    """Render ablation results for a single detector in tabs."""
    info = _DETECTOR_REGISTRY[det_key]
    all_cache = _get_ablation_cache()

    tab_names = []
    if "layers" in results:
        tab_names.append("Layer Comparison")
    if "pca_dims" in results:
        tab_names.append("PCA Dimensions")
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

            # Check if other detectors have layer results for comparison
            other_layer_results = {
                _DETECTOR_REGISTRY[k]["display"]: cache_val["layers"]
                for k, cache_val in all_cache.items()
                if k != det_key and "layers" in cache_val
            }

            compare_detectors = False
            if other_layer_results:
                compare_detectors = st.checkbox(
                    "Compare detectors", value=False,
                    key="abl_compare_layers",
                )

            if compare_detectors and other_layer_results:
                series: dict[str, tuple[list, list]] = {
                    info["display"]: (layers, aurocs),
                }
                for name, other_data in other_layer_results.items():
                    series[name] = (
                        [r["layer"] for r in other_data],
                        [r["auroc"] for r in other_data],
                    )
                fig = ablation_multi_line_chart(
                    series, "Layer", "AUROC",
                    title="AUROC by Transformer Layer (All Detectors)",
                )
            else:
                fig = ablation_line_chart(
                    layers, aurocs, "Layer", "AUROC",
                    title=f"AUROC by Transformer Layer — {info['display']}",
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
            dims = [r["pca_dim"] for r in pca_data]
            dim_labels = [str(d) if d is not None else "None" for d in dims]
            aurocs = [r["auroc"] for r in pca_data]

            # For the chart, use numeric x-axis; represent "None" as 0
            x_vals = [d if d is not None else 0 for d in dims]

            fig = ablation_line_chart(
                x_vals, aurocs, "PCA Dimensions (0 = raw)", "AUROC",
                title=f"AUROC by PCA Dimensions — {info['display']}",
            )
            st.plotly_chart(fig, use_container_width=True)

            best = max(pca_data, key=lambda r: r["auroc"])
            best_label = str(best["pca_dim"]) if best["pca_dim"] is not None else "None (raw)"
            st.info(f"**Best PCA dims: {best_label}** (AUROC = {best['auroc']:.4f}).")

            df = pd.DataFrame(pca_data)
            df["pca_dim"] = df["pca_dim"].apply(lambda d: "raw" if d is None else d)
            display_cols = ["pca_dim", "auroc", "auprc", "f1", "time_s"]
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
            st.plotly_chart(fig, use_container_width=True)

            # Find crossover point
            for eff_det_name, aurocs_list in eff["aurocs_by_detector"].items():
                if aurocs_list and eff.get("unsupervised_baselines"):
                    best_unsup = max(eff["unsupervised_baselines"].values())
                    for i, a in enumerate(aurocs_list):
                        if a >= best_unsup:
                            st.info(
                                f"**{eff_det_name}** matches the best unsupervised detector "
                                f"AUROC ({best_unsup:.3f}) with only "
                                f"**{eff['n_labeled'][i]}** labeled examples."
                            )
                            break

            rows = []
            for i, n in enumerate(eff["n_labeled"]):
                row: dict = {"n_labeled": n}
                for eff_det_name, aurocs_list in eff["aurocs_by_detector"].items():
                    if i < len(aurocs_list):
                        row[eff_det_name] = aurocs_list[i]
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
            st.plotly_chart(fig, use_container_width=True)

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
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            "Export JSON",
            data=json_str,
            file_name=f"ablation_{det_key}.json",
            mime="application/json",
            key=f"export_json_{det_key}",
        )


# Show cached results
if cached_results:
    st.success(f"Ablation results available for **{selected_display}**.")
    _show_results(selected_det_key, cached_results)

    # Show which other detectors have cached results
    other_cached = [
        _DETECTOR_REGISTRY[k]["display"]
        for k in cache if k != selected_det_key
    ]
    if other_cached:
        st.caption(f"Also cached: {', '.join(other_cached)}")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Re-run Ablations", key="rerun_ablations"):
            cache.pop(selected_det_key, None)
            st.rerun()
    with btn_col2:
        if st.button("Clear All Cached Results", key="clear_all_ablations"):
            st.session_state["_ablation_cache"] = {}
            st.rerun()
    st.stop()

# ------------------------------------------------------------------
# Run ablations
# ------------------------------------------------------------------

st.divider()

if st.button("Run Ablations", type="primary", key="run_ablations"):
    # Count configs for progress
    total_configs = 0
    if run_layers:
        total_configs += len(test_layers)
    if run_pca:
        total_configs += len(_PCA_DIMS)
    if run_label_eff:
        total_configs += len(_LABEL_BUDGETS) + 2  # +2 for unsupervised baselines
    if run_cross_source:
        total_configs += 9  # 3x3 matrix

    current = 0
    results: dict = {}
    progress = st.progress(0.0, text="Starting ablations...")

    # --- Layer ablation ---
    if run_layers:
        layer_results = []
        for layer in test_layers:
            progress.progress(
                current / max(total_configs, 1),
                text=f"Layer ablation: layer {layer}...",
            )

            t0 = time.perf_counter()
            metrics = _train_and_evaluate(
                selected_det_key, config.pca_dims, store, splits, layer,
            )
            elapsed = time.perf_counter() - t0
            layer_results.append({
                "layer": layer, "time_s": round(elapsed, 2), **metrics,
            })
            current += 1

        results["layers"] = layer_results

    # --- PCA dimensions ablation ---
    if run_pca and uses_pca:
        pca_results = []
        X_train, X_val = _get_train_val(store, splits, base_layer)
        X_test, labels = _get_test_data(store, splits, base_layer)

        for pca_dim in _PCA_DIMS:
            dim_label = str(pca_dim) if pca_dim is not None else "raw"
            progress.progress(
                current / max(total_configs, 1),
                text=f"PCA ablation: {dim_label} dims...",
            )

            t0 = time.perf_counter()

            # For "None" (raw), use the full hidden dim
            effective_dim = pca_dim if pca_dim is not None else X_train.shape[1]

            # Skip if PCA dims would exceed feature count
            if pca_dim is not None and pca_dim >= X_train.shape[1]:
                current += 1
                continue

            try:
                detector = selected_info["factory"](effective_dim)

                if is_supervised:
                    X_train_jb = store.get_activations(
                        base_layer, indices=splits["train_jailbreak"].indices.tolist()
                    )
                    X_combined = np.vstack([X_train, X_train_jb])
                    y_combined = np.concatenate([
                        np.zeros(len(X_train), dtype=np.int64),
                        np.ones(len(X_train_jb), dtype=np.int64),
                    ])
                    detector.fit(X_combined, y_combined)
                else:
                    detector.fit(X_train)

                scores = detector.score(X_test)
                metrics = evaluate_detector(scores, labels)
            except Exception:
                current += 1
                continue

            elapsed = time.perf_counter() - t0
            pca_results.append({
                "pca_dim": pca_dim, "time_s": round(elapsed, 2), **metrics,
            })
            current += 1

        if pca_results:
            results["pca_dims"] = pca_results

    # --- Labeled data efficiency ---
    if run_label_eff and is_supervised and has_labeled_jailbreaks:
        X_train, X_val = _get_train_val(store, splits, base_layer)
        X_test, labels = _get_test_data(store, splits, base_layer)

        X_train_jb_full = store.get_activations(
            base_layer, indices=splits["train_jailbreak"].indices.tolist()
        )

        # Unsupervised baselines
        unsup_baselines: dict[str, float] = {}
        for unsup_name, unsup_cls in [
            ("PCA + Mahalanobis", PCAMahalanobisDetector),
            ("Autoencoder", AutoencoderDetector),
        ]:
            progress.progress(
                current / max(total_configs, 1),
                text=f"Unsupervised baseline: {unsup_name}...",
            )
            try:
                det = unsup_cls(n_components=config.pca_dims)
                det.fit(X_train)
                s = det.score(X_test)
                m = evaluate_detector(s, labels)
                unsup_baselines[unsup_name] = m["auroc"]
            except Exception:
                pass
            current += 1

        # Selected detector at different label budgets
        aurocs_by_detector: dict[str, list[float]] = {selected_display: []}
        budgets_used = []

        for n_labels in _LABEL_BUDGETS:
            if n_labels > len(X_train_jb_full):
                current += 1
                continue

            progress.progress(
                current / max(total_configs, 1),
                text=f"Label efficiency: {n_labels} labeled jailbreaks...",
            )

            budgets_used.append(n_labels)
            X_train_jb_subset = X_train_jb_full[:n_labels]

            try:
                detector = selected_info["factory"](config.pca_dims)
                X_combined = np.vstack([X_train, X_train_jb_subset])
                y_combined = np.concatenate([
                    np.zeros(len(X_train), dtype=np.int64),
                    np.ones(len(X_train_jb_subset), dtype=np.int64),
                ])
                detector.fit(X_combined, y_combined)
                scores = detector.score(X_test)
                m = evaluate_detector(scores, labels)
                aurocs_by_detector[selected_display].append(m["auroc"])
            except Exception:
                aurocs_by_detector[selected_display].append(0.5)

            current += 1

        if budgets_used:
            results["label_efficiency"] = {
                "n_labeled": budgets_used,
                "aurocs_by_detector": aurocs_by_detector,
                "unsupervised_baselines": unsup_baselines,
            }

    # --- Cross-source generalization ---
    if run_cross_source and is_supervised and has_labeled_jailbreaks:
        progress.progress(
            current / max(total_configs, 1),
            text="Cross-source generalization...",
        )

        X_train, X_val = _get_train_val(store, splits, base_layer)

        test_jb_indices = splits["test_jailbreak"].indices.tolist()
        train_jb_indices = splits["train_jailbreak"].indices.tolist()

        sources = ["Group A", "Group B", "Group C"]
        n_train_jb = len(train_jb_indices)
        n_test_jb = len(test_jb_indices)

        rng = np.random.default_rng(config.random_seed)
        n_groups = len(sources)
        train_groups = rng.integers(0, n_groups, size=n_train_jb)
        test_groups = rng.integers(0, n_groups, size=n_test_jb)

        X_train_jb = store.get_activations(base_layer, indices=train_jb_indices)
        X_test_jb = store.get_activations(base_layer, indices=test_jb_indices)
        X_test_benign = store.get_activations(
            base_layer, indices=splits["test_benign"].indices.tolist()
        )

        X_val_jb = store.get_activations(
            base_layer, indices=splits["val_jailbreak"].indices.tolist()
        )

        auroc_matrix = np.zeros((n_groups, n_groups))
        for train_src in range(n_groups):
            src_mask = train_groups == train_src
            if src_mask.sum() < 3:
                auroc_matrix[train_src, :] = 0.5
                current += n_groups
                continue
            X_train_jb_src = X_train_jb[src_mask]

            # Train fresh detector for this source
            try:
                ens = DetectorEnsemble()
                ens.add_detector(
                    selected_det_key,
                    selected_info["factory"](config.pca_dims),
                    weight=1.0,
                    detector_type=DetectorType.SUPERVISED,
                )
                ens.fit(
                    X_train, X_val,
                    X_train_jailbreak=X_train_jb_src,
                    X_val_jailbreak=X_val_jb,
                )
            except Exception:
                auroc_matrix[train_src, :] = 0.5
                current += n_groups
                continue

            for test_src in range(n_groups):
                test_mask = test_groups == test_src
                if test_mask.sum() < 3:
                    auroc_matrix[train_src, test_src] = 0.5
                    current += 1
                    continue

                X_test_src = np.vstack([X_test_benign, X_test_jb[test_mask]])
                labels_src = np.concatenate([
                    np.zeros(len(X_test_benign), dtype=np.int64),
                    np.ones(int(test_mask.sum()), dtype=np.int64),
                ])
                r = ens.score(X_test_src)
                m = evaluate_detector(r.ensemble_scores, labels_src)
                auroc_matrix[train_src, test_src] = m["auroc"]
                current += 1

        results["cross_source"] = {
            "sources": sources,
            "auroc_matrix": auroc_matrix.tolist(),
        }

    progress.progress(1.0, text="Ablations complete!")

    # Cache results for this detector
    cache = _get_ablation_cache()
    cache[selected_det_key] = results

    _show_results(selected_det_key, results)
