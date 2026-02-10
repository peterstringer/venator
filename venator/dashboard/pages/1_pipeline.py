"""Pipeline page — guided setup merging Data, Extract, Split, and Train stages."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.components.charts import source_distribution_chart
from venator.dashboard.state import PipelineState
from venator.data.prompts import (
    PromptDataset,
    collect_benign_prompts,
    collect_jailbreak_prompts,
)
from venator.data.splits import SplitManager
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

st.header("Pipeline")
st.markdown(
    "Guided setup: collect data, extract activations, create splits, and train detectors."
)

# Source label -> internal key mapping
_BENIGN_SOURCES = {
    "Alpaca (instruction-following)": "alpaca",
    "MMLU (academic questions)": "mmlu",
    "Custom diverse prompts": "diverse",
}
_JAILBREAK_SOURCES = {
    "JailbreakBench": "jailbreakbench",
    "AdvBench": "advbench",
    "Dan-style jailbreaks": "dan_style",
}

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


# ==================================================================
# Helpers
# ==================================================================


def _show_dataset_stats(dataset: PromptDataset, key_prefix: str) -> None:
    """Display stats, pie chart, and sample prompts for a loaded dataset."""
    st.metric("Total prompts", len(dataset))
    counts = dataset.source_counts()
    if counts:
        fig = source_distribution_chart(counts, title="Source Distribution")
        st.plotly_chart(fig, width="stretch")
    with st.expander("Sample prompts (5 random)"):
        rng = random.Random(42)
        indices = rng.sample(range(len(dataset)), min(5, len(dataset)))
        for idx in indices:
            src = dataset.sources[idx]
            st.markdown(f"**[{src}]** {dataset.prompts[idx][:200]}")


def _parse_uploaded_jsonl(
    uploaded_file: object,
    default_label: int,
) -> PromptDataset | None:
    """Parse an uploaded JSONL file into a PromptDataset."""
    try:
        content = uploaded_file.read().decode("utf-8")  # type: ignore[union-attr]
        prompts: list[str] = []
        labels: list[int] = []
        sources: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(record["prompt"])
            labels.append(int(record.get("label", default_label)))
            sources.append(record.get("source", "uploaded"))
        if not prompts:
            st.error("Uploaded file contains no prompts.")
            return None
        return PromptDataset(prompts, labels, sources)
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Failed to parse JSONL: {e}")
        return None


def _show_store_summary(store_path: Path) -> None:
    """Display summary metrics for an existing activation store."""
    store = ActivationStore(store_path)
    file_size_mb = os.path.getsize(store_path) / (1024 * 1024)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Prompts", store.n_prompts)
    with cols[1]:
        st.metric("Hidden dim", store.hidden_dim)
    with cols[2]:
        st.metric("Layers", ", ".join(str(l) for l in store.layers))
    with cols[3]:
        st.metric("File size", f"{file_size_mb:.1f} MB")


def _show_split_summary(
    splits: dict, store: ActivationStore, manager: SplitManager
) -> None:
    """Display the unified split summary with visual diagram."""
    n_benign = int((store.get_labels() == 0).sum())
    n_jailbreak = int((store.get_labels() == 1).sum())

    n_tb = splits["train_benign"].n_samples
    n_tj = splits["train_jailbreak"].n_samples
    n_vb = splits["val_benign"].n_samples
    n_vj = splits["val_jailbreak"].n_samples
    n_eb = splits["test_benign"].n_samples
    n_ej = splits["test_jailbreak"].n_samples

    # Visual split diagram
    st.markdown(f"""```
YOUR DATA: {n_benign} benign + {n_jailbreak} jailbreak prompts

TRAIN
 |- {n_tb:>4} benign      -> unsupervised detectors
 '- {n_tj:>4} jailbreak   -> supervised detectors

VALIDATE
 |- {n_vb:>4} benign      -> all detectors
 '- {n_vj:>4} jailbreak   -> all detectors

TEST (never seen in training)
 |- {n_eb:>4} benign
 '- {n_ej:>4} jailbreak
```""")

    # Methodology verification
    st.markdown("**Methodology Verification**")

    validation_passed = True
    try:
        manager.validate_splits(splits, store)
    except ValueError as e:
        validation_passed = False
        st.error(f"Validation failed: {e}")

    if validation_passed:
        st.markdown(
            f":white_check_mark: Training jailbreaks: **{n_tj}** "
            f"({n_tj / n_jailbreak * 100:.0f}% of jailbreaks)" if n_jailbreak > 0
            else ":white_check_mark: Training jailbreaks: **0**"
        )
        st.markdown(
            f":white_check_mark: Test jailbreaks (uncontaminated): **{n_ej}** "
            f"({n_ej / n_jailbreak * 100:.0f}% of jailbreaks)" if n_jailbreak > 0
            else ":white_check_mark: Test jailbreaks: **0**"
        )
        st.markdown(":white_check_mark: No index overlap between splits")

        # Sample-to-feature ratio check
        ratio = n_tb / config.pca_dims
        ratio_ok = ratio >= 5.0
        icon = ":white_check_mark:" if ratio_ok else ":warning:"
        quality = "healthy" if ratio_ok else "low"
        st.markdown(
            f"{icon} Sample-to-feature ratio: "
            f"**{n_tb}** samples / "
            f"**{config.pca_dims}** PCA dims = "
            f"**{ratio:.1f}x** ({quality})"
        )


# ==================================================================
# SECTION 1: Data Collection
# ==================================================================

_data_title = ":white_check_mark: 1. Data Collection" if state.prompts_ready else "1. Data Collection"

with st.expander(_data_title, expanded=not state.prompts_ready):
    if state.prompts_ready:
        st.success("Datasets loaded.")
        benign_ds: PromptDataset | None = None
        jailbreak_ds: PromptDataset | None = None
        if state.benign_path and state.benign_path.exists():
            benign_ds = PromptDataset.load(state.benign_path)
        if state.jailbreak_path and state.jailbreak_path.exists():
            jailbreak_ds = PromptDataset.load(state.jailbreak_path)

        info_parts = []
        if benign_ds:
            info_parts.append(f"**{len(benign_ds)}** benign")
        if jailbreak_ds:
            info_parts.append(f"**{len(jailbreak_ds)}** jailbreak")
        if info_parts:
            st.info(" + ".join(info_parts) + " prompts")

        if st.button("Re-collect data", key="recollect_data"):
            state.reset_from(1)
            st.rerun()
    else:
        st.markdown("Collect or upload benign and jailbreak prompt datasets.")

        col_benign, col_jailbreak = st.columns(2)

        # --- Benign Prompts ---
        with col_benign:
            st.markdown("**Benign Prompts**")
            st.caption("Training distribution — used for train, validation, and test.")

            with st.form("collect_benign_form"):
                n_benign = st.slider(
                    "Number of prompts",
                    min_value=100,
                    max_value=1000,
                    value=500,
                    step=50,
                    key="benign_n_slider",
                )
                selected_benign_labels = st.multiselect(
                    "Sources to include",
                    options=list(_BENIGN_SOURCES.keys()),
                    default=list(_BENIGN_SOURCES.keys()),
                    key="benign_sources_select",
                )
                collect_benign_clicked = st.form_submit_button("Collect from datasets")

            if collect_benign_clicked:
                if not selected_benign_labels:
                    st.error("Select at least one source.")
                else:
                    allowed_keys = {_BENIGN_SOURCES[lbl] for lbl in selected_benign_labels}
                    with st.spinner("Collecting benign prompts..."):
                        raw = collect_benign_prompts(n=n_benign, seed=config.random_seed)
                        filtered = [(p, s) for p, s in raw if s in allowed_keys]
                        if not filtered:
                            st.error("No prompts collected. Check source availability.")
                        else:
                            prompts_list = [p for p, _ in filtered]
                            sources_list = [s for _, s in filtered]
                            ds = PromptDataset(prompts_list, [0] * len(prompts_list), sources_list)
                            st.session_state["_benign_dataset"] = ds
                            counts = ds.source_counts()
                            missing = allowed_keys - set(counts.keys())
                            st.success(f"Collected {len(ds)} benign prompts.")
                            if missing:
                                st.warning(
                                    f"Some sources returned no data: {', '.join(sorted(missing))}."
                                )

            uploaded_benign = st.file_uploader(
                "Or upload JSONL",
                type=["jsonl"],
                key="benign_upload",
                help='One JSON object per line: {"prompt": "...", "label": 0, "source": "..."}',
            )
            if uploaded_benign is not None:
                ds = _parse_uploaded_jsonl(uploaded_benign, default_label=0)
                if ds is not None:
                    st.session_state["_benign_dataset"] = ds
                    st.success(f"Loaded {len(ds)} prompts from upload.")

            loaded_benign: PromptDataset | None = st.session_state.get("_benign_dataset")
            if loaded_benign is not None:
                _show_dataset_stats(loaded_benign, "benign")

        # --- Jailbreak Prompts ---
        with col_jailbreak:
            st.markdown("**Jailbreak Prompts**")
            st.caption("Test only — never used in unsupervised training or validation.")

            with st.form("collect_jailbreak_form"):
                n_jailbreak = st.slider(
                    "Number of prompts",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    key="jailbreak_n_slider",
                )
                selected_jailbreak_labels = st.multiselect(
                    "Sources to include",
                    options=list(_JAILBREAK_SOURCES.keys()),
                    default=list(_JAILBREAK_SOURCES.keys()),
                    key="jailbreak_sources_select",
                )
                collect_jailbreak_clicked = st.form_submit_button("Collect from datasets")

            if collect_jailbreak_clicked:
                if not selected_jailbreak_labels:
                    st.error("Select at least one source.")
                else:
                    allowed_keys = {_JAILBREAK_SOURCES[lbl] for lbl in selected_jailbreak_labels}
                    with st.spinner("Collecting jailbreak prompts..."):
                        raw = collect_jailbreak_prompts(n=n_jailbreak, seed=config.random_seed)
                        filtered = [(p, s) for p, s in raw if s in allowed_keys]
                        if not filtered:
                            st.error("No prompts collected. Check source availability.")
                        else:
                            prompts_list = [p for p, _ in filtered]
                            sources_list = [s for _, s in filtered]
                            ds = PromptDataset(prompts_list, [1] * len(prompts_list), sources_list)
                            st.session_state["_jailbreak_dataset"] = ds
                            counts = ds.source_counts()
                            missing = allowed_keys - set(counts.keys())
                            if len(ds) < n_jailbreak:
                                st.warning(
                                    f"Collected **{len(ds)}** of {n_jailbreak} requested."
                                )
                            else:
                                st.success(f"Collected {len(ds)} jailbreak prompts.")
                            if missing:
                                st.warning(
                                    f"Sources returned no data: {', '.join(sorted(missing))}."
                                )

            uploaded_jailbreak = st.file_uploader(
                "Or upload JSONL",
                type=["jsonl"],
                key="jailbreak_upload",
                help='One JSON object per line: {"prompt": "...", "label": 1, "source": "..."}',
            )
            if uploaded_jailbreak is not None:
                ds = _parse_uploaded_jsonl(uploaded_jailbreak, default_label=1)
                if ds is not None:
                    st.session_state["_jailbreak_dataset"] = ds
                    st.success(f"Loaded {len(ds)} prompts from upload.")

            loaded_jailbreak: PromptDataset | None = st.session_state.get("_jailbreak_dataset")
            if loaded_jailbreak is not None:
                _show_dataset_stats(loaded_jailbreak, "jailbreak")

        # Save & mark complete
        both_loaded = (
            st.session_state.get("_benign_dataset") is not None
            and st.session_state.get("_jailbreak_dataset") is not None
        )

        if st.button("Save Datasets", disabled=not both_loaded, type="primary", key="save_data"):
            b_ds = st.session_state["_benign_dataset"]
            j_ds = st.session_state["_jailbreak_dataset"]

            benign_path = config.prompts_dir / "benign.jsonl"
            jailbreak_path = config.prompts_dir / "jailbreaks.jsonl"

            b_ds.save(benign_path)
            j_ds.save(jailbreak_path)

            state.reset_from(1)
            state.benign_path = benign_path
            state.jailbreak_path = jailbreak_path
            state.prompts_ready = True

            st.success(
                f"Saved {len(b_ds)} benign + {len(j_ds)} jailbreak prompts."
            )
            st.rerun()

        if not both_loaded:
            st.caption("Collect or upload both datasets to continue.")


# ==================================================================
# SECTION 2: Extract Activations
# ==================================================================

_extract_title = ":white_check_mark: 2. Extract Activations" if state.activations_ready else "2. Extract Activations"

with st.expander(_extract_title, expanded=state.prompts_ready and not state.activations_ready):
    if not state.prompts_ready:
        st.info("Complete data collection first.")
    elif state.activations_ready and state.store_path and state.store_path.exists():
        st.success("Activations extracted.")
        _show_store_summary(state.store_path)

        if st.button("Re-extract", key="re_extract"):
            state.reset_from(2)
            st.rerun()
    else:
        st.markdown(
            "Extract hidden state activations from an LLM for each prompt. "
            "This is the slowest step (~1-3 seconds per prompt)."
        )

        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            model_id = st.text_input("Model ID", value=config.model_id, key="extract_model_id")
            output_path_str = st.text_input(
                "Output path",
                value=str(config.activations_dir / "all.h5"),
                key="extract_output_path",
            )
        with cfg_col2:
            available_layers = list(range(0, 32))
            selected_layers = st.multiselect(
                "Layers to extract",
                options=available_layers,
                default=config.extraction_layers,
                key="extract_layers",
            )

        # Estimate
        benign_ds = PromptDataset.load(state.benign_path) if state.benign_path else None
        jailbreak_ds = PromptDataset.load(state.jailbreak_path) if state.jailbreak_path else None

        if benign_ds and jailbreak_ds:
            n_total = len(benign_ds) + len(jailbreak_ds)
            st.info(
                f"**{n_total} prompts** to extract "
                f"({len(benign_ds)} benign + {len(jailbreak_ds)} jailbreak)."
            )
        else:
            n_total = 0
            st.warning("Could not load prompt datasets.")

        can_start = bool(selected_layers) and n_total > 0

        if st.button("Start Extraction", disabled=not can_start, type="primary", key="start_extract"):
            output_path = Path(output_path_str)

            if output_path.exists():
                output_path.unlink()

            combined = PromptDataset.merge(benign_ds, jailbreak_ds)  # type: ignore[arg-type]

            with st.status("Extracting activations...", expanded=True) as status:
                from venator.activation.extractor import ActivationExtractor

                st.write("Loading model... (this may take a minute on first run)")
                extractor = ActivationExtractor(
                    model_id=model_id,
                    layers=selected_layers,
                    config=config,
                )

                hidden_dim = extractor.hidden_dim
                layers = sorted(extractor._target_layers)
                st.write(f"Model loaded. Hidden dim: {hidden_dim}, layers: {layers}")

                output_path.parent.mkdir(parents=True, exist_ok=True)
                store = ActivationStore.create(
                    output_path,
                    model_id=model_id,
                    layers=layers,
                    hidden_dim=hidden_dim,
                )

                progress_bar = st.progress(0)
                prompt_text = st.empty()
                stats_text = st.empty()

                t0 = time.perf_counter()

                for i in range(len(combined)):
                    prompt = combined.prompts[i]
                    label = combined.labels[i]

                    prompt_text.text(f"Extracting {i + 1}/{n_total}: {prompt[:80]}...")

                    activations = extractor.extract_single(prompt)
                    store.append(prompt, activations, label=label)

                    progress_bar.progress((i + 1) / n_total)

                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = (n_total - i - 1) / rate if rate > 0 else 0
                    stats_text.text(
                        f"Elapsed: {elapsed:.0f}s | "
                        f"{rate:.2f} prompts/s | "
                        f"ETA: {remaining:.0f}s"
                    )

                elapsed = time.perf_counter() - t0
                status.update(label="Extraction complete!", state="complete", expanded=True)

            state.reset_from(2)
            state.store_path = output_path
            state.activations_ready = True

            st.success(
                f"Extracted {n_total} prompts in {elapsed:.1f}s "
                f"({elapsed / n_total:.2f}s/prompt)."
            )
            _show_store_summary(output_path)
            st.rerun()


# ==================================================================
# SECTION 3: Split Data
# ==================================================================

_split_title = ":white_check_mark: 3. Split Data" if state.splits_ready else "3. Split Data"

with st.expander(_split_title, expanded=state.activations_ready and not state.splits_ready):
    if not state.activations_ready:
        st.info("Complete extraction first.")
    elif state.splits_ready and state.splits_path and state.splits_path.exists():
        st.success("Splits created.")

        store = ActivationStore(state.store_path)
        splits = SplitManager.load_splits(state.splits_path)
        manager = SplitManager(seed=config.random_seed)
        _show_split_summary(splits, store, manager)

        if st.button("Re-split", key="re_split"):
            state.reset_from(3)
            st.rerun()
    else:
        st.markdown(
            "Create train/validation/test splits. One split serves all detector types — "
            "the pipeline routes data automatically."
        )

        store = ActivationStore(state.store_path)
        n_jailbreak_total = int((store.get_labels() == 1).sum())

        # Jailbreak allocation (shown only when jailbreaks exist)
        if n_jailbreak_total > 0:
            st.markdown("**Jailbreak allocation**")
            jb_col1, jb_col2 = st.columns(2)
            with jb_col1:
                jailbreak_train_frac = st.slider(
                    "Jailbreak fraction (training)",
                    min_value=0.05, max_value=0.30, value=0.15, step=0.05,
                    key="jb_train_frac",
                    help="Small labeled subset for supervised detectors (e.g. linear probe).",
                )
            with jb_col2:
                jailbreak_val_frac = st.slider(
                    "Jailbreak fraction (validation)",
                    min_value=0.05, max_value=0.30, value=0.15, step=0.05,
                    key="jb_val_frac",
                    help="Used for threshold calibration with labeled data.",
                )

            jailbreak_test_frac = round(1.0 - jailbreak_train_frac - jailbreak_val_frac, 2)
            if jailbreak_test_frac < 0.40:
                st.warning(
                    f"Only {jailbreak_test_frac:.0%} of jailbreaks reserved for testing."
                )

            n_train_jb = int(n_jailbreak_total * jailbreak_train_frac)
            n_val_jb = int(n_jailbreak_total * jailbreak_val_frac)
            n_test_jb = n_jailbreak_total - n_train_jb - n_val_jb
            st.info(
                f"**{n_train_jb}** jailbreaks for training, "
                f"**{n_val_jb}** for validation, "
                f"**{n_test_jb}** reserved for uncontaminated testing."
            )
        else:
            jailbreak_train_frac = 0.15
            jailbreak_val_frac = 0.15

        # Benign split fractions
        st.markdown("**Benign allocation**")
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            train_frac = st.slider(
                "Train fraction (benign)",
                min_value=0.50, max_value=0.80, value=0.70, step=0.05,
                key="benign_train_frac",
            )
        with cfg_col2:
            val_frac = st.slider(
                "Validation fraction (benign)",
                min_value=0.05, max_value=0.25, value=0.15, step=0.05,
                key="benign_val_frac",
            )
        with cfg_col3:
            test_frac = round(1.0 - train_frac - val_frac, 2)
            st.metric("Test fraction (benign)", f"{test_frac:.0%}")

        seed = st.number_input("Random seed", min_value=0, value=config.random_seed, step=1, key="split_seed")

        if test_frac <= 0:
            st.error("Train + validation fractions must be less than 1.0.")
            st.stop()

        if st.button("Create Splits", type="primary", key="create_splits"):
            manager = SplitManager(seed=int(seed))

            with st.spinner("Creating splits..."):
                splits = manager.create_splits(
                    store,
                    benign_train_frac=train_frac,
                    benign_val_frac=val_frac,
                    jailbreak_train_frac=jailbreak_train_frac,
                    jailbreak_val_frac=jailbreak_val_frac,
                )

            _show_split_summary(splits, store, manager)

            st.session_state["_splits"] = splits
            st.session_state["_split_manager"] = manager

        if st.session_state.get("_splits") is not None:
            if st.button("Save Splits", type="primary", key="save_splits"):
                splits = st.session_state["_splits"]
                manager = st.session_state["_split_manager"]

                splits_path = config.data_dir / "splits.json"
                manager.save_splits(splits, splits_path)

                state.reset_from(3)
                state.splits_path = splits_path
                state.splits_ready = True

                st.success(f"Saved splits to `{splits_path}`.")
                st.rerun()


# ==================================================================
# SECTION 4: Train Detectors
# ==================================================================

_train_title = ":white_check_mark: 4. Train Detectors" if state.model_ready else "4. Train Detectors"

with st.expander(_train_title, expanded=state.splits_ready and not state.model_ready):
    if not state.splits_ready:
        st.info("Complete splitting first.")
    elif state.model_ready and state.model_path:
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
            primary = state.train_metrics.get("primary_detector")
            if primary:
                st.info(f"Primary detector: **{primary}**")

        if st.button("Re-train", key="re_train"):
            state.reset_from(4)
            st.rerun()
    else:
        st.markdown(
            "Train anomaly detection ensemble. Supports unsupervised, supervised, and hybrid."
        )

        splits = SplitManager.load_splits(state.splits_path)
        store = ActivationStore(state.store_path)
        available_layers = store.layers

        # Determine if labeled jailbreaks are available for supervised training
        has_labeled_jailbreaks = splits["train_jailbreak"].n_samples > 0

        # Ensemble type selector
        if has_labeled_jailbreaks:
            ensemble_type = st.radio(
                "Ensemble Type",
                options=["unsupervised", "supervised", "hybrid"],
                format_func=lambda x: x.capitalize(),
                horizontal=True,
                index=2,
                key="ensemble_type_radio",
                help=(
                    "**Unsupervised**: PCA+Mahalanobis, Isolation Forest, Autoencoder (benign-only). "
                    "**Supervised**: Linear Probe, Contrastive Direction, Contrastive Mahalanobis. "
                    "**Hybrid**: Linear Probe, Contrastive Direction + Autoencoder."
                ),
            )
        else:
            ensemble_type = "unsupervised"
            st.info("Using **unsupervised** ensemble (no labeled jailbreaks in training split).")

        st.markdown("**Detectors in this ensemble:**")
        for det_name, det_type, det_weight in _ENSEMBLE_DETECTORS[ensemble_type]:
            badge = ":green[supervised]" if det_type == "sup" else ":blue[unsupervised]"
            st.markdown(f"- {det_name} ({badge}, weight={det_weight})")

        # Layer and PCA controls
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            default_layer = available_layers[len(available_layers) // 2]
            layer = st.selectbox(
                "Layer for detection",
                options=available_layers,
                index=available_layers.index(default_layer),
                key="train_layer",
            )
            pca_dims = st.slider(
                "PCA dimensions",
                min_value=10, max_value=100, value=config.pca_dims, step=5,
                key="train_pca_dims",
            )

        with cfg_col2:
            threshold_pctile = st.slider(
                "Threshold percentile",
                min_value=90.0, max_value=99.0,
                value=config.anomaly_threshold_percentile, step=0.5,
                key="train_threshold",
            )

        # Manual detector selection (unsupervised only)
        if ensemble_type == "unsupervised":
            det_col1, det_col2, det_col3 = st.columns(3)
            with det_col1:
                use_pca_maha = st.checkbox("PCA + Mahalanobis", value=True, key="use_pca_maha")
                weight_pca_maha = st.number_input(
                    "Weight", value=config.weight_pca_mahalanobis, min_value=0.1, step=0.5,
                    key="w_pca",
                ) if use_pca_maha else 0.0
            with det_col2:
                use_iforest = st.checkbox("Isolation Forest", value=True, key="use_iforest")
                weight_iforest = st.number_input(
                    "Weight", value=config.weight_isolation_forest, min_value=0.1, step=0.5,
                    key="w_if",
                ) if use_iforest else 0.0
            with det_col3:
                use_autoencoder = st.checkbox("Autoencoder", value=True, key="use_ae")
                weight_ae = st.number_input(
                    "Weight", value=config.weight_autoencoder, min_value=0.1, step=0.5,
                    key="w_ae",
                ) if use_autoencoder else 0.0

            n_detectors = sum([use_pca_maha, use_iforest, use_autoencoder])
            if n_detectors == 0:
                st.error("Select at least one detector.")
                st.stop()

        if st.button("Train Ensemble", type="primary", key="train_ensemble"):
            # Get training data (unified 6-key format)
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
            else:
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
                status.update(label="Training complete!", state="complete")

            # Select primary detector
            from sklearn.metrics import roc_curve as _roc_curve  # type: ignore[import-untyped]

            _priority = ["linear_probe", "pca_mahalanobis", "contrastive_mahalanobis",
                         "isolation_forest", "autoencoder", "contrastive_direction"]
            det_names_in_ens = {n for n, _, _ in ensemble.detectors}
            primary_name = None
            primary_det = None
            for candidate in _priority:
                if candidate in det_names_in_ens:
                    primary_name = candidate
                    for n, d, _ in ensemble.detectors:
                        if n == candidate:
                            primary_det = d
                            break
                    break
            if primary_name is None and ensemble.detectors:
                primary_name = ensemble.detectors[0][0]
                primary_det = ensemble.detectors[0][1]

            # Calibrate primary threshold
            primary_val_scores = primary_det.score(X_val)
            if X_val_jailbreak is not None and len(X_val_jailbreak) > 0:
                primary_jb_scores = primary_det.score(X_val_jailbreak)
                all_scores = np.concatenate([primary_val_scores, primary_jb_scores])
                all_labels = np.concatenate([
                    np.zeros(len(primary_val_scores), dtype=np.int64),
                    np.ones(len(primary_jb_scores), dtype=np.int64),
                ])
                fpr_arr, tpr_arr, thresholds_arr = _roc_curve(all_labels, all_scores)
                j_stat = tpr_arr - fpr_arr
                best_idx = int(np.argmax(j_stat))
                primary_threshold = float(np.nextafter(thresholds_arr[best_idx], -np.inf))
            else:
                primary_threshold = float(np.percentile(primary_val_scores, threshold_pctile))

            val_fpr = float(np.mean(primary_val_scores > primary_threshold))

            # Save model
            output_dir = config.models_dir / f"detector_{ensemble_type}_v1"
            ensemble.save(output_dir)

            meta = {
                "layer": int(layer),
                "model_id": store.model_id,
                "extraction_layers": [int(l) for l in store.layers],
                "ensemble_type": ensemble_type,
                "primary_name": primary_name,
                "primary_threshold": primary_threshold,
            }
            with open(output_dir / "pipeline_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # Update state
            train_metrics = {
                "val_false_positive_rate": val_fpr,
                "threshold": primary_threshold,
                "ensemble_threshold": float(ensemble.threshold_),
                "layer": int(layer),
                "pca_dims": int(pca_dims),
                "n_detectors": int(n_detectors),
                "ensemble_type": ensemble_type,
                "primary_detector": primary_name,
                "training_time_s": round(elapsed, 1),
            }

            state.reset_from(4)
            state.model_path = output_dir
            state.train_metrics = train_metrics
            state.model_ready = True

            st.success(
                f"Trained {ensemble_type} ensemble in {elapsed:.1f}s. "
                f"Primary detector: **{primary_name}**"
            )

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Validation FPR", f"{val_fpr:.4f} ({val_fpr * 100:.1f}%)")
            with res_col2:
                st.metric("Primary Threshold", f"{primary_threshold:.4f}")
            with res_col3:
                st.metric("Training time", f"{elapsed:.1f}s")
