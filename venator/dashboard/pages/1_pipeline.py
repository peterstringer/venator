"""Pipeline page — guided setup merging Data, Extract, Split, and Train stages."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
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
)
from venator.detection.contrastive import (
    ContrastiveDirectionDetector,
    ContrastiveMahalanobisDetector,
)
from venator.detection.isolation_forest import IsolationForestDetector
from venator.detection.linear_probe import LinearProbeDetector, MLPProbeDetector
from venator.detection.metrics import evaluate_detector
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

# Detector registry: (internal_name, display_name, type, default_weight, factory)
# factory is a callable(pca_dims) -> detector instance
_DETECTOR_REGISTRY = {
    # Supervised
    "linear_probe": {
        "display": "Linear Probe",
        "type": "supervised",
        "default_weight": 2.5,
        "factory": lambda pca, **kw: LinearProbeDetector(
            n_components=pca, C=kw.get("C", 1.0),
        ),
    },
    "contrastive_mahalanobis": {
        "display": "Contrastive Mahalanobis",
        "type": "supervised",
        "default_weight": 1.5,
        "factory": lambda pca, **kw: ContrastiveMahalanobisDetector(
            n_components=pca, regularization=kw.get("regularization", 1e-5),
        ),
    },
    "contrastive_direction": {
        "display": "Contrastive Direction",
        "type": "supervised",
        "default_weight": 2.0,
        "factory": lambda pca, **kw: ContrastiveDirectionDetector(),
    },
    "mlp_probe": {
        "display": "MLP Probe",
        "type": "supervised",
        "default_weight": 2.0,
        "factory": lambda pca, **kw: MLPProbeDetector(
            n_components=pca,
            hidden1=kw.get("hidden1", 128),
            hidden2=kw.get("hidden2", 32),
            epochs=kw.get("epochs", 200),
        ),
    },
    # Unsupervised
    "autoencoder": {
        "display": "Autoencoder",
        "type": "unsupervised",
        "default_weight": 1.0,
        "factory": lambda pca, **kw: AutoencoderDetector(
            n_components=pca,
            hidden_dim=kw.get("hidden_dim", 64),
            epochs=kw.get("epochs", 200),
        ),
    },
    "pca_mahalanobis": {
        "display": "PCA + Mahalanobis",
        "type": "unsupervised",
        "default_weight": 2.0,
        "factory": lambda pca, **kw: PCAMahalanobisDetector(
            n_components=pca, regularization=kw.get("regularization", 1e-6),
        ),
    },
    "isolation_forest": {
        "display": "Isolation Forest",
        "type": "unsupervised",
        "default_weight": 1.5,
        "factory": lambda pca, **kw: IsolationForestDetector(
            n_components=pca, n_estimators=kw.get("n_estimators", 200),
        ),
    },
}

# Quick-select presets: name -> list of detector keys
_QUICK_PRESETS = {
    "Linear Probe Only": ["linear_probe"],
    "All Supervised": ["linear_probe", "contrastive_mahalanobis", "contrastive_direction", "mlp_probe"],
    "All Unsupervised": ["pca_mahalanobis", "isolation_forest", "autoencoder"],
    "Everything": list(_DETECTOR_REGISTRY.keys()),
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


def _select_primary_detector(
    ensemble: DetectorEnsemble,
) -> tuple[str, object]:
    """Select the primary detector from an ensemble by priority."""
    _priority = [
        "linear_probe", "pca_mahalanobis", "contrastive_mahalanobis",
        "isolation_forest", "autoencoder", "contrastive_direction",
        "mlp_probe",
    ]
    det_names_in_ens = {n for n, _, _ in ensemble.detectors}
    for candidate in _priority:
        if candidate in det_names_in_ens:
            for n, d, _ in ensemble.detectors:
                if n == candidate:
                    return candidate, d
    if ensemble.detectors:
        return ensemble.detectors[0][0], ensemble.detectors[0][1]
    raise RuntimeError("No detectors in ensemble")


def _calibrate_primary_threshold(
    primary_det: object,
    X_val: np.ndarray,
    X_val_jailbreak: np.ndarray | None,
    threshold_pctile: float,
) -> tuple[float, float]:
    """Calibrate threshold for primary detector. Returns (threshold, val_fpr)."""
    from sklearn.metrics import roc_curve as _roc_curve  # type: ignore[import-untyped]

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
        threshold = float(np.nextafter(thresholds_arr[best_idx], -np.inf))
    else:
        threshold = float(np.percentile(primary_val_scores, threshold_pctile))
    val_fpr = float(np.mean(primary_val_scores > threshold))
    return threshold, val_fpr


def _compute_detector_val_auroc(
    det: object,
    X_val: np.ndarray,
    X_val_jailbreak: np.ndarray | None,
) -> float | None:
    """Compute AUROC for a single detector on validation data."""
    if X_val_jailbreak is None or len(X_val_jailbreak) == 0:
        return None
    scores_b = det.score(X_val)
    scores_j = det.score(X_val_jailbreak)
    all_scores = np.concatenate([scores_b, scores_j])
    all_labels = np.concatenate([
        np.zeros(len(scores_b), dtype=np.int64),
        np.ones(len(scores_j), dtype=np.int64),
    ])
    from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
    try:
        return float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        return None


_train_title = ":white_check_mark: 4. Train Detectors" if state.model_ready else "4. Train Detectors"

with st.expander(_train_title, expanded=state.splits_ready and not state.model_ready):
    if not state.splits_ready:
        st.info("Complete splitting first.")
    elif state.model_ready and state.model_path:
        # --- Completed state: show training summary banner ---
        trained = state.trained_detectors or []
        if trained:
            st.success("Training Complete")
            for det_info in trained:
                auroc_str = f"Val AUROC: {det_info['val_auroc']:.3f}" if det_info.get("val_auroc") is not None else ""
                primary_tag = " **[PRIMARY]**" if det_info.get("is_primary") else ""
                st.markdown(
                    f"- **{det_info['display']}**{primary_tag} — "
                    f"Trained in {det_info['time_s']:.1f}s"
                    + (f" ({auroc_str})" if auroc_str else "")
                )
        else:
            st.success("Model trained.")

        if state.train_metrics:
            m = state.train_metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                val_fpr = m.get("val_false_positive_rate")
                if val_fpr is not None:
                    st.metric("Validation FPR", f"{val_fpr:.4f} ({val_fpr * 100:.1f}%)")
            with col2:
                primary = m.get("primary_detector")
                if primary:
                    st.metric("Primary Detector", primary)
            with col3:
                st.metric("Training Time", f"{m.get('training_time_s', 0):.1f}s")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Re-train", key="re_train"):
                state.reset_from(4)
                st.rerun()
        with btn_col2:
            if st.button("View Results", type="primary", key="goto_results"):
                st.switch_page("pages/2_results.py")
    else:
        # --- Training configuration ---
        st.markdown("Select detectors, configure, and train.")

        splits = SplitManager.load_splits(state.splits_path)
        store = ActivationStore(state.store_path)
        available_layers = store.layers
        has_labeled_jailbreaks = splits["train_jailbreak"].n_samples > 0

        # ---- SECTION 1: Detector Selection ----
        st.markdown("**Detector Selection**")

        # Quick-select buttons
        preset_cols = st.columns(len(_QUICK_PRESETS))
        for i, (preset_name, preset_keys) in enumerate(_QUICK_PRESETS.items()):
            with preset_cols[i]:
                # Filter supervised detectors if no jailbreaks
                valid_keys = [
                    k for k in preset_keys
                    if has_labeled_jailbreaks or _DETECTOR_REGISTRY[k]["type"] == "unsupervised"
                ]
                if st.button(preset_name, key=f"preset_{i}", disabled=len(valid_keys) == 0):
                    for k in _DETECTOR_REGISTRY:
                        st.session_state[f"sel_{k}"] = k in valid_keys
                    st.rerun()

        # Supervised group
        if has_labeled_jailbreaks:
            st.markdown(":green[**SUPERVISED**]")
            sup_cols = st.columns(4)
            sup_detectors = [
                (k, v) for k, v in _DETECTOR_REGISTRY.items()
                if v["type"] == "supervised"
            ]
            for i, (key, info) in enumerate(sup_detectors):
                with sup_cols[i % 4]:
                    default = key == "linear_probe"
                    st.checkbox(
                        info["display"], value=default, key=f"sel_{key}",
                    )
        else:
            st.info("No labeled jailbreaks — supervised detectors unavailable.")
            for key, info in _DETECTOR_REGISTRY.items():
                if info["type"] == "supervised":
                    st.session_state[f"sel_{key}"] = False

        # Unsupervised group
        st.markdown(":blue[**UNSUPERVISED**]")
        unsup_cols = st.columns(3)
        unsup_detectors = [
            (k, v) for k, v in _DETECTOR_REGISTRY.items()
            if v["type"] == "unsupervised"
        ]
        for i, (key, info) in enumerate(unsup_detectors):
            with unsup_cols[i % 3]:
                default = key == "autoencoder"
                st.checkbox(
                    info["display"], value=default, key=f"sel_{key}",
                )

        # Ensemble option
        st.markdown("**ENSEMBLE**")
        use_ensemble = st.checkbox(
            "Custom Ensemble (combine selected detectors)",
            value=False, key="sel_ensemble",
            help="Trains all selected detectors individually AND creates a weighted ensemble.",
        )

        # Collect selected detectors
        selected_keys = [
            k for k in _DETECTOR_REGISTRY
            if st.session_state.get(f"sel_{k}", False)
        ]

        if not selected_keys:
            st.warning("Select at least one detector.")
            st.stop()

        # ---- Ensemble weight sliders (only if ensemble checked) ----
        det_weights: dict[str, float] = {}
        if use_ensemble and len(selected_keys) > 1:
            st.markdown("**Ensemble Weights**")
            w_cols = st.columns(min(len(selected_keys), 4))
            for i, key in enumerate(selected_keys):
                info = _DETECTOR_REGISTRY[key]
                with w_cols[i % min(len(selected_keys), 4)]:
                    det_weights[key] = st.number_input(
                        info["display"],
                        value=info["default_weight"],
                        min_value=0.1, step=0.5,
                        key=f"w_{key}",
                    )
        else:
            for key in selected_keys:
                det_weights[key] = _DETECTOR_REGISTRY[key]["default_weight"]

        # ---- SECTION 2: Configuration ----
        st.markdown("---")
        st.markdown("**Configuration**")

        # Shared config: layer + PCA
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            default_layer = available_layers[len(available_layers) // 2]
            layer = st.selectbox(
                "Layer for detection",
                options=available_layers,
                index=available_layers.index(default_layer),
                key="train_layer",
            )
        with cfg_col2:
            pca_dims = st.slider(
                "PCA dimensions",
                min_value=10, max_value=200, value=config.pca_dims, step=5,
                key="train_pca_dims",
            )
        with cfg_col3:
            threshold_pctile = st.slider(
                "Threshold percentile",
                min_value=90.0, max_value=99.0,
                value=config.anomaly_threshold_percentile, step=0.5,
                key="train_threshold",
            )

        # Per-detector config (in expanders)
        det_kwargs: dict[str, dict] = {k: {} for k in selected_keys}

        has_configurable = any(
            k in ("linear_probe", "autoencoder", "pca_mahalanobis",
                   "isolation_forest", "contrastive_mahalanobis", "mlp_probe")
            for k in selected_keys
        )
        if has_configurable:
            with st.expander("Per-detector settings", expanded=False):
                for key in selected_keys:
                    if key == "linear_probe":
                        c1, c2 = st.columns(2)
                        with c1:
                            det_kwargs[key]["C"] = st.number_input(
                                "Linear Probe: Regularization C",
                                value=1.0, min_value=0.01, step=0.1,
                                key="cfg_lp_C",
                            )
                    elif key == "autoencoder":
                        c1, c2 = st.columns(2)
                        with c1:
                            det_kwargs[key]["epochs"] = st.slider(
                                "Autoencoder: Epochs",
                                min_value=50, max_value=500, value=200, step=50,
                                key="cfg_ae_epochs",
                            )
                        with c2:
                            det_kwargs[key]["hidden_dim"] = st.number_input(
                                "Autoencoder: Hidden dim",
                                value=64, min_value=16, step=16,
                                key="cfg_ae_hidden",
                            )
                    elif key == "mlp_probe":
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            det_kwargs[key]["hidden1"] = st.number_input(
                                "MLP: Hidden 1", value=128, min_value=16, step=16,
                                key="cfg_mlp_h1",
                            )
                        with c2:
                            det_kwargs[key]["hidden2"] = st.number_input(
                                "MLP: Hidden 2", value=32, min_value=8, step=8,
                                key="cfg_mlp_h2",
                            )
                        with c3:
                            det_kwargs[key]["epochs"] = st.slider(
                                "MLP: Epochs",
                                min_value=50, max_value=500, value=200, step=50,
                                key="cfg_mlp_epochs",
                            )
                    elif key == "pca_mahalanobis":
                        det_kwargs[key]["regularization"] = st.number_input(
                            "PCA+Mahalanobis: Regularization",
                            value=1e-6, min_value=0.0, step=1e-6, format="%.1e",
                            key="cfg_pcam_reg",
                        )
                    elif key == "isolation_forest":
                        det_kwargs[key]["n_estimators"] = st.number_input(
                            "Isolation Forest: N estimators",
                            value=200, min_value=50, step=50,
                            key="cfg_if_est",
                        )
                    elif key == "contrastive_mahalanobis":
                        det_kwargs[key]["regularization"] = st.number_input(
                            "Contrastive Mahalanobis: Regularization",
                            value=1e-5, min_value=0.0, step=1e-5, format="%.1e",
                            key="cfg_cm_reg",
                        )

        # ---- SECTION 3: Auto-Optimize ----
        st.markdown("---")
        auto_optimize = st.checkbox(
            "Auto-optimize hyperparameters",
            value=False, key="auto_optimize",
            help="Grid search over PCA dims and layers on validation data.",
        )

        if auto_optimize:
            auto_pca_grid = [10, 20, 30, 50, 75, 100]
            auto_layer_grid = available_layers
            n_configs = len(auto_pca_grid) * len(auto_layer_grid) * len(selected_keys)
            st.info(
                f"Will search {len(auto_pca_grid)} PCA dims x "
                f"{len(auto_layer_grid)} layers x "
                f"{len(selected_keys)} detectors = "
                f"**{n_configs} configurations**."
            )

        # ---- SECTION 4: Training Execution ----
        st.markdown("---")

        if st.button("Train Selected Detectors", type="primary", key="train_detectors"):
            # Load training data
            X_train = store.get_activations(layer, indices=splits["train_benign"].indices.tolist())
            X_val = store.get_activations(layer, indices=splits["val_benign"].indices.tolist())
            if has_labeled_jailbreaks:
                X_train_jailbreak = store.get_activations(
                    layer, indices=splits["train_jailbreak"].indices.tolist()
                )
                X_val_jailbreak = store.get_activations(
                    layer, indices=splits["val_jailbreak"].indices.tolist()
                )
            else:
                X_train_jailbreak = None
                X_val_jailbreak = None

            # Build ensemble with all selected detectors
            ensemble = DetectorEnsemble(threshold_percentile=threshold_pctile)
            for key in selected_keys:
                info = _DETECTOR_REGISTRY[key]
                det_type = (
                    DetectorType.SUPERVISED
                    if info["type"] == "supervised"
                    else DetectorType.UNSUPERVISED
                )
                detector = info["factory"](pca_dims, **det_kwargs[key])
                ensemble.add_detector(
                    key, detector, weight=det_weights[key],
                    detector_type=det_type,
                )

            # --- Auto-optimize path ---
            if auto_optimize:
                auto_pca_grid = [10, 20, 30, 50, 75, 100]
                auto_layer_grid = available_layers
                best_configs: dict[str, dict] = {}
                n_total = len(auto_pca_grid) * len(auto_layer_grid) * len(selected_keys)

                with st.status("Auto-optimizing...", expanded=True) as opt_status:
                    progress = st.progress(0)
                    status_text = st.empty()
                    results_area = st.empty()
                    step = 0

                    for key in selected_keys:
                        info = _DETECTOR_REGISTRY[key]
                        best_auroc = -1.0
                        best_cfg: dict = {}

                        for try_layer in auto_layer_grid:
                            X_tr_l = store.get_activations(
                                try_layer,
                                indices=splits["train_benign"].indices.tolist(),
                            )
                            X_va_l = store.get_activations(
                                try_layer,
                                indices=splits["val_benign"].indices.tolist(),
                            )
                            X_tr_jb_l = (
                                store.get_activations(
                                    try_layer,
                                    indices=splits["train_jailbreak"].indices.tolist(),
                                )
                                if has_labeled_jailbreaks else None
                            )
                            X_va_jb_l = (
                                store.get_activations(
                                    try_layer,
                                    indices=splits["val_jailbreak"].indices.tolist(),
                                )
                                if has_labeled_jailbreaks else None
                            )

                            for try_pca in auto_pca_grid:
                                step += 1
                                status_text.text(
                                    f"[{info['display']}] Config {step}/{n_total}: "
                                    f"Layer {try_layer}, PCA {try_pca}"
                                )
                                progress.progress(step / n_total)

                                try:
                                    det = info["factory"](try_pca, **det_kwargs.get(key, {}))
                                    det_type = info["type"]
                                    if det_type == "supervised":
                                        if X_tr_jb_l is None:
                                            continue
                                        X_combined = np.vstack([X_tr_l, X_tr_jb_l])
                                        y_combined = np.concatenate([
                                            np.zeros(len(X_tr_l), dtype=np.int64),
                                            np.ones(len(X_tr_jb_l), dtype=np.int64),
                                        ])
                                        det.fit(X_combined, y_combined)
                                    else:
                                        det.fit(X_tr_l)

                                    auroc = _compute_detector_val_auroc(
                                        det, X_va_l, X_va_jb_l,
                                    )
                                    if auroc is not None and auroc > best_auroc:
                                        best_auroc = auroc
                                        best_cfg = {
                                            "layer": try_layer, "pca": try_pca,
                                            "auroc": auroc,
                                        }
                                except Exception:
                                    continue

                        best_configs[key] = best_cfg

                    opt_status.update(label="Auto-optimize complete!", state="complete")

                # Show auto-optimize results
                st.markdown("**Auto-optimize results:**")
                for key in selected_keys:
                    cfg = best_configs.get(key, {})
                    info = _DETECTOR_REGISTRY[key]
                    if cfg:
                        st.markdown(
                            f"- **{info['display']}**: Layer {cfg['layer']}, "
                            f"PCA {cfg['pca']} → Val AUROC: {cfg['auroc']:.3f}"
                        )
                    else:
                        st.markdown(f"- **{info['display']}**: No valid config found")

                # Use best layer/PCA from primary detector
                primary_key = selected_keys[0]
                best_primary = best_configs.get(primary_key, {})
                if best_primary:
                    layer = best_primary["layer"]
                    pca_dims = best_primary["pca"]
                    st.info(
                        f"Using best config from **{_DETECTOR_REGISTRY[primary_key]['display']}**: "
                        f"Layer {layer}, PCA {pca_dims}"
                    )

                # Rebuild ensemble with optimal PCA
                X_train = store.get_activations(
                    layer, indices=splits["train_benign"].indices.tolist()
                )
                X_val = store.get_activations(
                    layer, indices=splits["val_benign"].indices.tolist()
                )
                if has_labeled_jailbreaks:
                    X_train_jailbreak = store.get_activations(
                        layer, indices=splits["train_jailbreak"].indices.tolist()
                    )
                    X_val_jailbreak = store.get_activations(
                        layer, indices=splits["val_jailbreak"].indices.tolist()
                    )

                ensemble = DetectorEnsemble(threshold_percentile=threshold_pctile)
                for key in selected_keys:
                    info = _DETECTOR_REGISTRY[key]
                    det_type_enum = (
                        DetectorType.SUPERVISED
                        if info["type"] == "supervised"
                        else DetectorType.UNSUPERVISED
                    )
                    detector = info["factory"](pca_dims, **det_kwargs.get(key, {}))
                    ensemble.add_detector(
                        key, detector, weight=det_weights[key],
                        detector_type=det_type_enum,
                    )

            # --- Train ensemble (final or only pass) ---
            trained_results: list[dict] = []

            with st.status("Training detectors...", expanded=True) as status:
                t0 = time.perf_counter()

                # Show what will be trained
                for name, det, weight in ensemble.detectors:
                    det_type = ensemble.detector_types_.get(name, DetectorType.UNSUPERVISED)
                    type_label = "supervised" if det_type == DetectorType.SUPERVISED else "unsupervised"
                    st.write(f"Training **{name}** ({type_label}, weight={weight})...")

                ensemble.fit(
                    X_train, X_val,
                    X_train_jailbreak=X_train_jailbreak,
                    X_val_jailbreak=X_val_jailbreak,
                )
                elapsed = time.perf_counter() - t0
                status.update(label="Training complete!", state="complete")

            # Select primary and calibrate threshold
            primary_name, primary_det = _select_primary_detector(ensemble)
            primary_threshold, val_fpr = _calibrate_primary_threshold(
                primary_det, X_val, X_val_jailbreak, threshold_pctile,
            )

            # Compute per-detector AUROC for the summary
            for name, det, weight in ensemble.detectors:
                det_t0 = time.perf_counter()
                auroc = _compute_detector_val_auroc(det, X_val, X_val_jailbreak)
                trained_results.append({
                    "name": name,
                    "display": _DETECTOR_REGISTRY[name]["display"],
                    "val_auroc": auroc,
                    "time_s": elapsed / len(ensemble.detectors),
                    "is_primary": name == primary_name,
                })

            # Determine ensemble type for metadata
            has_sup = any(
                _DETECTOR_REGISTRY[k]["type"] == "supervised" for k in selected_keys
            )
            has_unsup = any(
                _DETECTOR_REGISTRY[k]["type"] == "unsupervised" for k in selected_keys
            )
            if has_sup and has_unsup:
                ensemble_type = "hybrid"
            elif has_sup:
                ensemble_type = "supervised"
            else:
                ensemble_type = "unsupervised"

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

            # ---- SECTION 5: Post-Training — update state and show banner ----
            train_metrics = {
                "val_false_positive_rate": val_fpr,
                "threshold": primary_threshold,
                "ensemble_threshold": float(ensemble.threshold_),
                "layer": int(layer),
                "pca_dims": int(pca_dims),
                "n_detectors": len(selected_keys),
                "ensemble_type": ensemble_type,
                "primary_detector": primary_name,
                "training_time_s": round(elapsed, 1),
            }

            state.reset_from(4)
            state.model_path = output_dir
            state.train_metrics = train_metrics
            state.trained_detectors = trained_results
            state.model_ready = True

            # Force rerun so sidebar refreshes with checkmark
            st.rerun()
