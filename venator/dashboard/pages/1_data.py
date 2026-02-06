"""Data collection page — collect/upload prompts and preview datasets."""

from __future__ import annotations

import json
import random

import streamlit as st

from venator.dashboard.components.charts import source_distribution_chart
from venator.dashboard.state import PipelineState
from venator.data.prompts import (
    PromptDataset,
    collect_benign_prompts,
    collect_jailbreak_prompts,
)

state = PipelineState()
config = state.config

# Source label → internal key mapping
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


def _show_dataset_stats(dataset: PromptDataset, key_prefix: str) -> None:
    """Display stats, pie chart, and sample prompts for a loaded dataset."""
    st.metric("Total prompts", len(dataset))
    counts = dataset.source_counts()
    if counts:
        fig = source_distribution_chart(counts, title="Source Distribution")
        st.plotly_chart(fig, use_container_width=True)
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


# ======================================================================
# Page layout
# ======================================================================

st.header("1. Data Collection")
st.markdown(
    "Collect or upload benign and jailbreak prompt datasets for the pipeline."
)

col_benign, col_jailbreak = st.columns(2)

# ------------------------------------------------------------------
# LEFT COLUMN — Benign Prompts
# ------------------------------------------------------------------

with col_benign:
    st.subheader("Benign Prompts")
    st.caption("Training distribution — used for train, validation, and test.")

    # Option A: Collect
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
                # Filter by selected sources
                filtered = [(p, s) for p, s in raw if s in allowed_keys]
                if not filtered:
                    st.error("No prompts collected. Check source availability.")
                else:
                    prompts = [p for p, _ in filtered]
                    sources = [s for _, s in filtered]
                    ds = PromptDataset(prompts, [0] * len(prompts), sources)
                    st.session_state["_benign_dataset"] = ds
                    st.success(f"Collected {len(ds)} benign prompts.")

    # Option B: Upload
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

    # Show stats if dataset is loaded
    benign_ds: PromptDataset | None = st.session_state.get("_benign_dataset")
    if benign_ds is not None:
        _show_dataset_stats(benign_ds, "benign")

# ------------------------------------------------------------------
# RIGHT COLUMN — Jailbreak Prompts
# ------------------------------------------------------------------

with col_jailbreak:
    st.subheader("Jailbreak Prompts")
    st.caption("Test only — never used in training or validation.")
    st.warning(
        "These prompts are used **only for evaluation**. They will never "
        "appear in training data, per the unsupervised anomaly detection methodology."
    )

    # Option A: Collect
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
                    prompts = [p for p, _ in filtered]
                    sources = [s for _, s in filtered]
                    ds = PromptDataset(prompts, [1] * len(prompts), sources)
                    st.session_state["_jailbreak_dataset"] = ds
                    st.success(f"Collected {len(ds)} jailbreak prompts.")

    # Option B: Upload
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

    # Show stats if dataset is loaded
    jailbreak_ds: PromptDataset | None = st.session_state.get("_jailbreak_dataset")
    if jailbreak_ds is not None:
        _show_dataset_stats(jailbreak_ds, "jailbreak")

# ------------------------------------------------------------------
# BOTTOM — Save & Continue
# ------------------------------------------------------------------

st.divider()

both_loaded = (
    st.session_state.get("_benign_dataset") is not None
    and st.session_state.get("_jailbreak_dataset") is not None
)

if st.button("Save & Continue  \u2192", disabled=not both_loaded, type="primary"):
    benign_ds = st.session_state["_benign_dataset"]
    jailbreak_ds = st.session_state["_jailbreak_dataset"]

    benign_path = config.prompts_dir / "benign.jsonl"
    jailbreak_path = config.prompts_dir / "jailbreaks.jsonl"

    benign_ds.save(benign_path)
    jailbreak_ds.save(jailbreak_path)

    state.reset_from(1)
    state.benign_path = benign_path
    state.jailbreak_path = jailbreak_path
    state.prompts_ready = True

    st.success(
        f"Saved {len(benign_ds)} benign prompts to `{benign_path}` and "
        f"{len(jailbreak_ds)} jailbreak prompts to `{jailbreak_path}`."
    )
    st.switch_page("pages/2_extract.py")

if not both_loaded:
    st.caption("Collect or upload both benign and jailbreak datasets to continue.")
