"""Activation extraction page — run extraction with progress tracking."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.state import PipelineState
from venator.data.prompts import PromptDataset

state = PipelineState()
config = state.config

st.header("2. Extract Activations")
st.markdown(
    "Extract hidden state activations from an LLM for each prompt in the dataset. "
    "This is the slowest step (~1-3 seconds per prompt)."
)

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(2):
    st.warning("Complete the **Data** stage first to unlock extraction.")
    st.stop()

# ------------------------------------------------------------------
# Already complete — show results
# ------------------------------------------------------------------


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


if state.activations_ready and state.store_path and state.store_path.exists():
    st.success("Activations extracted.")
    _show_store_summary(state.store_path)

    col_re, col_cont = st.columns(2)
    with col_re:
        if st.button("Re-extract"):
            state.reset_from(2)
            st.rerun()
    with col_cont:
        if st.button("Continue  \u2192", type="primary"):
            st.switch_page("pages/3_split.py")
    st.stop()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

st.subheader("Configuration")

cfg_col1, cfg_col2 = st.columns(2)
with cfg_col1:
    model_id = st.text_input("Model ID", value=config.model_id)
    output_path_str = st.text_input(
        "Output path",
        value=str(config.activations_dir / "all.h5"),
    )
with cfg_col2:
    # Offer a range of layers for Mistral-7B (32 layers, 0-indexed)
    available_layers = list(range(0, 32))
    selected_layers = st.multiselect(
        "Layers to extract",
        options=available_layers,
        default=config.extraction_layers,
    )

# Estimate time
benign_ds = PromptDataset.load(state.benign_path) if state.benign_path else None
jailbreak_ds = PromptDataset.load(state.jailbreak_path) if state.jailbreak_path else None

if benign_ds and jailbreak_ds:
    n_total = len(benign_ds) + len(jailbreak_ds)
    st.info(
        f"**{n_total} prompts** to extract "
        f"({len(benign_ds)} benign + {len(jailbreak_ds)} jailbreak). "
        f"Estimated time: {n_total * 1.5 / 60:.0f}\u2013{n_total * 3 / 60:.0f} minutes."
    )
else:
    n_total = 0
    st.warning("Could not load prompt datasets. Check the Data stage.")

# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------

st.divider()

can_start = bool(selected_layers) and n_total > 0

if st.button("Start Extraction", disabled=not can_start, type="primary"):
    output_path = Path(output_path_str)

    # Handle existing file
    if output_path.exists():
        output_path.unlink()

    # Merge datasets
    combined = PromptDataset.merge(benign_ds, jailbreak_ds)  # type: ignore[arg-type]

    with st.status("Extracting activations...", expanded=True) as status:
        # Import extractor here to avoid loading MLX at page load
        from venator.activation.extractor import ActivationExtractor

        st.write("Loading model... (this may take a minute on first run)")
        extractor = ActivationExtractor(
            model_id=model_id,
            layers=selected_layers,
            config=config,
        )

        # Trigger model load and get metadata
        hidden_dim = extractor.hidden_dim
        layers = sorted(extractor._target_layers)
        st.write(f"Model loaded. Hidden dim: {hidden_dim}, layers: {layers}")

        # Create store
        output_path.parent.mkdir(parents=True, exist_ok=True)
        store = ActivationStore.create(
            output_path,
            model_id=model_id,
            layers=layers,
            hidden_dim=hidden_dim,
        )

        # Progress widgets
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

    # Update state
    state.reset_from(2)
    state.store_path = output_path
    state.activations_ready = True

    st.success(
        f"Extracted {n_total} prompts in {elapsed:.1f}s "
        f"({elapsed / n_total:.2f}s/prompt)."
    )
    _show_store_summary(output_path)

    if st.button("Continue to Split  \u2192", type="primary", key="post_extract_continue"):
        st.switch_page("pages/3_split.py")
