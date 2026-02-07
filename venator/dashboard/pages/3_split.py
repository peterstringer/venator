"""Data splitting page — create splits and verify methodology constraints."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from venator.activation.storage import ActivationStore
from venator.dashboard.state import PipelineState
from venator.data.splits import SplitManager, SplitMode

state = PipelineState()
config = state.config

st.header("3. Split Data")
st.markdown(
    "Create train/validation/test splits with proper methodology constraints: "
    "jailbreaks are reserved for the test set only."
)

# ------------------------------------------------------------------
# Prerequisite check
# ------------------------------------------------------------------

if not state.is_stage_available(3):
    st.warning("Complete the **Extract** stage first to unlock splitting.")
    st.stop()

# ------------------------------------------------------------------
# Helper to display split summary and methodology checks
# ------------------------------------------------------------------


def _show_split_summary(
    splits: dict, store: ActivationStore, manager: SplitManager
) -> None:
    """Display the split summary table and methodology verification panel."""
    n_benign = int((store.get_labels() == 0).sum())

    rows = []
    for name in ["train", "val", "test_benign", "test_jailbreak"]:
        s = splits[name]
        pct = f"{s.n_samples / n_benign * 100:.0f}%" if not s.contains_jailbreaks else "\u2014"
        jailbreak_flag = "YES (100%)" if s.contains_jailbreaks else "NO"
        rows.append({
            "Split": s.name,
            "N Samples": s.n_samples,
            "% of Benign": pct,
            "Contains Jailbreaks": jailbreak_flag,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Methodology verification
    st.subheader("Methodology Verification")

    # Run validation
    validation_passed = True
    try:
        manager.validate_splits(splits, store, mode=SplitMode.UNSUPERVISED)
    except ValueError as e:
        validation_passed = False
        st.error(f"Validation failed: {e}")

    if validation_passed:
        st.markdown(":white_check_mark: Training data contains **0 jailbreak prompts**")
        st.markdown(":white_check_mark: Validation data contains **0 jailbreak prompts**")
        st.markdown(":white_check_mark: All jailbreak prompts reserved for **test set**")
        st.markdown(":white_check_mark: No index overlap between splits")

        ratio = splits["train"].n_samples / config.pca_dims
        ratio_ok = ratio >= 5.0
        icon = ":white_check_mark:" if ratio_ok else ":warning:"
        quality = "healthy" if ratio_ok else "low"
        st.markdown(
            f"{icon} Sample-to-feature ratio: "
            f"**{splits['train'].n_samples}** samples / "
            f"**{config.pca_dims}** PCA dims = "
            f"**{ratio:.1f}x** ({quality})"
        )


# ------------------------------------------------------------------
# Already complete — show results
# ------------------------------------------------------------------

if state.splits_ready and state.splits_path and state.splits_path.exists():
    st.success("Splits created.")

    store = ActivationStore(state.store_path)
    splits = SplitManager.load_splits(state.splits_path)
    manager = SplitManager(seed=config.random_seed)
    _show_split_summary(splits, store, manager)

    col_re, col_cont = st.columns(2)
    with col_re:
        if st.button("Re-split"):
            state.reset_from(3)
            st.rerun()
    with col_cont:
        if st.button("Continue  \u2192", type="primary"):
            st.switch_page("pages/4_train.py")
    st.stop()

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

st.subheader("Configuration")

cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
with cfg_col1:
    train_frac = st.slider(
        "Train fraction",
        min_value=0.50,
        max_value=0.80,
        value=0.70,
        step=0.05,
    )
with cfg_col2:
    val_frac = st.slider(
        "Validation fraction",
        min_value=0.05,
        max_value=0.25,
        value=0.15,
        step=0.05,
    )
with cfg_col3:
    test_frac = round(1.0 - train_frac - val_frac, 2)
    st.metric("Test fraction (benign)", f"{test_frac:.0%}")

seed = st.number_input("Random seed", min_value=0, value=config.random_seed, step=1)

if test_frac <= 0:
    st.error("Train + validation fractions must be less than 1.0.")
    st.stop()

# ------------------------------------------------------------------
# Create splits
# ------------------------------------------------------------------

st.divider()

if st.button("Create Splits", type="primary"):
    store = ActivationStore(state.store_path)
    manager = SplitManager(seed=int(seed))

    with st.spinner("Creating splits..."):
        splits = manager.create_splits(
            store,
            mode=SplitMode.UNSUPERVISED,
            benign_train_frac=train_frac,
            benign_val_frac=val_frac,
        )

    _show_split_summary(splits, store, manager)

    # Store in session state for save step
    st.session_state["_splits"] = splits
    st.session_state["_split_manager"] = manager

# ------------------------------------------------------------------
# Save & Continue
# ------------------------------------------------------------------

if st.session_state.get("_splits") is not None:
    st.divider()
    if st.button("Save & Continue  \u2192", type="primary"):
        splits = st.session_state["_splits"]
        manager = st.session_state["_split_manager"]
        splits_path = config.data_dir / "splits.json"

        manager.save_splits(splits, splits_path, mode=SplitMode.UNSUPERVISED)

        state.reset_from(3)
        state.splits_path = splits_path
        state.splits_ready = True

        st.success(f"Saved splits to `{splits_path}`.")
        st.switch_page("pages/4_train.py")
