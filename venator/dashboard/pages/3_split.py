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
    "Create train/validation/test splits with proper methodology constraints."
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
    splits: dict, store: ActivationStore, manager: SplitManager, mode: SplitMode
) -> None:
    """Display the split summary table and methodology verification panel."""
    n_benign = int((store.get_labels() == 0).sum())
    n_jailbreak = int((store.get_labels() == 1).sum())

    rows = []
    for name, s in splits.items():
        if s.contains_jailbreaks:
            pct_label = "\u2014"
        else:
            pct_label = f"{s.n_samples / n_benign * 100:.0f}%" if n_benign > 0 else "0%"
        jailbreak_flag = "YES (100%)" if s.contains_jailbreaks else "NO"
        rows.append({
            "Split": s.name,
            "N Samples": s.n_samples,
            "% of Benign": pct_label,
            "Contains Jailbreaks": jailbreak_flag,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Methodology verification
    st.subheader("Methodology Verification")

    validation_passed = True
    try:
        manager.validate_splits(splits, store, mode=mode)
    except ValueError as e:
        validation_passed = False
        st.error(f"Validation failed: {e}")

    if validation_passed:
        if mode == SplitMode.UNSUPERVISED:
            st.markdown(":white_check_mark: Training data contains **0 jailbreak prompts**")
            st.markdown(":white_check_mark: Validation data contains **0 jailbreak prompts**")
            st.markdown(":white_check_mark: All jailbreak prompts reserved for **test set**")
        else:
            n_train_jb = splits["train_jailbreak"].n_samples if "train_jailbreak" in splits else 0
            n_val_jb = splits["val_jailbreak"].n_samples if "val_jailbreak" in splits else 0
            n_test_jb = splits["test_jailbreak"].n_samples if "test_jailbreak" in splits else 0
            st.markdown(
                f":white_check_mark: Training jailbreaks: **{n_train_jb}** "
                f"({n_train_jb / n_jailbreak * 100:.0f}% of jailbreaks)" if n_jailbreak > 0
                else ":white_check_mark: Training jailbreaks: **0**"
            )
            st.markdown(
                f":white_check_mark: Validation jailbreaks: **{n_val_jb}** "
                f"({n_val_jb / n_jailbreak * 100:.0f}% of jailbreaks)" if n_jailbreak > 0
                else ":white_check_mark: Validation jailbreaks: **0**"
            )
            st.markdown(
                f":white_check_mark: Test jailbreaks (uncontaminated): **{n_test_jb}** "
                f"({n_test_jb / n_jailbreak * 100:.0f}% of jailbreaks)" if n_jailbreak > 0
                else ":white_check_mark: Test jailbreaks: **0**"
            )
        st.markdown(":white_check_mark: No index overlap between splits")

        # Sample-to-feature ratio check
        train_key = "train" if "train" in splits else "train_benign"
        ratio = splits[train_key].n_samples / config.pca_dims
        ratio_ok = ratio >= 5.0
        icon = ":white_check_mark:" if ratio_ok else ":warning:"
        quality = "healthy" if ratio_ok else "low"
        st.markdown(
            f"{icon} Sample-to-feature ratio: "
            f"**{splits[train_key].n_samples}** samples / "
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
    loaded_mode = SplitManager.load_mode(state.splits_path)
    manager = SplitManager(seed=config.random_seed)
    _show_split_summary(splits, store, manager, loaded_mode)

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

# Split mode selector
mode_col1, mode_col2 = st.columns(2)
with mode_col1:
    split_mode_str = st.radio(
        "Split Mode",
        options=["unsupervised", "semi_supervised"],
        format_func=lambda x: "Unsupervised" if x == "unsupervised" else "Semi-Supervised",
        horizontal=True,
        help=(
            "**Unsupervised**: jailbreaks reserved for test set only (standard methodology). "
            "**Semi-Supervised**: a small labeled jailbreak fraction in train/val for "
            "supervised/hybrid detectors."
        ),
    )
split_mode = SplitMode(split_mode_str)

is_semi = split_mode == SplitMode.SEMI_SUPERVISED

# Show jailbreak budget info for semi-supervised mode
if is_semi:
    store = ActivationStore(state.store_path)
    n_jailbreak_total = int((store.get_labels() == 1).sum())
    n_benign_total = int((store.get_labels() == 0).sum())

    if n_jailbreak_total == 0:
        st.error(
            "No jailbreak prompts found in the activation store. "
            "Semi-supervised mode requires labeled jailbreaks. "
            "Use Unsupervised mode or re-extract with labeled data."
        )
        st.stop()

    ss_col1, ss_col2 = st.columns(2)
    with ss_col1:
        jailbreak_train_frac = st.slider(
            "Labeled jailbreak fraction (training)",
            min_value=0.05,
            max_value=0.30,
            value=0.15,
            step=0.05,
            help="Fraction of jailbreaks allocated to training set.",
        )
    with ss_col2:
        jailbreak_val_frac = st.slider(
            "Labeled jailbreak fraction (validation)",
            min_value=0.05,
            max_value=0.30,
            value=0.15,
            step=0.05,
            help="Fraction of jailbreaks allocated to validation set.",
        )

    jailbreak_test_frac = round(1.0 - jailbreak_train_frac - jailbreak_val_frac, 2)
    if jailbreak_test_frac < 0.40:
        st.warning(
            f"Only {jailbreak_test_frac:.0%} of jailbreaks reserved for testing. "
            "Consider reducing train/val fractions to maintain uncontaminated test data."
        )

    n_train_jb = int(n_jailbreak_total * jailbreak_train_frac)
    n_val_jb = int(n_jailbreak_total * jailbreak_val_frac)
    n_test_jb = n_jailbreak_total - n_train_jb - n_val_jb
    st.info(
        f"**{n_train_jb}** jailbreaks for training, "
        f"**{n_val_jb}** for validation, "
        f"**{n_test_jb}** reserved for uncontaminated testing "
        f"(out of {n_jailbreak_total} total)."
    )

# Benign split fractions
cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
with cfg_col1:
    train_frac = st.slider(
        "Train fraction (benign)",
        min_value=0.50,
        max_value=0.80,
        value=0.70,
        step=0.05,
    )
with cfg_col2:
    val_frac = st.slider(
        "Validation fraction (benign)",
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
            mode=split_mode,
            benign_train_frac=train_frac,
            benign_val_frac=val_frac,
        )

    _show_split_summary(splits, store, manager, split_mode)

    # Store in session state for save step
    st.session_state["_splits"] = splits
    st.session_state["_split_manager"] = manager
    st.session_state["_split_mode"] = split_mode

# ------------------------------------------------------------------
# Save & Continue
# ------------------------------------------------------------------

if st.session_state.get("_splits") is not None:
    st.divider()
    if st.button("Save & Continue  \u2192", type="primary"):
        splits = st.session_state["_splits"]
        manager = st.session_state["_split_manager"]
        saved_mode = st.session_state.get("_split_mode", SplitMode.UNSUPERVISED)

        # Use different filename for semi-supervised splits
        if saved_mode == SplitMode.SEMI_SUPERVISED:
            splits_path = config.data_dir / "splits_semi.json"
        else:
            splits_path = config.data_dir / "splits.json"

        manager.save_splits(splits, splits_path, mode=saved_mode)

        state.reset_from(3)
        state.splits_path = splits_path
        state.splits_ready = True

        st.success(f"Saved splits to `{splits_path}`.")
        st.switch_page("pages/4_train.py")
