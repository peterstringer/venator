"""Sortable/filterable prompt explorer component."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def render_prompt_explorer(
    prompts: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> None:
    """Render a tabbed prompt explorer with filtering.

    Shows a sortable table of test prompts with their scores and predictions,
    filterable by outcome category (TP, FP, FN, TN).

    Args:
        prompts: Test prompt texts.
        labels: Ground truth labels (0=benign, 1=jailbreak).
        scores: Ensemble anomaly scores.
        predictions: Binary predictions (0=benign, 1=anomaly).
        threshold: Decision threshold used for predictions.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)

    df = pd.DataFrame({
        "Prompt": [p[:120] + "..." if len(p) > 120 else p for p in prompts],
        "Label": ["Jailbreak" if l == 1 else "Benign" for l in labels],
        "Score": np.round(scores, 4),
        "Predicted": ["Anomaly" if p == 1 else "Normal" for p in predictions],
        "Correct": [
            "Yes" if (l == 1 and p == 1) or (l == 0 and p == 0) else "No"
            for l, p in zip(labels, predictions)
        ],
    })

    # Categorize
    is_tp = (labels == 1) & (predictions == 1)
    is_fp = (labels == 0) & (predictions == 1)
    is_fn = (labels == 1) & (predictions == 0)
    is_tn = (labels == 0) & (predictions == 0)

    tab_all, tab_tp, tab_fp, tab_fn, tab_tn = st.tabs([
        f"All ({len(df)})",
        f"True Positives ({int(is_tp.sum())})",
        f"False Positives ({int(is_fp.sum())})",
        f"False Negatives ({int(is_fn.sum())})",
        f"True Negatives ({int(is_tn.sum())})",
    ])

    with tab_all:
        st.dataframe(df, use_container_width=True, height=400)
    with tab_tp:
        st.dataframe(df[is_tp].reset_index(drop=True), use_container_width=True, height=400)
    with tab_fp:
        if is_fp.sum() > 0:
            st.dataframe(df[is_fp].reset_index(drop=True), use_container_width=True, height=400)
        else:
            st.info("No false positives.")
    with tab_fn:
        if is_fn.sum() > 0:
            st.dataframe(df[is_fn].reset_index(drop=True), use_container_width=True, height=400)
        else:
            st.info("No false negatives.")
    with tab_tn:
        st.dataframe(df[is_tn].reset_index(drop=True), use_container_width=True, height=400)


def render_error_analysis(
    prompts: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    n_show: int = 5,
) -> None:
    """Show top false positives and false negatives for error analysis.

    Args:
        prompts: Test prompt texts.
        labels: Ground truth labels.
        scores: Ensemble anomaly scores.
        predictions: Binary predictions.
        n_show: Number of examples to show per category.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)

    # Top False Positives: benign prompts with highest scores
    fp_mask = (labels == 0) & (predictions == 1)
    with st.expander(f"Top False Positives ({int(fp_mask.sum())} total)"):
        if fp_mask.sum() == 0:
            st.info("No false positives — all benign prompts correctly classified.")
        else:
            fp_indices = np.where(fp_mask)[0]
            fp_sorted = fp_indices[np.argsort(scores[fp_indices])[::-1]][:n_show]
            for idx in fp_sorted:
                st.markdown(
                    f"**Score: {scores[idx]:.4f}** — {prompts[idx][:200]}"
                )

    # Top False Negatives: jailbreaks with lowest scores
    fn_mask = (labels == 1) & (predictions == 0)
    with st.expander(f"Top False Negatives ({int(fn_mask.sum())} total)"):
        if fn_mask.sum() == 0:
            st.info("No false negatives — all jailbreaks detected.")
        else:
            fn_indices = np.where(fn_mask)[0]
            fn_sorted = fn_indices[np.argsort(scores[fn_indices])][:n_show]
            for idx in fn_sorted:
                st.markdown(
                    f"**Score: {scores[idx]:.4f}** — {prompts[idx][:200]}"
                )
