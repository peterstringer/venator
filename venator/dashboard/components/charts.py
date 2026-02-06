"""Reusable Plotly chart builders for the dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]


def source_distribution_chart(
    source_counts: dict[str, int],
    title: str = "",
) -> go.Figure:
    """Create a pie chart showing prompt source distribution.

    Args:
        source_counts: Mapping of source name to prompt count.
        title: Optional chart title.

    Returns:
        A Plotly Figure with a donut-style pie chart.
    """
    labels = list(source_counts.keys())
    values = list(source_counts.values())

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(t=40, b=20, l=20, r=20),
        height=280,
    )
    return fig


def score_distribution_chart(
    benign_scores: np.ndarray,
    jailbreak_scores: np.ndarray,
    threshold: float,
) -> go.Figure:
    """Overlaid histograms of ensemble scores for benign vs jailbreak.

    Args:
        benign_scores: Ensemble scores for benign test samples.
        jailbreak_scores: Ensemble scores for jailbreak test samples.
        threshold: Decision threshold (vertical dashed line).

    Returns:
        Plotly Figure with two overlaid histograms and threshold line.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=benign_scores,
            name="Benign",
            marker_color="rgba(55, 128, 191, 0.6)",
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=jailbreak_scores,
            name="Jailbreak",
            marker_color="rgba(219, 64, 82, 0.6)",
            nbinsx=40,
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Threshold: {threshold:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Score Distribution",
        xaxis_title="Ensemble Score",
        yaxis_title="Count",
        barmode="overlay",
        height=380,
        margin=dict(t=40, b=40, l=40, r=20),
    )
    return fig


def roc_curve_chart(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    op_fpr: float | None = None,
    op_tpr: float | None = None,
) -> go.Figure:
    """ROC curve with AUC annotation and optional operating point.

    Args:
        fpr: False positive rate values from sklearn.roc_curve.
        tpr: True positive rate values from sklearn.roc_curve.
        auroc: Area under the ROC curve.
        op_fpr: FPR at the operating point (threshold).
        op_tpr: TPR at the operating point (threshold).

    Returns:
        Plotly Figure with ROC curve, diagonal reference, and operating point.
    """
    fig = go.Figure()
    # Diagonal reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="lightgray"),
            showlegend=False,
        )
    )
    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auroc:.3f})",
            line=dict(color="rgba(55, 128, 191, 1)"),
        )
    )
    # Operating point
    if op_fpr is not None and op_tpr is not None:
        fig.add_trace(
            go.Scatter(
                x=[op_fpr],
                y=[op_tpr],
                mode="markers",
                name=f"Threshold ({op_fpr:.3f}, {op_tpr:.3f})",
                marker=dict(size=10, color="red", symbol="x"),
            )
        )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        margin=dict(t=40, b=40, l=40, r=20),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def precision_recall_chart(
    recall: np.ndarray,
    precision: np.ndarray,
    auprc: float,
) -> go.Figure:
    """Precision-Recall curve with AUPRC annotation.

    Args:
        recall: Recall values (x-axis).
        precision: Precision values (y-axis).
        auprc: Area under the precision-recall curve.

    Returns:
        Plotly Figure with PR curve.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR (AUC = {auprc:.3f})",
            line=dict(color="rgba(44, 160, 101, 1)"),
        )
    )
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=380,
        margin=dict(t=40, b=40, l=40, r=20),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def detector_comparison_chart(
    detector_aurocs: dict[str, float],
) -> go.Figure:
    """Bar chart comparing AUROC across individual detectors and ensemble.

    Args:
        detector_aurocs: Mapping of detector name to AUROC value.

    Returns:
        Plotly Figure with a horizontal bar chart.
    """
    names = list(detector_aurocs.keys())
    values = list(detector_aurocs.values())

    colors = [
        "rgba(55, 128, 191, 0.8)" if n != "Ensemble" else "rgba(219, 64, 82, 0.8)"
        for n in names
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Per-Detector AUROC Comparison",
        xaxis_title="AUROC",
        height=280,
        margin=dict(t=40, b=40, l=120, r=40),
        xaxis=dict(range=[0, 1.05]),
    )
    return fig
