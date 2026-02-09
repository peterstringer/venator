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
        template="plotly_white",
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
        template="plotly_white",
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
        template="plotly_white",
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
        template="plotly_white",
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
        template="plotly_white",
    )
    return fig


def score_gauge(
    score: float,
    threshold: float,
    title: str = "Anomaly Score",
) -> go.Figure:
    """Gauge/meter showing a score relative to the decision threshold.

    Args:
        score: The ensemble anomaly score.
        threshold: Decision threshold for anomaly classification.
        title: Gauge title.

    Returns:
        Plotly Figure with a gauge indicator.
    """
    axis_max = max(1.0, score * 1.2)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(valueformat=".4f"),
            title=dict(text=title),
            gauge=dict(
                axis=dict(range=[0, axis_max]),
                bar=dict(color="rgba(55, 128, 191, 0.8)"),
                steps=[
                    dict(range=[0, threshold * 0.7], color="rgba(44, 160, 101, 0.2)"),
                    dict(range=[threshold * 0.7, threshold], color="rgba(255, 193, 7, 0.2)"),
                    dict(range=[threshold, axis_max], color="rgba(219, 64, 82, 0.2)"),
                ],
                threshold=dict(
                    line=dict(color="black", width=4),
                    thickness=0.75,
                    value=threshold,
                ),
            ),
        )
    )
    fig.update_layout(
        height=250,
        margin=dict(t=60, b=20, l=30, r=30),
        template="plotly_white",
    )
    return fig


def ablation_line_chart(
    x_values: list[int | float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    title: str = "",
    highlight_best: bool = True,
) -> go.Figure:
    """Line chart for ablation results with optional best-point highlight.

    Args:
        x_values: X-axis values (e.g., layer indices or PCA dimensions).
        y_values: Y-axis metric values (e.g., AUROC).
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Optional chart title.
        highlight_best: Whether to highlight the point with the best y-value.

    Returns:
        Plotly Figure with a line chart.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name=y_label,
            line=dict(color="rgba(55, 128, 191, 1)"),
            marker=dict(size=8),
        )
    )

    if highlight_best and y_values:
        best_idx = int(np.argmax(y_values))
        best_x = x_values[best_idx]
        best_y = y_values[best_idx]
        fig.add_trace(
            go.Scatter(
                x=[best_x],
                y=[best_y],
                mode="markers",
                name=f"Best: {best_y:.4f}",
                marker=dict(size=14, color="rgba(219, 64, 82, 1)", symbol="star"),
            )
        )
        fig.add_hline(
            y=best_y,
            line_dash="dash",
            line_color="lightgray",
            annotation_text=f"{best_y:.4f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        margin=dict(t=40, b=40, l=40, r=20),
        template="plotly_white",
    )
    return fig


def correlation_heatmap(
    scores: dict[str, np.ndarray],
    title: str = "Score Correlation Heatmap",
) -> go.Figure:
    """Heatmap of pairwise correlation between detector scores.

    Args:
        scores: Mapping of detector name to score arrays (all same length).
        title: Chart title.

    Returns:
        Plotly Figure with a correlation heatmap.
    """
    names = list(scores.keys())
    arrays = np.array([scores[n] for n in names])
    corr = np.corrcoef(arrays)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=names,
            y=names,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            texttemplate="%{z:.2f}",
            textfont=dict(size=12),
        )
    )
    fig.update_layout(
        title=title,
        height=400,
        margin=dict(t=40, b=40, l=40, r=20),
        template="plotly_white",
        yaxis=dict(scaleanchor="x"),
    )
    return fig


def detector_comparison_grouped_bar(
    unsupervised_metrics: dict[str, float],
    supervised_metrics: dict[str, float],
    metric_name: str = "AUROC",
) -> go.Figure:
    """Grouped bar chart: unsupervised (blue) vs supervised (green) detectors.

    Args:
        unsupervised_metrics: Mapping of detector name to metric value.
        supervised_metrics: Mapping of detector name to metric value.
        metric_name: The metric being compared (for axis label).

    Returns:
        Plotly Figure with grouped bars color-coded by detector type.
    """
    all_names = list(unsupervised_metrics.keys()) + list(supervised_metrics.keys())
    unsup_vals = [unsupervised_metrics.get(n, 0) for n in all_names]
    sup_vals = [supervised_metrics.get(n, 0) for n in all_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Unsupervised",
        x=all_names,
        y=unsup_vals,
        marker_color="rgba(55, 128, 191, 0.8)",
        text=[f"{v:.3f}" if v > 0 else "" for v in unsup_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Supervised",
        x=all_names,
        y=sup_vals,
        marker_color="rgba(44, 160, 101, 0.8)",
        text=[f"{v:.3f}" if v > 0 else "" for v in sup_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title=f"{metric_name} by Detector (Unsupervised vs Supervised)",
        yaxis_title=metric_name,
        height=400,
        margin=dict(t=40, b=40, l=40, r=20),
        template="plotly_white",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def labeled_data_efficiency_chart(
    n_labeled: list[int],
    aurocs_by_detector: dict[str, list[float]],
    unsupervised_baselines: dict[str, float],
) -> go.Figure:
    """Line chart showing AUROC vs number of labeled jailbreaks.

    Dashed horizontal lines for unsupervised baselines.

    Args:
        n_labeled: X-axis values (number of labeled jailbreaks).
        aurocs_by_detector: Detector name -> list of AUROC values per n_labeled.
        unsupervised_baselines: Detector name -> AUROC baseline value.

    Returns:
        Plotly Figure with efficiency curves and baseline references.
    """
    # Supervised detector colors (green shades)
    sup_colors = [
        "rgba(44, 160, 101, 1)",
        "rgba(0, 128, 128, 1)",
        "rgba(76, 175, 80, 1)",
        "rgba(27, 94, 32, 1)",
    ]
    # Unsupervised baseline colors (blue shades)
    unsup_colors = [
        "rgba(55, 128, 191, 0.6)",
        "rgba(100, 149, 237, 0.6)",
        "rgba(70, 130, 180, 0.6)",
    ]

    fig = go.Figure()

    for i, (name, aurocs) in enumerate(aurocs_by_detector.items()):
        color = sup_colors[i % len(sup_colors)]
        fig.add_trace(go.Scatter(
            x=n_labeled,
            y=aurocs,
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            marker=dict(size=8),
        ))

    for i, (name, baseline) in enumerate(unsupervised_baselines.items()):
        color = unsup_colors[i % len(unsup_colors)]
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{name}: {baseline:.3f}",
            annotation_position="top left" if i % 2 == 0 else "bottom left",
        )

    fig.update_layout(
        title="Labeled Data Efficiency: AUROC vs Number of Labeled Jailbreaks",
        xaxis_title="Number of Labeled Jailbreaks",
        yaxis_title="AUROC",
        height=400,
        margin=dict(t=40, b=40, l=40, r=20),
        template="plotly_white",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def generalization_heatmap(
    sources: list[str],
    auroc_matrix: np.ndarray,
) -> go.Figure:
    """Heatmap of cross-source generalization performance.

    Args:
        sources: Source names (used for both axes).
        auroc_matrix: 2D array of AUROC values (rows=trained on, cols=tested on).

    Returns:
        Plotly Figure with a generalization heatmap.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=auroc_matrix,
            x=[f"Test: {s}" for s in sources],
            y=[f"Train: {s}" for s in sources],
            colorscale="Viridis",
            zmin=0.5,
            zmax=1.0,
            texttemplate="%{z:.3f}",
            textfont=dict(size=12),
        )
    )
    fig.update_layout(
        title="Cross-Source Generalization (AUROC)",
        height=400,
        margin=dict(t=40, b=80, l=100, r=20),
        template="plotly_white",
        yaxis=dict(scaleanchor="x"),
    )
    return fig


def ensemble_roc_comparison(
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> go.Figure:
    """Multiple ROC curves on same plot for ensemble comparison.

    Args:
        roc_curves: Mapping of ensemble name to (fpr, tpr, auroc).

    Returns:
        Plotly Figure with overlaid ROC curves.
    """
    colors = [
        "rgba(55, 128, 191, 1)",   # blue - unsupervised
        "rgba(44, 160, 101, 1)",   # green - supervised
        "rgba(219, 64, 82, 1)",    # red - hybrid
        "rgba(153, 102, 204, 1)",  # purple - extra
    ]

    fig = go.Figure()
    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="lightgray"),
        showlegend=False,
    ))

    for i, (name, (fpr, tpr, auroc)) in enumerate(roc_curves.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC={auroc:.3f})",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        margin=dict(t=40, b=40, l=40, r=20),
        template="plotly_white",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig
