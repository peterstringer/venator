"""Reusable Plotly chart builders for the dashboard."""

from __future__ import annotations

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
