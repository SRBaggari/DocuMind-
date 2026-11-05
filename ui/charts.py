"""Plotly charts. Colors come from a palette checked for color-blind separation.

Bars are horizontal because their labels are words. Series color marks identity only,
so every chart with two series also has a legend.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from core.document_comparator import SimilarityGrid
from core.keyword_extractor import KeywordComparison, Topic

SERIES_A = "#2a78d6"
SERIES_B = "#eb6834"
# One hue from light to dark, because a similarity score is a magnitude.
SEQUENTIAL = ((0.0, "#eaf2fb"), (0.5, "#6ea6e6"), (1.0, "#123f78"))
_BAR_HEIGHT = 26
_MIN_HEIGHT = 220
_CHROME_HEIGHT = 90


def _layout(figure: go.Figure, rows: int, x_title: str) -> go.Figure:
    figure.update_layout(
        height=max(_MIN_HEIGHT, rows * _BAR_HEIGHT + _CHROME_HEIGHT),
        margin={"l": 8, "r": 16, "t": 16, "b": 8},
        barcornerradius=4,
        bargap=0.35,
        xaxis_title=x_title,
        yaxis={"autorange": "reversed", "automargin": True},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return figure


def keyword_frequency(terms: Sequence[tuple[str, int]]) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=[count for _, count in terms],
            y=[term for term, _ in terms],
            orientation="h",
            marker_color=SERIES_A,
            hovertemplate="%{y}: %{x} uses<extra></extra>",
        )
    )
    return _layout(figure, len(terms), "Times used")


def topic_distribution(topics: Sequence[Topic]) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=[topic.share * 100 for topic in topics],
            y=[topic.label for topic in topics],
            orientation="h",
            marker_color=SERIES_A,
            hovertemplate="%{y}: %{x:.0f}% of topic weight<extra></extra>",
        )
    )
    return _layout(figure, len(topics), "Share of topic weight (%)")


def keyword_comparison(
    keywords: Sequence[KeywordComparison], name_a: str, name_b: str
) -> go.Figure:
    terms = [keyword.term for keyword in keywords]
    figure = go.Figure(
        [
            go.Bar(
                name=name,
                x=values,
                y=terms,
                orientation="h",
                marker_color=color,
                hovertemplate=f"{name}<br>%{{y}}: %{{x:.3f}}<extra></extra>",
            )
            for name, values, color in (
                (name_a, [k.score_a for k in keywords], SERIES_A),
                (name_b, [k.score_b for k in keywords], SERIES_B),
            )
        ]
    )
    figure.update_layout(barmode="group", bargroupgap=0.1)
    return _layout(figure, len(terms) * 2, "TF-IDF importance")


def similarity_heatmap(grid: SimilarityGrid, name_a: str, name_b: str) -> go.Figure:
    figure = go.Figure(
        go.Heatmap(
            z=[list(row) for row in grid.values],
            x=list(grid.labels_b),
            y=list(grid.labels_a),
            zmin=0,
            zmax=1,
            colorscale=[list(step) for step in SEQUENTIAL],
            colorbar={"title": "Similarity", "thickness": 12},
            xgap=1,
            ygap=1,
            hovertemplate="A %{y}<br>B %{x}<br>Strongest match: %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=460,
        margin={"l": 8, "r": 8, "t": 16, "b": 8},
        xaxis={"title": name_b, "showticklabels": len(grid.labels_b) <= 12},
        yaxis={
            "title": name_a,
            "autorange": "reversed",
            "showticklabels": len(grid.labels_a) <= 12,
        },
    )
    return figure
