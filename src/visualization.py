"""Visualization module for Instagram Influencer Analytics.

Provides modern, interactive Plotly figures designed for executive dashboards
and student project presentations. Implements clean palettes, custom hover templates,
and avoids misleading shared axes for metrics with different units.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Cohesive modern dashboard color palette
THEME_COLORS = [
    "#6366F1",  # Indigo
    "#06B6D4",  # Cyan
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#F43F5E",  # Rose
    "#8B5CF6",  # Violet
    "#EC4899",  # Pink
    "#3B82F6",  # Blue
]

TIER_COLORS = {
    "Nano": "#94A3B8",      # Slate
    "Micro": "#38BDF8",     # Sky blue
    "Mid-tier": "#34D399",  # Mint
    "Macro": "#FBBF24",     # Warm yellow
    "Mega": "#A78BFA",      # Lavender purple
}


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent modern typography and card-friendly styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "family": "Inter, Roboto, sans-serif"},
        },
        template="plotly_white",
        margin={"l": 40, "r": 30, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240, 244, 248, 0.4)",
        font={"family": "Inter, Roboto, sans-serif", "size": 12},
        hoverlabel={"bgcolor": "#1E293B", "font_size": 12, "font_family": "Inter, Roboto"},
        xaxis={"gridcolor": "rgba(203, 213, 225, 0.5)", "zeroline": False},
        yaxis={"gridcolor": "rgba(203, 213, 225, 0.5)", "zeroline": False},
    )
    return fig


def plot_top_n_bar(
    df: pd.DataFrame,
    metric: str = "followers",
    top_n: int = 10,
    title: Optional[str] = None,
) -> go.Figure:
    """Create a horizontal bar chart of top N influencers by a specified metric."""
    if df.empty or metric not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No data available")

    metric_labels = {
        "followers": "Followers",
        "engagement_rate": "Engagement Rate (%)",
        "avg_likes": "Average Likes per Post",
        "new_post_avg_like": "Recent Average Likes",
        "influence_score": "Source Influence Score",
    }
    label = metric_labels.get(metric, metric.replace("_", " ").title())
    clean_title = title or f"Top {top_n} Influencers by {label}"

    top_df = df.dropna(subset=[metric]).nlargest(top_n, metric).sort_values(metric, ascending=True)

    fig = px.bar(
        top_df,
        x=metric,
        y="handle",
        orientation="h",
        color="country" if "country" in top_df else None,
        color_discrete_sequence=THEME_COLORS,
        hover_data={
            "handle": True,
            "followers": ":,",
            "engagement_rate": ":.2f",
            metric: True,
        },
        labels={
            metric: label,
            "handle": "Instagram Account",
            "country": "Country",
        },
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _apply_theme(fig, clean_title)


def plot_followers_vs_engagement(
    df: pd.DataFrame, log_scale: bool = True
) -> go.Figure:
    """Scatter plot displaying audience scale (followers) vs engagement rate."""
    if df.empty:
        fig = go.Figure()
        return _apply_theme(fig, "No data available")

    plot_df = df[df["followers"] > 0].copy()
    fig = px.scatter(
        plot_df,
        x="followers",
        y="engagement_rate",
        color="tier" if "tier" in plot_df else "country",
        color_discrete_map=TIER_COLORS,
        hover_name="handle",
        hover_data={
            "followers": ":,",
            "engagement_rate": ":.2f",
            "country": True,
            "avg_likes": ":,",
        },
        log_x=log_scale,
        labels={
            "followers": "Followers (Log Scale)" if log_scale else "Followers",
            "engagement_rate": "Engagement Rate (%)",
            "tier": "Follower Tier",
            "country": "Country",
        },
    )
    # Add benchmark reference line for mean engagement rate
    mean_eng = plot_df["engagement_rate"].mean()
    fig.add_hline(
        y=mean_eng,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text=f"Dataset Avg: {mean_eng:.2f}%",
        annotation_position="bottom right",
    )
    return _apply_theme(fig, "Followers vs. Engagement Rate (Scale vs. Interaction)")


def plot_tier_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing creator distribution across follower tiers."""
    if df.empty or "tier" not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No Tier Data")

    tier_counts = df["tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]

    fig = px.pie(
        tier_counts,
        values="Count",
        names="Tier",
        hole=0.45,
        color="Tier",
        color_discrete_map=TIER_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_theme(fig, "Influencer Distribution by Follower Tier")


def plot_country_distribution(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bar chart of top creator countries in the dataset."""
    if df.empty or "country" not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No Country Data")

    country_df = (
        df[df["country"] != "Unknown"]["country"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    country_df.columns = ["Country", "Influencers"]

    fig = px.bar(
        country_df,
        x="Country",
        y="Influencers",
        color="Influencers",
        color_continuous_scale="Purples",
        labels={"Influencers": "Creator Count"},
    )
    fig.update_coloraxes(showscale=False)
    return _apply_theme(fig, f"Top {top_n} Creator Geographic Origins")


def plot_metric_distribution(
    df: pd.DataFrame, metric: str = "engagement_rate"
) -> go.Figure:
    """Distribution histogram for engagement rate or follower metrics."""
    if df.empty or metric not in df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No Distribution Data")

    label = "Engagement Rate (%)" if metric == "engagement_rate" else metric.title()
    clean_s = df[metric].dropna()

    fig = px.histogram(
        clean_s,
        x=metric,
        nbins=25,
        color_discrete_sequence=["#6366F1"],
        marginal="box",
        labels={metric: label},
    )
    mean_v = float(clean_s.mean())
    median_v = float(clean_s.median())
    fig.add_vline(
        x=mean_v,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean: {mean_v:.2f}",
    )
    fig.add_vline(
        x=median_v,
        line_dash="dot",
        line_color="#10B981",
        annotation_text=f"Median: {median_v:.2f}",
    )
    return _apply_theme(fig, f"{label} Distribution in Dataset")


def plot_comparison_bar(
    comparison_df: pd.DataFrame, metric: str
) -> go.Figure:
    """Create a side-by-side bar chart for 2-5 influencers for a single metric.

    Avoids confusing multi-metric shared axes by keeping metric units isolated.
    """
    if comparison_df.empty or metric not in comparison_df.columns:
        fig = go.Figure()
        return _apply_theme(fig, "No Comparison Data")

    metric_labels = {
        "followers": "Followers",
        "engagement_rate": "Engagement Rate (%)",
        "avg_likes": "Average Likes per Post",
        "new_post_avg_like": "Recent Average Likes",
        "influence_score": "Source Influence Score",
        "posts": "Total Posts",
        "total_likes": "Total Likes",
    }
    label = metric_labels.get(metric, metric.replace("_", " ").title())

    sorted_df = comparison_df.sort_values(metric, ascending=False)
    fig = px.bar(
        sorted_df,
        x="handle",
        y=metric,
        color="handle",
        color_discrete_sequence=THEME_COLORS,
        text_auto=".2s" if metric != "engagement_rate" else ".2f",
        labels={"handle": "Influencer", metric: label},
    )
    fig.update_layout(showlegend=False)
    return _apply_theme(fig, f"Comparison: {label}")


def plot_normalized_comparison_radar(
    comparison_df: pd.DataFrame, metrics: List[str]
) -> go.Figure:
    """Radar chart comparing multiple influencers on normalized 0-100 scales.

    Normalizing metrics puts unlike units (millions of followers vs 2% engagement)
    onto a statistically sound, comparable 0-100 relative index without unit distortion.
    """
    if comparison_df.empty or len(comparison_df) < 2:
        fig = go.Figure()
        return _apply_theme(fig, "At least 2 profiles required")

    categories = [
        "followers",
        "engagement_rate",
        "avg_likes",
        "new_post_avg_like",
        "influence_score",
    ]
    available_cats = [c for c in categories if c in comparison_df.columns and comparison_df[c].notna().any()]
    if len(available_cats) < 3:
        fig = go.Figure()
        return _apply_theme(fig, "Insufficient metric overlap for radar chart")

    labels_map = {
        "followers": "Follower Scale",
        "engagement_rate": "Engagement Rate",
        "avg_likes": "Avg Likes",
        "new_post_avg_like": "Recent Likes",
        "influence_score": "Influence Score",
    }

    fig = go.Figure()
    # Min-max normalization across the selected subset
    normed_df = comparison_df.copy()
    for cat in available_cats:
        c_min = normed_df[cat].min()
        c_max = normed_df[cat].max()
        if c_max == c_min or pd.isna(c_max):
            normed_df[cat] = 50.0
        else:
            normed_df[cat] = ((normed_df[cat] - c_min) / (c_max - c_min) * 100).fillna(0.0)

    cat_labels = [labels_map.get(c, c) for c in available_cats]
    # Close radar loop
    cat_labels_closed = cat_labels + [cat_labels[0]]

    for i, (_, row) in enumerate(normed_df.iterrows()):
        vals = [row[c] for c in available_cats]
        vals_closed = vals + [vals[0]]
        color = THEME_COLORS[i % len(THEME_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=cat_labels_closed,
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)",
                line={"color": color, "width": 2},
                name=f"@{row['handle']}",
            )
        )

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "ticksuffix": "%",
                "showline": False,
            }
        },
        showlegend=True,
    )
    return _apply_theme(fig, "Normalized Relative Profile Comparison (0-100 Relative Scale)")
