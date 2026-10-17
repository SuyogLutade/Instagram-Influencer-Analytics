"""Analytics and ranking engine for Instagram Influencer Analytics.

Implements KPIs, multi-metric leaderboard rankings, side-by-side influencer
comparisons with leader detection, and optional composite scoring.
Preserves legacy campaign analytics functions for backwards compatibility.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "followers": "Followers",
    "engagement_rate": "Engagement Rate (%)",
    "avg_likes": "Average Likes per Post",
    "new_post_avg_like": "Recent Average Likes",
    "total_likes": "Total Likes",
    "posts": "Total Posts",
    "influence_score": "Source Influence Score",
    "source_rank": "Original Source Rank",
}


def calculate_kpis(
    df: pd.DataFrame, ranking_metric: str = "followers"
) -> Dict[str, Any]:
    """Calculate key performance indicators from an influencer DataFrame."""
    if df.empty:
        return {
            "total_profiles": 0,
            "total_followers": 0,
            "avg_followers": 0.0,
            "median_followers": 0.0,
            "avg_engagement_rate": 0.0,
            "avg_likes_per_post": 0.0,
            "top_influencer": None,
            "unique_countries": 0,
            "tier_distribution": {},
        }

    total_profiles = len(df)
    total_followers = int(df["followers"].sum())
    avg_followers = float(df["followers"].mean())
    median_followers = float(df["followers"].median())

    # Engagement rate mean (ignoring NaNs)
    avg_eng = (
        float(df["engagement_rate"].dropna().mean())
        if "engagement_rate" in df and df["engagement_rate"].notna().any()
        else 0.0
    )

    # Average likes per post mean (ignoring NaNs)
    avg_likes = (
        float(df["avg_likes"].dropna().mean())
        if "avg_likes" in df and df["avg_likes"].notna().any()
        else 0.0
    )

    # Top influencer under selected metric
    top_influencer = None
    if ranking_metric in df.columns and df[ranking_metric].notna().any():
        valid_df = df.dropna(subset=[ranking_metric])
        if not valid_df.empty:
            top_row = valid_df.sort_values(ranking_metric, ascending=False).iloc[0]
            top_influencer = {
                "handle": top_row["handle"],
                "value": top_row[ranking_metric],
                "followers": top_row["followers"],
                "metric_name": ranking_metric,
                "metric_label": METRIC_LABELS.get(
                    ranking_metric, ranking_metric.replace("_", " ").title()
                ),
            }

    unique_countries = int(df["country"].nunique()) if "country" in df else 0
    tier_dist = (
        df["tier"].value_counts().to_dict() if "tier" in df else {}
    )

    return {
        "total_profiles": total_profiles,
        "total_followers": total_followers,
        "avg_followers": avg_followers,
        "median_followers": median_followers,
        "avg_engagement_rate": avg_eng,
        "avg_likes_per_post": avg_likes,
        "top_influencer": top_influencer,
        "unique_countries": unique_countries,
        "tier_distribution": tier_dist,
    }


def filter_and_rank(
    df: pd.DataFrame,
    sort_by: str = "followers",
    ascending: bool = False,
    selected_countries: Optional[List[str]] = None,
    selected_tiers: Optional[List[str]] = None,
    selected_categories: Optional[List[str]] = None,
    follower_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """Filter influencer profiles and compute rankings dynamically.

    Distinguishes the dataset's original ranking ('source_rank')
    from the application-calculated ranking ('app_rank').
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # Apply Country Filter
    if selected_countries and "country" in filtered.columns:
        filtered = filtered[filtered["country"].isin(selected_countries)]

    # Apply Tier Filter
    if selected_tiers and "tier" in filtered.columns:
        filtered = filtered[filtered["tier"].isin(selected_tiers)]

    # Apply Category Filter
    if selected_categories and "category" in filtered.columns:
        filtered = filtered[filtered["category"].isin(selected_categories)]

    # Apply Follower Range Filter
    if follower_range and len(follower_range) == 2:
        low, high = follower_range
        filtered = filtered[filtered["followers"].between(low, high)]

    if filtered.empty:
        return filtered

    # Sort records
    if sort_by not in filtered.columns:
        sort_by = "followers"

    # Put NaN values last regardless of ascending/descending
    ranked = filtered.sort_values(by=sort_by, ascending=ascending, na_position="last").copy()

    # Assign calculated application rank (1, 2, 3...)
    ranked["app_rank"] = range(1, len(ranked) + 1)

    return ranked.reset_index(drop=True)


def compare_influencers(
    df: pd.DataFrame, handles: List[str]
) -> Dict[str, Any]:
    """Perform a side-by-side comparison of 2 to 5 selected influencer profiles.

    Computes leading creator for each metric, formatted comparison table,
    and analytical observations on the scale-versus-engagement trade-off.
    """
    if not handles or len(handles) < 2:
        raise ValueError("At least 2 influencer handles are required for comparison.")
    if len(handles) > 5:
        handles = handles[:5]

    selected = df[df["handle"].isin(handles)].copy()
    if len(selected) < 2:
        raise ValueError("Could not find enough matching influencer records in the dataset.")

    # Maintain user's handle selection order
    selected["sort_order"] = selected["handle"].apply(
        lambda h: handles.index(h) if h in handles else 999
    )
    selected = selected.sort_values("sort_order").drop(columns=["sort_order"])

    metrics_to_compare = [
        ("followers", "Followers", "count"),
        ("engagement_rate", "Engagement Rate (%)", "pct"),
        ("avg_likes", "Average Likes / Post", "count"),
        ("new_post_avg_like", "Recent Avg. Likes", "count"),
        ("influence_score", "Source Influence Score", "score"),
        ("posts", "Total Posts", "count"),
        ("total_likes", "Total Likes", "count"),
    ]

    leaders: Dict[str, Dict[str, Any]] = {}
    table_rows = []

    for metric_col, metric_label, metric_type in metrics_to_compare:
        if metric_col not in selected.columns:
            continue

        valid_vals = selected.dropna(subset=[metric_col])
        if valid_vals.empty:
            continue

        # Find maximum value
        max_val = valid_vals[metric_col].max()
        leading_handles = valid_vals[valid_vals[metric_col] == max_val]["handle"].tolist()

        leaders[metric_col] = {
            "metric_label": metric_label,
            "max_value": max_val,
            "leading_handles": leading_handles,
            "metric_type": metric_type,
        }

        row_data: Dict[str, Any] = {"Metric": metric_label}
        for _, row in selected.iterrows():
            val = row[metric_col]
            if pd.isna(val):
                formatted_val = "N/A"
            elif metric_type == "pct":
                formatted_val = f"{val:.2f}%"
            elif metric_type == "score":
                formatted_val = f"{val:.0f}"
            else:
                formatted_val = f"{val:,.0f}" if val >= 1000 else f"{val}"
            row_data[row["handle"]] = formatted_val

        # Add winner column
        row_data["Leading Influencer"] = ", ".join(leading_handles)
        table_rows.append(row_data)

    comparison_table = pd.DataFrame(table_rows)

    # Generate scale vs engagement observation
    insights = []
    top_follower_creator = selected.sort_values("followers", ascending=False).iloc[0]
    top_eng_creator = selected.sort_values("engagement_rate", ascending=False).iloc[0]

    if top_follower_creator["handle"] != top_eng_creator["handle"]:
        fol_ratio = (
            top_follower_creator["followers"] / max(1, top_eng_creator["followers"])
        )
        eng_ratio = (
            top_eng_creator["engagement_rate"] / max(0.01, top_follower_creator["engagement_rate"])
        )
        insights.append(
            f"**Scale vs. Engagement Divergence:** @{top_follower_creator['handle']} has "
            f"**{fol_ratio:.1f}x** more followers, but @{top_eng_creator['handle']} achieves "
            f"a **{eng_ratio:.1f}x higher engagement rate** ({top_eng_creator['engagement_rate']:.2f}% vs {top_follower_creator['engagement_rate']:.2f}%)."
        )
    else:
        insights.append(
            f"@{top_follower_creator['handle']} uniquely dominates both maximum audience reach and highest engagement rate among the selected group."
        )

    return {
        "selected_profiles": selected,
        "comparison_table": comparison_table,
        "leaders": leaders,
        "insights": insights,
        "metrics_evaluated": [m[0] for m in metrics_to_compare if m[0] in selected.columns],
    }


def calculate_composite_score(
    df: pd.DataFrame,
    weight_followers: float = 0.35,
    weight_engagement: float = 0.45,
    weight_influence: float = 0.20,
) -> pd.Series:
    """Calculate an optional transparent Composite Influencer Score (0-100 scale).

    Formula:
        Score = w_fol * Norm(Followers) + w_eng * Norm(Engagement) + w_inf * Norm(Influence Score)

    Note:
        This is an educational multi-criteria index for demonstration,
        NOT an official Instagram industry standard.
    """
    if df.empty:
        return pd.Series(dtype=float)

    def min_max_norm(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        valid = s.dropna()
        if valid.empty:
            return pd.Series(50.0, index=series.index)
        low, high = valid.min(), valid.max()
        if high == low:
            return pd.Series(50.0, index=series.index)
        return ((s - low) / (high - low) * 100.0).fillna(0.0)

    # Normalize weights so sum == 1.0
    total_w = weight_followers + weight_engagement + weight_influence
    if total_w <= 0:
        total_w = 1.0
    w_fol = weight_followers / total_w
    w_eng = weight_engagement / total_w
    w_inf = weight_influence / total_w

    norm_fol = min_max_norm(df["followers"])
    norm_eng = min_max_norm(df["engagement_rate"])

    if "influence_score" in df and df["influence_score"].notna().any():
        norm_inf = min_max_norm(df["influence_score"])
    else:
        norm_inf = pd.Series(50.0, index=df.index)

    composite = (w_fol * norm_fol + w_eng * norm_eng + w_inf * norm_inf).round(1)
    return composite


# ---------------------------------------------------------
# Legacy Campaign Analytics Functions (Preserved)
# ---------------------------------------------------------
def enrich_influencers(
    influencers: pd.DataFrame, campaigns: pd.DataFrame
) -> pd.DataFrame:
    """Legacy function: merge campaign historical returns into synthetic influencer records."""
    df = influencers.copy()
    if "followers_90_days_ago" in df.columns:
        df["follower_growth_pct"] = np.where(
            df["followers_90_days_ago"] > 0,
            (df["followers"] - df["followers_90_days_ago"])
            / df["followers_90_days_ago"]
            * 100,
            0,
        )
    else:
        df["follower_growth_pct"] = 0.0

    if campaigns is None or campaigns.empty:
        return df

    campaign_stats = campaigns.groupby("influencer_id", as_index=False).agg(
        campaign_count=("campaign_id", "count"),
        total_campaign_cost=("campaign_cost", "sum"),
        total_revenue=("revenue", "sum"),
        total_conversions=("conversions", "sum"),
        total_reach=("reach", "sum"),
        total_engagements=("engagements", "sum"),
        total_clicks=("clicks", "sum"),
    )
    df = df.merge(campaign_stats, on="influencer_id", how="left")
    numeric_cols = [
        "campaign_count",
        "total_campaign_cost",
        "total_revenue",
        "total_conversions",
        "total_reach",
        "total_engagements",
        "total_clicks",
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df["campaign_roi_pct"] = np.where(
        df["total_campaign_cost"] > 0,
        (df["total_revenue"] - df["total_campaign_cost"])
        / df["total_campaign_cost"]
        * 100,
        0,
    )
    df["cost_per_engagement"] = np.where(
        df["total_engagements"] > 0,
        df["total_campaign_cost"] / df["total_engagements"],
        0,
    )

    def norm(series):
        low, high = series.min(), series.max()
        if pd.isna(low) or pd.isna(high) or high == low:
            return pd.Series(np.full(len(series), 50.0), index=series.index)
        return (series - low) / (high - low) * 100

    df["effectiveness_score"] = (
        0.35 * norm(df["engagement_rate"])
        + 0.25 * norm(df["follower_growth_pct"])
        + 0.40 * norm(df["campaign_roi_pct"])
    ).round(1)
    return df


def campaign_metrics(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Legacy function: compute conversion and ROI rates for campaigns."""
    df = campaigns.copy()
    df["engagement_rate_pct"] = np.where(
        df["reach"] > 0, df["engagements"] / df["reach"] * 100, 0
    )
    df["click_through_rate_pct"] = np.where(
        df["impressions"] > 0, df["clicks"] / df["impressions"] * 100, 0
    )
    df["conversion_rate_pct"] = np.where(
        df["clicks"] > 0, df["conversions"] / df["clicks"] * 100, 0
    )
    df["cost_per_engagement"] = np.where(
        df["engagements"] > 0, df["campaign_cost"] / df["engagements"], 0
    )
    df["roi_pct"] = np.where(
        df["campaign_cost"] > 0,
        (df["revenue"] - df["campaign_cost"]) / df["campaign_cost"] * 100,
        0,
    )
    return df
