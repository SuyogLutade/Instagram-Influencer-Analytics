"""Data cleaning and validation module for Instagram Influencer Analytics.

Provides reusable parsing, validation, and normalization functions that can handle
both genuine Kaggle datasets and user-uploaded CSVs with varying schemas.
"""
from typing import Any, Dict, Optional, Tuple
import re
import numpy as np
import pandas as pd


FOLLOWER_TIERS = [
    (0, 10_000, "Nano"),
    (10_000, 100_000, "Micro"),
    (100_000, 500_000, "Mid-tier"),
    (500_000, 1_000_000, "Macro"),
    (1_000_000, float("inf"), "Mega"),
]


def parse_numeric(val: Any) -> Optional[float]:
    """Safely parse numeric values from strings, handling commas, suffixes (k, m, b),
    percentages, and currency or spacing characters.

    Examples:
        '1,200,000' -> 1200000.0
        '475.8M'    -> 475800000.0
        '65.4k'     -> 65400.0
        '2.5B'      -> 2500000000.0
        '1.39%'     -> 1.39
        np.nan      -> np.nan
    """
    if pd.isna(val) or val is None:
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    s = str(val).strip().replace(",", "").replace("$", "").replace(" ", "")
    if not s or s.lower() in ("nan", "none", "null", "-", "—", "n/a", ""):
        return np.nan

    # Check for percentage string
    if s.endswith("%"):
        try:
            return float(s[:-1].strip())
        except ValueError:
            return np.nan

    # Check for multiplier suffixes
    multiplier = 1.0
    last_char = s[-1].upper()
    if last_char == "K":
        multiplier = 1e3
        s = s[:-1]
    elif last_char == "M":
        multiplier = 1e6
        s = s[:-1]
    elif last_char == "B":
        multiplier = 1e9
        s = s[:-1]

    try:
        res = float(s) * multiplier
        # Clean IEEE 754 precision noise like 65400.00000000001
        return round(res, 6) if abs(res - round(res)) < 1e-5 else res
    except ValueError:
        # Regex fallback to extract digits and decimal point
        match = re.search(r"[-+]?\d*\.?\d+", s)
        if match:
            try:
                res = float(match.group()) * multiplier
                return round(res, 6) if abs(res - round(res)) < 1e-5 else res
            except ValueError:
                return np.nan
        return np.nan


def detect_engagement_rate_format(series: pd.Series) -> str:
    """Detect whether engagement rates in a series are stored as decimals (e.g. 0.0139)
    or percentages (e.g. 1.39).

    Returns:
        'decimal' if values are formatted as fractions <= 1.0,
        'percentage' if already formatted as percentages > 1.0.
    """
    clean_series = series.dropna()
    if clean_series.empty:
        return "percentage"

    # Convert to numeric safely
    numeric_vals = pd.to_numeric(clean_series, errors="coerce").dropna()
    if numeric_vals.empty:
        return "percentage"

    # If the 75th percentile or median is <= 0.15 (15%), it is almost certainly a decimal fraction
    # Real-world Instagram engagement rates for top creators rarely exceed 15-20%.
    median_val = float(numeric_vals.median())
    p75 = float(numeric_vals.quantile(0.75))

    if median_val <= 0.20 and p75 <= 0.50:
        return "decimal"
    return "percentage"


def assign_follower_tier(followers: float) -> str:
    """Categorize an influencer into standard industry creator tiers:
    - Nano: < 10,000
    - Micro: 10,000 to 100,000
    - Mid-tier: 100,000 to 500,000
    - Macro: 500,000 to 1,000,000
    - Mega: >= 1,000,000
    """
    if pd.isna(followers) or followers < 0:
        return "Unknown"
    for low, high, label in FOLLOWER_TIERS:
        if low <= followers < high:
            return label
    return "Mega"


def clean_dataset(
    raw_df: pd.DataFrame,
    is_synthetic: bool = False,
    dataset_name: str = "Influencer Dataset",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean, normalize, and validate an influencer dataset DataFrame.

    Tolerates common schema variations from Kaggle datasets, GitHub mirrors,
    and user CSV uploads.

    Args:
        raw_df: The raw input DataFrame.
        is_synthetic: Flag indicating whether this dataset is synthetic demo data.
        dataset_name: Informational label for error reporting.

    Returns:
        Tuple of (cleaned_df, report_dict)
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("Cannot clean an empty or None DataFrame.")

    df = raw_df.copy()
    initial_rows = len(df)

    # 1. Normalize column names (lowercase, strip whitespace, replace separators)
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        for col in df.columns
    ]

    # 2. Map schema aliases
    alias_map = {
        # Username / Handle
        "channel_info": "handle",
        "username": "handle",
        "account_username": "handle",
        "influencer_name": "handle",
        "profile_name": "handle",
        "influencer": "handle",
        # Followers
        "follower_count": "followers",
        "followers_count": "followers",
        "total_followers": "followers",
        "follower": "followers",
        # Average Likes
        "avg_likes_per_post": "avg_likes",
        "average_likes": "avg_likes",
        "likes_avg": "avg_likes",
        "mean_likes": "avg_likes",
        # Engagement Rate
        "60_day_eng_rate": "engagement_rate_raw",
        "60dayeng_rate": "engagement_rate_raw",
        "sixty_day_eng_rate": "engagement_rate_raw",
        "eng_rate": "engagement_rate_raw",
        "engagement_rate": "engagement_rate_raw",
        "engagement": "engagement_rate_raw",
        # Influence score & rank
        "score": "influence_score",
        "rank": "source_rank",
        # Posts
        "posts_total": "posts",
        "total_posts": "posts",
        "post_count": "posts",
        # Recent likes
        "new_post_avg_like": "new_post_avg_like",
        "recent_avg_likes": "new_post_avg_like",
        "recent_likes": "new_post_avg_like",
        "new_post_avg_likes": "new_post_avg_like",
        # Total likes
        "tot_likes": "total_likes",
        # Country
        "country_name": "country",
        "region": "country",
        "location": "country",
        # Category
        "niche": "category",
        "topic": "category",
        "genre": "category",
    }

    # Apply aliases only when target column is not already present
    rename_dict = {}
    for old_col, new_col in alias_map.items():
        if old_col in df.columns and new_col not in df.columns:
            rename_dict[old_col] = new_col
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # 3. Validate required columns
    required_cols = ["handle", "followers"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        found_cols = ", ".join(df.columns)
        raise ValueError(
            f"The dataset '{dataset_name}' is missing required column(s): {', '.join(missing)}. "
            f"Found columns in file: [{found_cols}]. "
            "Please ensure the CSV includes at least a username/handle column and a followers column."
        )

    # 4. Clean usernames / handles
    df["handle"] = (
        df["handle"]
        .astype(str)
        .str.strip()
        .str.lstrip("@")
        .str.replace(" ", "", regex=False)
    )
    # Filter out empty, 'nan', 'none', 'unknown' handles
    invalid_mask = df["handle"].str.lower().isin(["", "nan", "none", "null", "unknown"])
    df = df[~invalid_mask].copy()

    # 5. Parse numeric columns safely
    numeric_columns = [
        "followers",
        "avg_likes",
        "new_post_avg_like",
        "total_likes",
        "posts",
        "influence_score",
        "source_rank",
        "engagement_rate_raw",
        "avg_comments",
        "avg_shares",
        "avg_saves",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_numeric)

    # Followers must be non-negative integer
    df["followers"] = df["followers"].fillna(0).clip(lower=0).round().astype("int64")

    # Drop duplicate handles, keeping record with the highest follower count
    pre_dedup = len(df)
    df = df.sort_values("followers", ascending=False).drop_duplicates(
        subset=["handle"], keep="first"
    )
    duplicates_removed = pre_dedup - len(df)

    # 6. Ensure optional columns exist with safe defaults
    optional_defaults = {
        "avg_likes": np.nan,
        "new_post_avg_like": np.nan,
        "total_likes": np.nan,
        "posts": np.nan,
        "influence_score": np.nan,
        "source_rank": np.nan,
        "category": "General / Influencer",
        "country": "Unknown",
    }
    for col, default_val in optional_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Clean country
    df["country"] = (
        df["country"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    )

    # Clean category
    df["category"] = (
        df["category"]
        .fillna("General / Influencer")
        .astype(str)
        .str.strip()
        .replace({"": "General / Influencer", "nan": "General / Influencer"})
    )

    # 7. Process Engagement Rate (handle decimal vs percent and missing values)
    detected_eng_format = "percentage"
    proxy_used_count = 0

    if "engagement_rate_raw" in df.columns and df["engagement_rate_raw"].notna().any():
        detected_eng_format = detect_engagement_rate_format(df["engagement_rate_raw"])
        raw_rates = pd.to_numeric(df["engagement_rate_raw"], errors="coerce")

        if detected_eng_format == "decimal":
            # e.g. 0.0139 -> 1.39%
            converted_rates = raw_rates * 100.0
        else:
            # Already percentage: keep as is
            converted_rates = raw_rates

        df["engagement_rate"] = converted_rates
    else:
        df["engagement_rate"] = np.nan

    # Track how engagement rate was obtained: 'Source Reported' vs 'Likes-to-Followers Proxy'
    df["engagement_metric_type"] = np.where(
        df["engagement_rate"].notna(),
        "Source Reported (60-Day)",
        "Calculated Proxy (Avg Likes / Followers)",
    )

    # Fill missing engagement rates using likes-to-followers proxy formula
    # Proxy (%) = (Average Likes / Followers) * 100
    missing_eng_mask = df["engagement_rate"].isna()
    if missing_eng_mask.any():
        proxy_rates = np.where(
            df["followers"] > 0,
            (df["avg_likes"].fillna(0) / df["followers"]) * 100.0,
            0.0,
        )
        df.loc[missing_eng_mask, "engagement_rate"] = proxy_rates[missing_eng_mask]
        proxy_used_count = int(missing_eng_mask.sum())

    # Engagement rate should not be negative
    df["engagement_rate"] = df["engagement_rate"].fillna(0.0).clip(lower=0.0).round(3)

    # 8. Follower Tier Assignment
    df["tier"] = df["followers"].apply(assign_follower_tier)

    # 9. Instagram Profile URL
    df["profile_url"] = "https://www.instagram.com/" + df["handle"] + "/"

    # 10. Origin Tracking
    df["is_synthetic"] = bool(is_synthetic)
    df["data_provenance"] = (
        "Synthetic Demonstration Fallback" if is_synthetic else "Genuine Kaggle Dataset"
    )

    # Sort primarily by followers descending
    df = df.sort_values("followers", ascending=False).reset_index(drop=True)

    report = {
        "dataset_name": dataset_name,
        "initial_rows": initial_rows,
        "cleaned_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "detected_engagement_format": detected_eng_format,
        "proxy_used_count": proxy_used_count,
        "is_synthetic": is_synthetic,
        "unique_countries": int(df["country"].nunique()),
        "columns_present": list(df.columns),
    }

    return df, report
