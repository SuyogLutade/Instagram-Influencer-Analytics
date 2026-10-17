"""Load and normalize a public Instagram influencer CSV dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_URL = (
    "https://raw.githubusercontent.com/AleenaHanjgikar/"
    "instagram-influencer-analytics/main/top_insta_influencers_Cleaned_Data.csv"
)
SOURCE_PAGE = "https://github.com/AleenaHanjgikar/instagram-influencer-analytics"


def normalize_influencers(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the source dataset and tolerate common column-name variants."""
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    aliases = {
        "channel_info": "handle", "username": "handle", "account_username": "handle",
        "followers_count": "followers", "follower_count": "followers",
        "avg_likes_per_post": "avg_likes", "average_likes": "avg_likes",
        "60_day_eng_rate": "engagement_rate_raw", "60dayeng_rate": "engagement_rate_raw",
        "sixty_day_eng_rate": "engagement_rate_raw", "engagement_rate": "engagement_rate_raw",
        "influence_score": "influence_score", "rank": "source_rank",
        "posts_total": "posts", "total_posts": "posts", "country_name": "country",
    }
    rename = {old: new for old, new in aliases.items() if old in df.columns and new not in df.columns}
    df = df.rename(columns=rename)

    required = ["handle", "followers"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    numeric_candidates = ["followers", "avg_likes", "new_post_avg_like", "total_likes", "posts", "influence_score", "source_rank", "engagement_rate_raw"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df["handle"] = df["handle"].astype(str).str.strip().str.lstrip("@").replace({"nan": "Unknown"})
    df["followers"] = df["followers"].fillna(0).clip(lower=0).astype("int64")
    if "avg_likes" not in df: df["avg_likes"] = pd.NA
    if "new_post_avg_like" not in df: df["new_post_avg_like"] = pd.NA
    if "total_likes" not in df: df["total_likes"] = pd.NA
    if "posts" not in df: df["posts"] = pd.NA
    if "influence_score" not in df: df["influence_score"] = pd.NA
    if "source_rank" not in df: df["source_rank"] = pd.NA
    if "country" not in df: df["country"] = "Unknown"
    df["country"] = df["country"].fillna("Unknown").astype(str).replace({"": "Unknown"})

    if "engagement_rate_raw" not in df:
        df["engagement_rate_raw"] = pd.NA
    # The selected GitHub dataset stores engagement as a fraction (0.0139 = 1.39%).
    # Other uploaded files may already store it as a percentage.
    rate = pd.to_numeric(df["engagement_rate_raw"], errors="coerce")
    finite = rate.dropna()
    if not finite.empty and finite.abs().median() <= 1:
        rate = rate * 100
    df["engagement_rate"] = rate
    # Derive a transparent likes-to-followers proxy only if no engagement field exists.
    fallback = (df["avg_likes"] / df["followers"].replace(0, pd.NA) * 100)
    df["engagement_rate"] = df["engagement_rate"].fillna(fallback)
    df["engagement_rate"] = df["engagement_rate"].fillna(0).clip(lower=0)
    df["tier"] = pd.cut(
        df["followers"],
        bins=[-1, 10_000, 100_000, 500_000, 1_000_000, float("inf")],
        labels=["Nano", "Micro", "Mid-tier", "Macro", "Mega"],
    ).astype(str)
    df["profile_url"] = "https://www.instagram.com/" + df["handle"].str.replace(" ", "", regex=False) + "/"
    df["data_quality_note"] = "Snapshot dataset; not live Instagram Insights"
    df = df[df["handle"].ne("Unknown")].drop_duplicates(subset=["handle"], keep="first")
    return df.sort_values("followers", ascending=False).reset_index(drop=True)


def load_online_dataset() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE_URL, encoding_errors="replace")
    return normalize_influencers(raw)


def load_local_dataset() -> pd.DataFrame:
    """Load a downloaded source CSV, or the project's sample data as a fallback."""
    real_path = DATA_DIR / "influencers_real.csv"
    if real_path.exists():
        return normalize_influencers(pd.read_csv(real_path, encoding_errors="replace"))
    sample_path = DATA_DIR / "influencers.csv"
    if sample_path.exists():
        raw = pd.read_csv(sample_path)
        if "handle" in raw and "followers" in raw:
            # Existing local demo data already stores engagement as a percentage.
            if "engagement_rate" in raw:
                raw["engagement_rate_raw"] = raw["engagement_rate"]
            return normalize_influencers(raw)
    raise FileNotFoundError("No local influencer CSV was found in the data folder.")
