"""Data loader module for Instagram Influencer Analytics.

Manages loading from genuine Kaggle local CSVs, online GitHub mirror,
user file uploads, and labeled synthetic demo fallbacks.
Preserves original raw data and persists processed data separately.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import io
import pandas as pd

from src.data_cleaning import clean_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

RAW_KAGGLE_PATH = RAW_DIR / "top_insta_influencers_data.csv"
PROCESSED_KAGGLE_PATH = PROCESSED_DIR / "influencers_cleaned.csv"
SYNTHETIC_PATH = SYNTHETIC_DIR / "sample_synthetic_influencers.csv"

# Exact dataset provenance from Kaggle
KAGGLE_METADATA = {
    "title": "Top Instagram Influencers Data (Cleaned)",
    "author": "Suraj Jha (Kaggle: @surajjha051)",
    "source_url": "https://www.kaggle.com/datasets/surajjha051/top-instagram-influencers-data-cleaned",
    "mirror_url": "https://raw.githubusercontent.com/AleenaHanjgikar/instagram-influencer-analytics/main/top_insta_influencers_Cleaned_Data.csv",
    "license": "CC0: Public Domain (No copyright restrictions)",
    "collection_date": "August 2022 snapshot",
    "description": "Historical snapshot of top 200 Instagram accounts globally, recording handle, follower count, average likes, 60-day engagement rate, post count, and country origin.",
    "is_synthetic": False,
    "provenance_type": "Genuine Kaggle Dataset",
}

SYNTHETIC_METADATA = {
    "title": "Synthetic Creator Demonstration Dataset (Fallback)",
    "author": "System Generated Demo Data",
    "source_url": "Local Simulation",
    "mirror_url": "None (Local simulation)",
    "license": "Educational / Demonstration Only",
    "collection_date": "Simulated Sample",
    "description": "20 synthetic creator profiles spanning Fitness, Fashion, Tech, Food, and Travel niches for offline demonstration when Kaggle data is unavailable.",
    "is_synthetic": True,
    "provenance_type": "Synthetic Demonstration Fallback",
}


def ensure_directories():
    """Ensure data/raw, data/processed, and data/synthetic folders exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)


def init_synthetic_fallback() -> pd.DataFrame:
    """Create or load the labeled synthetic demonstration dataset."""
    ensure_directories()
    if SYNTHETIC_PATH.exists():
        return pd.read_csv(SYNTHETIC_PATH)

    # Check if existing data/influencers.csv exists and can be adapted
    legacy_path = DATA_DIR / "influencers.csv"
    if legacy_path.exists():
        df = pd.read_csv(legacy_path)
        df["is_synthetic"] = True
        df["data_provenance"] = "Synthetic Demonstration Fallback"
        df.to_csv(SYNTHETIC_PATH, index=False)
        return df

    # Minimal synthetic fallback creator records
    creators = [
        {"handle": "AaravFit", "category": "Fitness", "followers": 18500, "avg_likes": 923, "engagement_rate_raw": 6.30, "country": "India"},
        {"handle": "MiraStyle", "category": "Fashion", "followers": 68400, "avg_likes": 2132, "engagement_rate_raw": 4.83, "country": "India"},
        {"handle": "TechWithRohan", "category": "Technology", "followers": 420000, "avg_likes": 9721, "engagement_rate_raw": 3.27, "country": "India"},
        {"handle": "FoodieNisha", "category": "Food", "followers": 52300, "avg_likes": 2251, "engagement_rate_raw": 5.58, "country": "India"},
        {"handle": "WanderWithKabir", "category": "Travel", "followers": 315000, "avg_likes": 6228, "engagement_rate_raw": 2.77, "country": "India"},
        {"handle": "FitBySana", "category": "Fitness", "followers": 74200, "avg_likes": 3478, "engagement_rate_raw": 6.84, "country": "India"},
        {"handle": "TheDailyDrape", "category": "Fashion", "followers": 268000, "avg_likes": 5503, "engagement_rate_raw": 3.09, "country": "India"},
        {"handle": "GadgetGuruDev", "category": "Technology", "followers": 91500, "avg_likes": 2539, "engagement_rate_raw": 3.53, "country": "India"},
        {"handle": "SpiceTrail", "category": "Food", "followers": 22400, "avg_likes": 1015, "engagement_rate_raw": 6.87, "country": "India"},
        {"handle": "RoamWithIra", "category": "Travel", "followers": 88700, "avg_likes": 2047, "engagement_rate_raw": 3.24, "country": "India"},
    ]
    df = pd.DataFrame(creators)
    df["is_synthetic"] = True
    df["data_provenance"] = "Synthetic Demonstration Fallback"
    df.to_csv(SYNTHETIC_PATH, index=False)
    return df


def load_raw_kaggle_csv() -> pd.DataFrame:
    """Load local raw Kaggle CSV, or download from mirror if not present."""
    ensure_directories()
    if RAW_KAGGLE_PATH.exists():
        return pd.read_csv(RAW_KAGGLE_PATH, encoding_errors="replace")

    # If raw CSV not locally present, attempt download from online mirror
    url = KAGGLE_METADATA["mirror_url"]
    try:
        raw_df = pd.read_csv(url, encoding_errors="replace")
        raw_df.to_csv(RAW_KAGGLE_PATH, index=False)
        return raw_df
    except Exception as exc:
        raise FileNotFoundError(
            f"Local Kaggle raw dataset missing at {RAW_KAGGLE_PATH} and online download failed: {exc}"
        )


def load_dataset(
    source_choice: str = "kaggle_local",
    uploaded_file: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Primary entry point to load, validate, and clean an influencer dataset.

    Args:
        source_choice: One of:
            - 'kaggle_local': genuine Kaggle dataset stored locally
            - 'kaggle_online': fetch latest from online mirror
            - 'uploaded': custom user-provided CSV upload
            - 'synthetic_demo': synthetic fallback dataset
        uploaded_file: Streamlit UploadedFile or file-like object if source_choice == 'uploaded'

    Returns:
        Tuple of (cleaned_df, cleaning_report, metadata_dict)
    """
    ensure_directories()

    if source_choice == "uploaded":
        if uploaded_file is None:
            raise ValueError("No uploaded file provided.")
        filename = getattr(uploaded_file, "name", "uploaded_file.csv")
        try:
            raw_df = pd.read_csv(uploaded_file, encoding_errors="replace")
        except Exception as exc:
            raise ValueError(f"Could not parse uploaded CSV file: {exc}")

        meta = {
            "title": f"User Upload: {filename}",
            "author": "User Provided",
            "source_url": "Uploaded by user in dashboard",
            "mirror_url": "N/A",
            "license": "User Supplied",
            "collection_date": "Current Session",
            "description": f"Custom user-uploaded CSV dataset ({len(raw_df)} rows).",
            "is_synthetic": False,
            "provenance_type": "User Uploaded CSV",
        }
        cleaned_df, report = clean_dataset(
            raw_df, is_synthetic=False, dataset_name=filename
        )
        return cleaned_df, report, meta

    elif source_choice == "synthetic_demo":
        raw_df = init_synthetic_fallback()
        meta = dict(SYNTHETIC_METADATA)
        cleaned_df, report = clean_dataset(
            raw_df, is_synthetic=True, dataset_name=meta["title"]
        )
        return cleaned_df, report, meta

    elif source_choice == "kaggle_online":
        url = KAGGLE_METADATA["mirror_url"]
        raw_df = pd.read_csv(url, encoding_errors="replace")
        meta = dict(KAGGLE_METADATA)
        meta["provenance_type"] = "Genuine Kaggle Dataset (Online Mirror)"
        cleaned_df, report = clean_dataset(
            raw_df, is_synthetic=False, dataset_name=meta["title"]
        )
        # Update local processed cache
        cleaned_df.to_csv(PROCESSED_KAGGLE_PATH, index=False)
        return cleaned_df, report, meta

    else:  # 'kaggle_local' default
        # If processed version already exists and is fresh, we can also load raw and re-clean
        # To guarantee consistency, we read the local raw file and clean it
        try:
            raw_df = load_raw_kaggle_csv()
            meta = dict(KAGGLE_METADATA)
            meta["provenance_type"] = "Genuine Kaggle Dataset (Local File)"
            cleaned_df, report = clean_dataset(
                raw_df, is_synthetic=False, dataset_name=meta["title"]
            )
            # Save processed dataset
            cleaned_df.to_csv(PROCESSED_KAGGLE_PATH, index=False)
            return cleaned_df, report, meta
        except Exception as exc:
            # Fall back to synthetic with clear warning
            raw_df = init_synthetic_fallback()
            meta = dict(SYNTHETIC_METADATA)
            meta["description"] += f" (Loaded because local Kaggle data was unavailable: {exc})"
            cleaned_df, report = clean_dataset(
                raw_df, is_synthetic=True, dataset_name="Fallback Synthetic Data"
            )
            return cleaned_df, report, meta
