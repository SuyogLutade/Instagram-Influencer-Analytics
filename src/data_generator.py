from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

def generate_sample_data(seed: int = 42):
    """Create realistic-looking synthetic influencer, campaign, and follower history data."""
    rng = np.random.default_rng(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    creators = [
        ("AaravFit", "Fitness", "Nano", 18500),
        ("MiraStyle", "Fashion", "Micro", 68400),
        ("TechWithRohan", "Technology", "Macro", 420000),
        ("FoodieNisha", "Food", "Micro", 52300),
        ("WanderWithKabir", "Travel", "Macro", 315000),
        ("FitBySana", "Fitness", "Micro", 74200),
        ("TheDailyDrape", "Fashion", "Macro", 268000),
        ("GadgetGuruDev", "Technology", "Micro", 91500),
        ("SpiceTrail", "Food", "Nano", 22400),
        ("RoamWithIra", "Travel", "Micro", 88700),
        ("UrbanStyleK", "Fashion", "Nano", 19800),
        ("HealthyBites", "Food", "Macro", 192000),
        ("CodeWithAnya", "Technology", "Nano", 16300),
        ("LiftWithNeil", "Fitness", "Macro", 510000),
        ("BeyondTheMap", "Travel", "Nano", 14700),
        ("StyleByTara", "Fashion", "Micro", 61900),
        ("KitchenDiaries", "Food", "Micro", 79800),
        ("NextGenTech", "Technology", "Macro", 355000),
        ("RunWithAadi", "Fitness", "Micro", 58700),
        ("PostcardPriya", "Travel", "Macro", 229000),
    ]
    rows = []
    for i, (handle, category, tier, followers) in enumerate(creators, start=1):
        engagement_base = {"Nano": 6.0, "Micro": 4.7, "Macro": 3.2}[tier]
        engagement = max(1.0, rng.normal(engagement_base, 1.0))
        growth = rng.uniform(2.0, 17.0)
        rows.append({
            "influencer_id": f"INF{i:03d}",
            "handle": handle,
            "category": category,
            "tier": tier,
            "followers": int(followers),
            "followers_90_days_ago": int(followers / (1 + growth / 100)),
            "engagement_rate": round(float(engagement), 2),
            "avg_likes": int(followers * engagement / 100 * rng.uniform(0.62, 0.82)),
            "avg_comments": int(followers * engagement / 100 * rng.uniform(0.08, 0.18)),
            "avg_shares": int(followers * engagement / 100 * rng.uniform(0.04, 0.12)),
            "avg_saves": int(followers * engagement / 100 * rng.uniform(0.05, 0.15)),
            "country": "India",
        })
    influencers = pd.DataFrame(rows)

    campaign_names = [
        "Festive Fashion Edit", "Fit For Life", "Smart Tech Launch",
        "Taste of India", "Weekend Getaways", "Monsoon Essentials",
        "Healthy Habits", "Creator Tech Week", "Summer Style Drop",
        "Discover Local", "Everyday Wellness", "Gadget Upgrade"
    ]
    brands = ["NovaWear", "Pulse Active", "ByteWorks", "SpiceBox", "TrailWays", "UrbanNest"]
    campaign_rows = []
    start = pd.Timestamp("2026-01-01")
    for i in range(1, 61):
        inf = influencers.iloc[int(rng.integers(0, len(influencers)))]
        start_date = start + pd.Timedelta(days=int(rng.integers(0, 285)))
        impressions = int(rng.integers(25000, 850000) * (0.5 + inf["followers"] / 500000))
        reach = int(impressions * rng.uniform(0.48, 0.82))
        clicks = int(reach * rng.uniform(0.012, 0.075))
        conversions = int(clicks * rng.uniform(0.025, 0.16))
        cost = round(float(rng.uniform(5000, 85000) * (0.5 + inf["followers"] / 250000)), 2)
        revenue = round(float(conversions * rng.uniform(350, 1800)), 2)
        engagements = int(reach * (inf["engagement_rate"] / 100) * rng.uniform(0.75, 1.3))
        campaign_rows.append({
            "campaign_id": f"CMP{i:03d}",
            "campaign_name": campaign_names[(i - 1) % len(campaign_names)],
            "brand": brands[(i - 1) % len(brands)],
            "influencer_id": inf["influencer_id"],
            "start_date": start_date.date().isoformat(),
            "category": inf["category"],
            "campaign_cost": cost,
            "impressions": impressions,
            "reach": reach,
            "clicks": clicks,
            "conversions": conversions,
            "engagements": engagements,
            "revenue": revenue,
            "status": str(rng.choice(["Completed", "Completed", "Completed", "Active"])),
        })
    campaigns = pd.DataFrame(campaign_rows)

    history_rows = []
    dates = pd.date_range(end=pd.Timestamp("2026-10-01"), periods=7, freq="15D")
    for _, inf in influencers.iterrows():
        start_followers = int(inf["followers_90_days_ago"])
        end_followers = int(inf["followers"])
        values = np.linspace(start_followers, end_followers, len(dates))
        for date, value in zip(dates, values):
            noise = int(rng.normal(0, max(10, value * 0.008)))
            history_rows.append({
                "influencer_id": inf["influencer_id"],
                "date": date.date().isoformat(),
                "followers": max(0, int(value) + noise),
            })
    history = pd.DataFrame(history_rows)

    influencers.to_csv(DATA_DIR / "influencers.csv", index=False)
    campaigns.to_csv(DATA_DIR / "campaigns.csv", index=False)
    history.to_csv(DATA_DIR / "follower_history.csv", index=False)
    return influencers, campaigns, history

def load_data(regenerate: bool = False):
    files = [
        DATA_DIR / "influencers.csv",
        DATA_DIR / "campaigns.csv",
        DATA_DIR / "follower_history.csv",
    ]
    if regenerate or not all(p.exists() for p in files):
        return generate_sample_data()
    influencers = pd.read_csv(files[0])
    campaigns = pd.read_csv(files[1], parse_dates=["start_date"])
    history = pd.read_csv(files[2], parse_dates=["date"])
    if not pd.api.types.is_datetime64_any_dtype(campaigns["start_date"]):
        campaigns["start_date"] = pd.to_datetime(campaigns["start_date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(history["date"]):
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
    return influencers, campaigns, history
