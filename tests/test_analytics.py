"""Unit tests for analytics, ranking, and comparison engine."""
import unittest
import numpy as np
import pandas as pd

from src.analytics import (
    calculate_kpis,
    filter_and_rank,
    compare_influencers,
    calculate_composite_score,
)


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        """Prepare sample clean test DataFrame."""
        self.sample_df = pd.DataFrame([
            {
                "handle": "mega_star",
                "followers": 100_000_000,
                "engagement_rate": 1.20,
                "avg_likes": 1_200_000,
                "new_post_avg_like": 900_000,
                "influence_score": 92.0,
                "country": "United States",
                "tier": "Mega",
                "source_rank": 1,
            },
            {
                "handle": "mid_creator",
                "followers": 200_000,
                "engagement_rate": 5.40,
                "avg_likes": 10_800,
                "new_post_avg_like": 8_500,
                "influence_score": 75.0,
                "country": "India",
                "tier": "Mid-tier",
                "source_rank": 50,
            },
            {
                "handle": "micro_niche",
                "followers": 30_000,
                "engagement_rate": 8.50,
                "avg_likes": 2_550,
                "new_post_avg_like": 2_100,
                "influence_score": 68.0,
                "country": "United States",
                "tier": "Micro",
                "source_rank": 120,
            },
        ])

    def test_calculate_kpis(self):
        """Test calculation of summary KPIs."""
        kpis = calculate_kpis(self.sample_df, ranking_metric="followers")
        self.assertEqual(kpis["total_profiles"], 3)
        self.assertEqual(kpis["total_followers"], 100_230_000)
        self.assertAlmostEqual(kpis["avg_engagement_rate"], (1.20 + 5.40 + 8.50) / 3, places=2)
        self.assertIsNotNone(kpis["top_influencer"])
        self.assertEqual(kpis["top_influencer"]["handle"], "mega_star")

    def test_kpis_empty_df(self):
        """Test KPI calculation on an empty DataFrame."""
        empty_kpis = calculate_kpis(pd.DataFrame())
        self.assertEqual(empty_kpis["total_profiles"], 0)
        self.assertEqual(empty_kpis["total_followers"], 0)
        self.assertIsNone(empty_kpis["top_influencer"])

    def test_filter_and_rank_followers(self):
        """Test sorting by followers descending."""
        ranked = filter_and_rank(self.sample_df, sort_by="followers", ascending=False)
        self.assertEqual(ranked.iloc[0]["handle"], "mega_star")
        self.assertEqual(ranked.iloc[0]["app_rank"], 1)
        self.assertEqual(ranked.iloc[2]["handle"], "micro_niche")
        self.assertEqual(ranked.iloc[2]["app_rank"], 3)

    def test_filter_and_rank_engagement(self):
        """Test sorting by engagement rate descending - smaller creator leads."""
        ranked = filter_and_rank(self.sample_df, sort_by="engagement_rate", ascending=False)
        self.assertEqual(ranked.iloc[0]["handle"], "micro_niche")
        self.assertEqual(ranked.iloc[0]["app_rank"], 1)
        self.assertEqual(ranked.iloc[2]["handle"], "mega_star")
        self.assertEqual(ranked.iloc[2]["app_rank"], 3)

    def test_filter_by_country_and_tier(self):
        """Test filtering subset by country and tier."""
        filtered = filter_and_rank(
            self.sample_df,
            selected_countries=["United States"],
            selected_tiers=["Micro"],
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["handle"], "micro_niche")

    def test_compare_influencers_leaders(self):
        """Test side-by-side comparison and accurate metric leader detection."""
        comp = compare_influencers(self.sample_df, ["mega_star", "micro_niche"])
        leaders = comp["leaders"]

        # mega_star leads in followers
        self.assertEqual(leaders["followers"]["leading_handles"], ["mega_star"])
        # micro_niche leads in engagement_rate
        self.assertEqual(leaders["engagement_rate"]["leading_handles"], ["micro_niche"])

        # Check table structure
        table = comp["comparison_table"]
        self.assertIn("Metric", table.columns)
        self.assertIn("mega_star", table.columns)
        self.assertIn("micro_niche", table.columns)
        self.assertIn("Leading Influencer", table.columns)

    def test_compare_influencers_validation(self):
        """Test comparison requires at least 2 profiles."""
        with self.assertRaises(ValueError):
            compare_influencers(self.sample_df, ["mega_star"])

    def test_composite_score_bounds(self):
        """Test composite score normalization produces scores between 0 and 100."""
        scores = calculate_composite_score(self.sample_df)
        self.assertEqual(len(scores), 3)
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 100).all())


if __name__ == "__main__":
    unittest.main()
