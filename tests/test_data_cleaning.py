"""Unit tests for data cleaning and schema validation module."""
import unittest
import numpy as np
import pandas as pd

from src.data_cleaning import (
    parse_numeric,
    detect_engagement_rate_format,
    assign_follower_tier,
    clean_dataset,
)


class TestDataCleaning(unittest.TestCase):
    def test_parse_numeric_strings(self):
        """Test numeric string parsing with commas, multipliers, and percentages."""
        self.assertEqual(parse_numeric("1,250,000"), 1250000.0)
        self.assertEqual(parse_numeric("475.8M"), 475800000.0)
        self.assertEqual(parse_numeric("65.4k"), 65400.0)
        self.assertEqual(parse_numeric("2.5B"), 2500000000.0)
        self.assertEqual(parse_numeric("1.39%"), 1.39)
        self.assertEqual(parse_numeric(12345), 12345.0)
        self.assertTrue(np.isnan(parse_numeric("N/A")))
        self.assertTrue(np.isnan(parse_numeric(None)))
        self.assertTrue(np.isnan(parse_numeric("")))

    def test_detect_engagement_rate_format(self):
        """Test detection of decimal fractions vs percentage representations."""
        # Decimal fractions like 0.0139, 0.025
        decimal_series = pd.Series([0.0139, 0.0162, 0.0124, 0.0097, 0.02])
        self.assertEqual(detect_engagement_rate_format(decimal_series), "decimal")

        # Percentage values like 1.39, 2.5, 4.8
        percentage_series = pd.Series([1.39, 1.62, 2.50, 4.83, 3.27])
        self.assertEqual(detect_engagement_rate_format(percentage_series), "percentage")

    def test_assign_follower_tier(self):
        """Test follower tier threshold assignments."""
        self.assertEqual(assign_follower_tier(5000), "Nano")
        self.assertEqual(assign_follower_tier(50000), "Micro")
        self.assertEqual(assign_follower_tier(250000), "Mid-tier")
        self.assertEqual(assign_follower_tier(750000), "Macro")
        self.assertEqual(assign_follower_tier(5000000), "Mega")
        self.assertEqual(assign_follower_tier(-100), "Unknown")

    def test_clean_dataset_with_aliases(self):
        """Test column name normalization and alias resolution."""
        raw_data = {
            "Channel Info": ["@cristiano", "kyliejenner"],
            "Followers Count": ["475.8M", "366,200,000"],
            "Avg Likes Per Post": ["8,700,000", "8.3M"],
            "60 Day Eng Rate": [0.0139, 0.0162],
            "Country Name": ["Spain", "United States"],
        }
        df = pd.DataFrame(raw_data)
        cleaned_df, report = clean_dataset(df, dataset_name="Test Raw")

        self.assertEqual(len(cleaned_df), 2)
        self.assertIn("handle", cleaned_df.columns)
        self.assertIn("followers", cleaned_df.columns)
        self.assertIn("engagement_rate", cleaned_df.columns)
        self.assertEqual(cleaned_df.iloc[0]["handle"], "cristiano")
        self.assertEqual(cleaned_df.iloc[0]["followers"], 475800000)
        # Decimal 0.0139 converted to 1.39%
        self.assertAlmostEqual(cleaned_df.iloc[0]["engagement_rate"], 1.39, places=2)
        self.assertEqual(report["duplicates_removed"], 0)

    def test_no_double_conversion_of_percentages(self):
        """Ensure engagement rates already in percentage format are not multiplied by 100 again."""
        raw_data = {
            "handle": ["AaravFit", "MiraStyle"],
            "followers": [18500, 68400],
            "avg_likes": [923, 2132],
            "engagement_rate": [6.30, 4.83],  # Already percentage
        }
        df = pd.DataFrame(raw_data)
        cleaned_df, report = clean_dataset(df)
        aarav_eng = cleaned_df.loc[cleaned_df["handle"] == "AaravFit", "engagement_rate"].iloc[0]
        mira_eng = cleaned_df.loc[cleaned_df["handle"] == "MiraStyle", "engagement_rate"].iloc[0]
        self.assertAlmostEqual(aarav_eng, 6.30, places=2)
        self.assertAlmostEqual(mira_eng, 4.83, places=2)
        self.assertEqual(report["detected_engagement_format"], "percentage")

    def test_likes_to_followers_proxy_fallback(self):
        """Test calculation of likes-to-followers engagement proxy when engagement is missing."""
        raw_data = {
            "handle": ["creator_a"],
            "followers": [100000],
            "avg_likes": [2500],
            # engagement rate omitted
        }
        df = pd.DataFrame(raw_data)
        cleaned_df, report = clean_dataset(df)
        # Proxy = (2500 / 100000) * 100 = 2.5%
        self.assertAlmostEqual(cleaned_df.iloc[0]["engagement_rate"], 2.50, places=2)
        self.assertIn("Proxy", cleaned_df.iloc[0]["engagement_metric_type"])

    def test_missing_required_columns_error(self):
        """Test that missing required columns raise an informative ValueError."""
        invalid_data = {"city": ["Paris", "Tokyo"], "score": [90, 85]}
        df = pd.DataFrame(invalid_data)
        with self.assertRaises(ValueError) as ctx:
            clean_dataset(df)
        self.assertIn("missing required column", str(ctx.exception).lower())

    def test_empty_dataframe_error(self):
        """Test that empty DataFrames raise ValueError."""
        with self.assertRaises(ValueError):
            clean_dataset(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
