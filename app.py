"""Instagram Influencer Analytics System (InfluencerIQ).

A comprehensive Streamlit analytics platform for exploring Instagram influencer
metrics, leaderboards, comparisons, and methodological transparency using real Kaggle datasets.
"""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import (
    load_dataset,
    KAGGLE_METADATA,
    SYNTHETIC_METADATA,
    RAW_KAGGLE_PATH,
    PROCESSED_KAGGLE_PATH,
)
from src.analytics import (
    calculate_kpis,
    filter_and_rank,
    compare_influencers,
    calculate_composite_score,
    METRIC_LABELS,
)
from src.visualization import (
    plot_top_n_bar,
    plot_followers_vs_engagement,
    plot_tier_distribution,
    plot_country_distribution,
    plot_metric_distribution,
    plot_comparison_bar,
    plot_normalized_comparison_radar,
)

# Page configuration
st.set_page_config(
    page_title="InfluencerIQ | Instagram Analytics System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Executive Aesthetic CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
}
.hero-banner {
    padding: 1.5rem 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #0e7490 100%);
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15);
}
.hero-banner h1 {
    margin: 0 0 0.3rem 0;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
}
.hero-banner p {
    margin: 0;
    font-size: 1.05rem;
    opacity: 0.92;
    color: #e0e7ff;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
}
.kpi-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}
.kpi-subtitle {
    font-size: 0.82rem;
    color: #059669;
    margin-top: 0.35rem;
    font-weight: 500;
}
.tier-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}
.tier-mega { background-color: #ede9fe; color: #6d28d9; }
.tier-macro { background-color: #fef3c7; color: #b45309; }
.tier-mid { background-color: #d1fae5; color: #065f46; }
.tier-micro { background-color: #e0f2fe; color: #0369a1; }
.tier-nano { background-color: #f1f5f9; color: #475569; }
.leader-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border-left: 4px solid #6366f1;
    padding: 0.85rem 1.1rem;
    border-radius: 8px;
    margin-bottom: 0.7rem;
}
.method-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


# Formatting Helpers
def format_number(val) -> str:
    if pd.isna(val) or val is None:
        return "—"
    num = float(val)
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if num >= 1e6:
        return f"{num / 1e6:.1f}M"
    if num >= 1e3:
        return f"{num / 1e3:.1f}K"
    return f"{num:,.0f}"


def format_exact(val) -> str:
    if pd.isna(val) or val is None:
        return "—"
    return f"{float(val):,.0f}"


# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    source_label_map = {
        "kaggle_local": "Genuine Kaggle Dataset (Local)",
        "kaggle_online": "Genuine Kaggle Dataset (Online Mirror)",
        "uploaded": "Upload Custom CSV File",
        "synthetic_demo": "Synthetic Fallback Dataset",
    }
    source_choice = st.radio(
        "Data Source",
        options=["kaggle_local", "kaggle_online", "uploaded", "synthetic_demo"],
        format_func=lambda x: source_label_map[x],
        index=0,
        help="Select between the genuine Kaggle snapshot, online GitHub mirror, custom CSV upload, or synthetic demo.",
    )

    uploaded_file = None
    if source_choice == "uploaded":
        uploaded_file = st.file_uploader(
            "Upload influencer CSV",
            type=["csv"],
            help="Upload a CSV with influencer metrics. Must have a handle/username column and followers.",
        )

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    nav_pages = [
        "📊 Overview Dashboard",
        "🔍 Influencer Explorer",
        "🏆 Rankings & Leaderboard",
        "⚖️ Influencer Comparison",
        "📘 How It Works",
        "💾 Data Management & Export",
    ]
    selected_page = st.radio("Go to Page", nav_pages, index=0)


# Data Loading with Streamlit Cache
@st.cache_data(show_spinner=False, ttl=1800)
def get_cached_dataset(source_key: str, file_bytes: bytes = None, file_name: str = None):
    if source_key == "uploaded" and file_bytes is not None:
        import io
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = file_name or "uploaded.csv"
        return load_dataset("uploaded", uploaded_file=file_obj)
    return load_dataset(source_key)


# Attempt Data Load
try:
    if source_choice == "uploaded":
        if uploaded_file is None:
            st.info("👋 Please upload a CSV file in the sidebar to begin analyzing your data.")
            st.stop()
        bytes_data = uploaded_file.getvalue()
        influencers_df, cleaning_report, dataset_meta = get_cached_dataset(
            "uploaded", file_bytes=bytes_data, file_name=uploaded_file.name
        )
    else:
        with st.spinner("Loading influencer dataset…"):
            influencers_df, cleaning_report, dataset_meta = get_cached_dataset(source_choice)
except Exception as exc:
    st.error(f"❌ Failed to load dataset: {exc}")
    st.warning(
        "You can switch to the local genuine Kaggle CSV, upload a CSV file, or use the synthetic fallback demo."
    )
    st.stop()

if influencers_df.empty:
    st.error("No usable records were found in the selected dataset.")
    st.stop()


# Global Sidebar Filters
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 Global Filters")

    # Countries Filter
    all_countries = sorted(influencers_df["country"].dropna().unique().tolist())
    selected_countries = st.multiselect(
        "Filter by Country",
        options=all_countries,
        default=all_countries,
        help="Filter profiles by their registered country.",
    )
    if not selected_countries:
        selected_countries = all_countries

    # Tiers Filter
    all_tiers = ["Nano", "Micro", "Mid-tier", "Macro", "Mega"]
    available_tiers = [t for t in all_tiers if t in influencers_df["tier"].unique()]
    selected_tiers = st.multiselect(
        "Filter by Follower Tier",
        options=available_tiers,
        default=available_tiers,
        help="Segment creators by audience tier.",
    )
    if not selected_tiers:
        selected_tiers = available_tiers

    # Follower Range Slider
    f_min = int(influencers_df["followers"].min())
    f_max = int(influencers_df["followers"].max())
    if f_min < f_max:
        follower_range = st.slider(
            "Follower Count Range",
            min_value=f_min,
            max_value=f_max,
            value=(f_min, f_max),
            format="%d",
        )
    else:
        follower_range = (f_min, f_max)

    st.markdown("---")
    # Dataset Provenance Badge
    is_synth = dataset_meta.get("is_synthetic", False)
    if is_synth:
        st.warning("⚠️ **Note:** Active data is a **Synthetic Demonstration Fallback**.")
    else:
        st.success("✅ **Dataset:** Genuine Kaggle Snapshot (Suraj Jha / CC0).")
    st.caption(f"Loaded: **{len(influencers_df):,}** profiles")


# Apply Global Filters
filtered_df = filter_and_rank(
    influencers_df,
    sort_by="followers",
    ascending=False,
    selected_countries=selected_countries,
    selected_tiers=selected_tiers,
    follower_range=follower_range,
)


# Header Banner
st.markdown(f"""
<div class="hero-banner">
    <h1>📊 InfluencerIQ</h1>
    <p>Empirical Instagram Influencer Analytics & Performance Evaluation System</p>
</div>
""", unsafe_allow_html=True)

# Data Provenance Alert bar
prov_col1, prov_col2 = st.columns([3, 1])
with prov_col1:
    if is_synth:
        st.info("⚠️ Currently displaying **Synthetic Demo Data** for demonstration purposes.")
    else:
        st.caption(
            f"📍 **Source:** [{dataset_meta.get('title')}]({dataset_meta.get('source_url', '#')}) · "
            f"Author: **{dataset_meta.get('author')}** · License: **{dataset_meta.get('license')}** · "
            "Snapshot data (not a real-time Instagram API stream)."
        )
with prov_col2:
    st.caption(f"Showing **{len(filtered_df):,}** of **{len(influencers_df):,}** profiles")


# =========================================================
# PAGE 1: OVERVIEW DASHBOARD
# =========================================================
if selected_page == "📊 Overview Dashboard":
    st.subheader("Executive Influencer Landscape")

    if filtered_df.empty:
        st.warning("No influencer profiles match the selected filters. Please expand your filter criteria.")
    else:
        # Dynamic Ranking metric selector for KPI card
        kpi_metric_choice = st.selectbox(
            "Select Benchmark Metric for Leader Highlight:",
            options=["followers", "engagement_rate", "avg_likes", "influence_score"],
            format_func=lambda x: METRIC_LABELS.get(x, x),
            index=0,
            key="kpi_metric_select",
        )
        kpis = calculate_kpis(filtered_df, ranking_metric=kpi_metric_choice)

        # KPI Row
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Profiles</div>
                <div class="kpi-value">{kpis['total_profiles']:,}</div>
                <div class="kpi-subtitle">Active in filter</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Followers</div>
                <div class="kpi-value">{format_number(kpis['total_followers'])}</div>
                <div class="kpi-subtitle">Combined reach</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg. Followers</div>
                <div class="kpi-value">{format_number(kpis['avg_followers'])}</div>
                <div class="kpi-subtitle">Median: {format_number(kpis['median_followers'])}</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg. Engagement</div>
                <div class="kpi-value">{kpis['avg_engagement_rate']:.2f}%</div>
                <div class="kpi-subtitle">Per post interaction</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Avg. Likes / Post</div>
                <div class="kpi-value">{format_number(kpis['avg_likes_per_post'])}</div>
                <div class="kpi-subtitle">Historical average</div>
            </div>
            """, unsafe_allow_html=True)
        with k6:
            top_inf = kpis["top_influencer"]
            top_name = f"@{top_inf['handle']}" if top_inf else "N/A"
            top_val = (
                f"{top_inf['value']:.2f}%"
                if top_inf and top_inf["metric_name"] == "engagement_rate"
                else format_number(top_inf["value"]) if top_inf else "—"
            )
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">#1 by {top_inf['metric_label'] if top_inf else ''}</div>
                <div class="kpi-value" style="font-size:1.4rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{top_name}</div>
                <div class="kpi-subtitle">{top_val}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1: Top 10 Followers & Top 10 Engagement
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_top_fol = plot_top_n_bar(
                filtered_df, metric="followers", top_n=10, title="Top 10 Influencers by Audience Scale (Followers)"
            )
            st.plotly_chart(fig_top_fol, use_container_width=True)

        with chart_col2:
            fig_top_eng = plot_top_n_bar(
                filtered_df, metric="engagement_rate", top_n=10, title="Top 10 Influencers by Engagement Rate (%)"
            )
            st.plotly_chart(fig_top_eng, use_container_width=True)

        # Charts Row 2: Scale vs Engagement & Follower Tiers
        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            fig_scatter = plot_followers_vs_engagement(filtered_df, log_scale=True)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with chart_col4:
            fig_tiers = plot_tier_distribution(filtered_df)
            st.plotly_chart(fig_tiers, use_container_width=True)

        # Charts Row 3: Country Breakdown & Engagement Rate Distribution
        chart_col5, chart_col6 = st.columns(2)
        with chart_col5:
            fig_countries = plot_country_distribution(filtered_df, top_n=10)
            st.plotly_chart(fig_countries, use_container_width=True)

        with chart_col6:
            fig_dist = plot_metric_distribution(filtered_df, metric="engagement_rate")
            st.plotly_chart(fig_dist, use_container_width=True)


# =========================================================
# PAGE 2: INFLUENCER EXPLORER
# =========================================================
elif selected_page == "🔍 Influencer Explorer":
    st.subheader("Individual Influencer Profile Explorer")

    if filtered_df.empty:
        st.warning("No profiles match the active filters.")
    else:
        handles_list = sorted(filtered_df["handle"].unique().tolist())
        selected_handle = st.selectbox(
            "Search or select an Instagram handle:",
            options=handles_list,
            index=0,
            help="Type to search through available creator handles.",
        )

        profile = filtered_df[filtered_df["handle"] == selected_handle].iloc[0]

        # Profile Summary Header
        p_head1, p_head2 = st.columns([3, 1])
        with p_head1:
            st.markdown(f"## [@{profile['handle']}]({profile['profile_url']})")
            tier_class = f"tier-{profile['tier'].lower()}" if profile['tier'] in ["Mega", "Macro", "Nano", "Micro"] else "tier-mid"
            st.markdown(
                f"<span class='tier-badge {tier_class}'>{profile['tier']} Tier</span> · "
                f"**Origin:** {profile['country']} · "
                f"**Category:** {profile.get('category', 'General / Influencer')} · "
                f"**Engagement Source:** {profile.get('engagement_metric_type', 'Source 60-Day')}",
                unsafe_allow_html=True,
            )
        with p_head2:
            st.link_button("🌐 Open Instagram Profile", profile["profile_url"])

        st.markdown("<br>", unsafe_allow_html=True)

        # Profile Metric Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total Followers", format_number(profile["followers"]), f"{profile['followers']:,} exact")
        with m2:
            st.metric("Engagement Rate", f"{profile['engagement_rate']:.2f}%")
        with m3:
            st.metric("Avg Likes / Post", format_number(profile["avg_likes"]) if pd.notna(profile["avg_likes"]) else "N/A")
        with m4:
            st.metric("Recent Avg Likes", format_number(profile["new_post_avg_like"]) if pd.notna(profile["new_post_avg_like"]) else "N/A")
        with m5:
            score_val = f"{profile['influence_score']:.0f}" if pd.notna(profile.get("influence_score")) else "N/A"
            st.metric("Influence Score", score_val)

        st.markdown("---")

        # Visual Metrics Breakdown
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("#### Performance Metrics Breakdown")
            metrics_dict = {
                "Followers": profile["followers"],
                "Average Likes": profile["avg_likes"],
                "Recent Avg Likes": profile["new_post_avg_like"],
                "Total Posts": profile["posts"] if pd.notna(profile["posts"]) else 0,
            }
            m_df = pd.DataFrame([
                {"Metric": k, "Value": float(v) if pd.notna(v) else 0.0}
                for k, v in metrics_dict.items()
            ])
            import plotly.express as px
            fig_prof = px.bar(
                m_df,
                x="Metric",
                y="Value",
                color="Metric",
                text_auto=".2s",
                title=f"Volume Metrics for @{profile['handle']}",
            )
            fig_prof.update_layout(showlegend=False, yaxis_type="log")
            st.plotly_chart(fig_prof, use_container_width=True)
            st.caption("Log scale applied to display metrics of vastly different magnitudes.")

        with v_col2:
            st.markdown("#### Benchmark Against Dataset Average")
            tier_profiles = influencers_df[influencers_df["tier"] == profile["tier"]]
            tier_avg_eng = tier_profiles["engagement_rate"].mean()
            dataset_avg_eng = influencers_df["engagement_rate"].mean()

            diff_tier = profile["engagement_rate"] - tier_avg_eng
            diff_all = profile["engagement_rate"] - dataset_avg_eng

            bench_data = pd.DataFrame({
                "Benchmark": [f"@{profile['handle']}", f"{profile['tier']} Tier Avg", "Overall Dataset Avg"],
                "Engagement Rate (%)": [profile["engagement_rate"], tier_avg_eng, dataset_avg_eng],
            })
            fig_bench = px.bar(
                bench_data,
                x="Benchmark",
                y="Engagement Rate (%)",
                color="Benchmark",
                text_auto=".2f",
                title=f"Engagement Comparison: @{profile['handle']} vs Benchmarks",
            )
            fig_bench.update_layout(showlegend=False)
            st.plotly_chart(fig_bench, use_container_width=True)

            if diff_tier >= 0:
                st.success(f"✨ @{profile['handle']} has a **+{diff_tier:.2f}% higher engagement rate** than the {profile['tier']} tier average.")
            else:
                st.info(f"ℹ️ @{profile['handle']} is **{abs(diff_tier):.2f}% below** the {profile['tier']} tier average.")

        st.caption(
            "⚠️ **Methodology Note:** This dataset represents a historical snapshot. "
            "No historical follower trajectory or growth curves are fabricated when only single-point snapshot data is provided."
        )


# =========================================================
# PAGE 3: RANKINGS AND LEADERBOARD
# =========================================================
elif selected_page == "🏆 Rankings & Leaderboard":
    st.subheader("Influencer Rankings & Competitive Leaderboard")
    st.markdown(
        "Sort and benchmark influencer profiles across multiple performance indicators. "
        "Distinguishes between **Application Calculated Rank** and the dataset's **Original Source Rank**."
    )

    if filtered_df.empty:
        st.warning("No profiles match the active filters.")
    else:
        rank_ctrl1, rank_ctrl2, rank_ctrl3 = st.columns([2, 1, 1])
        with rank_ctrl1:
            sort_metric = st.selectbox(
                "Rank By Metric:",
                options=["followers", "engagement_rate", "avg_likes", "new_post_avg_like", "influence_score"],
                format_func=lambda x: METRIC_LABELS.get(x, x),
                index=0,
            )
        with rank_ctrl2:
            sort_direction = st.radio("Sort Order:", options=["Descending (Highest First)", "Ascending (Lowest First)"], index=0)
            is_asc = sort_direction.startswith("Ascending")
        with rank_ctrl3:
            st.metric("Matching Influencers", f"{len(filtered_df):,}")

        # Compute dynamic leaderboard
        leaderboard_df = filter_and_rank(
            filtered_df,
            sort_by=sort_metric,
            ascending=is_asc,
        )

        # Principle reminder
        st.info(
            "💡 **Analytical Principle:** An influencer with the highest follower count is not necessarily the best-performing creator. "
            "Smaller niche creators often yield significantly higher engagement rates, resulting in superior audience resonance."
        )

        # Prepare Clean Display Table
        display_cols = [
            "app_rank",
            "source_rank",
            "handle",
            "tier",
            "followers",
            "engagement_rate",
            "avg_likes",
            "new_post_avg_like",
            "influence_score",
            "country",
        ]
        available_display = [c for c in display_cols if c in leaderboard_df.columns]
        view_df = leaderboard_df[available_display].copy()

        view_df = view_df.rename(columns={
            "app_rank": "App Rank",
            "source_rank": "Source Rank",
            "handle": "Username / Handle",
            "tier": "Tier",
            "followers": "Followers",
            "engagement_rate": "Engagement Rate (%)",
            "avg_likes": "Avg Likes / Post",
            "new_post_avg_like": "Recent Avg Likes",
            "influence_score": "Source Influence Score",
            "country": "Country",
        })

        st.dataframe(
            view_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "App Rank": st.column_config.NumberColumn("Calculated Rank", format="#%d"),
                "Source Rank": st.column_config.NumberColumn("Original Source Rank", format="#%d"),
                "Followers": st.column_config.NumberColumn("Followers", format="%d"),
                "Engagement Rate (%)": st.column_config.NumberColumn("Engagement Rate (%)", format="%.2f%%"),
                "Avg Likes / Post": st.column_config.NumberColumn("Avg Likes / Post", format="%d"),
                "Recent Avg Likes": st.column_config.NumberColumn("Recent Avg Likes", format="%d"),
                "Source Influence Score": st.column_config.NumberColumn("Influence Score", format="%.0f"),
            },
        )

        # Download Leaderboard CSV
        csv_data = view_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Download Current Leaderboard as CSV",
            data=csv_data,
            file_name=f"influencer_leaderboard_by_{sort_metric}.csv",
            mime="text/csv",
        )


# =========================================================
# PAGE 4: INFLUENCER COMPARISON
# =========================================================
elif selected_page == "⚖️ Influencer Comparison":
    st.subheader("Side-by-Side Influencer Comparison")
    st.markdown(
        "Select between **2 and 5 creator profiles** to compare audience scale, engagement depth, and interaction efficiency side-by-side."
    )

    if filtered_df.empty:
        st.warning("No profiles match the active filters.")
    else:
        available_handles = filtered_df["handle"].tolist()
        default_handles = available_handles[:min(3, len(available_handles))]

        selected_handles = st.multiselect(
            "Select 2 to 5 Influencers to Compare:",
            options=available_handles,
            default=default_handles,
            max_selections=5,
            help="Choose up to 5 profiles for direct comparative analysis.",
        )

        if len(selected_handles) < 2:
            st.info("👋 Please select at least **2 profiles** above to generate the comparative analysis.")
        else:
            try:
                comp_results = compare_influencers(filtered_df, selected_handles)
            except Exception as exc:
                st.error(f"Error computing comparison: {exc}")
                st.stop()

            # Comparative Insight Callout
            for insight in comp_results["insights"]:
                st.markdown(f"""
                <div class="leader-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)

            # Metric Leader Highlights
            st.markdown("#### 🥇 Metric Leaders")
            lead_cols = st.columns(min(4, len(comp_results["leaders"])))
            col_idx = 0
            for metric_key, leader_info in comp_results["leaders"].items():
                if metric_key in ["followers", "engagement_rate", "avg_likes", "influence_score"]:
                    with lead_cols[col_idx % len(lead_cols)]:
                        formatted_lead_val = (
                            f"{leader_info['max_value']:.2f}%"
                            if leader_info["metric_type"] == "pct"
                            else format_number(leader_info["max_value"])
                        )
                        st.markdown(f"""
                        <div class="kpi-card" style="padding:0.9rem;">
                            <div class="kpi-title">{leader_info['metric_label']} Leader</div>
                            <div class="kpi-value" style="font-size:1.3rem;">@{leader_info['leading_handles'][0]}</div>
                            <div class="kpi-subtitle">{formatted_lead_val}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        col_idx += 1

            st.markdown("<br>", unsafe_allow_html=True)

            # Comparison Table
            st.markdown("#### 📋 Detailed Metrics Matrix")
            st.dataframe(
                comp_results["comparison_table"],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            # Interactive Charts
            st.markdown("#### 📊 Comparative Visualizations")
            c_tab1, c_tab2, c_tab3 = st.tabs([
                "Individual Metric Comparison",
                "Normalized Radar Profile (0-100 Scale)",
                "Scale vs Engagement Matrix",
            ])

            with c_tab1:
                chart_metric = st.selectbox(
                    "Select metric to chart:",
                    options=["followers", "engagement_rate", "avg_likes", "new_post_avg_like", "influence_score"],
                    format_func=lambda x: METRIC_LABELS.get(x, x),
                    index=0,
                )
                fig_comp_bar = plot_comparison_bar(comp_results["selected_profiles"], metric=chart_metric)
                st.plotly_chart(fig_comp_bar, use_container_width=True)

            with c_tab2:
                fig_radar = plot_normalized_comparison_radar(
                    comp_results["selected_profiles"],
                    metrics=["followers", "engagement_rate", "avg_likes", "new_post_avg_like", "influence_score"],
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                st.caption(
                    "📌 **Why Normalization Matters:** Comparing followers (millions) and engagement rate (percentage) on a shared axis causes distortions. "
                    "Min-max normalization rescales every metric to a relative 0–100% index across the selected creators."
                )

            with c_tab3:
                fig_sub_scatter = plot_followers_vs_engagement(comp_results["selected_profiles"], log_scale=True)
                st.plotly_chart(fig_sub_scatter, use_container_width=True)


# =========================================================
# PAGE 5: HOW IT WORKS (METHODOLOGY)
# =========================================================
elif selected_page == "📘 How It Works":
    st.subheader("📘 System Architecture & Analysis Methodology")
    st.markdown(
        "A transparent, beginner-friendly guide explaining how InfluencerIQ loads data, cleans anomalies, "
        "calculates metrics, and evaluates creators without fabricating statistics."
    )

    # Section A: Data Source
    with st.expander("Section A: Data Source & Origin", expanded=True):
        st.markdown(f"""
        ### Where Does the Dataset Come From?
        InfluencerIQ integrates public benchmark data from **Kaggle**:
        - **Dataset Title:** `{dataset_meta.get('title')}`
        - **Author / Contributor:** `{dataset_meta.get('author')}`
        - **Kaggle URL:** [{dataset_meta.get('source_url')}]({dataset_meta.get('source_url')})
        - **License:** `{dataset_meta.get('license')}`
        - **Collection Date:** `{dataset_meta.get('collection_date')}`

        ### Columns Loaded in Memory:
        The active dataset contains **{len(influencers_df):,} records** with the following fields:
        """)
        cols_summary = pd.DataFrame({
            "Column Name": influencers_df.columns,
            "Non-Null Count": [int(influencers_df[c].notna().sum()) for c in influencers_df.columns],
            "Data Type": [str(influencers_df[c].dtype) for c in influencers_df.columns],
            "Sample Value": [str(influencers_df[c].iloc[0]) if len(influencers_df) else "N/A" for c in influencers_df.columns],
        })
        st.dataframe(cols_summary, use_container_width=True, hide_index=True)

        st.markdown("""
        ### Snapshot vs. Live Instagram Feed
        - **Historical Snapshot:** The dataset captures influencer statistics as recorded during the data collection period (2022).
        - **No Private Insights:** Metrics such as post reach, impressions, story saves, and audience demographics are private to account owners. The system analyzes only public signals.
        - **No Fabricated Trajectory:** Because this dataset is a snapshot without multi-timestamp records, we deliberately do not invent artificial historical follower charts.
        """)

    # Section B: Data Processing Workflow
    with st.expander("Section B: Data Processing Workflow", expanded=False):
        st.markdown("""
        ### End-to-End Pipeline
        InfluencerIQ implements a modular 7-step pipeline:
        """)
        st.markdown("""
```mermaid
graph LR
    A[Raw Kaggle CSV] --> B[Data Validation]
    B --> C[Data Cleaning & Aliasing]
    C --> D[Metric Calculation & Proxy]
    D --> E[Interactive Filtering]
    E --> F[Rankings & Comparisons]
    F --> G[Interactive Visualizations]
```
        """)
        st.markdown("""
        1. **Raw Kaggle CSV:** Loaded from `data/raw/` or fetched from the verified mirror.
        2. **Data Validation:** Verifies that required columns (`handle` and `followers`) exist; alerts the user if schema is unsupported.
        3. **Data Cleaning:** Normalizes column headers, removes `@` and spacing, parses numeric strings (`475.8M`, `65k`, commas), and eliminates duplicate usernames.
        4. **Metric Calculation:** Detects decimal vs percentage engagement rates, fills missing rates via likes-to-followers proxies, and categorizes follower tiers.
        5. **Interactive Filtering:** Dynamically slices records by country, tier, and audience scale.
        6. **Rankings & Comparison:** Computes relative leaderboards and side-by-side matrices.
        7. **Visualizations:** Generates isolated-scale Plotly charts and normalized radar plots.
        """)

    # Section C: Engagement Rate Calculation
    with st.expander("Section C: Engagement Rate Calculation & Conversion", expanded=False):
        st.markdown("""
        ### 1. Source Engagement Rate (When Supplied)
        When an engagement rate is included in the source dataset, it typically measures interaction over a 60-day period:
        $$\\text{Engagement Rate} = \\text{Source 60-Day Metric}$$

        ### 2. Likes-to-Followers Proxy Formula
        If engagement rate is missing for a creator (or when only likes and followers are present), we compute a transparent proxy:
        $$\\text{Likes-to-Followers Rate (\\%)} = \\left( \\frac{\\text{Average Likes}}{\\text{Followers}} \\right) \\times 100$$
        *Note: This is labeled as a likes-based engagement proxy rather than complete interaction rate (which would include comments and shares).*

        ### 3. Decimal vs. Percentage Conversion
        Different data sources format engagement rates differently. For example:
        - Fractional format: `0.025` represents `2.5%`.
        - Percentage format: `2.50` already represents `2.5%`.

        **Automated Detection:**
        Our cleaning algorithm inspects the distribution. If the 75th percentile is $\\le 0.20$ (20%), the series is recognized as fractional and multiplied by 100 once.
        If values are already $\\ge 1.0$, they are preserved as-is, preventing double-multiplication bugs.
        """)

    # Section D: Follower Tiers
    with st.expander("Section D: Creator Follower Tiers", expanded=False):
        st.markdown("""
        ### Tier Segmentation
        Influencers are grouped into standard creator categories based on audience size:
        | Tier Label | Follower Threshold | Strategic Characteristics |
        | :--- | :--- | :--- |
        | **Nano** | $< 10,000$ | Highly engaged, niche community, accessible |
        | **Micro** | $10,000 - 100,000$ | High engagement, high trust, targeted demographics |
        | **Mid-tier** | $100,000 - 500,000$ | Professional content, balanced reach and resonance |
        | **Macro** | $500,000 - 1,000,000$ | Broad reach, celebrity status within industry |
        | **Mega** | $\\ge 1,000,000$ | Mass culture reach, global celebrity appeal |

        *Note: These tiers are common operational guidelines used in influencer marketing, not rigid ISO standards.*
        """)

    # Section E: How Rankings Work
    with st.expander("Section E: Ranking Engine & Sorting Logic", expanded=False):
        st.markdown("""
        ### How the Leaderboard Sorts Influencers
        - **Calculated App Rank:** Generated dynamically for the filtered view ($1, 2, 3, \\dots, N$).
        - **Original Source Rank:** Preserved from the raw Kaggle dataset for auditing.
        - **Missing Value Handling:** Missing numbers (`NaN`) are placed at the very end of the leaderboard (`na_position='last'`), ensuring the application never crashes.
        - **Tie Handling:** If two creators have identical metrics, secondary sorting by follower volume breaks the tie consistently.
        """)
        # Live Example from Dataset
        top_by_fol = influencers_df.sort_values("followers", ascending=False).iloc[0]
        top_by_eng = influencers_df.sort_values("engagement_rate", ascending=False).iloc[0]
        st.markdown(f"""
        **Live Example from Active Dataset:**
        - **Rank by Followers:** #1 is **@{top_by_fol['handle']}** with **{format_number(top_by_fol['followers'])} followers** (Engagement: {top_by_fol['engagement_rate']:.2f}%).
        - **Rank by Engagement Rate:** #1 is **@{top_by_eng['handle']}** with **{top_by_eng['engagement_rate']:.2f}% engagement** (Followers: {format_number(top_by_eng['followers'])}).
        """)

    # Section F: Interactive Dynamic Comparison Example
    with st.expander("Section F: How Influencer Comparison Works (Interactive Dynamic Demo)", expanded=True):
        st.markdown("""
        ### Step-by-Step Comparison Mechanism
        1. User selects 2 or more creator handles.
        2. System isolates records and matches comparable metrics.
        3. Identifies the numerical leader for each metric individually.
        4. Analyzes the **Scale vs. Engagement Trade-off**.

        #### Try the Interactive Two-Creator Benchmark:
        """)
        demo_col1, demo_col2 = st.columns(2)
        demo_handles = influencers_df["handle"].tolist()
        with demo_col1:
            inf_a_handle = st.selectbox("Select Influencer A:", demo_handles, index=0, key="how_inf_a")
        with demo_col2:
            inf_b_handle = st.selectbox("Select Influencer B:", demo_handles, index=min(1, len(demo_handles)-1), key="how_inf_b")

        if inf_a_handle == inf_b_handle:
            st.info("Select two different creators to see the live comparison.")
        else:
            inf_a = influencers_df[influencers_df["handle"] == inf_a_handle].iloc[0]
            inf_b = influencers_df[influencers_df["handle"] == inf_b_handle].iloc[0]

            demo_rows = [
                {
                    "Metric": "Followers (Audience Scale)",
                    f"@{inf_a['handle']}": format_exact(inf_a["followers"]),
                    f"@{inf_b['handle']}": format_exact(inf_b["followers"]),
                    "Leading Creator": f"@{inf_a['handle']}" if inf_a["followers"] >= inf_b["followers"] else f"@{inf_b['handle']}",
                },
                {
                    "Metric": "Engagement Rate (%)",
                    f"@{inf_a['handle']}": f"{inf_a['engagement_rate']:.2f}%",
                    f"@{inf_b['handle']}": f"{inf_b['engagement_rate']:.2f}%",
                    "Leading Creator": f"@{inf_a['handle']}" if inf_a["engagement_rate"] >= inf_b["engagement_rate"] else f"@{inf_b['handle']}",
                },
                {
                    "Metric": "Average Likes per Post",
                    f"@{inf_a['handle']}": format_exact(inf_a["avg_likes"]),
                    f"@{inf_b['handle']}": format_exact(inf_b["avg_likes"]),
                    "Leading Creator": f"@{inf_a['handle']}" if inf_a["avg_likes"] >= inf_b["avg_likes"] else f"@{inf_b['handle']}",
                },
                {
                    "Metric": "Recent Average Likes",
                    f"@{inf_a['handle']}": format_exact(inf_a["new_post_avg_like"]),
                    f"@{inf_b['handle']}": format_exact(inf_b["new_post_avg_like"]),
                    "Leading Creator": f"@{inf_a['handle']}" if inf_a["new_post_avg_like"] >= inf_b["new_post_avg_like"] else f"@{inf_b['handle']}",
                },
            ]
            st.dataframe(pd.DataFrame(demo_rows), use_container_width=True, hide_index=True)

            # Trade-off explanation
            if (inf_a["followers"] > inf_b["followers"] and inf_a["engagement_rate"] < inf_b["engagement_rate"]) or \
               (inf_b["followers"] > inf_a["followers"] and inf_b["engagement_rate"] < inf_a["engagement_rate"]):
                st.markdown("""
                > **Marketing Takeaway:** As shown above, having more followers does **not** automatically make an influencer the better choice.
                > While mega influencers offer broad brand exposure, micro or mid-tier influencers frequently generate stronger audience connection and higher conversion potential per impression.
                """)

    # Section G: Overall Influencer Score
    with st.expander("Section G: Composite Influencer Score Index", expanded=False):
        st.markdown("""
        ### Optional Multi-Criteria Index
        To offer a holistic metric balancing both audience reach and engagement, InfluencerIQ provides an optional composite index:

        $$\\text{Score} = 0.35 \\times \\text{Norm}(\\text{Followers}) + 0.45 \\times \\text{Norm}(\\text{Engagement}) + 0.20 \\times \\text{Norm}(\\text{Influence Score})$$

        - **Min-Max Normalization:** Every metric is rescaled between 0 and 100:
          $$\\text{Norm}(X) = \\frac{X - X_{\\min}}{X_{\\max} - X_{\\min}} \\times 100$$
        - **Disclaimer:** This score is an analytical modeling tool designed for relative comparisons within a dataset. It is **not** an official Meta or Instagram rating.
        """)

    # Section H: Limitations & Ethical Considerations
    with st.expander("Section H: Limitations & Data Ethics", expanded=True):
        st.markdown("""
        ### Important Real-World Constraints
        1. **Snapshot Limitations:** Social media metrics fluctuate daily. A static dataset represents a single moment in time.
        2. **Public vs. Private Analytics:** Follower counts and average likes are public signals, but true business metrics (Reach, Saves, Link Clicks, Conversions) require authorized Meta Graph API access.
        3. **Audience Quality & Bot Inflation:** High follower numbers may include dormant or bot accounts. Engagement rate acts as a partial quality check, but cannot detect sophisticated fraud without follower-level auditing.
        4. **ROI Uncertainty:** A high engagement rate does not guarantee commercial sales; conversion rates depend on creative resonance, product pricing, and target audience alignment.
        """)


# =========================================================
# PAGE 6: DATA MANAGEMENT AND EXPORT
# =========================================================
elif selected_page == "💾 Data Management & Export":
    st.subheader("💾 Data Management & Export Center")
    st.markdown("Inspect metadata, upload replacement datasets, validate schemas, and export cleaned CSV files.")

    dm_tab1, dm_tab2, dm_tab3 = st.tabs([
        "Dataset Metadata & Schema",
        "Upload & Replace Dataset",
        "Export & Downloads",
    ])

    with dm_tab1:
        st.markdown("#### Active Dataset Metadata")
        st.json(dataset_meta)

        st.markdown("#### Schema & Data Health")
        schema_df = pd.DataFrame({
            "Column": influencers_df.columns,
            "Non-Null Count": [int(influencers_df[c].notna().sum()) for c in influencers_df.columns],
            "Null Count": [int(influencers_df[c].isna().sum()) for c in influencers_df.columns],
            "Data Type": [str(influencers_df[c].dtype) for c in influencers_df.columns],
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

        st.markdown("#### Raw Dataset Preview (First 50 Rows)")
        st.dataframe(influencers_df.head(50), use_container_width=True, hide_index=True)

    with dm_tab2:
        st.markdown("#### Upload Custom Influencer CSV")
        st.markdown("""
        You can upload your own influencer dataset for immediate analysis.
        - **Required Fields:** Username / Handle (`handle`, `channel_info`, `username`) and Followers (`followers`, `follower_count`).
        - **Recommended Fields:** `avg_likes`, `engagement_rate`, `country`, `posts`, `influence_score`.
        """)
        new_upload = st.file_uploader(
            "Choose a CSV file to validate and preview:",
            type=["csv"],
            key="dm_upload_file",
        )
        if new_upload is not None:
            try:
                test_df, test_report = clean_dataset(
                    pd.read_csv(new_upload, encoding_errors="replace"),
                    dataset_name=new_upload.name,
                )
                st.success(f"✅ Schema valid! Successfully parsed **{len(test_df):,} profiles**.")
                st.write("**Cleaning Report:**", test_report)
                st.dataframe(test_df.head(10), use_container_width=True, hide_index=True)
                st.info("To make this active for the entire dashboard, select 'Upload a CSV' in the sidebar.")
            except Exception as exc:
                st.error(f"❌ Upload validation failed: {exc}")

        st.markdown("---")
        st.markdown("#### Original Dataset Integrity")
        st.caption(
            "🔒 **Data Protection Policy:** The raw Kaggle dataset stored in `data/raw/` is protected and will never be overwritten by uploaded files or dashboard edits without explicit administrator action."
        )

    with dm_tab3:
        st.markdown("#### Export Processed Data")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.markdown("##### 1. Cleaned Dataset (All Profiles)")
            st.caption(f"Exports all {len(influencers_df):,} cleaned and validated influencer records.")
            full_clean_csv = influencers_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=full_clean_csv,
                file_name="influencers_cleaned_full.csv",
                mime="text/csv",
            )

        with e_col2:
            st.markdown("##### 2. Active Filtered Subset")
            st.caption(f"Exports only the {len(filtered_df):,} profiles currently matching your filters.")
            filtered_csv = filtered_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Download Filtered Subset (CSV)",
                data=filtered_csv,
                file_name="influencers_filtered_subset.csv",
                mime="text/csv",
            )
