import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "latest_bloom_risk_predictions.csv"

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data["chla"] = pd.to_numeric(data["chla"], errors="coerce")
    if "risk_probability" in data.columns:
        data["risk_probability"] = pd.to_numeric(
            data["risk_probability"], errors="coerce"
        )
    data = data.dropna(subset=["latitude", "longitude", "date"]).copy()
    return data

df = load_data(DATA_PATH)

# ------------------------------------------------------------
# SAFE COLUMN HANDLING
# ------------------------------------------------------------
if "risk_label" not in df.columns:
    st.error("The prediction file does not contain the required 'risk_label' column.")
    st.stop()

if "risk_probability" not in df.columns:
    df["risk_probability"] = np.nan

latest_date = df["date"].max()
risk_df = df[df["risk_label"] == "Potential Bloom Risk"].copy()

# Model score is used as an indicator, not a confirmed probability of HAB occurrence.
def severity(score):
    if pd.isna(score):
        return "Not available"
    if score >= 0.90:
        return "Very High"
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Moderate"
    return "Low"

df["risk_level"] = df["risk_probability"].apply(severity)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("🌊 BloomDetect AI")
st.sidebar.caption("Potential bloom-risk early-warning dashboard")

st.sidebar.markdown("### Map controls")

map_scope = st.sidebar.selectbox(
    "View",
    ["Potential Bloom Risk Only", "All Observation Locations"]
)

min_score = st.sidebar.slider(
    "Minimum model score",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    help="Filters points using the model's prediction score. This is not a confirmed HAB probability."
)

region = st.sidebar.selectbox(
    "Region",
    [
        "Indian Ocean & surrounding region",
        "Arabian Sea",
        "Bay of Bengal",
        "All available observations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick facts")
st.sidebar.write(f"**Latest observation:** {latest_date.strftime('%d %b %Y')}")
st.sidebar.write(f"**Grid locations:** {len(df):,}")
st.sidebar.write(f"**Risk locations:** {len(risk_df):,}")
st.sidebar.caption(
    "Satellite observations support early-warning analysis. "
    "They do not by themselves confirm a harmful algal bloom, species, or toxin."
)

# ------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------
def apply_region_filter(data, selected_region):
    if selected_region == "Arabian Sea":
        return data[
            (data["latitude"].between(5, 25))
            & (data["longitude"].between(50, 75))
        ].copy()

    if selected_region == "Bay of Bengal":
        return data[
            (data["latitude"].between(5, 25))
            & (data["longitude"].between(75, 100))
        ].copy()

    if selected_region == "Indian Ocean & surrounding region":
        return data[
            (data["latitude"].between(-40, 30))
            & (data["longitude"].between(20, 120))
        ].copy()

    return data.copy()

filtered_df = apply_region_filter(df, region)
filtered_risk = filtered_df[filtered_df["risk_label"] == "Potential Bloom Risk"].copy()

if "risk_probability" in filtered_df.columns:
    score_filtered = filtered_df[
        filtered_df["risk_probability"].fillna(0) >= min_score
    ].copy()
else:
    score_filtered = filtered_df.copy()

if map_scope == "Potential Bloom Risk Only":
    map_df = score_filtered[
        score_filtered["risk_label"] == "Potential Bloom Risk"
    ].copy()
else:
    map_df = score_filtered.copy()

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🌊 BloomDetect AI")
st.subheader("AI-Based Harmful Algal Bloom Detection and Early Warning System")

st.info(
    "BloomDetect AI identifies locations with **potential bloom risk** "
    "using EOS-06 OCM-3 chlorophyll-a observations and machine learning. "
    "It is designed as an early-warning support tool, not as a replacement for field validation."
)

# ------------------------------------------------------------
# TOP METRICS
# ------------------------------------------------------------
total_locations = len(filtered_df)
risk_locations = len(filtered_risk)
risk_percent = (risk_locations / total_locations * 100) if total_locations else 0
avg_chla = filtered_df["chla"].mean() if len(filtered_df) else np.nan
max_chla = filtered_df["chla"].max() if len(filtered_df) else np.nan

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Observation Locations", f"{total_locations:,}")

with c2:
    st.metric("Potential Bloom Risk", f"{risk_locations:,}")

with c3:
    st.metric("Risk Share", f"{risk_percent:.2f}%")

with c4:
    st.metric(
        "Average Chl-a",
        f"{avg_chla:.4f}" if pd.notna(avg_chla) else "N/A"
    )

with c5:
    st.metric(
        "Latest Date",
        latest_date.strftime("%d %b %Y")
    )

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "🗺️ Risk Map",
        "📍 Location Explorer",
        "🧠 How It Works",
        "📥 Data & Alerts"
    ]
)

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab1:
    st.markdown("## 🌍 What is happening?")

    o1, o2 = st.columns(2)

    with o1:
        st.markdown("### Potential-risk snapshot")
        st.write(
            f"On **{latest_date.strftime('%d %B %Y')}**, "
            f"the selected region contains **{risk_locations:,} locations** "
            f"flagged as potential bloom-risk locations."
        )

        if risk_locations > 0:
            high_count = (filtered_risk["risk_level"].isin(["Very High", "High"])).sum()
            st.write(
                f"Among those locations, **{high_count:,}** have a "
                "high or very-high model score."
            )

    with o2:
        st.markdown("### Who can use it?")
        st.markdown(
            """
            - 🐟 **Fisheries & fishing communities:** identify areas that may deserve attention.
            - 🏖️ **Coastal communities & tourism:** support awareness and planning.
            - 🧪 **Researchers:** inspect spatial chlorophyll-a patterns.
            - 🏛️ **Environmental planners:** prioritize areas for further investigation.
            - 🚢 **Marine operations:** add a satellite-based environmental indicator to planning.
            - 🎓 **Students & educators:** explore satellite data and AI in a real application.
            """
        )

    st.markdown("## 🚦 Risk-level distribution")

    if len(filtered_risk):
        level_order = ["Very High", "High", "Moderate", "Low", "Not available"]
        level_counts = (
            filtered_risk["risk_level"]
            .value_counts()
            .reindex(level_order, fill_value=0)
            .reset_index()
        )
        level_counts.columns = ["Risk Level", "Locations"]

        fig_bar = px.bar(
            level_counts,
            x="Risk Level",
            y="Locations",
            text="Locations",
            title="Potential Bloom-Risk Locations by Model Score"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=420)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No risk locations match the current filters.")

    st.markdown("## 🔎 Key interpretation")
    st.warning(
        "A potential bloom-risk flag is an AI-derived screening signal. "
        "High chlorophyll-a alone does not prove a harmful algal bloom. "
        "Confirmation requires appropriate oceanographic and field-based validation."
    )

# ============================================================
# TAB 2: MAP
# ============================================================
with tab2:
    st.markdown("## 🗺️ Potential Bloom Risk Map")
    st.caption(
        f"Latest EOS-06 OCM-3 observation date: {latest_date.strftime('%d %B %Y')}"
    )

    if len(map_df) == 0:
        st.warning("No locations match the selected map filters.")
    else:
        hover_cols = ["date", "chla", "risk_level"]
        if "risk_probability" in map_df.columns:
            hover_cols.append("risk_probability")

        fig = px.scatter_geo(
            map_df,
            lat="latitude",
            lon="longitude",
            color="risk_level" if map_scope == "Potential Bloom Risk Only" else "risk_label",
            size="risk_probability" if map_scope == "Potential Bloom Risk Only" else None,
            hover_data=hover_cols,
            projection="equirectangular",
            title=f"BloomDetect AI Spatial Risk View - {latest_date.strftime('%Y-%m-%d')}"
        )

        fig.update_geos(
            showcountries=True,
            showcoastlines=True,
            showland=True,
            landcolor="rgb(235,235,235)",
            coastlinecolor="black",
            lataxis_range=[-40, 30],
            lonaxis_range=[20, 120],
            projection_scale=1.0
        )

        fig.update_layout(
            height=700,
            margin=dict(l=0, r=0, t=60, b=0),
            legend_title_text="Indicator"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📌 Map interpretation")
        st.write(
            "Each point represents an observation grid location. "
            "Red/high-score areas indicate locations the model has flagged for potential bloom risk. "
            "The map is intended to help users decide where further investigation or field validation may be useful."
        )

# ============================================================
# TAB 3: LOCATION EXPLORER
# ============================================================
with tab3:
    st.markdown("## 📍 Location Explorer")
    st.write(
        "Enter approximate coordinates to find the nearest satellite observation "
        "and inspect its current model result."
    )

    a, b = st.columns(2)

    with a:
        user_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=17.4,
            step=0.1
        )

    with b:
        user_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.5,
            step=0.1
        )

    if st.button("🔎 Find Nearest Observation", type="primary"):
        working = df.dropna(subset=["latitude", "longitude"]).copy()

        # Simple geographic distance approximation for nearest grid cell.
        working["_distance"] = np.sqrt(
            (working["latitude"] - user_lat) ** 2
            + (working["longitude"] - user_lon) ** 2
        )

        nearest = working.loc[working["_distance"].idxmin()]

        st.markdown("### 📡 Nearest satellite observation")

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric("Nearest Latitude", f"{nearest['latitude']:.2f}°")

        with p2:
            st.metric("Nearest Longitude", f"{nearest['longitude']:.2f}°")

        with p3:
            st.metric(
                "Chlorophyll-a",
                f"{nearest['chla']:.4f}" if pd.notna(nearest["chla"]) else "N/A"
            )

        with p4:
            st.metric("Status", str(nearest["risk_label"]))

        score = nearest["risk_probability"]

        if pd.notna(score):
            st.progress(
                min(max(float(score), 0.0), 1.0),
                text=f"Model score: {float(score):.3f}"
            )

        if nearest["risk_label"] == "Potential Bloom Risk":
            st.error(
                "This location is flagged for **potential bloom risk** by the model. "
                "This is a screening signal and should be followed by appropriate validation."
            )
        else:
            st.success(
                "This location is not flagged as potential bloom risk by the current model output."
            )

        st.caption(
            f"Distance to nearest grid cell: {nearest['_distance']:.2f} degrees. "
            "The nearest-grid result is an approximation."
        )

# ============================================================
# TAB 4: HOW IT WORKS
# ============================================================
with tab4:
    st.markdown("## 🧠 How BloomDetect AI Works")

    st.markdown(
        """
        ### 1. 🛰️ Satellite observation
        EOS-06 OCM-3 provides ocean-colour observations from which
        chlorophyll-a information is used.

        ### 2. 🧹 Data preparation
        Invalid values, duplicate/misdated files and unusable observations
        are handled before machine-learning analysis.

        ### 3. 📈 Historical context
        The system uses previous observations and historical baseline information
        to represent how chlorophyll-a changes over time.

        ### 4. 🤖 Machine learning
        A Decision Tree model classifies locations into:
        - **Normal**
        - **Potential Bloom Risk**

        ### 5. 🗺️ Spatial risk mapping
        Model predictions are placed back onto geographic coordinates to produce
        a visual risk map.

        ### 6. 🚨 Early-warning support
        The resulting map can help users identify locations that may deserve
        additional investigation, monitoring or field validation.
        """
    )

    st.markdown("## 🧩 Why this is useful beyond ocean research")

    use_cases = pd.DataFrame(
        {
            "User": [
                "Fishing communities",
                "Coastal tourism",
                "Environmental agencies",
                "Marine researchers",
                "Port / marine operations",
                "Disaster & public-awareness teams",
                "Students / educators"
            ],
            "Possible use": [
                "Prioritize areas for additional checking",
                "Support environmental awareness",
                "Screen locations for follow-up monitoring",
                "Explore spatial chlorophyll patterns",
                "Add environmental information to planning",
                "Visualize areas requiring attention",
                "Demonstrate satellite + AI applications"
            ]
        }
    )

    st.dataframe(use_cases, use_container_width=True, hide_index=True)

    st.markdown("## ⚠️ Scientific limitation")
    st.warning(
        "BloomDetect AI does not identify algal species or toxins from this dataset. "
        "Satellite chlorophyll-a is an environmental indicator. "
        "A model-derived potential bloom-risk flag should not be interpreted as confirmed HAB occurrence."
    )

# ============================================================
# TAB 5: DATA + DOWNLOADS + ALERTS
# ============================================================
with tab5:
    st.markdown("## 📥 Data, Downloads & Alert Support")

    st.markdown("### 📚 Dataset information")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric("Observation Grid Points", f"{len(df):,}")

    with d2:
        st.metric(
            "Latest Chl-a",
            f"{df['chla'].max():.4f}" if df["chla"].notna().any() else "N/A"
        )

    with d3:
        st.metric("Latest Observation", latest_date.strftime("%d %b %Y"))

    st.write(
        "**Source:** EOS-06 / OCM-3 Level-4 analysed chlorophyll product "
        "(MOSDAC / ISRO data ecosystem)."
    )
    st.write("**Spatial resolution:** approximately 0.25° grid.")
    st.write(
        "**Current dataset role:** satellite-derived chlorophyll-a observations "
        "used for potential bloom-risk screening."
    )

    st.markdown("### 📊 Download filtered observations")

    download_df = map_df.copy()

    if len(download_df):
        csv_bytes = download_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Current View as CSV",
            data=csv_bytes,
            file_name="bloomdetect_current_view.csv",
            mime="text/csv"
        )

    st.markdown("### 🚨 Create a simple investigation alert")

    if len(risk_df):
        top_alerts = risk_df.sort_values(
            "risk_probability",
            ascending=False,
            na_position="last"
        ).head(10).copy()

        st.dataframe(
            top_alerts[
                [
                    "latitude",
                    "longitude",
                    "chla",
                    "risk_probability",
                    "risk_level"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are model-screened locations for follow-up investigation, not confirmed HAB alerts."
        )

        alert_text = (
            f"BloomDetect AI screening update: {len(risk_df):,} locations were "
            f"flagged for potential bloom risk on {latest_date.strftime('%d %B %Y')}. "
            "Further environmental or field validation is recommended before treating "
            "any location as a confirmed harmful algal bloom."
        )

        st.code(alert_text, language="text")

        st.download_button(
            "⬇️ Download Screening Alert",
            data=alert_text,
            file_name="bloomdetect_screening_alert.txt",
            mime="text/plain"
        )

    st.markdown("### 🔍 Data preview")

    preview_cols = [
        c for c in
        ["date", "latitude", "longitude", "chla", "risk_probability", "risk_label"]
        if c in df.columns
    ]

    st.dataframe(
        df[preview_cols].head(100),
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "BloomDetect AI | EOS-06 OCM-3 satellite-based potential bloom-risk screening | "
    "Research / educational prototype"
)
