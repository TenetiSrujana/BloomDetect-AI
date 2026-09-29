import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# BLOOMDETECT AI | FINAL DASHBOARD
# ============================================================
st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL OCEAN THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 78% 5%, rgba(32, 155, 173, .12), transparent 28%),
        radial-gradient(circle at 5% 25%, rgba(20, 88, 113, .10), transparent 26%),
        #061219;
    color: #eaf6f7;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #041017;
    border-right: 1px solid rgba(117, 194, 207, .12);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 48px 50px 44px;
    border-radius: 28px;
    border: 1px solid rgba(121, 213, 221, .16);
    background:
        linear-gradient(135deg, rgba(7, 51, 65, .98), rgba(4, 27, 37, .98));
    box-shadow: 0 25px 80px rgba(0,0,0,.25);
}

.hero::after {
    content: "";
    position: absolute;
    right: -110px;
    top: -150px;
    width: 390px;
    height: 390px;
    border-radius: 50%;
    border: 1px solid rgba(104, 211, 221, .15);
    box-shadow:
        0 0 0 40px rgba(104,211,221,.035),
        0 0 0 82px rgba(104,211,221,.018);
}

.eyebrow {
    color: #67d1d8;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    font-family: "Manrope", sans-serif;
    color: #f2fbfc;
    font-size: clamp(2.5rem, 5vw, 4.7rem);
    line-height: .95;
    letter-spacing: -.05em;
    margin: 0;
}

.hero p {
    color: #a9c7cd;
    max-width: 800px;
    font-size: 1.05rem;
    line-height: 1.7;
    margin: 20px 0 0;
}

.hero-tag {
    display: inline-block;
    margin-top: 20px;
    padding: 7px 13px;
    border-radius: 999px;
    border: 1px solid rgba(111, 210, 218, .22);
    background: rgba(72, 180, 193, .08);
    color: #c2edef;
    font-size: .8rem;
}

.section-title {
    font-family: "Manrope", sans-serif;
    color: #f0fafb;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.025em;
    margin: 34px 0 14px;
}

.section-subtitle {
    color: #78979f;
    font-size: .9rem;
    margin-top: -6px;
    margin-bottom: 18px;
}

.info-card {
    background: linear-gradient(145deg, rgba(11, 37, 48, .92), rgba(6, 25, 34, .92));
    border: 1px solid rgba(121, 194, 207, .12);
    border-radius: 17px;
    padding: 19px;
}

.info-label {
    color: #75959d;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .11em;
    font-weight: 700;
}

.info-value {
    color: #effbfc;
    font-family: "Manrope", sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    margin-top: 7px;
}

.info-note {
    color: #78959d;
    font-size: .77rem;
    margin-top: 4px;
}

.callout {
    border-left: 3px solid #5bc8d0;
    background: rgba(70, 183, 194, .065);
    padding: 15px 18px;
    border-radius: 0 12px 12px 0;
    color: #b9d5da;
    line-height: 1.65;
}

.map-card {
    background: #061a23;
    border: 1px solid rgba(125, 205, 215, .12);
    border-radius: 20px;
    padding: 8px;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(11,37,48,.92), rgba(6,25,34,.92));
    border: 1px solid rgba(121,194,207,.12);
    border-radius: 15px;
    padding: 14px 15px;
}

[data-testid="stMetricLabel"] { color: #7d9da5 !important; }
[data-testid="stMetricValue"] { color: #effbfc !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    padding: 6px;
    border-radius: 14px;
    background: rgba(4, 17, 24, .72);
    border: 1px solid rgba(121,194,207,.09);
}

.stTabs [data-baseweb="tab"] {
    color: #829da3;
    border-radius: 9px;
    padding: 9px 15px;
}

.stTabs [aria-selected="true"] {
    color: #e6fafb !important;
    background: rgba(75, 182, 194, .13);
}

.stButton > button[kind="primary"] {
    background: #43b8c1;
    border: 0;
    color: #041017;
    font-weight: 800;
}

.small-muted { color: #708d95; font-size: .8rem; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
DATA_PATH = Path(__file__).resolve().parent / "latest_bloom_risk_predictions.csv"

@st.cache_data(show_spinner="Loading EOS-06 observations...")
def load_data(path):
    data = pd.read_csv(path)

    required = ["date", "latitude", "longitude", "chla", "risk_label"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ["latitude", "longitude", "chla", "risk_probability"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "risk_probability" not in data.columns:
        data["risk_probability"] = np.nan

    data = data.dropna(subset=["date", "latitude", "longitude"]).copy()
    return data

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error("BloomDetect AI could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest_df = df[df["date"].eq(latest_date)].copy()

# ============================================================
# HELPERS
# ============================================================
def normalize_longitude(series):
    """Convert longitudes to -180..180 for a clean global map."""
    x = pd.to_numeric(series, errors="coerce")
    return ((x + 180) % 360) - 180


def normalize_one_longitude(value):
    return ((float(value) + 180) % 360) - 180


def geographic_window(data, choice):
    """Geographic filters over the SAME global dataset."""
    if choice == "Indian Ocean study window":
        return data[data["latitude"].between(-40, 30) & data["longitude"].between(20, 120)].copy()

    if choice == "Arabian Sea geographic window":
        return data[data["latitude"].between(5, 25) & data["longitude"].between(50, 75)].copy()

    if choice == "Bay of Bengal geographic window":
        return data[data["latitude"].between(5, 25) & data["longitude"].between(75, 100)].copy()

    return data.copy()


def numeric_score(series):
    return pd.to_numeric(series, errors="coerce")


def make_ocean_map(data, risk_data, window_name, show_all=True):
    """Create full observation map with potential-risk overlay."""
    normal = data.copy()
    risks = risk_data.copy()

    normal["plot_lon"] = normalize_longitude(normal["longitude"])
    risks["plot_lon"] = normalize_longitude(risks["longitude"])

    fig = go.Figure()

    if show_all and len(normal):
        fig.add_trace(go.Scattergeo(
            lon=normal["plot_lon"],
            lat=normal["latitude"],
            mode="markers",
            name="All observations",
            marker=dict(
                size=3.2,
                color="#2b7893",
                opacity=0.48,
                line=dict(width=0),
            ),
            customdata=np.column_stack([
                normal["chla"].fillna(np.nan),
                normal["risk_label"].astype(str),
            ]),
            hovertemplate=(
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°<br>"
                "Chl-a: %{customdata[0]:.4f}<br>"
                "Prediction: %{customdata[1]}<extra></extra>"
            ),
        ))

    if len(risks):
        fig.add_trace(go.Scattergeo(
            lon=risks["plot_lon"],
            lat=risks["latitude"],
            mode="markers",
            name="Potential bloom risk",
            marker=dict(
                size=8,
                color="#ff5b5f",
                opacity=0.96,
                line=dict(color="#ffd7d8", width=1),
            ),
            customdata=np.column_stack([
                risks["chla"].fillna(np.nan),
                numeric_score(risks["risk_probability"]).fillna(np.nan),
                risks["risk_label"].astype(str),
            ]),
            hovertemplate=(
                "Latitude: %{lat:.2f}°<br>"
                "Longitude: %{lon:.2f}°<br>"
                "Chl-a: %{customdata[0]:.4f}<br>"
                "Model score: %{customdata[1]:.3f}<br>"
                "Status: %{customdata[2]}<extra></extra>"
            ),
        ))

    if window_name == "Global dataset":
        geo = dict(
            projection_type="equirectangular",
            showland=True,
            landcolor="#d8dedf",
            showocean=True,
            oceancolor="#061b25",
            showcountries=True,
            countrycolor="#66767b",
            coastlinecolor="#819095",
            coastlinewidth=0.8,
            showlakes=True,
            lakecolor="#061b25",
            lonaxis=dict(range=[-180, 180], showgrid=True, gridcolor="rgba(160,190,195,.12)"),
            lataxis=dict(range=[-90, 90], showgrid=True, gridcolor="rgba(160,190,195,.12)"),
        )
    elif window_name == "Arabian Sea geographic window":
        geo = dict(
            projection_type="mercator",
            showland=True,
            landcolor="#d8dedf",
            showocean=True,
            oceancolor="#061b25",
            showcountries=True,
            countrycolor="#66767b",
            coastlinecolor="#819095",
            lonaxis=dict(range=[48, 78]),
            lataxis=dict(range=[2, 28]),
        )
    elif window_name == "Bay of Bengal geographic window":
        geo = dict(
            projection_type="mercator",
            showland=True,
            landcolor="#d8dedf",
            showocean=True,
            oceancolor="#061b25",
            showcountries=True,
            countrycolor="#66767b",
            coastlinecolor="#819095",
            lonaxis=dict(range=[72, 103]),
            lataxis=dict(range=[2, 28]),
        )
    else:
        geo = dict(
            projection_type="mercator",
            showland=True,
            landcolor="#d8dedf",
            showocean=True,
            oceancolor="#061b25",
            showcountries=True,
            countrycolor="#66767b",
            coastlinecolor="#819095",
            lonaxis=dict(range=[15, 125]),
            lataxis=dict(range=[-43, 33]),
        )

    fig.update_geos(**geo)
    fig.update_layout(
        height=650 if window_name != "Global dataset" else 600,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b8d0d5"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(4,17,24,.72)",
            bordercolor="rgba(121,194,207,.12)",
            borderwidth=1,
        ),
    )
    return fig

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🌊 BloomDetect AI")
st.sidebar.caption("Satellite-based potential bloom-risk screening")

st.sidebar.markdown("### Map view")
window = st.sidebar.selectbox(
    "Geographic window",
    [
        "Indian Ocean study window",
        "Global dataset",
        "Arabian Sea geographic window",
        "Bay of Bengal geographic window",
    ],
    help="These are geographic filters on the same global dataset, not separate datasets.",
)

score_cutoff = st.sidebar.slider(
    "Risk-score threshold",
    0.0,
    1.0,
    0.05,
    0.01,
    help="Only potential-risk locations at or above this model score are highlighted. This is not a calibrated HAB probability.",
)

show_all = st.sidebar.checkbox(
    "Show all ocean observations",
    value=True,
    help="Keep the full available observation grid visible and overlay potential-risk locations.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset status**")
st.sidebar.write(f"Latest observation  ·  **{latest_date:%d %b %Y}**")
st.sidebar.write(f"Latest grid cells  ·  **{len(latest_df):,}**")

all_latest_risk = latest_df[latest_df["risk_label"].eq("Potential Bloom Risk")].copy()
st.sidebar.write(f"Latest flagged  ·  **{len(all_latest_risk):,}**")
st.sidebar.caption("EOS-06 OCM-3 E06OCM_L4_AC is a global-ocean product. Regional names here are geographic viewing windows.")

# ============================================================
# CURRENT FILTERED DATA
# ============================================================
focus = geographic_window(latest_df, window)
focus_risk_all = focus[focus["risk_label"].eq("Potential Bloom Risk")].copy()
focus_risk_all["_score"] = numeric_score(focus_risk_all["risk_probability"])
focus_risk = focus_risk_all[focus_risk_all["_score"].fillna(0) >= score_cutoff].copy()

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">EOS-06 · OCM-3 · E06OCM_L4_AC</div>
    <h1>BloomDetect AI</h1>
    <p>
        Satellite-based spatial screening of chlorophyll-a patterns to identify
        <b>potential bloom-risk locations</b> and support further environmental investigation.
    </p>
    <span class="hero-tag">Latest available observation · {latest_date:%d %B %Y}</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP METRICS
# ============================================================
st.markdown('<div class="section-title">Latest ocean view</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Available grid locations", f"{len(latest_df):,}")
with m2:
    st.metric("Potential-risk locations", f"{len(all_latest_risk):,}")
with m3:
    share = (len(all_latest_risk) / len(latest_df) * 100) if len(latest_df) else 0
    st.metric("Flagged share", f"{share:.2f}%")
with m4:
    mean_chla = latest_df["chla"].mean()
    st.metric("Mean chlorophyll-a", f"{mean_chla:.4f}" if pd.notna(mean_chla) else "N/A")

st.markdown("""
<div class="callout">
<b>Interpretation:</b> The full available observation grid is analysed first. Potential-risk locations are then highlighted as a model-derived screening signal. High chlorophyll-a alone does not confirm a harmful algal bloom; species, toxins and confirmed bloom occurrence require appropriate environmental and field validation.
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab_map, tab_location, tab_insights, tab_method, tab_data = st.tabs(
    ["◉ Ocean map", "⌖ Location check", "◒ Insights", "⌁ Method", "↓ Data"]
)

# ============================================================
# OCEAN MAP
# ============================================================
with tab_map:
    st.markdown('<div class="section-title">Ocean observation map</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-subtitle">{window} · {len(focus):,} available observations · {len(focus_risk):,} highlighted potential-risk locations</div>',
        unsafe_allow_html=True,
    )

    fig = make_ocean_map(focus, focus_risk, window, show_all=show_all)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**● Observation grid**")
        st.caption("The muted points represent the available satellite-derived observation locations for the selected date and geographic window.")
    with c2:
        st.markdown("**● Potential-risk overlay**")
        st.caption(f"The highlighted points meet the selected risk-score threshold of {score_cutoff:.2f}.")

    if window != "Global dataset":
        st.info("This is a geographic viewing window over the same global EOS-06 analysed chlorophyll dataset. It is not a separate regional dataset.")

# ============================================================
# LOCATION CHECK
# ============================================================
with tab_location:
    st.markdown('<div class="section-title">Location check</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter coordinates to inspect the nearest available observation on the latest date.</div>', unsafe_allow_html=True)

    x1, x2 = st.columns(2)
    with x1:
        user_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=17.4, step=0.25, format="%.2f")
    with x2:
        user_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.5, step=0.25, format="%.2f")

    if st.button("Find nearest observation", type="primary"):
        working = latest_df.dropna(subset=["latitude", "longitude"]).copy()

        # Handle 0..360 source longitudes and -180..180 user input consistently.
        target_lon = normalize_one_longitude(user_lon)
        working_plot_lon = normalize_longitude(working["longitude"])

        lat_scale = 111.0
        lon_scale = max(111.0 * np.cos(np.deg2rad(user_lat)), 1.0)

        working["_distance_km"] = np.sqrt(
            ((working["latitude"] - user_lat) * lat_scale) ** 2 +
            ((working_plot_lon - target_lon) * lon_scale) ** 2
        )

        nearest = working.loc[working["_distance_km"].idxmin()]

        st.markdown("### Nearest available grid cell")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Latitude", f"{nearest['latitude']:.2f}°")
        with q2:
            st.metric("Longitude", f"{normalize_one_longitude(nearest['longitude']):.2f}°")
        with q3:
            val = nearest["chla"]
            st.metric("Chlorophyll-a", f"{val:.4f}" if pd.notna(val) else "N/A")
        with q4:
            st.metric("Approx. distance", f"{nearest['_distance_km']:.1f} km")

        status = str(nearest["risk_label"])
        score = pd.to_numeric(pd.Series([nearest.get("risk_probability", np.nan)]), errors="coerce").iloc[0]

        if status == "Potential Bloom Risk":
            st.warning("The nearest grid cell is flagged for potential bloom risk.")
        else:
            st.success("The nearest grid cell is not flagged for potential bloom risk.")

        if pd.notna(score):
            st.progress(float(np.clip(score, 0, 1)), text=f"Model score · {float(score):.3f}")

        st.caption("This location check is an approximate nearest-grid lookup. The model score is a screening indicator, not a calibrated probability of a harmful algal bloom.")

# ============================================================
# INSIGHTS
# ============================================================
with tab_insights:
    st.markdown('<div class="section-title">Screening insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">A compact view of the latest model-derived screening results.</div>', unsafe_allow_html=True)

    scores = numeric_score(all_latest_risk["risk_probability"]).dropna()

    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.metric("Flagged locations", f"{len(all_latest_risk):,}")
    with i2:
        st.metric("Highest model score", f"{scores.max():.3f}" if len(scores) else "N/A")
    with i3:
        st.metric("Median model score", f"{scores.median():.3f}" if len(scores) else "N/A")
    with i4:
        threshold_count = int((scores >= score_cutoff).sum()) if len(scores) else 0
        st.metric("Above selected threshold", f"{threshold_count:,}")

    if len(scores):
        chart = go.Figure()
        chart.add_trace(go.Histogram(
            x=scores,
            nbinsx=24,
            marker_color="#39aeb9",
            opacity=.88,
            name="Model score",
        ))
        chart.add_vline(x=score_cutoff, line_dash="dash", line_color="#ff6a6d", annotation_text="Selected threshold")
        chart.update_layout(
            title="Model-score distribution among flagged locations",
            height=380,
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b8d0d5"),
            xaxis_title="Model score",
            yaxis_title="Locations",
        )
        st.plotly_chart(chart, use_container_width=True)

        st.markdown("### Screening queue")
        queue_cols = ["latitude", "longitude", "chla", "risk_probability"]
        queue = all_latest_risk[queue_cols].copy()
        queue["risk_probability"] = numeric_score(queue["risk_probability"])
        queue = queue.sort_values("risk_probability", ascending=False, na_position="last").head(25)
        queue.columns = ["Latitude", "Longitude", "Chl-a", "Model score"]
        st.dataframe(queue, use_container_width=True, hide_index=True)
    else:
        st.info("No potential-risk locations are present in the latest observation.")

    st.markdown("""
    <div class="callout">
    The screening queue is intended to help prioritize locations for additional investigation. It is not an official fisheries, public-health or emergency warning.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# METHOD
# ============================================================
with tab_method:
    st.markdown('<div class="section-title">How BloomDetect works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">From global satellite observations to a spatial screening signal.</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Observe", "EOS-06 OCM-3 analysed chlorophyll-a fields provide daily global-ocean observations."),
        ("02", "Prepare", "Source records are cleaned and organised by location and date before feature construction."),
        ("03", "Build temporal context", "Previous chlorophyll-a values, historical baseline and recent behaviour provide context for each observation."),
        ("04", "Classify", "The trained machine-learning model assigns each available location a Normal or Potential Bloom Risk prediction."),
        ("05", "Map", "Predictions are returned to their geographic grid cells so users can inspect spatial patterns."),
        ("06", "Investigate", "Flagged locations can support prioritization for additional environmental or field validation."),
    ]

    for num, title, desc in steps:
        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom:10px;">
                <span style="color:#67d1d8;font-weight:800;letter-spacing:.08em;">{num}</span>
                <span style="color:#effbfc;font-size:1.05rem;font-weight:700;margin-left:12px;">{title}</span>
                <div style="color:#8faeb5;margin-top:7px;line-height:1.55;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Scientific interpretation")
    st.markdown("""
    - A potential-risk flag is a **model-derived screening signal**.
    - High chlorophyll-a alone does **not** prove a harmful algal bloom.
    - The current dataset does not provide confirmed HAB species or toxin labels.
    - Satellite observations complement, rather than replace, field and environmental validation.
    - The application is an **early-warning support prototype**, not a guaranteed real-time warning service.
    """)

    st.markdown("### Dataset context")
    st.caption("EOS-06 / OCM-3 E06OCM_L4_AC is documented by MOSDAC as a daily analysed chlorophyll product for the global ocean at 0.25° × 0.25° spatial resolution. Regional options in this dashboard are geographic filters applied to that same dataset.")

# ============================================================
# DATA
# ============================================================
with tab_data:
    st.markdown('<div class="section-title">Data & downloads</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Inspect the latest observation grid and export the screening results.</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Rows in prediction file", f"{len(df):,}")
    with d2:
        st.metric("Dates represented", f"{df['date'].dt.date.nunique():,}")
    with d3:
        st.metric("Latest observation", f"{latest_date:%d %b %Y}")

    st.markdown("### Current geographic view")
    download_cols = ["date", "latitude", "longitude", "chla", "risk_probability", "risk_label"]
    download_cols = [c for c in download_cols if c in focus.columns]
    current_download = focus[download_cols].copy()

    st.download_button(
        "Download current observations CSV",
        data=current_download.to_csv(index=False).encode("utf-8"),
        file_name="bloomdetect_current_observations.csv",
        mime="text/csv",
        type="primary",
    )

    st.markdown("### Latest potential-risk locations")
    if len(all_latest_risk):
        alert_cols = ["latitude", "longitude", "chla", "risk_probability"]
        alert_table = all_latest_risk[alert_cols].copy()
        alert_table["risk_probability"] = numeric_score(alert_table["risk_probability"])
        alert_table = alert_table.sort_values("risk_probability", ascending=False, na_position="last").head(50)
        alert_table.columns = ["Latitude", "Longitude", "Chl-a", "Model score"]
        st.dataframe(alert_table, use_container_width=True, hide_index=True)

        alert_text = (
            f"BloomDetect AI screening summary — {latest_date:%d %B %Y}\n\n"
            f"Available latest-date grid locations: {len(latest_df):,}\n"
            f"Potential bloom-risk locations: {len(all_latest_risk):,}\n\n"
            "This is a model-derived screening signal. It does not confirm a harmful algal bloom, species, toxin, or public-health event.\n"
            "Further environmental and/or field validation is recommended."
        )

        st.download_button(
            "Download screening summary",
            data=alert_text,
            file_name="bloomdetect_screening_summary.txt",
            mime="text/plain",
        )
    else:
        st.info("No potential-risk locations are present in the latest observation.")

    st.markdown("### Latest observation preview")
    preview_cols = ["date", "latitude", "longitude", "chla", "risk_probability", "risk_label"]
    preview_cols = [c for c in preview_cols if c in latest_df.columns]
    st.dataframe(latest_df[preview_cols].head(100), use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<div class="small-muted">BloomDetect AI · EOS-06 OCM-3 · Potential bloom-risk screening · Research prototype</div>',
    unsafe_allow_html=True,
)
