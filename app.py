import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ============================================================
# BLOOMDETECT AI | FINAL PUBLIC DASHBOARD
# ============================================================
st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 82% 4%, rgba(25, 126, 157, 0.14), transparent 28%),
        radial-gradient(circle at 10% 18%, rgba(16, 78, 110, 0.10), transparent 26%),
        #07141b;
}

.block-container {
    max-width: 1380px;
    padding-top: 2.0rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #061119;
    border-right: 1px solid rgba(127, 196, 210, 0.13);
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 42px 46px 38px 46px;
    border: 1px solid rgba(131, 207, 224, 0.17);
    border-radius: 26px;
    background:
        linear-gradient(135deg, rgba(8, 48, 64, .96), rgba(5, 24, 34, .97)),
        radial-gradient(circle at 85% 25%, rgba(54, 190, 205, .18), transparent 28%);
    box-shadow: 0 24px 70px rgba(0,0,0,.28);
}

.hero:after {
    content: "";
    position: absolute;
    width: 310px;
    height: 310px;
    right: -90px;
    top: -130px;
    border-radius: 50%;
    border: 1px solid rgba(101, 209, 220, .15);
    box-shadow: 0 0 0 38px rgba(101,209,220,.035), 0 0 0 78px rgba(101,209,220,.02);
}

.eyebrow {
    color: #70d2d9;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    font-family: "Manrope", sans-serif;
    font-size: clamp(2.25rem, 5vw, 4.25rem);
    line-height: .98;
    margin: 0;
    color: #f2fbfc;
    letter-spacing: -.045em;
}

.hero p {
    color: #a9c5cb;
    max-width: 760px;
    font-size: 1.04rem;
    line-height: 1.65;
    margin-top: 20px;
}

.pill {
    display: inline-block;
    margin-top: 16px;
    padding: 7px 12px;
    border: 1px solid rgba(111, 211, 217, .2);
    border-radius: 999px;
    color: #bdebee;
    background: rgba(75, 182, 194, .07);
    font-size: .82rem;
}

.section-title {
    font-family: "Manrope", sans-serif;
    color: #edf8fa;
    font-size: 1.45rem;
    font-weight: 800;
    margin: 30px 0 12px 0;
    letter-spacing: -.025em;
}

.card {
    background: linear-gradient(145deg, rgba(12, 34, 44, .92), rgba(7, 25, 34, .92));
    border: 1px solid rgba(126, 194, 208, .13);
    border-radius: 18px;
    padding: 20px;
    min-height: 118px;
}

.card-label {
    color: #83aab2;
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-weight: 700;
}

.card-value {
    color: #effbfc;
    font-family: "Manrope", sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 8px;
}

.card-note {
    color: #78959d;
    font-size: .79rem;
    margin-top: 5px;
}

.callout {
    border-left: 3px solid #58c7d0;
    background: rgba(63, 174, 187, .07);
    padding: 16px 18px;
    border-radius: 0 12px 12px 0;
    color: #b9d5da;
    line-height: 1.65;
}

.small-muted {
    color: #78959d;
    font-size: .82rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(12,34,44,.92), rgba(7,25,34,.92));
    border: 1px solid rgba(126,194,208,.13);
    padding: 15px 16px;
    border-radius: 16px;
}

[data-testid="stMetricLabel"] {
    color: #83aab2 !important;
}

[data-testid="stMetricValue"] {
    color: #effbfc !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(6, 20, 27, .75);
    padding: 7px;
    border-radius: 14px;
    border: 1px solid rgba(126,194,208,.10);
}

.stTabs [data-baseweb="tab"] {
    color: #8eabb1;
    border-radius: 9px;
    padding: 9px 15px;
}

.stTabs [aria-selected="true"] {
    color: #dff9fa !important;
    background: rgba(73, 179, 190, .13);
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

button[kind="primary"] {
    background: #39aeb9;
    border: none;
    color: #061119;
    font-weight: 700;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Data --------------------------
DATA_PATH = Path(__file__).resolve().parent / "latest_bloom_risk_predictions.csv"

@st.cache_data(show_spinner="Loading satellite observations...")
def load_data(path):
    data = pd.read_csv(path)
    for col in ["date", "latitude", "longitude", "chla", "risk_probability"]:
        if col in data.columns:
            if col == "date":
                data[col] = pd.to_datetime(data[col], errors="coerce")
            else:
                data[col] = pd.to_numeric(data[col], errors="coerce")
    required = ["date", "latitude", "longitude", "chla", "risk_label"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if "risk_probability" not in data.columns:
        data["risk_probability"] = np.nan
    return data.dropna(subset=["date", "latitude", "longitude"]).copy()

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error("BloomDetect could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest_df = df[df["date"].eq(latest_date)].copy()
latest_risk = latest_df[latest_df["risk_label"].eq("Potential Bloom Risk")].copy()

# ----------------------------- Helpers ----------------------
def geographic_window(data, choice):
    # These are geographic filters applied to the same dataset.
    # They are NOT separate ocean datasets.
    if choice == "Indian Ocean study window":
        return data[data["latitude"].between(-40, 30) & data["longitude"].between(20, 120)].copy()
    if choice == "Arabian Sea geographic window":
        return data[data["latitude"].between(5, 25) & data["longitude"].between(50, 75)].copy()
    if choice == "Bay of Bengal geographic window":
        return data[data["latitude"].between(5, 25) & data["longitude"].between(75, 100)].copy()
    return data.copy()

def score_value(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)

# ----------------------------- Sidebar -----------------------
st.sidebar.markdown("## 🌊 BloomDetect")
st.sidebar.caption("Satellite-based potential bloom-risk screening")

st.sidebar.markdown("### Explore")
view = st.sidebar.radio(
    "Map layer",
    ["Potential risk", "All observations"],
    label_visibility="collapsed"
)

window = st.sidebar.selectbox(
    "Geographic window",
    [
        "Indian Ocean study window",
        "Arabian Sea geographic window",
        "Bay of Bengal geographic window",
        "Global dataset"
    ],
)

score_cutoff = st.sidebar.slider(
    "Minimum model score",
    0.0, 1.0, 0.50, 0.05,
    help="A model score used to filter the displayed predictions. It is not a calibrated probability of a harmful algal bloom."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset status**")
st.sidebar.write(f"Latest observation  ·  **{latest_date:%d %b %Y}**")
st.sidebar.write(f"Latest grid cells  ·  **{len(latest_df):,}**")
st.sidebar.write(f"Latest flagged  ·  **{len(latest_risk):,}**")
st.sidebar.caption("The geographic windows above are filters on the same dataset, not separate datasets.")

# ----------------------------- Filter ------------------------
focus = geographic_window(latest_df, window)
focus_risk = focus[focus["risk_label"].eq("Potential Bloom Risk")].copy()
focus_risk = focus_risk[score_value(focus_risk["risk_probability"]) >= score_cutoff]

# ----------------------------- Hero --------------------------
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">EOS-06 · OCM-3 · AI SCREENING</div>
  <h1>BloomDetect AI</h1>
  <p>
    A clean spatial view of satellite-derived chlorophyll-a patterns and
    machine-learning screening signals for <b>potential bloom risk</b>.
    Built to help people see where additional investigation may be useful.
  </p>
  <span class="pill">Latest observation · {latest_date:%d %B %Y}</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Latest situation</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Grid locations", f"{len(latest_df):,}")
with m2:
    st.metric("Potential-risk locations", f"{len(latest_risk):,}")
with m3:
    share = len(latest_risk) / len(latest_df) * 100 if len(latest_df) else 0
    st.metric("Share flagged", f"{share:.2f}%")
with m4:
    mean_chla = latest_df["chla"].mean()
    st.metric("Mean Chl-a", f"{mean_chla:.4f}" if pd.notna(mean_chla) else "N/A")

st.markdown("""
<div class="callout">
<b>How to read this:</b> BloomDetect produces a model-derived screening signal.
A high chlorophyll-a value alone does not prove a harmful algal bloom. Species,
toxins and confirmed bloom occurrence require appropriate environmental and
field-based validation.
</div>
""", unsafe_allow_html=True)

# ----------------------------- Tabs --------------------------
tab_map, tab_explore, tab_insights, tab_method, tab_data = st.tabs(
    ["◉ Risk map", "⌖ Location check", "◒ Insights", "⌁ Method", "↓ Data"]
)

# ============================================================
# MAP
# ============================================================
with tab_map:
    st.markdown('<div class="section-title">Spatial screening map</div>', unsafe_allow_html=True)
    st.caption(
        f"{window} · {len(focus_risk):,} displayed potential-risk locations · "
        f"model score ≥ {score_cutoff:.2f}"
    )

    if view == "Potential risk":
        map_df = focus_risk.copy()
    else:
        map_df = focus.copy()
        if "risk_probability" in map_df.columns:
            map_df = map_df[score_value(map_df["risk_probability"]) >= score_cutoff]

    if len(map_df) == 0:
        st.info("No observations match the selected filters.")
    else:
        # Keep all-observation maps lightweight.
        plot_df = map_df.copy()
        if view == "All observations" and len(plot_df) > 12000:
            plot_df = plot_df.sample(12000, random_state=42)

        hover = {
            "latitude": ":.2f",
            "longitude": ":.2f",
            "chla": ":.4f",
            "risk_probability": ":.3f",
            "risk_label": True,
        }
        hover = {k: v for k, v in hover.items() if k in plot_df.columns}

        fig = px.scatter_geo(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="risk_label" if view == "All observations" else None,
            size="risk_probability" if view == "Potential risk" and plot_df["risk_probability"].notna().any() else None,
            hover_data=hover,
            projection="natural earth",
        )
        if view == "Potential risk":
            fig.update_traces(marker=dict(color="#ff6673", opacity=.82, line=dict(width=0)))
        fig.update_geos(
            showcountries=True,
            showcoastlines=True,
            coastlinecolor="rgba(170,210,216,.55)",
            showland=True,
            landcolor="#d8dfdc",
            showocean=True,
            oceancolor="#081d28",
            lakecolor="#081d28",
        )
        fig.update_layout(
            height=680,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#b9d5da"),
            legend_title_text="Prediction",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("### What this map is for")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**01 · Discover**")
        st.caption("See where the model is flagging potential bloom-risk patterns.")
    with b:
        st.markdown("**02 · Prioritize**")
        st.caption("Use the score filter to build a smaller screening list.")
    with c:
        st.markdown("**03 · Validate**")
        st.caption("Use flagged locations to support further environmental or field investigation.")

# ============================================================
# LOCATION CHECK
# ============================================================
with tab_explore:
    st.markdown('<div class="section-title">Check a location</div>', unsafe_allow_html=True)
    st.caption("Enter approximate coordinates. BloomDetect returns the nearest available satellite grid cell.")

    x1, x2 = st.columns(2)
    with x1:
        user_lat = st.number_input("Latitude", -90.0, 90.0, 17.4, 0.1)
    with x2:
        user_lon = st.number_input("Longitude", -180.0, 180.0, 78.5, 0.1)

    if st.button("Find nearest observation", type="primary"):
        working = latest_df.dropna(subset=["latitude", "longitude"]).copy()
        lat_scale = 111.0
        lon_scale = 111.0 * np.cos(np.deg2rad(user_lat))
        lon_scale = max(lon_scale, 1.0)

        working["_distance_km"] = np.sqrt(
            ((working["latitude"] - user_lat) * lat_scale) ** 2 +
            ((working["longitude"] - user_lon) * lon_scale) ** 2
        )

        nearest = working.loc[working["_distance_km"].idxmin()]

        st.markdown("### Nearest satellite grid cell")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Latitude", f"{nearest['latitude']:.2f}°")
        with q2:
            st.metric("Longitude", f"{nearest['longitude']:.2f}°")
        with q3:
            val = nearest["chla"]
            st.metric("Chl-a", f"{val:.4f}" if pd.notna(val) else "N/A")
        with q4:
            st.metric("Distance", f"{nearest['_distance_km']:.1f} km")

        score = nearest.get("risk_probability", np.nan)
        status = nearest["risk_label"]

        if status == "Potential Bloom Risk":
            st.error("Potential bloom-risk flag at the nearest grid cell.")
        else:
            st.success("No potential bloom-risk flag at the nearest grid cell.")

        if pd.notna(score):
            st.progress(
                float(np.clip(score, 0, 1)),
                text=f"Model score · {float(score):.3f}"
            )

        st.caption(
            "The nearest-grid result is a spatial approximation. The model score is a screening indicator, not a confirmed HAB probability."
        )

# ============================================================
# INSIGHTS
# ============================================================
with tab_insights:
    st.markdown('<div class="section-title">Decision-support view</div>', unsafe_allow_html=True)

    risk_scores = score_value(latest_risk["risk_probability"])
    i1, i2, i3 = st.columns(3)
    with i1:
        st.metric("Flagged now", f"{len(latest_risk):,}")
    with i2:
        st.metric("Highest model score", f"{risk_scores.max():.3f}" if len(risk_scores) else "N/A")
    with i3:
        st.metric("Median model score", f"{risk_scores.median():.3f}" if len(risk_scores) else "N/A")

    if len(latest_risk):
        chart_df = latest_risk[["risk_probability"]].copy()
        chart_df["Model score"] = score_value(chart_df["risk_probability"])

        fig_hist = px.histogram(
            chart_df,
            x="Model score",
            nbins=20,
            title="Distribution of model scores among flagged locations"
        )
        fig_hist.update_layout(
            height=360,
            margin=dict(l=0, r=0, t=55, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hist, width="stretch")

        st.markdown("### Screening queue")
        queue_cols = ["latitude", "longitude", "chla", "risk_probability"]
        queue = latest_risk[queue_cols].copy()
        queue = queue.sort_values("risk_probability", ascending=False, na_position="last").head(20)
        queue.columns = ["Latitude", "Longitude", "Chl-a", "Model score"]
        st.dataframe(queue, width="stretch", hide_index=True)

        st.caption(
            "The queue is intended for prioritization of follow-up investigation. It is not an official public-health or fisheries warning."
        )
    else:
        st.info("No potential-risk locations are present in the current latest-date dataset.")

# ============================================================
# METHOD
# ============================================================
with tab_method:
    st.markdown('<div class="section-title">From satellite data to screening signal</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Observe", "EOS-06 OCM-3 ocean-colour observations provide chlorophyll-a information."),
        ("02", "Prepare", "Invalid observations and duplicate or unusable source records are handled before analysis."),
        ("03", "Build context", "Previous observations and historical baseline features describe how chlorophyll-a changes over time."),
        ("04", "Classify", "A machine-learning model produces a Normal or Potential Bloom Risk prediction."),
        ("05", "Map", "Predictions are returned to their geographic grid locations."),
        ("06", "Investigate", "Flagged locations can support targeted review and field/environmental validation."),
    ]

    for num, title, desc in steps:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:10px; min-height:0;">
                <span style="color:#70d2d9;font-weight:800;letter-spacing:.08em;">{num}</span>
                <span style="color:#effbfc;font-size:1.05rem;font-weight:700;margin-left:12px;">{title}</span>
                <div style="color:#8faeb5;margin-top:7px;line-height:1.55;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### What BloomDetect does not claim")
    st.markdown("""
    - It does **not** identify a specific algal species.
    - It does **not** measure algal toxins.
    - It does **not** confirm that a flagged location contains a harmful algal bloom.
    - It is **not** a live sensor network or guaranteed real-time warning service.
    - Satellite screening should be complemented by appropriate environmental and field observations.
    """)

    st.markdown("### Data provenance")
    st.caption(
        "The project uses EOS-06 / OCM-3 analysed chlorophyll data from the MOSDAC / ISRO data ecosystem. "
        "The MOSDAC algorithm documentation describes daily analysed global-ocean chlorophyll fields."
    )

# ============================================================
# DATA
# ============================================================
with tab_data:
    st.markdown('<div class="section-title">Data & downloads</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Dataset rows", f"{len(df):,}")
    with d2:
        st.metric("Dates represented", f"{df['date'].dt.date.nunique():,}")
    with d3:
        st.metric("Latest date", f"{latest_date:%d %b %Y}")

    st.markdown("### Download the current screening view")
    download_cols = ["date", "latitude", "longitude", "chla", "risk_probability", "risk_label"]
    download_cols = [c for c in download_cols if c in focus.columns]
    current_download = focus[download_cols].copy()

    st.download_button(
        "Download observations as CSV",
        data=current_download.to_csv(index=False).encode("utf-8"),
        file_name="bloomdetect_current_observations.csv",
        mime="text/csv",
        on_click="ignore",
        type="primary",
    )

    st.markdown("### Latest flagged locations")
    if len(latest_risk):
        alert_cols = ["latitude", "longitude", "chla", "risk_probability"]
        alert_table = latest_risk[alert_cols].copy()
        alert_table = alert_table.sort_values("risk_probability", ascending=False, na_position="last").head(25)
        alert_table.columns = ["Latitude", "Longitude", "Chl-a", "Model score"]
        st.dataframe(alert_table, width="stretch", hide_index=True)

        alert_text = (
            f"BloomDetect AI screening summary — {latest_date:%d %B %Y}\n\n"
            f"{len(latest_risk):,} locations were flagged for potential bloom risk in the latest available observation.\n"
            "This is a model-derived screening signal. It does not confirm a harmful algal bloom, species, toxin, or public-health event.\n"
            "Further environmental and/or field validation is recommended."
        )

        st.download_button(
            "Download screening summary",
            data=alert_text,
            file_name="bloomdetect_screening_summary.txt",
            mime="text/plain",
            on_click="ignore",
        )
    else:
        st.info("There are no flagged locations in the latest observation.")

    st.markdown("### Data preview")
    preview_cols = ["date", "latitude", "longitude", "chla", "risk_probability", "risk_label"]
    preview_cols = [c for c in preview_cols if c in latest_df.columns]
    st.dataframe(latest_df[preview_cols].head(100), width="stretch", hide_index=True)

# ----------------------------- Footer ------------------------
st.markdown("---")
st.markdown(
    '<div class="small-muted">BloomDetect AI · EOS-06 OCM-3 · Potential bloom-risk screening · Research prototype</div>',
    unsafe_allow_html=True
)
