
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Home | Risk Map | Location | Insights | Data
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
IMAGE_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# Proxy-screening thresholds used by the project.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading latest satellite field...")
def load_latest():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv is missing beside app.py."
        )

    d = pd.read_csv(DATA_PATH)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for c in [
        "latitude",
        "longitude",
        "chla",
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()

    d["risk_flag"] = (
        d["risk_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("potential bloom risk")
    )

    # Keep longitudes stable for the Indian Ocean study window.
    d["plot_lon"] = d["longitude"].astype(float)

    return d


try:
    df = load_latest()
except Exception as exc:
    st.error("BloomDetect AI could not load the project data.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
latest_risk = latest[latest["risk_flag"]].copy()


# ============================================================
# NAVIGATION
# ============================================================

PAGES = ["home", "map", "location", "insights", "data"]
NAV_LABELS = {
    "home": "⌂ Home",
    "map": "🗺 Risk Map",
    "location": "📍 Location",
    "insights": "📊 Insights",
    "data": "⇩ Data",
}

if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"


def navigate(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
    --ink:#103f49;
    --deep:#07566a;
    --aqua:#0aa8b5;
    --muted:#63828a;
    --line:#a7d8dc;
    --pale:#f1fbfb;
    --green:#22a879;
    --red:#ef4f5e;
    --blue:#197da5;
    --shadow:0 14px 38px rgba(9,80,94,.09);
}

html,body,[data-testid="stAppViewContainer"]{
    background:#f1fbfb!important;
    color:var(--ink)!important;
    font-family:'DM Sans',sans-serif!important;
}

.stApp{
    background:
      radial-gradient(circle at 8% 10%,rgba(55,211,210,.08),transparent 25%),
      radial-gradient(circle at 92% 72%,rgba(19,156,175,.07),transparent 28%),
      linear-gradient(180deg,#fbffff 0%,#effafa 52%,#fbffff 100%)!important;
}

[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,
[data-testid="stSidebar"]{
    display:none!important;
}

.block-container{
    max-width:1260px!important;
    padding:24px 34px 70px!important;
}

.brand-bar{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    padding:15px 20px;
    border:1.5px solid #c5e4e6;
    background:rgba(255,255,255,.94);
    border-radius:22px;
    box-shadow:var(--shadow);
}

.brand-left{
    display:flex;
    align-items:center;
    gap:13px;
}

.logo{
    width:48px;
    height:48px;
    border-radius:16px;
    background:linear-gradient(145deg,#29cdd0,#087b8e);
    display:grid;
    place-items:center;
    color:#fff;
    font-size:20px;
    font-weight:800;
}

.brand-name{
    font:800 1.22rem Manrope,sans-serif;
    color:#103f49;
}

.brand-sub{
    font-size:.73rem;
    color:#78959b;
}

.latest-label{
    text-align:right;
    color:#78959b;
    font-size:.68rem;
}

.latest-label b{
    color:#174b56;
    font-size:.78rem;
}

.nav-wrap{
    margin:18px 0 18px;
}

.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
    min-height:45px!important;
    border-radius:14px!important;
    background:#fff!important;
    border:2px solid #164e5b!important;
    color:#123f49!important;
    font-weight:800!important;
    font-size:.87rem!important;
    box-shadow:0 6px 16px rgba(15,70,82,.09)!important;
}

.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{
    background:#e2f8f8!important;
    border-color:#087d8c!important;
}

.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#d8f7f7,#b9eeee)!important;
    border-color:#078c9b!important;
}

.section{
    padding:28px 0 16px;
}

.kicker{
    font:800 .68rem Manrope,sans-serif;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:11px;
}

.section h2{
    font:800 clamp(2rem,3.6vw,3.15rem)/1.08 Manrope,sans-serif;
    letter-spacing:-.055em;
    color:#103f49;
    margin:0 0 13px;
}

.section p{
    color:#5f8088;
    line-height:1.62;
    margin:0;
    max-width:1080px;
    font-size:.98rem;
}

.hero{
    min-height:390px;
    border-radius:30px;
    padding:58px;
    background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);
    box-shadow:0 28px 72px rgba(6,86,100,.16);
    position:relative;
    overflow:hidden;
}

.hero:after{
    content:"";
    position:absolute;
    width:420px;
    height:420px;
    right:-120px;
    top:-160px;
    border:75px solid rgba(210,255,252,.10);
    border-radius:50%;
}

.hero-content{
    position:relative;
    z-index:2;
    max-width:820px;
}

.hero .kicker{
    color:#a6fffa;
}

.hero h1{
    font:800 clamp(3rem,6vw,5.7rem)/.92 Manrope,sans-serif;
    letter-spacing:-.075em;
    color:#e4ffff;
    margin:0 0 20px;
}

.hero p{
    font-size:1.05rem;
    line-height:1.78;
    color:#e0fbfb;
    max-width:760px;
}

.hero-badges{
    display:flex;
    gap:9px;
    flex-wrap:wrap;
    margin-top:23px;
}

.badge{
    padding:9px 13px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.25);
    color:#efffff;
    font-size:.78rem;
    font-weight:700;
}

.metric{
    min-height:112px;
    height:100%;
    box-sizing:border-box;
    padding:17px;
    background:linear-gradient(145deg,#fff,#eaf8f8);
    border:1.5px solid #a6d5da;
    border-radius:18px;
    box-shadow:0 11px 28px rgba(15,91,101,.08);
    display:flex;
    flex-direction:column;
    justify-content:center;
}

.metric .label{
    font:800 .64rem Manrope,sans-serif;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:#6d8c93;
}

.metric .value{
    font:800 1.45rem Manrope,sans-serif;
    color:#123f49;
    margin-top:6px;
    white-space:nowrap;
}

.metric .note{
    font-size:.73rem;
    color:#78959b;
    margin-top:4px;
}

.card{
    background:rgba(255,255,255,.90);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:var(--shadow);
    padding:22px;
}

.card h3{
    font:800 1.28rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 8px;
}

.card p{
    color:#66848b;
    line-height:1.62;
    margin:0;
    font-size:.93rem;
}

.story-grid{
    display:grid;
    grid-template-columns:1.05fr .95fr;
    gap:24px;
    align-items:stretch;
    margin-top:22px;
}

.home-image img{
    width:100%;
    max-height:330px;
    object-fit:cover;
    border-radius:18px;
    border:1.5px solid #a9d6da;
    box-shadow:var(--shadow);
}

.tool-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:12px}

.tool-card{
    min-height:150px;
    padding:20px;
    background:rgba(255,255,255,.86);
    border:1.5px solid #b5dde0;
    border-radius:19px;
    box-shadow:var(--shadow);
}

.tool-card .icon{
    font-size:1.35rem;
    margin-bottom:10px;
}

.tool-card h3{
    font:800 1.05rem Manrope;
    color:#123f49;
    margin:0 0 7px;
}

.tool-card p{
    font-size:.86rem;
    line-height:1.55;
    color:#66848b;
    margin:0;
}

.map-shell{
    background:#dff7f8;
    border-radius:23px;
    padding:4px;
    border:2px solid #8fcbd1;
    box-shadow:0 16px 42px rgba(15,91,101,.09);
    overflow:hidden;
}

.map-title{
    min-height:44px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 12px;
    color:#174b56;
}

.map-title b{
    font-size:.92rem;
}

.map-title span{
    font-size:.70rem;
    color:#6b8b92;
}

.map-legend{
    display:flex;
    align-items:center;
    gap:9px;
    flex-wrap:wrap;
    color:#4f747b;
    font-size:.79rem;
    padding:12px 14px;
}

.legend{
    width:15px;
    height:10px;
    border-radius:4px;
    display:inline-block;
}

.blue{background:#197da5}
.green{background:#22a879}
.red{background:#ef4f5e;border-radius:50%;width:11px;height:11px}

.timeline-card{
    margin:16px 0;
    padding:16px 18px;
    background:rgba(255,255,255,.92);
    border:1.5px solid #a9d6da;
    border-radius:18px;
    box-shadow:var(--shadow);
}

.note{
    margin-top:17px;
    padding:13px 16px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 14px 14px 0;
    color:#52757c;
    font-size:.86rem;
    line-height:1.58;
}

.status{
    margin-top:16px;
    padding:15px 16px;
    border-radius:15px;
    border:2px solid;
}

.status.risk{
    background:#fff0f2;
    border-color:#f06a78;
    color:#9e2d3c;
}

.status.normal{
    background:#eafaf4;
    border-color:#49b995;
    color:#176f58;
}

.status-title{
    font:800 .92rem Manrope;
}

.result-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px;
    margin-top:15px;
}

.result-item{
    padding:12px 13px;
    background:#f5fcfc;
    border:1px solid #c7e5e7;
    border-radius:13px;
}

.result-item span{
    display:block;
    font-size:.70rem;
    color:#78959b;
    margin-bottom:4px;
}

.result-item b{
    color:#194b55;
    font-size:.91rem;
}

.signal-grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:10px;
    margin-top:16px;
}

.signal{
    padding:12px 13px;
    border-radius:14px;
    background:#f7fcfc;
    border:1px solid #c9e5e7;
}

.signal .label{
    font-size:.69rem;
    color:#78959b;
}

.signal .value{
    font:800 .92rem Manrope;
    color:#164a55;
    margin-top:4px;
}

.chart-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:18px;
    margin-top:18px;
}

.chart-card{
    background:#fff;
    border:1.5px solid #b6dfe2;
    border-radius:20px;
    box-shadow:var(--shadow);
    padding:14px;
}

.chart-title{
    font:800 .95rem Manrope;
    color:#123f49;
    padding:4px 4px 10px;
}

/* Plotly chart labels: keep titles and ticks dark and readable. */
[data-testid="stPlotlyChart"]{
    color:#123f49 !important;
}


.hotspot-table{
    border-radius:18px;
    overflow:hidden;
    border:1px solid #c5e3e5;
    box-shadow:var(--shadow);
    background:#fff;
}

.hotspot-table table{
    width:100%;
    border-collapse:collapse;
    font-size:.87rem;
}

.hotspot-table th{
    background:#0e5663;
    color:#fff;
    text-align:left;
    padding:11px 13px;
}

.hotspot-table td{
    padding:10px 13px;
    border-top:1px solid #e3eeee;
    color:#315b63;
    background:#fff;
}

.download-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:18px;
    margin-top:18px;
}

.download-card{
    min-height:145px;
    padding:20px;
    border-radius:19px;
    background:linear-gradient(135deg,#087b8b,#13adb3);
    border:2px solid #07576a;
    box-shadow:0 13px 28px rgba(8,91,102,.16);
    color:#fff;
}

.download-card h3{
    font:800 1.12rem Manrope;
    color:#fff;
    margin:0 0 6px;
}

.download-card p{
    font-size:.84rem;
    color:#e5ffff;
    line-height:1.45;
    margin:0;
}

.mini-label{
    font:800 .64rem Manrope,sans-serif;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:#6d8c93;
    margin-bottom:7px;
}

.footer{
    border-top:1px solid #d5ebed;
    margin-top:42px;
    padding-top:16px;
    color:#76959b;
    font-size:.70rem;
}

.stNumberInput input{
    border:2px solid #164e5b!important;
    border-radius:12px!important;
    background:#fff!important;
}

.stSlider label{
    color:#174b56!important;
    font-weight:800!important;
}

/* FINAL SPACING + FORM VISIBILITY */
.section{padding:48px 0 28px!important;margin-top:10px!important}
.nav-wrap{margin:22px 0 42px!important}
.hero{margin-bottom:30px!important}
.metric{margin-bottom:2px!important}
.story-grid{margin-top:30px!important;margin-bottom:42px!important}
.card{margin-bottom:0!important}
.chart-grid{margin-top:28px!important;gap:28px!important}
.chart-card{padding:18px!important}
.stNumberInput label,.stNumberInput label p{color:#174b56!important;font-weight:800!important;font-size:.82rem!important}
.stNumberInput input{color:#123f49!important;font-size:1rem!important;font-weight:700!important;caret-color:#123f49!important}
.stNumberInput div[data-baseweb="input"]{background:#fff!important;border-radius:12px!important}
[data-testid="stMetric"]{background:#f7fcfc!important;border:1px solid #c9e5e7!important;border-radius:14px!important;padding:12px 14px!important}
[data-testid="stMetricLabel"]{color:#6d8c93!important}
[data-testid="stMetricValue"]{color:#123f49!important}
[data-testid="stFormSubmitButton"] button{margin-top:8px!important}
@media(max-width:950px){
    .block-container{padding:18px 18px 55px!important}
    .tool-grid,.story-grid,.chart-grid,.download-grid{grid-template-columns:1fr 1fr}
    .signal-grid{grid-template-columns:1fr 1fr}
}

@media(max-width:620px){
    .tool-grid,.story-grid,.chart-grid,.download-grid,.signal-grid{grid-template-columns:1fr}
    .hero{padding:38px 28px;min-height:320px}
    .hero h1{font-size:3rem}
    .block-container{padding:14px 12px 45px!important}
    .latest-label{display:none}
    .section h2{font-size:2.15rem}
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SMALL HELPERS
# ============================================================

def metric(label, value, note):
    st.markdown(
        f"""
        <div class="metric">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(kicker, title, copy):
    st.markdown(
        f"""
        <div class="section">
            <div class="kicker">{kicker}</div>
            <h2>{title}</h2>
            <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(value, digits=4):
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):.{digits}f}"


def pct(value):
    if pd.isna(value):
        return "Unavailable"
    x = float(value)
    if x <= 1:
        x *= 100
    return f"{x:.1f}%"


def region_name(lat, lon):
    if 5 <= lat <= 30 and 45 <= lon <= 75:
        return "Arabian Sea"
    if 0 <= lat <= 25 and 75 < lon <= 100:
        return "Bay of Bengal"
    if -30 <= lat < 5 and 40 <= lon <= 100:
        return "Southern Indian Ocean"
    if 5 <= lat <= 30 and 75 < lon <= 120:
        return "Northern Indian Ocean"
    return "Other study area"


def nearest_row(lat, lon):
    """
    Safe nearest-cell lookup.
    This intentionally avoids applying float() to a whole pandas Series,
    which was the source of the previous Location-page TypeError.
    """
    work = latest.copy()

    lat_arr = work["latitude"].to_numpy(dtype=float)
    lon_arr = work["longitude"].to_numpy(dtype=float)

    target_lat = float(lat)
    target_lon = float(lon)

    # Approximate geographic distance. Good enough for nearest grid cell.
    lon_scale = max(float(np.cos(np.deg2rad(target_lat))), 0.25)

    distance = (
        (lat_arr - target_lat) ** 2
        + ((lon_arr - target_lon) * lon_scale) ** 2
    )

    return work.iloc[int(np.argmin(distance))]


def field_for_date(date_value):
    date_value = pd.Timestamp(date_value)

    field = df[df["date"].eq(date_value)].copy()

    return field[
        field["latitude"].between(LAT_MIN, LAT_MAX)
        & field["plot_lon"].between(LON_MIN, LON_MAX)
    ].copy()


# ============================================================
# MAP
# ============================================================

def make_latest_map(field):
    """
    Stable Scattergeo map.
    No numpy clipping on Plotly marker sizes.
    No update_geos projection tricks.
    """
    fig = go.Figure()

    if field.empty:
        return fig

    # Background processed cells.
    background = field.copy()

    background["lat_plot"] = background["latitude"]
    background["lon_plot"] = background["plot_lon"]

    fig.add_trace(
        go.Scattergeo(
            lat=background["lat_plot"],
            lon=background["lon_plot"],
            mode="markers",
            marker=dict(
                size=5,
                color="#197da5",
                opacity=0.42,
            ),
            text=[
                f"Chl-a: {v:.4f}"
                for v in background["chla"].astype(float)
            ],
            hovertemplate=(
                "<b>Processed cell</b><br>"
                "Latitude: %{lat:.4f}<br>"
                "Longitude: %{lon:.4f}<br>"
                "%{text}<extra></extra>"
            ),
            name="Processed field",
        )
    )

    # Green concentration layer using 2° bins.
    flagged = field[field["risk_flag"]].copy()

    if not flagged.empty:
        flagged["lat_zone"] = np.floor(flagged["latitude"] / 2) * 2 + 1
        flagged["lon_zone"] = np.floor(flagged["plot_lon"] / 2) * 2 + 1

        zones = (
            flagged.groupby(["lat_zone", "lon_zone"], as_index=False)
            .size()
            .rename(columns={"size": "flagged_cells"})
        )

        fig.add_trace(
            go.Scattergeo(
                lat=zones["lat_zone"],
                lon=zones["lon_zone"],
                mode="markers",
                marker=dict(
                    size=20,
                    color="#22a879",
                    opacity=0.26,
                    line=dict(width=1, color="#22a879"),
                ),
                text=zones["flagged_cells"],
                hovertemplate=(
                    "<b>Flag concentration</b><br>"
                    "Latitude zone: %{lat:.0f}°<br>"
                    "Longitude zone: %{lon:.0f}°<br>"
                    "Flagged cells: %{text}<extra></extra>"
                ),
                name="Flag concentration",
            )
        )

        # Red individual risk cells.
        fig.add_trace(
            go.Scattergeo(
                lat=flagged["latitude"],
                lon=flagged["plot_lon"],
                mode="markers",
                marker=dict(
                    size=7,
                    color="#ef4f5e",
                    opacity=0.95,
                    line=dict(width=0.7, color="#ffffff"),
                ),
                text=[
                    f"Chl-a: {v:.4f}"
                    for v in flagged["chla"].astype(float)
                ],
                hovertemplate=(
                    "<b>Potential bloom-risk screening cell</b><br>"
                    "Latitude: %{lat:.4f}<br>"
                    "Longitude: %{lon:.4f}<br>"
                    "%{text}<extra></extra>"
                ),
                name="Potential bloom risk",
            )
        )

    fig.update_geos(
        projection_type="equirectangular",
        showland=True,
        landcolor="#dce9e7",
        showocean=True,
        oceancolor="#dff7f8",
        showcoastlines=True,
        coastlinecolor="#5b9198",
        showcountries=True,
        countrycolor="#9ab9be",
        bgcolor="#dff7f8",
        lataxis=dict(range=[LAT_MIN, LAT_MAX]),
        lonaxis=dict(range=[LON_MIN, LON_MAX]),
    )

    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=5, b=5),
        paper_bgcolor="#dff7f8",
        showlegend=False,
        font=dict(color="#174b56"),
        geo=dict(
            center=dict(lat=-5, lon=70),
            projection_scale=1.0,
        ),
    )

    return fig


# ============================================================
# HOTSPOT TABLE
# ============================================================

def hotspot_table(field):
    flagged = field[field["risk_flag"]].copy()

    if flagged.empty:
        return pd.DataFrame()

    flagged["lat_zone"] = np.floor(flagged["latitude"] / 2) * 2 + 1
    flagged["lon_zone"] = np.floor(flagged["plot_lon"] / 2) * 2 + 1

    zones = (
        flagged.groupby(["lat_zone", "lon_zone"], as_index=False)
        .agg(
            flagged_cells=("risk_flag", "size"),
            mean_chla=("chla", "mean"),
            max_chla=("chla", "max"),
        )
        .sort_values("flagged_cells", ascending=False)
        .head(12)
    )

    zones["Investigation zone"] = zones.apply(
        lambda r: (
            f"{r['lat_zone']:.0f}°–{r['lat_zone']+2:.0f}°, "
            f"{r['lon_zone']:.0f}°–{r['lon_zone']+2:.0f}°"
        ),
        axis=1,
    )

    out = zones[
        ["Investigation zone", "flagged_cells", "mean_chla", "max_chla"]
    ].copy()

    out.columns = [
        "Investigation zone",
        "Potential-risk cells",
        "Mean Chl-a",
        "Max Chl-a",
    ]

    out["Mean Chl-a"] = out["Mean Chl-a"].round(4)
    out["Max Chl-a"] = out["Max Chl-a"].round(4)

    return out


# ============================================================
# TOP BRAND
# ============================================================

st.markdown(
    f"""
    <div class="brand-bar">
        <div class="brand-left">
            <div class="logo">≈</div>
            <div>
                <div class="brand-name">BloomDetect AI</div>
                <div class="brand-sub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div>
            </div>
        </div>
        <div class="latest-label">
            Latest processed field<br>
            <b>{latest_date.strftime("%d %b %Y")}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION BUTTONS
# ============================================================

st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)

nav_cols = st.columns(5, gap="small")

for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(
            NAV_LABELS[page],
            key=f"nav_{page}",
            width="stretch",
            type="primary" if st.session_state.page == page else "secondary",
        ):
            navigate(page)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="kicker">EOS-06 · OCM-3 · SATELLITE INTELLIGENCE</div>
                <h1>Read the ocean signal.</h1>
                <p>
                    BloomDetect AI turns satellite-derived chlorophyll-a observations
                    into an early-warning support view for identifying locations
                    whose patterns may deserve closer investigation.
                </p>
                <div class="hero-badges">
                    <span class="badge">🌊 Ocean colour</span>
                    <span class="badge">🛰 EOS-06 OCM-3</span>
                    <span class="badge">🌱 Potential bloom-risk screening</span>
                    <span class="badge">📍 Spatial intelligence</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(latest)
    flags = int(latest["risk_flag"].sum())
    share = 100 * flags / total if total else 0
    max_chla = latest["chla"].max()

    m1, m2, m3, m4 = st.columns(4, gap="medium")

    with m1:
        metric("Processed cells", f"{total:,}", "latest field")
    with m2:
        metric("Potential-risk cells", f"{flags:,}", "screening output")
    with m3:
        metric("Risk share", f"{share:.2f}%", "of processed cells")
    with m4:
        metric("Maximum Chl-a", fmt(max_chla), "latest field")

    left, right = st.columns([1.0, 1.0], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="mini-label">FROM SATELLITE OBSERVATION TO INVESTIGATION SUPPORT</div>
                <h3>Chlorophyll-a is a signal, not a verdict.</h3>
                <p>
                    BloomDetect combines satellite-derived chlorophyll-a with
                    temporal context to screen for unusual patterns. The result is
                    a potential bloom-risk screening signal, not confirmation of a
                    harmful algal bloom.
                </p>
                <div class="note">
                    <b>Scientific note:</b> high chlorophyll-a alone does not prove
                    a harmful algal bloom. Species, toxin presence and ecological
                    impact require additional evidence and field validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if IMAGE_PATH.exists():
            st.image(
                str(IMAGE_PATH),
                caption="How satellite ocean-colour observations can support bloom-risk investigation",
                width="stretch",
            )
        else:
            st.markdown(
                """
                <div class="card">
                    <h3>EOS-06 ocean-colour observation</h3>
                    <p>
                        Add bloomdetect_bloom_process.png beside app.py to display
                        the project illustration here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )



# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":

    section(
        "01 · SPATIAL INTELLIGENCE",
        "See the current field clearly.",
        "Explore the latest processed EOS-06 OCM-3 field and the current potential bloom-risk screening output.",
    )

    selected_date = latest_date
    selected_full = field_for_date(selected_date)
    processed_count = len(selected_full)
    risk_count = int(selected_full["risk_flag"].sum())
    share = 100 * risk_count / processed_count if processed_count else 0

    a, b, c, d = st.columns(4, gap="medium")
    with a:
        metric("Observation date", pd.Timestamp(selected_date).strftime("%d %b %Y"), "latest processed field")
    with b:
        metric("Processed cells", f"{processed_count:,}", "selected field")
    with c:
        metric("Potential-risk", f"{risk_count:,}", "screening layer")
    with d:
        metric("Risk share", f"{share:.2f}%", "selected field")

    st.markdown(
        """
        <div class="map-shell">
            <div class="map-title">
                <b>Indian Ocean · Arabian Sea · Bay of Bengal</b>
                <span>Blue field · green concentration · red screening flags</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        make_latest_map(selected_full),
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    st.markdown(
        """
            <div class="map-legend">
                <span class="legend blue"></span><span>Blue = processed ocean field</span>
                <span class="legend green"></span><span>Green = flagged-cell concentration</span>
                <span class="legend red"></span><span>Red = individual potential-risk cell</span>
            </div>
        </div>
        <div class="note">
            <b>Reading the map:</b> green areas show spatial concentration of screening flags.
            Red cells are potential-risk screening results, not confirmed harmful algal blooms.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page == "location":

    section(
        "02 · LOCATION INTELLIGENCE",
        "Check one coordinate across the available evidence.",
        "Your input stays separate from the nearest valid 0.25° satellite cell, so it is always clear which location was evaluated.",
    )

    if "checked_lat" not in st.session_state:
        st.session_state.checked_lat = 17.40
    if "checked_lon" not in st.session_state:
        st.session_state.checked_lon = 78.50

    left, right = st.columns([0.86, 1.14], gap="large")

    with left:
        st.markdown('<div class="card"><div class="mini-label">YOUR INPUT</div><h3>Coordinates</h3><p>Enter decimal degrees. Example: 17.38, 78.49.</p></div>', unsafe_allow_html=True)
        st.write("")
        with st.form("location_form", clear_on_submit=False):
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=float(st.session_state.checked_lat), step=0.25, format="%.4f")
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=float(st.session_state.checked_lon), step=0.25, format="%.4f")
            submitted = st.form_submit_button("🔎 Check location", width="stretch", type="primary")
        if submitted:
            st.session_state.checked_lat = float(lat)
            st.session_state.checked_lon = float(lon)
            st.rerun()

    lat = float(st.session_state.checked_lat)
    lon = float(st.session_state.checked_lon)
    row = nearest_row(lat, lon)
    flagged = bool(row["risk_flag"])
    risk_probability = row.get("risk_probability", np.nan)

    with right:
        st.markdown('<div class="card"><div class="mini-label">NEAREST PROCESSED OCEAN CELL</div><h3>Nearest satellite observation</h3><p>This is the processed 0.25° grid cell actually used for the lookup.</p></div>', unsafe_allow_html=True)
        st.write("")
        r1, r2 = st.columns(2, gap="small")
        with r1:
            st.metric("Processed latitude", f"{float(row['latitude']):.4f}°")
        with r2:
            st.metric("Processed longitude", f"{float(row['longitude']):.4f}°")
        st.caption(f"Your input: {lat:.4f}° · {lon:.4f}°")
        r3, r4 = st.columns(2, gap="small")
        with r3:
            st.metric("Observation date", row["date"].strftime("%d %b %Y"))
        with r4:
            st.metric("Region", region_name(float(row["latitude"]), float(row["longitude"])))
        r5, r6 = st.columns(2, gap="small")
        with r5:
            st.metric("Chlorophyll-a", fmt(row["chla"]))
        with r6:
            st.metric("Risk probability", pct(risk_probability))

        if flagged:
            st.error("🔴 POTENTIAL BLOOM-RISK FLAG\n\nThis processed cell is included in the current screening shortlist.")
        else:
            st.success("🟢 NOT FLAGGED\n\nThis processed cell is not included in the current potential-risk shortlist.")

    st.markdown('<div class="mini-label" style="margin-top:34px">SUPPORTING SIGNALS</div>', unsafe_allow_html=True)
    signals = [
        ("Current Chl-a", fmt(row["chla"])),
        ("Historical baseline", fmt(row.get("historical_baseline", np.nan))),
        ("Anomaly", fmt(row.get("chla_anomaly", np.nan))),
        ("Recent change", fmt(row.get("chla_change", np.nan))),
    ]
    cols = st.columns(4, gap="small")
    for col, (label, value) in zip(cols, signals):
        with col:
            st.markdown(f'<div class="signal"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

    report = pd.DataFrame([{
        "input_latitude": lat, "input_longitude": lon,
        "nearest_processed_latitude": row["latitude"], "nearest_processed_longitude": row["longitude"],
        "date": row["date"].strftime("%Y-%m-%d"), "chla": row["chla"],
        "historical_baseline": row.get("historical_baseline", np.nan),
        "chla_anomaly": row.get("chla_anomaly", np.nan), "chla_change": row.get("chla_change", np.nan),
        "risk_label": row["risk_label"], "risk_probability": row.get("risk_probability", np.nan),
    }])
    st.download_button("⬇ Download location report", report.to_csv(index=False).encode("utf-8"), "bloomdetect_location_report.csv", "text/csv", width="stretch")


# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":

    section(
        "03 · INSIGHTS",
        "See the patterns behind the screening output.",
        "Hotspot concentration and data insights are combined here. This page does not create another independent risk score.",
    )

    flags = int(latest["risk_flag"].sum())
    normal_count = len(latest) - flags
    share = 100 * flags / len(latest) if len(latest) else 0

    cols = st.columns(4, gap="medium")

    values = [
        ("Observation cells", f"{len(latest):,}", "latest field"),
        ("Potential-risk", f"{flags:,}", "screening output"),
        ("Normal", f"{normal_count:,}", "latest field"),
        ("Risk share", f"{share:.2f}%", "latest field"),
    ]

    for col, (label, value, note) in zip(cols, values):
        with col:
            metric(label, value, note)

    zones = hotspot_table(latest)

    section(
        "HOTSPOT INTELLIGENCE",
        "Where are the flags concentrating?",
        "Nearby screening flags are grouped into 2° × 2° investigation zones. This is a spatial concentration view, not a severity ranking.",
    )

    if zones.empty:

        st.markdown(
            """
            <div class="card">
                <h3>No potential-risk cells in the latest study area.</h3>
                <p>
                    The latest processed field contains no screening flags inside
                    the selected study window.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        flagged = latest[latest["risk_flag"]].copy()

        flagged["lat_zone"] = (
            np.floor(flagged["latitude"] / 2) * 2 + 1
        )
        flagged["lon_zone"] = (
            np.floor(flagged["plot_lon"] / 2) * 2 + 1
        )

        plot_zones = (
            flagged.groupby(
                ["lat_zone", "lon_zone"], as_index=False
            )
            .size()
            .rename(columns={"size": "flagged_cells"})
            .sort_values("flagged_cells", ascending=False)
            .head(12)
        )

        plot_zones["zone"] = plot_zones.apply(
            lambda r: (
                f"{r['lat_zone']:.0f}°–{r['lat_zone']+2:.0f}°, "
                f"{r['lon_zone']:.0f}°–{r['lon_zone']+2:.0f}°"
            ),
            axis=1,
        )

        fig = px.bar(
            plot_zones,
            x="flagged_cells",
            y="zone",
            orientation="h",
            text="flagged_cells",
        )

        fig.update_traces(
            marker_color="#22a879",
            textposition="outside",
            textfont=dict(color="#123f49", size=13),
            cliponaxis=False,
        )

        fig.update_layout(
            height=430,
            margin=dict(l=190, r=65, t=25, b=75),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Potential-risk cells in zone",
                title_font=dict(color="#123f49", size=15),
                tickfont=dict(color="#315f67", size=12),
                gridcolor="#d6e7e8",
                automargin=True,
            ),
            yaxis=dict(
                title="Investigation zone",
                title_font=dict(color="#123f49", size=15),
                tickfont=dict(color="#315f67", size=12),
                automargin=True,
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="chart-title">Top investigation zones</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="mini-label" style="margin-top:18px">ZONE DETAILS</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="hotspot-table">{zones.to_html(index=False, border=0)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="mini-label" style="margin-top:42px">CURRENT FIELD</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:

        clean = (
            latest["chla"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        hist_fig = px.histogram(
            pd.DataFrame({"Chlorophyll-a": clean}),
            x="Chlorophyll-a",
            nbins=32,
        )

        hist_fig.update_traces(marker_color="#197da5")

        hist_fig.update_layout(
            height=390,
            margin=dict(l=70, r=25, t=25, b=75),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(color="#123f49", size=13),
            xaxis=dict(
                title="Satellite-derived chlorophyll-a",
                title_font=dict(color="#123f49", size=16),
                tickfont=dict(color="#123f49", size=13),
                gridcolor="#d6e7e8",
                automargin=True,
            ),
            yaxis=dict(
                title="Number of processed cells",
                title_font=dict(color="#123f49", size=16),
                tickfont=dict(color="#123f49", size=13),
                gridcolor="#d6e7e8",
                automargin=True,
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="chart-title">Chlorophyll-a distribution</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            hist_fig,
            width="stretch",
            config={"displaylogo": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:

        normal_mean = (
            latest.loc[~latest["risk_flag"], "chla"].mean()
            if normal_count
            else np.nan
        )

        risk_mean = (
            latest.loc[latest["risk_flag"], "chla"].mean()
            if flags
            else np.nan
        )

        comparison = pd.DataFrame(
            {
                "Screening group": [
                    "Normal",
                    "Potential bloom risk",
                ],
                "Mean chlorophyll-a": [
                    normal_mean,
                    risk_mean,
                ],
            }
        )

        comp_fig = px.bar(
            comparison,
            x="Screening group",
            y="Mean chlorophyll-a",
            text="Mean chlorophyll-a",
        )

        comp_fig.update_traces(
            marker_color=["#197da5", "#ef4f5e"],
            texttemplate="%{text:.4f}",
            textposition="outside",
            textfont=dict(color="#123f49", size=13),
            cliponaxis=False,
        )

        comp_fig.update_layout(
            height=390,
            margin=dict(l=70, r=25, t=25, b=75),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Screening group",
                title_font=dict(color="#123f49", size=15),
                tickfont=dict(color="#315f67", size=12),
                automargin=True,
            ),
            yaxis=dict(
                title="Mean chlorophyll-a",
                title_font=dict(color="#123f49", size=15),
                tickfont=dict(color="#315f67", size=12),
                gridcolor="#d6e7e8",
                automargin=True,
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="chart-title">Mean chlorophyll-a by screening group</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            comp_fig,
            width="stretch",
            config={"displaylogo": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note">
            <b>Interpretation:</b> these charts describe satellite-derived
            chlorophyll-a and the project screening output. They do not independently
            establish harmfulness, species identity or toxin presence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not zones.empty:
        st.download_button(
            "⬇ Download hotspot report",
            zones.to_csv(index=False).encode("utf-8"),
            "bloomdetect_hotspot_report.csv",
            "text/csv",
            width="stretch",
        )


# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    section(
        "04 · DATA",
        "Source data and project outputs.",
        "Everything related to the dataset and downloads stays here, so the other pages remain focused on analysis.",
    )

    cols = st.columns(4, gap="medium")

    info = [
        ("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3"),
        ("Grid", "0.25°", "latitude × longitude"),
        ("Latest cells", f"{len(latest):,}", "processed observation"),
        ("Latest date", latest_date.strftime("%d %b %Y"), "processed dataset"),
    ]

    for col, (label, value, note) in zip(cols, info):
        with col:
            metric(label, value, note)

    st.markdown(
        """
        <div class="card" style="margin-top:18px">
            <div class="mini-label">SOURCE PRODUCT</div>
            <h3>EOS-06 OCM-3 analysed chlorophyll-a</h3>
            <p>
                The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4
                analysed chlorophyll product, E06OCM_L4_AC. The project output
                is a potential bloom-risk screening layer built from the
                satellite-derived chlorophyll-a signal and temporal context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section(
        "DOWNLOADS",
        "Take the actual project outputs.",
        "These downloads are generated directly from the data used by the dashboard.",
    )

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown(
            """
            <div class="download-card">
                <h3>Latest observation table</h3>
                <p>
                    All processed cells from the latest available field,
                    including the stored screening result and supporting signals.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download latest observations",
            latest.to_csv(index=False).encode("utf-8"),
            "bloomdetect_latest_observations.csv",
            "text/csv",
            width="stretch",
        )

    with d2:
        st.markdown(
            """
            <div class="download-card">
                <h3>Potential-risk shortlist</h3>
                <p>
                    Only cells currently screened as potential bloom risk in
                    the latest processed field.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download potential-risk locations",
            latest_risk.to_csv(index=False).encode("utf-8"),
            "bloomdetect_potential_risk.csv",
            "text/csv",
            width="stretch",
        )

    section(
        "SCIENTIFIC USE",
        "Use the screening result correctly.",
        "",
    )

    st.markdown(
        """
        <div class="card">
            <p>
                <b>Potential bloom risk</b> is a project screening output.
                It is not confirmation of a harmful algal bloom, species identity
                or toxin presence. Satellite observations should be combined with
                field observations and additional environmental evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening ·
        Latest processed field: {latest_date.strftime("%d %B %Y")}
    </div>
    """,
    unsafe_allow_html=True,
)
