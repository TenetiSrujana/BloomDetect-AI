from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Clean single-file Streamlit dashboard
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"

# Optional project images. The app still works if these are absent.
BLOOM_IMAGE = BASE_DIR / "bloomdetect_bloom_process.png"

# Project study window used for the dashboard.
LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# Proxy screening thresholds used when the project label was created.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv was not found next to app.py."
        )

    d = pd.read_csv(DATA_PATH)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            "The CSV is missing these required columns: "
            + ", ".join(sorted(missing))
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for col in ["latitude", "longitude", "chla"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    optional_numeric = [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]
    for col in optional_numeric:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(
        subset=["latitude", "longitude", "date", "chla"]
    ).copy()

    d["risk_flag"] = (
        d["risk_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("potential bloom risk")
    )

    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180

    return d


try:
    df = load_data()
except Exception as exc:
    st.error("BloomDetect AI could not load the dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()
risk = latest[latest["risk_flag"]].copy()

# ------------------------------------------------------------
# PAGE STATE
# ------------------------------------------------------------

PAGES = ["home", "explore", "map", "location", "hotspots", "insights", "data"]

NAV = {
    "home": "⌂ Home",
    "explore": "◈ Explore",
    "map": "🗺 Risk Map",
    "location": "📍 Location",
    "hotspots": "🔥 Hotspots",
    "insights": "📊 Insights",
    "data": "⇩ Data",
}

if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"

# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------

st.markdown(
    r"""
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

html, body, [data-testid="stAppViewContainer"]{
    background:#f1fbfb !important;
    color:var(--ink) !important;
    font-family:'DM Sans',sans-serif !important;
}

.stApp{
    background:
        radial-gradient(circle at 10% 12%, rgba(55,211,210,.08), transparent 25%),
        radial-gradient(circle at 90% 72%, rgba(19,156,175,.07), transparent 28%),
        linear-gradient(180deg,#fbffff 0%,#effafa 52%,#fbffff 100%) !important;
    overflow-x:hidden;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer,
[data-testid="stSidebar"]{
    display:none !important;
}

.block-container{
    max-width:1260px !important;
    padding:24px 34px 70px !important;
}

/* subtle water motion */
.stApp:before{
    content:"";
    position:fixed;
    left:-10%;
    right:-10%;
    bottom:-170px;
    height:300px;
    z-index:0;
    pointer-events:none;
    background:
        radial-gradient(ellipse at 18% 55%,rgba(24,193,201,.12) 0 17%,transparent 18%),
        radial-gradient(ellipse at 56% 42%,rgba(69,221,218,.09) 0 20%,transparent 21%),
        radial-gradient(ellipse at 88% 60%,rgba(15,171,191,.10) 0 18%,transparent 19%);
    animation:waterMove 13s ease-in-out infinite alternate;
}

@keyframes waterMove{
    from{transform:translateX(-2%)}
    to{transform:translateX(2%)}
}

/* Header */
.brand-bar{
    position:relative;
    z-index:10;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    padding:15px 20px;
    border:1.5px solid #c5e4e6;
    background:rgba(255,255,255,.90);
    border-radius:22px;
    box-shadow:var(--shadow);
    backdrop-filter:blur(18px);
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
    color:white;
    font-size:20px;
    font-weight:800;
    box-shadow:0 10px 24px rgba(9,144,157,.18);
}

.brand-name{
    font:800 1.22rem Manrope,sans-serif;
    color:#103f49;
    letter-spacing:-.03em;
}

.brand-sub{
    font-size:.73rem;
    color:#78959b;
    margin-top:2px;
}

.latest-label{
    text-align:right;
    color:#78959b;
    font-size:.68rem;
    line-height:1.35;
}

.latest-label b{
    color:#174b56;
    font-size:.78rem;
}

/* Navigation */
.nav-wrap{
    position:relative;
    z-index:12;
    margin:13px 0 6px;
}

.stButton>button,
.stDownloadButton>button,
.stFormSubmitButton>button{
    min-height:45px !important;
    border-radius:14px !important;
    background:#ffffff !important;
    border:2px solid #164e5b !important;
    color:#123f49 !important;
    font-weight:800 !important;
    font-size:.87rem !important;
    box-shadow:0 6px 16px rgba(15,70,82,.09) !important;
    transition:all .18s ease !important;
}

.stButton>button:hover,
.stDownloadButton>button:hover,
.stFormSubmitButton>button:hover{
    background:#e2f8f8 !important;
    border-color:#087d8c !important;
    transform:translateY(-1px) !important;
    box-shadow:0 10px 22px rgba(15,91,101,.13) !important;
}

.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#d8f7f7,#b9eeee) !important;
    border-color:#078c9b !important;
    color:#0a5966 !important;
    box-shadow:0 8px 20px rgba(8,140,155,.16),0 0 0 2px rgba(16,174,183,.08) !important;
}

/* Typography */
.section{
    position:relative;
    z-index:2;
    padding:34px 0 18px;
}

.kicker{
    font:800 .68rem Manrope,sans-serif;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:11px;
}

.section h2{
    font:800 clamp(2.15rem,4vw,3.65rem)/1.03 Manrope,sans-serif;
    letter-spacing:-.06em;
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

/* Hero */
.hero{
    position:relative;
    z-index:2;
    overflow:hidden;
    min-height:430px;
    border-radius:30px;
    padding:62px 58px;
    background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);
    box-shadow:0 28px 72px rgba(6,86,100,.16);
}

.hero:before{
    content:"";
    position:absolute;
    inset:-25%;
    background:repeating-radial-gradient(
        ellipse at 20% 115%,
        transparent 0 55px,
        rgba(181,255,251,.12) 57px 59px,
        transparent 61px 105px
    );
    transform:rotate(-7deg);
    animation:waveLines 14s linear infinite;
}

.hero:after{
    content:"";
    position:absolute;
    left:-5%;
    right:-5%;
    bottom:-135px;
    height:275px;
    background:rgba(142,244,237,.14);
    border-radius:50%;
    animation:heroWave 8s ease-in-out infinite alternate;
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
    font:800 clamp(3.0rem,6vw,5.8rem)/.91 Manrope,sans-serif;
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

@keyframes waveLines{
    from{transform:translateX(-4%) rotate(-7deg)}
    to{transform:translateX(4%) rotate(-7deg)}
}

@keyframes heroWave{
    from{transform:translateX(-2%) rotate(-1deg)}
    to{transform:translateX(2%) rotate(1deg)}
}

/* Cards */
.card{
    position:relative;
    z-index:2;
    background:rgba(255,255,255,.88);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:var(--shadow);
    padding:22px;
    backdrop-filter:blur(12px);
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

.metric{
    position:relative;
    z-index:2;
    min-height:112px;
    height:100%;
    box-sizing:border-box;
    padding:17px;
    background:linear-gradient(145deg,#ffffff,#eaf8f8);
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

/* Home */
.intro-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:20px;
    margin-top:20px;
}

.image-card{
    padding:9px;
    overflow:hidden;
}

.image-card img{
    display:block;
    width:100%;
    height:315px;
    object-fit:cover;
    border-radius:16px;
}

.feature-grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:15px;
    margin-top:18px;
}

.feature{
    position:relative;
    z-index:2;
    padding:21px;
    background:rgba(255,255,255,.84);
    border:1.5px solid #b5dde0;
    border-radius:19px;
    box-shadow:var(--shadow);
    min-height:165px;
    transition:.2s;
}

.feature:hover{
    transform:translateY(-3px);
}

.feature .icon{
    font-size:1.35rem;
    margin-bottom:10px;
}

.feature h3{
    font:800 1.05rem Manrope;
    color:#123f49;
    margin:0 0 7px;
}

.feature p{
    font-size:.87rem;
    line-height:1.56;
    color:#66848b;
    margin:0;
}

/* map */
.map-card{
    position:relative;
    z-index:2;
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
    gap:15px;
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

/* explorer */
.module-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:16px;
    margin-top:18px;
}

.module{
    min-height:205px;
    position:relative;
    z-index:2;
    background:rgba(255,255,255,.87);
    border:1.5px solid #abd8dc;
    border-radius:20px;
    box-shadow:var(--shadow);
    padding:22px;
}

.module h3{
    margin:0 0 8px;
    font:800 1.18rem Manrope;
    color:#123f49;
}

.module p{
    margin:0 0 16px;
    color:#66848b;
    font-size:.88rem;
    line-height:1.58;
}

/* location */
.location-grid{
    display:grid;
    grid-template-columns:.82fr 1.18fr;
    gap:20px;
    align-items:start;
}

.input-card{
    padding:22px;
}

.result-card{
    padding:22px;
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

.note{
    position:relative;
    z-index:2;
    margin-top:17px;
    padding:13px 16px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 14px 14px 0;
    color:#52757c;
    font-size:.86rem;
    line-height:1.58;
}

/* hotspot */
.hotspot-table{
    position:relative;
    z-index:2;
    border-radius:18px;
    overflow:hidden;
    border:1px solid #c5e3e5;
    box-shadow:var(--shadow);
    background:white;
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

.hotspot-table tr:nth-child(even) td{
    background:#f7fcfc;
}

/* insight cards */
.insight-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:18px;
    margin-top:18px;
}

.chart-card{
    position:relative;
    z-index:2;
    background:#fff;
    border:1.5px solid #b6dfe2;
    border-radius:20px;
    box-shadow:var(--shadow);
    padding:16px;
}

/* downloads */
.download-card{
    min-height:138px;
    box-sizing:border-box;
    padding:20px;
    border-radius:19px;
    background:linear-gradient(135deg,#087b8b,#13adb3);
    border:2px solid #07576a;
    box-shadow:0 13px 28px rgba(8,91,102,.16);
    color:white;
}

.download-card h3{
    font:800 1.12rem Manrope;
    color:white;
    margin:0 0 6px;
}

.download-card p{
    font-size:.84rem;
    color:#e5ffff;
    line-height:1.45;
    margin:0;
}

/* small label */
.mini-label{
    font:800 .64rem Manrope,sans-serif;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:#6d8c93;
    margin-bottom:7px;
}

/* footer */
.footer{
    position:relative;
    z-index:2;
    border-top:1px solid #d5ebed;
    margin-top:42px;
    padding-top:16px;
    color:#76959b;
    font-size:.70rem;
}

/* remove Streamlit widget label spacing where possible */
div[data-testid="stNumberInput"]{
    margin-bottom:8px;
}

/* mobile */
@media(max-width:950px){
    .block-container{padding:18px 18px 55px !important}
    .feature-grid{grid-template-columns:1fr 1fr}
    .intro-grid,.location-grid,.insight-grid{grid-template-columns:1fr}
    .module-grid{grid-template-columns:1fr 1fr}
    .signal-grid{grid-template-columns:1fr 1fr}
    .hero{padding:48px 35px}
}

@media(max-width:620px){
    .feature-grid,.module-grid,.signal-grid{grid-template-columns:1fr}
    .hero h1{font-size:3rem}
    .hero{padding:42px 25px}
    .block-container{padding:14px 12px 45px !important}
    .brand-bar{padding:13px}
    .latest-label{display:none}
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

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


def safe_num(value, digits=4):
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):.{digits}f}"


def probability_text(value):
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


def in_study_area(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def nearest_row(lat, lon):
    a = latest
    dist = (a["latitude"].to_numpy() - lat) ** 2 + (
        a["longitude"].to_numpy() - lon
    ) ** 2
    return a.iloc[int(np.argmin(dist))]


def study_data():
    return latest[
        latest["latitude"].between(LAT_MIN, LAT_MAX)
        & latest["plot_lon"].between(LON_MIN, LON_MAX)
    ].copy()


def make_risk_map():
    """
    Responsive full study-area map:
    BLUE  = normal processed ocean field
    GREEN = concentration zones containing flagged cells
    RED   = individual potential bloom-risk cells
    """
    data = study_data()

    if data.empty:
        data = latest.copy()

    # Blue base field: aggregate to ~1° cells.
    blue = data.copy()
    blue["lat_bin"] = np.floor(blue["latitude"]).astype(float) + 0.5
    blue["lon_bin"] = np.floor(blue["plot_lon"]).astype(float) + 0.5

    blue = (
        blue.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(mean_chla=("chla", "mean"), cells=("chla", "size"))
    )

    fig = px.scatter_geo(
        blue,
        lat="lat_bin",
        lon="lon_bin",
        custom_data=["lat_bin", "lon_bin", "mean_chla", "cells"],
    )

    fig.update_traces(
        marker=dict(
            size=8,
            color="#197da5",
            opacity=0.72,
            line=dict(width=0),
        ),
        name="Normal processed field",
        hovertemplate=(
            "<b>Normal processed field</b><br>"
            "Latitude: %{customdata[0]:.1f}°<br>"
            "Longitude: %{customdata[1]:.1f}°<br>"
            "Mean Chl-a: %{customdata[2]:.4f}<br>"
            "Cells represented: %{customdata[3]}<extra></extra>"
        ),
    )

    risk_data = data[data["risk_flag"]].copy()

    if not risk_data.empty:
        risk_data["lat_zone"] = np.floor(risk_data["latitude"] / 2) * 2 + 1
        risk_data["lon_zone"] = np.floor(risk_data["plot_lon"] / 2) * 2 + 1

        zones = (
            risk_data.groupby(["lat_zone", "lon_zone"], as_index=False)
            .agg(flagged_cells=("risk_flag", "size"), mean_chla=("chla", "mean"))
        )

        zone_fig = px.scatter_geo(
            zones,
            lat="lat_zone",
            lon="lon_zone",
            size="flagged_cells",
            custom_data=["lat_zone", "lon_zone", "flagged_cells", "mean_chla"],
        )

        zone_fig.update_traces(
            marker=dict(
                color="#22a879",
                opacity=0.58,
                line=dict(width=2, color="#ffffff"),
                sizemin=7,
            ),
            name="Flag concentration zone",
            hovertemplate=(
                "<b>Flag concentration zone</b><br>"
                "Zone centre: %{customdata[0]:.1f}°, %{customdata[1]:.1f}°<br>"
                "Flagged cells: %{customdata[2]}<br>"
                "Mean Chl-a: %{customdata[3]:.4f}<extra></extra>"
            ),
        )

        for trace in zone_fig.data:
            fig.add_trace(trace)

        risk_fig = px.scatter_geo(
            risk_data,
            lat="latitude",
            lon="plot_lon",
            custom_data=["latitude", "longitude", "chla"],
        )

        risk_fig.update_traces(
            marker=dict(
                size=7.5,
                color="#ef4f5e",
                opacity=0.96,
                line=dict(width=1.2, color="#ffffff"),
            ),
            name="Potential bloom-risk cell",
            hovertemplate=(
                "<b>Potential bloom-risk screening flag</b><br>"
                "Latitude: %{customdata[0]:.2f}°<br>"
                "Longitude: %{customdata[1]:.2f}°<br>"
                "Chl-a: %{customdata[2]:.4f}<extra></extra>"
            ),
        )

        for trace in risk_fig.data:
            fig.add_trace(trace)

    fig.update_geos(
        showland=True,
        landcolor="#dce9e7",
        showocean=True,
        oceancolor="#dff7f8",
        showcoastlines=True,
        coastlinecolor="#4d9099",
        coastlinewidth=1.0,
        showcountries=True,
        countrycolor="#9ab9be",
        bgcolor="#dff7f8",
        lataxis_range=[LAT_MIN, LAT_MAX],
        lonaxis_range=[LON_MIN, LON_MAX],
        center=dict(lat=-5, lon=70),
        projection="equirectangular",
    )

    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#dff7f8",
        plot_bgcolor="#dff7f8",
        font=dict(color="#174b56"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,.86)",
            bordercolor="#b9dfe2",
            borderwidth=1,
            font=dict(size=11, color="#174b56"),
        ),
    )

    return fig


def hotspot_table(data):
    r = data[data["risk_flag"]].copy()

    if r.empty:
        return pd.DataFrame()

    r["lat_zone"] = np.floor(r["latitude"] / 2) * 2
    r["lon_zone"] = np.floor(r["plot_lon"] / 2) * 2

    zones = (
        r.groupby(["lat_zone", "lon_zone"], as_index=False)
        .agg(
            flagged_cells=("risk_flag", "size"),
            mean_chla=("chla", "mean"),
            max_chla=("chla", "max"),
        )
        .sort_values(["flagged_cells", "mean_chla"], ascending=False)
        .head(12)
    )

    zones["Zone"] = zones.apply(
        lambda x: f"{x.lat_zone:.0f}°–{x.lat_zone + 2:.0f}°, "
        f"{x.lon_zone:.0f}°–{x.lon_zone + 2:.0f}°",
        axis=1,
    )

    zones["Flagged cells"] = zones["flagged_cells"].astype(int)
    zones["Mean Chl-a"] = zones["mean_chla"].round(4)
    zones["Max Chl-a"] = zones["max_chla"].round(4)

    return zones[["Zone", "Flagged cells", "Mean Chl-a", "Max Chl-a"]]


def image_uri(path):
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    f"""
    <div class="brand-bar">
        <div class="brand-left">
            <div class="logo">≈</div>
            <div>
                <div class="brand-name">BloomDetect AI</div>
                <div class="brand-sub">
                    Coastal & Ocean Intelligence · EOS-06 OCM-3
                </div>
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

st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)

nav_cols = st.columns(len(PAGES), gap="small")

for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(
            NAV[page],
            key=f"nav_{page}",
            width="stretch",
            type="primary" if st.session_state.page == page else "secondary",
        ):
            st.session_state.page = page
            st.rerun()

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
                    into a practical screening dashboard for finding locations whose
                    recent patterns may deserve closer investigation.
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

    st.markdown(
        """
        <div class="section">
            <div class="kicker">WHAT THE PLATFORM DOES</div>
            <h2>One signal. Several useful views.</h2>
            <p>
                Start with the latest satellite field, locate screening flags,
                inspect a coordinate, investigate concentrations and download
                the actual processed outputs. No repeated theory blocks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_left, intro_right = st.columns([1, 1], gap="large")

    with intro_left:
        st.markdown(
            """
            <div class="card" style="min-height:315px">
                <div class="mini-label">WHY IT MATTERS</div>
                <h3>Chlorophyll-a is a signal, not a verdict.</h3>
                <p>
                    Changes in satellite-derived chlorophyll-a can reveal unusual
                    ocean-colour patterns and help identify areas for investigation.
                    BloomDetect uses these signals for <b>potential bloom-risk
                    screening</b>.
                </p>
                <div class="note">
                    <b>Important:</b> high chlorophyll-a alone does not prove a
                    harmful algal bloom. Species, toxins and ecological impacts
                    require additional evidence and field validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with intro_right:
        uri = image_uri(BLOOM_IMAGE)
        if uri:
            st.markdown(
                f"""
                <div class="card image-card">
                    <img src="{uri}" alt="Bloom formation and satellite observation">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="card" style="min-height:315px;display:grid;place-items:center">
                    <div style="text-align:center;color:#6b8b92">
                        <div style="font-size:3rem">🛰️🌊</div>
                        <b>EOS-06 ocean-colour observation</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section" style="padding-top:34px"><div class="kicker">LATEST FIELD</div></div>',
        unsafe_allow_html=True,
    )

    home_metrics = st.columns(4, gap="medium")
    risk_share = 100 * len(risk) / len(latest) if len(latest) else 0

    for col, (label, value, note) in zip(
        home_metrics,
        [
            ("Processed cells", f"{len(latest):,}", "latest field"),
            ("Potential-risk cells", f"{len(risk):,}", "screening output"),
            ("Risk share", f"{risk_share:.2f}%", "of processed cells"),
            ("Maximum Chl-a", safe_num(latest["chla"].max(), 4), "latest field"),
        ],
    ):
        with col:
            metric(label, value, note)

    st.markdown(
        """
        <div class="section" style="padding-top:34px">
            <div class="kicker">EXPLORE</div>
            <h2>Useful tools, without the clutter.</h2>
            <p>
                Each page has one clear job. The same numbers are not repeated
                everywhere like a badly designed conference poster.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = [
        ("🗺️", "Risk Map", "See the full study area with the normal field, green flag concentrations and red flagged cells."),
        ("📍", "Location", "Check one coordinate and compare your input with the nearest processed satellite cell."),
        ("🔥", "Hotspots", "Group nearby flags into compact investigation zones instead of staring at hundreds of points."),
        ("📊", "Insights", "Compare regional screening patterns and the latest chlorophyll-a distribution."),
    ]

    fcols = st.columns(4, gap="medium")
    for col, (icon, title, desc) in zip(fcols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature">
                    <div class="icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# EXPLORE
# ============================================================

elif st.session_state.page == "explore":

    section(
        "EXPLORE",
        "Choose the analysis you actually need.",
        "The current dataset supports several useful operational views. This page is a clean launcher, not another wall of documentation.",
    )

    modules = [
        (
            "🗺️",
            "Ocean & Bloom Risk",
            "Spatial screening of the latest field with blue normal coverage, green concentration zones and red potential-risk cells.",
            "map",
            "Open Risk Map",
        ),
        (
            "🔥",
            "Hotspot Intelligence",
            "Group flagged cells into 2° × 2° investigation zones and inspect their Chl-a statistics.",
            "hotspots",
            "Open Hotspots",
        ),
        (
            "📍",
            "Location Intelligence",
            "Enter a latitude and longitude and retrieve the nearest processed satellite cell and screening result.",
            "location",
            "Check a Location",
        ),
        (
            "📊",
            "Ocean-Colour Insights",
            "Understand the current Chl-a distribution and compare screening patterns across broad study regions.",
            "insights",
            "Open Insights",
        ),
        (
            "🧾",
            "Data & Outputs",
            "See the actual source product, current dataset statistics and download useful project outputs.",
            "data",
            "Open Data",
        ),
    ]

    cols = st.columns(3, gap="medium")

    for i, item in enumerate(modules):
        icon, title, desc, target, button = item
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="module">
                    <div style="font-size:1.5rem;margin-bottom:10px">{icon}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(button, key=f"explore_{target}", width="stretch"):
                st.session_state.page = target
                st.rerun()

    st.markdown(
        """
        <div class="note">
            <b>Current scope:</b> the present dashboard is built from the
            EOS-06 OCM-3 analysed chlorophyll-a dataset and the project's
            potential bloom-risk screening output. Advanced fisheries,
            temperature, salinity, pollution and water-quality indicators
            should only be added when the corresponding datasets are actually
            available.
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
        "Where is the signal showing up?",
        "The map stays focused on the Indian Ocean study area. Blue represents the latest processed field, green shows concentrations of flagged cells, and red shows individual cells screened as potential bloom risk.",
    )

    data = study_data()
    risk_study = data[data["risk_flag"]]

    m1, m2, m3 = st.columns(3, gap="medium")

    with m1:
        metric("Study area", "20°–120°E", "40°S–30°N")

    with m2:
        metric("Flagged cells", f"{len(risk_study):,}", "latest processed field")

    with m3:
        share = 100 * len(risk_study) / len(data) if len(data) else 0
        metric("Risk share", f"{share:.2f}%", "of study-area cells")

    st.markdown(
        """
        <div class="map-card">
            <div class="map-title">
                <b>Indian Ocean · Arabian Sea · Bay of Bengal</b>
                <span>Latest processed field · spatial screening</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    fig = make_risk_map()

    st.plotly_chart(
        fig,
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
            <span class="legend blue"></span><span>Blue = normal processed field</span>
            <span class="legend green"></span><span>Green = flagged-cell concentration zone</span>
            <span class="legend red"></span><span>Red = individual potential bloom-risk screening flag</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="note">
            <b>Interpretation:</b> green zones are spatial concentrations of
            screening flags, not a separate severity score. Red cells are
            potential-risk screening results, not confirmed harmful algal blooms.
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
        "Check one coordinate.",
        "Enter the location you want to investigate. The dashboard keeps your input separate from the nearest processed 0.25° satellite grid cell.",
    )

    if "checked_lat" not in st.session_state:
        st.session_state.checked_lat = 18.0
    if "checked_lon" not in st.session_state:
        st.session_state.checked_lon = 78.0

    left, right = st.columns([.82, 1.18], gap="large")

    with left:
        st.markdown(
            """
            <div class="card input-card">
                <div class="mini-label">YOUR INPUT</div>
                <h3>Coordinates</h3>
                <p>Use decimal degrees. Example: 17.38, 78.49.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("location_form", clear_on_submit=False):
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.checked_lat),
                step=0.25,
                format="%.2f",
            )

            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.checked_lon),
                step=0.25,
                format="%.2f",
            )

            submitted = st.form_submit_button(
                "🔎 Check location",
                width="stretch",
                type="primary",
            )

        if submitted:
            st.session_state.checked_lat = float(lat)
            st.session_state.checked_lon = float(lon)

    lat = float(st.session_state.checked_lat)
    lon = float(st.session_state.checked_lon)

    row = nearest_row(lat, lon)
    flagged = bool(row["risk_flag"])
    inside = in_study_area(lat, lon)
    probability = probability_text(row.get("risk_probability", np.nan))

    with right:
        st.markdown(
            f"""
            <div class="card result-card">
                <div class="mini-label">NEAREST PROCESSED CELL</div>
                <h3 style="font-size:1.75rem">
                    {float(row.latitude):.4f}° · {float(row.longitude):.4f}°
                </h3>
                <p>
                    Satellite lookup cell actually used for this result.
                    <br>
                    <b>Your input:</b> {lat:.4f}° · {lon:.4f}°
                </p>

                <div class="result-grid">
                    <div class="result-item">
                        <span>Observation date</span>
                        <b>{row.date.strftime("%d %b %Y")}</b>
                    </div>
                    <div class="result-item">
                        <span>Chlorophyll-a</span>
                        <b>{safe_num(row.chla)}</b>
                    </div>
                    <div class="result-item">
                        <span>Risk probability</span>
                        <b>{probability}</b>
                    </div>
                    <div class="result-item">
                        <span>Region</span>
                        <b>{region_name(row.latitude, row.longitude)}</b>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if flagged:
            st.markdown(
                """
                <div class="status risk">
                    <div class="status-title">🔴 POTENTIAL BLOOM-RISK FLAG</div>
                    <div>This processed cell is included in the current screening shortlist.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status normal">
                    <div class="status-title">🟢 NOT FLAGGED</div>
                    <div>This processed cell is not included in the current potential-risk shortlist.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not inside:
            st.markdown(
                """
                <div class="note">
                    <b>Outside dashboard study window:</b> the nearest processed
                    cell is still shown, but the main spatial dashboard is focused
                    on 40°S–30°N and 20°E–120°E.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="mini-label" style="margin-top:22px">SUPPORTING SIGNALS</div>',
        unsafe_allow_html=True,
    )

    signals = [
        ("Current Chl-a", safe_num(row.chla)),
        ("Historical baseline", safe_num(row.get("historical_baseline", np.nan))),
        ("Anomaly", safe_num(row.get("chla_anomaly", np.nan))),
        ("Recent change", safe_num(row.get("chla_change", np.nan))),
    ]

    s_cols = st.columns(4, gap="small")
    for col, (label, value) in zip(s_cols, signals):
        with col:
            st.markdown(
                f"""
                <div class="signal">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    report = pd.DataFrame(
        [
            {
                "input_latitude": lat,
                "input_longitude": lon,
                "nearest_processed_latitude": row.latitude,
                "nearest_processed_longitude": row.longitude,
                "date": row.date.strftime("%Y-%m-%d"),
                "chla": row.chla,
                "historical_baseline": row.get("historical_baseline", np.nan),
                "chla_anomaly": row.get("chla_anomaly", np.nan),
                "chla_change": row.get("chla_change", np.nan),
                "risk_label": row.risk_label,
                "risk_probability": row.get("risk_probability", np.nan),
            }
        ]
    )

    st.download_button(
        "⬇ Download location report",
        report.to_csv(index=False).encode("utf-8"),
        "bloomdetect_location_report.csv",
        "text/csv",
        width="stretch",
    )

# ============================================================
# HOTSPOTS
# ============================================================

elif st.session_state.page == "hotspots":

    section(
        "03 · HOTSPOT INTELLIGENCE",
        "Where are the flags concentrating?",
        "Nearby flagged cells are grouped into broad 2° × 2° zones. The purpose is to turn many individual flags into a manageable investigation list.",
    )

    data = study_data()
    zones = hotspot_table(data)

    if zones.empty:
        st.markdown(
            """
            <div class="card">
                <h3>No potential-risk cells in the current study area.</h3>
                <p>The latest processed field contains no screening flags in the selected study window.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        total_flags = int(data["risk_flag"].sum())
        top_flags = int(zones.iloc[0]["Flagged cells"])

        a, b, c = st.columns(3, gap="medium")
        with a:
            metric("Flagged cells", f"{total_flags:,}", "study area")
        with b:
            metric("Concentration zones", f"{len(zones):,}", "top 12 shown")
        with c:
            metric("Largest zone", f"{top_flags:,}", "flagged cells")

        st.markdown(
            '<div class="section" style="padding-top:28px"><div class="kicker">CONCENTRATIONS</div><h2>Investigation areas.</h2><p>Green concentrations show where screening flags are grouped. They are not a new risk label.</p></div>',
            unsafe_allow_html=True,
        )

        r = data[data["risk_flag"]].copy()
        r["lat_zone"] = np.floor(r["latitude"] / 2) * 2 + 1
        r["lon_zone"] = np.floor(r["plot_lon"] / 2) * 2 + 1

        plot_zones = (
            r.groupby(["lat_zone", "lon_zone"], as_index=False)
            .agg(flagged_cells=("risk_flag", "size"), mean_chla=("chla", "mean"))
            .sort_values("flagged_cells", ascending=False)
            .head(20)
        )

        hfig = px.scatter_geo(
            plot_zones,
            lat="lat_zone",
            lon="lon_zone",
            size="flagged_cells",
            custom_data=["lat_zone", "lon_zone", "flagged_cells", "mean_chla"],
        )

        hfig.update_traces(
            marker=dict(
                color="#22a879",
                opacity=.72,
                line=dict(width=2, color="#ffffff"),
            ),
            hovertemplate=(
                "<b>Investigation zone</b><br>"
                "Centre: %{customdata[0]:.1f}°, %{customdata[1]:.1f}°<br>"
                "Flagged cells: %{customdata[2]}<br>"
                "Mean Chl-a: %{customdata[3]:.4f}<extra></extra>"
            ),
        )

        hfig.update_geos(
            showland=True,
            landcolor="#dce9e7",
            showocean=True,
            oceancolor="#dff7f8",
            showcoastlines=True,
            coastlinecolor="#4d9099",
            showcountries=True,
            countrycolor="#9ab9be",
            bgcolor="#dff7f8",
            lataxis_range=[LAT_MIN, LAT_MAX],
            lonaxis_range=[LON_MIN, LON_MAX],
            center=dict(lat=-5, lon=70),
            projection="equirectangular",
        )

        hfig.update_layout(
            height=480,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#dff7f8",
            plot_bgcolor="#dff7f8",
            font=dict(color="#174b56"),
            showlegend=False,
        )

        st.plotly_chart(
            hfig,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )

        st.markdown(
            '<div class="mini-label" style="margin-top:18px">TOP INVESTIGATION ZONES</div>',
            unsafe_allow_html=True,
        )

        table_html = zones.to_html(index=False, classes="", border=0)

        st.markdown(
            f'<div class="hotspot-table">{table_html}</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download hotspot table",
            zones.to_csv(index=False).encode("utf-8"),
            "bloomdetect_hotspot_zones.csv",
            "text/csv",
            width="stretch",
        )

# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":

    section(
        "04 · OCEAN-COLOUR INSIGHTS",
        "What stands out in the latest field?",
        "A compact analysis page for the current observation: field distribution, screening composition and broad regional patterns.",
    )

    normal_count = int((~latest["risk_flag"]).sum())
    risk_count = int(latest["risk_flag"].sum())
    share = 100 * risk_count / len(latest) if len(latest) else 0

    top = st.columns(4, gap="medium")
    for col, (label, value, note) in zip(
        top,
        [
            ("Observation cells", f"{len(latest):,}", "latest field"),
            ("Potential-risk", f"{risk_count:,}", "screening output"),
            ("Normal", f"{normal_count:,}", "latest field"),
            ("Risk share", f"{share:.2f}%", "latest field"),
        ],
    ):
        with col:
            metric(label, value, note)

    regional_defs = [
        (
            "Arabian Sea",
            latest["latitude"].between(5, 30)
            & latest["plot_lon"].between(45, 75),
        ),
        (
            "Bay of Bengal",
            latest["latitude"].between(0, 25)
            & latest["plot_lon"].between(75.01, 100),
        ),
        (
            "Southern Indian Ocean",
            latest["latitude"].between(-30, 5)
            & latest["plot_lon"].between(40, 100),
        ),
        (
            "Northern Indian Ocean",
            latest["latitude"].between(5, 30)
            & latest["plot_lon"].between(75.01, 120),
        ),
    ]

    regional_rows = []

    for name, condition in regional_defs:
        sub = latest[condition]
        if len(sub):
            regional_rows.append(
                {
                    "Region": name,
                    "Cells": len(sub),
                    "Potential-risk cells": int(sub["risk_flag"].sum()),
                    "Risk share": 100 * sub["risk_flag"].mean(),
                    "Mean Chl-a": sub["chla"].mean(),
                    "Max Chl-a": sub["chla"].max(),
                }
            )

    regional = pd.DataFrame(regional_rows)

    st.markdown(
        '<div class="mini-label" style="margin-top:24px">REGIONAL VIEW</div>',
        unsafe_allow_html=True,
    )

    if not regional.empty:
        display_regional = regional.copy()
        display_regional["Risk share"] = display_regional["Risk share"].map(
            lambda x: f"{x:.2f}%"
        )
        display_regional["Mean Chl-a"] = display_regional["Mean Chl-a"].round(4)
        display_regional["Max Chl-a"] = display_regional["Max Chl-a"].round(4)

        st.dataframe(
            display_regional,
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2, gap="large")

    with left:
        chart = pd.DataFrame(
            {
                "Classification": ["Normal", "Potential bloom risk"],
                "Cells": [normal_count, risk_count],
            }
        )

        fig1 = px.bar(
            chart,
            x="Classification",
            y="Cells",
            text="Cells",
        )

        fig1.update_traces(
            marker_color=["#197da5", "#ef4f5e"],
            textposition="outside",
            cliponaxis=False,
        )

        fig1.update_layout(
            height=360,
            margin=dict(l=55, r=20, t=25, b=65),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Screening outcome",
                title_font=dict(size=13),
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="Number of cells",
                title_font=dict(size=13),
                tickfont=dict(size=12),
                gridcolor="#d6e7e8",
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="mini-label">SCREENING COMPOSITION</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig1, width="stretch", config={"displaylogo": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        clean_chla = latest["chla"].replace([np.inf, -np.inf], np.nan).dropna()

        hist_df = pd.DataFrame({"chla": clean_chla})

        fig2 = px.histogram(
            hist_df,
            x="chla",
            nbins=32,
        )

        fig2.update_traces(marker_color="#0aa8b5")

        fig2.update_layout(
            height=360,
            margin=dict(l=55, r=20, t=25, b=65),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Satellite-derived chlorophyll-a",
                title_font=dict(size=13),
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="Number of cells",
                title_font=dict(size=13),
                tickfont=dict(size=12),
                gridcolor="#d6e7e8",
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="mini-label">CHLOROPHYLL-A DISTRIBUTION</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section" style="padding-top:28px">
            <div class="kicker">INVESTIGATION QUEUE</div>
            <h2>Flagged cells to inspect.</h2>
            <p>A compact sample of current potential-risk cells. This is not a severity ranking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    queue = risk.copy().head(20)

    keep = [
        c
        for c in [
            "latitude",
            "longitude",
            "chla",
            "chla_anomaly",
            "chla_change",
            "risk_probability",
        ]
        if c in queue.columns
    ]

    queue = queue[keep].copy()

    for c in ["chla", "chla_anomaly", "chla_change", "risk_probability"]:
        if c in queue.columns:
            queue[c] = queue[c].round(4)

    st.dataframe(queue, width="stretch", hide_index=True)

# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    section(
        "05 · DATA & OUTPUTS",
        "The project data, without the lecture.",
        "This page keeps the source information and downloadable outputs in one place. The analysis pages stay focused on doing things with the data.",
    )

    cards = st.columns(4, gap="medium")

    for col, (label, value, note) in zip(
        cards,
        [
            ("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3"),
            ("Grid", "0.25°", "latitude × longitude"),
            ("Latest cells", f"{len(latest):,}", "processed observation"),
            ("Latest date", latest_date.strftime("%d %b %Y"), "processed dataset"),
        ],
    ):
        with col:
            metric(label, value, note)

    st.markdown(
        """
        <div class="card" style="margin-top:18px">
            <div class="mini-label">SOURCE</div>
            <h3>EOS-06 OCM-3 analysed chlorophyll-a</h3>
            <p>
                The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4
                analysed chlorophyll product (<b>E06OCM_L4_AC</b>).
                The current project output is a screening layer built from
                the satellite-derived chlorophyll-a signal and temporal context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section" style="padding-top:30px">
            <div class="kicker">PROJECT OUTPUTS</div>
            <h2>Download what you need.</h2>
            <p>These are the actual latest-field outputs used by the dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dl1, dl2 = st.columns(2, gap="large")

    with dl1:
        st.markdown(
            """
            <div class="download-card">
                <h3>Latest observation table</h3>
                <p>All processed cells from the latest available field.</p>
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

    with dl2:
        st.markdown(
            """
            <div class="download-card">
                <h3>Potential-risk shortlist</h3>
                <p>Only cells currently screened as potential bloom risk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download potential-risk locations",
            risk.to_csv(index=False).encode("utf-8"),
            "bloomdetect_potential_risk.csv",
            "text/csv",
            width="stretch",
        )

    st.markdown(
        """
        <div class="section" style="padding-top:30px">
            <div class="kicker">INTERPRETATION</div>
            <h2>Use the result correctly.</h2>
        </div>

        <div class="card">
            <p>
                <b>Potential bloom risk</b> is a screening output from this
                project. It is not confirmation of a harmful algal bloom,
                species identity or toxin presence. Satellite observations
                should be combined with field observations and additional
                environmental evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown(
    f"""
    <div class="footer">
        BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening ·
        Latest processed field: {latest_date.strftime("%d %B %Y")}
    </div>
    """,
    unsafe_allow_html=True,
)
