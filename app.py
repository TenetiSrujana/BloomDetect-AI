from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Satellite-based Coastal & Ocean Intelligence Platform
#
# Navigation:
# Home | Risk Map | Location | Insights | Data
#
# Data used:
#   latest_bloom_risk_predictions.csv
#   bloomdetect_history_web.csv.gz
#   bloomdetect_bloom_process.png
#
# No synthetic/demo values are created by this app.
# Historical views are derived only from the compact history layer.
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

HISTORY_CANDIDATES = [
    BASE_DIR / "bloomdetect_history_web.csv.gz",
    BASE_DIR / "bloomdetect_history.csv.gz",
    BASE_DIR / "bloomdetect_history.csv",
]

# Study window used by the project.
LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# Proxy-screening thresholds used when the project screening label
# was created. These are NOT HAB biological thresholds.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

PAGE_KEYS = ["home", "map", "location", "insights", "data"]


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading satellite observations...")
def load_predictions():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv must be beside app.py."
        )

    d = pd.read_csv(DATA_PATH)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            "Prediction CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    numeric_cols = [
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
    ]

    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
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

    d["plot_lon"] = d["longitude"].astype(float)

    return d


@st.cache_data(show_spinner="Loading compact historical layer...")
def load_history():
    path = next((p for p in HISTORY_CANDIDATES if p.exists()), None)

    if path is None:
        return pd.DataFrame(), None

    h = pd.read_csv(path)

    required = {"date", "lat_bin", "lon_bin", "chla", "risk"}
    if not required.issubset(h.columns):
        return pd.DataFrame(), None

    h["date"] = pd.to_datetime(h["date"], errors="coerce")

    for col in [
        "lat_bin",
        "lon_bin",
        "chla",
        "risk",
        "cells",
        "max_chla",
    ]:
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")

    h = h.replace([np.inf, -np.inf], np.nan)
    h = h.dropna(
        subset=["date", "lat_bin", "lon_bin", "chla"]
    ).copy()

    h["risk"] = pd.to_numeric(
        h["risk"], errors="coerce"
    ).fillna(0).astype("int8")

    # Derive temporal signals only if they are not already stored.
    # These describe the compact historical layer and do not create
    # another ML prediction.
    h = h.sort_values(
        ["lat_bin", "lon_bin", "date"]
    ).reset_index(drop=True)

    group = h.groupby(
        ["lat_bin", "lon_bin"],
        sort=False
    )["chla"]

    if "previous_chla" not in h.columns:
        h["previous_chla"] = group.shift(1)

    if "historical_baseline" not in h.columns:
        h["historical_baseline"] = group.transform(
            lambda x: x.shift(1).expanding().mean()
        )

    if "chla_anomaly" not in h.columns:
        h["chla_anomaly"] = (
            h["chla"] - h["historical_baseline"]
        )

    if "chla_change" not in h.columns:
        h["chla_change"] = (
            h["chla"] - h["previous_chla"]
        )

    h["risk"] = h["risk"].astype("int8")

    return h, path.name


try:
    df = load_predictions()
    history, history_name = load_history()
except Exception as exc:
    st.error("BloomDetect AI could not load the project data.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
latest_risk = latest[latest["risk_flag"]].copy()


# ============================================================
# PRECOMPUTED RESEARCH SIGNALS
# ============================================================

@st.cache_data(show_spinner=False)
def build_history_summary(h):
    if h.empty:
        return pd.DataFrame()

    summary = (
        h.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(
            observed_dates=("date", "nunique"),
            risk_days=("risk", "sum"),
            mean_chla=("chla", "mean"),
            max_chla=("chla", "max"),
            mean_anomaly=("chla_anomaly", "mean"),
        )
    )

    summary["risk_persistence"] = (
        summary["risk_days"] / summary["observed_dates"]
    )

    return summary


history_summary = build_history_summary(history)


@st.cache_data(show_spinner=False)
def latest_signal_counts(latest_df):
    total = len(latest_df)

    anomaly = (
        latest_df["chla_anomaly"].ge(ANOMALY_THRESHOLD).fillna(False)
        if "chla_anomaly" in latest_df.columns
        else pd.Series(False, index=latest_df.index)
    )

    change = (
        latest_df["chla_change"].ge(CHANGE_THRESHOLD).fillna(False)
        if "chla_change" in latest_df.columns
        else pd.Series(False, index=latest_df.index)
    )

    return {
        "positive_change": int(change.sum()),
        "anomalous": int(anomaly.sum()),
        "risk": int(latest_df["risk_flag"].sum()),
        "normal": int(total - latest_df["risk_flag"].sum()),
    }


signal_counts = latest_signal_counts(latest)


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
    --aqua:#08a7b3;
    --muted:#587980;
    --line:#a8d9dd;
    --bg:#f2fbfb;
    --green:#20a477;
    --red:#ef4f5e;
    --blue:#197da5;
    --gold:#c58a1b;
}

html, body,
[data-testid="stAppViewContainer"]{
    background:var(--bg)!important;
    color:var(--ink)!important;
    font-family:'DM Sans',sans-serif!important;
}

.stApp{
    background:
        radial-gradient(circle at 8% 8%,rgba(45,210,212,.08),transparent 24%),
        radial-gradient(circle at 92% 75%,rgba(24,160,180,.06),transparent 28%),
        linear-gradient(180deg,#fbffff 0%,#effafa 55%,#fbffff 100%)!important;
    overflow-x:hidden!important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer,
[data-testid="stSidebar"]{
    display:none!important;
}

.block-container{
    width:100%!important;
    max-width:1240px!important;
    padding:24px 28px 65px!important;
    margin:0 auto!important;
    overflow-x:hidden!important;
}

/* HEADER */

.brand-bar{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    width:100%;
    box-sizing:border-box;
    padding:15px 20px;
    border:1.5px solid #c4e4e6;
    background:rgba(255,255,255,.96);
    border-radius:22px;
    box-shadow:0 12px 30px rgba(9,80,94,.08);
}

.brand-left{
    display:flex;
    align-items:center;
    gap:13px;
    min-width:0;
}

.logo{
    width:50px;
    height:50px;
    flex:0 0 50px;
    border-radius:16px;
    background:linear-gradient(145deg,#2bcdd0,#087b8e);
    display:grid;
    place-items:center;
    color:#fff;
    font-size:21px;
    font-weight:800;
}

.brand-name{
    font:800 1.25rem Manrope,sans-serif;
    color:var(--ink);
    letter-spacing:-.035em;
}

.brand-sub{
    color:#78959b;
    font-size:.73rem;
    margin-top:2px;
}

.latest-label{
    color:#78959b;
    font-size:.67rem;
    line-height:1.35;
    text-align:right;
    white-space:nowrap;
}

.latest-label b{
    color:#174b56;
    font-size:.78rem;
}

/* NAV */

.nav-wrap{
    margin:12px 0 25px;
}

.stButton>button,
.stDownloadButton>button,
.stFormSubmitButton>button{
    min-height:44px!important;
    border-radius:14px!important;
    background:#fff!important;
    border:2px solid #164e5b!important;
    color:#123f49!important;
    font-weight:800!important;
    font-size:.84rem!important;
    box-shadow:0 5px 15px rgba(15,70,82,.07)!important;
}

.stButton>button:hover,
.stDownloadButton>button:hover,
.stFormSubmitButton>button:hover{
    background:#e2f8f8!important;
    border-color:#087d8c!important;
}

.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#d9f7f7,#bceeee)!important;
    border-color:#078c9b!important;
}

/* TYPOGRAPHY */

.section{
    padding:14px 0 10px;
    width:100%;
    box-sizing:border-box;
}

.kicker{
    font:800 .65rem Manrope,sans-serif;
    letter-spacing:.17em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:9px;
}

.section h2{
    font:800 clamp(2rem,4vw,3.25rem)/1.04 Manrope,sans-serif;
    letter-spacing:-.055em;
    color:var(--ink);
    margin:0 0 11px;
    overflow-wrap:anywhere;
}

.section p{
    color:#5b7b82;
    line-height:1.65;
    margin:0;
    max-width:1060px;
    font-size:.94rem;
}

.card{
    box-sizing:border-box;
    width:100%;
    background:rgba(255,255,255,.94);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:0 13px 34px rgba(9,80,94,.08);
    padding:21px;
}

.card h3{
    font:800 1.22rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 8px;
    overflow-wrap:anywhere;
}

.card p{
    color:#5f7e85;
    line-height:1.62;
    margin:0;
    font-size:.89rem;
}

.mini-label{
    font:800 .62rem Manrope,sans-serif;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:#6b898f;
    margin-bottom:8px;
}

.note{
    box-sizing:border-box;
    width:100%;
    margin-top:15px;
    padding:12px 14px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 13px 13px 0;
    color:#52757c;
    font-size:.82rem;
    line-height:1.55;
}

.note b{
    color:#174b56;
}

/* HERO */

.hero{
    box-sizing:border-box;
    width:100%;
    min-height:385px;
    position:relative;
    overflow:hidden;
    border-radius:30px;
    padding:52px 56px;
    background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);
    box-shadow:0 24px 65px rgba(6,86,100,.15);
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
}

.hero:after{
    content:"";
    position:absolute;
    width:270px;
    height:270px;
    right:-75px;
    top:-90px;
    border:46px solid rgba(176,249,247,.13);
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
    font:800 clamp(3rem,6vw,5.45rem)/.92 Manrope,sans-serif;
    letter-spacing:-.075em;
    color:#e4ffff;
    margin:0 0 19px;
    overflow-wrap:anywhere;
}

.hero p{
    font-size:1rem;
    line-height:1.72;
    color:#e0fbfb;
    max-width:760px;
    margin:0;
}

.hero-badges{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-top:22px;
}

.badge{
    padding:8px 12px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.25);
    color:#efffff;
    font-size:.75rem;
    font-weight:700;
}

/* METRICS */

.metrics{
    margin-top:17px;
}

.metric{
    min-height:108px;
    box-sizing:border-box;
    padding:17px;
    background:linear-gradient(145deg,#fff,#eaf8f8);
    border:1.5px solid #a6d5da;
    border-radius:19px;
    box-shadow:0 10px 25px rgba(15,91,101,.07);
    display:flex;
    flex-direction:column;
    justify-content:center;
}

.metric .label{
    font:800 .62rem Manrope,sans-serif;
    letter-spacing:.1em;
    text-transform:uppercase;
    color:#6d8c93;
}

.metric .value{
    font:800 1.35rem Manrope,sans-serif;
    color:#123f49;
    margin-top:6px;
    overflow-wrap:anywhere;
}

.metric .note{
    margin:4px 0 0;
    padding:0;
    border:0;
    background:none;
    font-size:.7rem;
    color:#78959b;
}

/* STORY */

.story-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:20px;
    align-items:stretch;
    margin-top:18px;
}

.story-image{
    width:100%;
    height:100%;
    box-sizing:border-box;
    border:1.5px solid #a9d6da;
    border-radius:22px;
    overflow:hidden;
    background:#dff7f8;
    box-shadow:0 13px 34px rgba(9,80,94,.08);
}

.story-image img{
    display:block;
    width:100%;
    height:100%;
    object-fit:cover;
}

/* TOOL CARDS */

.tools-heading{
    margin-top:30px;
}

.tool-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:15px;
    margin-top:15px;
}

.tool-card{
    box-sizing:border-box;
    min-height:145px;
    padding:18px;
    background:rgba(255,255,255,.92);
    border:1.5px solid #b5dde0;
    border-radius:19px;
    box-shadow:0 10px 25px rgba(9,80,94,.07);
}

.tool-card .icon{
    font-size:1.2rem;
    margin-bottom:8px;
}

.tool-card h3{
    font:800 1rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 6px;
}

.tool-card p{
    font-size:.8rem;
    line-height:1.5;
    color:#66848b;
    margin:0;
}

/* MAP */

.timeline-card{
    box-sizing:border-box;
    width:100%;
    margin:12px 0 17px;
    padding:14px 17px;
    background:rgba(255,255,255,.94);
    border:1.5px solid #a9d6da;
    border-radius:18px;
    box-shadow:0 10px 25px rgba(9,80,94,.07);
}

.timeline-top{
    display:flex;
    justify-content:space-between;
    gap:16px;
    align-items:end;
}

.timeline-title{
    font:800 .86rem Manrope;
    color:#174b56;
}

.timeline-help{
    font-size:.71rem;
    color:#6c8b92;
    margin-top:3px;
}

.timeline-date{
    font:800 1rem Manrope;
    color:#103f49;
    white-space:nowrap;
}

.map-shell{
    box-sizing:border-box;
    width:100%;
    background:#dff7f8;
    border-radius:23px;
    padding:4px;
    border:2px solid #8fcbd1;
    box-shadow:0 16px 42px rgba(15,91,101,.09);
    overflow:hidden;
}

.map-title{
    min-height:42px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    padding:0 12px;
    color:#174b56;
}

.map-title b{
    font-size:.88rem;
}

.map-title span{
    font-size:.68rem;
    color:#6b8b92;
}

.map-legend{
    display:flex;
    align-items:center;
    gap:8px 10px;
    flex-wrap:wrap;
    color:#4f747b;
    font-size:.74rem;
    padding:11px 14px;
}

.legend{
    width:14px;
    height:10px;
    border-radius:4px;
    display:inline-block;
}

.blue{background:#197da5}
.green{background:#20a477}
.red{
    background:#ef4f5e;
    width:11px;
    height:11px;
    border-radius:50%;
}

/* LOCATION */

.result-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:9px;
    margin-top:13px;
}

.result-item{
    padding:11px 12px;
    background:#f5fcfc;
    border:1px solid #c7e5e7;
    border-radius:13px;
}

.result-item span{
    display:block;
    font-size:.67rem;
    color:#78959b;
    margin-bottom:4px;
}

.result-item b{
    color:#194b55;
    font-size:.88rem;
    overflow-wrap:anywhere;
}

.status{
    margin-top:13px;
    padding:13px 14px;
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
    font:800 .88rem Manrope;
    margin-bottom:3px;
}

.signal-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:9px;
    margin-top:12px;
}

.signal{
    min-width:0;
    padding:12px;
    border-radius:14px;
    background:#f7fcfc;
    border:1px solid #c9e5e7;
}

.signal .label{
    font-size:.66rem;
    color:#78959b;
}

.signal .value{
    font:800 .96rem Manrope;
    color:#164a55;
    margin-top:4px;
    overflow-wrap:anywhere;
}

/* CHARTS */

.chart-card{
    box-sizing:border-box;
    width:100%;
    background:#fff;
    border:1.5px solid #b6dfe2;
    border-radius:20px;
    box-shadow:0 10px 28px rgba(9,80,94,.07);
    padding:13px 13px 7px;
}

.chart-title{
    font:800 .95rem Manrope;
    color:#123f49;
    padding:4px 5px 8px;
}

.hotspot-table{
    width:100%;
    overflow-x:auto;
    border-radius:18px;
    border:1px solid #c5e3e5;
    background:#fff;
}

.hotspot-table table{
    width:100%;
    border-collapse:collapse;
    font-size:.82rem;
}

.hotspot-table th{
    background:#0e5663;
    color:#fff;
    text-align:left;
    padding:10px 12px;
}

.hotspot-table td{
    padding:9px 12px;
    border-top:1px solid #e3eeee;
    color:#315b63;
    background:#fff;
}

.insight-callout{
    padding:16px 17px;
    border-radius:18px;
    background:linear-gradient(135deg,#edfafa,#fff);
    border:1.5px solid #b9dfe2;
    box-shadow:0 9px 25px rgba(9,80,94,.06);
}

.insight-callout .big{
    font:800 1.55rem Manrope;
    color:#103f49;
}

.insight-callout .small{
    margin-top:4px;
    font-size:.78rem;
    color:#64838a;
    line-height:1.45;
}

.download-card{
    box-sizing:border-box;
    min-height:135px;
    padding:18px;
    border-radius:19px;
    background:linear-gradient(135deg,#087b8b,#13adb3);
    border:2px solid #07576a;
    box-shadow:0 13px 28px rgba(8,91,102,.15);
    color:#fff;
}

.download-card h3{
    font:800 1.06rem Manrope;
    color:#fff;
    margin:0 0 6px;
}

.download-card p{
    font-size:.8rem;
    color:#e5ffff;
    line-height:1.45;
    margin:0;
}

.footer{
    border-top:1px solid #d5ebed;
    margin-top:38px;
    padding-top:14px;
    color:#76959b;
    font-size:.68rem;
}

/* INPUTS */

.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{
    border-radius:12px!important;
}

.stSlider label,
.stSelectbox label,
.stNumberInput label{
    color:#174b56!important;
    font-weight:800!important;
}

/* RESPONSIVE */

@media(max-width:900px){
    .story-grid{grid-template-columns:1fr;}
    .tool-grid{grid-template-columns:1fr 1fr;}
    .signal-grid{grid-template-columns:1fr 1fr;}
}

@media(max-width:620px){
    .block-container{padding:14px 11px 42px!important;}
    .brand-bar{padding:12px 13px;}
    .latest-label{display:none;}
    .hero{padding:36px 23px;min-height:380px;}
    .hero h1{font-size:3rem;}
    .tool-grid{grid-template-columns:1fr;}
    .signal-grid,.result-grid{grid-template-columns:1fr;}
    .timeline-top{align-items:flex-start;flex-direction:column;}
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def navigate(page):
    st.session_state.page = page
    st.rerun()


def metric(label, value, note=""):
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


def page_heading(kicker, title, copy):
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


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fmt(value, digits=4):
    value = safe_float(value)
    if pd.isna(value):
        return "Unavailable"
    return f"{value:.{digits}f}"


def pct(value):
    value = safe_float(value)
    if pd.isna(value):
        return "Unavailable"
    if value <= 1:
        value *= 100
    return f"{value:.1f}%"


def normalize_lon(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return ((value + 180.0) % 360.0) - 180.0


def region_name(lat, lon):
    lat = safe_float(lat)
    lon = safe_float(lon)

    if pd.isna(lat) or pd.isna(lon):
        return "Unknown"

    if 5 <= lat <= 30 and 45 <= lon <= 75:
        return "Arabian Sea"

    if 0 <= lat <= 25 and 75 < lon <= 100:
        return "Bay of Bengal"

    if -30 <= lat < 5 and 40 <= lon <= 100:
        return "Southern Indian Ocean"

    if 5 <= lat <= 30 and 75 < lon <= 120:
        return "Northern Indian Ocean"

    return "Other study area"


def available_history_dates():
    if history.empty:
        return []

    return sorted(
        pd.to_datetime(
            history["date"].dropna().unique()
        )
    )


def history_for_date(selected_date):
    if history.empty:
        return pd.DataFrame()

    selected_date = pd.Timestamp(selected_date)

    h = history[
        history["date"].eq(selected_date)
    ].copy()

    return h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()


def nearest_latest(lat, lon):
    lat = safe_float(lat)
    lon = safe_float(lon)

    if pd.isna(lat) or pd.isna(lon) or latest.empty:
        return None

    lat_values = pd.to_numeric(
        latest["latitude"], errors="coerce"
    ).to_numpy(dtype=float)

    lon_values = pd.to_numeric(
        latest["longitude"], errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.isfinite(lat_values) & np.isfinite(lon_values)

    if not valid.any():
        return None

    lat_values = lat_values[valid]
    lon_values = lon_values[valid]
    positions = np.flatnonzero(valid)

    # Longitude distance is scaled by latitude so the nearest
    # geographic cell is not distorted by longitude convergence.
    lon_scale = max(
        np.cos(np.deg2rad(lat)),
        0.25
    )

    dist = (
        (lat_values - lat) ** 2
        + ((lon_values - lon) * lon_scale) ** 2
    )

    return latest.iloc[int(positions[int(np.argmin(dist))])]


def nearest_history_cell(lat, lon):
    if history.empty:
        return None

    lat = safe_float(lat)
    lon = safe_float(lon)

    if pd.isna(lat) or pd.isna(lon):
        return None

    h = history[
        history["lat_bin"].between(LAT_MIN, LAT_MAX)
        & history["lon_bin"].between(LON_MIN, LON_MAX)
    ]

    if h.empty:
        return None

    lat_values = h["lat_bin"].to_numpy(dtype=float)
    lon_values = h["lon_bin"].to_numpy(dtype=float)

    lon_scale = max(
        np.cos(np.deg2rad(lat)),
        0.25
    )

    dist = (
        (lat_values - lat) ** 2
        + ((lon_values - lon) * lon_scale) ** 2
    )

    return h.iloc[int(np.argmin(dist))]


def location_history(lat, lon):
    if history.empty:
        return pd.DataFrame()

    lat = safe_float(lat)
    lon = safe_float(lon)

    if pd.isna(lat) or pd.isna(lon):
        return pd.DataFrame()

    h = history[
        history["lat_bin"].between(LAT_MIN, LAT_MAX)
        & history["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()

    if h.empty:
        return h

    lon_scale = max(
        np.cos(np.deg2rad(lat)),
        0.25
    )

    distances = (
        (h["lat_bin"].to_numpy() - lat) ** 2
        + (
            (h["lon_bin"].to_numpy() - lon)
            * lon_scale
        ) ** 2
    )

    h["_distance"] = distances

    # Keep only observations for the nearest historical 1° grid cell.
    best = h.loc[
        h["_distance"].eq(h["_distance"].min())
    ]

    return best.sort_values("date").drop(
        columns=["_distance"],
        errors="ignore"
    )


def map_source(selected_date):
    selected_date = pd.Timestamp(selected_date)

    if selected_date == latest_date:
        field = latest.copy()

        field["lat_bin"] = (
            np.floor(field["latitude"]) + 0.5
        )
        field["lon_bin"] = (
            np.floor(field["longitude"]) + 0.5
        )

        return field, True

    h = history_for_date(selected_date)

    if h.empty:
        return latest.copy(), True

    return h, False


def map_figure(selected_date, detail_mode):
    field, is_latest = map_source(selected_date)

    if is_latest:
        processed = field.copy()

        base = (
            processed.groupby(
                ["lat_bin", "lon_bin"],
                as_index=False
            )
            .agg(
                mean_chla=("chla", "mean"),
                cells=("chla", "size"),
            )
        )

        risk_cells = processed[
            processed["risk_flag"]
        ].copy()

        if detail_mode == "Change from previous observation":
            value_col = "chla_change"
            legend_title = "Chl-a change"
            if value_col in processed.columns:
                base["signal"] = pd.to_numeric(
                    processed.groupby(
                        ["lat_bin", "lon_bin"]
                    )[value_col].mean(),
                    errors="coerce"
                ).to_numpy()
            else:
                base["signal"] = np.nan

        elif detail_mode == "Historical anomaly":
            value_col = "chla_anomaly"
            legend_title = "Chl-a anomaly"
            if value_col in processed.columns:
                base["signal"] = pd.to_numeric(
                    processed.groupby(
                        ["lat_bin", "lon_bin"]
                    )[value_col].mean(),
                    errors="coerce"
                ).to_numpy()
            else:
                base["signal"] = np.nan

        else:
            base["signal"] = base["mean_chla"]
            legend_title = (
                "Chl-a"
                if detail_mode == "Chlorophyll-a"
                else "Processed field"
            )

    else:
        processed = field.copy()

        base = processed.rename(
            columns={
                "lat_bin": "lat_bin",
                "lon_bin": "lon_bin",
                "chla": "mean_chla",
            }
        ).copy()

        base["cells"] = pd.to_numeric(
            base.get("cells", 1),
            errors="coerce"
        ).fillna(1)

        if detail_mode == "Change from previous observation":
            base["signal"] = pd.to_numeric(
                base["chla_change"],
                errors="coerce"
            )
            legend_title = "Chl-a change"

        elif detail_mode == "Historical anomaly":
            base["signal"] = pd.to_numeric(
                base["chla_anomaly"],
                errors="coerce"
            )
            legend_title = "Chl-a anomaly"

        else:
            base["signal"] = pd.to_numeric(
                base["mean_chla"],
                errors="coerce"
            )
            legend_title = (
                "Chl-a"
                if detail_mode == "Chlorophyll-a"
                else "Historical screening field"
            )

        risk_cells = processed[
            processed["risk"].fillna(0).astype(int).gt(0)
        ].copy()

        risk_cells = risk_cells.rename(
            columns={
                "lat_bin": "latitude",
                "lon_bin": "longitude",
                "mean_chla": "chla",
            }
        )

    base["signal"] = pd.to_numeric(
        base["signal"],
        errors="coerce"
    )

    fig = go.Figure()

    if detail_mode in [
        "Chlorophyll-a",
        "Change from previous observation",
        "Historical anomaly",
    ]:
        valid = base.dropna(
            subset=["signal"]
        ).copy()

        if not valid.empty:
            fig.add_trace(
                go.Scattergeo(
                    lat=valid["lat_bin"],
                    lon=valid["lon_bin"],
                    mode="markers",
                    name=legend_title,
                    marker=dict(
                        size=np.clip(
                            pd.to_numeric(
                                valid["cells"],
                                errors="coerce"
                            ).fillna(1).to_numpy() ** 0.5 * 2.0 + 5,
                            5,
                            15,
                        ),
                        color=valid["signal"].to_numpy(),
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(
                            title=legend_title,
                            thickness=12,
                        ),
                        opacity=0.78,
                    ),
                    customdata=np.column_stack(
                        [
                            valid["signal"].to_numpy(),
                            valid["cells"].to_numpy(),
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{lat:.2f}°, %{lon:.2f}°</b><br>"
                        + legend_title
                        + ": %{customdata[0]:.4f}<br>"
                        "Cells: %{customdata[1]:.0f}"
                        "<extra></extra>"
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Scattergeo(
                lat=base["lat_bin"],
                lon=base["lon_bin"],
                mode="markers",
                name="Processed ocean field",
                marker=dict(
                    size=np.clip(
                        pd.to_numeric(
                            base["cells"],
                            errors="coerce"
                        ).fillna(1).to_numpy() ** 0.5 * 2.0 + 5,
                        5,
                        13,
                    ),
                    color="#197da5",
                    opacity=0.58,
                ),
                customdata=base[
                    ["mean_chla", "cells"]
                ].to_numpy(),
                hovertemplate=(
                    "<b>Processed ocean field</b><br>"
                    "Mean Chl-a: %{customdata[0]:.4f}<br>"
                    "Cells represented: %{customdata[1]:.0f}"
                    "<extra></extra>"
                ),
            )
        )

        if not risk_cells.empty:
            rc = risk_cells.copy()

            rc["latitude"] = pd.to_numeric(
                rc["latitude"], errors="coerce"
            )
            rc["longitude"] = pd.to_numeric(
                rc["longitude"], errors="coerce"
            )
            rc["chla"] = pd.to_numeric(
                rc["chla"], errors="coerce"
            )

            rc = rc.dropna(
                subset=["latitude", "longitude"]
            )

            # Explicit numeric array avoids the old Plotly marker-size crash.
            rc["zone_lat"] = (
                np.floor(rc["latitude"] / 2) * 2 + 1
            )
            rc["zone_lon"] = (
                np.floor(rc["longitude"] / 2) * 2 + 1
            )

            zones = (
                rc.groupby(
                    ["zone_lat", "zone_lon"],
                    as_index=False
                )
                .size()
                .rename(columns={"size": "flagged_cells"})
            )

            flag_values = pd.to_numeric(
                zones["flagged_cells"],
                errors="coerce"
            ).fillna(0).to_numpy(dtype=float)

            marker_sizes = np.clip(
                flag_values * 1.2 + 8,
                9,
                30,
            )

            fig.add_trace(
                go.Scattergeo(
                    lat=zones["zone_lat"],
                    lon=zones["zone_lon"],
                    mode="markers",
                    name="Flag concentration",
                    marker=dict(
                        size=marker_sizes,
                        color="#20a477",
                        opacity=0.43,
                        line=dict(
                            width=1.1,
                            color="#ffffff"
                        ),
                    ),
                    text=flag_values,
                    hovertemplate=(
                        "<b>Investigation zone</b><br>"
                        "Potential-risk cells: %{text:.0f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scattergeo(
                    lat=rc["latitude"],
                    lon=rc["longitude"],
                    mode="markers",
                    name="Potential-risk screening",
                    marker=dict(
                        size=6,
                        color="#ef4f5e",
                        opacity=0.92,
                        line=dict(
                            width=0.7,
                            color="#ffffff"
                        ),
                    ),
                    text=rc["chla"],
                    hovertemplate=(
                        "<b>Potential bloom-risk screening</b><br>"
                        "Chl-a: %{text:.4f}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_geos(
        showland=True,
        landcolor="#dce9e7",
        showocean=True,
        oceancolor="#dff7f8",
        showcoastlines=True,
        coastlinecolor="#4d9099",
        coastlinewidth=1,
        showcountries=True,
        countrycolor="#9ab9be",
        bgcolor="#dff7f8",
        lataxis_range=[LAT_MIN, LAT_MAX],
        lonaxis_range=[LON_MIN, LON_MAX],
        center=dict(lat=-5, lon=70),
        projection_type="equirectangular",
    )

    fig.update_layout(
        height=590,
        margin=dict(l=0, r=0, t=5, b=5),
        paper_bgcolor="#dff7f8",
        plot_bgcolor="#dff7f8",
        font=dict(
            color="#174b56",
            size=11,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,.93)",
            bordercolor="#b9dfe2",
            borderwidth=1,
            font=dict(
                size=10,
                color="#174b56"
            ),
        ),
    )

    processed_count = len(field)

    if is_latest:
        risk_count = int(
            field["risk_flag"].sum()
        )
    else:
        risk_count = int(
            field["risk"].fillna(0).astype(int).sum()
        )

    return (
        fig,
        processed_count,
        risk_count,
        pd.Timestamp(selected_date),
        is_latest,
    )


def hotspot_table(data):
    if data.empty:
        return pd.DataFrame()

    r = data[
        data["risk_flag"]
    ].copy()

    if r.empty:
        return pd.DataFrame()

    r["lat_zone"] = (
        np.floor(r["latitude"] / 2) * 2
    )
    r["lon_zone"] = (
        np.floor(r["plot_lon"] / 2) * 2
    )

    z = (
        r.groupby(
            ["lat_zone", "lon_zone"],
            as_index=False
        )
        .agg(
            flagged_cells=("risk_flag", "size"),
            mean_chla=("chla", "mean"),
            max_chla=("chla", "max"),
        )
        .sort_values(
            ["flagged_cells", "mean_chla"],
            ascending=False
        )
        .head(12)
    )

    z["Zone"] = z.apply(
        lambda x:
            f"{x.lat_zone:.0f}°–{x.lat_zone + 2:.0f}°, "
            f"{x.lon_zone:.0f}°–{x.lon_zone + 2:.0f}°",
        axis=1,
    )

    z["Potential-risk cells"] = (
        pd.to_numeric(
            z["flagged_cells"],
            errors="coerce"
        ).fillna(0).astype(int)
    )

    z["Mean Chl-a"] = z["mean_chla"].round(4)
    z["Max Chl-a"] = z["max_chla"].round(4)

    return z[
        [
            "Zone",
            "Potential-risk cells",
            "Mean Chl-a",
            "Max Chl-a",
        ]
    ]


def make_chart_layout(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(
            l=78,
            r=32,
            t=58,
            b=78,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(
            family="DM Sans",
            color="#244f58",
            size=12,
        ),
        title_font=dict(
            family="Manrope",
            color="#103f49",
            size=18,
        ),
        xaxis=dict(
            title_font=dict(
                color="#244f58",
                size=13,
            ),
            tickfont=dict(
                color="#315a62",
                size=11,
            ),
            gridcolor="#d6e7e8",
            zerolinecolor="#bcd5d8",
        ),
        yaxis=dict(
            title_font=dict(
                color="#244f58",
                size=13,
            ),
            tickfont=dict(
                color="#315a62",
                size=11,
            ),
            gridcolor="#d6e7e8",
            zerolinecolor="#bcd5d8",
        ),
    )
    return fig


def regional_summary():
    definitions = [
        (
            "Arabian Sea",
            latest.latitude.between(5, 30)
            & latest.longitude.between(45, 75),
        ),
        (
            "Bay of Bengal",
            latest.latitude.between(0, 25)
            & latest.longitude.between(75.01, 100),
        ),
        (
            "Southern Indian Ocean",
            latest.latitude.between(-30, 5)
            & latest.longitude.between(40, 100),
        ),
        (
            "Northern Indian Ocean",
            latest.latitude.between(5, 30)
            & latest.longitude.between(75.01, 120),
        ),
    ]

    rows = []

    for name, mask in definitions:
        sub = latest[mask]

        if sub.empty:
            continue

        rows.append(
            {
                "Region": name,
                "Processed cells": len(sub),
                "Potential-risk cells": int(
                    sub["risk_flag"].sum()
                ),
                "Risk share": (
                    100 * sub["risk_flag"].mean()
                ),
                "Mean Chl-a": sub["chla"].mean(),
                "Maximum Chl-a": sub["chla"].max(),
            }
        )

    return pd.DataFrame(rows)


def latest_risk_history():
    if history.empty or latest_risk.empty:
        return pd.DataFrame()

    flagged = latest_risk.copy()

    flagged["lat_bin"] = np.floor(
        flagged["latitude"]
    )
    flagged["lon_bin"] = np.floor(
        flagged["longitude"]
    )

    merged = flagged.merge(
        history_summary[
            [
                "lat_bin",
                "lon_bin",
                "observed_dates",
                "risk_days",
                "risk_persistence",
                "mean_chla",
                "max_chla",
            ]
        ],
        on=["lat_bin", "lon_bin"],
        how="left",
    )

    return merged


# ============================================================
# HEADER + NAVIGATION
# ============================================================

st.markdown(
    f"""
    <div class="brand-bar">
        <div class="brand-left">
            <div class="logo">≈</div>
            <div>
                <div class="brand-name">BloomDetect AI</div>
                <div class="brand-sub">
                    Coastal &amp; Ocean Intelligence · EOS-06 OCM-3
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

PAGES = [
    ("home", "⌂ Home"),
    ("map", "🗺 Risk Map"),
    ("location", "📍 Location"),
    ("insights", "📊 Insights"),
    ("data", "⇩ Data"),
]

if st.session_state.get("page") not in PAGE_KEYS:
    st.session_state.page = "home"

nav_cols = st.columns(5, gap="small")

for col, (page, label) in zip(nav_cols, PAGES):
    with col:
        if st.button(
            label,
            key=f"nav_{page}",
            width="stretch",
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            ),
        ):
            navigate(page)


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="kicker">
                    EOS-06 · OCM-3 · SATELLITE INTELLIGENCE
                </div>

                <h1>Read the ocean signal.</h1>

                <p>
                    BloomDetect AI combines satellite-derived chlorophyll-a
                    with temporal and spatial context to screen unusual
                    ocean patterns and prioritize locations for closer
                    investigation.
                </p>

                <div class="hero-badges">
                    <span class="badge">🌊 Ocean colour</span>
                    <span class="badge">🛰 EOS-06 OCM-3</span>
                    <span class="badge">📈 Temporal signals</span>
                    <span class="badge">🗺 Spatial intelligence</span>
                    <span class="badge">⚠️ Early-warning support</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_share = (
        100 * len(latest_risk) / len(latest)
        if len(latest)
        else 0
    )

    metric_cols = st.columns(4, gap="medium")

    home_metrics = [
        (
            "Processed cells",
            f"{len(latest):,}",
            "latest field",
        ),
        (
            "Potential-risk cells",
            f"{len(latest_risk):,}",
            "screening output",
        ),
        (
            "Risk share",
            f"{risk_share:.2f}%",
            "of processed cells",
        ),
        (
            "Maximum Chl-a",
            fmt(latest["chla"].max()),
            "latest field",
        ),
    ]

    for col, item in zip(metric_cols, home_metrics):
        with col:
            metric(*item)

    left, right = st.columns(
        [1.0, 1.0],
        gap="large"
    )

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="mini-label">PROJECT PURPOSE</div>
                <h3>From ocean-colour observations to investigation support.</h3>
                <p>
                    The platform is designed to help analysts move from a
                    satellite observation to a more useful question:
                    where is the signal unusual, how is it changing, and
                    which locations deserve closer attention?
                </p>

                <div class="note">
                    <b>Scientific boundary:</b>
                    a potential bloom-risk screening flag is not confirmation
                    of a harmful algal bloom. High chlorophyll-a alone cannot
                    establish harmfulness, species identity or toxin presence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if IMAGE_PATH.exists():
            st.image(
                str(IMAGE_PATH),
                width="stretch",
                caption=(
                    "Satellite ocean-colour observations can support "
                    "bloom-risk investigation."
                ),
            )
        else:
            st.markdown(
                """
                <div class="card">
                    <div class="mini-label">PROJECT ASSET</div>
                    <h3>Ocean-colour observation workflow</h3>
                    <p>
                        The project illustration is not available in the
                        application folder, so no substitute image is shown.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="tools-heading">
            <div class="kicker">INTELLIGENCE LAYERS</div>
            <div class="section" style="padding:0">
                <h2>Four views. Four different jobs.</h2>
                <p>
                    The dashboard avoids repeating the same analysis.
                    Each view exposes a different part of the evidence.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tool_cols = st.columns(4, gap="medium")

    tools = [
        (
            "🗺️",
            "Risk Map",
            "Move through observations and inspect spatial screening signals.",
            "map",
            "Open Risk Map",
        ),
        (
            "📍",
            "Location",
            "Evaluate one coordinate against its nearest processed cell and history.",
            "location",
            "Open Location",
        ),
        (
            "📊",
            "Insights",
            "Study anomalies, change, concentration, persistence and regional patterns.",
            "insights",
            "Open Insights",
        ),
        (
            "⇩",
            "Data",
            "Access the source details and download the actual project outputs.",
            "data",
            "Open Data",
        ),
    ]

    for col, (icon, title, desc, target, button_text) in zip(
        tool_cols, tools
    ):
        with col:
            st.markdown(
                f"""
                <div class="tool-card">
                    <div class="icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                button_text,
                key=f"home_tool_{target}",
                width="stretch",
            ):
                navigate(target)

    if not history.empty:
        coverage_start = history["date"].min()
        coverage_end = history["date"].max()

        st.markdown(
            f"""
            <div class="note" style="margin-top:20px">
                <b>Temporal coverage:</b>
                compact historical screening support is available from
                {coverage_start.strftime("%d %b %Y")} to
                {coverage_end.strftime("%d %b %Y")}.
                Historical views are descriptive support layers, not new ML predictions.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":

    page_heading(
        "01 · SPATIAL INTELLIGENCE",
        "See the field, then change the lens.",
        "The map keeps one geographic frame while letting you inspect the screening result, chlorophyll-a, recent change or historical anomaly.",
    )

    dates = available_history_dates()

    if len(dates) > 1:

        if (
            "map_date" not in st.session_state
            or pd.Timestamp(
                st.session_state.map_date
            ) not in dates
        ):
            st.session_state.map_date = dates[-1]

        selected = st.slider(
            "Observation date",
            min_value=dates[0].date(),
            max_value=dates[-1].date(),
            value=pd.Timestamp(
                st.session_state.map_date
            ).date(),
            format="DD MMM YYYY",
            key="map_date_slider",
        )

        selected_date = pd.Timestamp(selected)
        st.session_state.map_date = selected_date

    else:
        selected_date = latest_date

        if history.empty:
            st.markdown(
                """
                <div class="note">
                    <b>Historical layer unavailable:</b>
                    the latest processed field is shown because no compact
                    history file is present beside the application.
                </div>
                """,
                unsafe_allow_html=True,
            )

    mode = st.selectbox(
        "Map detail",
        [
            "Current screening",
            "Chlorophyll-a",
            "Change from previous observation",
            "Historical anomaly",
        ],
        index=0,
    )

    fig, processed_count, risk_count, actual_date, is_latest = map_figure(
        selected_date,
        mode,
    )

    share = (
        100 * risk_count / processed_count
        if processed_count
        else 0
    )

    metric_cols = st.columns(4, gap="medium")

    values = [
        (
            "Selected date",
            actual_date.strftime("%d %b %Y"),
            "observation",
        ),
        (
            "Processed cells",
            f"{processed_count:,}",
            "selected field",
        ),
        (
            "Potential-risk",
            f"{risk_count:,}",
            "screening layer",
        ),
        (
            "Risk share",
            f"{share:.2f}%",
            "selected field",
        ),
    ]

    for col, item in zip(metric_cols, values):
        with col:
            metric(*item)

    source_note = (
        "latest 0.25° processed screening field"
        if is_latest
        else "compact historical 1° screening support layer"
    )

    st.markdown(
        f"""
        <div class="map-shell">
            <div class="map-title">
                <b>Indian Ocean · Arabian Sea · Bay of Bengal</b>
                <span>{source_note}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
            ],
        },
    )

    st.markdown(
        """
            <div class="map-legend">
                <span class="legend blue"></span>
                <span>Blue = processed ocean field</span>

                <span class="legend green"></span>
                <span>Green = spatial concentration of screening flags</span>

                <span class="legend red"></span>
                <span>Red = individual potential-risk screening cell</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "Current screening":
        st.markdown(
            """
            <div class="note">
                <b>Reading the screening map:</b>
                green marks show where potential-risk screening cells
                concentrate. They are not a separate severity score.
                Red cells are screening results, not confirmed harmful
                algal blooms.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "Change from previous observation":
        st.markdown(
            """
            <div class="note">
                <b>Reading change:</b>
                positive values indicate chlorophyll-a increased relative
                to the previous available observation for the same compact
                historical grid cell.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "Historical anomaly":
        st.markdown(
            """
            <div class="note">
                <b>Reading anomaly:</b>
                anomaly is the difference between the current value and the
                historical mean available before that observation for the
                same compact grid cell.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page == "location":

    page_heading(
        "02 · LOCATION INTELLIGENCE",
        "Interrogate one coordinate.",
        "Your input remains separate from the nearest processed 0.25° satellite cell. Historical context is shown from the nearest compact 1° grid cell when available.",
    )

    left, right = st.columns(
        [0.72, 1.28],
        gap="large"
    )

    if "loc_lat" not in st.session_state:
        st.session_state.loc_lat = 17.4

    if "loc_lon" not in st.session_state:
        st.session_state.loc_lon = 78.5

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="mini-label">YOUR INPUT</div>
                <h3>Coordinates</h3>
                <p>
                    Enter decimal degrees inside the project study window.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        lat = st.number_input(
            "Latitude",
            min_value=float(LAT_MIN),
            max_value=float(LAT_MAX),
            value=float(st.session_state.loc_lat),
            step=0.01,
            format="%.4f",
        )

        lon = st.number_input(
            "Longitude",
            min_value=float(LON_MIN),
            max_value=float(LON_MAX),
            value=float(st.session_state.loc_lon),
            step=0.01,
            format="%.4f",
        )

        st.session_state.loc_lat = lat
        st.session_state.loc_lon = lon

        st.markdown(
            f"""
            <div class="note">
                <b>Input location:</b>
                {lat:.4f}°, {lon:.4f}°
            </div>
            """,
            unsafe_allow_html=True,
        )

    row = nearest_latest(lat, normalize_lon(lon))

    with right:
        if row is None:
            st.error(
                "No processed satellite cell is available for this lookup."
            )
        else:
            flagged = bool(row["risk_flag"])

            risk_probability = row.get(
                "risk_probability",
                np.nan
            )

            st.markdown(
                """
                <div class="card">
                    <div class="mini-label">
                        NEAREST PROCESSED OCEAN CELL
                    </div>
                    <h3>Satellite observation used for lookup</h3>
                    <p>
                        This is the actual processed cell returned by the
                        nearest-cell search.
                    </p>

                    <div class="result-grid">
                """,
                unsafe_allow_html=True,
            )

            result_items = [
                (
                    "Processed latitude",
                    f"{float(row['latitude']):.4f}°"
                ),
                (
                    "Processed longitude",
                    f"{float(row['longitude']):.4f}°"
                ),
                (
                    "Observation date",
                    pd.Timestamp(row["date"]).strftime("%d %b %Y")
                ),
                (
                    "Region",
                    region_name(
                        row["latitude"],
                        row["longitude"]
                    )
                ),
                (
                    "Chlorophyll-a",
                    fmt(row["chla"])
                ),
                (
                    "Model screening probability",
                    pct(risk_probability)
                ),
            ]

            for label, value in result_items:
                st.markdown(
                    f"""
                    <div class="result-item">
                        <span>{label}</span>
                        <b>{value}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            if flagged:
                st.markdown(
                    """
                    <div class="status risk">
                        <div class="status-title">
                            🔴 POTENTIAL BLOOM-RISK SCREENING
                        </div>
                        <div>
                            This processed cell is included in the current
                            potential-risk shortlist.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="status normal">
                        <div class="status-title">
                            🟢 NOT FLAGGED
                        </div>
                        <div>
                            This processed cell is not included in the
                            current potential-risk shortlist.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if row is not None:

        support = nearest_history_cell(
            row["latitude"],
            row["longitude"]
        )

        baseline = (
            row.get("historical_baseline", np.nan)
        )
        anomaly = (
            row.get("chla_anomaly", np.nan)
        )
        change = (
            row.get("chla_change", np.nan)
        )

        # Use historical support only when the latest row does not contain
        # the corresponding temporal value.
        if support is not None:
            if pd.isna(baseline):
                baseline = support.get(
                    "historical_baseline",
                    np.nan
                )

            if pd.isna(anomaly):
                anomaly = support.get(
                    "chla_anomaly",
                    np.nan
                )

            if pd.isna(change):
                change = support.get(
                    "chla_change",
                    np.nan
                )

        st.markdown(
            '<div class="mini-label" style="margin-top:20px">SUPPORTING SIGNALS</div>',
            unsafe_allow_html=True,
        )

        signal_cols = st.columns(
            4,
            gap="small"
        )

        signals = [
            (
                "Current Chl-a",
                fmt(row["chla"])
            ),
            (
                "Historical baseline",
                fmt(baseline)
            ),
            (
                "Anomaly",
                fmt(anomaly)
            ),
            (
                "Recent change",
                fmt(change)
            ),
        ]

        for col, (label, value) in zip(
            signal_cols,
            signals
        ):
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

        hist = location_history(
            row["latitude"],
            row["longitude"]
        )

        if not hist.empty:

            st.markdown(
                '<div class="mini-label" style="margin-top:22px">LOCATION HISTORY</div>',
                unsafe_allow_html=True,
            )

            hist_plot = hist[
                [
                    "date",
                    "chla",
                    "historical_baseline",
                    "chla_anomaly",
                    "risk",
                ]
            ].copy()

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=hist_plot["date"],
                    y=hist_plot["chla"],
                    mode="lines",
                    name="Chl-a",
                    line=dict(
                        color="#197da5",
                        width=2.4
                    ),
                )
            )

            baseline_series = hist_plot[
                "historical_baseline"
            ].dropna()

            if not baseline_series.empty:
                fig.add_trace(
                    go.Scatter(
                        x=hist_plot.loc[
                            baseline_series.index,
                            "date"
                        ],
                        y=baseline_series,
                        mode="lines",
                        name="Historical baseline",
                        line=dict(
                            color="#20a477",
                            width=2,
                            dash="dot"
                        ),
                    )
                )

            risk_points = hist_plot[
                hist_plot["risk"].eq(1)
            ]

            if not risk_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=risk_points["date"],
                        y=risk_points["chla"],
                        mode="markers",
                        name="Historical screening flag",
                        marker=dict(
                            color="#ef4f5e",
                            size=7,
                            line=dict(
                                color="#ffffff",
                                width=1
                            ),
                        ),
                    )
                )

            fig.update_layout(
                title=dict(
                    text="Chlorophyll-a history for nearest compact grid cell",
                    font=dict(
                        size=18,
                        color="#103f49"
                    ),
                ),
                height=400,
                margin=dict(
                    l=70,
                    r=30,
                    t=65,
                    b=70,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font=dict(
                    color="#244f58",
                    size=12
                ),
                legend=dict(
                    font=dict(
                        color="#244f58",
                        size=11
                    )
                ),
                xaxis=dict(
                    title="Observation date",
                    title_font=dict(
                        color="#244f58",
                        size=13
                    ),
                    tickfont=dict(
                        color="#315a62",
                        size=10
                    ),
                    gridcolor="#d6e7e8",
                ),
                yaxis=dict(
                    title="Chlorophyll-a",
                    title_font=dict(
                        color="#244f58",
                        size=13
                    ),
                    tickfont=dict(
                        color="#315a62",
                        size=10
                    ),
                    gridcolor="#d6e7e8",
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={"displaylogo": False},
            )

            hist_risk_days = int(
                hist_plot["risk"].sum()
            )

            hist_mean = safe_float(
                hist_plot["chla"].mean()
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(
                    f"""
                    <div class="insight-callout">
                        <div class="big">{hist_risk_days}</div>
                        <div class="small">
                            historical screening-flag observations
                            in the nearest compact grid cell
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="insight-callout">
                        <div class="big">{fmt(hist_mean)}</div>
                        <div class="small">
                            mean chlorophyll-a across the available
                            historical observations for this grid cell
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        report = pd.DataFrame(
            [
                {
                    "input_latitude": lat,
                    "input_longitude": lon,
                    "nearest_processed_latitude": row["latitude"],
                    "nearest_processed_longitude": row["longitude"],
                    "date": pd.Timestamp(
                        row["date"]
                    ).strftime("%Y-%m-%d"),
                    "chla": row["chla"],
                    "historical_baseline": baseline,
                    "chla_anomaly": anomaly,
                    "chla_change": change,
                    "risk_label": row["risk_label"],
                    "risk_probability": risk_probability,
                }
            ]
        )

        st.download_button(
            "⬇ Download location report",
            report.to_csv(
                index=False
            ).encode("utf-8"),
            "bloomdetect_location_report.csv",
            "text/csv",
            width="stretch",
        )


# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":

    page_heading(
        "03 · OCEAN INTELLIGENCE",
        "What patterns stand out?",
        "This page turns the current field into evidence: change, anomaly, spatial concentration, persistence and regional differences.",
    )

    insight_cols = st.columns(
        4,
        gap="medium"
    )

    insight_values = [
        (
            "Potential-risk cells",
            f"{signal_counts['risk']:,}",
            "current screening",
        ),
        (
            "Positive Chl-a change",
            f"{signal_counts['positive_change']:,}",
            "above change threshold",
        ),
        (
            "Anomalous cells",
            f"{signal_counts['anomalous']:,}",
            "above anomaly threshold",
        ),
        (
            "Normal screening",
            f"{signal_counts['normal']:,}",
            "current field",
        ),
    ]

    for col, item in zip(
        insight_cols,
        insight_values
    ):
        with col:
            metric(*item)

    # --------------------------------------------------------
    # SIGNAL INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="tools-heading">
            <div class="kicker">SIGNAL SUMMARY</div>
            <div class="section" style="padding:0">
                <h2>What is driving attention?</h2>
                <p>
                    The screening layer combines the project's chlorophyll-a
                    temporal signals. These counts describe the current field;
                    they are not independent HAB diagnoses.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_risk = len(latest_risk)

    if "chla_change" in latest.columns:
        positive_change = int(
            latest["chla_change"]
            .ge(CHANGE_THRESHOLD)
            .fillna(False)
            .sum()
        )
    else:
        positive_change = 0

    if "chla_anomaly" in latest.columns:
        anomalous = int(
            latest["chla_anomaly"]
            .ge(ANOMALY_THRESHOLD)
            .fillna(False)
            .sum()
        )
    else:
        anomalous = 0

    drivers = pd.DataFrame(
        {
            "Signal": [
                "Potential-risk screening",
                "Positive Chl-a change",
                "Positive anomaly",
            ],
            "Cells": [
                current_risk,
                positive_change,
                anomalous,
            ],
        }
    )

    drivers_plot = px.bar(
        drivers.sort_values(
            "Cells",
            ascending=True
        ),
        x="Cells",
        y="Signal",
        orientation="h",
        text="Cells",
        title="Current signal counts",
    )

    drivers_plot.update_traces(
        marker_color="#20a477",
        textposition="outside",
        cliponaxis=False,
    )

    make_chart_layout(
        drivers_plot,
        height=360
    )

    st.markdown(
        '<div class="chart-card">',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        drivers_plot,
        width="stretch",
        config={"displaylogo": False},
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # HOTSPOT CONCENTRATION + PERSISTENCE
    # --------------------------------------------------------

    zones = hotspot_table(latest)

    if not zones.empty:

        st.markdown(
            """
            <div class="tools-heading">
                <div class="kicker">SPATIAL CONCENTRATION</div>
                <div class="section" style="padding:0">
                    <h2>Where are screening flags clustering?</h2>
                    <p>
                        Potential-risk cells are grouped into 2° × 2°
                        investigation zones. Concentration is not a severity score.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        hz = latest_risk.copy()

        hz["lat_zone"] = (
            np.floor(
                hz["latitude"] / 2
            ) * 2 + 1
        )

        hz["lon_zone"] = (
            np.floor(
                hz["plot_lon"] / 2
            ) * 2 + 1
        )

        plot_zones = (
            hz.groupby(
                ["lat_zone", "lon_zone"],
                as_index=False
            )
            .size()
            .rename(
                columns={
                    "size": "flagged_cells"
                }
            )
            .sort_values(
                "flagged_cells",
                ascending=False
            )
            .head(12)
        )

        plot_zones["Zone"] = plot_zones.apply(
            lambda r:
                f"{r.lat_zone:.0f}°–{r.lat_zone + 2:.0f}°, "
                f"{r.lon_zone:.0f}°–{r.lon_zone + 2:.0f}°",
            axis=1,
        )

        hotfig = px.bar(
            plot_zones.sort_values(
                "flagged_cells",
                ascending=True
            ),
            x="flagged_cells",
            y="Zone",
            orientation="h",
            text="flagged_cells",
            title="Top investigation zones",
        )

        hotfig.update_traces(
            marker_color="#20a477",
            textposition="outside",
            cliponaxis=False,
        )

        make_chart_layout(
            hotfig,
            height=450
        )

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            hotfig,
            width="stretch",
            config={"displaylogo": False},
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="mini-label" style="margin-top:17px">ZONE DETAILS</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="hotspot-table">
                {zones.to_html(index=False, border=0)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    if not history_summary.empty:

        persistent = latest_risk_history()

        if not persistent.empty:

            persistent = persistent.dropna(
                subset=["risk_persistence"]
            ).copy()

            persistent = persistent.sort_values(
                [
                    "risk_persistence",
                    "risk_days",
                    "chla",
                ],
                ascending=False,
            ).head(12)

            persistent["Location"] = (
                persistent.apply(
                    lambda r:
                        f"{r.latitude:.2f}°, {r.longitude:.2f}°",
                    axis=1
                )
            )

            persistent["Persistence"] = (
                persistent["risk_persistence"] * 100
            ).round(1)

            persistence_plot = px.bar(
                persistent.sort_values(
                    "Persistence",
                    ascending=True
                ),
                x="Persistence",
                y="Location",
                orientation="h",
                text="Persistence",
                title="Current flagged cells with strongest historical screening persistence",
            )

            persistence_plot.update_traces(
                marker_color="#ef4f5e",
                texttemplate="%{text:.1f}%",
                textposition="outside",
                cliponaxis=False,
            )

            make_chart_layout(
                persistence_plot,
                height=450
            )

            st.markdown(
                """
                <div class="tools-heading">
                    <div class="kicker">TEMPORAL INTELLIGENCE</div>
                    <div class="section" style="padding:0">
                        <h2>Which current flags have history behind them?</h2>
                        <p>
                            Persistence is the share of observed dates on which
                            the nearest compact historical grid cell was flagged.
                            It is a screening-history measure, not a biological
                            severity measure.
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="chart-card">',
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                persistence_plot,
                width="stretch",
                config={"displaylogo": False},
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # CURRENT CHL-A DISTRIBUTION
    # --------------------------------------------------------

    clean = (
        latest["chla"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    distribution = px.histogram(
        pd.DataFrame(
            {"Chlorophyll-a": clean}
        ),
        x="Chlorophyll-a",
        nbins=36,
        title="Current chlorophyll-a distribution",
    )

    distribution.update_traces(
        marker_color="#197da5"
    )

    make_chart_layout(
        distribution,
        height=390
    )

    st.markdown(
        '<div class="chart-card">',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        distribution,
        width="stretch",
        config={"displaylogo": False},
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RISK VS NORMAL CHL-A
    # --------------------------------------------------------

    risk_mean = safe_float(
        latest.loc[
            latest["risk_flag"],
            "chla"
        ].mean()
    )

    normal_mean = safe_float(
        latest.loc[
            ~latest["risk_flag"],
            "chla"
        ].mean()
    )

    comparison = pd.DataFrame(
        {
            "Screening group": [
                "Not flagged",
                "Potential-risk screening",
            ],
            "Mean chlorophyll-a": [
                normal_mean,
                risk_mean,
            ],
        }
    )

    comparison_plot = px.bar(
        comparison,
        x="Screening group",
        y="Mean chlorophyll-a",
        text="Mean chlorophyll-a",
        title="Mean chlorophyll-a by screening group",
    )

    comparison_plot.update_traces(
        marker_color=[
            "#197da5",
            "#ef4f5e",
        ],
        texttemplate="%{text:.4f}",
        textposition="outside",
        cliponaxis=False,
    )

    make_chart_layout(
        comparison_plot,
        height=390
    )

    st.markdown(
        '<div class="chart-card">',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        comparison_plot,
        width="stretch",
        config={"displaylogo": False},
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # REGIONAL SIGNAL
    # --------------------------------------------------------

    regional = regional_summary()

    if not regional.empty:

        st.markdown(
            """
            <div class="tools-heading">
                <div class="kicker">REGIONAL INTELLIGENCE</div>
                <div class="section" style="padding:0">
                    <h2>How does the signal differ by region?</h2>
                    <p>
                        Descriptive regional summaries of the latest processed field.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        regional_display = regional.copy()
        regional_display["Risk share"] = regional_display[
            "Risk share"
        ].map(
            lambda x: f"{x:.2f}%"
        )
        regional_display["Mean Chl-a"] = regional_display[
            "Mean Chl-a"
        ].round(4)
        regional_display["Maximum Chl-a"] = regional_display[
            "Maximum Chl-a"
        ].round(4)

        st.dataframe(
            regional_display,
            hide_index=True,
            width="stretch",
        )

    # --------------------------------------------------------
    # HIGHEST CURRENT-RISK CELLS
    # --------------------------------------------------------

    if not latest_risk.empty:

        risk_rank = latest_risk.copy()

        score_col = None
        for candidate in [
            "risk_probability",
            "model_score",
        ]:
            if candidate in risk_rank.columns:
                score_col = candidate
                break

        if score_col:
            risk_rank[score_col] = pd.to_numeric(
                risk_rank[score_col],
                errors="coerce"
            )
            risk_rank = risk_rank.sort_values(
                [score_col, "chla"],
                ascending=False
            )
        else:
            risk_rank = risk_rank.sort_values(
                "chla",
                ascending=False
            )

        cols_to_show = [
            "latitude",
            "longitude",
            "chla",
        ]

        if score_col:
            cols_to_show.append(score_col)

        if "chla_anomaly" in risk_rank.columns:
            cols_to_show.append("chla_anomaly")

        if "chla_change" in risk_rank.columns:
            cols_to_show.append("chla_change")

        top_risk = risk_rank[
            cols_to_show
        ].head(12).copy()

        rename = {
            "latitude": "Latitude",
            "longitude": "Longitude",
            "chla": "Chl-a",
            "risk_probability": "Model screening probability",
            "model_score": "Model score",
            "chla_anomaly": "Anomaly",
            "chla_change": "Recent change",
        }

        top_risk = top_risk.rename(
            columns=rename
        )

        st.markdown(
            """
            <div class="tools-heading">
                <div class="kicker">CURRENT PRIORITISATION</div>
                <div class="section" style="padding:0">
                    <h2>Highest current screening signals.</h2>
                    <p>
                        These rows are ordered using the model screening field
                        when available, with chlorophyll-a used as the fallback.
                        They are investigation candidates, not confirmed events.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            top_risk.round(4),
            hide_index=True,
            width="stretch",
        )

    st.markdown(
        """
        <div class="note">
            <b>Scientific interpretation:</b>
            BloomDetect AI is an early-warning support system. Satellite
            chlorophyll-a is a useful ocean-colour signal, but it cannot by
            itself establish harmfulness, species identity or toxin presence.
            Field validation and additional evidence remain necessary.
        </div>
        """,
        unsafe_allow_html=True,
    )

    report = latest.copy()

    st.download_button(
        "⬇ Download current intelligence table",
        report.to_csv(
            index=False
        ).encode("utf-8"),
        "bloomdetect_current_intelligence.csv",
        "text/csv",
        width="stretch",
    )


# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    page_heading(
        "04 · DATA & OUTPUTS",
        "The evidence behind the intelligence.",
        "This page is deliberately simple: source product, coverage, processing layer and actual downloadable outputs.",
    )

    coverage_start = (
        history["date"].min()
        if not history.empty
        else df["date"].min()
    )

    coverage_end = (
        history["date"].max()
        if not history.empty
        else df["date"].max()
    )

    data_cols = st.columns(
        4,
        gap="medium"
    )

    data_values = [
        (
            "Product",
            "E06OCM_L4_AC",
            "EOS-06 / OCM-3",
        ),
        (
            "Grid",
            "0.25°",
            "latest processed field",
        ),
        (
            "Coverage",
            (
                f"{coverage_start.strftime('%b %Y')} – "
                f"{coverage_end.strftime('%b %Y')}"
            ),
            "available observations",
        ),
        (
            "Latest field",
            f"{len(latest):,}",
            latest_date.strftime("%d %b %Y"),
        ),
    ]

    for col, item in zip(
        data_cols,
        data_values
    ):
        with col:
            metric(*item)

    st.markdown(
        """
        <div class="card" style="margin-top:18px">
            <div class="mini-label">SOURCE PRODUCT</div>
            <h3>EOS-06 / Oceansat-3 OCM-3 Level-4 analysed chlorophyll-a</h3>
            <p>
                The dashboard uses the E06OCM_L4_AC analysed chlorophyll-a
                product as the satellite-derived ocean-colour signal.
                The current project output is a potential bloom-risk
                screening layer built from that signal and temporal context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if history_name:
        st.markdown(
            f"""
            <div class="note">
                <b>Historical support layer:</b>
                {history_name}. This is the compact temporal visualization
                layer, not the original large ML dataset and not a second
                independent model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="tools-heading">
            <div class="kicker">PROJECT OUTPUTS</div>
            <div class="section" style="padding:0">
                <h2>Download the actual evidence.</h2>
                <p>
                    Every download below is generated from data used by the application.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dl1, dl2 = st.columns(
        2,
        gap="large"
    )

    with dl1:
        st.markdown(
            """
            <div class="download-card">
                <h3>Latest observations</h3>
                <p>
                    All processed cells in the latest available field,
                    including stored screening and supporting signals.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download latest observations",
            latest.to_csv(
                index=False
            ).encode("utf-8"),
            "bloomdetect_latest_observations.csv",
            "text/csv",
            width="stretch",
        )

    with dl2:
        st.markdown(
            """
            <div class="download-card">
                <h3>Potential-risk shortlist</h3>
                <p>
                    Only the cells currently included in the latest
                    potential bloom-risk screening output.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download potential-risk locations",
            latest_risk.to_csv(
                index=False
            ).encode("utf-8"),
            "bloomdetect_potential_risk.csv",
            "text/csv",
            width="stretch",
        )

    # Model evidence is included here once, instead of being repeated
    # throughout the dashboard. These are the project's reported metrics,
    # not values invented by the UI.
    st.markdown(
        """
        <div class="tools-heading">
            <div class="kicker">MODEL EVIDENCE</div>
            <div class="section" style="padding:0">
                <h2>Reported screening-model performance.</h2>
                <p>
                    The final project model is a Decision Tree trained for
                    the project's proxy screening target. These metrics should
                    not be interpreted as field-validated HAB detection accuracy.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
            ],
            "Final Decision Tree": [
                0.9993,
                0.8331,
                0.9988,
                0.9085,
                0.9997,
            ],
        }
    )

    model_metrics["Final Decision Tree"] = (
        model_metrics["Final Decision Tree"]
        .map(lambda x: f"{x:.4f}")
    )

    st.dataframe(
        model_metrics,
        hide_index=True,
        width="stretch",
    )

    st.markdown(
        """
        <div class="note">
            <b>Important:</b>
            the model target is derived from chlorophyll-a temporal behaviour.
            Therefore these metrics evaluate the project's proxy screening target,
            not confirmed harmful algal bloom events. Satellite observations
            cannot independently identify toxins or species.
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
