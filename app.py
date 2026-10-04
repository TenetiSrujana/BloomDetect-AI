from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# FINAL STABLE STREAMLIT VERSION
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
PREDICTION_FILE = BASE_DIR / "latest_bloom_risk_predictions.csv"
HISTORY_FILE = BASE_DIR / "bloomdetect_history.csv"
IMAGE_FILE = BASE_DIR / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
    --ink:#103f49;
    --muted:#64838a;
    --aqua:#0aa8b5;
    --line:#acd9dd;
    --pale:#f1fbfb;
    --green:#22a879;
    --red:#ef4f5e;
    --blue:#2383a8;
}

html, body, [data-testid="stAppViewContainer"]{
    background:#f1fbfb !important;
    color:var(--ink) !important;
    font-family:'DM Sans',sans-serif !important;
}

.stApp{
    background:linear-gradient(180deg,#fbffff 0%,#effafa 55%,#fbffff 100%) !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu,
footer, [data-testid="stSidebar"]{
    display:none !important;
}

.block-container{
    max-width:1260px !important;
    padding:22px 28px 65px !important;
}

.brand{
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:14px 18px;
    border:1.5px solid #c5e4e6;
    border-radius:20px;
    background:rgba(255,255,255,.94);
    box-shadow:0 12px 32px rgba(9,80,94,.08);
}

.brand-left{
    display:flex;
    align-items:center;
    gap:12px;
}

.logo{
    width:46px;
    height:46px;
    border-radius:15px;
    display:grid;
    place-items:center;
    background:linear-gradient(145deg,#29cdd0,#087b8e);
    color:white;
    font-weight:800;
    font-size:20px;
}

.brand-name{
    font:800 1.2rem Manrope,sans-serif;
    letter-spacing:-.03em;
}

.brand-sub{
    color:#78959b;
    font-size:.72rem;
}

.latest{
    text-align:right;
    color:#78959b;
    font-size:.68rem;
}
.latest b{color:#174b56;font-size:.78rem}

.stButton>button,
.stFormSubmitButton>button,
.stDownloadButton>button{
    min-height:44px !important;
    border-radius:13px !important;
    border:1.7px solid #164e5b !important;
    background:#fff !important;
    color:#123f49 !important;
    font-weight:800 !important;
    box-shadow:0 5px 14px rgba(15,70,82,.07) !important;
}

.stButton>button:hover,
.stFormSubmitButton>button:hover,
.stDownloadButton>button:hover{
    background:#e3f8f8 !important;
    border-color:#078c9b !important;
}

.stButton>button[kind="primary"],
.stFormSubmitButton>button[kind="primary"]{
    background:#d9f7f7 !important;
    border-color:#078c9b !important;
}

.kicker{
    font:800 .66rem Manrope,sans-serif;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:10px;
}

.page-title{
    font:800 clamp(2.2rem,4.4vw,4rem)/1.04 Manrope,sans-serif;
    letter-spacing:-.06em;
    color:#103f49;
    margin:0 0 13px;
}

.page-copy{
    color:#5f8088;
    line-height:1.65;
    max-width:1050px;
    font-size:.96rem;
}

.hero{
    margin-top:18px;
    position:relative;
    overflow:hidden;
    border-radius:30px;
    padding:60px 55px;
    min-height:365px;
    background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);
    box-shadow:0 25px 65px rgba(6,86,100,.16);
}

.hero:after{
    content:"";
    position:absolute;
    width:500px;
    height:500px;
    right:-170px;
    top:-180px;
    border-radius:50%;
    border:80px solid rgba(190,255,252,.08);
    box-shadow:0 0 0 65px rgba(190,255,252,.05);
}

.hero-content{position:relative;z-index:2;max-width:900px}
.hero .kicker{color:#a6fffa}
.hero h1{
    font:800 clamp(3rem,6vw,5.7rem)/.92 Manrope,sans-serif;
    letter-spacing:-.075em;
    color:#e4ffff;
    margin:0 0 20px;
}
.hero p{
    color:#e0fbfb;
    line-height:1.75;
    max-width:790px;
    font-size:1.02rem;
}

.badges{
    display:flex;
    flex-wrap:wrap;
    gap:9px;
    margin-top:22px;
}
.badge{
    padding:8px 12px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.25);
    color:#efffff;
    font-size:.77rem;
    font-weight:700;
}

.card{
    background:rgba(255,255,255,.92);
    border:1.5px solid var(--line);
    border-radius:20px;
    box-shadow:0 12px 30px rgba(15,91,101,.07);
    padding:21px;
}

.card h3{
    font:800 1.22rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 8px;
}

.card p{
    color:#66848b;
    line-height:1.62;
    margin:0;
    font-size:.91rem;
}

.metric{
    min-height:108px;
    padding:16px;
    border-radius:17px;
    background:linear-gradient(145deg,#fff,#eaf8f8);
    border:1.5px solid #a6d5da;
    box-shadow:0 10px 25px rgba(15,91,101,.07);
}

.metric-label{
    font:800 .63rem Manrope,sans-serif;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:#6d8c93;
}
.metric-value{
    font:800 1.45rem Manrope,sans-serif;
    color:#123f49;
    margin-top:6px;
}
.metric-note{
    color:#78959b;
    font-size:.72rem;
    margin-top:4px;
}

.note{
    margin-top:16px;
    padding:13px 15px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 13px 13px 0;
    color:#52757c;
    font-size:.84rem;
    line-height:1.55;
}

.section-gap{margin-top:30px}

.image-box{
    padding:8px;
    overflow:hidden;
}
.image-caption{
    color:#6f8c92;
    text-align:center;
    font-size:.72rem;
    padding:7px 4px 1px;
}

.small-label{
    font:800 .63rem Manrope,sans-serif;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:#6d8c93;
    margin-bottom:7px;
}

.status{
    padding:14px 16px;
    border-radius:14px;
    border:2px solid;
    margin-top:14px;
}
.status-risk{background:#fff0f2;border-color:#f06a78;color:#9e2d3c}
.status-normal{background:#eafaf4;border-color:#49b995;color:#176f58}
.status-title{font:800 .9rem Manrope,sans-serif}

.footer{
    border-top:1px solid #d5ebed;
    margin-top:42px;
    padding-top:15px;
    color:#76959b;
    font-size:.69rem;
}

[data-testid="stMetric"]{
    background:#fff;
    border:1px solid #c5e3e5;
    border-radius:15px;
    padding:12px;
}

@media(max-width:800px){
    .block-container{padding:15px 13px 45px !important}
    .hero{padding:42px 25px}
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_predictions():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv is missing beside app.py."
        )

    d = pd.read_csv(PREDICTION_FILE)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            "Prediction CSV is missing: " + ", ".join(sorted(missing))
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for c in ["latitude", "longitude", "chla"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    numeric_cols = [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]

    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(
        subset=["latitude", "longitude", "date", "chla"]
    ).copy()

    d["risk_flag"] = (
        d["risk_label"].astype(str).str.strip().str.lower()
        == "potential bloom risk"
    )

    return d


@st.cache_data(show_spinner=False)
def load_history():
    """
    Optional compact historical layer.
    Expected columns:
    date, lat_bin, lon_bin, chla, risk, cells, max_chla

    If the file is missing or clearly too large, the app safely falls back
    to the latest prediction field instead of crashing.
    """
    if not HISTORY_FILE.exists():
        return None, "compact history file is not available"

    # Do not let an accidentally uploaded 800 MB / 1 GB file kill Streamlit.
    try:
        size_mb = HISTORY_FILE.stat().st_size / (1024 * 1024)
        if size_mb > 180:
            return None, "history file is too large for this deployment"
    except OSError:
        return None, "history file could not be inspected"

    try:
        h = pd.read_csv(HISTORY_FILE)

        required = {
            "date", "lat_bin", "lon_bin", "chla", "risk"
        }
        missing = required - set(h.columns)
        if missing:
            return None, "history file is missing required columns"

        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        for c in ["lat_bin", "lon_bin", "chla", "risk"]:
            h[c] = pd.to_numeric(h[c], errors="coerce")

        h = h.dropna(
            subset=["date", "lat_bin", "lon_bin", "chla"]
        ).copy()

        h["risk"] = h["risk"].fillna(0).astype("int8")
        return h, None

    except Exception:
        return None, "history file could not be read"


try:
    df = load_predictions()
except Exception as exc:
    st.error("BloomDetect AI could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()
latest_risk = latest[latest["risk_flag"]].copy()

history, history_message = load_history()

# ============================================================
# HELPERS
# ============================================================

def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(kicker, title, copy):
    st.markdown(
        f"""
        <div class="section-gap">
            <div class="kicker">{kicker}</div>
            <div class="page-title">{title}</div>
            <div class="page-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_value(value, digits=4):
    try:
        if pd.isna(value):
            return "Unavailable"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "Unavailable"


def probability_value(row):
    value = row.get("risk_probability", np.nan)

    try:
        if pd.isna(value):
            value = row.get("model_score", np.nan)

        if pd.isna(value):
            return "Unavailable"

        value = float(value)
        if value <= 1:
            value *= 100

        return f"{value:.1f}%"
    except Exception:
        return "Unavailable"


def region_name(lat, lon):
    lat = float(lat)
    lon = float(lon)

    if 5 <= lat <= 30 and 45 <= lon <= 75:
        return "Arabian Sea"

    if 0 <= lat <= 25 and 75 < lon <= 100:
        return "Bay of Bengal"

    if -30 <= lat < 5 and 40 <= lon <= 100:
        return "Southern Indian Ocean"

    if 5 <= lat <= 30 and 75 < lon <= 120:
        return "Northern Indian Ocean"

    return "Other study area"


def nearest_latest(lat, lon):
    """
    Simple Euclidean nearest-cell lookup.
    Longitudes are used directly because the project study area is
    entirely within 20E–120E.
    """
    a = latest[
        latest["latitude"].between(LAT_MIN, LAT_MAX)
        & latest["longitude"].between(LON_MIN, LON_MAX)
    ].copy()

    if a.empty:
        a = latest.copy()

    distance = (
        (a["latitude"].to_numpy(dtype=float) - float(lat)) ** 2
        + (a["longitude"].to_numpy(dtype=float) - float(lon)) ** 2
    )

    return a.iloc[int(np.argmin(distance))]


def study_latest():
    return latest[
        latest["latitude"].between(LAT_MIN, LAT_MAX)
        & latest["longitude"].between(LON_MIN, LON_MAX)
    ].copy()


def make_latest_map(data):
    """
    Stable Scattergeo map.
    No projection helper tricks, no marker-size expressions that can
    fail when a selected date has no risk cells.
    """
    d = data.copy()

    if d.empty:
        return go.Figure()

    d["lat_bin"] = np.floor(d["latitude"]).astype(float) + 0.5
    d["lon_bin"] = np.floor(d["longitude"]).astype(float) + 0.5

    blue = (
        d.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(
            mean_chla=("chla", "mean"),
            cells=("chla", "size"),
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=blue["lat_bin"],
            lon=blue["lon_bin"],
            mode="markers",
            name="Processed ocean field",
            marker=dict(
                size=7,
                color="#2383a8",
                opacity=0.68,
            ),
            customdata=np.column_stack(
                [
                    blue["lat_bin"],
                    blue["lon_bin"],
                    blue["mean_chla"],
                    blue["cells"],
                ]
            ),
            hovertemplate=(
                "<b>Processed ocean field</b><br>"
                "Latitude: %{customdata[0]:.1f}°<br>"
                "Longitude: %{customdata[1]:.1f}°<br>"
                "Mean Chl-a: %{customdata[2]:.4f}<br>"
                "Cells represented: %{customdata[3]}<extra></extra>"
            ),
        )
    )

    flagged = d[d["risk_flag"]].copy()

    if not flagged.empty:
        flagged["zone_lat"] = np.floor(flagged["latitude"] / 2) * 2 + 1
        flagged["zone_lon"] = np.floor(flagged["longitude"] / 2) * 2 + 1

        zones = (
            flagged.groupby(
                ["zone_lat", "zone_lon"], as_index=False
            )
            .agg(
                flagged_cells=("risk_flag", "size"),
                mean_chla=("chla", "mean"),
            )
        )

        # Fixed marker size avoids Plotly/NumPy errors when values are
        # empty, zero or unexpectedly typed.
        zone_size = np.clip(
            np.sqrt(zones["flagged_cells"].astype(float)) * 5 + 8,
            10,
            30,
        )

        fig.add_trace(
            go.Scattergeo(
                lat=zones["zone_lat"],
                lon=zones["zone_lon"],
                mode="markers",
                name="Flag concentration zone",
                marker=dict(
                    size=zone_size,
                    color="#22a879",
                    opacity=0.55,
                    line=dict(width=1.5, color="white"),
                ),
                customdata=np.column_stack(
                    [
                        zones["zone_lat"],
                        zones["zone_lon"],
                        zones["flagged_cells"],
                        zones["mean_chla"],
                    ]
                ),
                hovertemplate=(
                    "<b>Flag concentration zone</b><br>"
                    "Centre: %{customdata[0]:.1f}°, %{customdata[1]:.1f}°<br>"
                    "Flagged cells: %{customdata[2]}<br>"
                    "Mean Chl-a: %{customdata[3]:.4f}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lat=flagged["latitude"],
                lon=flagged["longitude"],
                mode="markers",
                name="Potential bloom-risk cell",
                marker=dict(
                    size=7,
                    color="#ef4f5e",
                    opacity=0.96,
                    line=dict(width=1, color="white"),
                ),
                customdata=np.column_stack(
                    [
                        flagged["latitude"],
                        flagged["longitude"],
                        flagged["chla"],
                    ]
                ),
                hovertemplate=(
                    "<b>Potential bloom-risk screening flag</b><br>"
                    "Latitude: %{customdata[0]:.2f}°<br>"
                    "Longitude: %{customdata[1]:.2f}°<br>"
                    "Chl-a: %{customdata[2]:.4f}<extra></extra>"
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
        lonaxis=dict(range=[LON_MIN, LON_MAX]),
        lataxis=dict(range=[LAT_MIN, LAT_MAX]),
        center=dict(lat=-5, lon=70),
        projection_type="equirectangular",
    )

    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="#dff7f8",
        geo=dict(bgcolor="#dff7f8"),
        font=dict(color="#174b56"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#b9dfe2",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    return fig


def make_history_map(h):
    if h is None or h.empty:
        return go.Figure()

    h = h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()

    if h.empty:
        return go.Figure()

    blue = h.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=blue["lat_bin"],
            lon=blue["lon_bin"],
            mode="markers",
            name="Processed ocean field",
            marker=dict(
                size=6,
                color="#2383a8",
                opacity=0.58,
            ),
            customdata=np.column_stack(
                [
                    blue["lat_bin"],
                    blue["lon_bin"],
                    blue["chla"],
                ]
            ),
            hovertemplate=(
                "<b>Historical Chl-a field</b><br>"
                "Latitude: %{customdata[0]:.1f}°<br>"
                "Longitude: %{customdata[1]:.1f}°<br>"
                "Mean Chl-a: %{customdata[2]:.4f}<extra></extra>"
            ),
        )
    )

    flagged = h[h["risk"].astype(int) == 1].copy()

    if not flagged.empty:
        flagged["zone_lat"] = np.floor(flagged["lat_bin"] / 2) * 2 + 1
        flagged["zone_lon"] = np.floor(flagged["lon_bin"] / 2) * 2 + 1

        zones = (
            flagged.groupby(
                ["zone_lat", "zone_lon"], as_index=False
            )
            .agg(flagged_cells=("risk", "sum"))
        )

        zone_size = np.clip(
            np.sqrt(zones["flagged_cells"].astype(float)) * 5 + 8,
            10,
            28,
        )

        fig.add_trace(
            go.Scattergeo(
                lat=zones["zone_lat"],
                lon=zones["zone_lon"],
                mode="markers",
                name="Flag concentration zone",
                marker=dict(
                    size=zone_size,
                    color="#22a879",
                    opacity=0.55,
                    line=dict(width=1.5, color="white"),
                ),
                customdata=zones[["flagged_cells"]].to_numpy(),
                hovertemplate=(
                    "<b>Historical screening concentration</b><br>"
                    "Flagged cells: %{customdata[0]}<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lat=flagged["lat_bin"],
                lon=flagged["lon_bin"],
                mode="markers",
                name="Potential bloom-risk cell",
                marker=dict(
                    size=6,
                    color="#ef4f5e",
                    opacity=0.90,
                    line=dict(width=1, color="white"),
                ),
                customdata=flagged[["chla"]].to_numpy(),
                hovertemplate=(
                    "<b>Historical potential bloom-risk screening</b><br>"
                    "Chl-a: %{customdata[0]:.4f}<extra></extra>"
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
        showcountries=True,
        countrycolor="#9ab9be",
        bgcolor="#dff7f8",
        lonaxis=dict(range=[LON_MIN, LON_MAX]),
        lataxis=dict(range=[LAT_MIN, LAT_MAX]),
        center=dict(lat=-5, lon=70),
        projection_type="equirectangular",
    )

    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="#dff7f8",
        geo=dict(bgcolor="#dff7f8"),
        font=dict(color="#174b56"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#b9dfe2",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    return fig


def regional_table(d):
    definitions = [
        (
            "Arabian Sea",
            d["latitude"].between(5, 30)
            & d["longitude"].between(45, 75),
        ),
        (
            "Bay of Bengal",
            d["latitude"].between(0, 25)
            & d["longitude"].between(75.01, 100),
        ),
        (
            "Southern Indian Ocean",
            d["latitude"].between(-30, 5)
            & d["longitude"].between(40, 100),
        ),
        (
            "Northern Indian Ocean",
            d["latitude"].between(5, 30)
            & d["longitude"].between(75.01, 120),
        ),
    ]

    rows = []

    for name, mask in definitions:
        sub = d[mask]
        if not sub.empty:
            rows.append(
                {
                    "Region": name,
                    "Cells": len(sub),
                    "Potential-risk cells": int(sub["risk_flag"].sum()),
                    "Risk share": 100 * float(sub["risk_flag"].mean()),
                    "Mean Chl-a": float(sub["chla"].mean()),
                    "Max Chl-a": float(sub["chla"].max()),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# HEADER + NAVIGATION
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

PAGES = ["home", "map", "location", "insights", "data"]
LABELS = {
    "home": "⌂ Home",
    "map": "🗺 Risk Map",
    "location": "📍 Location",
    "insights": "📊 Insights",
    "data": "⇩ Data",
}

st.markdown(
    f"""
    <div class="brand">
        <div class="brand-left">
            <div class="logo">≈</div>
            <div>
                <div class="brand-name">BloomDetect AI</div>
                <div class="brand-sub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div>
            </div>
        </div>
        <div class="latest">
            Latest processed field<br>
            <b>{latest_date.strftime("%d %b %Y")}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_cols = st.columns(5, gap="small")

for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(
            LABELS[page],
            key=f"nav_{page}",
            width="stretch",
            type="primary" if st.session_state.page == page else "secondary",
        ):
            st.session_state.page = page
            st.rerun()

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
                <div class="badges">
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
        '<div class="section-gap"><div class="kicker">PROJECT SNAPSHOT</div>'
        '<div class="page-title">One clear view of the signal.</div></div>',
        unsafe_allow_html=True,
    )

    risk_share = 100 * len(latest_risk) / len(latest) if len(latest) else 0

    mcols = st.columns(4, gap="medium")

    values = [
        ("Processed cells", f"{len(latest):,}", "latest field"),
        ("Potential-risk cells", f"{len(latest_risk):,}", "screening output"),
        ("Risk share", f"{risk_share:.2f}%", "of processed cells"),
        ("Maximum Chl-a", safe_value(latest["chla"].max()), "latest field"),
    ]

    for col, item in zip(mcols, values):
        with col:
            metric_card(*item)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">FROM SATELLITE OBSERVATION TO INVESTIGATION SUPPORT</div>
                <h3>Chlorophyll-a is a signal, not a verdict.</h3>
                <p>
                    BloomDetect combines satellite-derived chlorophyll-a with
                    temporal context to screen for unusual patterns.
                    The result is a <b>potential bloom-risk screening signal</b>,
                    not confirmation of a harmful algal bloom.
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
        if IMAGE_FILE.exists():
            st.markdown('<div class="card image-box">', unsafe_allow_html=True)
            st.image(
                str(IMAGE_FILE),
                width="stretch",
                caption="How satellite ocean-colour observations can support bloom-risk investigation",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="card" style="min-height:300px;display:grid;place-items:center">
                    <div style="text-align:center;color:#6b8b92">
                        <div style="font-size:3rem">🛰️🌊</div>
                        <b>EOS-06 ocean-colour observation</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-gap"><div class="kicker">FOUR FOCUSED VIEWS</div>'
        '<div class="page-title" style="font-size:2.4rem">Everything has one job.</div></div>',
        unsafe_allow_html=True,
    )

    fcols = st.columns(4, gap="medium")

    features = [
        ("🗺️", "Risk Map", "Move through available dates and inspect spatial screening patterns."),
        ("📍", "Location", "Check one coordinate against the nearest processed satellite cell."),
        ("📊", "Insights", "Inspect regional patterns, concentrations and the latest Chl-a distribution."),
        ("⇩", "Data", "Keep source information and downloadable outputs in one place."),
    ]

    for col, (icon, title, desc) in zip(fcols, features):
        with col:
            st.markdown(
                f"""
                <div class="card" style="min-height:145px">
                    <div style="font-size:1.4rem;margin-bottom:8px">{icon}</div>
                    <h3 style="font-size:1.05rem">{title}</h3>
                    <p>{desc}</p>
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
        "See how the field changes.",
        "Use the date slider to move through available observations. The latest date uses the stored model screening output; earlier dates use the compact historical Chl-a screening layer.",
    )

    # Date handling
    if history is not None and not history.empty:
        all_dates = sorted(
            pd.concat(
                [
                    pd.Series(history["date"].dropna().unique()),
                    pd.Series([latest_date]),
                ]
            ).drop_duplicates()
        )
        all_dates = [pd.Timestamp(x) for x in all_dates]
    else:
        all_dates = [latest_date]

    if len(all_dates) > 1:
        selected_date = st.slider(
            "Map timeline",
            min_value=min(all_dates).date(),
            max_value=max(all_dates).date(),
            value=max(all_dates).date(),
            format="DD MMM YYYY",
        )
        selected_date = pd.Timestamp(selected_date)
    else:
        selected_date = latest_date
        st.info(
            "The compact history file is not available in this deployment. "
            "The latest processed field remains fully available."
        )

    using_latest = selected_date == latest_date

    if using_latest:
        selected = study_latest()
        selected_risk = selected[selected["risk_flag"]]
        map_fig = make_latest_map(selected)
        selected_cells = len(selected)
        selected_flags = len(selected_risk)
        map_note = "Latest date: stored model screening output."
    else:
        if history is None:
            selected = pd.DataFrame()
            selected_flags = 0
            selected_cells = 0
            map_fig = go.Figure()
            map_note = "Historical layer unavailable."
        else:
            selected = history[
                history["date"] == selected_date
            ].copy()

            selected = selected[
                selected["lat_bin"].between(LAT_MIN, LAT_MAX)
                & selected["lon_bin"].between(LON_MIN, LON_MAX)
            ]

            selected_flags = int(
                selected["risk"].fillna(0).astype(int).sum()
            )
            selected_cells = len(selected)
            map_fig = make_history_map(selected)
            map_note = (
                "Earlier date: compact historical screening layer derived "
                "from aggregated Chl-a history."
            )

    risk_share = (
        100 * selected_flags / selected_cells
        if selected_cells
        else 0
    )

    cards = st.columns(4, gap="medium")

    card_values = [
        ("Selected date", selected_date.strftime("%d %b %Y"), "observation"),
        ("Map cells", f"{selected_cells:,}", "selected field"),
        ("Potential-risk", f"{selected_flags:,}", "screening layer"),
        ("Risk share", f"{risk_share:.2f}%", "selected field"),
    ]

    for col, item in zip(cards, card_values):
        with col:
            metric_card(*item)

    st.markdown(
        '<div class="card" style="padding:7px;margin-top:14px">',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        map_fig,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;color:#55777e;font-size:.8rem">
            <span>🔵 Blue = processed ocean field</span>
            <span>🟢 Green = flagged-cell concentration</span>
            <span>🔴 Red = individual potential-risk cell</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="note">
            <b>Reading the map:</b> {map_note}
            Green areas indicate spatial concentration of screening flags.
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
        "Your input stays separate from the nearest valid processed 0.25° satellite cell, so it is always clear which location was actually evaluated.",
    )

    if "location_lat" not in st.session_state:
        st.session_state.location_lat = 17.40

    if "location_lon" not in st.session_state:
        st.session_state.location_lon = 78.50

    left, right = st.columns([.82, 1.18], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">YOUR INPUT</div>
                <h3>Coordinates</h3>
                <p>Enter decimal degrees. Example: 17.38, 78.49.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("location_form", clear_on_submit=False):
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.location_lat),
                step=0.25,
                format="%.2f",
            )

            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.location_lon),
                step=0.25,
                format="%.2f",
            )

            submitted = st.form_submit_button(
                "🔎 Check location",
                width="stretch",
                type="primary",
            )

        if submitted:
            st.session_state.location_lat = float(lat)
            st.session_state.location_lon = float(lon)
            st.rerun()

    lat = float(st.session_state.location_lat)
    lon = float(st.session_state.location_lon)

    try:
        row = nearest_latest(lat, lon)
    except Exception:
        row = None

    if row is None:
        st.error("No valid processed observation is available for lookup.")
    else:
        flagged = bool(row["risk_flag"])

        with right:
            st.markdown(
                f"""
                <div class="card">
                    <div class="small-label">NEAREST PROCESSED OCEAN CELL</div>
                    <h3 style="font-size:1.65rem">
                        {float(row["latitude"]):.4f}° · {float(row["longitude"]):.4f}°
                    </h3>
                    <p>
                        This is the satellite grid cell actually used for the lookup.
                        <br><b>Your input:</b> {lat:.4f}° · {lon:.4f}°
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rcols = st.columns(2, gap="small")

            details = [
                ("Observation date", row["date"].strftime("%d %b %Y")),
                ("Chlorophyll-a", safe_value(row["chla"])),
                ("Model score", probability_value(row)),
                ("Region", region_name(row["latitude"], row["longitude"])),
            ]

            for col, (label, value) in zip(rcols * 2, details):
                with col:
                    st.metric(label, value)

            if flagged:
                st.markdown(
                    """
                    <div class="status status-risk">
                        <div class="status-title">🔴 POTENTIAL BLOOM-RISK FLAG</div>
                        <div>This processed cell is included in the current screening shortlist.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="status status-normal">
                        <div class="status-title">🟢 NOT FLAGGED</div>
                        <div>This processed cell is not included in the current potential-risk shortlist.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            inside = (
                LAT_MIN <= lat <= LAT_MAX
                and LON_MIN <= lon <= LON_MAX
            )

            if not inside:
                st.warning(
                    "Your input is outside the main dashboard study window. "
                    "The nearest processed cell is still shown."
                )

        st.markdown(
            '<div class="section-gap"><div class="small-label">SUPPORTING SIGNALS</div></div>',
            unsafe_allow_html=True,
        )

        signal_cols = st.columns(4, gap="small")

        signals = [
            ("Current Chl-a", safe_value(row["chla"])),
            (
                "Historical baseline",
                safe_value(row.get("historical_baseline", np.nan)),
            ),
            (
                "Anomaly",
                safe_value(row.get("chla_anomaly", np.nan)),
            ),
            (
                "Recent change",
                safe_value(row.get("chla_change", np.nan)),
            ),
        ]

        for col, (label, value) in zip(signal_cols, signals):
            with col:
                st.metric(label, value)

        report = pd.DataFrame(
            [
                {
                    "input_latitude": lat,
                    "input_longitude": lon,
                    "nearest_processed_latitude": row["latitude"],
                    "nearest_processed_longitude": row["longitude"],
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "chla": row["chla"],
                    "historical_baseline": row.get(
                        "historical_baseline", np.nan
                    ),
                    "chla_anomaly": row.get(
                        "chla_anomaly", np.nan
                    ),
                    "chla_change": row.get(
                        "chla_change", np.nan
                    ),
                    "risk_label": row["risk_label"],
                    "risk_probability": row.get(
                        "risk_probability", np.nan
                    ),
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
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":

    section(
        "03 · INSIGHTS",
        "See the patterns behind the screening output.",
        "This page combines the useful concentration and distribution views into one place. It does not create another independent risk score.",
    )

    risk_count = int(latest["risk_flag"].sum())
    normal_count = int((~latest["risk_flag"]).sum())
    share = 100 * risk_count / len(latest) if len(latest) else 0

    cards = st.columns(4, gap="medium")

    vals = [
        ("Observation cells", f"{len(latest):,}", "latest field"),
        ("Potential-risk", f"{risk_count:,}", "screening output"),
        ("Normal", f"{normal_count:,}", "latest field"),
        ("Risk share", f"{share:.2f}%", "latest field"),
    ]

    for col, item in zip(cards, vals):
        with col:
            metric_card(*item)

    regional = regional_table(latest)

    if not regional.empty:
        st.markdown(
            '<div class="section-gap"><div class="small-label">REGIONAL VIEW</div></div>',
            unsafe_allow_html=True,
        )

        display_regional = regional.copy()
        display_regional["Risk share"] = display_regional[
            "Risk share"
        ].map(lambda x: f"{x:.2f}%")
        display_regional["Mean Chl-a"] = display_regional[
            "Mean Chl-a"
        ].round(4)
        display_regional["Max Chl-a"] = display_regional[
            "Max Chl-a"
        ].round(4)

        st.dataframe(
            display_regional,
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="card"><div class="small-label">SCREENING COMPOSITION</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure(
            go.Bar(
                x=["Normal", "Potential bloom risk"],
                y=[normal_count, risk_count],
                marker_color=["#2383a8", "#ef4f5e"],
                text=[normal_count, risk_count],
                textposition="outside",
                cliponaxis=False,
            )
        )

        fig.update_layout(
            height=370,
            margin=dict(l=60, r=20, t=35, b=70),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Screening outcome",
                title_font=dict(size=14),
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="Number of cells",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor="#d6e7e8",
            ),
            showlegend=False,
        )

        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="card"><div class="small-label">CHLOROPHYLL-A DISTRIBUTION</div>',
            unsafe_allow_html=True,
        )

        clean = latest["chla"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        fig = go.Figure(
            go.Histogram(
                x=clean,
                nbinsx=32,
                marker_color="#0aa8b5",
            )
        )

        fig.update_layout(
            height=370,
            margin=dict(l=60, r=20, t=35, b=70),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Satellite-derived chlorophyll-a",
                title_font=dict(size=14),
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="Number of cells",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor="#d6e7e8",
            ),
            showlegend=False,
        )

        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-gap"><div class="small-label">CURRENT INVESTIGATION QUEUE</div>'
        '<div class="page-title" style="font-size:2.2rem">Flagged cells to inspect.</div>',
        unsafe_allow_html=True,
    )

    queue = latest_risk.copy().head(20)

    keep = [
        c for c in [
            "latitude",
            "longitude",
            "chla",
            "chla_anomaly",
            "chla_change",
            "risk_probability",
        ]
        if c in queue.columns
    ]

    if keep:
        queue = queue[keep].copy()

        for c in [
            "chla",
            "chla_anomaly",
            "chla_change",
            "risk_probability",
        ]:
            if c in queue.columns:
                queue[c] = queue[c].round(4)

        st.dataframe(
            queue,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No potential-risk cells are present in the latest field.")

    if history is not None:
        st.markdown(
            """
            <div class="note">
                Historical dates in the Risk Map use a compact 1° aggregated
                Chl-a screening layer. Those earlier-date flags are screening
                proxies, not additional stored model predictions.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    section(
        "04 · DATA",
        "Source data and project outputs.",
        "Everything related to the dataset and downloads stays here, so the other pages can remain focused on analysis.",
    )

    cards = st.columns(4, gap="medium")

    data_values = [
        ("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3"),
        ("Grid", "0.25°", "latitude × longitude"),
        ("Latest cells", f"{len(latest):,}", "processed observation"),
        ("Latest date", latest_date.strftime("%d %b %Y"), "processed dataset"),
    ]

    for col, item in zip(cards, data_values):
        with col:
            metric_card(*item)

    st.markdown(
        """
        <div class="section-gap card">
            <div class="small-label">SOURCE</div>
            <h3>EOS-06 OCM-3 analysed chlorophyll-a</h3>
            <p>
                The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4
                analysed chlorophyll product (<b>E06OCM_L4_AC</b>).
                The current project output is a potential bloom-risk screening
                layer built from the satellite-derived chlorophyll-a signal
                and temporal context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-gap"><div class="small-label">DOWNLOADS</div>'
        '<div class="page-title" style="font-size:2.25rem">Project outputs.</div></div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown(
            """
            <div class="card">
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

    with d2:
        st.markdown(
            """
            <div class="card">
                <h3>Potential-risk shortlist</h3>
                <p>Only cells currently screened as potential bloom risk.</p>
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

    st.markdown(
        """
        <div class="section-gap card">
            <div class="small-label">INTERPRETATION</div>
            <h3>Use the result correctly.</h3>
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
