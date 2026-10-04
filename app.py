
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI - FINAL DEPLOYMENT VERSION
# Home | Risk Map | Location | Insights | Data
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
PRED_PATH = BASE / "latest_bloom_risk_predictions.csv"
IMAGE_PATH = BASE / "bloomdetect_bloom_process.png"

HISTORY_CANDIDATES = [
    BASE / "bloomdetect_history_web.csv.gz",
    BASE / "bloomdetect_history.csv.gz",
    BASE / "bloomdetect_history.csv",
]

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# Project proxy-screening thresholds.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner="Loading project data...")
def load_predictions():
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv is missing beside app.py."
        )

    d = pd.read_csv(PRED_PATH)

    required = {"latitude", "longitude", "date", "chla"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            "Prediction CSV is missing: " + ", ".join(sorted(missing))
        )

    # Support both the full deployed file and a compact risk column.
    if "risk_label" not in d.columns:
        if "risk" in d.columns:
            d["risk"] = pd.to_numeric(d["risk"], errors="coerce").fillna(0)
            d["risk_label"] = np.where(
                d["risk"].astype(int).eq(1),
                "Potential Bloom Risk",
                "Normal",
            )
        else:
            raise ValueError("Prediction CSV needs risk_label or risk.")

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for col in ["latitude", "longitude", "chla"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    for col in [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]:
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

    # Keep only the intended study window for dashboard statistics.
    d["plot_lon"] = d["longitude"]
    return d


@st.cache_data(show_spinner="Loading map history...")
def load_history():
    path = next((p for p in HISTORY_CANDIDATES if p.exists()), None)
    if path is None:
        return pd.DataFrame()

    h = pd.read_csv(path)

    required = {"date", "lat_bin", "lon_bin", "chla", "risk"}
    if not required.issubset(h.columns):
        return pd.DataFrame()

    h["date"] = pd.to_datetime(h["date"], errors="coerce")

    for col in ["lat_bin", "lon_bin", "chla", "risk"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")

    for col in ["cells", "max_chla"]:
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")

    h = h.replace([np.inf, -np.inf], np.nan)
    h = h.dropna(
        subset=["date", "lat_bin", "lon_bin", "chla"]
    ).copy()

    h = h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()

    h["risk"] = h["risk"].fillna(0).astype("int8")
    return h


try:
    df = load_predictions()
except Exception as exc:
    st.error("BloomDetect AI could not load the prediction dataset.")
    st.warning(str(exc))
    st.stop()

history = load_history()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()

study_latest = latest[
    latest["latitude"].between(LAT_MIN, LAT_MAX)
    & latest["longitude"].between(LON_MIN, LON_MAX)
].copy()

risk_latest = study_latest[study_latest["risk_flag"]].copy()


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --ink: #103f49;
    --muted: #66848b;
    --line: #a9d6da;
    --blue: #277fa4;
    --green: #22a879;
    --red: #ed5260;
    --teal: #0797a5;
    --shadow: 0 12px 32px rgba(9,80,94,.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background: #f2fbfb !important;
    color: var(--ink) !important;
    font-family: "DM Sans", sans-serif !important;
}

.stApp {
    background: linear-gradient(180deg,#fbffff 0%,#effafa 52%,#fbffff 100%) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer,
[data-testid="stSidebar"] {
    display: none !important;
}

.block-container {
    max-width: 1240px !important;
    padding: 22px 30px 65px !important;
}

h1, h2, h3, h4 {
    font-family: "Manrope", sans-serif !important;
    color: var(--ink) !important;
}

p, label, .stMarkdown {
    color: var(--muted);
}

.brand {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    border: 1.5px solid #c3e2e5;
    border-radius: 21px;
    background: rgba(255,255,255,.96);
    box-shadow: var(--shadow);
    margin-bottom: 12px;
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo {
    width: 47px;
    height: 47px;
    border-radius: 15px;
    background: linear-gradient(145deg,#28cbd0,#08798b);
    display: grid;
    place-items: center;
    color: white;
    font-weight: 800;
    font-size: 20px;
}

.brand-name {
    font: 800 1.2rem "Manrope";
    color: var(--ink);
}

.brand-sub {
    font-size: .71rem;
    color: #78959b;
}

.latest {
    text-align: right;
    font-size: .67rem;
    color: #78959b;
}

.latest b {
    color: #174b56;
    font-size: .78rem;
}

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    min-height: 42px !important;
    border-radius: 14px !important;
    background: white !important;
    border: 1.8px solid #164e5b !important;
    color: #123f49 !important;
    font-weight: 800 !important;
    box-shadow: 0 6px 15px rgba(15,70,82,.07) !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #e2f8f8 !important;
    border-color: #087d8c !important;
}

.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: #d8f7f7 !important;
    border-color: #078c9b !important;
}

.section {
    padding: 28px 0 15px;
}

.kicker {
    font: 800 .66rem "Manrope";
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 9px;
}

.section h2 {
    font: 800 clamp(2rem,4vw,3.35rem)/1.04 "Manrope";
    letter-spacing: -.06em;
    margin: 0 0 10px;
}

.section-copy {
    font-size: .94rem;
    line-height: 1.62;
    color: var(--muted);
    max-width: 1080px;
}

.hero-box {
    min-height: 350px;
    border-radius: 29px;
    padding: 48px;
    display: flex;
    align-items: center;
    background:
        radial-gradient(ellipse at 80% 20%,rgba(46,226,218,.30),transparent 45%),
        linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);
    box-shadow: 0 24px 62px rgba(6,86,100,.15);
    position: relative;
    overflow: hidden;
}

.hero-box:after {
    content: "";
    position: absolute;
    width: 900px;
    height: 900px;
    right: -430px;
    top: -450px;
    border: 2px solid rgba(220,255,252,.13);
    border-radius: 50%;
    box-shadow:
        0 0 0 65px rgba(220,255,252,.10),
        0 0 0 130px rgba(220,255,252,.08),
        0 0 0 195px rgba(220,255,252,.06);
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 760px;
}

.hero-kicker {
    color: #a6fffa;
    font: 800 .68rem "Manrope";
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.hero-title {
    color: #e4ffff !important;
    font: 800 clamp(2.8rem,6vw,5.1rem)/.94 "Manrope" !important;
    letter-spacing: -.075em;
    margin: 0 0 18px;
}

.hero-copy {
    color: #e0fbfb !important;
    font-size: 1rem;
    line-height: 1.75;
    max-width: 720px;
}

.badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.badge {
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.25);
    color: #efffff;
    font-size: .73rem;
    font-weight: 700;
}

.metric-card {
    background: rgba(255,255,255,.94);
    border: 1.5px solid var(--line);
    border-radius: 20px;
    padding: 15px;
    min-height: 104px;
    box-shadow: var(--shadow);
}

.metric-label {
    font: 800 .61rem "Manrope";
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #6d8c93;
}

.metric-value {
    font: 800 1.38rem "Manrope";
    color: var(--ink);
    margin-top: 5px;
}

.metric-note {
    font-size: .69rem;
    color: #78959b;
    margin-top: 4px;
}

.card {
    background: rgba(255,255,255,.94);
    border: 1.5px solid var(--line);
    border-radius: 20px;
    padding: 21px;
    box-shadow: var(--shadow);
}

.card-title {
    font: 800 1.2rem "Manrope";
    color: var(--ink);
    margin-bottom: 7px;
}

.card-copy {
    color: var(--muted);
    line-height: 1.6;
    font-size: .9rem;
}

.note {
    margin-top: 14px;
    padding: 12px 15px;
    background: #e6f8f8;
    border-left: 4px solid #11a9b2;
    border-radius: 0 13px 13px 0;
    color: #52757c;
    font-size: .82rem;
    line-height: 1.55;
}

.tool {
    padding: 19px;
    min-height: 142px;
    border: 1.5px solid var(--line);
    border-radius: 19px;
    background: white;
    box-shadow: var(--shadow);
}

.tool h3 {
    font: 800 1rem "Manrope";
    margin: 8px 0 6px;
    color: var(--ink);
}

.tool p {
    font-size: .83rem;
    line-height: 1.5;
    color: var(--muted);
}

.map-shell {
    border: 2px solid #8fcbd1;
    border-radius: 22px;
    background: #dff7f8;
    padding: 5px;
    overflow: hidden;
    box-shadow: var(--shadow);
}

.map-head {
    display: flex;
    justify-content: space-between;
    padding: 10px 12px;
    color: #174b56;
    font-size: .82rem;
}

.map-head span {
    color: #6b8b92;
    font-size: .68rem;
}

.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 9px;
    margin-top: 14px;
}

.result {
    padding: 11px 12px;
    border-radius: 12px;
    background: #f6fcfc;
    border: 1px solid #c8e5e7;
}

.result span,
.signal span {
    display: block;
    font-size: .65rem;
    color: #78959b;
    margin-bottom: 4px;
}

.result b,
.signal b {
    font-size: .88rem;
    color: #194b55;
}

.signal {
    padding: 13px;
    background: white;
    border: 1.5px solid var(--line);
    border-radius: 14px;
    box-shadow: var(--shadow);
}

.status {
    margin-top: 14px;
    padding: 14px;
    border-radius: 14px;
    border: 2px solid;
    font-size: .85rem;
}

.status b {
    display: block;
    font: 800 .9rem "Manrope";
    margin-bottom: 4px;
}

.status.risk {
    background: #fff0f2;
    border-color: #ed6976;
    color: #9b2d3c;
}

.status.normal {
    background: #eafaf4;
    border-color: #49b995;
    color: #176f58;
}

.chart-card {
    background: white;
    border: 1.5px solid var(--line);
    border-radius: 20px;
    padding: 16px;
    box-shadow: var(--shadow);
}

.footer {
    border-top: 1px solid #d5ebed;
    margin-top: 38px;
    padding-top: 15px;
    color: #76959b;
    font-size: .68rem;
}

[data-testid="stMetricValue"] {
    color: #103f49 !important;
}

[data-testid="stMetricLabel"] {
    color: #66848b !important;
}

@media(max-width:900px) {
    .block-container { padding: 18px !important; }
    .hero-box { padding: 36px 26px; }
}

@media(max-width:620px) {
    .hero-box { padding: 30px 20px; min-height: 320px; }
    .latest { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
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
        <div class="section">
            <div class="kicker">{kicker}</div>
            <h2>{title}</h2>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(value, digits=4):
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):.{digits}f}"


def model_score_text(row):
    value = row.get("risk_probability", np.nan)
    if pd.isna(value):
        value = row.get("model_score", np.nan)
    if pd.isna(value):
        return "Unavailable"
    value = float(value)
    if 0 <= value <= 1:
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


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
    return "Indian Ocean study area"


def nearest_valid(lat, lon):
    valid = study_latest[
        study_latest["chla"].gt(0) & study_latest["chla"].notna()
    ].copy()

    if valid.empty:
        return None

    scale = max(float(np.cos(np.deg2rad(float(lat)))), 0.25)

    lat_values = valid["latitude"].to_numpy(dtype=float)
    lon_values = valid["longitude"].to_numpy(dtype=float)

    distance = (
        (lat_values - float(lat)) ** 2
        + ((lon_values - float(lon)) * scale) ** 2
    )

    return valid.iloc[int(np.argmin(distance))]


def history_for_cell(lat, lon):
    if history.empty:
        return pd.DataFrame()

    lat_bin = np.floor(float(lat)) + 0.5
    lon_bin = np.floor(float(lon)) + 0.5

    h = history[
        history["lat_bin"].eq(lat_bin)
        & history["lon_bin"].eq(lon_bin)
    ].copy()

    return h.sort_values("date")


def location_signals(row):
    # Prefer the actual 0.25° features stored with the latest prediction.
    values = {
        "Current Chl-a": row.get("chla", np.nan),
        "Historical baseline": row.get("historical_baseline", np.nan),
        "Anomaly": row.get("chla_anomaly", np.nan),
        "Recent change": row.get("chla_change", np.nan),
    }

    # If the feature columns are unavailable, derive descriptive context
    # from the compact history layer only for display.
    if all(pd.isna(v) for v in values.values()):
        h = history_for_cell(row["latitude"], row["longitude"])

        if not h.empty:
            before = h[h["date"] < pd.Timestamp(row["date"])].copy()
            baseline = before["chla"].mean() if not before.empty else np.nan
            previous = before["chla"].iloc[-1] if not before.empty else np.nan

            values = {
                "Current Chl-a": float(row["chla"]),
                "Historical baseline": baseline,
                "Anomaly": (
                    float(row["chla"]) - baseline
                    if pd.notna(baseline)
                    else np.nan
                ),
                "Recent change": (
                    float(row["chla"]) - previous
                    if pd.notna(previous)
                    else np.nan
                ),
            }

    return values


def make_map(processed, risk_rows):
    fig = go.Figure()

    if processed.empty:
        return fig

    blue = processed.copy()

    # Use compact grid cells for the visual layer.
    if "lat_bin" not in blue.columns:
        blue["lat_bin"] = np.floor(blue["latitude"]) + 0.5
        blue["lon_bin"] = np.floor(blue["longitude"]) + 0.5

        blue = (
            blue.groupby(["lat_bin", "lon_bin"], as_index=False)
            .agg(
                mean_chla=("chla", "mean"),
                cells=("chla", "size"),
            )
        )
    else:
        blue = blue.rename(
            columns={
                "chla": "mean_chla",
                "cells": "cells",
            }
        )

    fig.add_trace(
        go.Scattergeo(
            lat=blue["lat_bin"].tolist(),
            lon=blue["lon_bin"].tolist(),
            mode="markers",
            name="Processed ocean field",
            marker=dict(
                size=7,
                color="#277fa4",
                opacity=0.68,
            ),
            customdata=np.column_stack(
                [
                    blue["mean_chla"].to_numpy(),
                    blue["cells"].fillna(0).to_numpy(),
                ]
            ),
            hovertemplate=(
                "<b>Processed ocean field</b><br>"
                "Mean Chl-a: %{customdata[0]:.4f}<br>"
                "Cells: %{customdata[1]:.0f}"
                "<extra></extra>"
            ),
        )
    )

    if not risk_rows.empty:
        r = risk_rows.copy()

        if "latitude" not in r.columns:
            r = r.rename(
                columns={
                    "lat_bin": "latitude",
                    "lon_bin": "longitude",
                }
            )

        r["longitude"] = pd.to_numeric(r["longitude"], errors="coerce")
        r = r.dropna(subset=["latitude", "longitude"])

        # Broad 2° × 2° concentration zones.
        r["zone_lat"] = np.floor(r["latitude"] / 2) * 2 + 1
        r["zone_lon"] = np.floor(r["longitude"] / 2) * 2 + 1

        zones = (
            r.groupby(["zone_lat", "zone_lon"], as_index=False)
            .size()
            .rename(columns={"size": "flags"})
        )

        zone_sizes = (
            (zones["flags"].astype(float) * 1.4 + 10)
            .clip(11, 38)
            .tolist()
        )

        fig.add_trace(
            go.Scattergeo(
                lat=zones["zone_lat"].tolist(),
                lon=zones["zone_lon"].tolist(),
                mode="markers",
                name="Flag concentration zone",
                marker=dict(
                    size=zone_sizes,
                    color="#22a879",
                    opacity=0.45,
                    line=dict(color="white", width=1.3),
                ),
                text=zones["flags"].astype(int).tolist(),
                hovertemplate=(
                    "<b>Flag concentration zone</b><br>"
                    "Flagged cells: %{text}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lat=r["latitude"].tolist(),
                lon=r["longitude"].tolist(),
                mode="markers",
                name="Potential bloom-risk cell",
                marker=dict(
                    size=6,
                    color="#ed5260",
                    opacity=0.92,
                    line=dict(color="white", width=0.5),
                ),
                text=r["chla"].map(lambda x: fmt(x)).tolist()
                if "chla" in r.columns
                else ["Unavailable"] * len(r),
                hovertemplate=(
                    "<b>Potential bloom-risk screening</b><br>"
                    "Chl-a: %{text}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#dff7f8",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.02,
            y=0.01,
            bgcolor="rgba(255,255,255,.92)",
            font=dict(size=11, color="#174b56"),
        ),
        geo=dict(
            showland=True,
            landcolor="#d7e5e3",
            showocean=True,
            oceancolor="#dff7f8",
            showlakes=True,
            lakecolor="#d5f1f3",
            showcountries=True,
            countrycolor="#8db4ba",
            showcoastlines=True,
            coastlinecolor="#739da4",
            coastlinewidth=0.8,
            projection_type="equirectangular",
            lataxis=dict(
                range=[LAT_MIN, LAT_MAX],
                showgrid=True,
                gridcolor="rgba(90,140,150,.18)",
            ),
            lonaxis=dict(
                range=[LON_MIN, LON_MAX],
                showgrid=True,
                gridcolor="rgba(90,140,150,.18)",
            ),
            bgcolor="#dff7f8",
        ),
    )

    return fig


# ============================================================
# HEADER + NAVIGATION
# ============================================================

latest_display = latest_date.strftime("%d %b %Y")

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
            <b>{latest_display}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = ["home", "map", "location", "insights", "data"]
NAV = {
    "home": "⌂ Home",
    "map": "🗺 Risk Map",
    "location": "📍 Location",
    "insights": "📊 Insights",
    "data": "⇩ Data",
}

if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"

nav_cols = st.columns(5, gap="small")

for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(
            NAV[page],
            key=f"nav_{page}",
            type="primary" if st.session_state.page == page else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = page
            st.rerun()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-content">
                <div class="hero-kicker">EOS-06 · OCM-3 · SATELLITE INTELLIGENCE</div>
                <div class="hero-title">Read the ocean signal.</div>
                <div class="hero-copy">
                    BloomDetect AI turns satellite-derived chlorophyll-a observations
                    and project screening output into an early-warning support view
                    for locations whose patterns may deserve closer investigation.
                </div>
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

    total = len(study_latest)
    flags = len(risk_latest)
    share = 100 * flags / total if total else 0
    maximum = study_latest["chla"].max() if total else np.nan

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    cards = st.columns(4, gap="medium")
    values = [
        ("Processed cells", f"{total:,}", "latest field"),
        ("Potential-risk cells", f"{flags:,}", "screening output"),
        ("Risk share", f"{share:.2f}%", "of processed cells"),
        ("Maximum Chl-a", fmt(maximum), "latest field"),
    ]

    for col, (label, value, note) in zip(cards, values):
        with col:
            metric_card(label, value, note)

    section(
        "PROJECT SNAPSHOT",
        "One clear view of the latest field.",
        "The home page gives the project context and current field status. Detailed investigation stays inside Risk Map, Location, Insights and Data.",
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="kicker">SCIENTIFIC CONTEXT</div>
                <div class="card-title">Chlorophyll-a is a signal, not a verdict.</div>
                <div class="card-copy">
                    The platform screens locations using satellite-derived
                    chlorophyll-a and temporal context. A potential-risk flag is
                    an investigation aid, not confirmation of a harmful algal bloom,
                    species or toxin.
                </div>
                <div class="note">
                    <b>Important:</b> high chlorophyll-a alone does not prove a
                    harmful algal bloom. Field observations and additional
                    environmental evidence are required for confirmation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if IMAGE_PATH.exists():
            st.image(
                str(IMAGE_PATH),
                use_container_width=True,
                caption="How satellite ocean-colour observations support bloom-risk investigation",
            )
        else:
            st.info("Project illustration is not included in the deployment folder.")

    section(
        "FOUR FOCUSED VIEWS",
        "Everything has one job.",
        "The website avoids duplicate pages and keeps each investigation task in one place.",
    )

    tools = [
        ("🗺️", "Risk Map", "Move through available dates and inspect spatial screening."),
        ("📍", "Location", "Check one coordinate against the nearest valid ocean observation."),
        ("📊", "Insights", "Study hotspot concentration and current Chl-a patterns."),
        ("⇩", "Data", "Review source information and download project outputs."),
    ]

    tool_cols = st.columns(4, gap="medium")

    for col, (icon, title, description) in zip(tool_cols, tools):
        with col:
            st.markdown(
                f"""
                <div class="tool">
                    <div style="font-size:1.3rem">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
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
        "Use the date slider to move through available observations. The latest date uses the final model screening output; earlier dates use the compact historical Chl-a screening layer.",
    )

    if not history.empty:
        available_dates = sorted(history["date"].dropna().unique())

        selected_date = st.slider(
            "Map timeline",
            min_value=available_dates[0].date(),
            max_value=available_dates[-1].date(),
            value=available_dates[-1].date(),
            format="DD MMM YYYY",
        )

        selected_date = pd.Timestamp(selected_date)
    else:
        selected_date = latest_date
        st.info(
            "The compact history file is not included yet. "
            "The map is showing the latest processed field."
        )

    if selected_date == latest_date:

        selected_full = study_latest.copy()
        selected_risk = risk_latest.copy()

        processed = len(selected_full)
        risk_count = len(selected_risk)

        blue = selected_full.copy()
        blue["lat_bin"] = np.floor(blue["latitude"]) + 0.5
        blue["lon_bin"] = np.floor(blue["longitude"]) + 0.5

        blue = (
            blue.groupby(["lat_bin", "lon_bin"], as_index=False)
            .agg(
                mean_chla=("chla", "mean"),
                cells=("chla", "size"),
            )
        )

        mode = "latest model screening output"

    else:

        h = history[history["date"].eq(selected_date)].copy()

        h = h[
            h["lat_bin"].between(LAT_MIN, LAT_MAX)
            & h["lon_bin"].between(LON_MIN, LON_MAX)
        ].copy()

        blue = h.copy()
        blue["cells"] = blue["cells"] if "cells" in blue.columns else 1
        blue["mean_chla"] = blue["chla"]

        selected_risk = blue[blue["risk"].gt(0)].copy()
        selected_risk = selected_risk.rename(
            columns={
                "lat_bin": "latitude",
                "lon_bin": "longitude",
            }
        )

        processed = len(blue)
        risk_count = len(selected_risk)
        mode = "historical screening layer"

    share = 100 * risk_count / processed if processed else 0

    metric_cols = st.columns(4, gap="medium")

    metric_values = [
        ("Selected date", selected_date.strftime("%d %b %Y"), "observation"),
        ("Processed cells", f"{processed:,}", "selected field"),
        ("Potential-risk", f"{risk_count:,}", "screening layer"),
        ("Risk share", f"{share:.2f}%", "selected field"),
    ]

    for col, (label, value, note) in zip(metric_cols, metric_values):
        with col:
            metric_card(label, value, note)

    st.markdown(
        """
        <div class="map-shell">
            <div class="map-head">
                <b>Indian Ocean · Arabian Sea · Bay of Bengal</b>
                <span>Blue field · green concentration · red screening flags</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    fig = make_map(blue, selected_risk)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )

    st.markdown(
        """
            <div class="map-head" style="justify-content:flex-start;gap:20px;flex-wrap:wrap">
                <span><b style="color:#277fa4">●</b> Processed ocean field</span>
                <span><b style="color:#22a879">●</b> Flag concentration zone</span>
                <span><b style="color:#ed5260">●</b> Potential bloom-risk cell</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "historical screening layer":
        st.info(
            "Earlier dates use a compact 1° historical screening layer derived "
            "from aggregated Chl-a history. These flags are screening proxies, "
            "not additional model predictions."
        )
    else:
        st.info(
            "Reading the map: green areas show spatial concentration of screening "
            "flags, not a separate severity score. Red cells are potential-risk "
            "screening results, not confirmed harmful algal blooms."
        )


# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page == "location":

    section(
        "02 · LOCATION INTELLIGENCE",
        "Check one coordinate across the available evidence.",
        "Your input is kept separate from the nearest valid processed ocean cell. Invalid or zero-value observations are excluded from the lookup.",
    )

    left, right = st.columns([0.82, 1.18], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="kicker">YOUR INPUT</div>
                <div class="card-title">Coordinates</div>
                <div class="card-copy">
                    Enter decimal degrees. Example: 17.38, 78.49.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("location_form", clear_on_submit=False):
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.get("lookup_lat", 17.38)),
                step=0.01,
                format="%.2f",
            )

            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.get("lookup_lon", 78.49)),
                step=0.01,
                format="%.2f",
            )

            submit = st.form_submit_button(
                "🔎 Check location",
                type="primary",
                use_container_width=True,
            )

        if submit:
            st.session_state.lookup_lat = float(lat)
            st.session_state.lookup_lon = float(lon)
            st.rerun()

    lat = float(st.session_state.get("lookup_lat", 17.38))
    lon = float(st.session_state.get("lookup_lon", 78.49))

    row = nearest_valid(lat, lon)

    with right:

        if row is None:
            st.error("No valid ocean observation is available for this lookup.")

        else:
            flagged = bool(row["risk_flag"])

            lat_scale = max(float(np.cos(np.deg2rad(lat))), 0.25)

            distance = np.sqrt(
                (float(row["latitude"]) - lat) ** 2
                + ((float(row["longitude"]) - lon) * lat_scale) ** 2
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="kicker">NEAREST VALID OCEAN CELL</div>
                    <div class="card-title">
                        {float(row["latitude"]):.4f}° · {float(row["longitude"]):.4f}°
                    </div>
                    <div class="card-copy">
                        <b>Your input:</b> {lat:.4f}° · {lon:.4f}°<br>
                        <b>Approx. angular separation:</b> {distance:.2f}°
                    </div>

                    <div class="result-grid">
                        <div class="result">
                            <span>Observation date</span>
                            <b>{pd.Timestamp(row["date"]).strftime("%d %b %Y")}</b>
                        </div>
                        <div class="result">
                            <span>Chlorophyll-a</span>
                            <b>{fmt(row["chla"])}</b>
                        </div>
                        <div class="result">
                            <span>Model score</span>
                            <b>{model_score_text(row)}</b>
                        </div>
                        <div class="result">
                            <span>Region</span>
                            <b>{region_name(row["latitude"], row["longitude"])}</b>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if flagged:
                st.markdown(
                    """
                    <div class="status risk">
                        <b>🔴 POTENTIAL BLOOM-RISK FLAG</b>
                        This valid ocean cell is included in the latest screening shortlist.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="status normal">
                        <b>🟢 NOT FLAGGED</b>
                        This valid ocean cell is not included in the latest potential-risk shortlist.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if row is not None:

        st.markdown(
            '<div class="kicker" style="margin-top:25px">SUPPORTING SIGNALS</div>',
            unsafe_allow_html=True,
        )

        signals = location_signals(row)

        signal_cols = st.columns(4, gap="small")

        for col, (label, value) in zip(signal_cols, signals.items()):
            with col:
                st.markdown(
                    f"""
                    <div class="signal">
                        <span>{label}</span>
                        <b>{fmt(value)}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        local_history = history_for_cell(
            row["latitude"],
            row["longitude"],
        )

        if not local_history.empty:

            chart = px.line(
                local_history,
                x="date",
                y="chla",
            )

            chart.update_traces(
                line_color="#0aa8b5",
                line_width=2.7,
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Chl-a: %{y:.4f}<extra></extra>"
                ),
            )

            chart.update_layout(
                height=340,
                margin=dict(l=70, r=25, t=30, b=70),
                paper_bgcolor="white",
                plot_bgcolor="white",
                showlegend=False,
                font=dict(
                    color="#174b56",
                    size=12,
                ),
                xaxis=dict(
                    title="Observation date",
                    title_font=dict(size=13),
                    tickfont=dict(size=11),
                    gridcolor="#e3eeee",
                ),
                yaxis=dict(
                    title="Mean chlorophyll-a",
                    title_font=dict(size=13),
                    tickfont=dict(size=11),
                    gridcolor="#e3eeee",
                ),
            )

            st.markdown(
                """
                <div class="chart-card">
                    <div class="kicker">LOCATION HISTORY</div>
                    <div class="card-title">How the local Chl-a signal changed</div>
                """,
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
                config={"displaylogo": False},
            )

            st.markdown("</div>", unsafe_allow_html=True)

        report = pd.DataFrame(
            [
                {
                    "input_latitude": lat,
                    "input_longitude": lon,
                    "nearest_latitude": row["latitude"],
                    "nearest_longitude": row["longitude"],
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "chla": row["chla"],
                    "historical_baseline": signals["Historical baseline"],
                    "chla_anomaly": signals["Anomaly"],
                    "chla_change": signals["Recent change"],
                    "risk_label": row["risk_label"],
                    "model_score": row.get(
                        "risk_probability",
                        row.get("model_score", np.nan),
                    ),
                    "region": region_name(
                        row["latitude"],
                        row["longitude"],
                    ),
                }
            ]
        )

        st.download_button(
            "⬇ Download location report",
            report.to_csv(index=False).encode("utf-8"),
            "bloomdetect_location_report.csv",
            "text/csv",
            use_container_width=True,
        )

        st.caption(
            "Model score is the stored classifier output. It is not a calibrated "
            "probability that a harmful algal bloom is present."
        )


# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":

    section(
        "03 · OCEAN INTELLIGENCE",
        "Where should the current signal receive closer attention?",
        "Hotspot concentration and current-field statistics are combined here because they answer the same investigation question without duplicating the Risk Map.",
    )

    total = len(study_latest)
    flags = len(risk_latest)
    share = 100 * flags / total if total else 0

    metric_cols = st.columns(4, gap="medium")

    insight_values = [
        ("Observation cells", f"{total:,}", "latest field"),
        ("Potential-risk", f"{flags:,}", "screening output"),
        ("Risk share", f"{share:.2f}%", "latest field"),
        ("Mean Chl-a", fmt(study_latest["chla"].mean()), "latest field"),
    ]

    for col, (label, value, note) in zip(metric_cols, insight_values):
        with col:
            metric_card(label, value, note)

    # --------------------------------------------------------
    # Hotspot zones
    # --------------------------------------------------------

    risk_zone = risk_latest.copy()

    if not risk_zone.empty:

        risk_zone["lat_zone"] = (
            np.floor(risk_zone["latitude"] / 2) * 2 + 1
        )
        risk_zone["lon_zone"] = (
            np.floor(risk_zone["longitude"] / 2) * 2 + 1
        )

        zones = (
            risk_zone.groupby(
                ["lat_zone", "lon_zone"],
                as_index=False,
            )
            .agg(
                flagged_cells=("risk_flag", "size"),
                mean_chla=("chla", "mean"),
                max_chla=("chla", "max"),
            )
            .sort_values("flagged_cells", ascending=False)
            .head(12)
        )

    else:
        zones = pd.DataFrame()

    st.markdown(
        """
        <div class="section">
            <div class="kicker">HOTSPOT INTELLIGENCE</div>
            <h2>Where are the flags concentrating?</h2>
            <div class="section-copy">
                Nearby screening flags are grouped into broad 2° × 2°
                investigation zones. This is a spatial concentration view,
                not a severity ranking.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if zones.empty:
        st.info("No potential-risk cells are available in the latest study field.")

    else:

        zones = zones.copy()

        zones["Zone"] = zones.apply(
            lambda x: (
                f'{x["lat_zone"]:.0f}°–{x["lat_zone"] + 2:.0f}°, '
                f'{x["lon_zone"]:.0f}°–{x["lon_zone"] + 2:.0f}°'
            ),
            axis=1,
        )

        chart_data = zones.iloc[::-1]

        fig = px.bar(
            chart_data,
            x="flagged_cells",
            y="Zone",
            orientation="h",
            text="flagged_cells",
        )

        fig.update_traces(
            marker_color="#22a879",
            textposition="outside",
            cliponaxis=False,
            textfont=dict(
                size=12,
                color="#174b56",
            ),
        )

        fig.update_layout(
            height=450,
            margin=dict(l=190, r=70, t=25, b=70),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            font=dict(
                color="#174b56",
                size=12,
            ),
            xaxis=dict(
                title="Potential-risk cells in zone",
                title_font=dict(size=13),
                tickfont=dict(size=11),
                gridcolor="#e3eeee",
            ),
            yaxis=dict(
                title="Investigation zone",
                title_font=dict(size=13),
                tickfont=dict(size=11),
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="kicker">TOP INVESTIGATION ZONES</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)

        table = zones[
            ["Zone", "flagged_cells", "mean_chla", "max_chla"]
        ].rename(
            columns={
                "flagged_cells": "Flagged cells",
                "mean_chla": "Mean Chl-a",
                "max_chla": "Max Chl-a",
            }
        )

        table["Mean Chl-a"] = table["Mean Chl-a"].round(4)
        table["Max Chl-a"] = table["Max Chl-a"].round(4)

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Current field
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section">
            <div class="kicker">CURRENT FIELD</div>
            <h2>What does the latest observation look like?</h2>
            <div class="section-copy">
                These charts describe the latest field without repeating the spatial map.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:

        hist_fig = px.histogram(
            study_latest,
            x="chla",
            nbins=32,
        )

        hist_fig.update_traces(
            marker_color="#277fa4",
        )

        hist_fig.update_layout(
            height=410,
            margin=dict(l=70, r=35, t=30, b=80),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            font=dict(
                color="#174b56",
                size=12,
            ),
            xaxis=dict(
                title="Satellite-derived chlorophyll-a",
                title_font=dict(size=13),
                tickfont=dict(size=11),
                gridcolor="#e3eeee",
            ),
            yaxis=dict(
                title="Number of processed cells",
                title_font=dict(size=13),
                tickfont=dict(size=11),
                gridcolor="#e3eeee",
            ),
        )

        st.markdown(
            """
            <div class="chart-card">
                <div class="kicker">DISTRIBUTION</div>
                <div class="card-title">Chlorophyll-a distribution</div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with chart_right:

        normal_mean = study_latest.loc[
            ~study_latest["risk_flag"],
            "chla",
        ].mean()

        risk_mean = study_latest.loc[
            study_latest["risk_flag"],
            "chla",
        ].mean()

        comparison = pd.DataFrame(
            {
                "Screening group": [
                    "Normal",
                    "Potential bloom risk",
                ],
                "Mean Chl-a": [
                    normal_mean,
                    risk_mean,
                ],
            }
        )

        comp_fig = px.bar(
            comparison,
            x="Screening group",
            y="Mean Chl-a",
            text="Mean Chl-a",
        )

        comp_fig.update_traces(
            marker_color=["#277fa4", "#ed5260"],
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False,
            textfont=dict(
                size=12,
                color="#174b56",
            ),
        )

        comp_fig.update_layout(
            height=410,
            margin=dict(l=70, r=60, t=30, b=90),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            font=dict(
                color="#174b56",
                size=12,
            ),
            xaxis=dict(
                title="Screening group",
                title_font=dict(size=13),
                tickfont=dict(size=11),
                gridcolor="#ffffff",
            ),
            yaxis=dict(
                title="Mean chlorophyll-a",
                title_font=dict(size=13),
                tickfont=dict(size=11),
                gridcolor="#e3eeee",
            ),
        )

        st.markdown(
            """
            <div class="chart-card">
                <div class="kicker">SCREENING GROUPS</div>
                <div class="card-title">Mean chlorophyll-a by screening group</div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            comp_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Regional summary
    # --------------------------------------------------------

    regional = []

    region_rules = [
        (
            "Arabian Sea",
            study_latest["latitude"].between(5, 30)
            & study_latest["longitude"].between(45, 75),
        ),
        (
            "Bay of Bengal",
            study_latest["latitude"].between(0, 25)
            & study_latest["longitude"].between(75.01, 100),
        ),
        (
            "Southern Indian Ocean",
            study_latest["latitude"].between(-30, 5)
            & study_latest["longitude"].between(40, 100),
        ),
        (
            "Northern Indian Ocean",
            study_latest["latitude"].between(5, 30)
            & study_latest["longitude"].between(75.01, 120),
        ),
    ]

    for name, condition in region_rules:
        subset = study_latest[condition]

        if len(subset):
            regional.append(
                {
                    "Region": name,
                    "Cells": len(subset),
                    "Potential-risk cells": int(
                        subset["risk_flag"].sum()
                    ),
                    "Risk share": 100 * subset["risk_flag"].mean(),
                    "Mean Chl-a": subset["chla"].mean(),
                }
            )

    regional_df = pd.DataFrame(regional)

    if not regional_df.empty:

        regional_df["Risk share"] = regional_df["Risk share"].map(
            lambda x: f"{x:.2f}%"
        )

        regional_df["Mean Chl-a"] = regional_df["Mean Chl-a"].round(4)

        st.markdown(
            """
            <div class="section">
                <div class="kicker">REGIONAL SIGNAL</div>
                <h2>Broad-area comparison.</h2>
                <div class="section-copy">
                    Descriptive summaries of the latest processed field.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            regional_df,
            hide_index=True,
            use_container_width=True,
        )

    st.info(
        "Interpretation: these views describe satellite-derived chlorophyll-a "
        "and project screening output. They do not independently establish "
        "harmfulness, species identity or toxin presence."
    )


# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    section(
        "04 · DATA & OUTPUTS",
        "The evidence behind the dashboard.",
        "Source information and downloadable project outputs live here so scientific details are not repeated across the other pages.",
    )

    data_cols = st.columns(4, gap="medium")

    data_values = [
        ("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3"),
        ("Grid", "0.25°", "latitude × longitude"),
        ("Latest cells", f"{len(study_latest):,}", "processed observation"),
        ("Latest date", latest_display, "processed dataset"),
    ]

    for col, (label, value, note) in zip(data_cols, data_values):
        with col:
            metric_card(label, value, note)

    st.markdown(
        """
        <div class="card" style="margin-top:18px">
            <div class="kicker">SOURCE PRODUCT</div>
            <div class="card-title">EOS-06 OCM-3 analysed chlorophyll-a</div>
            <div class="card-copy">
                The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4
                analysed chlorophyll product, E06OCM_L4_AC. The source field
                is provided on a 0.25° × 0.25° grid.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section(
        "DOWNLOADS",
        "Take the actual project outputs.",
        "These files are generated directly from the data used by the dashboard.",
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Latest observation table</div>
                <div class="card-copy">
                    Latest processed cells, including screening and available supporting signals.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download latest observations",
            latest.to_csv(index=False).encode("utf-8"),
            "bloomdetect_latest_observations.csv",
            "text/csv",
            use_container_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Potential-risk shortlist</div>
                <div class="card-copy">
                    Latest cells currently screened as potential bloom risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download potential-risk locations",
            risk_latest.to_csv(index=False).encode("utf-8"),
            "bloomdetect_potential_risk.csv",
            "text/csv",
            use_container_width=True,
        )

    st.info(
        "Scientific note: a potential bloom-risk flag is a project screening "
        "output. It is not confirmation of a harmful algal bloom, species identity "
        "or toxin presence. Satellite observations should be combined with field "
        "observations and additional environmental evidence."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening ·
        Latest processed field: {latest_display}
    </div>
    """,
    unsafe_allow_html=True,
)
