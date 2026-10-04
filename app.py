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

# History file can be any ONE of these names.
HISTORY_CANDIDATES = [
    BASE_DIR / "bloomdetect_history_web.csv.gz",
    BASE_DIR / "bloomdetect_history.csv.gz",
    BASE_DIR / "bloomdetect_history.csv",
]

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# Proxy-screening thresholds used by the project.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner="Loading BloomDetect data...")
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
            "Prediction CSV is missing: " + ", ".join(sorted(missing))
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for c in ["latitude", "longitude", "chla"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    numeric = [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]

    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

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

    return d


@st.cache_data(show_spinner="Loading map timeline...")
def load_history():
    path = next((p for p in HISTORY_CANDIDATES if p.exists()), None)

    if path is None:
        return pd.DataFrame(), None

    h = pd.read_csv(path)

    required = {"date", "lat_bin", "lon_bin", "chla", "risk"}
    if not required.issubset(h.columns):
        return pd.DataFrame(), None

    h["date"] = pd.to_datetime(h["date"], errors="coerce")

    for c in ["lat_bin", "lon_bin", "chla", "risk"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")

    h = h.replace([np.inf, -np.inf], np.nan)
    h = h.dropna(
        subset=["date", "lat_bin", "lon_bin", "chla"]
    ).copy()

    h["risk"] = h["risk"].fillna(0).astype("int8")

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
# STYLE
# ============================================================

st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
    --ink:#103f49;
    --deep:#07566a;
    --aqua:#08a7b3;
    --muted:#64838a;
    --line:#a8d9dd;
    --bg:#f2fbfb;
    --green:#20a477;
    --red:#ef4f5e;
    --blue:#197da5;
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
    max-width:1220px!important;
    padding:26px 28px 65px!important;
    margin:0 auto!important;
    overflow-x:hidden!important;
}

/* ---------- header ---------- */

.brand-bar{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    width:100%;
    box-sizing:border-box;
    padding:16px 20px;
    border:1.5px solid #c4e4e6;
    background:rgba(255,255,255,.95);
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
    font-size:.74rem;
    margin-top:2px;
}

.latest-label{
    color:#78959b;
    font-size:.68rem;
    line-height:1.35;
    text-align:right;
    white-space:nowrap;
}

.latest-label b{
    color:#174b56;
    font-size:.78rem;
}

.nav-wrap{
    margin:12px 0 24px;
}

/* ---------- buttons ---------- */

.stButton>button,
.stDownloadButton>button,
.stFormSubmitButton>button{
    min-height:45px!important;
    border-radius:14px!important;
    background:#fff!important;
    border:2px solid #164e5b!important;
    color:#123f49!important;
    font-weight:800!important;
    font-size:.86rem!important;
    box-shadow:0 5px 15px rgba(15,70,82,.08)!important;
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

.stSlider label{
    color:#174b56!important;
    font-weight:800!important;
}

/* ---------- common ---------- */

.section{
    padding:18px 0 10px;
    width:100%;
    box-sizing:border-box;
}

.kicker{
    font:800 .67rem Manrope,sans-serif;
    letter-spacing:.17em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:10px;
}

.section h2{
    font:800 clamp(2rem,4vw,3.2rem)/1.05 Manrope,sans-serif;
    letter-spacing:-.055em;
    color:var(--ink);
    margin:0 0 12px;
    overflow-wrap:anywhere;
    word-break:normal;
}

.section p{
    color:#5f8088;
    line-height:1.65;
    margin:0;
    max-width:1040px;
    font-size:.95rem;
    overflow-wrap:anywhere;
}

.card{
    box-sizing:border-box;
    width:100%;
    background:rgba(255,255,255,.92);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:0 13px 34px rgba(9,80,94,.08);
    padding:22px;
}

.card h3{
    font:800 1.25rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 9px;
    overflow-wrap:anywhere;
}

.card p{
    color:#66848b;
    line-height:1.62;
    margin:0;
    font-size:.91rem;
}

.mini-label{
    font:800 .63rem Manrope,sans-serif;
    letter-spacing:.14em;
    text-transform:uppercase;
    color:#6d8c93;
    margin-bottom:8px;
}

.note{
    box-sizing:border-box;
    width:100%;
    margin-top:16px;
    padding:13px 15px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 13px 13px 0;
    color:#52757c;
    font-size:.84rem;
    line-height:1.58;
    overflow-wrap:anywhere;
}

/* ---------- home ---------- */

.hero{
    box-sizing:border-box;
    width:100%;
    min-height:420px;
    position:relative;
    overflow:hidden;
    border-radius:30px;
    padding:58px;
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

.hero-content{
    position:relative;
    z-index:2;
    max-width:820px;
}

.hero .kicker{
    color:#a6fffa;
}

.hero h1{
    font:800 clamp(3.1rem,6vw,5.7rem)/.92 Manrope,sans-serif;
    letter-spacing:-.075em;
    color:#e4ffff;
    margin:0 0 20px;
    overflow-wrap:anywhere;
}

.hero p{
    font-size:1.03rem;
    line-height:1.75;
    color:#e0fbfb;
    max-width:760px;
    margin:0;
}

.hero-badges{
    display:flex;
    flex-wrap:wrap;
    gap:9px;
    margin-top:24px;
}

.badge{
    padding:9px 13px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.25);
    color:#efffff;
    font-size:.77rem;
    font-weight:700;
}

.metrics{
    margin-top:18px;
}

.metric{
    min-height:116px;
    box-sizing:border-box;
    padding:18px;
    background:linear-gradient(145deg,#fff,#eaf8f8);
    border:1.5px solid #a6d5da;
    border-radius:19px;
    box-shadow:0 10px 25px rgba(15,91,101,.07);
    display:flex;
    flex-direction:column;
    justify-content:center;
}

.metric .label{
    font:800 .63rem Manrope,sans-serif;
    letter-spacing:.11em;
    text-transform:uppercase;
    color:#6d8c93;
}

.metric .value{
    font:800 1.42rem Manrope,sans-serif;
    color:#123f49;
    margin-top:6px;
    overflow-wrap:anywhere;
}

.metric .note{
    margin:4px 0 0;
    padding:0;
    border:0;
    background:none;
    font-size:.71rem;
    color:#78959b;
}

/* ---------- story ---------- */

.story-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:22px;
    align-items:center;
    margin-top:20px;
}

.story-image{
    width:100%;
    box-sizing:border-box;
    border:1.5px solid #a9d6da;
    border-radius:22px;
    overflow:hidden;
    background:#dff7f8;
    box-shadow:0 13px 34px rgba(9,80,94,.08);
    padding:0;
}

.story-image img{
    display:block;
    width:100%;
    height:auto;
    border-radius:20px;
}

/* ---------- tools ---------- */

.tools-heading{
    margin-top:36px;
    padding:0;
}

.tools-heading h2{
    font-size:clamp(2rem,3.5vw,3rem);
}

.tool-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:16px;
    margin-top:18px;
    width:100%;
}

.tool-item{
    min-width:0;
}

.tool-card{
    box-sizing:border-box;
    width:100%;
    min-height:156px;
    padding:19px;
    background:rgba(255,255,255,.9);
    border:1.5px solid #b5dde0;
    border-radius:19px;
    box-shadow:0 10px 25px rgba(9,80,94,.07);
}

.tool-card .icon{
    font-size:1.28rem;
    margin-bottom:9px;
}

.tool-card h3{
    font:800 1.02rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 7px;
    overflow-wrap:anywhere;
}

.tool-card p{
    font-size:.82rem;
    line-height:1.52;
    color:#66848b;
    margin:0;
    overflow-wrap:anywhere;
}

.tool-button{
    margin-top:9px;
}

/* ---------- map ---------- */

.timeline-card{
    box-sizing:border-box;
    width:100%;
    margin:14px 0 18px;
    padding:15px 18px;
    background:rgba(255,255,255,.93);
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
    font:800 .88rem Manrope;
    color:#174b56;
}

.timeline-help{
    font-size:.73rem;
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
    min-height:44px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    padding:0 12px;
    color:#174b56;
}

.map-title b{
    font-size:.9rem;
}

.map-title span{
    font-size:.69rem;
    color:#6b8b92;
}

.map-legend{
    display:flex;
    align-items:center;
    gap:8px 10px;
    flex-wrap:wrap;
    color:#4f747b;
    font-size:.76rem;
    padding:12px 14px;
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

/* ---------- location ---------- */

.result-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:9px;
    margin-top:14px;
}

.result-item{
    padding:11px 12px;
    background:#f5fcfc;
    border:1px solid #c7e5e7;
    border-radius:13px;
}

.result-item span{
    display:block;
    font-size:.68rem;
    color:#78959b;
    margin-bottom:4px;
}

.result-item b{
    color:#194b55;
    font-size:.88rem;
    overflow-wrap:anywhere;
}

.status{
    margin-top:14px;
    padding:14px 15px;
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
    font:800 .9rem Manrope;
}

.signal-grid{
    display:grid;
    grid-template-columns:repeat(4,minmax(0,1fr));
    gap:10px;
    margin-top:13px;
}

.signal{
    min-width:0;
    padding:12px;
    border-radius:14px;
    background:#f7fcfc;
    border:1px solid #c9e5e7;
}

.signal .label{
    font-size:.67rem;
    color:#78959b;
}

.signal .value{
    font:800 .9rem Manrope;
    color:#164a55;
    margin-top:4px;
    overflow-wrap:anywhere;
}

/* ---------- charts/data ---------- */

.chart-card{
    box-sizing:border-box;
    width:100%;
    background:#fff;
    border:1.5px solid #b6dfe2;
    border-radius:20px;
    box-shadow:0 10px 28px rgba(9,80,94,.07);
    padding:14px;
}

.chart-title{
    font:800 .93rem Manrope;
    color:#123f49;
    padding:4px 4px 8px;
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
    font-size:.84rem;
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

.download-card{
    box-sizing:border-box;
    min-height:140px;
    padding:19px;
    border-radius:19px;
    background:linear-gradient(135deg,#087b8b,#13adb3);
    border:2px solid #07576a;
    box-shadow:0 13px 28px rgba(8,91,102,.15);
    color:#fff;
}

.download-card h3{
    font:800 1.08rem Manrope;
    color:#fff;
    margin:0 0 6px;
}

.download-card p{
    font-size:.82rem;
    color:#e5ffff;
    line-height:1.45;
    margin:0;
}

.footer{
    border-top:1px solid #d5ebed;
    margin-top:42px;
    padding-top:15px;
    color:#76959b;
    font-size:.69rem;
}

/* ---------- inputs ---------- */

.stNumberInput input{
    border:2px solid #164e5b!important;
    border-radius:12px!important;
    background:#fff!important;
}

/* ---------- responsive ---------- */

@media(max-width:900px){
    .story-grid{grid-template-columns:1fr;}
    .tool-grid{grid-template-columns:1fr 1fr;}
    .signal-grid{grid-template-columns:1fr 1fr;}
}

@media(max-width:620px){
    .block-container{padding:15px 12px 45px!important;}
    .brand-bar{padding:13px 14px;}
    .latest-label{display:none;}
    .hero{padding:38px 24px;min-height:390px;}
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


def safe_num(value, digits=4):
    try:
        if pd.isna(value):
            return "Unavailable"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "Unavailable"


def probability_text(value):
    try:
        if pd.isna(value):
            return "Unavailable"
        x = float(value)
        if x <= 1:
            x *= 100
        return f"{x:.1f}%"
    except Exception:
        return "Unavailable"


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


def nearest_latest(lat, lon):
    a = latest

    lon_scale = max(np.cos(np.deg2rad(float(lat))), 0.25)

    dist = (
        (a["latitude"].to_numpy() - float(lat)) ** 2
        + (
            (a["longitude"].to_numpy() - float(lon))
            * lon_scale
        ) ** 2
    )

    return a.iloc[int(np.argmin(dist))]


def history_dates():
    if history.empty:
        return []

    return sorted(
        pd.to_datetime(history["date"].dropna().unique())
    )


def history_for_date(selected_date):
    if history.empty:
        return pd.DataFrame()

    h = history[
        history["date"].eq(pd.Timestamp(selected_date))
    ].copy()

    return h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()


def map_figure(selected_date):
    selected_date = pd.Timestamp(selected_date)

    # Use the original 0.25° latest prediction field for the newest date.
    if selected_date == latest_date:
        field = latest.copy()

        field["lat_bin"] = (
            np.floor(field["latitude"]) + 0.5
        )
        field["lon_bin"] = (
            np.floor(field["longitude"]) + 0.5
        )

        blue = (
            field.groupby(
                ["lat_bin", "lon_bin"],
                as_index=False
            )
            .agg(
                mean_chla=("chla", "mean"),
                cells=("chla", "size")
            )
        )

        risk_cells = field[field["risk_flag"]].copy()

        processed_count = len(field)
        risk_count = int(field["risk_flag"].sum())

    else:
        # Historical timeline is intentionally aggregated for the website.
        h = history_for_date(selected_date)

        if h.empty:
            selected_date = latest_date
            field = latest.copy()

            field["lat_bin"] = (
                np.floor(field["latitude"]) + 0.5
            )
            field["lon_bin"] = (
                np.floor(field["longitude"]) + 0.5
            )

            blue = (
                field.groupby(
                    ["lat_bin", "lon_bin"],
                    as_index=False
                )
                .agg(
                    mean_chla=("chla", "mean"),
                    cells=("chla", "size")
                )
            )

            risk_cells = field[field["risk_flag"]].copy()

            processed_count = len(field)
            risk_count = int(field["risk_flag"].sum())

        else:
            blue = h.rename(
                columns={
                    "lat_bin": "lat",
                    "lon_bin": "lon",
                    "chla": "mean_chla",
                }
            ).copy()

            risk_cells = blue[
                blue["risk"].fillna(0).astype(int).eq(1)
            ].copy()

            risk_cells = risk_cells.rename(
                columns={
                    "lat": "latitude",
                    "lon": "longitude",
                    "mean_chla": "chla",
                }
            )

            processed_count = len(blue)
            risk_count = int(
                blue["risk"].fillna(0).astype(int).sum()
            )

    fig = go.Figure()

    # Blue processed field
    if not blue.empty:
        lat_col = "lat_bin" if "lat_bin" in blue.columns else "lat"
        lon_col = "lon_bin" if "lon_bin" in blue.columns else "lon"

        fig.add_trace(
            go.Scattergeo(
                lat=blue[lat_col],
                lon=blue[lon_col],
                mode="markers",
                name="Processed ocean field",
                marker=dict(
                    size=7,
                    color="#197da5",
                    opacity=0.58,
                ),
                text=[
                    f"Mean Chl-a: {x:.4f}"
                    for x in blue["mean_chla"]
                ],
                hovertemplate=(
                    "<b>Processed ocean field</b><br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

    # Green concentration zones
    if not risk_cells.empty:
        rc = risk_cells.copy()

        rc["zone_lat"] = (
            np.floor(
                rc["latitude"] / 2
            ) * 2 + 1
        )
        rc["zone_lon"] = (
            np.floor(
                rc["longitude"] / 2
            ) * 2 + 1
        )

        zones = (
            rc.groupby(
                ["zone_lat", "zone_lon"],
                as_index=False
            )
            .size()
            .rename(columns={"size": "flagged_cells"})
        )

        fig.add_trace(
            go.Scattergeo(
                lat=zones["zone_lat"],
                lon=zones["zone_lon"],
                mode="markers",
                name="Flag concentration zone",
                marker=dict(
                    size=np.clip(
                        zones["flagged_cells"] * 1.3 + 9,
                        10,
                        34
                    ),
                    color="#20a477",
                    opacity=0.42,
                    line=dict(
                        width=1.2,
                        color="#ffffff"
                    ),
                ),
                text=zones["flagged_cells"],
                hovertemplate=(
                    "<b>Flag concentration zone</b><br>"
                    "Flagged cells: %{text}"
                    "<extra></extra>"
                ),
            )
        )

        # Red individual screening cells
        fig.add_trace(
            go.Scattergeo(
                lat=rc["latitude"],
                lon=rc["longitude"],
                mode="markers",
                name="Potential bloom-risk cell",
                marker=dict(
                    size=6,
                    color="#ef4f5e",
                    opacity=0.92,
                    line=dict(
                        width=0.7,
                        color="#ffffff"
                    ),
                ),
                text=[
                    f"Chl-a: {x:.4f}"
                    for x in rc["chla"]
                ],
                hovertemplate=(
                    "<b>Potential bloom-risk screening</b><br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

    # IMPORTANT: projection_type, not projection.
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
        height=610,
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
            bgcolor="rgba(255,255,255,.92)",
            bordercolor="#b9dfe2",
            borderwidth=1,
            font=dict(
                size=10,
                color="#174b56"
            ),
        ),
    )

    return fig, processed_count, risk_count, selected_date


def hotspot_table(data):
    r = data[data["risk_flag"]].copy()

    if r.empty:
        return pd.DataFrame()

    r["lat_zone"] = (
        np.floor(r["latitude"] / 2) * 2
    )
    r["lon_zone"] = (
        np.floor(r["longitude"] / 2) * 2
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

    z["Flagged cells"] = z["flagged_cells"].astype(int)
    z["Mean Chl-a"] = z["mean_chla"].round(4)
    z["Max Chl-a"] = z["max_chla"].round(4)

    return z[
        ["Zone", "Flagged cells", "Mean Chl-a", "Max Chl-a"]
    ]


def navigate(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# HEADER
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

if st.session_state.get("page") not in [x[0] for x in PAGES]:
    st.session_state.page = "home"

st.markdown('<div class="nav-wrap"></div>', unsafe_allow_html=True)

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
                    BloomDetect AI turns satellite-derived chlorophyll-a
                    observations and the project screening output into a
                    practical view of where unusual patterns may deserve
                    closer investigation.
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

    risk_share = (
        100 * len(latest_risk) / len(latest)
        if len(latest)
        else 0
    )

    st.markdown('<div class="metrics"></div>', unsafe_allow_html=True)

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
            safe_num(latest["chla"].max()),
            "latest field",
        ),
    ]

    for col, item in zip(metric_cols, home_metrics):
        with col:
            metric(*item)

    page_heading(
        "PROJECT SNAPSHOT",
        "One clear view of the signal.",
        "The home page introduces the project and points to the four analysis tools. Detailed investigation stays inside the relevant page instead of being repeated everywhere.",
    )

    left, right = st.columns(
        [1.0, 1.0],
        gap="large"
    )

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="mini-label">WHAT BLOOMDETECT ADDS</div>
                <h3>From satellite observation to investigation support.</h3>
                <p>
                    The platform combines the latest satellite-derived
                    chlorophyll-a field with the project screening result,
                    then exposes that signal through a temporal map,
                    coordinate lookup and compact spatial insights.
                </p>

                <div class="note">
                    <b>Scientific note:</b>
                    a potential bloom-risk flag is not confirmation of a
                    harmful algal bloom. Species, toxin presence and
                    ecological impact require additional evidence and
                    field validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if IMAGE_PATH.exists():
            # st.image is intentionally used directly.
            # This keeps Streamlit's normal image hover/full-view controls.
            st.image(
                str(IMAGE_PATH),
                width="stretch",
                caption=(
                    "How satellite ocean-colour observations can support "
                    "bloom-risk investigation"
                ),
            )
        else:
            st.markdown(
                """
                <div class="card">
                    <h3>EOS-06 ocean-colour observation</h3>
                    <p>
                        Add bloomdetect_bloom_process.png beside app.py
                        to display the project illustration here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="tools-heading">
            <div class="kicker">EXPLORE THE PLATFORM</div>
            <div class="section" style="padding:0">
                <h2>Four focused tools.</h2>
                <p>
                    Each page has one clear purpose, so the same information
                    is not repeated across the dashboard.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tools = [
        (
            "🗺️",
            "Risk Map",
            "Move through available dates and compare the spatial screening field.",
            "map",
            "Open Risk Map",
        ),
        (
            "📍",
            "Location",
            "Check a coordinate against the nearest processed satellite cell.",
            "location",
            "Open Location",
        ),
        (
            "📊",
            "Insights",
            "Review hotspot concentration, regional patterns and current Chl-a signals.",
            "insights",
            "Open Insights",
        ),
        (
            "⇩",
            "Data",
            "Download the latest observations and potential-risk shortlist.",
            "data",
            "Open Data",
        ),
    ]

    tool_cols = st.columns(4, gap="medium")

    for col, (icon, title, desc, target, button_text) in zip(
        tool_cols, tools
    ):
        with col:
            st.markdown(
                f"""
                <div class="tool-item">
                    <div class="tool-card">
                        <div class="icon">{icon}</div>
                        <h3>{title}</h3>
                        <p>{desc}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="tool-button"></div>',
                unsafe_allow_html=True,
            )

            if st.button(
                button_text,
                key=f"home_tool_{target}",
                width="stretch",
            ):
                navigate(target)


# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":

    page_heading(
        "01 · SPATIAL INTELLIGENCE",
        "See how the field changes.",
        "Use the date slider to move through the available processed observations. The map keeps the same Indian Ocean study window so spatial changes are easy to compare.",
    )

    dates = history_dates()

    if len(dates) > 1:

        if (
            "map_date" not in st.session_state
            or pd.Timestamp(st.session_state.map_date) not in dates
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

        st.markdown(
            f"""
            <div class="timeline-card">
                <div class="timeline-top">
                    <div>
                        <div class="timeline-title">MAP TIMELINE</div>
                        <div class="timeline-help">
                            Drag the slider to compare available satellite fields.
                        </div>
                    </div>
                    <div class="timeline-date">
                        {selected_date.strftime("%d %b %Y")}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        selected_date = latest_date

        st.markdown(
            """
            <div class="note">
                <b>Date history is not available in the deployed app.</b>
                The latest processed field is shown below. Add the compact
                history file beside app.py to activate the date slider.
            </div>
            """,
            unsafe_allow_html=True,
        )

    fig, processed_count, risk_count, actual_date = map_figure(
        selected_date
    )

    share = (
        100 * risk_count / processed_count
        if processed_count
        else 0
    )

    mcols = st.columns(4, gap="medium")

    map_metrics = [
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
            "selected field",
        ),
        (
            "Risk share",
            f"{share:.2f}%",
            "selected field",
        ),
    ]

    for col, item in zip(mcols, map_metrics):
        with col:
            metric(*item)

    st.markdown(
        """
        <div class="map-shell">
            <div class="map-title">
                <b>Indian Ocean · Arabian Sea · Bay of Bengal</b>
                <span>
                    Blue field · green concentration · red screening flags
                </span>
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
                <span>Green = flagged-cell concentration</span>

                <span class="legend red"></span>
                <span>Red = individual potential-risk cell</span>
            </div>
        </div>

        <div class="note">
            <b>Reading the map:</b>
            green areas indicate spatial concentration of screening flags.
            They are not a separate severity score. Red cells are
            potential-risk screening results, not confirmed harmful algal blooms.
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
        "Check one coordinate across the available evidence.",
        "Your input is shown separately from the nearest processed 0.25° satellite cell, so there is no confusion about which location was actually evaluated.",
    )

    if "checked_lat" not in st.session_state:
        st.session_state.checked_lat = 18.0

    if "checked_lon" not in st.session_state:
        st.session_state.checked_lon = 78.0

    left, right = st.columns(
        [0.82, 1.18],
        gap="large"
    )

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="mini-label">YOUR INPUT</div>
                <h3>Coordinates</h3>
                <p>
                    Enter decimal degrees. Example: 17.38, 78.49.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form(
            "location_form",
            clear_on_submit=False
        ):
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(
                    st.session_state.checked_lat
                ),
                step=0.25,
                format="%.4f",
            )

            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(
                    st.session_state.checked_lon
                ),
                step=0.25,
                format="%.4f",
            )

            submitted = st.form_submit_button(
                "🔎 Check location",
                width="stretch",
                type="primary",
            )

        if submitted:
            st.session_state.checked_lat = float(lat)
            st.session_state.checked_lon = float(lon)
            st.rerun()

    lat = float(st.session_state.checked_lat)
    lon = float(st.session_state.checked_lon)

    row = nearest_latest(lat, lon)
    flagged = bool(row["risk_flag"])

    with right:
        st.markdown(
            f"""
            <div class="card">
                <div class="mini-label">
                    NEAREST PROCESSED CELL
                </div>

                <h3 style="font-size:1.55rem">
                    {float(row.latitude):.4f}° ·
                    {float(row.longitude):.4f}°
                </h3>

                <p>
                    This is the satellite grid cell used for the lookup.
                    <b>Your input:</b>
                    {lat:.4f}° · {lon:.4f}°
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
                        <b>
                            {probability_text(
                                row.get(
                                    "risk_probability",
                                    np.nan
                                )
                            )}
                        </b>
                    </div>

                    <div class="result-item">
                        <span>Region</span>
                        <b>
                            {region_name(
                                float(row.latitude),
                                float(row.longitude)
                            )}
                        </b>
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
                    <div class="status-title">
                        🔴 POTENTIAL BLOOM-RISK FLAG
                    </div>
                    <div>
                        This processed cell is included in the current
                        screening shortlist.
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
                        This processed cell is not included in the current
                        potential-risk shortlist.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="mini-label" style="margin-top:22px">SUPPORTING SIGNALS</div>',
        unsafe_allow_html=True,
    )

    signals = [
        (
            "Current Chl-a",
            safe_num(row.chla)
        ),
        (
            "Historical baseline",
            safe_num(
                row.get(
                    "historical_baseline",
                    np.nan
                )
            ),
        ),
        (
            "Anomaly",
            safe_num(
                row.get(
                    "chla_anomaly",
                    np.nan
                )
            ),
        ),
        (
            "Recent change",
            safe_num(
                row.get(
                    "chla_change",
                    np.nan
                )
            ),
        ),
    ]

    scols = st.columns(4, gap="small")

    for col, (label, value) in zip(
        scols, signals
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

    report = pd.DataFrame(
        [{
            "input_latitude": lat,
            "input_longitude": lon,
            "nearest_processed_latitude": row.latitude,
            "nearest_processed_longitude": row.longitude,
            "date": row.date.strftime("%Y-%m-%d"),
            "chla": row.chla,
            "historical_baseline": row.get(
                "historical_baseline", np.nan
            ),
            "chla_anomaly": row.get(
                "chla_anomaly", np.nan
            ),
            "chla_change": row.get(
                "chla_change", np.nan
            ),
            "risk_label": row.risk_label,
            "risk_probability": row.get(
                "risk_probability", np.nan
            ),
        }]
    )

    st.download_button(
        "⬇ Download location report",
        report.to_csv(index=False).encode(),
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
        "What stands out in the latest field?",
        "Hotspot concentration and data insights are combined here because they answer the same question: where should the current satellite signal receive closer investigation?",
    )

    total_flags = int(latest["risk_flag"].sum())

    share = (
        100 * total_flags / len(latest)
        if len(latest)
        else 0
    )

    icols = st.columns(4, gap="medium")

    insight_metrics = [
        (
            "Observation cells",
            f"{len(latest):,}",
            "latest field",
        ),
        (
            "Potential-risk",
            f"{total_flags:,}",
            "screening output",
        ),
        (
            "Risk share",
            f"{share:.2f}%",
            "latest field",
        ),
        (
            "Mean Chl-a",
            safe_num(latest["chla"].mean()),
            "latest field",
        ),
    ]

    for col, item in zip(icols, insight_metrics):
        with col:
            metric(*item)

    zones = hotspot_table(latest)

    page_heading(
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
                    The latest processed field contains no screening flags
                    inside the selected study window.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        hz = latest[latest["risk_flag"]].copy()

        hz["lat_zone"] = (
            np.floor(hz["latitude"] / 2) * 2 + 1
        )
        hz["lon_zone"] = (
            np.floor(hz["longitude"] / 2) * 2 + 1
        )

        plot_zones = (
            hz.groupby(
                ["lat_zone", "lon_zone"],
                as_index=False
            )
            .size()
            .rename(columns={"size": "flagged_cells"})
            .sort_values(
                "flagged_cells",
                ascending=False
            )
            .head(12)
        )

        plot_zones["Zone"] = plot_zones.apply(
            lambda x:
                f"{x.lat_zone:.0f}°–{x.lat_zone + 2:.0f}°, "
                f"{x.lon_zone:.0f}°–{x.lon_zone + 2:.0f}°",
            axis=1,
        )

        hfig = px.bar(
            plot_zones.sort_values(
                "flagged_cells",
                ascending=True
            ),
            x="flagged_cells",
            y="Zone",
            orientation="h",
            text="flagged_cells",
        )

        hfig.update_traces(
            marker_color="#20a477",
            textposition="outside",
            cliponaxis=False,
        )

        hfig.update_layout(
            height=430,
            margin=dict(
                l=150, r=55, t=25, b=70
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(
                color="#174b56",
                size=12
            ),
            xaxis=dict(
                title="Potential-risk cells in zone",
                title_font=dict(
                    color="#174b56",
                    size=14
                ),
                tickfont=dict(
                    color="#174b56",
                    size=11
                ),
                gridcolor="#d6e7e8",
            ),
            yaxis=dict(
                title="Investigation zone",
                title_font=dict(
                    color="#174b56",
                    size=14
                ),
                tickfont=dict(
                    color="#174b56",
                    size=11
                ),
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="chart-title">Top investigation zones</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            hfig,
            width="stretch",
            config={"displaylogo": False},
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="mini-label" style="margin-top:18px">ZONE DETAILS</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="hotspot-table">{zones.to_html(index=False, border=0)}</div>',
            unsafe_allow_html=True,
        )

    page_heading(
        "CURRENT FIELD",
        "What does the latest observation look like?",
        "These charts describe the latest processed field without repeating the spatial map.",
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        clean = (
            latest["chla"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        f1 = px.histogram(
            pd.DataFrame(
                {"Chlorophyll-a": clean}
            ),
            x="Chlorophyll-a",
            nbins=32,
        )

        f1.update_traces(
            marker_color="#197da5"
        )

        f1.update_layout(
            height=390,
            margin=dict(
                l=70, r=25, t=20, b=75
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(
                color="#174b56",
                size=12
            ),
            xaxis=dict(
                title="Satellite-derived chlorophyll-a",
                title_font=dict(
                    color="#174b56",
                    size=14
                ),
                tickfont=dict(
                    color="#174b56",
                    size=11
                ),
                gridcolor="#d6e7e8",
            ),
            yaxis=dict(
                title="Number of processed cells",
                title_font=dict(
                    color="#174b56",
                    size=14
                ),
                tickfont=dict(
                    color="#174b56",
                    size=11
                ),
                gridcolor="#d6e7e8",
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="chart-title">Chlorophyll-a distribution</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            f1,
            width="stretch",
            config={"displaylogo": False},
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        comp = pd.DataFrame(
            {
                "Screening group": [
                    "Normal",
                    "Potential bloom risk",
                ],
                "Mean chlorophyll-a": [
                    latest.loc[
                        ~latest.risk_flag,
                        "chla"
                    ].mean(),
                    latest.loc[
                        latest.risk_flag,
                        "chla"
                    ].mean(),
                ],
            }
        )

        f2 = px.bar(
            comp,
            x="Screening group",
            y="Mean chlorophyll-a",
            text="Mean chlorophyll-a",
        )

        f2.update_traces(
            marker_color=[
                "#197da5",
                "#ef4f5e"
            ],
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False,
        )

        f2.update_layout(
            height=390,
            margin=dict(
                l=70, r=25, t=20, b=75
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(
                color="#174b56",
                size=12
            ),
            xaxis=dict(
                title="Screening group",
                title_font=dict(
                    color="#174b56",
                    size=14
                ),
                tickfont=dict(
                    color="#174b56",
                    size=11
                ),
            ),
            yaxis=dict(
                title="Mean chlorophyll-a",
                title_font=dict(
                    color="#174b56",
                    size=14
                ),
                tickfont=dict(
                    color="#174b56",
                    size=11
                ),
                gridcolor="#d6e7e8",
            ),
        )

        st.markdown(
            '<div class="chart-card"><div class="chart-title">Mean chlorophyll-a by screening group</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            f2,
            width="stretch",
            config={"displaylogo": False},
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    # Regional summary
    regional_defs = [
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

    regional_rows = []

    for name, cond in regional_defs:
        sub = latest[cond]

        if len(sub):
            regional_rows.append(
                {
                    "Region": name,
                    "Cells": len(sub),
                    "Potential-risk cells": int(
                        sub["risk_flag"].sum()
                    ),
                    "Risk share": 100 * sub["risk_flag"].mean(),
                    "Mean Chl-a": sub["chla"].mean(),
                }
            )

    regional = pd.DataFrame(regional_rows)

    if not regional.empty:
        regional["Risk share"] = regional[
            "Risk share"
        ].map(lambda x: f"{x:.2f}%")

        regional["Mean Chl-a"] = regional[
            "Mean Chl-a"
        ].round(4)

        page_heading(
            "REGIONAL SIGNAL",
            "Broad-area comparison.",
            "These are descriptive regional summaries of the latest processed field.",
        )

        st.dataframe(
            regional,
            hide_index=True,
            width="stretch",
        )

    st.markdown(
        """
        <div class="note">
            <b>Interpretation:</b>
            the charts describe satellite-derived chlorophyll-a and the
            project screening output. They do not independently establish
            harmfulness, species identity or toxin presence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not zones.empty:
        st.download_button(
            "⬇ Download hotspot report",
            zones.to_csv(index=False).encode(),
            "bloomdetect_hotspot_report.csv",
            "text/csv",
            width="stretch",
        )


# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    page_heading(
        "04 · DATA & OUTPUTS",
        "The evidence behind the dashboard.",
        "Keep the source product, current processed field and downloadable outputs in one place. No duplicate analysis panels here.",
    )

    dcols = st.columns(4, gap="medium")

    data_metrics = [
        (
            "Product",
            "E06OCM_L4_AC",
            "EOS-06 / OCM-3",
        ),
        (
            "Grid",
            "0.25°",
            "latitude × longitude",
        ),
        (
            "Latest cells",
            f"{len(latest):,}",
            "processed observation",
        ),
        (
            "Latest date",
            latest_date.strftime("%d %b %Y"),
            "processed dataset",
        ),
    ]

    for col, item in zip(
        dcols, data_metrics
    ):
        with col:
            metric(*item)

    st.markdown(
        """
        <div class="card" style="margin-top:18px">
            <div class="mini-label">SOURCE PRODUCT</div>
            <h3>EOS-06 OCM-3 analysed chlorophyll-a</h3>
            <p>
                The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4
                analysed chlorophyll product, E06OCM_L4_AC. The current
                project output is a potential bloom-risk screening layer
                built from satellite-derived chlorophyll-a and temporal context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_heading(
        "DOWNLOADS",
        "Take the actual project outputs.",
        "These downloads are generated directly from the data used by the dashboard.",
    )

    dl1, dl2 = st.columns(
        2,
        gap="large"
    )

    with dl1:
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
            latest.to_csv(index=False).encode(),
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
                    Only cells currently screened as potential bloom risk
                    in the latest processed field.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download potential-risk locations",
            latest_risk.to_csv(index=False).encode(),
            "bloomdetect_potential_risk.csv",
            "text/csv",
            width="stretch",
        )

    page_heading(
        "SCIENTIFIC USE",
        "Use the screening result correctly.",
        "",
    )

    st.markdown(
        """
        <div class="card">
            <p>
                <b>Potential bloom risk</b> is a project screening output.
                It is not confirmation of a harmful algal bloom, species
                identity or toxin presence. Satellite observations should
                be combined with field observations and additional
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
