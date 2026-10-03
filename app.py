
from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Practical Coastal & Ocean Intelligence Dashboard
# Dataset: EOS-06 OCM-3 analysed chlorophyll-a
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

# These are the project proxy-screening thresholds.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv must be in the same folder as app.py"
        )

    d = pd.read_csv(DATA_PATH)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(sorted(missing))
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

study = latest[
    latest["latitude"].between(LAT_MIN, LAT_MAX)
    & latest["plot_lon"].between(LON_MIN, LON_MAX)
].copy()

risk = study[study["risk_flag"]].copy()


# ============================================================
# PAGE STATE
# ============================================================

PAGES = ["home", "map", "location", "hotspots", "insights", "data"]

NAV = {
    "home": "⌂ Home",
    "map": "🗺 Risk Map",
    "location": "📍 Location",
    "hotspots": "🔥 Hotspots",
    "insights": "📊 Insights",
    "data": "⇩ Data",
}

if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"


# ============================================================
# CSS
# ============================================================

st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
    --ink:#103f49;
    --deep:#07566a;
    --aqua:#0aa8b5;
    --muted:#64838a;
    --line:#a8d7db;
    --pale:#f1fbfb;
    --blue:#277fa4;
    --green:#22a879;
    --red:#ed5260;
    --shadow:0 12px 34px rgba(9,80,94,.08);
}

html,body,[data-testid="stAppViewContainer"]{
    background:#f2fbfb !important;
    color:var(--ink) !important;
    font-family:'DM Sans',sans-serif !important;
}

.stApp{
    background:
        radial-gradient(circle at 8% 10%,rgba(35,201,205,.08),transparent 25%),
        radial-gradient(circle at 92% 68%,rgba(15,156,175,.06),transparent 28%),
        linear-gradient(180deg,#fbffff 0%,#effafa 55%,#fbffff 100%) !important;
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
    max-width:1240px !important;
    padding:22px 30px 65px !important;
}

/* moving water effect */
.stApp:before{
    content:"";
    position:fixed;
    left:-10%;
    right:-10%;
    bottom:-170px;
    height:300px;
    pointer-events:none;
    z-index:0;
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

/* header */
.brand-bar{
    position:relative;
    z-index:10;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:18px;
    padding:14px 18px;
    border:1.5px solid #c3e2e5;
    background:rgba(255,255,255,.92);
    border-radius:21px;
    box-shadow:var(--shadow);
    backdrop-filter:blur(15px);
}

.brand-left{
    display:flex;
    align-items:center;
    gap:12px;
}

.logo{
    width:47px;
    height:47px;
    border-radius:15px;
    background:linear-gradient(145deg,#28cbd0,#08798b);
    display:grid;
    place-items:center;
    color:white;
    font-size:20px;
    font-weight:800;
}

.brand-name{
    font:800 1.2rem Manrope,sans-serif;
    color:var(--ink);
    letter-spacing:-.03em;
}

.brand-sub{
    font-size:.71rem;
    color:#78959b;
    margin-top:2px;
}

.latest-label{
    text-align:right;
    color:#78959b;
    font-size:.66rem;
    line-height:1.35;
}

.latest-label b{
    color:#174b56;
    font-size:.78rem;
}

/* nav */
.nav-wrap{
    position:relative;
    z-index:12;
    margin:12px 0 5px;
}

.stButton>button,
.stDownloadButton>button{
    min-height:44px !important;
    border-radius:14px !important;
    background:#ffffff !important;
    border:2px solid #164e5b !important;
    color:#123f49 !important;
    font-weight:800 !important;
    font-size:.84rem !important;
    box-shadow:0 6px 15px rgba(15,70,82,.08) !important;
    transition:.18s ease !important;
}

.stButton>button:hover,
.stDownloadButton>button:hover{
    background:#e2f8f8 !important;
    border-color:#087d8c !important;
    transform:translateY(-1px) !important;
}

.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#d8f7f7,#b8eeee) !important;
    border-color:#078c9b !important;
    color:#0a5966 !important;
}

/* typography */
.section{
    position:relative;
    z-index:2;
    padding:32px 0 16px;
}

.kicker{
    font:800 .66rem Manrope,sans-serif;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:10px;
}

.section h2{
    font:800 clamp(2.1rem,4vw,3.55rem)/1.03 Manrope,sans-serif;
    letter-spacing:-.06em;
    color:var(--ink);
    margin:0 0 12px;
}

.section p{
    color:#5f8088;
    line-height:1.62;
    margin:0;
    max-width:1050px;
    font-size:.95rem;
}

/* hero */
.hero{
    position:relative;
    z-index:2;
    overflow:hidden;
    min-height:390px;
    border-radius:29px;
    padding:55px 54px;
    background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);
    box-shadow:0 25px 65px rgba(6,86,100,.15);
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
    max-width:790px;
}

.hero .kicker{
    color:#a6fffa;
}

.hero h1{
    font:800 clamp(3rem,6vw,5.5rem)/.92 Manrope,sans-serif;
    letter-spacing:-.075em;
    color:#e4ffff;
    margin:0 0 18px;
}

.hero p{
    font-size:1.02rem;
    line-height:1.75;
    color:#e0fbfb;
    max-width:730px;
}

.hero-badges{
    display:flex;
    gap:8px;
    flex-wrap:wrap;
    margin-top:21px;
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

@keyframes waveLines{
    from{transform:translateX(-4%) rotate(-7deg)}
    to{transform:translateX(4%) rotate(-7deg)}
}

@keyframes heroWave{
    from{transform:translateX(-2%) rotate(-1deg)}
    to{transform:translateX(2%) rotate(1deg)}
}

/* cards */
.card{
    position:relative;
    z-index:2;
    background:rgba(255,255,255,.9);
    border:1.5px solid #a9d6da;
    border-radius:21px;
    box-shadow:var(--shadow);
    padding:21px;
}

.card h3{
    font:800 1.2rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 7px;
}

.card p{
    color:#66848b;
    line-height:1.6;
    margin:0;
    font-size:.9rem;
}

.metric{
    position:relative;
    z-index:2;
    min-height:108px;
    height:100%;
    box-sizing:border-box;
    padding:16px;
    background:linear-gradient(145deg,#ffffff,#eaf8f8);
    border:1.5px solid #a6d5da;
    border-radius:18px;
    box-shadow:0 10px 27px rgba(15,91,101,.07);
}

.metric .label{
    font:800 .62rem Manrope,sans-serif;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:#6d8c93;
}

.metric .value{
    font:800 1.42rem Manrope,sans-serif;
    color:#123f49;
    margin-top:5px;
    white-space:nowrap;
}

.metric .note{
    font-size:.7rem;
    color:#78959b;
    margin-top:4px;
}

/* image */
.image-card{
    position:relative;
    z-index:2;
    padding:8px;
    border-radius:22px;
    background:#fff;
    border:1.5px solid #acd9dd;
    box-shadow:var(--shadow);
}

.image-card img{
    display:block;
    width:100%;
    height:285px;
    object-fit:cover;
    border-radius:16px;
}

/* feature cards */
.feature{
    position:relative;
    z-index:2;
    padding:20px;
    background:rgba(255,255,255,.86);
    border:1.5px solid #b5dde0;
    border-radius:19px;
    box-shadow:var(--shadow);
    min-height:160px;
}

.feature .icon{
    font-size:1.3rem;
    margin-bottom:8px;
}

.feature h3{
    font:800 1.03rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 7px;
}

.feature p{
    font-size:.85rem;
    line-height:1.55;
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
    min-height:42px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:15px;
    padding:0 12px;
    color:#174b56;
}

.map-title b{font-size:.9rem}
.map-title span{font-size:.68rem;color:#6b8b92}

.map-legend{
    display:flex;
    align-items:center;
    gap:8px;
    flex-wrap:wrap;
    color:#4f747b;
    font-size:.77rem;
    padding:10px 13px;
}

.legend{
    width:15px;
    height:10px;
    border-radius:4px;
    display:inline-block;
}

.legend-blue{background:#277fa4}
.legend-green{background:#22a879}
.legend-red{background:#ed5260;border-radius:50%;width:11px;height:11px}

/* location */
.location-card{
    position:relative;
    z-index:2;
    background:rgba(255,255,255,.9);
    border:1.5px solid #a9d6da;
    border-radius:21px;
    box-shadow:var(--shadow);
    padding:22px;
}

.location-title{
    font:800 1.45rem Manrope,sans-serif;
    color:#123f49;
    margin-bottom:5px;
}

.small-copy{
    color:#66848b;
    font-size:.84rem;
    line-height:1.55;
}

.status{
    margin-top:15px;
    padding:14px 15px;
    border-radius:15px;
    border:2px solid;
}

.status.risk{
    background:#fff0f2;
    border-color:#ed6976;
    color:#9b2d3c;
}

.status.normal{
    background:#eafaf4;
    border-color:#49b995;
    color:#176f58;
}

.status-title{
    font:800 .9rem Manrope;
}

.result-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:9px;
    margin-top:14px;
}

.result-item{
    padding:11px 12px;
    background:#f6fcfc;
    border:1px solid #c8e5e7;
    border-radius:12px;
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
}

.signal-grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:10px;
    margin-top:15px;
}

.signal{
    padding:12px;
    border-radius:13px;
    background:#f7fcfc;
    border:1px solid #c9e5e7;
}

.signal .label{
    font-size:.66rem;
    color:#78959b;
}

.signal .value{
    font:800 .9rem Manrope;
    color:#164a55;
    margin-top:4px;
}

/* hotspot table */
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

.hotspot-table tr:nth-child(even) td{
    background:#f7fcfc;
}

/* chart cards */
.chart-card{
    position:relative;
    z-index:2;
    background:#fff;
    border:1.5px solid #b6dfe2;
    border-radius:20px;
    box-shadow:var(--shadow);
    padding:13px;
}

.chart-title{
    font:800 1rem Manrope,sans-serif;
    color:#164b56;
    padding:5px 6px 8px;
}

/* downloads */
.download-card{
    min-height:132px;
    box-sizing:border-box;
    padding:18px;
    border-radius:18px;
    background:linear-gradient(135deg,#087b8b,#13adb3);
    border:2px solid #07576a;
    box-shadow:0 13px 28px rgba(8,91,102,.14);
    color:white;
}

.download-card h3{
    font:800 1.06rem Manrope,sans-serif;
    color:white;
    margin:0 0 5px;
}

.download-card p{
    font-size:.8rem;
    color:#e5ffff;
    line-height:1.45;
    margin:0;
}

/* notes */
.note{
    position:relative;
    z-index:2;
    margin-top:15px;
    padding:12px 15px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 13px 13px 0;
    color:#52757c;
    font-size:.83rem;
    line-height:1.55;
}

.footer{
    position:relative;
    z-index:2;
    border-top:1px solid #d5ebed;
    margin-top:38px;
    padding-top:15px;
    color:#76959b;
    font-size:.68rem;
}

/* cleaner Streamlit inputs */
div[data-testid="stNumberInput"]{
    margin-bottom:7px;
}

@media(max-width:900px){
    .block-container{padding:18px 17px 55px !important}
    .hero{padding:43px 34px}
    .signal-grid{grid-template-columns:1fr 1fr}
}

@media(max-width:620px){
    .hero{padding:37px 24px}
    .hero h1{font-size:3rem}
    .block-container{padding:13px 11px 45px !important}
    .latest-label{display:none}
    .signal-grid{grid-template-columns:1fr}
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


def nearest_row(lat, lon):
    # Longitude is adjusted by latitude so that the simple distance
    # behaves more reasonably across the study region.
    a = study
    if a.empty:
        return None

    lat_scale = 1.0
    lon_scale = max(np.cos(np.radians(lat)), 0.25)

    dist = (
        ((a["latitude"].to_numpy() - lat) / lat_scale) ** 2
        + ((a["longitude"].to_numpy() - lon) * lon_scale) ** 2
    )

    return a.iloc[int(np.argmin(dist))]


def image_data_uri(path):
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        return "data:image/png;base64," + base64.b64encode(raw).decode("utf-8")
    except Exception:
        return None


def make_risk_map(data):
    """
    Safe spatial map using Scattergeo.
    No update_geos() is used because that was the source of the
    Streamlit Cloud Plotly ValueError in the previous version.
    """
    if data.empty:
        return None

    blue = data.copy()
    blue["lat_bin"] = np.floor(blue["latitude"]).astype(float) + 0.5
    blue["lon_bin"] = np.floor(blue["plot_lon"]).astype(float) + 0.5

    blue = (
        blue.groupby(["lat_bin", "lon_bin"], as_index=False)
        .agg(mean_chla=("chla", "mean"), cells=("chla", "size"))
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=blue["lat_bin"],
            lon=blue["lon_bin"],
            mode="markers",
            name="Normal processed field",
            marker=dict(
                size=7,
                color="#277fa4",
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
                "<b>Normal processed field</b><br>"
                "Latitude: %{customdata[0]:.1f}°<br>"
                "Longitude: %{customdata[1]:.1f}°<br>"
                "Mean Chl-a: %{customdata[2]:.4f}<br>"
                "Cells: %{customdata[3]}<extra></extra>"
            ),
        )
    )

    flagged = data[data["risk_flag"]].copy()

    if not flagged.empty:
        flagged["lat_zone"] = np.floor(flagged["latitude"] / 2) * 2 + 1
        flagged["lon_zone"] = np.floor(flagged["plot_lon"] / 2) * 2 + 1

        zones = (
            flagged.groupby(["lat_zone", "lon_zone"], as_index=False)
            .agg(
                flagged_cells=("risk_flag", "size"),
                mean_chla=("chla", "mean"),
            )
        )

        zone_sizes = np.clip(zones["flagged_cells"].to_numpy() * 1.4 + 9, 12, 42)

        fig.add_trace(
            go.Scattergeo(
                lat=zones["lat_zone"],
                lon=zones["lon_zone"],
                mode="markers",
                name="Flag concentration zone",
                marker=dict(
                    size=zone_sizes,
                    color="#22a879",
                    opacity=0.42,
                    line=dict(color="#ffffff", width=1.5),
                ),
                customdata=np.column_stack(
                    [
                        zones["lat_zone"],
                        zones["lon_zone"],
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
                lon=flagged["plot_lon"],
                mode="markers",
                name="Potential bloom-risk cell",
                marker=dict(
                    size=5.5,
                    color="#ed5260",
                    opacity=0.92,
                    line=dict(color="#ffffff", width=0.5),
                ),
                customdata=np.column_stack(
                    [
                        flagged["latitude"],
                        flagged["plot_lon"],
                        flagged["chla"],
                    ]
                ),
                hovertemplate=(
                    "<b>Potential bloom-risk cell</b><br>"
                    "Latitude: %{customdata[0]:.3f}°<br>"
                    "Longitude: %{customdata[1]:.3f}°<br>"
                    "Chl-a: %{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="#dff7f8",
        plot_bgcolor="#dff7f8",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,.86)",
            bordercolor="#9bcbd0",
            borderwidth=1,
            font=dict(size=11),
        ),
        geo=dict(
            showland=True,
            landcolor="#d6e5e3",
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
                gridcolor="rgba(90,140,150,.20)",
            ),
            lonaxis=dict(
                range=[LON_MIN, LON_MAX],
                showgrid=True,
                gridcolor="rgba(90,140,150,.20)",
            ),
            bgcolor="#dff7f8",
        ),
    )

    return fig


def hotspot_table(flagged):
    if flagged.empty:
        return pd.DataFrame(
            columns=["Zone centre", "Flagged cells", "Mean Chl-a", "Maximum Chl-a"]
        )

    x = flagged.copy()
    x["lat_zone"] = np.floor(x["latitude"] / 2) * 2 + 1
    x["lon_zone"] = np.floor(x["plot_lon"] / 2) * 2 + 1

    out = (
        x.groupby(["lat_zone", "lon_zone"], as_index=False)
        .agg(
            flagged_cells=("risk_flag", "size"),
            mean_chla=("chla", "mean"),
            max_chla=("chla", "max"),
        )
        .sort_values("flagged_cells", ascending=False)
        .head(12)
    )

    out["Zone centre"] = out.apply(
        lambda r: f"{r['lat_zone']:.1f}°, {r['lon_zone']:.1f}°", axis=1
    )

    return out[
        ["Zone centre", "flagged_cells", "mean_chla", "max_chla"]
    ].rename(
        columns={
            "flagged_cells": "Flagged cells",
            "mean_chla": "Mean Chl-a",
            "max_chla": "Maximum Chl-a",
        }
    )


# ============================================================
# HEADER
# ============================================================

latest_display = latest_date.strftime("%d %b %Y")

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
            <b>{latest_display}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="nav-wrap"></div>', unsafe_allow_html=True)

nav_cols = st.columns(len(NAV), gap="small")

for col, (page_key, label) in zip(nav_cols, NAV.items()):
    with col:
        if st.button(
            label,
            key=f"nav_{page_key}",
            type="primary" if st.session_state.page == page_key else "secondary",
            use_container_width=True,
        ):
            st.session_state.page = page_key
            st.rerun()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-content">
                <div class="kicker">Satellite ocean-colour intelligence</div>
                <h1>Read the ocean signal before it becomes a problem.</h1>
                <p>
                    BloomDetect AI turns EOS-06 OCM-3 chlorophyll-a observations
                    into a practical spatial screening view for potential bloom-risk
                    investigation. Explore the latest field, locate individual cells,
                    inspect concentrations and study the broader signal.
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

    flagged_count = len(risk)
    processed_count = len(study)
    risk_share = (flagged_count / processed_count * 100) if processed_count else 0
    max_chla = study["chla"].max() if not study.empty else np.nan

    st.markdown('<div class="section"><div class="kicker">LATEST FIELD</div><h2>What is happening in the current scene?</h2><p>The four numbers below summarize the latest processed study field. Detailed analysis stays on its own page, so the home screen does not become a museum of repeated statistics.</p></div>', unsafe_allow_html=True)

    mcols = st.columns(4, gap="medium")
    with mcols[0]:
        metric("Processed cells", f"{processed_count:,}", "latest study field")
    with mcols[1]:
        metric("Potential-risk cells", f"{flagged_count:,}", "screening output")
    with mcols[2]:
        metric("Risk share", f"{risk_share:.2f}%", "of processed cells")
    with mcols[3]:
        metric("Maximum Chl-a", safe_num(max_chla), "latest field value")

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="kicker">WHY IT MATTERS</div>
                <h3>Chlorophyll-a is a signal, not a verdict.</h3>
                <p>
                    The platform uses changes and persistence in satellite-derived
                    chlorophyll-a to screen locations that deserve closer attention.
                    A potential-risk flag is an investigation aid, not confirmation
                    of a harmful algal bloom, species or toxin.
                </p>
                <div class="note">
                    <b>Scientific caution:</b> high chlorophyll-a alone does not
                    prove a harmful algal bloom. Field observations and additional
                    environmental evidence are needed for confirmation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        uri = image_data_uri(IMAGE_PATH)
        if uri:
            st.markdown(
                f"""
                <div class="image-card">
                    <img src="{uri}" alt="Bloom development and satellite observation">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="card">
                    <h3>Satellite-to-risk workflow</h3>
                    <p>
                        Add <b>bloomdetect_bloom_process.png</b> beside app.py
                        to show the project visual here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section"><div class="kicker">PLATFORM</div><h2>Useful views, without the clutter.</h2><p>Each section answers a different operational question. No duplicate Explore page, no separate How It Works page, and no tiny buttons pretending to be features.</p></div>', unsafe_allow_html=True)

    features = [
        ("🗺️", "Risk Map", "See the full Indian Ocean study area with normal processed coverage, concentration zones and flagged cells."),
        ("📍", "Location", "Enter one coordinate and inspect the nearest processed satellite cell and screening result."),
        ("🔥", "Hotspots", "Find where flagged cells are concentrating and inspect the strongest investigation zones."),
        ("📊", "Insights", "Study the latest chlorophyll-a distribution and the time pattern available in the dataset."),
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

    st.markdown(
        """
        <div class="note">
            <b>Current data scope:</b> this version uses the EOS-06 OCM-3
            analysed chlorophyll-a product and the project's potential
            bloom-risk screening output. It does not invent temperature,
            salinity, pollution or fisheries measurements that are not present
            in the current dataset.
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
        "Where is the potential bloom-risk signal showing up?",
        "The map is focused on the Indian Ocean study area. Blue shows the processed field, green shows concentrations of flagged cells and red shows individual cells screened as potential bloom risk.",
    )

    flagged_count = len(risk)
    processed_count = len(study)
    risk_share = flagged_count / processed_count * 100 if processed_count else 0

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric("Study area", "20°–120°E", "40°S–30°N")
    with c2:
        metric("Flagged cells", f"{flagged_count:,}", "latest processed field")
    with c3:
        metric("Risk share", f"{risk_share:.2f}%", "of study-area cells")

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

    fig = make_risk_map(study)

    if fig is None:
        st.warning("No study-area observations are available for the map.")
    else:
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
            },
        )

    st.markdown(
        """
        <div class="map-legend">
            <span class="legend legend-blue"></span> Normal processed field
            &nbsp;&nbsp;
            <span class="legend legend-green"></span> Flag concentration zone
            &nbsp;&nbsp;
            <span class="legend legend-red"></span> Potential bloom-risk cell
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="note">
            Green areas are concentration summaries of flagged cells. They are
            not a second risk label. Red cells are the project's potential
            bloom-risk screening output.
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
        "Enter the location you want to investigate. The dashboard compares your input with the nearest processed 0.25° satellite grid cell.",
    )

    left, right = st.columns([0.82, 1.18], gap="large")

    with left:
        st.markdown(
            """
            <div class="location-card">
                <div class="kicker">YOUR INPUT</div>
                <div class="location-title">Coordinates</div>
                <div class="small-copy">
                    Use decimal degrees. Example: 17.38, 78.49.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=17.38,
            step=0.01,
            format="%.2f",
            key="lookup_lat",
        )

        lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.49,
            step=0.01,
            format="%.2f",
            key="lookup_lon",
        )

        if st.button(
            "🔎 Check location",
            key="check_location",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.location_result = (float(lat), float(lon))

    coords = st.session_state.get(
        "location_result",
        (17.38, 78.49),
    )

    input_lat, input_lon = coords
    nearest = nearest_row(input_lat, input_lon)

    with right:
        if nearest is None:
            st.error("No processed cell is available.")
        else:
            flagged = bool(nearest["risk_flag"])
            nearest_lat = float(nearest["latitude"])
            nearest_lon = float(nearest["longitude"])

            st.markdown(
                f"""
                <div class="location-card">
                    <div class="kicker">NEAREST PROCESSED CELL</div>
                    <div class="location-title">
                        {nearest_lat:.4f}° · {nearest_lon:.4f}°
                    </div>
                    <div class="small-copy">
                        This is the satellite grid cell actually used for the lookup.
                    </div>

                    <div class="result-grid">
                        <div class="result-item">
                            <span>Your input latitude</span>
                            <b>{input_lat:.4f}°</b>
                        </div>
                        <div class="result-item">
                            <span>Your input longitude</span>
                            <b>{input_lon:.4f}°</b>
                        </div>
                        <div class="result-item">
                            <span>Chlorophyll-a</span>
                            <b>{safe_num(nearest["chla"])}</b>
                        </div>
                        <div class="result-item">
                            <span>Risk probability</span>
                            <b>{probability_text(nearest.get("risk_probability", np.nan))}</b>
                        </div>
                        <div class="result-item">
                            <span>Observation date</span>
                            <b>{pd.Timestamp(nearest["date"]).strftime("%d %b %Y")}</b>
                        </div>
                        <div class="result-item">
                            <span>Region</span>
                            <b>{region_name(nearest_lat, nearest_lon)}</b>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            status_class = "risk" if flagged else "normal"
            status_title = (
                "🔴 POTENTIAL BLOOM RISK FLAGGED"
                if flagged
                else "🟢 NOT FLAGGED"
            )
            status_copy = (
                "This processed cell is included in the current potential-risk screening output."
                if flagged
                else "This processed cell is not included in the current potential-risk screening shortlist."
            )

            st.markdown(
                f"""
                <div class="status {status_class}">
                    <div class="status-title">{status_title}</div>
                    <div>{status_copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if nearest is not None:
        st.markdown(
            '<div class="kicker" style="margin-top:24px">SUPPORTING SIGNALS</div>',
            unsafe_allow_html=True,
        )

        signal_cols = st.columns(4, gap="small")

        signals = [
            ("Current Chl-a", nearest.get("chla", np.nan)),
            ("Historical baseline", nearest.get("historical_baseline", np.nan)),
            ("Anomaly", nearest.get("chla_anomaly", np.nan)),
            ("Recent change", nearest.get("chla_change", np.nan)),
        ]

        for col, (label, value) in zip(signal_cols, signals):
            with col:
                st.markdown(
                    f"""
                    <div class="signal">
                        <div class="label">{label}</div>
                        <div class="value">{safe_num(value)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        location_report = pd.DataFrame(
            [{
                "Input latitude": input_lat,
                "Input longitude": input_lon,
                "Nearest latitude": nearest_lat,
                "Nearest longitude": nearest_lon,
                "Observation date": nearest["date"].strftime("%Y-%m-%d"),
                "Chl-a": nearest["chla"],
                "Risk label": nearest["risk_label"],
                "Risk probability": nearest.get("risk_probability", np.nan),
                "Region": region_name(nearest_lat, nearest_lon),
            }]
        )

        st.download_button(
            "⇩ Download location report",
            data=location_report.to_csv(index=False).encode("utf-8"),
            file_name="bloomdetect_location_report.csv",
            mime="text/csv",
            key="download_location_report",
            use_container_width=True,
        )


# ============================================================
# HOTSPOTS
# ============================================================

elif st.session_state.page == "hotspots":

    section(
        "03 · HOTSPOT INTELLIGENCE",
        "Where are the screening flags concentrating?",
        "Flagged cells are grouped into broad 2° × 2° investigation zones. The zones make the spatial pattern easier to inspect without creating another risk label.",
    )

    flagged_count = len(risk)

    zones_df = hotspot_table(risk)

    top_count = int(zones_df["Flagged cells"].max()) if not zones_df.empty else 0
    top_zone = (
        str(zones_df.iloc[0]["Zone centre"])
        if not zones_df.empty
        else "No flagged zone"
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric("Flagged cells", f"{flagged_count:,}", "latest study area")
    with c2:
        metric("Investigation zones", f"{len(zones_df):,}", "top zones shown")
    with c3:
        metric("Largest zone", f"{top_count:,}", top_zone)

    if zones_df.empty:
        st.info("No potential-risk cells are available for hotspot analysis.")
    else:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Top concentration zones</div>',
            unsafe_allow_html=True,
        )

        chart_df = zones_df.head(12).iloc[::-1]

        fig = go.Figure(
            go.Bar(
                x=chart_df["Flagged cells"],
                y=chart_df["Zone centre"],
                orientation="h",
                marker_color="#22a879",
                text=chart_df["Flagged cells"],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Flagged cells: %{x}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            height=470,
            margin=dict(l=15, r=55, t=10, b=45),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis_title="Potential-risk cells in zone",
            yaxis_title="Approximate zone centre",
            font=dict(color="#315b63"),
            xaxis=dict(showgrid=True, gridcolor="#e3eeee"),
            yaxis=dict(showgrid=False),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div style="height:10px"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="hotspot-table">',
            unsafe_allow_html=True,
        )

        display_zones = zones_df.copy()
        display_zones["Mean Chl-a"] = display_zones["Mean Chl-a"].map(
            lambda x: f"{x:.4f}"
        )
        display_zones["Maximum Chl-a"] = display_zones["Maximum Chl-a"].map(
            lambda x: f"{x:.4f}"
        )

        st.markdown(
            display_zones.to_html(index=False, escape=True),
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "⇩ Download hotspot report",
            data=zones_df.to_csv(index=False).encode("utf-8"),
            file_name="bloomdetect_hotspot_report.csv",
            mime="text/csv",
            key="download_hotspots",
            use_container_width=True,
        )


# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":

    section(
        "04 · OCEAN-COLOUR INSIGHTS",
        "What does the data say beyond the map?",
        "These views use the actual chlorophyll-a observations and screening output. They focus on patterns that help investigation rather than repeating the map.",
    )

    mean_chla = study["chla"].mean() if not study.empty else np.nan
    median_chla = study["chla"].median() if not study.empty else np.nan
    max_chla = study["chla"].max() if not study.empty else np.nan
    mean_anomaly = (
        study["chla_anomaly"].mean()
        if "chla_anomaly" in study.columns
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric("Mean Chl-a", safe_num(mean_chla), "latest field")
    with c2:
        metric("Median Chl-a", safe_num(median_chla), "latest field")
    with c3:
        metric("Maximum Chl-a", safe_num(max_chla), "latest field")
    with c4:
        metric("Mean anomaly", safe_num(mean_anomaly), "latest field")

    # Latest distribution
    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Latest chlorophyll-a distribution</div>',
            unsafe_allow_html=True,
        )

        hist_data = study["chla"].replace([np.inf, -np.inf], np.nan).dropna()

        if not hist_data.empty:
            fig = go.Figure(
                go.Histogram(
                    x=hist_data,
                    nbinsx=35,
                    marker_color="#277fa4",
                    opacity=0.88,
                    hovertemplate="Chl-a: %{x:.4f}<br>Cells: %{y}<extra></extra>",
                )
            )
            fig.update_layout(
                height=370,
                margin=dict(l=45, r=20, t=10, b=50),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                xaxis_title="Chlorophyll-a",
                yaxis_title="Processed cells",
                xaxis=dict(showgrid=True, gridcolor="#e3eeee"),
                yaxis=dict(showgrid=True, gridcolor="#e3eeee"),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True},
            )
        else:
            st.info("No valid Chl-a values are available.")

        st.markdown("</div>", unsafe_allow_html=True)

    with chart_right:
        st.markdown(
            '<div class="chart-card"><div class="chart-title">Potential-risk cells by broad region</div>',
            unsafe_allow_html=True,
        )

        region_df = risk.copy()
        if not region_df.empty:
            region_df["Region"] = region_df.apply(
                lambda r: region_name(r["latitude"], r["longitude"]),
                axis=1,
            )
            region_counts = (
                region_df.groupby("Region")
                .size()
                .sort_values(ascending=True)
            )

            fig = go.Figure(
                go.Bar(
                    x=region_counts.values,
                    y=region_counts.index,
                    orientation="h",
                    marker_color="#ed5260",
                    text=region_counts.values,
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Potential-risk cells: %{x}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                height=370,
                margin=dict(l=20, r=55, t=10, b=50),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                xaxis_title="Potential-risk cells",
                yaxis_title="Broad region",
                xaxis=dict(showgrid=True, gridcolor="#e3eeee"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True},
            )
        else:
            st.info("No potential-risk cells are available.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Historical field trend
    st.markdown(
        '<div style="height:18px"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="chart-card"><div class="chart-title">Chlorophyll-a trend across the available processed dates</div>',
        unsafe_allow_html=True,
    )

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            mean_chla=("chla", "mean"),
            processed_cells=("chla", "size"),
        )
        .sort_values("date")
    )

    if len(daily) > 1:
        fig = go.Figure(
            go.Scatter(
                x=daily["date"],
                y=daily["mean_chla"],
                mode="lines",
                line=dict(color="#0aa8b5", width=2.5),
                hovertemplate=(
                    "Date: %{x|%d %b %Y}<br>"
                    "Mean Chl-a: %{y:.4f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            height=360,
            margin=dict(l=55, r=20, t=10, b=55),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            xaxis_title="Observation date",
            yaxis_title="Mean chlorophyll-a",
            xaxis=dict(showgrid=True, gridcolor="#e3eeee"),
            yaxis=dict(showgrid=True, gridcolor="#e3eeee"),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
    else:
        st.info("The current file does not contain enough dates for a trend chart.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note">
            <b>Interpretation:</b> these charts describe the satellite-derived
            chlorophyll-a signal and the project's screening output. They do not
            independently establish harmfulness, species identity or toxin
            presence.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    section(
        "05 · DATA & OUTPUTS",
        "Use the actual project data.",
        "This page keeps source information, dataset coverage and downloadable outputs together. No extra scientific claims are created from unavailable datasets.",
    )

    date_min = df["date"].min()
    date_max = df["date"].max()

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric("Total records", f"{len(df):,}", "current CSV")
    with c2:
        metric("Processed dates", f"{df['date'].nunique():,}", "available dates")
    with c3:
        metric("Grid locations", f"{df[['latitude','longitude']].drop_duplicates().shape[0]:,}", "unique cells")
    with c4:
        metric("Coverage", f"{date_min:%d %b %Y} → {date_max:%d %b %Y}", "CSV date range")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="kicker">SOURCE PRODUCT</div>
                <h3>EOS-06 OCM-3 analysed chlorophyll-a</h3>
                <p>
                    Product: <b>E06OCM_L4_AC</b><br>
                    Observation type: satellite-derived ocean-colour
                    chlorophyll-a<br>
                    Spatial grid used in the project: approximately 0.25°.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="card">
                <div class="kicker">PROJECT OUTPUT</div>
                <h3>Potential bloom-risk screening</h3>
                <p>
                    The current result is an early-warning support layer. It
                    highlights cells for closer investigation using the project's
                    screening model and derived chlorophyll-a signals.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown(
            """
            <div class="download-card">
                <h3>Latest processed field</h3>
                <p>Download the exact latest field used by the dashboard.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⇩ Download latest field CSV",
            data=latest.to_csv(index=False).encode("utf-8"),
            file_name="bloomdetect_latest_field.csv",
            mime="text/csv",
            key="download_latest_field",
            use_container_width=True,
        )

    with d2:
        st.markdown(
            """
            <div class="download-card">
                <h3>Current screening output</h3>
                <p>Download only the cells currently screened as potential bloom risk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(
            "⇩ Download potential-risk cells",
            data=risk.to_csv(index=False).encode("utf-8"),
            file_name="bloomdetect_potential_risk_cells.csv",
            mime="text/csv",
            key="download_risk_cells",
            use_container_width=True,
        )

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="kicker">LATEST FIELD PREVIEW</div><h3>Selected columns</h3>',
        unsafe_allow_html=True,
    )

    preview_cols = [
        c for c in [
            "latitude",
            "longitude",
            "date",
            "chla",
            "risk_label",
            "risk_probability",
        ]
        if c in latest.columns
    ]

    preview = latest[preview_cols].copy().head(100)

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="note">
            <b>Important:</b> the dataset contains satellite-derived
            chlorophyll-a and the project's screening result. Confirmed HAB
            species, toxin measurements and field-validation observations are
            not contained in this dashboard unless separately added.
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
        BloomDetect AI · EOS-06 OCM-3 · Potential bloom-risk screening ·
        Latest processed field: {latest_display}
    </div>
    """,
    unsafe_allow_html=True,
)
