from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# BLOOMDETECT AI — FINAL INTERACTIVE OCEAN EXPERIENCE
# One Streamlit app, internal navigation, no multipage routing.
# Uses the latest processed prediction CSV only.
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"

BLOOM_PROCESS_IMAGE = BASE_DIR / "algal_bloom_process.png"

# -----------------------------
# DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv was not found beside app.py."
        )

    data = pd.read_csv(DATA_PATH)
    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "The prediction CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ["latitude", "longitude", "chla"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

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
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["latitude", "longitude", "date"]).copy()
    data["plot_lon"] = ((data["longitude"] + 180) % 360) - 180
    data["risk_flag"] = (
        data["risk_label"].astype(str).str.strip().str.lower()
        == "potential bloom risk"
    )
    return data


try:
    df = load_data()
except Exception as exc:
    st.error("BloomDetect could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()
risk = latest[latest["risk_flag"]].copy()
normal = latest[~latest["risk_flag"]].copy()

# -----------------------------
# NAVIGATION
# -----------------------------
PAGES = ["home", "map", "location", "insights", "method", "data"]
LABELS = {
    "home": "Home",
    "map": "Ocean Map",
    "location": "Location",
    "insights": "Insights",
    "method": "How It Works",
    "data": "Data",
}

if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"


def go(page):
    st.session_state.page = page


# -----------------------------
# GEOGRAPHIC FILTERS
# -----------------------------
def region_mask(frame, region):
    lat = frame["latitude"]
    lon = frame["plot_lon"]
    if region == "Global Ocean":
        return pd.Series(True, index=frame.index)
    if region == "Indian Ocean":
        return lat.between(-40, 30) & lon.between(20, 120)
    if region == "Arabian Sea":
        return lat.between(5, 30) & lon.between(45, 78)
    if region == "Bay of Bengal":
        return lat.between(5, 23) & lon.between(78, 100)
    return pd.Series(True, index=frame.index)


def map_figure(observations, risk_points=None, region="Global Ocean", height=650):
    """Satellite-style chlorophyll field with an optional red risk overlay."""
    obs = observations.dropna(subset=["latitude", "plot_lon", "chla"]).copy()

    fig = px.scatter_geo(
        obs,
        lat="latitude",
        lon="plot_lon",
        color="chla",
        hover_name="risk_label",
        hover_data={
            "latitude": ":.2f",
            "plot_lon": ":.2f",
            "chla": ":.5f",
            "risk_label": True,
        },
        projection="mercator",
        color_continuous_scale=[
            [0.00, "#061b3a"],
            [0.18, "#0a4e83"],
            [0.38, "#0e8fb0"],
            [0.58, "#28c9cf"],
            [0.78, "#a5e46b"],
            [1.00, "#ffe58a"],
        ],
        range_color=(0, max(float(obs["chla"].quantile(0.995)), 0.01)),
    )

    fig.update_traces(
        marker=dict(size=4.2, opacity=0.78, line=dict(width=0)),
        name="Chlorophyll-a field",
        selector=dict(type="scattergeo"),
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>"
            "Latitude: %{lat:.2f}<br>"
            "Longitude: %{lon:.2f}<br>"
            "Chlorophyll-a: %{customdata[2]:.5f}<extra></extra>"
        ),
    )

    if risk_points is not None and len(risk_points):
        rp = risk_points.dropna(subset=["latitude", "plot_lon"]).copy()
        risk_fig = px.scatter_geo(
            rp,
            lat="latitude",
            lon="plot_lon",
            hover_name="risk_label",
            hover_data={
                "latitude": ":.2f",
                "plot_lon": ":.2f",
                "chla": ":.5f",
            },
            projection="mercator",
        )
        risk_trace = risk_fig.data[0]
        risk_trace.marker = dict(
            size=11,
            color="#ff5b67",
            opacity=0.98,
            line=dict(width=1.8, color="#ffffff"),
        )
        risk_trace.name = "Potential bloom risk"
        fig.add_trace(risk_trace)

    if region == "Indian Ocean":
        geo_update = dict(lataxis_range=[-40, 30], lonaxis_range=[20, 120])
    elif region == "Arabian Sea":
        geo_update = dict(lataxis_range=[5, 30], lonaxis_range=[45, 78])
    elif region == "Bay of Bengal":
        geo_update = dict(lataxis_range=[5, 23], lonaxis_range=[78, 100])
    else:
        geo_update = {}

    fig.update_geos(
        showcountries=True,
        countrycolor="rgba(190,225,230,.38)",
        showcoastlines=True,
        coastlinecolor="rgba(230,250,250,.65)",
        showland=True,
        landcolor="#203c48",
        showocean=True,
        oceancolor="#041c39",
        showlakes=False,
        bgcolor="#041a34",
        **geo_update,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="#041a34",
        plot_bgcolor="#041a34",
        font=dict(color="#dffcff"),
        coloraxis_colorbar=dict(
            title="Chlorophyll-a",
            thickness=14,
            len=0.72,
            outlinewidth=0,
            tickfont=dict(color="#dffcff"),
            titlefont=dict(color="#dffcff"),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            x=0.02,
            bgcolor="rgba(4,26,52,.82)",
            bordercolor="rgba(160,240,245,.20)",
            borderwidth=1,
            font=dict(color="#efffff"),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#58cdd1",
            font_color="#143b43",
        ),
    )
    return fig


def metric_card(label, value, note=""):
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


def section_title(kicker, title, copy):
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="kicker">{kicker}</div>
            <h2>{title}</h2>
            <p>{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# GLOBAL OCEAN UI
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

:root {{
    --ink:#123942;
    --muted:#66848b;
    --aqua:#079eab;
    --aqua2:#62dddd;
    --deep:#075e70;
    --pale:#effcfc;
    --line:rgba(10,103,115,.12);
    --shadow:0 24px 70px rgba(17,93,104,.12);
}}

html, body {{
    background:#effafa !important;
    scroll-behavior:smooth;
}}

.stApp {{
    background:
      radial-gradient(circle at 7% 6%, rgba(84,220,219,.26), transparent 22%),
      radial-gradient(circle at 94% 24%, rgba(103,202,240,.18), transparent 25%),
      linear-gradient(180deg,#fbffff 0%,#eefbfb 52%,#f9ffff 100%);
    color:var(--ink);
    overflow-x:hidden;
}}

/* Hide only Streamlit chrome. Keep our own navigation visible. */
header[data-testid="stHeader"] {{display:none !important;}}
[data-testid="stToolbar"] {{display:none !important;}}
#MainMenu {{display:none !important;}}
footer {{display:none !important;}}
section[data-testid="stSidebar"] {{display:none !important;}}

.block-container {{
    max-width:1480px;
    padding:0 46px 90px !important;
}}

* {{box-sizing:border-box;}}

/* A subtle animated water layer across every page. */
.stApp::before {{
    content:"";
    position:fixed;
    z-index:0;
    left:-5%; right:-5%; bottom:-90px;
    height:190px;
    pointer-events:none;
    opacity:.52;
    background:
      radial-gradient(ellipse at 20% 60%, rgba(69,210,215,.22) 0 22%, transparent 23%),
      radial-gradient(ellipse at 70% 40%, rgba(43,181,205,.18) 0 26%, transparent 27%);
    filter:blur(5px);
    animation:waterDrift 10s ease-in-out infinite alternate;
}}

@keyframes waterDrift {{
    from {{transform:translateX(-2%) scaleX(1.02);}}
    to {{transform:translateX(2%) scaleX(1.08);}}
}}

/* Floating bubbles */
.stApp::after {{
    content:"·   °       ·        °    ·        °      ·   °       ·";
    position:fixed;
    z-index:0;
    left:0; right:0; bottom:-20px;
    height:100vh;
    pointer-events:none;
    color:rgba(8,155,170,.16);
    font-size:32px;
    letter-spacing:70px;
    line-height:95px;
    white-space:pre-wrap;
    animation:bubbles 18s linear infinite;
}}

@keyframes bubbles {{
    0% {{transform:translateY(80px); opacity:.1;}}
    50% {{opacity:.35;}}
    100% {{transform:translateY(-110px); opacity:.08;}}
}}

/* Navigation */
.nav-wrap {
    position:sticky;
    top:0;
    z-index:10000;
    margin:0 -46px;
    padding:14px 46px 15px;
    background:rgba(250,255,255,.94);
    border-bottom:1px solid rgba(16,107,116,.12);
    box-shadow:0 10px 30px rgba(17,87,99,.07);
    backdrop-filter:blur(24px);
    -webkit-backdrop-filter:blur(24px);
}
.brand-line {display:flex;align-items:center;gap:12px;margin-bottom:4px;}
.logo {
    width:46px;height:46px;border-radius:15px;display:grid;place-items:center;
    background:linear-gradient(145deg,#2fcfd1,#087e91);color:#fff;font-size:18px;font-weight:800;
    box-shadow:0 10px 26px rgba(7,140,153,.22);
}
.brand {font:800 1.16rem Manrope,sans-serif;color:#123942;letter-spacing:-.03em;}
.tagline {font-size:.72rem;color:#719097;margin-left:58px;margin-top:0;margin-bottom:12px;}

.nav-btn button {
    min-height:54px !important;
    border-radius:16px !important;
    border:1.5px solid rgba(12,113,124,.18) !important;
    background:#ffffff !important;
    color:#123c45 !important;
    font-weight:800 !important;
    font-size:.88rem !important;
    box-shadow:0 6px 18px rgba(18,91,102,.08) !important;
    transition:transform .18s ease,background .18s ease,box-shadow .18s ease,border .18s ease !important;
}
.nav-btn button p,.nav-btn button span {color:#123c45 !important;}
.nav-btn button:hover {
    transform:translateY(-2px) !important;
    background:#dff9fa !important;
    border-color:#18aab2 !important;
    color:#075f70 !important;
    box-shadow:0 12px 26px rgba(8,133,145,.14) !important;
}
.nav-btn button:focus, .nav-btn button:active {
    background:#dff9fa !important;
    border-color:#18aab2 !important;
    color:#075f70 !important;
    box-shadow:0 0 0 2px rgba(24,170,178,.16), 0 10px 24px rgba(8,133,145,.12) !important;
}
.nav-btn button:focus p, .nav-btn button:active p, .nav-btn button:focus span, .nav-btn button:active span {
    color:#075f70 !important;
}

/* General */
.section-heading {{position:relative;z-index:2;padding:68px 0 28px;}}
.kicker {{font:800 .68rem Manrope,sans-serif;letter-spacing:.22em;text-transform:uppercase;color:#079ba9;margin-bottom:18px;}}
.section-heading h2 {{font:800 clamp(2.25rem,4.2vw,4.6rem)/1.02 Manrope,sans-serif;letter-spacing:-.055em;color:#123a43;margin:0 0 20px;max-width:1050px;}}
.section-heading p {{font:500 1.04rem/1.75 'DM Sans',sans-serif;color:#648189;max-width:850px;margin:0;}}

.metric-card, .glass-card, .feature-card {{
    position:relative; z-index:2;
    background:rgba(255,255,255,.70);
    border:1px solid rgba(13,107,117,.12);
    border-radius:25px;
    box-shadow:var(--shadow);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
}}
.metric-card {{padding:27px 28px;min-height:138px;}}
.metric-label {{font:800 .67rem Manrope,sans-serif;letter-spacing:.16em;text-transform:uppercase;color:#769097;}}
.metric-value {{font:800 2.2rem Manrope,sans-serif;color:#153d45;margin-top:12px;letter-spacing:-.04em;}}
.metric-note {{font-size:.78rem;color:#79939a;margin-top:6px;}}
.feature-card {{padding:30px;min-height:200px;transition:transform .25s ease, box-shadow .25s ease;}}
.feature-card:hover {{transform:translateY(-6px);box-shadow:0 30px 80px rgba(17,98,108,.16);}}
.feature-icon {{font-size:1.65rem;margin-bottom:14px;}}
.feature-card h3 {{font:800 1.35rem Manrope,sans-serif;color:#163c44;margin:0 0 10px;}}
.feature-card p {{font-size:.94rem;line-height:1.7;color:#6c858b;margin:0;}}

/* Home */
.hero {
    position:relative;z-index:2;min-height:560px;margin-top:30px;overflow:hidden;
    border-radius:34px;
    background:
      radial-gradient(circle at 78% 22%,rgba(96,236,232,.30),transparent 22%),
      radial-gradient(circle at 20% 90%,rgba(24,155,185,.30),transparent 30%),
      linear-gradient(135deg,#073848 0%,#075d70 46%,#0aa1a9 100%);
    box-shadow:0 34px 100px rgba(5,77,87,.20);
}
.hero::before {
    content:"";position:absolute;inset:-15%;
    background:
      repeating-radial-gradient(ellipse at 30% 110%,transparent 0 46px,rgba(161,250,246,.13) 47px 49px,transparent 50px 90px);
    transform:rotate(-5deg);
    animation:heroWater 13s linear infinite;
}
.hero::after {
    content:"";position:absolute;left:-10%;right:-10%;bottom:-100px;height:260px;
    background:rgba(135,239,237,.18);border-radius:50% 50% 0 0;
    animation:wave 7s ease-in-out infinite alternate;
}
@keyframes heroWater {from{transform:translateX(-4%) rotate(-5deg)}to{transform:translateX(4%) rotate(-5deg)}}
@keyframes wave {from{transform:translateX(-2%) rotate(-1deg)}to{transform:translateX(2%) rotate(1deg)}}
.hero-content {position:relative;z-index:3;padding:105px 82px 145px;max-width:950px;}
.hero-kicker {font:800 .72rem Manrope,sans-serif;letter-spacing:.22em;color:#a7ffff;margin-bottom:24px;}
.hero h1 {font:800 clamp(4rem,7vw,7.4rem)/.9 Manrope,sans-serif;color:#d9ffff;letter-spacing:-.075em;margin:0 0 25px;}
.hero p {font:500 1.1rem/1.75 'DM Sans',sans-serif;color:rgba(242,255,255,.90);max-width:760px;}
.hero-pill {display:inline-block;margin-top:22px;padding:11px 16px;border:1px solid rgba(161,249,248,.30);background:rgba(4,57,66,.25);border-radius:999px;color:#e9ffff;font-size:.78rem;backdrop-filter:blur(10px);}
.scroll-cue {position:absolute;z-index:4;right:44px;bottom:34px;color:#dffefe;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;animation:cue 2s ease-in-out infinite;}
@keyframes cue {0%,100%{transform:translateY(0);opacity:.55}50%{transform:translateY(8px);opacity:1}}

/* Home education */
.bloom-story {
    position:relative;z-index:2;
    display:grid;grid-template-columns:1fr 1.05fr;gap:28px;align-items:stretch;
}
.story-card {
    background:rgba(255,255,255,.86);border:1px solid var(--line);border-radius:28px;
    padding:34px;box-shadow:var(--shadow);
}
.story-card h3 {font:800 2rem/1.1 Manrope;color:#123a43;margin:0 0 16px;letter-spacing:-.04em;}
.story-card p {font-size:1rem;line-height:1.8;color:#648189;margin:0 0 14px;}
.bloom-diagram {
    background:#fff;border-radius:28px;border:1px solid var(--line);box-shadow:var(--shadow);
    padding:12px;overflow:hidden;
}
.bloom-diagram img {width:100%;height:100%;min-height:520px;object-fit:contain;display:block;border-radius:20px;}
.remember-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
.remember-card {
    position:relative;z-index:2;padding:28px;border-radius:24px;background:rgba(255,255,255,.78);
    border:1px solid var(--line);box-shadow:0 18px 50px rgba(19,93,104,.08);
}
.remember-card .num {font:800 1rem Manrope;color:#079ba9;margin-bottom:12px;}
.remember-card h3 {font:800 1.18rem Manrope;color:#123b43;margin:0 0 9px;}
.remember-card p {font-size:.92rem;line-height:1.7;color:#6c858b;margin:0;}
/* Data dictionary */
.data-table-wrap {{position:relative;z-index:2;overflow:hidden;border-radius:20px;border:1px solid rgba(13,107,117,.12);box-shadow:var(--shadow);background:#ffffff;}}
.data-table {{width:100%;border-collapse:collapse;font-size:.82rem;color:#173e46;}}
.data-table th {{text-align:left;padding:13px 16px;background:#123943;color:#ffffff;font-weight:800;}}
.data-table td {{padding:11px 16px;border-top:1px solid #e6f0f0;color:#315a62;background:#ffffff;}}
.data-table tr:nth-child(even) td {{background:#f7fcfc;}}

/* Map / charts */
.control-card {{padding:24px;border-radius:25px;background:rgba(255,255,255,.86);border:1px solid var(--line);box-shadow:var(--shadow);}}\n.map-shell {{position:relative;z-index:2;padding:10px;border-radius:28px;background:#041a34;box-shadow:0 28px 80px rgba(4,35,55,.22);border:1px solid rgba(126,235,239,.18);}}\n.map-note {{padding:16px 18px;border-radius:16px;background:rgba(5,33,54,.86);color:#dffcff;font-size:.86rem;line-height:1.6;margin-bottom:12px;border:1px solid rgba(124,230,236,.16);}}
.download-box {{padding:32px;border-radius:28px;background:linear-gradient(135deg,#087f90,#19aab0);box-shadow:0 24px 65px rgba(8,126,141,.20);color:white;}}
.download-box h3 {{font:800 1.6rem Manrope;margin:0 0 7px;color:white;}}
.download-box p {{color:rgba(255,255,255,.80);margin:0 0 18px;}}
.download-box button {{background:white!important;color:#087d8b!important;border:none!important;font-weight:800!important;}}
.quote {{position:relative;z-index:2;margin:35px 0;padding:25px 30px;border-left:4px solid #1ab0b6;background:rgba(222,250,250,.65);border-radius:0 20px 20px 0;color:#42666d;line-height:1.7;}}
.footer {{position:relative;z-index:2;border-top:1px solid var(--line);padding:25px 0;color:#78939a;font-size:.76rem;margin-top:70px;}}

/* Location result */
.location-result {{padding:32px;}}
.location-result h3 {{font:800 2.15rem/1.05 Manrope;color:#143b43;margin:0 0 25px;letter-spacing:-.04em;}}
.result-grid {{display:grid;grid-template-columns:1fr 1fr;gap:20px 28px;}}
.result-grid span {{display:block;font-size:.68rem;text-transform:uppercase;letter-spacing:.14em;font-weight:800;color:#78939a;margin-bottom:6px;}}
.result-grid b {{font-size:1rem;color:#214a52;}}
.result-grid .risk-result {{color:#d84e5e;}}
.result-grid .normal-result {{color:#21858c;}}

/* Buttons */
.stButton button p,.stButton button span,.stDownloadButton button p,.stDownloadButton button span {color:inherit !important;}
.stDownloadButton button {
    background:linear-gradient(135deg,#087f90,#16aeb3)!important;color:#ffffff!important;border:0!important;
}
.stButton button {border-radius:14px!important;min-height:48px!important;font-weight:800!important;color:#123c45!important;}
.stDownloadButton button {border-radius:15px!important;min-height:54px!important;font-weight:800!important;color:#ffffff!important;}
/* Streamlit controls */
div[data-baseweb="select"] > div {{border-radius:13px!important;background:rgba(255,255,255,.88)!important;border-color:rgba(13,107,117,.13)!important;}}
.stNumberInput input {{border-radius:13px!important;background:rgba(255,255,255,.88)!important;}}
.stButton button {{border-radius:14px!important;min-height:48px!important;font-weight:700!important;}}
.stDownloadButton button {{border-radius:15px!important;min-height:52px!important;font-weight:800!important;}}
[data-testid="stDataFrame"] {{border-radius:18px;overflow:hidden;background:#ffffff!important;}}
[data-testid="stDataFrame"] * {{color:#173e46 !important;}}
.stNumberInput label, .stSelectbox label {{color:#315f68 !important;font-weight:700 !important;}}
.stNumberInput input {{color:#173e46 !important;font-weight:700 !important;}}
.stNumberInput button {{color:#173e46 !important;background:#ffffff !important;}}

/* Visible water motion across the whole app */
.water-motion {{position:fixed;left:0;right:0;bottom:0;height:115px;z-index:1;pointer-events:none;overflow:hidden;opacity:.75;}}
.water-motion .wave {{position:absolute;left:-10%;width:120%;height:70px;border-radius:50% 50% 0 0;filter:blur(.2px);}}
.water-motion .wave.one {{bottom:-38px;background:rgba(24,181,192,.17);animation:waveMove1 7s ease-in-out infinite alternate;}}
.water-motion .wave.two {{bottom:-52px;background:rgba(70,211,219,.15);animation:waveMove2 9s ease-in-out infinite alternate-reverse;}}
.water-motion .wave.three {{bottom:-67px;background:rgba(7,126,144,.10);animation:waveMove3 11s ease-in-out infinite alternate;}}
@keyframes waveMove1 {{from{{transform:translateX(-3%) rotate(-1deg) scaleX(1.02)}}to{{transform:translateX(3%) rotate(1deg) scaleX(1.10)}}}}
@keyframes waveMove2 {{from{{transform:translateX(4%) rotate(1deg) scaleX(1.05)}}to{{transform:translateX(-4%) rotate(-1deg) scaleX(1.12)}}}}
@keyframes waveMove3 {{from{{transform:translateX(-2%) scaleX(1.04)}}to{{transform:translateX(2%) scaleX(1.13)}}}}

@media(max-width:900px) {{
    .block-container {{padding:0 18px 60px!important;}}
    .nav-wrap {{margin:0 -18px;padding:10px 18px;}}
    .hero-content {{padding:70px 35px 130px;}}
    .hero h1 {{font-size:4rem;}}
    .process {{grid-template-columns:1fr;}}
}}
</style>
""",
    unsafe_allow_html=True,
)

# A visible animated water layer behind the interface.
st.markdown("""<div class="water-motion"><div class="wave one"></div><div class="wave two"></div><div class="wave three"></div></div>""", unsafe_allow_html=True)

# -----------------------------
# NAV BAR
# -----------------------------
st.markdown('<div class="nav-wrap"><div class="brand-line"><div class="logo">≈</div><div class="brand">BloomDetect AI</div></div><div class="tagline">Satellite-based potential bloom-risk screening</div>', unsafe_allow_html=True)
nav_cols = st.columns(6, gap="small")
for col, page_name in zip(nav_cols, PAGES):
    with col:
        st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
        st.button(
            LABELS[page_name],
            key=f"nav_{page_name}",
            use_container_width=True,
            on_click=go,
            args=(page_name,),
        )
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page

# ============================================================
# HOME
# ============================================================
if page == "home":
    st.markdown(
        """
        <section class="hero">
            <div class="hero-content">
                <div class="hero-kicker">BLOOMDETECT AI</div>
                <h1>Reading the ocean<br>from space.</h1>
                <p>A visual screening experience for finding ocean locations where chlorophyll patterns may deserve a closer look. Explore the map, inspect a grid cell, and understand the science behind the signal.</p>
                <span class="hero-pill">Explore · Inspect · Understand</span>
            </div>
            <div class="scroll-cue">scroll · explore ↓</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    section_title(
        "Start here",
        "What is an algal bloom?",
        "An algal bloom happens when algae or phytoplankton increase rapidly and become unusually concentrated in part of a water body. Some blooms are harmless; some can create ecological or health impacts.",
    )

    left, right = st.columns([.92, 1.08], gap="large")
    with left:
        st.markdown(
            """
            <div class="story-card">
                <div class="kicker">The simple picture</div>
                <h3>Small organisms can create a very visible change in water.</h3>
                <p>Phytoplankton use light and nutrients to grow. When conditions allow large numbers of cells to accumulate, the water can develop a stronger chlorophyll signal.</p>
                <p>A bloom is an ecological event, not automatically a harmful event. Harm depends on the organisms involved and the effects they produce.</p>
                <p><b>BloomDetect focuses on the observable signal:</b> it screens for patterns that may deserve further investigation rather than declaring a confirmed harmful bloom.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if BLOOM_PROCESS_IMAGE.exists():
            st.markdown('<div class="bloom-diagram">', unsafe_allow_html=True)
            st.image(str(BLOOM_PROCESS_IMAGE), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Illustration: nutrient input and warming can support algal growth; decomposition can reduce oxygen in the water.")
        else:
            st.error("The algal-bloom explanation image is missing. Keep algal_bloom_process.png beside app.py.")

    section_title(
        "Three things to remember",
        "The signal needs context.",
        "These are the three ideas that keep the dashboard scientifically honest.",
    )
    remember = st.columns(3, gap="medium")
    cards = [
        ("01", "Not every bloom is harmful", "A high concentration of algae does not by itself mean toxins or harmful effects are present."),
        ("02", "Chlorophyll is a signal", "Chlorophyll-a can indicate changes in phytoplankton biomass, but it cannot identify every species or confirm toxicity."),
        ("03", "Screening comes before validation", "A flagged location is a place to investigate further, alongside oceanographic observations and field validation."),
    ]
    for col, (num, title, copy) in zip(remember, cards):
        with col:
            st.markdown(
                f'<div class="remember-card"><div class="num">{num}</div><h3>{title}</h3><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )

    section_title(
        "Explore the experience",
        "Three views. Three different jobs.",
        "Nothing here repeats the dataset documentation. Each screen has one purpose.",
    )
    explore = st.columns(3, gap="medium")
    explore_cards = [
        ("🌊", "Ocean Map", "See the chlorophyll field and the locations highlighted for potential bloom risk."),
        ("📍", "Location", "Enter coordinates and inspect the nearest processed grid cell."),
        ("📊", "Insights", "See compact summaries of the latest field and geographic concentration."),
    ]
    for col, (icon, title, copy) in zip(explore, explore_cards):
        with col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="quote">The dashboard is designed to help identify locations worth looking at more closely. A model flag is not a confirmation of a harmful algal bloom.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# MAP
# ============================================================
elif page == "map":
    section_title(
        "01 · Spatial explorer",
        "See the chlorophyll field.",
        "The map uses colour to show the satellite-derived chlorophyll field. Red markers sit on top only where the model screening result is potential bloom risk.",
    )

    c1, c2, c3 = st.columns([1.15, 1.05, .8], gap="medium")
    with c1:
        region = st.selectbox(
            "Geographic view",
            ["Global Ocean", "Indian Ocean", "Arabian Sea", "Bay of Bengal"],
            key="map_region",
        )
    with c2:
        highlight = st.toggle("Show potential-risk overlay", value=True, key="risk_overlay")
    with c3:
        metric_card("Visible cells", f"{len(latest[region_mask(latest, region)]):,}", "current view")

    visible = latest[region_mask(latest, region)].copy()
    visible_risk = risk[region_mask(risk, region)].copy() if highlight else pd.DataFrame()

    st.markdown(
        f'<div class="map-note"><b>{region}</b> · Colour represents chlorophyll-a concentration. Bright warm tones indicate higher values within the displayed field. Red circles mark model-screened potential bloom-risk locations.</div>',
        unsafe_allow_html=True,
    )

    fig = map_figure(visible, visible_risk, region=region, height=700)
    st.markdown('<div class="map-shell">', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="quote">This is a visual screening map, not a photographic satellite image. The colour field represents the processed chlorophyll-a observations available in the dashboard.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# LOCATION
# ============================================================
elif page == "location":
    section_title(
        "02 · Location explorer",
        "Inspect one grid cell.",
        "Move the coordinates and the nearest available observation changes with them. The page is intentionally focused on one location.",
    )

    default_lat = float(risk.iloc[0]["latitude"]) if len(risk) else 15.0
    default_lon = float(risk.iloc[0]["plot_lon"]) if len(risk) else 75.0

    left, right = st.columns([.82, 1.38], gap="large")
    with left:
        st.markdown(
            '<div class="control-card"><div class="kicker">Choose a point</div><p style="color:#67828a;margin:0">Use any latitude and longitude. The app finds the nearest 0.25° observation in the current processed grid.</p></div>',
            unsafe_allow_html=True,
        )
        lat_in = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat, step=0.25, format="%.2f", key="location_lat")
        lon_in = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon, step=0.25, format="%.2f", key="location_lon")

    lon_norm = ((lon_in + 180) % 360) - 180
    dist = np.sqrt((latest["latitude"] - lat_in) ** 2 + (latest["plot_lon"] - lon_norm) ** 2)
    row = latest.loc[dist.idxmin()]

    with right:
        risk_text = str(row["risk_label"])
        risk_class = "risk-result" if risk_text.lower() == "potential bloom risk" else "normal-result"
        prob = row.get("risk_probability", np.nan)
        prob_text = "Not available" if pd.isna(prob) else f"{float(prob)*100:.1f}%" if float(prob) <= 1 else f"{float(prob):.1f}%"
        st.markdown(
            f"""
            <div class="glass-card location-result">
                <div class="kicker">Nearest available observation</div>
                <h3>{row['latitude']:.2f}° · {row['plot_lon']:.2f}°</h3>
                <div class="result-grid">
                    <div><span>Observation date</span><b>{row['date'].strftime('%d %B %Y')}</b></div>
                    <div><span>Chlorophyll-a</span><b>{row['chla']:.5f}</b></div>
                    <div><span>Screening result</span><b class="{risk_class}">{risk_text}</b></div>
                    <div><span>Model probability</span><b>{prob_text}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title(
        "Signals at this point",
        "What changes when you move the marker?",
        "These six values are the model-context signals stored for the selected grid cell.",
    )
    available = [
        ("Previous chlorophyll-a", "previous_chla"),
        ("Historical baseline", "historical_baseline"),
        ("Recent mean", "recent_mean"),
        ("Recent maximum", "recent_max"),
        ("Chlorophyll anomaly", "chla_anomaly"),
        ("Chlorophyll change", "chla_change"),
    ]
    cards = st.columns(3, gap="medium")
    for i, (label, col) in enumerate(available):
        with cards[i % 3]:
            value = row[col] if col in row.index and pd.notna(row[col]) else None
            text_value = "Not available" if value is None else f"{float(value):.5f}"
            note = "selected grid cell" if value is not None else "not available"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.45rem">{text_value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

# ============================================================
# INSIGHTS
# ============================================================
elif page == "insights":
    section_title(
        "03 · Insights",
        "Useful signals from the latest field.",
        "Only summaries that add something beyond the map and location lookup appear here.",
    )

    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Observation cells", f"{len(latest):,}", "latest field")
    with cols[1]: metric_card("Potential-risk cells", f"{len(risk):,}", "model screening")
    with cols[2]: metric_card("Normal cells", f"{len(normal):,}", "latest field")
    with cols[3]: metric_card("Risk share", f"{len(risk)/len(latest)*100:.2f}%", "latest field")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="section-heading"><div class="kicker">Screening composition</div><h2>Normal vs potential-risk.</h2><p>A direct count of the two screening outcomes in the latest field.</p></div>',
            unsafe_allow_html=True,
        )
        composition = pd.DataFrame({"Classification": ["Normal", "Potential bloom risk"], "Cells": [len(normal), len(risk)]})
        fig1 = px.bar(
            composition, x="Classification", y="Cells", text="Cells",
            color="Classification",
            color_discrete_map={"Normal":"#7fcbd0", "Potential bloom risk":"#ef6471"},
        )
        fig1.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig1.update_layout(height=390, showlegend=False, margin=dict(l=10,r=10,t=25,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", font_color="#183f47")
        st.plotly_chart(fig1, width="stretch", config={"displaylogo": False})

    with c2:
        st.markdown(
            '<div class="section-heading"><div class="kicker">Chlorophyll field</div><h2>How the values are distributed.</h2><p>This shows the spread of the current chlorophyll-a observations, not harmfulness.</p></div>',
            unsafe_allow_html=True,
        )
        clean_chla = latest["chla"].replace([np.inf, -np.inf], np.nan).dropna()
        fig2 = px.histogram(clean_chla.to_frame(name="Chlorophyll-a"), x="Chlorophyll-a", nbins=45, color_discrete_sequence=["#18a3af"])
        fig2.update_layout(height=390, margin=dict(l=10,r=10,t=25,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", font_color="#183f47", yaxis_title="Observation cells")
        st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})

    region_rows = []
    for name in ["Indian Ocean", "Arabian Sea", "Bay of Bengal"]:
        m = region_mask(risk, name)
        region_rows.append({"Geographic view": name, "Potential-risk cells": int(m.sum())})
    region_df = pd.DataFrame(region_rows)

    st.markdown(
        '<div class="section-heading"><div class="kicker">Geographic concentration</div><h2>Where do flagged cells appear?</h2><p>A geographic count across three study windows, using the same latest field.</p></div>',
        unsafe_allow_html=True,
    )
    fig3 = px.bar(region_df, x="Geographic view", y="Potential-risk cells", text="Potential-risk cells", color_discrete_sequence=["#1499a8"])
    fig3.update_traces(texttemplate="%{text}", textposition="outside")
    fig3.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", font_color="#183f47", showlegend=False)
    st.plotly_chart(fig3, width="stretch", config={"displaylogo": False})

    st.markdown(
        '<div class="section-heading"><div class="kicker">Follow-up queue</div><h2>Locations to inspect next.</h2><p>These are the highest chlorophyll-a values among the model-screened potential-risk cells. This is a screening queue, not a confirmed severity ranking.</p></div>',
        unsafe_allow_html=True,
    )
    queue = risk.sort_values("chla", ascending=False).head(15).copy()
    show_cols = [c for c in ["latitude", "longitude", "chla", "risk_label"] if c in queue.columns]
    if len(queue):
        st.dataframe(
            queue[show_cols].rename(columns={"latitude":"Latitude", "longitude":"Longitude", "chla":"Chlorophyll-a", "risk_label":"Screening"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No potential-risk cells are available in the latest field.")

# ============================================================
# METHOD
# ============================================================
elif page == "method":
    section_title(
        "04 · How it works",
        "From observation to screening support.",
        "The dashboard turns a satellite-derived environmental signal into a transparent sequence of processing, modelling and visual inspection.",
    )

    steps = [
        ("01", "Observe", "Start with satellite-derived chlorophyll-a observations."),
        ("02", "Clean", "Remove invalid records and organize the observations into a consistent spatial-temporal grid."),
        ("03", "Add context", "Compare a location with its previous observation and historical behaviour."),
        ("04", "Screen", "A Decision Tree classifies locations against the project's proxy potential-risk target."),
        ("05", "Map", "Place the screening result back onto the geographic grid."),
        ("06", "Validate", "Use flagged locations as candidates for further oceanographic or field investigation."),
    ]
    for num, title, copy in steps:
        st.markdown(
            f'<div class="glass-card" style="padding:28px;margin:13px 0;display:flex;gap:24px;align-items:flex-start"><div style="font:800 1.8rem Manrope;color:#079ba9;min-width:58px">{num}</div><div><h3 style="margin:0 0 7px;font:800 1.25rem Manrope;color:#163c44">{title}</h3><p style="margin:0;color:#6a858c;line-height:1.7">{copy}</p></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="quote">Potential bloom risk is a model-derived screening signal. High chlorophyll-a alone does not prove a harmful algal bloom. The prototype does not identify toxin-producing species and does not replace field validation.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# DATA
# ============================================================
elif page == "data":
    section_title(
        "05 · Data",
        "Everything about the dataset lives here.",
        "Source, coverage, grid structure, model fields and downloadable outputs are kept on this page so the other screens can stay focused.",
    )

    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3")
    with cols[1]: metric_card("Grid spacing", "0.25°", "latitude × longitude")
    with cols[2]: metric_card("Latest cells", f"{len(latest):,}", "current observation date")
    with cols[3]: metric_card("Latest date", latest_date.strftime("%d %b %Y"), "processed dataset")

    st.markdown(
        """
        <div class="story-card" style="margin:25px 0">
            <div class="kicker">Source & coverage</div>
            <h3>What the dashboard is built from.</h3>
            <p><b>Satellite product:</b> EOS-06 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC).</p>
            <p><b>Variable used:</b> chlorophyll-a (<code>chla</code>).</p>
            <p><b>Spatial structure:</b> 0.25° latitude × 0.25° longitude global-ocean grid.</p>
            <p><b>Current dashboard view:</b> the latest processed observation date represented in the prediction table.</p>
            <p><b>Interpretation:</b> the dataset provides an environmental observation signal. It does not contain confirmed harmful-bloom species or toxin labels.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading"><div class="kicker">Data dictionary</div><h2>Fields available in the processed file.</h2><p>The public dashboard reads the latest processed prediction table.</p></div>',
        unsafe_allow_html=True,
    )
    rows = [
        ("latitude", "Grid latitude", "degrees"),
        ("longitude", "Grid longitude", "degrees"),
        ("date", "Observation date", "date"),
        ("chla", "Satellite-derived chlorophyll-a", "product value"),
        ("risk_label", "Model screening result", "Normal / Potential Bloom Risk"),
        ("risk_probability", "Model probability when stored", "probability"),
        ("previous_chla", "Previous observation", "derived feature"),
        ("historical_baseline", "Historical baseline", "derived feature"),
        ("recent_mean", "Recent mean", "derived feature"),
        ("recent_max", "Recent maximum", "derived feature"),
        ("chla_anomaly", "Chlorophyll anomaly", "derived feature"),
        ("chla_change", "Change from previous observation", "derived feature"),
    ]
    table_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a,b,c in rows)
    st.markdown(
        f"""
        <div class="data-table-wrap">
          <table class="data-table">
            <thead><tr><th>Field</th><th>Meaning</th><th>Type</th></tr></thead>
            <tbody>{table_rows}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="download-box"><h3>Downloads</h3><p>These are the only download controls in the website. Take the processed observations or the current potential-risk screening list with you.</p></div>',
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.download_button(
            "↓ Download latest observations",
            data=latest.to_csv(index=False).encode("utf-8"),
            file_name=f"bloomdetect_latest_{latest_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="data_latest_download",
        )
    with d2:
        st.download_button(
            "↓ Download potential-risk locations",
            data=risk.to_csv(index=False).encode("utf-8"),
            file_name=f"bloomdetect_risk_{latest_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="data_risk_download",
        )

    st.markdown(
        '<div class="quote">Important limitation: the model target is a proxy potential-bloom-risk label derived from chlorophyll behaviour. It is not confirmed HAB ground truth, and high chlorophyll-a alone does not prove a harmful algal bloom.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    "<div class='footer'>BloomDetect AI · Potential bloom-risk screening · Explore the signal, then validate it.</div>",
    unsafe_allow_html=True,
)
