
from pathlib import Path
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI — OCEAN EXPLORER
# Single-file Streamlit app with page-style navigation.
# Current data source: latest_bloom_risk_predictions.csv
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"

# Free-to-use Unsplash image shown as a visual reference in the home page.
# Source: Kristin Hoel, "underwater view of seaweed in the ocean"
HERO_IMAGE = (
    "https://images.unsplash.com/photo-1717292742444-8a7c392da4dd"
    "?auto=format&fit=crop&fm=jpg&q=82&w=1800"
)

# -----------------------------
# DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "The prediction CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["chla"] = pd.to_numeric(df["chla"], errors="coerce")

    if "risk_probability" in df.columns:
        df["risk_probability"] = pd.to_numeric(
            df["risk_probability"], errors="coerce"
        )

    if "model_score" in df.columns:
        df["model_score"] = pd.to_numeric(
            df["model_score"], errors="coerce"
        )

    df = df.dropna(subset=["latitude", "longitude", "date"]).copy()

    # Convert 0–360 longitudes to -180–180 for a normal world map.
    df["plot_lon"] = ((df["longitude"] + 180) % 360) - 180

    return df


try:
    df = load_data()
except Exception as exc:
    st.error("BloomDetect could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()

risk_mask = latest["risk_label"].astype(str).str.lower().eq(
    "potential bloom risk".lower()
)
risk = latest[risk_mask].copy()

# -----------------------------
# PAGE ROUTING
# -----------------------------
try:
    page = st.query_params.get("page", "home")
except Exception:
    page = "home"

valid_pages = {"home", "map", "location", "insights", "method", "data"}
if page not in valid_pages:
    page = "home"


# -----------------------------
# HELPERS
# -----------------------------
def nav_link(label, target, active=False):
    cls = "nav-link active" if active else "nav-link"
    return f'<a class="{cls}" href="?page={target}">{label}</a>'


def fmt_number(value):
    return f"{int(value):,}"


def region_mask(frame, region):
    lat = frame["latitude"]
    lon = frame["plot_lon"]

    if region == "Global Ocean":
        return pd.Series(True, index=frame.index)

    if region == "Indian Ocean":
        return (
            lat.between(-40, 30)
            & lon.between(20, 120)
        )

    if region == "Arabian Sea":
        return (
            lat.between(5, 30)
            & lon.between(45, 78)
        )

    if region == "Bay of Bengal":
        return (
            lat.between(5, 23)
            & lon.between(78, 100)
        )

    return pd.Series(True, index=frame.index)


def make_map(frame, risk_frame=None, title="Ocean observations", height=650):
    fig = go.Figure()

    # Full observation field.
    if len(frame):
        fig.add_trace(
            go.Scattergeo(
                lon=frame["plot_lon"],
                lat=frame["latitude"],
                mode="markers",
                name="Observations",
                marker=dict(
                    size=3.4,
                    color="rgba(94, 184, 205, 0.42)",
                    line=dict(width=0),
                ),
                customdata=np.column_stack(
                    [
                        frame["latitude"].round(3),
                        frame["longitude"].round(3),
                        frame["chla"].round(5),
                        frame["risk_label"].astype(str),
                    ]
                ),
                hovertemplate=(
                    "<b>Observation</b><br>"
                    "Lat: %{customdata[0]}°<br>"
                    "Lon: %{customdata[1]}°<br>"
                    "Chlorophyll-a: %{customdata[2]}<br>"
                    "Screening: %{customdata[3]}"
                    "<extra></extra>"
                ),
            )
        )

    # Risk overlay.
    if risk_frame is not None and len(risk_frame):
        fig.add_trace(
            go.Scattergeo(
                lon=risk_frame["plot_lon"],
                lat=risk_frame["latitude"],
                mode="markers",
                name="Potential bloom risk",
                marker=dict(
                    size=8,
                    color="#ff6b6b",
                    line=dict(width=1.5, color="#fff1ed"),
                    opacity=0.96,
                ),
                customdata=np.column_stack(
                    [
                        risk_frame["latitude"].round(3),
                        risk_frame["longitude"].round(3),
                        risk_frame["chla"].round(5),
                    ]
                ),
                hovertemplate=(
                    "<b>Potential bloom-risk location</b><br>"
                    "Lat: %{customdata[0]}°<br>"
                    "Lon: %{customdata[1]}°<br>"
                    "Chlorophyll-a: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_geos(
        showcountries=True,
        countrycolor="rgba(150,190,194,0.45)",
        showcoastlines=True,
        coastlinecolor="rgba(220,240,240,0.60)",
        showland=True,
        landcolor="#dce8e6",
        showocean=True,
        oceancolor="#082d38",
        bgcolor="rgba(0,0,0,0)",
        showlakes=False,
    )

    fig.update_layout(
        title=None,
        height=height,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dff8f8"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(3,17,22,.72)",
            bordercolor="rgba(120,220,220,.12)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#06212a",
            bordercolor="#63d6d7",
            font_color="#efffff",
        ),
    )
    return fig


def metric_card(label, value, note=""):
    return f"""
    <div class="metric-card glass">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def page_header(kicker, title, description):
    st.markdown(
        f"""
        <div class="page-header">
            <div class="kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# GLOBAL STYLE
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
    --bg:#031218;
    --bg2:#061d25;
    --glass:rgba(8,34,42,.62);
    --glass2:rgba(9,43,52,.48);
    --line:rgba(140,225,225,.15);
    --text:#eefdfd;
    --muted:#9db9be;
    --soft:#6f9198;
    --aqua:#69dfe0;
    --aqua2:#31bfc8;
    --green:#79d7a8;
    --danger:#ff6f72;
}

html{scroll-behavior:smooth;}
.stApp{
    background:
      radial-gradient(circle at 80% 5%, rgba(50,210,220,.12), transparent 24%),
      radial-gradient(circle at 10% 30%, rgba(25,115,140,.10), transparent 25%),
      linear-gradient(180deg,#020c11 0%,#03151c 48%,#020b10 100%);
    color:var(--text);
}
.block-container{
    max-width:1420px;
    padding:0 2.5rem 5rem;
}
footer{visibility:hidden;}
#MainMenu{visibility:hidden;}
[data-testid="stToolbar"]{visibility:hidden;}
header[data-testid="stHeader"]{background:rgba(2,10,14,.65);}

*{box-sizing:border-box;}
a{text-decoration:none!important;}

.glass{
    background:linear-gradient(145deg,rgba(11,45,54,.72),rgba(3,23,30,.62));
    border:1px solid var(--line);
    box-shadow:0 24px 80px rgba(0,0,0,.18);
    backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);
}

/* TOP NAV */
.topnav{
    position:sticky;
    top:0;
    z-index:1000;
    margin:0 -2.5rem;
    padding:12px 2.5rem;
    background:rgba(2,12,17,.76);
    border-bottom:1px solid rgba(140,225,225,.08);
    backdrop-filter:blur(22px);
    -webkit-backdrop-filter:blur(22px);
}
.nav-inner{
    max-width:1420px;
    margin:auto;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:22px;
}
.brand{
    display:flex;
    align-items:center;
    gap:10px;
    color:#f5ffff!important;
    font-family:Manrope,sans-serif;
    font-weight:800;
    font-size:1rem;
    letter-spacing:-.02em;
}
.logo{
    width:31px;height:31px;
    border-radius:10px;
    display:grid;
    place-items:center;
    background:linear-gradient(145deg,#7be9e3,#138c9b);
    box-shadow:0 0 24px rgba(77,221,220,.25);
    color:#022027;
    font-size:17px;
}
.navlinks{display:flex;align-items:center;gap:5px;flex-wrap:wrap;}
.nav-link{
    padding:8px 11px;
    border-radius:999px;
    color:#8eaeb4!important;
    font-size:.78rem;
    font-weight:600;
    transition:.22s ease;
}
.nav-link:hover,.nav-link.active{
    color:#ecffff!important;
    background:rgba(105,223,224,.10);
}

/* HERO */
.hero{
    min-height:700px;
    position:relative;
    overflow:hidden;
    margin-top:16px;
    border-radius:34px;
    background:
      linear-gradient(90deg,rgba(2,18,24,.94) 0%,rgba(2,22,28,.78) 45%,rgba(2,16,21,.30) 100%),
      linear-gradient(180deg,rgba(2,20,27,.12),rgba(2,15,20,.78)),
      url("https://images.unsplash.com/photo-1717292742444-8a7c392da4dd?auto=format&fit=crop&fm=jpg&q=82&w=1800") center/cover;
    border:1px solid rgba(140,225,225,.14);
}
.hero:before{
    content:"";
    position:absolute;
    inset:0;
    background:
      radial-gradient(circle at 74% 30%,rgba(104,227,223,.22),transparent 18%),
      radial-gradient(circle at 84% 74%,rgba(30,178,190,.25),transparent 28%);
    pointer-events:none;
}
.hero:after{
    content:"";
    position:absolute;
    left:-10%;
    bottom:-115px;
    width:120%;
    height:240px;
    background:rgba(18,151,164,.25);
    border-radius:50% 50% 0 0;
    filter:blur(2px);
    animation:wave 7s ease-in-out infinite;
}
@keyframes wave{
    0%,100%{transform:translateX(-2%) rotate(-1deg);}
    50%{transform:translateX(2%) rotate(1deg);}
}
.hero-content{
    position:relative;
    z-index:4;
    max-width:790px;
    padding:95px 8%;
}
.eyebrow{
    color:#72e5e3;
    text-transform:uppercase;
    letter-spacing:.2em;
    font-size:.72rem;
    font-weight:800;
}
.hero h1{
    font-family:Manrope,sans-serif;
    font-size:clamp(3.2rem,7vw,6.7rem);
    line-height:.91;
    letter-spacing:-.065em;
    margin:22px 0 25px;
}
.hero h1 span{color:#6de1df;}
.hero-copy{
    color:#b7d1d4;
    font-size:1.08rem;
    line-height:1.75;
    max-width:650px;
}
.hero-actions{
    display:flex;
    gap:10px;
    margin-top:30px;
    flex-wrap:wrap;
}
.hero-button{
    display:inline-block;
    padding:12px 18px;
    border-radius:999px;
    background:#7ae5e1;
    color:#032027!important;
    font-weight:800;
    font-size:.82rem;
}
.hero-button.secondary{
    background:rgba(255,255,255,.06);
    color:#dff8f8!important;
    border:1px solid rgba(150,230,230,.16);
}
.hero-meta{
    margin-top:22px;
    display:flex;
    gap:10px;
    flex-wrap:wrap;
}
.pill{
    display:inline-block;
    padding:8px 12px;
    border-radius:999px;
    color:#c3e8e9;
    background:rgba(4,38,47,.58);
    border:1px solid rgba(110,220,220,.14);
    font-size:.74rem;
}

/* FLOATING BUBBLES */
.bubble{
    position:absolute;
    border-radius:50%;
    border:1px solid rgba(150,245,240,.22);
    background:rgba(100,220,220,.035);
    animation:float 8s ease-in-out infinite;
}
.b1{width:18px;height:18px;right:18%;top:18%;animation-delay:-2s;}
.b2{width:9px;height:9px;right:29%;top:38%;animation-delay:-5s;}
.b3{width:28px;height:28px;right:9%;top:55%;animation-delay:-1s;}
.b4{width:12px;height:12px;right:40%;top:70%;animation-delay:-6s;}
@keyframes float{
    0%,100%{transform:translateY(0) translateX(0);opacity:.35}
    50%{transform:translateY(-28px) translateX(8px);opacity:.8}
}

/* PAGE */
.page-header{
    padding:76px 0 32px;
    max-width:900px;
}
.kicker{
    color:var(--aqua);
    text-transform:uppercase;
    letter-spacing:.18em;
    font-size:.72rem;
    font-weight:800;
}
.page-header h1{
    font-family:Manrope,sans-serif;
    font-size:clamp(2.4rem,5vw,4.7rem);
    line-height:1;
    letter-spacing:-.055em;
    margin:12px 0 16px;
}
.page-header p{
    color:var(--muted);
    max-width:780px;
    line-height:1.75;
    font-size:1rem;
}
.section-space{height:28px;}
.metric-grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin:22px 0;
}
.metric-card{
    min-height:130px;
    padding:22px;
    border-radius:22px;
}
.metric-label{
    color:#7f9fa5;
    font-size:.69rem;
    text-transform:uppercase;
    letter-spacing:.12em;
    font-weight:700;
}
.metric-value{
    font-family:Manrope,sans-serif;
    font-size:2rem;
    font-weight:800;
    margin-top:9px;
}
.metric-note{color:#759399;font-size:.74rem;margin-top:5px;}
.section-title{
    font-family:Manrope,sans-serif;
    font-size:1.6rem;
    letter-spacing:-.035em;
    margin:0 0 8px;
}
.section-copy{color:var(--muted);line-height:1.7;}
.info-card{
    padding:26px;
    border-radius:24px;
    margin:16px 0;
}
.feature-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:14px;
}
.feature-card{
    min-height:190px;
    padding:24px;
    border-radius:24px;
    transition:transform .25s ease,border-color .25s ease;
}
.feature-card:hover{
    transform:translateY(-5px);
    border-color:rgba(110,225,225,.30);
}
.feature-icon{font-size:1.45rem;margin-bottom:20px;}
.feature-title{font-family:Manrope,sans-serif;font-weight:800;font-size:1.05rem;}
.feature-copy{color:#8faeb3;line-height:1.65;font-size:.85rem;margin-top:8px;}
.small-note{color:#718f95;font-size:.76rem;line-height:1.6;}
.quote{
    padding:28px;
    border-left:2px solid var(--aqua);
    color:#d8eeee;
    background:rgba(83,213,213,.035);
    border-radius:0 20px 20px 0;
    font-family:Manrope,sans-serif;
    font-size:1.25rem;
    line-height:1.55;
}
.footer{
    margin-top:80px;
    padding:35px 0 15px;
    border-top:1px solid rgba(130,210,218,.09);
    color:#718e94;
    font-size:.75rem;
}

/* STREAMLIT WIDGET POLISH */
.stButton>button,.stDownloadButton>button{
    border-radius:999px!important;
    border:1px solid rgba(120,225,225,.16)!important;
    background:rgba(10,42,51,.72)!important;
    color:#e9ffff!important;
    font-weight:700!important;
}
.stButton>button:hover,.stDownloadButton>button:hover{
    border-color:rgba(120,225,225,.42)!important;
    background:rgba(15,65,76,.82)!important;
}
div[data-baseweb="select"]>div{
    background:rgba(5,29,37,.72)!important;
    border-color:rgba(120,225,225,.14)!important;
    border-radius:14px!important;
}
.stNumberInput input{
    background:rgba(5,29,37,.72)!important;
    color:#efffff!important;
    border-radius:12px!important;
}
[data-testid="stMetric"]{
    background:rgba(8,34,42,.52);
    border:1px solid rgba(140,225,225,.10);
    padding:18px;
    border-radius:18px;
}
.stDataFrame{
    border:1px solid rgba(130,210,218,.10);
    border-radius:16px;
}
@media(max-width:900px){
    .metric-grid,.feature-grid{grid-template-columns:1fr 1fr;}
    .hero{min-height:620px;}
    .hero-content{padding:70px 7%;}
}
@media(max-width:620px){
    .block-container{padding:0 1rem 3rem;}
    .topnav{margin:0 -1rem;padding:10px 1rem;}
    .nav-inner{align-items:flex-start;}
    .navlinks{gap:2px;}
    .nav-link{font-size:.7rem;padding:7px 8px;}
    .metric-grid,.feature-grid{grid-template-columns:1fr;}
    .hero h1{font-size:3.2rem;}
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# NAVIGATION
# -----------------------------
st.markdown(
    f"""
<div class="topnav">
  <div class="nav-inner">
    <a class="brand" href="?page=home">
      <span class="logo">≈</span>
      <span>Bloom<span style="color:#69dfe0">Detect</span> AI</span>
    </a>
    <div class="navlinks">
      {nav_link("Home", "home", page=="home")}
      {nav_link("Ocean Map", "map", page=="map")}
      {nav_link("Location", "location", page=="location")}
      {nav_link("Insights", "insights", page=="insights")}
      {nav_link("Method", "method", page=="method")}
      {nav_link("Data", "data", page=="data")}
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HOME
# ============================================================
if page == "home":

    st.markdown(
        f"""
        <section class="hero">
            <div class="bubble b1"></div>
            <div class="bubble b2"></div>
            <div class="bubble b3"></div>
            <div class="bubble b4"></div>
            <div class="hero-content">
                <div class="eyebrow">EOS-06 · OCM-3 · E06OCM_L4_AC</div>
                <h1>Reading the<br><span>ocean</span> from space.</h1>
                <div class="hero-copy">
                    BloomDetect AI turns satellite-derived chlorophyll-a observations
                    into a spatial screening view of locations showing patterns
                    associated with potential bloom risk.
                </div>
                <div class="hero-actions">
                    <a class="hero-button" href="?page=map">Explore the ocean →</a>
                    <a class="hero-button secondary" href="?page=location">Check a location</a>
                </div>
                <div class="hero-meta">
                    <span class="pill">Latest observation · {latest_date.strftime("%d %B %Y")}</span>
                    <span class="pill">{fmt_number(len(latest))} observation cells</span>
                    <span class="pill">{fmt_number(len(risk))} potential-risk cells</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-grid">
            {metric_card("Observation cells", fmt_number(len(latest)), "latest available grid")}
            {metric_card("Potential-risk cells", fmt_number(len(risk)), "model screening result")}
            {metric_card("Normal cells", fmt_number(len(latest)-len(risk)), "latest observation")}
            {metric_card("Observation date", latest_date.strftime("%d %b %Y"), "latest in this file")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-space"></div>
        <div class="kicker">Start exploring</div>
        <h2 class="section-title" style="font-size:2.35rem;">The ocean, in four views.</h2>
        <p class="section-copy">
        No giant dashboard wall. Each part of BloomDetect has one job, so the
        user can move from the ocean field to a location, then to the reasoning
        behind the screening result.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
          <a class="feature-card glass" href="?page=map">
            <div class="feature-icon">🌊</div>
            <div class="feature-title">Ocean Map</div>
            <div class="feature-copy">
              Explore the available observation grid and see potential-risk
              locations highlighted on top of it.
            </div>
          </a>
          <a class="feature-card glass" href="?page=location">
            <div class="feature-icon">⌖</div>
            <div class="feature-title">Location Explorer</div>
            <div class="feature-copy">
              Enter coordinates and inspect the nearest satellite grid cell,
              chlorophyll-a and screening result.
            </div>
          </a>
          <a class="feature-card glass" href="?page=insights">
            <div class="feature-icon">◒</div>
            <div class="feature-title">Insights</div>
            <div class="feature-copy">
              See the latest screening pattern, regional counts and the
              locations that deserve closer investigation.
            </div>
          </a>
          <a class="feature-card glass" href="?page=method">
            <div class="feature-icon">◎</div>
            <div class="feature-title">How it works</div>
            <div class="feature-copy">
              Follow the path from satellite observation to historical context,
              machine learning and spatial screening.
            </div>
          </a>
          <a class="feature-card glass" href="?page=data">
            <div class="feature-icon">⌁</div>
            <div class="feature-title">Data & downloads</div>
            <div class="feature-copy">
              See the dataset details, coverage, limitations and download the
              latest screening table.
            </div>
          </a>
          <div class="feature-card glass">
            <div class="feature-icon">◌</div>
            <div class="feature-title">For real-world screening</div>
            <div class="feature-copy">
              Designed as an early-warning support layer for researchers,
              environmental teams, marine monitoring and education.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-space"></div>
        <div class="quote">
          A satellite signal is the beginning of an investigation, not the conclusion.
        </div>
        <p class="small-note" style="margin-top:12px;">
        Potential bloom risk is a model-derived screening indicator. High
        chlorophyll-a alone does not confirm a harmful algal bloom, species,
        toxin or health risk.
        </p>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MAP
# ============================================================
elif page == "map":

    page_header(
        "01 · SPATIAL EXPLORER",
        "See the ocean field.",
        "The map starts with the full latest observation grid. Potential-risk "
        "locations are an overlay, not a replacement for the observations.",
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        region = st.selectbox(
            "Geographic view",
            ["Global Ocean", "Indian Ocean", "Arabian Sea", "Bay of Bengal"],
        )
    with c2:
        show_risk = st.toggle("Highlight potential-risk locations", value=True)
    with c3:
        st.metric("Latest cells", fmt_number(len(latest)))

    view = latest[region_mask(latest, region)].copy()
    view_risk = risk[region_mask(risk, region)].copy() if show_risk else pd.DataFrame()

    st.markdown(
        f"""
        <div class="info-card glass">
          <div class="section-title">{region}</div>
          <div class="section-copy">
            {fmt_number(len(view))} available observation cells are shown.
            {fmt_number(len(view_risk)) if show_risk else "No"} potential-risk
            locations are highlighted for the selected geographic view.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = make_map(
        view,
        view_risk if show_risk else None,
        title=region,
        height=680,
    )
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
        <div class="feature-grid">
          <div class="feature-card glass">
            <div class="feature-icon">•</div>
            <div class="feature-title">Observation field</div>
            <div class="feature-copy">
              Small blue points represent the available latest observation grid.
            </div>
          </div>
          <div class="feature-card glass">
            <div class="feature-icon">●</div>
            <div class="feature-title">Potential-risk overlay</div>
            <div class="feature-copy">
              Coral points identify locations flagged by the project's
              potential-risk screening rule.
            </div>
          </div>
          <div class="feature-card glass">
            <div class="feature-icon">⌖</div>
            <div class="feature-title">Next step</div>
            <div class="feature-copy">
              Use Location Explorer to inspect a specific coordinate in more detail.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# LOCATION
# ============================================================
elif page == "location":

    page_header(
        "02 · LOCATION EXPLORER",
        "Inspect one place.",
        "Enter a coordinate and BloomDetect will find the nearest available "
        "observation cell in the latest dataset.",
    )

    left, right = st.columns([1, 1.7], gap="large")

    with left:
        st.markdown('<div class="info-card glass">', unsafe_allow_html=True)
        lat_in = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=15.0,
            step=0.25,
        )
        lon_in = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=75.0,
            step=0.25,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Nearest grid cell.
    work = latest.copy()
    lat_scale = np.cos(np.deg2rad(lat_in))
    lon_delta = ((work["plot_lon"] - lon_in + 180) % 360) - 180
    distance = np.sqrt(
        (work["latitude"] - lat_in) ** 2
        + (lon_delta * max(abs(lat_scale), 0.2)) ** 2
    )
    nearest_idx = distance.idxmin()
    row = work.loc[nearest_idx].copy()
    approx_distance = float(distance.loc[nearest_idx]) * 111.0

    is_risk = str(row["risk_label"]).lower() == "potential bloom risk"

    with right:
        status_text = "Potential bloom risk" if is_risk else "Normal screening result"
        status_class = "risk" if is_risk else "normal"

        st.markdown(
            f"""
            <div class="info-card glass">
              <div class="kicker">Nearest observation</div>
              <div class="section-title">
                {float(row["latitude"]):.2f}° · {float(row["plot_lon"]):.2f}°
              </div>
              <div style="margin:10px 0 18px;">
                <span class="pill">{status_text}</span>
              </div>
              <div class="metric-grid" style="grid-template-columns:1fr 1fr;margin:0;">
                {metric_card("Chlorophyll-a", f"{float(row['chla']):.4f}", "satellite-derived")}
                {metric_card("Approx. distance", f"{approx_distance:.1f} km", "from entered point")}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_risk:
            st.warning(
                "This location is flagged for potential bloom risk by the current "
                "screening output. It is not a confirmed harmful algal bloom."
            )
        else:
            st.success(
                "This location is not flagged by the current screening output."
            )

    st.markdown('<div class="section-space"></div>', unsafe_allow_html=True)

    # Values that actually exist in the current CSV.
    value_cols = [
        ("Latitude", row["latitude"]),
        ("Longitude", row["longitude"]),
        ("Observation date", row["date"].strftime("%d %B %Y")),
        ("Chlorophyll-a", row["chla"]),
        ("Screening result", row["risk_label"]),
    ]

    if "previous_chla" in row.index:
        value_cols.append(("Previous chlorophyll-a", row["previous_chla"]))
    if "historical_baseline" in row.index:
        value_cols.append(("Historical baseline", row["historical_baseline"]))
    if "recent_mean" in row.index:
        value_cols.append(("Recent mean", row["recent_mean"]))
    if "recent_max" in row.index:
        value_cols.append(("Recent maximum", row["recent_max"]))

    table = pd.DataFrame(
        {
            "Measurement": [x[0] for x in value_cols],
            "Value": [x[1] for x in value_cols],
        }
    )

    st.markdown(
        '<div class="section-title">Observation details</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="quote">
          Use a flagged location as a candidate for further investigation,
          not as proof that a harmful algal bloom is present.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# INSIGHTS
# ============================================================
elif page == "insights":

    page_header(
        "03 · SCREENING INSIGHTS",
        "What the latest field contains.",
        "A compact view of the latest observation distribution and the locations "
        "that the current screening process has flagged.",
    )

    total = len(latest)
    n_risk = len(risk)
    n_normal = total - n_risk
    risk_share = (100 * n_risk / total) if total else 0
    mean_chla = latest["chla"].mean()
    risk_mean_chla = risk["chla"].mean() if len(risk) else np.nan

    st.markdown(
        f"""
        <div class="metric-grid">
            {metric_card("Latest observations", fmt_number(total), "grid cells")}
            {metric_card("Potential-risk cells", fmt_number(n_risk), "screening output")}
            {metric_card("Risk share", f"{risk_share:.2f}%", "of latest cells")}
            {metric_card("Mean chlorophyll-a", f"{mean_chla:.4f}", "latest field")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown(
            '<div class="info-card glass"><div class="section-title">Observation vs screening result</div>',
            unsafe_allow_html=True,
        )
        fig = go.Figure(
            go.Bar(
                x=["Normal", "Potential risk"],
                y=[n_normal, n_risk],
                marker_color=["#55b8d0", "#ff6f72"],
                text=[n_normal, n_risk],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dff8f8",
            yaxis_title="Locations",
            xaxis_title=None,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b:
        st.markdown(
            '<div class="info-card glass"><div class="section-title">Chlorophyll-a distribution</div>',
            unsafe_allow_html=True,
        )
        hist = latest["chla"].replace([np.inf, -np.inf], np.nan).dropna()
        # Clip only for display so a few extreme values do not flatten the chart.
        if len(hist) > 1000:
            lo, hi = hist.quantile([0.01, 0.99])
            hist = hist.clip(lo, hi)

        fig2 = go.Figure(
            go.Histogram(
                x=hist,
                nbinsx=36,
                marker_color="#5dd6d7",
                opacity=.82,
            )
        )
        fig2.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dff8f8",
            xaxis_title="Chlorophyll-a",
            yaxis_title="Locations",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title" style="margin-top:35px;">Latest screening queue</div>',
        unsafe_allow_html=True,
    )

    queue_cols = [
        c for c in [
            "latitude", "longitude", "date", "chla",
            "previous_chla", "historical_baseline",
            "recent_mean", "recent_max", "risk_label"
        ] if c in risk.columns
    ]

    queue = risk[queue_cols].copy()

    sort_col = None
    for candidate in ["chla", "recent_max", "recent_mean"]:
        if candidate in queue.columns:
            sort_col = candidate
            break
    if sort_col:
        queue = queue.sort_values(sort_col, ascending=False)

    st.dataframe(
        queue.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing up to 100 of {len(risk):,} flagged locations. "
        "The queue is for screening and prioritization, not confirmation."
    )

# ============================================================
# METHOD
# ============================================================
elif page == "method":

    page_header(
        "04 · METHOD",
        "From satellite signal to screening.",
        "BloomDetect uses the available chlorophyll-a observations and temporal "
        "context to create a machine-learning screening layer.",
    )

    steps = [
        ("01", "Satellite observation", "EOS-06 OCM-3 analysed chlorophyll-a observations form the input field."),
        ("02", "Data preparation", "Invalid values and duplicate/misdated source files were handled during preprocessing."),
        ("03", "Historical context", "Previous observations, historical baseline and recent statistics provide temporal context."),
        ("04", "Machine learning", "A Decision Tree is used for the current potential-risk screening prototype."),
        ("05", "Spatial screening", "Predictions are mapped back to their geographic observation cells."),
        ("06", "Investigation support", "Flagged cells can be inspected and downloaded for further environmental investigation."),
    ]

    for number, title, copy in steps:
        st.markdown(
            f"""
            <div class="info-card glass" style="display:flex;gap:24px;align-items:flex-start;">
              <div style="
                min-width:54px;height:54px;border-radius:18px;
                display:grid;place-items:center;
                background:rgba(104,223,224,.10);
                border:1px solid rgba(104,223,224,.16);
                color:#70e1df;font-weight:800;">
                {number}
              </div>
              <div>
                <div class="section-title">{title}</div>
                <div class="section-copy">{copy}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="quote" style="margin-top:30px;">
          Potential bloom risk is a model-derived screening signal.
          High chlorophyll-a alone does not prove a harmful algal bloom.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid" style="margin-top:22px;">
          <div class="feature-card glass">
            <div class="feature-title">What the system can say</div>
            <div class="feature-copy">
              It can identify observation cells whose chlorophyll-related temporal
              pattern matches the project's potential-risk screening target.
            </div>
          </div>
          <div class="feature-card glass">
            <div class="feature-title">What it cannot say</div>
            <div class="feature-copy">
              It cannot confirm a harmful species, toxin, health effect or
              field-verified HAB event from this dataset alone.
            </div>
          </div>
          <div class="feature-card glass">
            <div class="feature-title">Why validation matters</div>
            <div class="feature-copy">
              Flagged locations should be interpreted with appropriate
              oceanographic information and, where required, field observations.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DATA
# ============================================================
elif page == "data":

    page_header(
        "05 · DATA",
        "Know what you are looking at.",
        "A transparent view of the current BloomDetect dataset and what the present prototype does and does not contain.",
    )

    st.markdown(
        f"""
        <div class="metric-grid">
            {metric_card("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3")}
            {metric_card("Latest date", latest_date.strftime("%d %b %Y"), "in prediction file")}
            {metric_card("Grid cells", fmt_number(len(latest)), "latest observation")}
            {metric_card("Flagged", fmt_number(len(risk)), "potential-risk screening")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-grid">
          <div class="feature-card glass">
            <div class="feature-title">Global-ocean source</div>
            <div class="feature-copy">
              The source product is a global-ocean analysed chlorophyll-a product.
              Regional views in this app are geographic filters over the same data.
            </div>
          </div>
          <div class="feature-card glass">
            <div class="feature-title">Spatial resolution</div>
            <div class="feature-copy">
              The source product is documented at 0.25° × 0.25° resolution.
            </div>
          </div>
          <div class="feature-card glass">
            <div class="feature-title">Current website file</div>
            <div class="feature-copy">
              The website currently loads the latest prediction CSV. It does not
              pretend that the complete historical ML dataset is inside this file.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title" style="margin-top:40px;">Download the current screening table</div>',
        unsafe_allow_html=True,
    )

    download_df = latest.copy()
    csv_bytes = download_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download latest observations CSV",
        data=csv_bytes,
        file_name=f"bloomdetect_latest_{latest_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    risk_csv = risk.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download potential-risk locations",
        data=risk_csv,
        file_name=f"bloomdetect_potential_risk_{latest_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        <div class="info-card glass" style="margin-top:28px;">
          <div class="section-title">Interpretation</div>
          <div class="section-copy">
            This application is an early-warning support prototype. A potential-risk
            flag is not a confirmed harmful algal bloom. Satellite chlorophyll-a is
            an environmental indicator and should be interpreted with additional
            oceanographic or field evidence when decisions require confirmation.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    """
    <div class="footer">
      <b style="color:#b9d9dc;">BloomDetect AI</b><br>
      Satellite-based potential bloom-risk screening using EOS-06 OCM-3
      chlorophyll-a observations. Built as a research and decision-support prototype.
    </div>
    """,
    unsafe_allow_html=True,
)
