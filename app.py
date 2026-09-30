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

HERO_IMAGE = (
    "https://images.unsplash.com/photo-1717292742444-8a7c392da4dd"
    "?auto=format&fit=crop&fm=jpg&q=88&w=1900"
)
ALGAE_IMAGE = (
    "https://images.unsplash.com/photo-1717292742444-8a7c392da4dd"
    "?auto=format&fit=crop&fm=jpg&q=82&w=1100"
)

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
    # Plotly Express avoids the graph_objects API that caused the previous deployment error.
    fig = px.scatter_geo(
        observations,
        lat="latitude",
        lon="plot_lon",
        hover_name="risk_label",
        hover_data={
            "latitude": ":.2f",
            "plot_lon": ":.2f",
            "chla": ":.5f",
            "risk_label": True,
        },
        projection="natural earth",
    )
    fig.update_traces(
        marker=dict(size=3.2, color="#179bad", opacity=0.34),
        name="Observation grid",
        selector=dict(type="scattergeo"),
    )

    if risk_points is not None and len(risk_points):
        risk_fig = px.scatter_geo(
            risk_points,
            lat="latitude",
            lon="plot_lon",
            hover_name="risk_label",
            hover_data={
                "latitude": ":.2f",
                "plot_lon": ":.2f",
                "chla": ":.5f",
            },
            projection="natural earth",
        )
        risk_trace = risk_fig.data[0]
        risk_trace.marker = dict(
            size=9,
            color="#ef5f6d",
            opacity=0.95,
            line=dict(width=1.5, color="#ffffff"),
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
        countrycolor="rgba(35,82,91,.42)",
        showcoastlines=True,
        coastlinecolor="rgba(35,82,91,.55)",
        showland=True,
        landcolor="#edf5f2",
        showocean=True,
        oceancolor="#d9f4f6",
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
        **geo_update,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#143b43"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=0,
            bgcolor="rgba(255,255,255,.80)",
            bordercolor="rgba(10,120,130,.10)",
            borderwidth=1,
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
    f"""
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
.nav-wrap {{
    position:sticky;
    top:0;
    z-index:10000;
    margin:0 -46px;
    padding:12px 46px 10px;
    background:rgba(249,255,255,.86);
    border-bottom:1px solid rgba(16,107,116,.10);
    box-shadow:0 8px 30px rgba(17,87,99,.07);
    backdrop-filter:blur(24px);
    -webkit-backdrop-filter:blur(24px);
}}

.brand-line {{
    display:flex; align-items:center; gap:12px;
    margin-bottom:10px;
}}

.logo {{
    width:44px;height:44px;border-radius:15px;
    display:grid;place-items:center;
    background:linear-gradient(145deg,#39d1d0,#087e91);
    color:#fff;font-size:18px;font-weight:800;
    box-shadow:0 8px 24px rgba(7,140,153,.25);
}}

.brand {{font:800 1.14rem Manrope,sans-serif;color:#163942;letter-spacing:-.03em;}}
.tagline {{font-size:.72rem;color:#78949a;margin-left:56px;margin-top:-8px;margin-bottom:9px;}}

.nav-btn button {{
    min-height:48px !important;
    border-radius:15px !important;
    border:1px solid rgba(15,103,114,.13) !important;
    background:rgba(255,255,255,.84) !important;
    color:#315a62 !important;
    font-weight:700 !important;
    font-size:.84rem !important;
    box-shadow:0 5px 18px rgba(18,91,102,.05) !important;
    transition:transform .2s ease, background .2s ease, box-shadow .2s ease !important;
}}
.nav-btn button:hover {{
    transform:translateY(-2px) !important;
    background:#e7fbfb !important;
    color:#057e8b !important;
    box-shadow:0 12px 28px rgba(8,133,145,.12) !important;
}}

/* General */
.section-heading {{position:relative;z-index:2;padding:68px 0 28px;}}
.kicker {{font:800 .68rem Manrope,sans-serif;letter-spacing:.22em;text-transform:uppercase;color:#079ba9;margin-bottom:18px;}}
.section-heading h2 {{font:800 clamp(2.5rem,5vw,5.4rem)/.98 Manrope,sans-serif;letter-spacing:-.055em;color:#123a43;margin:0 0 20px;max-width:1050px;}}
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
.hero {{
    position:relative; z-index:2;
    min-height:690px;
    margin-top:26px;
    overflow:hidden;
    border-radius:38px;
    background:
      linear-gradient(90deg,rgba(2,55,65,.86),rgba(4,112,121,.45) 55%,rgba(0,38,47,.14)),
      url('{HERO_IMAGE}') center/cover;
    box-shadow:0 30px 100px rgba(5,77,87,.22);
}}
.hero::before {{
    content:"";position:absolute;inset:-20%;
    background:radial-gradient(ellipse at 65% 25%,rgba(117,239,240,.28),transparent 32%);
    animation:waterGlow 7s ease-in-out infinite alternate;
}}
.hero::after {{
    content:"";position:absolute;left:-5%;right:-5%;bottom:-110px;height:250px;
    background:rgba(101,229,229,.17);
    border-radius:50% 50% 0 0;
    animation:wave 8s ease-in-out infinite alternate;
}}
@keyframes waterGlow {{from{{transform:scale(.96);opacity:.55}}to{{transform:scale(1.08);opacity:1}}}}
@keyframes wave {{from{{transform:translateX(-2%) rotate(-1deg)}}to{{transform:translateX(2%) rotate(1deg)}}}}
.hero-content {{position:relative;z-index:3;padding:105px 82px 150px;max-width:900px;}}
.hero-kicker {{font:800 .7rem Manrope,sans-serif;letter-spacing:.25em;color:#8af0ee;margin-bottom:28px;}}
.hero h1 {{font:800 clamp(4rem,8vw,8.6rem)/.88 Manrope,sans-serif;color:#70efed;letter-spacing:-.075em;margin:0 0 28px;}}
.hero p {{font:500 1.1rem/1.75 'DM Sans',sans-serif;color:rgba(242,255,255,.88);max-width:760px;}}
.hero-pill {{display:inline-block;margin-top:22px;padding:12px 17px;border:1px solid rgba(161,249,248,.28);background:rgba(4,57,66,.32);border-radius:999px;color:#e9ffff;font-size:.8rem;backdrop-filter:blur(10px);}}

.scroll-cue {{position:absolute;z-index:4;right:44px;bottom:34px;color:#dffefe;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;animation:cue 2s ease-in-out infinite;}}
@keyframes cue {{0%,100%{{transform:translateY(0);opacity:.55}}50%{{transform:translateY(8px);opacity:1}}}}

/* Home educational cards */
.algae-image {{width:100%;height:340px;object-fit:cover;border-radius:28px;border:1px solid rgba(9,103,115,.12);box-shadow:var(--shadow);}}
.process {{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}}
.process-item {{background:rgba(255,255,255,.66);border:1px solid var(--line);border-radius:22px;padding:23px;box-shadow:0 16px 50px rgba(19,93,104,.08);}}
.process-item strong {{display:block;font:800 1.1rem Manrope;color:#123b43;margin-bottom:8px;}}
.process-item span {{font-size:.9rem;color:#708a90;line-height:1.6;}}

/* Map / charts */
.control-card {{padding:24px;border-radius:25px;background:rgba(255,255,255,.70);border:1px solid var(--line);box-shadow:var(--shadow);}}
.download-box {{padding:32px;border-radius:28px;background:linear-gradient(135deg,#087f90,#19aab0);box-shadow:0 24px 65px rgba(8,126,141,.20);color:white;}}
.download-box h3 {{font:800 1.6rem Manrope;margin:0 0 7px;color:white;}}
.download-box p {{color:rgba(255,255,255,.80);margin:0 0 18px;}}
.download-box button {{background:white!important;color:#087d8b!important;border:none!important;font-weight:800!important;}}
.quote {{position:relative;z-index:2;margin:35px 0;padding:25px 30px;border-left:4px solid #1ab0b6;background:rgba(222,250,250,.65);border-radius:0 20px 20px 0;color:#42666d;line-height:1.7;}}
.footer {{position:relative;z-index:2;border-top:1px solid var(--line);padding:25px 0;color:#78939a;font-size:.76rem;margin-top:70px;}}

/* Streamlit controls */
div[data-baseweb="select"] > div {{border-radius:13px!important;background:rgba(255,255,255,.88)!important;border-color:rgba(13,107,117,.13)!important;}}
.stNumberInput input {{border-radius:13px!important;background:rgba(255,255,255,.88)!important;}}
.stButton button {{border-radius:14px!important;min-height:48px!important;font-weight:700!important;}}
.stDownloadButton button {{border-radius:15px!important;min-height:52px!important;font-weight:800!important;}}
[data-testid="stDataFrame"] {{border-radius:18px;overflow:hidden;}}

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
        f"""
        <section class="hero">
            <div class="hero-content">
                <div class="hero-kicker">EOS-06 · OCM-3 · E06OCM_L4_AC</div>
                <h1>Reading the ocean<br>from space.</h1>
                <p>BloomDetect AI turns satellite-derived chlorophyll-a observations into a spatial screening experience for locations showing patterns associated with potential bloom risk.</p>
                <span class="hero-pill">Latest available observation · {latest_date.strftime('%d %B %Y')}</span>
            </div>
            <div class="scroll-cue">scroll · explore the ocean ↓</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    section_title(
        "Start here",
        "What is an algal bloom?",
        "Algae and phytoplankton are normal and essential parts of aquatic ecosystems. A bloom happens when their population increases rapidly and becomes unusually concentrated. Not every bloom is harmful.",
    )

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.markdown(
            """
            <div class="glass-card" style="padding:32px">
                <div class="kicker">The simple version</div>
                <h3 style="font:800 2rem Manrope;color:#143b43;margin:0 0 14px">Tiny organisms. Big environmental signal.</h3>
                <p style="font-size:1rem;line-height:1.8;color:#69838a">Phytoplankton use sunlight and nutrients to grow. When conditions allow many cells to accumulate in one area, the water can show a strong chlorophyll signal. Some blooms are beneficial, while a smaller subset can become harmful.</p>
                <p style="font-size:.92rem;line-height:1.7;color:#69838a">BloomDetect uses chlorophyll-a as an observation signal. It does not identify toxin-producing species or confirm that a bloom is harmful.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.image(ALGAE_IMAGE, caption="Underwater algae / seaweed environment", use_container_width=True)

    section_title(
        "How a bloom can develop",
        "Sunlight + nutrients + water movement → growth.",
        "The exact causes vary by ecosystem. NOAA notes that sunlight, nutrients, water movement and other environmental conditions can contribute to algal growth and harmful blooms.",
    )
    st.markdown(
        """
        <div class="process">
            <div class="process-item"><strong>01 · Growth conditions</strong><span>Light and nutrients support phytoplankton growth.</span></div>
            <div class="process-item"><strong>02 · Accumulation</strong><span>Water movement can concentrate cells in particular areas.</span></div>
            <div class="process-item"><strong>03 · Visible signal</strong><span>Higher phytoplankton biomass can contribute to changes in chlorophyll-a observed from space.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title(
        "What BloomDetect adds",
        "From a huge ocean field to places worth investigating.",
        "The current prototype screens the available observation grid and highlights model-derived potential-risk locations for further investigation.",
    )
    stats = st.columns(4, gap="medium")
    with stats[0]: metric_card("Observation cells", f"{len(latest):,}", "latest available grid")
    with stats[1]: metric_card("Potential-risk cells", f"{len(risk):,}", "screening output")
    with stats[2]: metric_card("Normal cells", f"{len(normal):,}", "remaining latest grid")
    with stats[3]: metric_card("Risk share", f"{len(risk)/len(latest)*100:.2f}%", "latest field")

    section_title(
        "Who is this for?",
        "Built for people who need to look closer.",
        "The dashboard is a screening and exploration tool, not a replacement for field measurements.",
    )
    users = st.columns(4, gap="medium")
    user_cards = [
        ("🎣", "Fisheries & monitoring", "Explore areas that may deserve additional environmental attention."),
        ("🔬", "Researchers", "Inspect spatial chlorophyll patterns and download the current screening table."),
        ("🏛️", "Environmental teams", "Use a large-area screening view to help prioritize follow-up investigation."),
        ("🎓", "Students & educators", "Explore how satellite data and machine learning can support an ocean application."),
    ]
    for col, (icon, title, copy) in zip(users, user_cards):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.info("Important: a potential bloom-risk flag is a model-derived screening signal. High chlorophyll-a alone does not prove a harmful algal bloom.")

# ============================================================
# MAP
# ============================================================
elif page == "map":
    section_title(
        "01 · Spatial explorer",
        "See the ocean field.",
        "Start with the full latest observation grid. Potential-risk locations can be highlighted on top of it.",
    )

    c1, c2, c3 = st.columns([1.2, 1.1, .8], gap="medium")
    with c1:
        region = st.selectbox(
            "Geographic view",
            ["Global Ocean", "Indian Ocean", "Arabian Sea", "Bay of Bengal"],
            key="map_region",
        )
    with c2:
        highlight = st.toggle("Highlight potential-risk locations", value=True, key="risk_overlay")
    with c3:
        metric_card("Latest cells", f"{len(latest):,}", "full available grid")

    visible = latest[region_mask(latest, region)].copy()
    visible_risk = risk[region_mask(risk, region)].copy() if highlight else pd.DataFrame()

    st.markdown(
        f'<div class="glass-card" style="padding:18px 24px;margin:18px 0"><b>{region}</b> · {len(visible):,} observation cells · {len(visible_risk):,} potential-risk cells highlighted</div>',
        unsafe_allow_html=True,
    )

    fig = map_figure(visible, visible_risk, region=region)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": False})

    st.markdown('<div class="quote">The map shows the available observation field first. Red points are a model-derived overlay and should be treated as locations for further investigation, not confirmed HAB occurrences.</div>', unsafe_allow_html=True)

# ============================================================
# LOCATION
# ============================================================
elif page == "location":
    section_title(
        "02 · Location explorer",
        "Look closer at one place.",
        "Enter coordinates to find the nearest available observation in the current processed dataset.",
    )

    left, right = st.columns([.85, 1.35], gap="large")
    with left:
        lat_in = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=15.0, step=0.25, format="%.2f")
        lon_in = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=75.0, step=0.25, format="%.2f")
        inspect = st.button("Inspect nearest observation →", use_container_width=True, key="inspect_location")

    # Show the default nearest point immediately, then update when inspected.
    lon_norm = ((lon_in + 180) % 360) - 180
    dist = np.sqrt((latest["latitude"] - lat_in) ** 2 + (latest["plot_lon"] - lon_norm) ** 2)
    row = latest.loc[dist.idxmin()]

    with right:
        st.markdown(
            f"""
            <div class="glass-card" style="padding:32px">
                <div class="kicker">Nearest grid cell</div>
                <h3 style="font:800 2.1rem Manrope;color:#143b43;margin:0 0 16px">{row['latitude']:.2f}° · {row['plot_lon']:.2f}°</h3>
                <p style="color:#67828a">Observation date: <b>{row['date'].strftime('%d %B %Y')}</b></p>
                <p style="color:#67828a">Chlorophyll-a: <b>{row['chla']:.5f}</b></p>
                <p style="color:#67828a">Screening result: <b>{row['risk_label']}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_title(
        "Available signals",
        "What is actually present for this grid cell?",
        "These values come directly from the processed prediction table. Missing fields are shown as unavailable rather than invented.",
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
            text = "Not available" if value is None else f"{float(value):.5f}"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.45rem">{text}</div><div class="metric-note">processed dataset</div></div>',
                unsafe_allow_html=True,
            )

# ============================================================
# INSIGHTS
# ============================================================
elif page == "insights":
    section_title(
        "03 · Insights",
        "Useful signals from the latest field.",
        "Only summaries that can be calculated directly from the current observation and screening table are shown here.",
    )

    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Observation cells", f"{len(latest):,}", "latest grid")
    with cols[1]: metric_card("Potential-risk cells", f"{len(risk):,}", "model screening")
    with cols[2]: metric_card("Normal cells", f"{len(normal):,}", "remaining grid")
    with cols[3]: metric_card("Risk share", f"{len(risk)/len(latest)*100:.2f}%", "latest field")

    # 1. Classification composition: genuinely useful.
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-heading"><div class="kicker">Screening composition</div><h2>How the latest grid is classified.</h2><p>Normal observations versus model-flagged potential-risk observations.</p></div>', unsafe_allow_html=True)
        composition = pd.DataFrame({"Classification": ["Normal", "Potential bloom risk"], "Cells": [len(normal), len(risk)]})
        fig1 = px.bar(composition, x="Classification", y="Cells", text="Cells", color="Classification", color_discrete_map={"Normal":"#8ecfd3", "Potential bloom risk":"#ef6471"})
        fig1.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig1.update_layout(height=400, showlegend=False, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", font_color="#183f47")
        st.plotly_chart(fig1, width="stretch", config={"displaylogo": False})

    with c2:
        st.markdown('<div class="section-heading"><div class="kicker">Chlorophyll field</div><h2>Where does chlorophyll sit?</h2><p>The distribution describes the current satellite-derived chlorophyll-a field. It is not a direct measure of harmfulness.</p></div>', unsafe_allow_html=True)
        clean_chla = latest["chla"].replace([np.inf, -np.inf], np.nan).dropna()
        # Remove only extreme non-finite values; keep zeros because they are part of the processed field.
        fig2 = px.histogram(clean_chla.to_frame(name="Chlorophyll-a"), x="Chlorophyll-a", nbins=45, color_discrete_sequence=["#1ea5b1"])
        fig2.update_layout(height=400, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", font_color="#183f47", yaxis_title="Observation cells")
        st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})

    # 2. Region counts: useful, but explicitly geographic views of the same dataset.
    region_rows = []
    for name in ["Indian Ocean", "Arabian Sea", "Bay of Bengal"]:
        m = region_mask(risk, name)
        region_rows.append({"Geographic view": name, "Potential-risk cells": int(m.sum())})
    region_df = pd.DataFrame(region_rows)
    st.markdown('<div class="section-heading"><div class="kicker">Geographic concentration</div><h2>Where are the flagged cells within common study windows?</h2><p>These are geographic filters over the same global-ocean source, not separate datasets.</p></div>', unsafe_allow_html=True)
    fig3 = px.bar(region_df, x="Geographic view", y="Potential-risk cells", text="Potential-risk cells", color_discrete_sequence=["#1499a8"])
    fig3.update_traces(texttemplate="%{text}", textposition="outside")
    fig3.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", font_color="#183f47", showlegend=False)
    st.plotly_chart(fig3, width="stretch", config={"displaylogo": False})

    # 3. Investigation queue: useful action list.
    st.markdown('<div class="section-heading"><div class="kicker">Investigation queue</div><h2>Potential-risk locations.</h2><p>A compact list for follow-up inspection. It does not rank confirmed HAB severity.</p></div>', unsafe_allow_html=True)
    queue = risk.sort_values("chla", ascending=False).head(25).copy()
    show_cols = [c for c in ["latitude", "longitude", "chla", "risk_label"] if c in queue.columns]
    if len(queue):
        st.dataframe(
            queue[show_cols].rename(columns={"latitude":"Latitude", "longitude":"Longitude", "chla":"Chlorophyll-a", "risk_label":"Screening"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No potential-risk cells are available in the latest field.")

    st.markdown('<div class="download-box"><h3>Take the screening list with you.</h3><p>Download the current flagged locations as a CSV for further investigation.</p></div>', unsafe_allow_html=True)
    st.download_button(
        "⬇ Download potential-risk locations",
        data=risk.to_csv(index=False).encode("utf-8"),
        file_name=f"bloomdetect_screening_{latest_date.strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="insights_download",
    )

# ============================================================
# METHOD
# ============================================================
elif page == "method":
    section_title(
        "04 · How it works",
        "From satellite signal to investigation support.",
        "A transparent six-step pipeline, without pretending the model knows more than the data can tell us.",
    )

    steps = [
        ("01", "Satellite observation", "EOS-06 OCM-3 provides analysed chlorophyll-a observations across the global ocean."),
        ("02", "Data preparation", "The source observations are cleaned, standardized and organized into an analysis-ready grid."),
        ("03", "Historical context", "Previous observations and historical context are used to describe changes in chlorophyll-a."),
        ("04", "Machine learning", "A Decision Tree screens locations against the project's proxy potential-risk target."),
        ("05", "Spatial screening", "The model output is placed back onto the geographic grid for visual exploration."),
        ("06", "Further validation", "Flagged locations can be prioritized for oceanographic and field-based investigation."),
    ]
    for num, title, copy in steps:
        st.markdown(
            f'<div class="glass-card" style="padding:28px;margin:13px 0;display:flex;gap:24px;align-items:flex-start"><div style="font:800 1.8rem Manrope;color:#079ba9;min-width:58px">{num}</div><div><h3 style="margin:0 0 7px;font:800 1.25rem Manrope;color:#163c44">{title}</h3><p style="margin:0;color:#6a858c;line-height:1.7">{copy}</p></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="quote">Potential bloom risk is a model-derived screening signal. High chlorophyll-a alone does not prove a harmful algal bloom. The current prototype does not identify toxin-producing species or replace field validation.</div>', unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
elif page == "data":
    section_title(
        "05 · Data",
        "The dataset behind the experience.",
        "This page is deliberately different from Home: it is a compact data sheet and download area, not another explanation of algae or the project.",
    )

    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Product", "OCM-3", "EOS-06")
    with cols[1]: metric_card("Latest cells", f"{len(latest):,}", "available observation grid")
    with cols[2]: metric_card("Grid spacing", "0.25°", "source product")
    with cols[3]: metric_card("Latest date", latest_date.strftime("%d %b %Y"), "processed data")

    st.markdown('<div class="section-heading"><div class="kicker">Data dictionary</div><h2>What is actually in the file?</h2><p>The public dashboard uses the latest processed prediction table.</p></div>', unsafe_allow_html=True)
    dictionary = pd.DataFrame([
        ["latitude", "Grid latitude", "degrees"],
        ["longitude", "Grid longitude", "degrees"],
        ["date", "Observation date", "date"],
        ["chla", "Satellite-derived chlorophyll-a", "product value"],
        ["risk_label", "Model screening result", "Normal / Potential Bloom Risk"],
        ["previous_chla", "Previous observation", "derived feature when present"],
        ["historical_baseline", "Historical baseline", "derived feature when present"],
        ["recent_mean", "Recent mean", "derived feature when present"],
        ["recent_max", "Recent maximum", "derived feature when present"],
        ["chla_anomaly", "Chlorophyll anomaly", "derived feature when present"],
        ["chla_change", "Change from previous observation", "derived feature when present"],
    ], columns=["Field", "Meaning", "Type"])
    st.dataframe(dictionary, width="stretch", hide_index=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="download-box"><h3>Download the data.</h3><p>Choose the complete latest observation table or the smaller screening list.</p></div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="large")
    with d1:
        st.download_button(
            "⬇ Download latest observations",
            data=latest.to_csv(index=False).encode("utf-8"),
            file_name=f"bloomdetect_latest_{latest_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="data_latest_download",
        )
    with d2:
        st.download_button(
            "⬇ Download screening locations",
            data=risk.to_csv(index=False).encode("utf-8"),
            file_name=f"bloomdetect_risk_{latest_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="data_risk_download",
        )

    st.markdown('<div class="quote">Source context: EOS-06 / OCM-3 analysed chlorophyll product. Regional views in the app are geographic filters over the same global-ocean source. The current dashboard is not a live real-time monitoring feed.</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    f"<div class='footer'>BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Current data through {latest_date.strftime('%d %B %Y')}</div>",
    unsafe_allow_html=True,
)
