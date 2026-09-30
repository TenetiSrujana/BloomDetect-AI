from pathlib import Path
import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI — OCEAN INTERACTIVE FINAL
# Single Streamlit app. Navigation stays inside the same app.
# Current source: latest_bloom_risk_predictions.csv
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
    "?auto=format&fit=crop&fm=jpg&q=86&w=1900"
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
    for col in ["latitude", "longitude", "chla"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "date"]).copy()
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
risk = latest[
    latest["risk_label"].astype(str).str.lower().eq("potential bloom risk")
].copy()

# -----------------------------
# STATE NAVIGATION
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

if "page" not in st.session_state or st.session_state.page not in PAGES:
    st.session_state.page = "home"


def go(page):
    st.session_state.page = page


# -----------------------------
# HELPERS
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


def make_map(frame, risk_frame=None, title=None, height=620):
    fig = go.Figure()

    if len(frame):
        hover = [
            frame["latitude"].round(3),
            frame["longitude"].round(3),
            frame["chla"].round(5),
            frame["risk_label"].astype(str),
        ]
        fig.add_trace(
            go.Scattergeo(
                lon=frame["plot_lon"],
                lat=frame["latitude"],
                mode="markers",
                name="Observation grid",
                marker=dict(size=3.2, color="rgba(27,150,176,0.42)"),
                customdata=np.column_stack(hover),
                hovertemplate=(
                    "<b>Ocean observation</b><br>"
                    "Latitude: %{customdata[0]}°<br>"
                    "Longitude: %{customdata[1]}°<br>"
                    "Chlorophyll-a: %{customdata[2]}<br>"
                    "Screening: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    if risk_frame is not None and len(risk_frame):
        fig.add_trace(
            go.Scattergeo(
                lon=risk_frame["plot_lon"],
                lat=risk_frame["latitude"],
                mode="markers",
                name="Potential bloom risk",
                marker=dict(
                    size=8.5,
                    color="#ef6471",
                    line=dict(width=1.5, color="#ffffff"),
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
                    "Latitude: %{customdata[0]}°<br>"
                    "Longitude: %{customdata[1]}°<br>"
                    "Chlorophyll-a: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    fig.update_geos(
        showcountries=True,
        countrycolor="rgba(70,110,120,0.55)",
        showcoastlines=True,
        coastlinecolor="rgba(40,90,100,0.7)",
        showland=True,
        landcolor="#eef4f1",
        showocean=True,
        oceancolor="#d9f3f4",
        bgcolor="rgba(0,0,0,0)",
        showlakes=False,
    )
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=35 if title else 0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#18343b"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=0,
            bgcolor="rgba(255,255,255,.78)",
            bordercolor="rgba(30,100,110,.10)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#65cdd0",
            font_color="#16343a",
        ),
    )
    return fig


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
# GLOBAL WHITE OCEAN UI
# -----------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

:root {{
    --ink:#12333b;
    --muted:#647e84;
    --aqua:#079aa7;
    --aqua2:#53d5d4;
    --sky:#dff7f8;
    --sea:#0d8290;
    --line:rgba(16,87,96,.12);
    --glass:rgba(255,255,255,.70);
    --shadow:0 24px 70px rgba(22,91,100,.12);
}}

html, body {{
    background:#f4fbfb!important;
    scroll-behavior:smooth;
}}

.stApp {{
    background:
      radial-gradient(circle at 8% 4%, rgba(80,211,214,.24), transparent 22%),
      radial-gradient(circle at 92% 15%, rgba(115,211,245,.20), transparent 24%),
      linear-gradient(180deg,#f8ffff 0%,#eefafb 48%,#f8ffff 100%);
    color:var(--ink);
}}

/* Remove Streamlit chrome that was blocking the nav. */
header[data-testid="stHeader"] {{ display:none!important; }}
[data-testid="stToolbar"] {{ display:none!important; }}
#MainMenu {{ display:none!important; }}
footer {{ display:none!important; }}
section[data-testid="stSidebar"] {{ display:none!important; }}

.block-container {{
    max-width:1450px;
    padding:0 42px 70px!important;
}}

* {{ box-sizing:border-box; }}
a {{ text-decoration:none!important; }}

/* NAV */
.nav-shell {{
    position:sticky;
    top:0;
    z-index:9999;
    margin:0 -42px;
    padding:14px 42px;
    background:rgba(248,255,255,.88);
    border-bottom:1px solid rgba(25,110,120,.10);
    box-shadow:0 8px 30px rgba(26,90,100,.07);
    backdrop-filter:blur(22px);
    -webkit-backdrop-filter:blur(22px);
}}

.brand-row {{
    display:flex;
    align-items:center;
    gap:11px;
    margin-bottom:10px;
}}

.brand-mark {{
    width:40px;height:40px;border-radius:13px;
    display:grid;place-items:center;
    color:white;font-size:18px;font-weight:800;
    background:linear-gradient(145deg,#2bc6c8,#057f90);
    box-shadow:0 8px 25px rgba(8,151,163,.25);
}}

.brand-name {{
    font-family:Manrope,sans-serif;
    font-size:1.08rem;font-weight:800;letter-spacing:-.03em;
    color:#14353c;
}}

.nav-caption {{
    color:#769197;font-size:.72rem;margin-left:51px;margin-top:-8px;
}}

.nav-btn {{
    width:100%!important;
    min-height:48px!important;
    border-radius:14px!important;
    border:1px solid rgba(22,111,120,.11)!important;
    background:rgba(255,255,255,.72)!important;
    color:#41636a!important;
    font-size:.82rem!important;
    font-weight:700!important;
    box-shadow:0 4px 15px rgba(25,100,110,.04)!important;
    transition:all .2s ease!important;
}}
.nav-btn:hover {{
    background:#e4fafa!important;
    color:#067f8b!important;
    border-color:rgba(8,151,163,.30)!important;
    transform:translateY(-1px);
}}

/* HOME HERO */
.hero {{
    min-height:720px;
    margin-top:25px;
    position:relative;
    overflow:hidden;
    border-radius:36px;
    border:1px solid rgba(16,107,116,.12);
    background:
      linear-gradient(90deg,rgba(3,42,50,.90) 0%,rgba(5,91,101,.58) 48%,rgba(0,33,41,.18) 100%),
      url('{HERO_IMAGE}') center/cover;
    box-shadow:var(--shadow);
}}

.hero::before {{
    content:"";position:absolute;inset:0;
    background:
      radial-gradient(circle at 73% 25%,rgba(154,255,247,.25),transparent 14%),
      radial-gradient(circle at 88% 65%,rgba(55,215,225,.24),transparent 22%);
    animation:glow 7s ease-in-out infinite alternate;
}}

@keyframes glow {{ from {{opacity:.55}} to {{opacity:1}} }}

.wave {{
    position:absolute;left:-8%;bottom:-105px;width:116%;height:250px;
    background:rgba(222,252,252,.20);
    border-radius:50% 50% 0 0;
    filter:blur(1px);
    animation:waveMove 7s ease-in-out infinite;
}}
.wave.two {{
    bottom:-145px;opacity:.35;height:280px;
    animation-duration:10s;animation-delay:-3s;
}}
@keyframes waveMove {{
    0%,100% {{ transform:translateX(-2%) rotate(-1deg); }}
    50% {{ transform:translateX(2%) rotate(1deg); }}
}}

.bubble {{
    position:absolute;border-radius:50%;
    border:1px solid rgba(255,255,255,.55);
    background:rgba(255,255,255,.09);
    box-shadow:0 0 18px rgba(160,255,255,.18);
    animation:bubbleUp 8s linear infinite;
}}
.b1{{width:14px;height:14px;right:16%;bottom:20%;animation-delay:-1s}}
.b2{{width:8px;height:8px;right:27%;bottom:15%;animation-delay:-4s}}
.b3{{width:22px;height:22px;right:8%;bottom:26%;animation-delay:-6s}}
.b4{{width:10px;height:10px;right:38%;bottom:12%;animation-delay:-2s}}
.b5{{width:16px;height:16px;right:47%;bottom:19%;animation-delay:-5s}}
@keyframes bubbleUp {{
    0%{{transform:translateY(80px) scale(.8);opacity:0}}
    15%{{opacity:.7}}
    100%{{transform:translateY(-430px) translateX(25px) scale(1.1);opacity:0}}
}}

.hero-content {{
    position:relative;z-index:5;
    padding:100px 8%;max-width:900px;
}}
.hero-kicker {{
    color:#b8ffff;text-transform:uppercase;letter-spacing:.22em;
    font-size:.72rem;font-weight:800;
}}
.hero h1 {{
    font-family:Manrope,sans-serif;color:#fff;
    font-size:clamp(3.3rem,7vw,7rem);
    line-height:.91;letter-spacing:-.065em;
    margin:20px 0 25px;
}}
.hero h1 span {{color:#6ef2ec}}
.hero p {{
    max-width:690px;color:rgba(240,255,255,.84);
    line-height:1.8;font-size:1.05rem;
}}
.hero-chip {{
    display:inline-block;margin-top:25px;padding:9px 14px;
    border:1px solid rgba(255,255,255,.24);
    background:rgba(255,255,255,.10);border-radius:999px;
    color:#e8ffff;font-size:.75rem;
    backdrop-filter:blur(14px);
}}

/* GENERAL */
.section-heading {{ padding:75px 0 28px;max-width:900px; }}
.kicker {{color:var(--aqua);text-transform:uppercase;letter-spacing:.2em;font-size:.70rem;font-weight:800;}}
.section-heading h2 {{font-family:Manrope,sans-serif;color:#12343b;font-size:clamp(2.2rem,5vw,4.5rem);line-height:1;letter-spacing:-.055em;margin:10px 0 14px;}}
.section-heading p {{color:var(--muted);font-size:1rem;line-height:1.75;max-width:780px;}}

.glass-card {{
    background:rgba(255,255,255,.68);
    border:1px solid rgba(20,105,115,.11);
    border-radius:26px;
    box-shadow:0 20px 55px rgba(29,98,108,.08);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
}}

.metric-card {{
    min-height:145px;padding:24px;border-radius:22px;
    background:rgba(255,255,255,.72);
    border:1px solid rgba(20,105,115,.10);
    box-shadow:0 16px 45px rgba(29,98,108,.07);
}}
.metric-label {{font-size:.69rem;text-transform:uppercase;letter-spacing:.13em;color:#789096;font-weight:800;}}
.metric-value {{font-family:Manrope,sans-serif;font-size:2.15rem;color:#14383f;font-weight:800;margin-top:9px;}}
.metric-note {{font-size:.75rem;color:#789096;margin-top:4px;}}

.feature-card {{
    padding:28px;min-height:205px;border-radius:25px;
    background:rgba(255,255,255,.73);
    border:1px solid rgba(20,105,115,.10);
    box-shadow:0 16px 45px rgba(29,98,108,.06);
    transition:.25s ease;
}}
.feature-card:hover {{transform:translateY(-6px);box-shadow:0 24px 55px rgba(20,110,120,.13);border-color:rgba(8,155,165,.28)}}
.feature-icon {{font-size:1.5rem;margin-bottom:18px;}}
.feature-card h3 {{font-family:Manrope,sans-serif;color:#173940;font-size:1.12rem;margin:0 0 8px;}}
.feature-card p {{color:#71878d;font-size:.86rem;line-height:1.65;margin:0;}}

.quote {{
    margin:25px 0;padding:30px;border-radius:22px;
    background:linear-gradient(135deg,#e8fbfb,#f8ffff);
    border:1px solid rgba(8,155,165,.12);
    color:#22474e;font-family:Manrope,sans-serif;font-size:1.2rem;line-height:1.55;
}}

.footer {{margin-top:80px;padding:30px 0;border-top:1px solid rgba(20,105,115,.10);color:#71888e;font-size:.75rem;}}

/* Streamlit controls */
.stButton>button,.stDownloadButton>button {{
    min-height:46px!important;border-radius:14px!important;
    border:1px solid rgba(22,111,120,.12)!important;
    background:rgba(255,255,255,.82)!important;
    color:#31565e!important;font-weight:700!important;
    box-shadow:0 6px 20px rgba(25,100,110,.05)!important;
}}
.stButton>button:hover,.stDownloadButton>button:hover {{
    background:#e8fbfb!important;color:#087d88!important;
    border-color:rgba(8,155,165,.32)!important;
}}
div[data-baseweb="select"]>div {{
    background:rgba(255,255,255,.85)!important;
    border-color:rgba(20,105,115,.13)!important;
    border-radius:14px!important;
}}
.stNumberInput input {{background:rgba(255,255,255,.86)!important;color:#173940!important;border-radius:12px!important;}}
[data-testid="stMetric"] {{background:rgba(255,255,255,.72);border:1px solid rgba(20,105,115,.10);padding:16px;border-radius:18px;}}
.stDataFrame {{border:1px solid rgba(20,105,115,.10);border-radius:16px;}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# TOP NAVIGATION
# -----------------------------
st.markdown('<div class="nav-shell"><div class="brand-row"><div class="brand-mark">≈</div><div class="brand-name">BloomDetect AI</div></div><div class="nav-caption">Satellite-based potential bloom-risk screening</div></div>', unsafe_allow_html=True)

nav_cols = st.columns([1.05, 1.25, 1.25, 1.15, 1.35, 1.0], gap="small")
for col, key in zip(nav_cols, PAGES):
    with col:
        if st.button(LABELS[key], key=f"nav_{key}", use_container_width=True):
            go(key)
            st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

page = st.session_state.page

# ============================================================
# HOME
# ============================================================
if page == "home":
    st.markdown(
        f"""
        <div class="hero">
            <div class="bubble b1"></div><div class="bubble b2"></div>
            <div class="bubble b3"></div><div class="bubble b4"></div><div class="bubble b5"></div>
            <div class="wave"></div><div class="wave two"></div>
            <div class="hero-content">
                <div class="hero-kicker">EOS-06 · OCM-3 · E06OCM_L4_AC</div>
                <h1>Read the <span>ocean</span><br>from space.</h1>
                <p>
                    BloomDetect AI turns satellite-derived chlorophyll-a observations
                    into a spatial screening experience for locations showing patterns
                    associated with potential bloom risk.
                </p>
                <div class="hero-chip">Latest available observation · {latest_date.strftime('%d %B %Y')}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading"><div class="kicker">Explore</div><h2>Choose your way into the ocean.</h2><p>Each section is a focused tool. No endless dashboard scrolling, no mystery buttons.</p></div>', unsafe_allow_html=True)

    # Home feature buttons: each opens a section in the same app.
    feature_cols = st.columns(3, gap="large")
    feature_data = [
        ("🗺️", "Ocean Map", "Explore the complete latest observation grid and highlight potential-risk locations.", "map"),
        ("📍", "Location Explorer", "Enter coordinates and inspect the nearest available satellite observation.", "location"),
        ("📊", "Insights", "See the current observation picture, flagged locations and useful screening statistics.", "insights"),
        ("🧠", "How It Works", "Understand exactly how satellite data becomes an AI-assisted screening signal.", "method"),
        ("🌐", "Data", "See the source, coverage, resolution, latest date and available downloads.", "data"),
        ("🌊", "Start exploring", "Jump into the map and see what the latest ocean observation field looks like.", "map"),
    ]
    for i, (icon, title, copy, target) in enumerate(feature_data):
        with feature_cols[i % 3]:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)
            if st.button(f"Open {title}  →", key=f"home_{i}", use_container_width=True):
                go(target)
                st.rerun()

    st.markdown('<div class="section-heading"><div class="kicker">At a glance</div><h2>The field we actually observe.</h2><p>The website keeps the complete latest observation field visible. The model-generated potential-risk locations are an overlay, not a replacement for the observations.</p></div>', unsafe_allow_html=True)

    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Latest grid cells", f"{len(latest):,}", "available observations")
    with cols[1]: metric_card("Potential-risk cells", f"{len(risk):,}", "model-screened locations")
    with cols[2]: metric_card("Risk share", f"{len(risk)/len(latest)*100:.2f}%", "of latest observation grid")
    with cols[3]: metric_card("Observation date", latest_date.strftime("%d %b %Y"), "latest available")

    st.markdown('<div class="quote">A satellite signal is the beginning of an investigation, not the conclusion.</div>', unsafe_allow_html=True)

# ============================================================
# MAP
# ============================================================
elif page == "map":
    section_title("01 · Spatial explorer", "See the ocean field.", "Start with the full latest observation grid. Potential-risk locations can be highlighted on top of it.")

    controls = st.columns([1.4, 1.4, 1.1], gap="large")
    with controls[0]:
        region = st.selectbox("Geographic view", ["Global Ocean", "Indian Ocean", "Arabian Sea", "Bay of Bengal"], key="map_region")
    with controls[1]:
        highlight = st.toggle("Highlight potential-risk locations", value=True, key="map_highlight")
    with controls[2]:
        metric_card("Latest cells", f"{len(latest):,}", "full available grid")

    mask = region_mask(latest, region)
    visible = latest[mask].copy()
    visible_risk = risk[region_mask(risk, region)].copy()

    st.markdown(f'<div class="glass-card" style="padding:18px 22px;margin:18px 0;color:#58747a"><b>{region}</b> · {len(visible):,} observation cells · {len(visible_risk):,} potential-risk cells available in this geographic view.</div>', unsafe_allow_html=True)
    st.plotly_chart(make_map(visible, visible_risk if highlight else None, height=670), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    st.markdown('<div class="section-heading"><div class="kicker">Read the map</div><h2>What you are seeing.</h2><p>Blue points represent the available observation grid. Coral points are the locations flagged by the current model-derived screening output.</p></div>', unsafe_allow_html=True)

    a, b, c = st.columns(3, gap="large")
    with a:
        st.markdown('<div class="feature-card"><div class="feature-icon">🔵</div><h3>Observation</h3><p>A location represented in the latest processed satellite grid.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="feature-card"><div class="feature-icon">🔴</div><h3>Potential-risk flag</h3><p>A model-derived screening signal that deserves further investigation.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="feature-card"><div class="feature-icon">🌍</div><h3>Geographic view</h3><p>Regional selections are filters over the same global-ocean source, not separate datasets.</p></div>', unsafe_allow_html=True)

# ============================================================
# LOCATION
# ============================================================
elif page == "location":
    section_title("02 · Location explorer", "Look closer at one place.", "Enter coordinates to find the nearest available observation in the current dataset.")

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        lat_in = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=15.0, step=0.25, format="%.2f")
        lon_in = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=75.0, step=0.25, format="%.2f")
        inspect = st.button("Inspect nearest observation  →", use_container_width=True, key="inspect_location")

    if inspect:
        lon_norm = ((lon_in + 180) % 360) - 180
        dist = np.sqrt((latest["latitude"] - lat_in) ** 2 + (latest["plot_lon"] - lon_norm) ** 2)
        idx = dist.idxmin()
        row = latest.loc[idx]

        with right:
            st.markdown('<div class="glass-card" style="padding:28px">', unsafe_allow_html=True)
            st.markdown('<div class="kicker">Nearest grid cell</div>', unsafe_allow_html=True)
            st.markdown(f"### {row['latitude']:.2f}° · {row['plot_lon']:.2f}°")
            st.write(f"Observation date: **{row['date'].strftime('%d %B %Y')}**")
            st.write(f"Chlorophyll-a: **{row['chla']:.5f}**")
            st.write(f"Screening result: **{row['risk_label']}**")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-heading"><div class="kicker">Context</div><h2>Available signals.</h2><p>These values come directly from the processed prediction table. Fields that are not present in the public dashboard dataset are not invented.</p></div>', unsafe_allow_html=True)
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
                value = row[col] if col in row.index and pd.notna(row[col]) else "Not available"
                if isinstance(value, (float, np.floating)):
                    value = f"{value:.5f}"
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.45rem">{value}</div><div class="metric-note">processed dataset</div></div>', unsafe_allow_html=True)

# ============================================================
# INSIGHTS
# ============================================================
elif page == "insights":
    section_title("03 · Insights", "What does the latest field look like?", "A compact view of the current observation grid and the model-derived screening output.")

    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Observations", f"{len(latest):,}", "latest grid")
    with cols[1]: metric_card("Potential risk", f"{len(risk):,}", "flagged locations")
    with cols[2]: metric_card("Normal", f"{len(latest)-len(risk):,}", "remaining grid")
    with cols[3]: metric_card("Risk share", f"{len(risk)/len(latest)*100:.2f}%", "latest field")

    st.markdown('<div class="section-heading"><div class="kicker">Chlorophyll field</div><h2>Distribution at the latest observation date.</h2><p>This describes the satellite-derived chlorophyll-a values in the current processed grid. It is not a direct measure of harmfulness.</p></div>', unsafe_allow_html=True)

    clean_chla = latest["chla"].replace([np.inf, -np.inf], np.nan).dropna()
    fig = go.Figure(go.Histogram(x=clean_chla, nbinsx=55, marker_color="#22a8b4", opacity=.82))
    fig.update_layout(height=430, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.55)", font=dict(color="#18343b"), xaxis_title="Chlorophyll-a", yaxis_title="Observation cells")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.markdown('<div class="section-heading"><div class="kicker">Investigation queue</div><h2>Potential-risk locations.</h2><p>These are screening outputs for further investigation, not confirmed harmful algal bloom observations.</p></div>', unsafe_allow_html=True)

    queue = risk.sort_values(["chla"], ascending=False).head(25).copy()
    show_cols = [c for c in ["latitude", "longitude", "chla", "risk_label"] if c in queue.columns]
    st.dataframe(queue[show_cols].rename(columns={"latitude":"Latitude","longitude":"Longitude","chla":"Chlorophyll-a","risk_label":"Screening"}), use_container_width=True, hide_index=True)

    csv = risk.to_csv(index=False).encode("utf-8")
    st.download_button("Download current screening locations", data=csv, file_name=f"bloomdetect_screening_{latest_date.strftime('%Y-%m-%d')}.csv", mime="text/csv", use_container_width=True)

# ============================================================
# METHOD
# ============================================================
elif page == "method":
    section_title("04 · How it works", "From satellite observation to screening.", "BloomDetect turns a large observation field into a spatial screening view while keeping the limits of the data visible.")

    steps = [
        ("01", "Satellite observation", "EOS-06 OCM-3 provides analysed chlorophyll-a observations across the global ocean."),
        ("02", "Data preparation", "The source observations are cleaned, standardized and organized into an analysis-ready grid."),
        ("03", "Historical context", "The project uses previous observations and historical context to describe changes in chlorophyll-a."),
        ("04", "Machine learning", "A Decision Tree model screens locations against the project's proxy potential-risk target."),
        ("05", "Spatial screening", "The model output is placed back onto the geographic grid so patterns can be explored visually."),
        ("06", "Investigation support", "Flagged locations can be prioritized for further oceanographic or field-based validation."),
    ]
    for num, title, copy in steps:
        st.markdown(f'<div class="glass-card" style="padding:27px;margin:12px 0;display:flex;gap:25px;align-items:flex-start"><div style="font-family:Manrope;font-size:1.8rem;font-weight:800;color:#079aa7;min-width:55px">{num}</div><div><h3 style="margin:0 0 7px;color:#173940;font-family:Manrope">{title}</h3><p style="margin:0;color:#6b858b;line-height:1.7">{copy}</p></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="quote">High chlorophyll-a alone does not prove a harmful algal bloom. BloomDetect provides a model-derived potential-risk screening signal that requires appropriate environmental and field validation.</div>', unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
elif page == "data":
    section_title("05 · Data", "Know what is underneath the interface.", "The dashboard should be transparent about what it has, what it does not have, and what the screening result means.")

    cols = st.columns(4, gap="medium")
    with cols[0]: metric_card("Product", "OCM-3", "EOS-06")
    with cols[1]: metric_card("Grid cells", f"{len(latest):,}", "latest observation")
    with cols[2]: metric_card("Resolution", "0.25°", "source grid")
    with cols[3]: metric_card("Latest", latest_date.strftime("%d %b %Y"), "available observation")

    info = [
        ("Dataset", "E06OCM_L4_AC — analysed chlorophyll product"),
        ("Coverage", "Global ocean source; regional selections are geographic views"),
        ("Current dashboard", "Latest processed prediction table"),
        ("Risk meaning", "Potential bloom-risk screening signal, not confirmed HAB occurrence"),
        ("Validation", "Further oceanographic and field-based validation is required"),
    ]
    for title, text in info:
        st.markdown(f'<div class="feature-card" style="margin:12px 0;min-height:auto"><div class="feature-icon">◦</div><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading"><div class="kicker">Download</div><h2>Take the latest screening with you.</h2><p>Download the current observation table or just the locations currently flagged by the model.</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.download_button("Download latest observations", data=latest.to_csv(index=False).encode("utf-8"), file_name=f"bloomdetect_latest_{latest_date.strftime('%Y-%m-%d')}.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("Download screening locations", data=risk.to_csv(index=False).encode("utf-8"), file_name=f"bloomdetect_risk_{latest_date.strftime('%Y-%m-%d')}.csv", mime="text/csv", use_container_width=True)

    st.markdown('<div class="quote">This application is an early-warning support prototype. Satellite-derived chlorophyll patterns complement, but do not replace, field measurements and other oceanographic evidence.</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    "<div class='footer'>BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Current data through "
    + latest_date.strftime("%d %B %Y")
    + "</div>",
    unsafe_allow_html=True,
)
