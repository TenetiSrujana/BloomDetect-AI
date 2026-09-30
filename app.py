from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="BloomDetect AI", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
INFOGRAPHIC_PATH = BASE_DIR / "bloomdetect_bloom_process.png"
PHYTO_IMAGE_PATH = BASE_DIR / "phytoplankton_signal.png"
SAT_IMAGE_PATH = BASE_DIR / "satellite_signal_illustration.png"

# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("latest_bloom_risk_predictions.csv must be in the same folder as app.py")
    d = pd.read_csv(DATA_PATH)
    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["latitude", "longitude", "chla"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["risk_probability", "model_score", "previous_chla", "historical_baseline", "recent_mean", "recent_max", "chla_anomaly", "chla_change"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
    d["risk_flag"] = d["risk_label"].astype(str).str.strip().str.lower().eq("potential bloom risk")
    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
    return d

try:
    df = load_data()
except Exception as e:
    st.error("BloomDetect could not load the dataset.")
    st.code(str(e))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()
risk = latest[latest["risk_flag"]].copy()

PAGES = ["home", "map", "checker", "compare", "insights", "method", "data"]
NAV = {
    "home": "Home",
    "map": "Risk Map",
    "checker": "Risk Checker",
    "compare": "Compare",
    "insights": "Insights",
    "method": "How It Works",
    "data": "Data",
}
if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"

def go(page):
    st.session_state.page = page

# -----------------------------
# CSS
# -----------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#103f49;--muted:#65848b;--aqua:#0aa7b2;--aqua2:#7be6e1;--pale:#f4ffff;--deep:#07576a;--line:#d8eeee;--shadow:0 18px 55px rgba(15,91,101,.10)}
html,body,[data-testid="stAppViewContainer"]{background:#f4ffff!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif!important}
.stApp{background:linear-gradient(180deg,#fbffff 0%,#effcfc 45%,#f9ffff 100%)!important;overflow-x:hidden}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer, [data-testid="stSidebar"]{display:none!important}
.block-container{max-width:1480px!important;padding:18px 46px 90px!important}
/* full-page animated ocean */
.stApp:before{content:"";position:fixed;inset:auto -10% 0;height:230px;z-index:0;pointer-events:none;background:radial-gradient(ellipse at 18% 60%,rgba(35,198,205,.20) 0 20%,transparent 21%),radial-gradient(ellipse at 62% 45%,rgba(63,214,224,.16) 0 25%,transparent 26%),radial-gradient(ellipse at 92% 60%,rgba(21,170,192,.15) 0 22%,transparent 23%);filter:blur(2px);animation:waterMove 9s ease-in-out infinite alternate}
.stApp:after{content:"◦   ·       ◦       ·    ◦        ·       ◦";position:fixed;left:4%;bottom:-80px;z-index:0;pointer-events:none;color:rgba(8,153,169,.16);font-size:24px;letter-spacing:60px;line-height:80px;animation:bubbleRise 16s linear infinite}
@keyframes waterMove{from{transform:translateX(-3%) scaleX(1.02)}to{transform:translateX(3%) scaleX(1.08)}}
@keyframes bubbleRise{from{transform:translateY(70px);opacity:.05}50%{opacity:.25}to{transform:translateY(-95vh);opacity:.03}}
/* brand */
.brand-bar{position:relative;z-index:10;display:flex;align-items:center;justify-content:space-between;padding:16px 22px;border:1px solid var(--line);background:rgba(255,255,255,.88);border-radius:22px;box-shadow:var(--shadow);backdrop-filter:blur(18px)}
.brand-left{display:flex;align-items:center;gap:14px}.logo{width:50px;height:50px;border-radius:17px;background:linear-gradient(145deg,#35d6d3,#087e91);display:grid;place-items:center;color:white;font-size:20px;font-weight:800;box-shadow:0 12px 28px rgba(9,144,157,.22)}
.brand-name{font:800 1.25rem Manrope,sans-serif;color:#103f49;letter-spacing:-.03em}.brand-sub{font-size:.72rem;color:#76959b;margin-top:3px}
/* nav buttons */
.nav{position:relative;z-index:20;display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin:14px 0 28px}
.nav .stButton>button{height:50px!important;border-radius:15px!important;background:#fff!important;border:1px solid #cde4e6!important;color:#174650!important;font-weight:800!important;font-size:.82rem!important;box-shadow:0 8px 20px rgba(15,91,101,.07)!important;transition:all .18s ease!important}
.nav .stButton>button:hover{background:#e6fbfb!important;border-color:#17abb4!important;color:#075d6c!important;transform:translateY(-2px)!important}
.nav .stButton>button:focus,.nav .stButton>button:active{background:#d9f7f7!important;border-color:#0aa7b2!important;color:#075d6c!important;box-shadow:0 0 0 3px rgba(10,167,178,.14)!important}
.nav .stButton>button p,.nav .stButton>button span{color:inherit!important}
/* typography */
.kicker{font:800 .68rem Manrope,sans-serif;letter-spacing:.2em;text-transform:uppercase;color:#0798a6;margin-bottom:12px}.title{font:800 clamp(2.4rem,5vw,5.2rem)/.98 Manrope,sans-serif;letter-spacing:-.06em;color:#103f49;margin:0 0 16px}.lead{font-size:1rem;line-height:1.75;color:#66848b;max-width:850px}.section{position:relative;z-index:2;padding:46px 0 20px}.section h2{font:800 clamp(2rem,3.5vw,3.5rem)/1.05 Manrope,sans-serif;letter-spacing:-.05em;color:#103f49;margin:0 0 12px}.section p{color:#66848b;line-height:1.7;margin:0;max-width:900px}
/* cards */
.card{position:relative;z-index:2;background:rgba(255,255,255,.84);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow);padding:26px}.card h3{font:800 1.3rem Manrope,sans-serif;color:#123f49;margin:0 0 9px}.card p{color:#66848b;line-height:1.7;margin:0}.mini-label{font:800 .66rem Manrope,sans-serif;letter-spacing:.16em;text-transform:uppercase;color:#78959b;margin-bottom:8px}.metric{position:relative;z-index:2;padding:23px 24px;background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}.metric .label{font:800 .65rem Manrope,sans-serif;letter-spacing:.15em;text-transform:uppercase;color:#78959b}.metric .value{font:800 2rem Manrope,sans-serif;color:#123f49;margin-top:9px}.metric .note{font-size:.76rem;color:#78959b;margin-top:4px}
/* hero */
.hero{position:relative;z-index:2;overflow:hidden;min-height:570px;border-radius:34px;padding:88px 76px;background:linear-gradient(135deg,#063d51 0%,#076b7c 48%,#18aeb1 100%);box-shadow:0 32px 90px rgba(6,86,100,.18)}
.hero:before{content:"";position:absolute;inset:-25%;background:repeating-radial-gradient(ellipse at 20% 115%,transparent 0 55px,rgba(181,255,251,.13) 57px 59px,transparent 61px 105px);transform:rotate(-7deg);animation:waveLines 13s linear infinite}.hero:after{content:"";position:absolute;left:-5%;right:-5%;bottom:-130px;height:280px;background:rgba(142,244,237,.15);border-radius:50%;animation:heroWave 7s ease-in-out infinite alternate}.hero-content{position:relative;z-index:2;max-width:820px}.hero .kicker{color:#a6fffa}.hero h1{font:800 clamp(3.6rem,7vw,7.1rem)/.88 Manrope,sans-serif;letter-spacing:-.075em;color:#e4ffff;margin:0 0 22px}.hero p{font-size:1.08rem;line-height:1.8;color:#e0fbfb;max-width:760px}.hero-badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}.badge{padding:9px 13px;border-radius:999px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);color:#efffff;font-size:.76rem;font-weight:700;backdrop-filter:blur(8px)}
@keyframes waveLines{from{transform:translateX(-4%) rotate(-7deg)}to{transform:translateX(4%) rotate(-7deg)}}@keyframes heroWave{from{transform:translateX(-2%) rotate(-1deg)}to{transform:translateX(2%) rotate(1deg)}}
/* educational image */
.info-image{position:relative;z-index:2;background:#fff;padding:10px;border-radius:26px;border:1px solid var(--line);box-shadow:var(--shadow);overflow:hidden}.info-image img{width:100%;height:auto;max-height:560px;object-fit:cover;object-position:center;border-radius:18px;display:block}
/* feature cards */
.feature-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.feature{position:relative;z-index:2;padding:25px;background:#fff;border:1px solid var(--line);border-radius:22px;box-shadow:var(--shadow);min-height:190px;transition:.2s}.feature:hover{transform:translateY(-4px);box-shadow:0 25px 65px rgba(16,91,102,.13)}.feature .icon{font-size:1.65rem;margin-bottom:15px}.feature h3{font:800 1.1rem Manrope;color:#123f49;margin:0 0 8px}.feature p{font-size:.86rem;line-height:1.65;color:#66848b;margin:0}
/* map */
.map-card{position:relative;z-index:2;background:#06233a;border-radius:26px;padding:10px;border:1px solid rgba(37,188,201,.25);box-shadow:0 28px 75px rgba(5,42,60,.18)}
.map-legend{display:flex;align-items:center;gap:16px;flex-wrap:wrap;color:#e8ffff;font-size:.78rem;padding:12px 14px}.legend-gradient{width:180px;height:10px;border-radius:999px;background:linear-gradient(90deg,#09264a,#0b76a3,#19c8c6,#b7e76b,#ffe27c)}.risk-dot{width:12px;height:12px;border-radius:50%;background:#ff5364;border:2px solid white}
/* forms */
div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] div{border-radius:13px!important}.stNumberInput label,.stSelectbox label{font-weight:800!important;color:#285761!important}.stNumberInput button{color:#075d6c!important}
/* table */
.table-wrap{position:relative;z-index:2;border-radius:20px;overflow:hidden;border:1px solid var(--line);box-shadow:var(--shadow);background:#fff}.table-wrap table{width:100%;border-collapse:collapse;font-size:.82rem}.table-wrap th{background:#103f49;color:#fff;text-align:left;padding:12px 14px}.table-wrap td{padding:11px 14px;border-top:1px solid #e3eeee;color:#315b63;background:#fff}.table-wrap tr:nth-child(even) td{background:#f7fcfc}
/* method */
.step{position:relative;z-index:2;display:grid;grid-template-columns:70px 1fr;gap:22px;align-items:start;padding:25px 0;border-bottom:1px solid var(--line)}.step-num{width:58px;height:58px;border-radius:18px;background:#dff8f8;color:#078c99;display:grid;place-items:center;font:800 1rem Manrope}.step h3{font:800 1.15rem Manrope;color:#123f49;margin:0 0 7px}.step p{color:#66848b;line-height:1.7;margin:0}
/* downloads */
.download-panel{position:relative;z-index:2;background:linear-gradient(135deg,#087f90,#18aeb1);border-radius:28px;padding:30px;box-shadow:0 25px 70px rgba(8,126,141,.18)}.download-panel h3{font:800 1.55rem Manrope;color:#fff;margin:0 0 7px}.download-panel p{color:#d9ffff;margin:0 0 18px}.download-panel .stDownloadButton button{background:#fff!important;color:#087381!important;border:0!important}.download-panel .stDownloadButton button p,.download-panel .stDownloadButton button span{color:#087381!important}
.note{position:relative;z-index:2;padding:18px 20px;background:#e7fafa;border-left:4px solid #13aab2;border-radius:0 18px 18px 0;color:#4d7078;line-height:1.7}
.footer{position:relative;z-index:2;margin-top:60px;padding-top:22px;border-top:1px solid var(--line);color:#78959b;font-size:.74rem}
@media(max-width:1000px){.nav{grid-template-columns:repeat(4,1fr)}.feature-grid{grid-template-columns:repeat(2,1fr)}.hero{padding:60px 42px}.bloom-grid{grid-template-columns:1fr!important}.two-col{grid-template-columns:1fr!important}}
@media(max-width:650px){.block-container{padding:12px 18px 60px!important}.nav{grid-template-columns:repeat(2,1fr)}.feature-grid{grid-template-columns:1fr}.hero{padding:50px 28px;min-height:500px}.hero h1{font-size:3.3rem}.brand-sub{display:none}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header + navigation
# -----------------------------
st.markdown("""
<div class="brand-bar">
  <div class="brand-left">
    <div class="logo">≈</div>
    <div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Satellite-based potential bloom-risk screening</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

nav_cols = st.columns(7, gap="small")
for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(NAV[page], key=f"nav_{page}", use_container_width=True):
            go(page)
            st.rerun()

# -----------------------------
# Helpers
# -----------------------------
def metric(label, value, note):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)

def section(kicker, title, copy):
    st.markdown(f'<div class="section"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{copy}</p></div>', unsafe_allow_html=True)

def status_text(row):
    return "Potential bloom risk" if bool(row["risk_flag"]) else "No potential bloom-risk flag"

def safe(v, digits=5):
    return "Unavailable" if pd.isna(v) else f"{float(v):.{digits}f}"

def probability_text(v):
    if pd.isna(v):
        return "Unavailable"
    x=float(v)
    if x <= 1.0:
        x *= 100.0
    return f"{x:.1f}%"

def checker_row(lat, lon):
    dist = (latest["latitude"] - lat) ** 2 + (latest["longitude"] - lon) ** 2
    return latest.loc[dist.idxmin()]

def map_figure(view, show_risk=True):
    data = latest.copy()
    if view == "Indian Ocean": data = data[data.latitude.between(-40,30) & data.plot_lon.between(20,120)]
    elif view == "Arabian Sea": data = data[data.latitude.between(5,30) & data.plot_lon.between(45,78)]
    elif view == "Bay of Bengal": data = data[data.latitude.between(5,23) & data.plot_lon.between(78,100)]
    if data.empty: data = latest.copy()
    # Keep the map responsive while retaining every flagged location.
    risk_subset = data[data.risk_flag].copy()
    normal_subset = data[~data.risk_flag].copy()
    if len(normal_subset) > 18000:
        normal_subset = normal_subset.sample(18000, random_state=42)
    data = pd.concat([normal_subset, risk_subset], ignore_index=True).drop_duplicates(subset=["latitude","longitude"])
    vmax = max(float(data.chla.quantile(.995)), .01)
    fig = px.scatter_geo(data, lat="latitude", lon="plot_lon", color="chla", custom_data=["latitude","longitude","chla","risk_label"], projection="mercator", color_continuous_scale=["#09264a","#0b76a3","#19c8c6","#b7e76b","#ffe27c"], range_color=(0,vmax))
    fig.update_traces(marker=dict(size=5,opacity=.78,line=dict(width=0)), hovertemplate="Lat %{customdata[0]:.2f}°<br>Lon %{customdata[1]:.2f}°<br>Chl-a %{customdata[2]:.5f}<br>%{customdata[3]}<extra></extra>")
    if show_risk:
        rp = data[data.risk_flag]
        if not rp.empty:
            fig.add_trace(px.scatter_geo(rp,lat="latitude",lon="plot_lon",custom_data=["latitude","longitude","chla"],projection="mercator").data[0])
            fig.data[-1].marker = dict(size=10,color="#ff5264",opacity=1,line=dict(width=1.5,color="#ffffff"))
            fig.data[-1].name="Potential bloom risk"
            fig.data[-1].hovertemplate="Potential bloom-risk flag<br>Lat %{customdata[0]:.2f}°<br>Lon %{customdata[1]:.2f}°<br>Chl-a %{customdata[2]:.5f}<extra></extra>"
    ranges = {"Indian Ocean":([-40,30],[20,120]),"Arabian Sea":([5,30],[45,78]),"Bay of Bengal":([5,23],[78,100])}
    geo = dict(showland=True,landcolor="#d7eee8",showocean=True,oceancolor="#e8f8f8",showcoastlines=True,coastlinecolor="#76aab0",showcountries=True,countrycolor="#9bc4c8",bgcolor="#06233a")
    if view in ranges: geo.update(lataxis_range=ranges[view][0],lonaxis_range=ranges[view][1])
    fig.update_geos(**geo)
    fig.update_layout(height=620,margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor="#06233a",font=dict(color="#eaffff"),legend=dict(bgcolor="rgba(6,35,58,.80)",font=dict(color="#eaffff")),coloraxis_colorbar=dict(title=dict(text="Chl-a",font=dict(color="#eaffff")),tickfont=dict(color="#eaffff")))
    return fig

# -----------------------------
# HOME
# -----------------------------
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
      <div class="hero-content">
        <div class="kicker">EOS-06 · OCM-3 · BloomDetect AI</div>
        <h1>See the ocean<br>more clearly.</h1>
        <p>BloomDetect AI turns satellite-derived chlorophyll-a observations into an interactive screening system for finding ocean locations whose recent patterns may deserve a closer look.</p>
        <div class="hero-badges"><span class="badge">🌊 Spatial screening</span><span class="badge">🛰️ Satellite observations</span><span class="badge">🔎 Investigation support</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section("01 · Understand the signal", "What is an algal bloom?", "An algal bloom occurs when algae or phytoplankton become unusually concentrated in part of a water body. Some blooms are harmless, while some can have ecological or health effects. A chlorophyll-a signal can help identify where conditions deserve closer investigation, but it does not by itself prove a harmful bloom or identify toxins.")
    c1,c2 = st.columns([.82,1.18], gap="large")
    with c1:
        st.markdown("""
        <div class="card" style="height:100%">
          <div class="mini-label">Why this matters</div>
          <h3>Small organisms can create a large environmental signal.</h3>
          <p>Phytoplankton use light and nutrients to grow. When many cells accumulate, ocean colour and chlorophyll-a can change. BloomDetect uses that observable signal to screen locations for potential bloom risk, then points people toward places worth investigating with additional evidence.</p>
          <br><p><b>Important:</b> the dashboard is an early-warning support tool, not a species, toxin, or laboratory confirmation system.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if INFOGRAPHIC_PATH.exists():
            st.markdown('<div class="info-image">', unsafe_allow_html=True)
            st.image(str(INFOGRAPHIC_PATH), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Bloom process illustration is missing from the app folder.")

    section("02 · Know what the system can tell you", "Three rules keep the screening honest.", "The interface is designed around the difference between an observable satellite signal and a confirmed harmful event.")
    cards = [
        ("01","Not every bloom is harmful","A high chlorophyll-a value is not proof of toxins, harmful species, or ecological damage."),
        ("02","A flag means investigate","A potential-risk flag identifies a location for closer inspection, not a confirmed HAB."),
        ("03","Context changes the meaning","Previous observations, historical baseline and recent behaviour help put the latest signal in context."),
    ]
    cols=st.columns(3,gap="medium")
    for col,(n,h,p) in zip(cols,cards):
        with col: st.markdown(f'<div class="card"><div class="mini-label">{n}</div><h3>{h}</h3><p>{p}</p></div>',unsafe_allow_html=True)

    section("03 · Read the signal visually", "From tiny cells to a satellite view.", "These small visuals explain the two scales the project connects: microscopic phytoplankton and the large-area satellite observation.")
    v1,v2=st.columns(2,gap="large")
    with v1:
        if PHYTO_IMAGE_PATH.exists():
            st.image(str(PHYTO_IMAGE_PATH), use_container_width=True)
    with v2:
        if SAT_IMAGE_PATH.exists():
            st.image(str(SAT_IMAGE_PATH), use_container_width=True)

    section("04 · Who can use it", "One dashboard, different jobs.", "The same screening result can support different next steps depending on who is using it.")
    users=[
        ("🧭","Environmental teams","Find flagged coordinates and decide where field or local observations deserve attention."),
        ("🐟","Fisheries & coastal observers","Check a location before looking for supporting observations from the same area."),
        ("🔬","Researchers & students","Explore chlorophyll patterns, derived context features and model screening outputs."),
        ("📢","Community awareness","Turn a technical satellite signal into an understandable starting point without claiming a confirmed harmful event."),
    ]
    cols=st.columns(4,gap="medium")
    for col,(ic,h,p) in zip(cols,users):
        with col: st.markdown(f'<div class="feature"><div class="icon">{ic}</div><h3>{h}</h3><p>{p}</p></div>',unsafe_allow_html=True)

    section("05 · Explore", "Four useful actions instead of four decorative screens.", "Each feature has a different job, so the same dataset is not repeated across the site.")
    feats=[
        ("🌍","Risk Map","See the latest chlorophyll-a field and potential-risk flags spatially."),
        ("📍","Risk Checker","Enter latitude and longitude and inspect the nearest processed grid cell."),
        ("↔️","Compare","Compare two locations using the same latest observation fields."),
        ("📊","Insights","Understand the latest field, risk share and geographic concentration."),
    ]
    cols=st.columns(4,gap="medium")
    for col,(ic,h,p) in zip(cols,feats):
        with col: st.markdown(f'<div class="feature"><div class="icon">{ic}</div><h3>{h}</h3><p>{p}</p></div>',unsafe_allow_html=True)

# -----------------------------
# MAP
# -----------------------------
elif st.session_state.page == "map":
    section("01 · Spatial explorer", "See where the signal changes.", "The map shows the latest processed chlorophyll-a field. Red markers are model-screened potential bloom-risk locations. They are screening flags, not confirmed harmful blooms.")
    a,b,c=st.columns([1.5,1.2,.7],gap="medium")
    with a: view=st.selectbox("Geographic view",["Global Ocean","Indian Ocean","Arabian Sea","Bay of Bengal"])
    with b: show=st.toggle("Show potential-risk overlay",True)
    with c: metric("Visible cells",f"{len(latest):,}","latest observation")
    st.markdown('<div class="map-card">',unsafe_allow_html=True)
    st.plotly_chart(map_figure(view,show),use_container_width=True,config={"scrollZoom":False,"displaylogo":False,"modeBarButtonsToRemove":["lasso2d","select2d"]})
    st.markdown('<div class="map-legend"><span>Chlorophyll-a field</span><span class="legend-gradient"></span><span>Lower → higher within this view</span><span class="risk-dot"></span><span>Potential bloom risk</span></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="note"><b>How to use this:</b> zoom to a region, identify a flagged coordinate, then open <b>Risk Checker</b> to inspect the values behind that location.</div>',unsafe_allow_html=True)

# -----------------------------
# CHECKER
# -----------------------------
elif st.session_state.page == "checker":
    section("02 · Coordinate screening", "Is this location flagged?", "Enter a latitude and longitude. BloomDetect finds the nearest processed 0.25° grid cell and reports its latest screening result and supporting signals.")
    left,right=st.columns([.75,1.25],gap="large")
    with left:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        lat=st.number_input("Latitude",min_value=-90.0,max_value=90.0,value=18.0,step=.25,format="%.2f")
        lon=st.number_input("Longitude",min_value=-180.0,max_value=180.0,value=78.0,step=.25,format="%.2f")
        st.markdown('<p style="color:#66848b;font-size:.84rem">Tip: choose a point inside the Indian Ocean, Arabian Sea or Bay of Bengal to explore the current study area.</p></div>',unsafe_allow_html=True)
    row=checker_row(lat,lon)
    with right:
        risk_yes=bool(row["risk_flag"])
        status="Potential bloom risk" if risk_yes else "No potential bloom-risk flag"
        status_color="#d94d5e" if risk_yes else "#087f8d"
        prob=probability_text(row.get("risk_probability",np.nan))
        st.markdown(f'''<div class="card" style="min-height:300px"><div class="mini-label">Nearest processed observation</div><h3 style="font-size:2rem">{row.latitude:.2f}° · {row.longitude:.2f}°</h3><div class="result-grid"><div><span>Date</span><b>{row.date.strftime('%d %B %Y')}</b></div><div><span>Chlorophyll-a</span><b>{safe(row.chla)}</b></div><div><span>Screening</span><b style="color:{status_color}">{status}</b></div><div><span>Model probability</span><b>{prob}</b></div></div></div>''',unsafe_allow_html=True)
    section("Context at the selected cell", "What is changing around this point?", "These values are stored in the processed prediction table and help explain why the latest observation may look different from the location's recent behaviour.")
    vals=[("Previous chlorophyll-a",row.get("previous_chla",np.nan)),("Historical baseline",row.get("historical_baseline",np.nan)),("Recent mean",row.get("recent_mean",np.nan)),("Recent maximum",row.get("recent_max",np.nan)),("Chlorophyll anomaly",row.get("chla_anomaly",np.nan)),("Chlorophyll change",row.get("chla_change",np.nan))]
    cols=st.columns(3,gap="medium")
    for col,(lab,val) in zip(cols,vals):
        with col: metric(lab,safe(val),"selected grid cell")
    if risk_yes:
        st.markdown('<div class="note"><b>Screening interpretation:</b> this coordinate is currently flagged as potential bloom risk by the model. Confirming a harmful algal bloom requires additional ecological, field, laboratory or species-specific evidence.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="note"><b>Screening interpretation:</b> this coordinate is not currently flagged by the model. That does not mean algae are absent; it means the current screening rule did not flag this processed grid cell.</div>',unsafe_allow_html=True)

# -----------------------------
# COMPARE
# -----------------------------
elif st.session_state.page == "compare":
    section("03 · Location comparison", "Compare two places side by side.", "Use this when you want to inspect whether two coordinates show different latest chlorophyll-a signals or screening results.")
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown('<h3>Location A</h3>',unsafe_allow_html=True)
        a_lat=st.number_input("Latitude A",-90.,90.,18.,.25,format="%.2f",key="a_lat"); a_lon=st.number_input("Longitude A",-180.,180.,78.,.25,format="%.2f",key="a_lon"); st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.markdown('<h3>Location B</h3>',unsafe_allow_html=True)
        b_lat=st.number_input("Latitude B",-90.,90.,20.,.25,format="%.2f",key="b_lat"); b_lon=st.number_input("Longitude B",-180.,180.,90.,.25,format="%.2f",key="b_lon"); st.markdown('</div>',unsafe_allow_html=True)
    ra,rb=checker_row(a_lat,a_lon),checker_row(b_lat,b_lon)
    st.markdown('<div style="height:20px"></div>',unsafe_allow_html=True)
    for name,row in [("Location A",ra),("Location B",rb)]:
        status="Potential bloom risk" if row.risk_flag else "No potential bloom-risk flag"
        st.markdown(f'<div class="card" style="margin-bottom:14px"><div class="mini-label">{name}</div><h3>{row.latitude:.2f}° · {row.longitude:.2f}°</h3><div class="result-grid"><div><span>Chlorophyll-a</span><b>{safe(row.chla)}</b></div><div><span>Screening</span><b>{status}</b></div><div><span>Anomaly</span><b>{safe(row.get("chla_anomaly",np.nan))}</b></div><div><span>Recent change</span><b>{safe(row.get("chla_change",np.nan))}</b></div></div></div>',unsafe_allow_html=True)

# -----------------------------
# INSIGHTS
# -----------------------------
elif st.session_state.page == "insights":
    section("04 · Latest-field insights", "What stands out in the current observation?", "These summaries answer practical questions about the latest processed field without repeating the dataset documentation.")
    normal_count=int((~latest.risk_flag).sum()); risk_count=int(latest.risk_flag.sum()); share=100*risk_count/len(latest) if len(latest) else 0
    cols=st.columns(4,gap="medium")
    for col,(l,v,n) in zip(cols,[("Observation cells",f"{len(latest):,}","latest field"),("Potential-risk cells",f"{risk_count:,}","model screening"),("Normal cells",f"{normal_count:,}","latest field"),("Risk share",f"{share:.2f}%","latest field")]):
        with col: metric(l,v,n)
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="card"><div class="mini-label">Screening composition</div><h3>Normal vs potential bloom-risk</h3><p>Count of the two model screening outcomes in the latest field.</p></div>',unsafe_allow_html=True)
        chart=pd.DataFrame({"Classification":["Normal","Potential bloom risk"],"Cells":[normal_count,risk_count]})
        fig=px.bar(chart,x="Classification",y="Cells",text="Cells",color="Classification",color_discrete_map={"Normal":"#73c9cf","Potential bloom risk":"#ff6472"})
        fig.update_layout(height=360,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.72)",showlegend=False,font=dict(color="#285761"))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown('<div class="card"><div class="mini-label">Chlorophyll-a distribution</div><h3>Where values cluster</h3><p>This is the distribution of the current satellite-derived field, not a direct measure of harmfulness.</p></div>',unsafe_allow_html=True)
        fig=px.histogram(latest,x="chla",nbins=30,color_discrete_sequence=["#14a6b1"])
        fig.update_layout(height=360,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.72)",font=dict(color="#285761"))
        st.plotly_chart(fig,use_container_width=True)
    section("Investigation queue", "Which coordinates are currently flagged?", "This list is for follow-up inspection. It does not rank confirmed HAB severity.")
    q=risk.copy()
    if "risk_probability" in q.columns: q=q.sort_values(["risk_probability","chla"],ascending=False)
    else: q=q.sort_values("chla",ascending=False)
    q=q.head(20)[[c for c in ["latitude","longitude","chla","risk_probability","risk_label"] if c in q.columns]].copy()
    if "risk_probability" in q.columns: q["risk_probability"]=q["risk_probability"].round(3)
    st.dataframe(q,use_container_width=True,hide_index=True)

# -----------------------------
# METHOD
# -----------------------------
elif st.session_state.page == "method":
    section("05 · Transparent pipeline", "From satellite observation to screening support.", "The project turns an environmental observation into a reproducible sequence of cleaning, temporal context, machine learning and spatial inspection.")
    steps=[
        ("01","Observe","Use EOS-06 OCM-3 analysed chlorophyll-a observations on the global-ocean grid."),
        ("02","Clean","Remove invalid records and organize the observations into a consistent spatial-temporal table."),
        ("03","Build context","Create previous observation, historical baseline, recent mean, recent maximum, anomaly and change features."),
        ("04","Screen","A Decision Tree model screens locations using the prepared context features."),
        ("05","Map","Project the latest screening results back onto geographic coordinates for spatial inspection."),
        ("06","Investigate","Use the flagged coordinates as an early-warning shortlist for additional observations and validation."),
    ]
    for n,h,p in steps:
        st.markdown(f'<div class="step"><div class="step-num">{n}</div><div><h3>{h}</h3><p>{p}</p></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin-top:28px"><b>Scientific boundary:</b> the current dataset does not contain confirmed harmful-bloom species or toxin labels. Therefore the model output is presented as <b>potential bloom risk</b>, not confirmed HAB detection.</div>',unsafe_allow_html=True)

# -----------------------------
# DATA
# -----------------------------
elif st.session_state.page == "data":
    section("06 · Dataset & outputs", "Everything about the dataset lives here.", "This page contains the source product, coverage, grid structure, processed fields, interpretation limits and downloads. The other pages intentionally avoid repeating this documentation.")
    cols=st.columns(4,gap="medium")
    for col,(l,v,n) in zip(cols,[("Product","E06OCM_L4_AC","EOS-06 / OCM-3"),("Grid","0.25°","latitude × longitude"),("Latest cells",f"{len(latest):,}","processed observation"),("Latest date",latest_date.strftime("%d %b %Y"),"processed dataset")]):
        with col: metric(l,v,n)
    st.markdown('<div class="card" style="margin-top:20px"><div class="mini-label">Source & interpretation</div><h3>What the dashboard is built from.</h3><p><b>Satellite product:</b> EOS-06 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC).</p><p><b>Primary variable:</b> chlorophyll-a (<code>chla</code>).</p><p><b>Spatial structure:</b> 0.25° latitude × 0.25° longitude global-ocean grid.</p><p><b>Current dashboard view:</b> the latest processed observation represented in the prediction table.</p><p><b>Interpretation:</b> the dataset provides an environmental observation signal. It does not contain confirmed harmful-bloom species or toxin labels.</p></div>',unsafe_allow_html=True)
    section("Data dictionary", "Fields available in the processed file.", "These are the fields the public dashboard can inspect.")
    fields=[("latitude","Grid latitude","degrees"),("longitude","Grid longitude","degrees"),("date","Observation date","date"),("chla","Satellite-derived chlorophyll-a","product value"),("risk_label","Model screening result","Normal / Potential Bloom Risk"),("risk_probability","Model probability when available","stored model probability") ,("previous_chla","Previous observation","derived feature"),("historical_baseline","Historical baseline","derived feature"),("recent_mean","Recent mean","derived feature"),("recent_max","Recent maximum","derived feature"),("chla_anomaly","Chlorophyll anomaly","derived feature"),("chla_change","Change from previous observation","derived feature")]
    rows=''.join([f'<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>' for a,b,c in fields])
    st.markdown(f'<div class="table-wrap"><table><thead><tr><th>Field</th><th>Meaning</th><th>Type</th></tr></thead><tbody>{rows}</tbody></table></div>',unsafe_allow_html=True)
    section("Useful outputs", "Take the screening result with you.", "Downloads are kept here and nowhere else in the website.")
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="download-panel"><h3>Latest observation table</h3><p>Download the complete latest processed prediction table used by the dashboard.</p></div>',unsafe_allow_html=True)
        st.download_button("⬇ Download latest observations",latest.to_csv(index=False).encode(),"bloomdetect_latest_observations.csv","text/csv",use_container_width=True)
    with c2:
        st.markdown('<div class="download-panel"><h3>Potential-risk shortlist</h3><p>Download only the latest locations currently flagged for potential bloom risk.</p></div>',unsafe_allow_html=True)
        st.download_button("⬇ Download potential-risk locations",risk.to_csv(index=False).encode(),"bloomdetect_potential_risk.csv","text/csv",use_container_width=True)
    st.markdown('<div class="note" style="margin-top:25px"><b>Data boundary:</b> this is not a live real-time monitoring feed. Satellite screening should be combined with field observations and other environmental evidence before decisions about a harmful event are made.</div>',unsafe_allow_html=True)

st.markdown(f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Current data through {latest_date.strftime("%d %B %Y")}</div>',unsafe_allow_html=True)
