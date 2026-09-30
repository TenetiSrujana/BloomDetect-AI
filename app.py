from pathlib import Path
import base64
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
:root{--ink:#123f49;--muted:#64858d;--aqua:#10aeb7;--aqua2:#8be9e5;--pale:#f2fcfc;--deep:#07576a;--line:#d3ebec;--glass:rgba(255,255,255,.66);--shadow:0 18px 55px rgba(15,91,101,.10)}
html,body,[data-testid="stAppViewContainer"]{background:#f2fcfc!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif!important}
.stApp{background:linear-gradient(180deg,#fbffff 0%,#eefbfb 46%,#fbffff 100%)!important;overflow-x:hidden}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}
.block-container{max-width:1320px!important;padding:22px 38px 80px!important}
/* Whole-site ocean motion */
.stApp:before{content:"";position:fixed;left:-10%;right:-10%;bottom:-110px;height:260px;z-index:0;pointer-events:none;background:radial-gradient(ellipse at 15% 55%,rgba(42,205,211,.20) 0 18%,transparent 19%),radial-gradient(ellipse at 55% 45%,rgba(71,224,222,.15) 0 22%,transparent 23%),radial-gradient(ellipse at 90% 60%,rgba(15,171,191,.15) 0 20%,transparent 21%);filter:blur(2px);animation:waterMove 10s ease-in-out infinite alternate}
.stApp:after{content:"◦     ·        ◦        ·        ◦        ·        ◦";position:fixed;left:5%;bottom:-80px;z-index:0;pointer-events:none;color:rgba(8,153,169,.14);font-size:25px;letter-spacing:58px;animation:bubbleRise 18s linear infinite}
@keyframes waterMove{from{transform:translateX(-3%) scaleX(1.02)}to{transform:translateX(3%) scaleX(1.08)}}
@keyframes bubbleRise{from{transform:translateY(80px);opacity:.04}50%{opacity:.2}to{transform:translateY(-100vh);opacity:.02}}
/* Brand */
.brand-bar{position:relative;z-index:10;display:flex;align-items:center;padding:16px 22px;border:1px solid rgba(205,232,233,.9);background:rgba(255,255,255,.76);border-radius:24px;box-shadow:var(--shadow);backdrop-filter:blur(22px)}
.brand-left{display:flex;align-items:center;gap:14px}.logo{width:50px;height:50px;border-radius:17px;background:linear-gradient(145deg,#38d7d2,#087d90);display:grid;place-items:center;color:#fff;font-size:20px;font-weight:800;box-shadow:0 12px 28px rgba(9,144,157,.20)}
.brand-name{font:800 1.25rem Manrope,sans-serif;color:#103f49;letter-spacing:-.03em}.brand-sub{font-size:.76rem;color:#73939a;margin-top:3px}
/* All Streamlit buttons: visible glassmorphism, never black */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{height:48px!important;border-radius:15px!important;background:rgba(255,255,255,.96)!important;border:2px solid #184b57!important;color:#123f49!important;font-weight:800!important;font-size:.94rem!important;box-shadow:0 7px 18px rgba(15,70,82,.10),inset 0 1px 0 #fff!important;backdrop-filter:blur(20px)!important;-webkit-backdrop-filter:blur(20px)!important;transition:transform .18s ease,background .18s ease,border-color .18s ease,box-shadow .18s ease!important}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:#e3fbfb!important;border-color:#087f90!important;color:#075d6c!important;transform:translateY(-2px)!important;box-shadow:0 13px 26px rgba(15,91,101,.15),0 0 0 3px rgba(18,174,184,.10)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#0a5364,#087f90)!important;border-color:#0a3f4d!important;color:#fff!important;box-shadow:0 12px 28px rgba(8,82,99,.25),0 0 0 3px rgba(16,174,183,.12)!important}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#084b5b,#067585)!important;color:#fff!important}
.stButton>button:focus,.stButton>button:active,.stDownloadButton>button:focus,.stFormSubmitButton>button:focus{background:#e3fbfb!important;border-color:#087f90!important;color:#075d6c!important;box-shadow:0 0 0 3px rgba(16,174,183,.16),0 10px 24px rgba(15,91,101,.11)!important}
.stButton>button p,.stButton>button span,.stDownloadButton>button p,.stDownloadButton>button span,.stFormSubmitButton>button p,.stFormSubmitButton>button span{color:inherit!important}
/* Navigation */
.nav-spacer{height:8px}.nav-wrap{position:relative;z-index:12;margin-bottom:8px}
/* Typography */
.kicker{font:800 .72rem Manrope,sans-serif;letter-spacing:.19em;text-transform:uppercase;color:#0798a6;margin-bottom:13px}
.title{font:800 clamp(2.5rem,5vw,5.2rem)/.98 Manrope,sans-serif;letter-spacing:-.06em;color:#103f49;margin:0 0 16px}
.lead{font-size:1.06rem;line-height:1.78;color:#62838b;max-width:1100px}
.section{position:relative;z-index:2;padding:38px 0 20px}.section h2{font:800 clamp(2.15rem,3.15vw,3.25rem)/1.08 Manrope,sans-serif;letter-spacing:-.05em;color:#103f49;margin:0 0 16px}.section p{color:#5f7f87;line-height:1.65;margin:0;max-width:1120px;font-size:1.02rem}
/* Cards */
.card{position:relative;z-index:2;background:rgba(255,255,255,.78);border:1.5px solid #b8dfe2;border-radius:25px;box-shadow:var(--shadow);padding:24px;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px)}
.card h3{font:800 1.42rem Manrope,sans-serif;color:#123f49;margin:0 0 11px}.card p{color:#66848b;line-height:1.78;margin:0;font-size:1rem}.mini-label{font:800 .68rem Manrope,sans-serif;letter-spacing:.15em;text-transform:uppercase;color:#78959b;margin-bottom:9px}
.metric{position:relative;z-index:2;min-height:128px;height:100%;box-sizing:border-box;padding:19px;background:linear-gradient(145deg,rgba(255,255,255,.97),rgba(231,249,249,.82));border:1.5px solid rgba(78,169,179,.38);border-radius:22px;box-shadow:0 16px 38px rgba(15,91,101,.10),inset 0 1px 0 rgba(255,255,255,1);backdrop-filter:blur(18px);display:flex;flex-direction:column;justify-content:center}.metric .label{font:800 .68rem Manrope,sans-serif;letter-spacing:.13em;text-transform:uppercase;color:#6d8c93}.metric .value{font:800 1.55rem Manrope,sans-serif;color:#123f49;margin-top:8px;white-space:nowrap}.metric .note{font-size:.78rem;color:#78959b;margin-top:6px}
/* Hero */
.hero{position:relative;z-index:2;overflow:hidden;min-height:535px;border-radius:34px;padding:80px 70px;background:linear-gradient(135deg,#063d51 0%,#076b7c 48%,#18aeb1 100%);box-shadow:0 32px 90px rgba(6,86,100,.18)}
.hero:before{content:"";position:absolute;inset:-25%;background:repeating-radial-gradient(ellipse at 20% 115%,transparent 0 55px,rgba(181,255,251,.13) 57px 59px,transparent 61px 105px);transform:rotate(-7deg);animation:waveLines 13s linear infinite}.hero:after{content:"";position:absolute;left:-5%;right:-5%;bottom:-130px;height:280px;background:rgba(142,244,237,.15);border-radius:50%;animation:heroWave 7s ease-in-out infinite alternate}.hero-content{position:relative;z-index:2;max-width:900px}.hero .kicker{color:#a6fffa}.hero h1{font:800 clamp(3.2rem,6.5vw,6.5rem)/.9 Manrope,sans-serif;letter-spacing:-.075em;color:#e4ffff;margin:0 0 23px}.hero p{font-size:1.12rem;line-height:1.82;color:#e0fbfb;max-width:800px}.hero-badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:25px}.badge{padding:10px 14px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.26);color:#efffff;font-size:.82rem;font-weight:700;backdrop-filter:blur(10px)}
@keyframes waveLines{from{transform:translateX(-4%) rotate(-7deg)}to{transform:translateX(4%) rotate(-7deg)}}@keyframes heroWave{from{transform:translateX(-2%) rotate(-1deg)}to{transform:translateX(2%) rotate(1deg)}}
/* Equal-height educational split */
.bloom-grid{position:relative;z-index:2;display:grid;grid-template-columns:1fr 1.18fr;gap:24px;align-items:stretch}.bloom-panel{height:100%;min-height:430px;background:rgba(255,255,255,.70);border:1px solid rgba(211,235,236,.95);border-radius:25px;box-shadow:var(--shadow);padding:30px;backdrop-filter:blur(16px)}.bloom-panel h3{font:800 1.5rem/1.2 Manrope;color:#123f49;margin:0 0 16px}.bloom-panel p{font-size:1rem;line-height:1.82;color:#66848b;margin:0}.bloom-panel .important{margin-top:22px;padding:16px 18px;background:#e5f9f9;border-left:4px solid #13aab2;border-radius:0 16px 16px 0;color:#3f6971;line-height:1.7}.bloom-visual{height:100%;min-height:430px;background:rgba(255,255,255,.70);border:1px solid rgba(211,235,236,.95);border-radius:25px;box-shadow:var(--shadow);padding:10px;backdrop-filter:blur(16px);display:flex;align-items:stretch}.bloom-visual img{display:block;width:100%;height:100%;min-height:408px;object-fit:cover;object-position:center;border-radius:18px}
/* Feature cards */
.feature-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:17px}.feature{position:relative;z-index:2;padding:26px;background:rgba(255,255,255,.70);border:1px solid rgba(211,235,236,.95);border-radius:22px;box-shadow:var(--shadow);min-height:200px;transition:.2s;backdrop-filter:blur(14px)}.feature:hover{transform:translateY(-4px);box-shadow:0 25px 65px rgba(16,91,102,.13)}.feature .icon{font-size:1.65rem;margin-bottom:15px}.feature h3{font:800 1.16rem Manrope;color:#123f49;margin:0 0 9px}.feature p{font-size:.96rem;line-height:1.7;color:#66848b;margin:0}
/* Map */
.map-card{position:relative;z-index:2;background:rgba(255,255,255,.74);border-radius:24px;padding:5px;border:2px solid #9bcfd4;box-shadow:0 18px 48px rgba(15,91,101,.10);backdrop-filter:blur(16px);overflow:hidden}
.map-study-label{height:48px;display:flex;align-items:center;gap:12px;padding:0 2px;color:#174b56}.map-study-label span{font:800 .66rem Manrope,sans-serif;letter-spacing:.15em;color:#0a9ba8}.map-study-label b{font-size:.95rem}.map-legend{display:flex;align-items:center;gap:13px;flex-wrap:wrap;color:#4f747b;font-size:.86rem;padding:13px 14px}.legend-gradient{width:180px;height:10px;border-radius:999px;background:linear-gradient(90deg,#09264a,#0b76a3,#19c8c6,#b7e76b,#ffe27c)}.risk-dot{width:12px;height:12px;border-radius:50%;background:#ff5364;border:2px solid white}
/* Inputs */
div[data-testid="stNumberInput"]{position:relative!important;z-index:4!important}
div[data-testid="stNumberInput"] input,div[data-testid="stNumberInput"] input[type="number"]{border-radius:13px!important;background:rgba(255,255,255,.94)!important;color:#174b56!important;border:1.5px solid rgba(75,169,179,.46)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,1),0 7px 18px rgba(15,91,101,.06)!important;font-size:1.02rem!important;font-weight:800!important;height:44px!important}
div[data-testid="stNumberInput"] button{width:32px!important;height:32px!important;margin-right:4px!important;border-radius:9px!important;background:rgba(222,249,249,.98)!important;color:#087f8d!important;border:1px solid rgba(88,181,188,.58)!important;box-shadow:0 3px 8px rgba(15,91,101,.08)!important;opacity:1!important}
div[data-testid="stNumberInput"] button:hover{background:#c9f3f2!important;color:#075d6c!important}
div[data-testid="stSelectbox"] [data-baseweb="select"]>div{border-radius:14px!important;background:rgba(255,255,255,.72)!important;border:1px solid rgba(135,192,198,.58)!important;color:#174b56!important}
.stNumberInput label,.stSelectbox label,.stToggle label{font-weight:800!important;color:#285761!important;font-size:.9rem!important}
div[data-testid="stNumberInput"] [data-baseweb="input"],
div[data-testid="stNumberInput"] [data-baseweb="input"]>div{
    background:rgba(255,255,255,.94)!important;
    border:1.5px solid rgba(75,169,179,.46)!important;
    border-radius:13px!important;
    box-shadow:0 7px 18px rgba(15,91,101,.06)!important;
}
div[data-testid="stNumberInput"] [data-baseweb="input"] input{
    background:transparent!important;
    color:#174b56!important;
    -webkit-text-fill-color:#174b56!important;
}

/* Coordinate checker */
.checker-input-card{position:relative;z-index:3;background:rgba(255,255,255,.84);border:1.5px solid rgba(113,190,197,.40);border-radius:25px;box-shadow:0 16px 42px rgba(15,91,101,.09);padding:25px 27px 22px;backdrop-filter:blur(18px)}
.entered-line{margin-top:10px;padding:13px 15px;border-radius:14px;background:#eafafa;border:1px solid #d4eeee;color:#53777e;font-size:.92rem;line-height:1.6}
.nearest-line{margin-top:8px;color:#66848b;font-size:.92rem;line-height:1.6}
/* Result grids */
.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}.result-grid div{padding:12px 14px;background:linear-gradient(145deg,#f5fdfd,#e9f8f8);border:1px solid #cbe8e9;border-radius:15px}.result-grid span{display:block;font-size:.76rem;color:#78959b;margin-bottom:5px}.result-grid b{font-size:.98rem;color:#194b55}
/* Tables */
.table-wrap{position:relative;z-index:2;border-radius:20px;overflow:hidden;border:1px solid var(--line);box-shadow:var(--shadow);background:#fff}.table-wrap table{width:100%;border-collapse:collapse;font-size:.90rem}.table-wrap th{background:#0e5663;color:#fff;text-align:left;padding:14px 15px}.table-wrap td{padding:13px 15px;border-top:1px solid #e3eeee;color:#315b63;background:#fff}.table-wrap tr:nth-child(even) td{background:#f7fcfc}
/* Method */
.step{position:relative;z-index:2;display:grid;grid-template-columns:58px 1fr;gap:16px;align-items:center;padding:16px 18px;margin:10px 0;border:1.5px solid #9ccfd3;border-radius:18px;background:rgba(255,255,255,.80);box-shadow:0 10px 26px rgba(15,91,101,.08);backdrop-filter:blur(12px)}.step-num{width:48px;height:48px;border-radius:14px;background:#0d5264;color:#fff;display:grid;place-items:center;font:800 .92rem Manrope;border:2px solid #082f3b}.step h3{font:800 1.22rem Manrope;color:#123f49;margin:0 0 4px}.step p{color:#66848b;line-height:1.55;margin:0;font-size:.96rem}
/* Downloads */
.download-panel{position:relative;z-index:2;min-height:150px;box-sizing:border-box;background:linear-gradient(135deg,#087f90,#18aeb1);border-radius:26px;padding:28px;box-shadow:0 25px 70px rgba(8,126,141,.18)}.download-panel h3{font:800 1.55rem Manrope;color:#fff;margin:0 0 8px}.download-panel p{color:#d9ffff;margin:0 0 18px;font-size:1rem;line-height:1.7}.download-panel .stDownloadButton>button{background:rgba(255,255,255,.82)!important;color:#087381!important;border:1px solid rgba(255,255,255,.95)!important}.download-panel .stDownloadButton>button p,.download-panel .stDownloadButton>button span{color:#087381!important}
.note{position:relative;z-index:2;padding:18px 20px;background:#e7fafa;border-left:4px solid #13aab2;border-radius:0 18px 18px 0;color:#4d7078;line-height:1.75;font-size:.96rem}.footer{position:relative;z-index:2;margin-top:60px;padding-top:22px;border-top:1px solid var(--line);color:#78959b;font-size:.78rem}

.data-metrics .metric{min-height:136px!important;height:136px!important}.data-metrics .metric .value{font-size:1.48rem!important}.data-source{background:rgba(255,255,255,.88)!important;border:1.5px solid rgba(113,190,197,.40)!important}
.data-source p{font-size:1rem!important;line-height:1.75!important}
.data-source h3{font-size:1.5rem!important}
.input-title{font:800 1.3rem Manrope,sans-serif!important;color:#123f49!important;margin:0 0 10px!important}.stNumberInput{margin-bottom:8px!important}.chart-card + div[data-testid="stPlotlyChart"]{margin-top:-10px!important}.download-panel + div .stDownloadButton>button{border:2px solid #184b57!important;background:#ffffff!important;color:#0b5c6d!important;font-weight:800!important}.download-panel + div .stDownloadButton>button:hover{background:#e1f8f8!important}.data-source{padding:24px!important}.table-wrap{margin-top:10px}.table-wrap th,.table-wrap td{font-size:.92rem!important}.plotly .xtick text,.plotly .ytick text{font-size:12px!important;fill:#174b56!important}.plotly .gtitle,.plotly .xtitle,.plotly .ytitle{fill:#174b56!important}
.compare-result{min-height:0;padding:22px!important}
.chart-card{height:auto!important;min-height:0!important;padding:20px!important}
.chart-card h3{font-size:1.35rem!important;margin-bottom:8px!important}.chart-card p{font-size:.96rem!important;line-height:1.55!important}
@media(max-width:1000px){.feature-grid{grid-template-columns:repeat(2,1fr)}.hero{padding:60px 42px}.bloom-grid{grid-template-columns:1fr}.bloom-panel,.bloom-visual{min-height:auto}.bloom-visual img{min-height:0}.result-grid{grid-template-columns:1fr}}
@media(max-width:650px){.block-container{padding:12px 18px 60px!important}.hero{padding:48px 28px;min-height:470px}.hero h1{font-size:3.2rem}.brand-sub{display:none}.feature-grid{grid-template-columns:1fr}.section{padding-top:42px}.section h2{font-size:2.15rem}}
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

st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)
nav_cols = st.columns(7, gap="small")
for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(NAV[page], key=f"nav_{page}", use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
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

def safe(v, digits=3):
    if pd.isna(v):
        return "Unavailable"
    x = float(v)
    if abs(x) < 0.0005:
        return "0.0"
    return f"{x:.{digits}f}"

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
    # The satellite product covers the global ocean, but the project visualization
    # is intentionally centered on the Indian Ocean study area.
    data = latest[latest.latitude.between(-40, 30) & latest.plot_lon.between(20, 120)].copy()
    if data.empty:
        data = latest.copy()

    risk_subset = data[data.risk_flag].copy()
    normal_subset = data[~data.risk_flag].copy()

    # Keep every normal cell visible when possible. The tiny points form the field,
    # while the larger red points make the screened locations immediately visible.
    vmax = max(float(data.chla.quantile(.995)), .01)
    fig = px.scatter_geo(
        normal_subset,
        lat="latitude", lon="plot_lon", color="chla",
        custom_data=["latitude", "longitude", "chla"],
        projection="equirectangular",
        color_continuous_scale=[[0,"#176b96"],[0.55,"#1aa6c0"],[1,"#56c8d1"]],
        range_color=(0, vmax),
    )
    fig.update_traces(
        marker=dict(size=2.15, opacity=.62, line=dict(width=0)),
        hovertemplate="Lat %{customdata[0]:.2f}°<br>Lon %{customdata[1]:.2f}°<br>Chl-a %{customdata[2]:.4f}<extra></extra>"
    )

    if show_risk and not risk_subset.empty:
        risk_fig = px.scatter_geo(
            risk_subset,
            lat="latitude", lon="plot_lon",
            custom_data=["latitude", "longitude", "chla"],
            projection="equirectangular",
        )
        fig.add_trace(risk_fig.data[0])
        fig.data[-1].marker = dict(size=8.5, color="#ff5365", opacity=.98, line=dict(width=1.8, color="#ffffff"))
        fig.data[-1].name = "Potential bloom risk"
        fig.data[-1].hovertemplate = "Potential bloom-risk flag<br>Lat %{customdata[0]:.2f}°<br>Lon %{customdata[1]:.2f}°<br>Chl-a %{customdata[2]:.4f}<extra></extra>"

    fig.update_geos(
        showland=True, landcolor="#dcebe8",
        showocean=True, oceancolor="#edf9fa",
        showcoastlines=True, coastlinecolor="#4f9aa4", coastlinewidth=1.1,
        showcountries=True, countrycolor="#9abdc2",
        bgcolor="#edf9fa",
        lataxis_range=[-40,30], lonaxis_range=[20,120],
        projection_scale=1.08,
        center=dict(lat=-5, lon=70),
    )
    fig.update_layout(
        height=590,
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="#edf9fa",
        plot_bgcolor="#edf9fa",
        font=dict(color="#174b56"),
        showlegend=False,
        coloraxis_colorbar=dict(
            title=dict(text="Chl-a", font=dict(color="#174b56")),
            tickfont=dict(color="#174b56"),
            bgcolor="rgba(255,255,255,.82)", outlinewidth=0,
            thickness=14, len=.58,
        ),
    )
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
    def image_data_uri(path):
        if not path.exists():
            return ""
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    infographic_uri = image_data_uri(INFOGRAPHIC_PATH)
    if infographic_uri:
        st.markdown(f"""
        <div class="bloom-grid">
          <div class="bloom-panel">
            <div class="mini-label">Why this matters</div>
            <h3>Small organisms can create a large environmental signal.</h3>
            <p>Phytoplankton use light and nutrients to grow. When many cells accumulate, ocean colour and chlorophyll-a can change. BloomDetect uses that observable signal to screen locations for potential bloom risk, then points people toward places worth investigating with additional evidence.</p>
            <div class="important"><b>Important:</b> the dashboard is an early-warning support tool, not a species, toxin, or laboratory confirmation system.</div>
          </div>
          <div class="bloom-visual"><img src="{infographic_uri}" alt="How an algal bloom can develop and how satellite observations provide a chlorophyll signal"></div>
        </div>
        """, unsafe_allow_html=True)
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
    section("01 · Spatial map", "See where the signal changes.", "The map stays focused on the Indian Ocean study area used for this project. Blue marks the latest chlorophyll-a field; red marks model-screened potential bloom-risk locations.")
    a,b=st.columns([1.55,.7],gap="medium")
    with a:
        st.markdown('<div class="map-study-label"><span>STUDY AREA</span><b>Indian Ocean · Arabian Sea · Bay of Bengal</b></div>',unsafe_allow_html=True)
    with b:
        show=st.toggle("Show potential-risk overlay",True)
    st.markdown('<div class="map-card">',unsafe_allow_html=True)
    st.plotly_chart(map_figure("Indian Ocean",show),use_container_width=True,config={"scrollZoom":False,"displaylogo":False,"modeBarButtonsToRemove":["lasso2d","select2d"]})
    st.markdown('<div class="map-legend"><span>Chlorophyll-a field</span><span class="legend-gradient"></span><span>Lower → higher within this view</span><span class="risk-dot"></span><span>Potential bloom risk</span></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="note"><b>How to use this:</b> zoom to a region, identify a flagged coordinate, then open <b>Risk Checker</b> to inspect the values behind that location.</div>',unsafe_allow_html=True)

# -----------------------------
# CHECKER
# -----------------------------
elif st.session_state.page == "checker":
    section("02 · Coordinate screening", "Check a location.", "Enter a latitude and longitude. BloomDetect finds the nearest processed 0.25° grid cell and reports its latest screening result.")
    left,right=st.columns([.78,1.22],gap="large")
    with left:
        st.markdown('<div class="checker-input-card">',unsafe_allow_html=True)
        lat=st.number_input("Latitude",min_value=-90.0,max_value=90.0,value=18.0,step=.25,format="%.2f",key="checker_lat")
        lon=st.number_input("Longitude",min_value=-180.0,max_value=180.0,value=78.0,step=.25,format="%.2f",key="checker_lon")
        st.markdown(f'<div class="entered-line"><b>Selected:</b> {lat:.2f}° latitude · {lon:.2f}° longitude</div>',unsafe_allow_html=True)
        st.markdown('<div class="nearest-line">Screening uses the nearest processed 0.25° grid cell.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    row=checker_row(lat,lon)
    with right:
        risk_yes=bool(row["risk_flag"])
        status="Potential bloom risk" if risk_yes else "No potential bloom-risk flag"
        status_color="#d94d5e" if risk_yes else "#087f8d"
        prob=probability_text(row.get("risk_probability",np.nan))
        st.markdown(f"""
        <div class="card" style="min-height:330px">
          <div class="mini-label">Nearest processed observation</div>
          <h3 style="font-size:2rem;margin-bottom:8px">{lat:.2f}° · {lon:.2f}°</h3><div class="nearest-line">Nearest processed cell used for screening: {row.latitude:.2f}° · {row.longitude:.2f}°</div>
          <div class="result-grid">
            <div><span>Date</span><b>{row.date.strftime('%d %B %Y')}</b></div>
            <div><span>Chlorophyll-a</span><b>{safe(row.chla)}</b></div>
            <div><span>Screening</span><b style="color:{status_color}">{status}</b></div>
            <div><span>Model probability</span><b>{prob}</b></div>
          </div>
        </div>
        """,unsafe_allow_html=True)
    st.markdown(
        f'<div class="note" style="margin-top:16px"><b>How it works:</b> Your selected coordinate stays visible above. The model checks the nearest processed grid cell, so nearby coordinates can map to the same observation. '
        f' '
        f'The result updates immediately when you change the coordinates. '
        f'Nearby inputs can show the same result when they still map to the same grid cell.</div>',
        unsafe_allow_html=True
    )

# -----------------------------
# COMPARE
# -----------------------------
elif st.session_state.page == "compare":
    section("03 · Compare locations", "Compare two places side by side.", "Use this when you want to inspect whether two coordinates show different latest chlorophyll-a signals or screening results.")
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<h3 class="input-title">Location A</h3>',unsafe_allow_html=True)
        a_lat=st.number_input("Latitude A",-90.,90.,18.,.25,format="%.2f",key="a_lat")
        a_lon=st.number_input("Longitude A",-180.,180.,78.,.25,format="%.2f",key="a_lon")
    with c2:
        st.markdown('<h3 class="input-title">Location B</h3>',unsafe_allow_html=True)
        b_lat=st.number_input("Latitude B",-90.,90.,20.,.25,format="%.2f",key="b_lat")
        b_lon=st.number_input("Longitude B",-180.,180.,90.,.25,format="%.2f",key="b_lon")
    ra,rb=checker_row(a_lat,a_lon),checker_row(b_lat,b_lon)
    st.markdown('<div style="height:20px"></div>',unsafe_allow_html=True)
    for name,row in [("Location A",ra),("Location B",rb)]:
        status="Potential bloom risk" if row.risk_flag else "No potential bloom-risk flag"
        st.markdown(f'<div class="card compare-result" style="margin-bottom:14px"><div class="mini-label">{name}</div><h3>{(a_lat if name=="Location A" else b_lat):.2f}° · {(a_lon if name=="Location A" else b_lon):.2f}°</h3><div class="nearest-line">Nearest processed cell used for screening: {row.latitude:.2f}° · {row.longitude:.2f}°</div><div class="result-grid"><div><span>Chlorophyll-a</span><b>{safe(row.chla)}</b></div><div><span>Screening</span><b>{status}</b></div><div><span>Anomaly</span><b>{safe(row.get("chla_anomaly",np.nan))}</b></div><div><span>Recent change</span><b>{safe(row.get("chla_change",np.nan))}</b></div></div></div>',unsafe_allow_html=True)

# -----------------------------
# INSIGHTS
# -----------------------------
elif st.session_state.page == "insights":
    section("04 · Latest-field insights", "What stands out right now?", "These summaries answer practical questions about the latest processed field without repeating the dataset documentation.")
    normal_count=int((~latest.risk_flag).sum()); risk_count=int(latest.risk_flag.sum()); share=100*risk_count/len(latest) if len(latest) else 0
    cols=st.columns(4,gap="medium")
    for col,(l,v,n) in zip(cols,[("Observation cells",f"{len(latest):,}","latest field"),("Potential-risk cells",f"{risk_count:,}","model screening"),("Normal cells",f"{normal_count:,}","latest field"),("Risk share",f"{share:.2f}%","latest field")]):
        with col: metric(l,v,n)
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="card chart-card"><div class="mini-label">Screening composition</div><h3>Normal vs potential bloom-risk</h3><p>Count of the two model screening outcomes in the latest field.</p></div>',unsafe_allow_html=True)
        chart=pd.DataFrame({"Classification":["Normal","Potential bloom risk"],"Cells":[normal_count,risk_count]})
        fig=px.bar(chart,x="Classification",y="Cells",text="Cells",color="Classification",color_discrete_map={"Normal":"#73c9cf","Potential bloom risk":"#ff6472"})
        fig.update_layout(height=300,margin=dict(l=45,r=15,t=12,b=50),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.72)",showlegend=False,font=dict(color="#174b56"),xaxis=dict(title="Screening outcome",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56")),yaxis=dict(title="Number of cells",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56"),gridcolor="#c6d9dc"))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown('<div class="card chart-card"><div class="mini-label">Chlorophyll-a distribution</div><h3>Where values cluster</h3><p>This is the distribution of the current satellite-derived field, not a direct measure of harmfulness.</p></div>',unsafe_allow_html=True)
        fig=px.histogram(latest,x="chla",nbins=30,color_discrete_sequence=["#14a6b1"])
        fig.update_layout(height=300,margin=dict(l=45,r=15,t=12,b=50),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.72)",font=dict(color="#174b56"),xaxis=dict(title="Chlorophyll-a",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56")),yaxis=dict(title="Number of cells",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56"),gridcolor="#c6d9dc"))
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
    section("05 · Project pipeline", "From satellite observation to screening support.", "The project turns an environmental observation into a reproducible sequence of cleaning, temporal context, machine learning and spatial inspection.")
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
    section("06 · Dataset & outputs", "Dataset, fields and useful downloads.", "This page contains the source product, coverage, grid structure, processed fields, interpretation limits and downloads. The other pages intentionally avoid repeating this documentation.")
    st.markdown('<div class="data-metrics">',unsafe_allow_html=True)
    cols=st.columns(4,gap="medium")
    for col,(l,v,n) in zip(cols,[("Product","E06OCM_L4_AC","EOS-06 / OCM-3"),("Grid","0.25°","latitude × longitude"),("Latest cells",f"{len(latest):,}","processed observation"),("Latest date",latest_date.strftime("%d %b %Y"),"processed dataset")]):
        with col: metric(l,v,n)
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="card data-source" style="margin-top:18px"><div class="mini-label">Source & interpretation</div><h3>What the dashboard is built from.</h3><p><b>Satellite product:</b> EOS-06 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC).</p><p><b>Primary variable:</b> chlorophyll-a (<code>chla</code>).</p><p><b>Spatial structure:</b> 0.25° latitude × 0.25° longitude global-ocean grid.</p><p><b>Current dashboard view:</b> the latest processed observation represented in the prediction table.</p><p><b>Interpretation:</b> the dataset provides an environmental observation signal. It does not contain confirmed harmful-bloom species or toxin labels.</p></div>',unsafe_allow_html=True)
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
