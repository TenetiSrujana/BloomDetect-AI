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

# -----------------------------
# DATA
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
    numeric = ["risk_probability", "model_score", "previous_chla", "historical_baseline", "recent_mean", "recent_max", "chla_anomaly", "chla_change"]
    for c in numeric:
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

# Project screening thresholds used when the proxy target was created.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

PAGES = ["home", "map", "checker", "hotspots", "insights", "method", "data"]
NAV = {
    "home": "Home",
    "map": "Risk Map",
    "checker": "Risk Checker",
    "hotspots": "Hotspots",
    "insights": "Insights",
    "method": "How It Works",
    "data": "Data",
}
if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"

# -----------------------------
# CSS
# -----------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#103f49;--muted:#5f8088;--aqua:#0ca7b2;--deep:#07566a;--line:#9fcfd4;--pale:#f1fbfb;--shadow:0 14px 38px rgba(9,80,94,.09)}
html,body,[data-testid="stAppViewContainer"]{background:#f1fbfb!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif!important}
.stApp{background:linear-gradient(180deg,#fbffff 0%,#effafa 50%,#fbffff 100%)!important;overflow-x:hidden}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}
.block-container{max-width:1260px!important;padding:24px 34px 72px!important}
/* Subtle ocean motion, deliberately lightweight so the app stays responsive. */
.stApp:before{content:"";position:fixed;left:-10%;right:-10%;bottom:-145px;height:260px;z-index:0;pointer-events:none;background:radial-gradient(ellipse at 15% 55%,rgba(28,193,201,.13) 0 17%,transparent 18%),radial-gradient(ellipse at 55% 45%,rgba(69,221,218,.10) 0 20%,transparent 21%),radial-gradient(ellipse at 90% 60%,rgba(15,171,191,.10) 0 18%,transparent 19%);animation:waterMove 12s ease-in-out infinite alternate}
.stApp:after{content:"◦   ·     ◦      ·     ◦      ·";position:fixed;left:5%;bottom:-90px;z-index:0;pointer-events:none;color:rgba(8,153,169,.10);font-size:24px;letter-spacing:55px;animation:bubbleRise 20s linear infinite}
@keyframes waterMove{from{transform:translateX(-2%)}to{transform:translateX(2%)}}
@keyframes bubbleRise{from{transform:translateY(80px);opacity:.02}50%{opacity:.13}to{transform:translateY(-100vh);opacity:.01}}
/* Brand */
.brand-bar{position:relative;z-index:10;display:flex;align-items:center;padding:15px 20px;border:1.5px solid #c5e4e6;background:rgba(255,255,255,.86);border-radius:22px;box-shadow:var(--shadow);backdrop-filter:blur(18px)}
.brand-left{display:flex;align-items:center;gap:13px}.logo{width:48px;height:48px;border-radius:16px;background:linear-gradient(145deg,#28cdd0,#087b8e);display:grid;place-items:center;color:white;font-size:20px;font-weight:800;box-shadow:0 10px 24px rgba(9,144,157,.18)}
.brand-name{font:800 1.22rem Manrope,sans-serif;color:#103f49;letter-spacing:-.03em}.brand-sub{font-size:.73rem;color:#78959b;margin-top:2px}
/* Buttons */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{height:46px!important;border-radius:14px!important;background:#ffffff!important;border:2px solid #164e5b!important;color:#123f49!important;font-weight:800!important;font-size:.91rem!important;box-shadow:0 6px 16px rgba(15,70,82,.09)!important;transition:all .18s ease!important}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:#e2f8f8!important;border-color:#087d8c!important;transform:translateY(-1px)!important;box-shadow:0 10px 22px rgba(15,91,101,.13)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#07576a,#0b8b99)!important;border-color:#063d4b!important;color:#fff!important;box-shadow:0 9px 22px rgba(8,82,99,.22),0 0 0 3px rgba(16,174,183,.10)!important}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#064b5c,#087c8b)!important;color:#fff!important}
.stButton>button:focus,.stButton>button:active{outline:none!important;box-shadow:0 0 0 3px rgba(16,174,183,.16),0 9px 22px rgba(15,91,101,.12)!important}
.stButton>button p,.stButton>button span,.stDownloadButton>button p,.stDownloadButton>button span{color:inherit!important}
.nav-wrap{position:relative;z-index:12;margin:14px 0 8px}.nav-spacer{height:2px}
/* Type */
.kicker{font:800 .70rem Manrope,sans-serif;letter-spacing:.19em;text-transform:uppercase;color:#0797a5;margin-bottom:12px}
.section{position:relative;z-index:2;padding:36px 0 18px}.section h2{font:800 clamp(2.25rem,4vw,4.0rem)/1.02 Manrope,sans-serif;letter-spacing:-.06em;color:#103f49;margin:0 0 14px}.section p{color:#5f8088;line-height:1.62;margin:0;max-width:1120px;font-size:1rem}
/* Cards */
.card{position:relative;z-index:2;background:rgba(255,255,255,.86);border:1.5px solid #a9d6da;border-radius:23px;box-shadow:var(--shadow);padding:22px;backdrop-filter:blur(12px)}
.card h3{font:800 1.35rem Manrope,sans-serif;color:#123f49;margin:0 0 9px}.card p{color:#66848b;line-height:1.62;margin:0;font-size:.96rem}.mini-label{font:800 .67rem Manrope,sans-serif;letter-spacing:.15em;text-transform:uppercase;color:#6d8c93;margin-bottom:8px}
.metric{position:relative;z-index:2;min-height:116px;height:100%;box-sizing:border-box;padding:18px;background:linear-gradient(145deg,#ffffff,#eaf8f8);border:1.5px solid #a6d5da;border-radius:19px;box-shadow:0 11px 28px rgba(15,91,101,.08);display:flex;flex-direction:column;justify-content:center}.metric .label{font:800 .66rem Manrope,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:#6d8c93}.metric .value{font:800 1.48rem Manrope,sans-serif;color:#123f49;margin-top:7px;white-space:nowrap}.metric .note{font-size:.75rem;color:#78959b;margin-top:5px}
/* Hero */
.hero{position:relative;z-index:2;overflow:hidden;min-height:470px;border-radius:31px;padding:70px 62px;background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);box-shadow:0 28px 72px rgba(6,86,100,.16)}
.hero:before{content:"";position:absolute;inset:-25%;background:repeating-radial-gradient(ellipse at 20% 115%,transparent 0 55px,rgba(181,255,251,.12) 57px 59px,transparent 61px 105px);transform:rotate(-7deg);animation:waveLines 14s linear infinite}.hero:after{content:"";position:absolute;left:-5%;right:-5%;bottom:-130px;height:270px;background:rgba(142,244,237,.14);border-radius:50%;animation:heroWave 8s ease-in-out infinite alternate}.hero-content{position:relative;z-index:2;max-width:870px}.hero .kicker{color:#a6fffa}.hero h1{font:800 clamp(3.2rem,6vw,6.0rem)/.9 Manrope,sans-serif;letter-spacing:-.075em;color:#e4ffff;margin:0 0 22px}.hero p{font-size:1.08rem;line-height:1.78;color:#e0fbfb;max-width:790px}.hero-badges{display:flex;gap:9px;flex-wrap:wrap;margin-top:23px}.badge{padding:9px 13px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.80rem;font-weight:700}
@keyframes waveLines{from{transform:translateX(-4%) rotate(-7deg)}to{transform:translateX(4%) rotate(-7deg)}}@keyframes heroWave{from{transform:translateX(-2%) rotate(-1deg)}to{transform:translateX(2%) rotate(1deg)}}
/* Educational section */
.bloom-grid{position:relative;z-index:2;display:grid;grid-template-columns:1fr 1.15fr;gap:22px;align-items:stretch}.bloom-panel,.bloom-visual{background:rgba(255,255,255,.80);border:1.5px solid #b9dfe2;border-radius:23px;box-shadow:var(--shadow);padding:26px}.bloom-panel{min-height:360px}.bloom-panel h3{font:800 1.4rem/1.2 Manrope;color:#123f49;margin:0 0 13px}.bloom-panel p{font-size:.97rem;line-height:1.75;color:#66848b;margin:0}.bloom-panel .important{margin-top:20px;padding:14px 16px;background:#e5f9f9;border-left:4px solid #13aab2;border-radius:0 14px 14px 0;color:#3f6971;line-height:1.62}.bloom-visual{padding:9px;min-height:360px}.bloom-visual img{display:block;width:100%;height:100%;min-height:340px;object-fit:cover;border-radius:17px}
.feature-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.feature{position:relative;z-index:2;padding:23px;background:rgba(255,255,255,.80);border:1.5px solid #b5dde0;border-radius:20px;box-shadow:var(--shadow);min-height:175px;transition:.2s}.feature:hover{transform:translateY(-3px)}.feature .icon{font-size:1.5rem;margin-bottom:12px}.feature h3{font:800 1.1rem Manrope;color:#123f49;margin:0 0 7px}.feature p{font-size:.92rem;line-height:1.62;color:#66848b;margin:0}
/* Map */
.map-card{position:relative;z-index:2;background:#eefafa;border-radius:23px;padding:4px;border:2px solid #8fcbd1;box-shadow:0 16px 42px rgba(15,91,101,.09);overflow:hidden}.map-study-label{height:45px;display:flex;align-items:center;gap:11px;padding:0 12px;color:#174b56}.map-study-label span{font:800 .65rem Manrope,sans-serif;letter-spacing:.15em;color:#0a9ba8}.map-study-label b{font-size:.93rem}.map-legend{display:flex;align-items:center;gap:11px;flex-wrap:wrap;color:#4f747b;font-size:.83rem;padding:12px 14px}.legend-blue{width:16px;height:10px;border-radius:4px;background:#2384a5}.risk-dot{width:12px;height:12px;border-radius:50%;background:#f04e5e;border:2px solid white}
/* Inputs */
.coord-label{font:800 .80rem Manrope,sans-serif;color:#285761;margin:0 0 6px!important}.input-help{color:#6b8b92;font-size:.78rem;margin-top:5px}.checker-grid{display:grid;grid-template-columns:.9fr 1.1fr;gap:28px;align-items:start}.checker-form{padding-top:2px}.entered-line{margin-top:15px;padding:12px 14px;border-radius:14px;background:#e7f8f8;border:1px solid #c9e9eb;color:#4f747b;font-size:.88rem;line-height:1.5}.nearest-line{color:#66848b;font-size:.88rem;line-height:1.5}.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-top:15px}.result-grid div{padding:13px 14px;background:linear-gradient(145deg,#f5fdfd,#e9f8f8);border:1px solid #c7e5e7;border-radius:14px;min-height:59px}.result-grid span{display:block;font-size:.72rem;color:#78959b;margin-bottom:5px}.result-grid b{font-size:.94rem;color:#194b55}.status-box{margin-top:15px;padding:14px 16px;border-radius:15px;border:2px solid}.status-box.yes{background:#fff0f2;border-color:#f06a78;color:#a62d3c}.status-box.no{background:#eafaf4;border-color:#49b995;color:#176f58}.status-box strong{font:800 .92rem Manrope}.signal-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:15px}.signal{padding:12px 13px;border-radius:14px;background:#f7fcfc;border:1px solid #c9e5e7}.signal .slabel{font-size:.70rem;color:#78959b}.signal .sval{font:800 .94rem Manrope;color:#164a55;margin-top:4px}.note{position:relative;z-index:2;margin-top:17px;padding:13px 16px;background:#e6f8f8;border-left:4px solid #11a9b2;border-radius:0 14px 14px 0;color:#52757c;font-size:.88rem;line-height:1.6}
/* Tables */
.table-wrap{position:relative;z-index:2;border-radius:18px;overflow:hidden;border:1px solid #c5e3e5;box-shadow:var(--shadow);background:#fff}.table-wrap table{width:100%;border-collapse:collapse;font-size:.88rem}.table-wrap th{background:#0e5663;color:#fff;text-align:left;padding:12px 14px}.table-wrap td{padding:11px 14px;border-top:1px solid #e3eeee;color:#315b63;background:#fff}.table-wrap tr:nth-child(even) td{background:#f7fcfc}
/* Method */
.step{position:relative;z-index:2;display:grid;grid-template-columns:52px 1fr;gap:15px;align-items:center;padding:15px 17px;margin:9px 0;border:2px solid #174e5b;border-radius:17px;background:linear-gradient(100deg,rgba(255,255,255,.96),rgba(232,249,249,.90));box-shadow:0 9px 23px rgba(15,91,101,.07)}.step-num{width:43px;height:43px;border-radius:12px;background:#0d5264;color:#fff;display:grid;place-items:center;font:800 .85rem Manrope;border:2px solid #082f3b}.step h3{font:800 1.12rem Manrope;color:#123f49;margin:0 0 4px}.step p{color:#66848b;line-height:1.5;margin:0;font-size:.92rem}
/* Downloads */
.download-panel{height:132px;box-sizing:border-box;padding:21px;border-radius:19px;background:linear-gradient(135deg,#087b8b,#13adb3);border:2px solid #07576a;box-shadow:0 13px 28px rgba(8,91,102,.16);color:white}.download-panel h3{font:800 1.2rem Manrope;color:white;margin:0 0 7px}.download-panel p{font-size:.88rem;color:#e5ffff;line-height:1.45;margin:0}
/* Charts */
.chart-card{padding:18px!important;min-height:112px!important}.chart-card h3{font-size:1.25rem!important;margin-bottom:6px!important}.chart-card p{font-size:.90rem!important;line-height:1.5!important}.plotly .xtick text,.plotly .ytick text{font-size:12px!important;fill:#174b56!important}.plotly .gtitle,.plotly .xtitle,.plotly .ytitle{fill:#174b56!important}
.footer{position:relative;z-index:2;border-top:1px solid #d5ebed;margin-top:44px;padding-top:18px;color:#76959b;font-size:.72rem}
@media(max-width:900px){.block-container{padding:18px 18px 55px!important}.feature-grid{grid-template-columns:1fr 1fr}.bloom-grid,.checker-grid{grid-template-columns:1fr}.signal-grid{grid-template-columns:1fr 1fr}.hero{padding:48px 35px}.section{padding-top:30px}}
@media(max-width:620px){.feature-grid{grid-template-columns:1fr}.signal-grid{grid-template-columns:1fr}.hero h1{font-size:3rem}.block-container{padding:15px 12px 45px!important}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# NAVIGATION
# -----------------------------
st.markdown('<div class="brand-bar"><div class="brand-left"><div class="logo">≈</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Satellite-based potential bloom-risk screening</div></div></div></div>', unsafe_allow_html=True)
nav_cols = st.columns(7, gap="small")
for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(NAV[page], key=f"nav_{page}", use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()

# -----------------------------
# HELPERS
# -----------------------------
def metric(label, value, note):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)

def section(kicker, title, copy):
    st.markdown(f'<div class="section"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{copy}</p></div>', unsafe_allow_html=True)

def safe(v, digits=3):
    if pd.isna(v): return "Unavailable"
    x=float(v)
    if abs(x)<0.0005: return "0.0"
    return f"{x:.{digits}f}"

def probability_text(v):
    if pd.isna(v): return "Unavailable"
    x=float(v)
    if x<=1: x*=100
    return f"{x:.1f}%"

def nearest_row(lat, lon):
    # Normalize longitude distance at the dateline even though this project's view is far from it.
    dlat=(latest["latitude"].to_numpy()-lat)**2
    dlon=(np.abs(latest["longitude"].to_numpy()-lon).clip(0,180))**2
    idx=int(np.argmin(dlat+dlon))
    return latest.iloc[idx]

def study_area(d):
    return d[d["latitude"].between(-40,30) & d["plot_lon"].between(20,120)].copy()

def map_figure(show_risk=True):
    """Fast study-area map: 1-degree blue background cells + all red risk cells."""
    data=study_area(latest)
    if data.empty: data=latest.copy()
    # Aggregate normal cells to ~1° cells. This makes the blue ocean field visible while
    # keeping the browser responsive. All risk cells remain individually visible.
    normal=data[~data.risk_flag].copy()
    normal["lat_bin"]=(np.floor(normal.latitude)+0.5).round(2)
    normal["lon_bin"]=(np.floor(normal.plot_lon)+0.5).round(2)
    blue=(normal.groupby(["lat_bin","lon_bin"],as_index=False)
          .agg(chla=("chla","mean")))
    risk_data=data[data.risk_flag].copy()
    fig=px.scatter_geo(
        blue,lat="lat_bin",lon="lon_bin",color="chla",custom_data=["lat_bin","lon_bin","chla"],
        projection="equirectangular",color_continuous_scale=[[0,"#0c5b87"],[.45,"#148db1"],[1,"#39c4c8"]],
        range_color=(0,max(float(data.chla.quantile(.995)),.01)))
    fig.update_traces(marker=dict(size=7.5,opacity=.80,line=dict(width=0)),
        hovertemplate="Lat %{customdata[0]:.1f}°<br>Lon %{customdata[1]:.1f}°<br>Mean Chl-a %{customdata[2]:.4f}<extra></extra>")
    if show_risk and not risk_data.empty:
        rf=px.scatter_geo(risk_data,lat="latitude",lon="plot_lon",custom_data=["latitude","longitude","chla"])
        fig.add_trace(rf.data[0])
        fig.data[-1].marker=dict(size=8.5,color="#ef4f5e",opacity=.98,line=dict(width=1.5,color="#ffffff"))
        fig.data[-1].name="Potential bloom risk"
        fig.data[-1].hovertemplate="Potential bloom-risk flag<br>Lat %{customdata[0]:.2f}°<br>Lon %{customdata[1]:.2f}°<br>Chl-a %{customdata[2]:.4f}<extra></extra>"
    fig.update_geos(showland=True,landcolor="#dceae8",showocean=True,oceancolor="#dff7f8",
        showcoastlines=True,coastlinecolor="#3f8d98",coastlinewidth=1.1,showcountries=True,countrycolor="#93b9be",
        bgcolor="#dff7f8",lataxis_range=[-40,30],lonaxis_range=[20,120],projection_scale=1.08,center=dict(lat=-5,lon=70))
    fig.update_layout(height=570,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="#dff7f8",plot_bgcolor="#dff7f8",
        font=dict(color="#174b56"),showlegend=False,
        coloraxis_colorbar=dict(title=dict(text="Chl-a",font=dict(color="#174b56")),tickfont=dict(color="#174b56"),
        bgcolor="rgba(255,255,255,.86)",outlinewidth=0,thickness=13,len=.55))
    return fig

def region_name(lat, lon):
    if 5 <= lat <= 30 and 45 <= lon <= 75: return "Arabian Sea"
    if 0 <= lat <= 25 and 75 < lon <= 100: return "Bay of Bengal"
    if -30 <= lat < 5 and 40 <= lon <= 100: return "Southern Indian Ocean"
    if 5 <= lat <= 30 and 75 < lon <= 120: return "Northern Indian Ocean"
    return "Other study area"

def image_data_uri(path):
    if not path.exists(): return ""
    mime="image/png" if path.suffix.lower()==".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

# -----------------------------
# HOME
# -----------------------------
if st.session_state.page == "home":
    # HOME ONLY: visual and practical landing-page content.
    st.markdown("""
    <style>
    .home-hero-wrap{position:relative;z-index:2;overflow:hidden;border-radius:31px;background:linear-gradient(135deg,#063d51 0%,#076d7e 52%,#10a9ad 100%);box-shadow:0 28px 72px rgba(6,86,100,.18);border:1px solid rgba(180,255,250,.18)}
    .home-hero-wrap:before{content:"";position:absolute;inset:-35%;background:repeating-radial-gradient(ellipse at 18% 112%,transparent 0 52px,rgba(191,255,251,.12) 54px 57px,transparent 59px 105px);transform:rotate(-7deg);animation:homeWave 15s linear infinite}
    .home-hero-wrap:after{content:"";position:absolute;left:-10%;right:-10%;bottom:-175px;height:300px;border-radius:50%;background:rgba(153,250,242,.12);animation:homeFloat 8s ease-in-out infinite alternate}
    .home-hero-grid{position:relative;z-index:2;display:grid;grid-template-columns:1.2fr .8fr;gap:28px;align-items:stretch;padding:58px 58px 50px}
    .home-hero-copy{display:flex;flex-direction:column;justify-content:center}
    .home-hero-kicker{font:800 .72rem Manrope,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:#a9fffa;margin-bottom:14px}
    .home-hero-title{font:800 clamp(3rem,6vw,5.8rem)/.91 Manrope,sans-serif;letter-spacing:-.075em;color:#e8ffff;margin:0 0 21px}
    .home-hero-text{font-size:1.08rem;line-height:1.78;color:#ddfbfb;max-width:760px;margin:0}
    .home-pills{display:flex;flex-wrap:wrap;gap:9px;margin-top:22px}
    .home-pill{padding:9px 13px;border-radius:999px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.80rem;font-weight:800}
    .home-ocean-card{position:relative;min-height:330px;border-radius:24px;overflow:hidden;border:1px solid rgba(220,255,253,.35);background:linear-gradient(180deg,rgba(255,255,255,.08),rgba(0,30,45,.24)),url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1100&q=85') center/cover}
    .home-ocean-card:before{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,65,82,.04),rgba(0,39,55,.52));}
    .home-ocean-card:after{content:"";position:absolute;left:-15%;right:-15%;bottom:25px;height:80px;border-top:2px solid rgba(198,255,251,.38);border-radius:50%;animation:homeSea 5s ease-in-out infinite alternate}
    .home-ocean-label{position:absolute;z-index:2;left:18px;bottom:18px;right:18px;padding:14px 16px;border-radius:16px;background:rgba(2,42,57,.62);border:1px solid rgba(220,255,253,.24);backdrop-filter:blur(10px);color:#eaffff}
    .home-ocean-label b{display:block;font:800 1rem Manrope,sans-serif;margin-bottom:3px}.home-ocean-label span{font-size:.78rem;color:#c7f3f3}
    .home-live-row{position:relative;z-index:2;display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:0 58px 50px}
    .home-live{padding:17px 18px;border-radius:18px;background:rgba(255,255,255,.11);border:1px solid rgba(216,255,252,.22);backdrop-filter:blur(9px)}
    .home-live .v{font:800 1.45rem Manrope,sans-serif;color:#f1ffff}.home-live .l{font-size:.72rem;color:#bfe8e8;margin-top:4px}
    @keyframes homeWave{from{transform:translateX(-5%) rotate(-7deg)}to{transform:translateX(5%) rotate(-7deg)}}
    @keyframes homeFloat{from{transform:translateX(-2%) rotate(-1deg)}to{transform:translateX(2%) rotate(1deg)}}
    @keyframes homeSea{from{transform:translateX(-4%) scaleX(1)}to{transform:translateX(4%) scaleX(1.05)}}
    .home-purpose-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .home-purpose{position:relative;z-index:2;min-height:190px;padding:23px;border-radius:20px;background:rgba(255,255,255,.84);border:1.5px solid #b5dde0;box-shadow:var(--shadow);transition:.2s}
    .home-purpose:hover{transform:translateY(-3px);border-color:#70bcc4}.home-purpose .icon{font-size:1.45rem;margin-bottom:12px}.home-purpose h3{font:800 1.12rem Manrope;color:#123f49;margin:0 0 8px}.home-purpose p{font-size:.93rem;line-height:1.64;color:#66848b;margin:0}
    .home-action{position:relative;z-index:2;padding:24px;border-radius:22px;background:linear-gradient(135deg,#e9fafa,#ffffff);border:1.5px solid #a9d6da;box-shadow:var(--shadow)}
    .home-action h3{font:800 1.35rem Manrope;color:#123f49;margin:0 0 7px}.home-action p{color:#66848b;line-height:1.6;margin:0 0 17px;font-size:.94rem}
    .home-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.home-flow-step{padding:17px;border-radius:17px;background:#fff;border:1.5px solid #b7dfe2;min-height:105px}.home-flow-step b{display:block;font:800 .72rem Manrope;color:#0a99a6;letter-spacing:.12em;margin-bottom:7px}.home-flow-step strong{display:block;font:800 1rem Manrope;color:#164b55;margin-bottom:5px}.home-flow-step span{font-size:.80rem;line-height:1.45;color:#6a898f}
    .home-boundary{position:relative;z-index:2;margin-top:20px;padding:17px 19px;border-radius:17px;background:#e6f8f8;border-left:4px solid #10a9b2;color:#52757c;line-height:1.62;font-size:.89rem}
    @media(max-width:900px){.home-hero-grid{grid-template-columns:1fr;padding:45px 35px 35px}.home-live-row{padding:0 35px 35px}.home-purpose-grid{grid-template-columns:1fr}.home-flow{grid-template-columns:1fr 1fr}}
    @media(max-width:620px){.home-hero-grid{padding:35px 22px 25px}.home-live-row{grid-template-columns:1fr;padding:0 22px 25px}.home-hero-title{font-size:3rem}.home-flow{grid-template-columns:1fr}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="home-hero-wrap">
      <div class="home-hero-grid">
        <div class="home-hero-copy">
          <div class="home-hero-kicker">EOS-06 · OCM-3 · BLOOMDETECT AI</div>
          <h1 class="home-hero-title">See change<br>beneath the surface.</h1>
          <p class="home-hero-text">BloomDetect AI turns a large satellite-derived chlorophyll-a dataset into a practical screening system that helps people find ocean locations whose recent patterns may deserve closer investigation.</p>
          <div class="home-pills">
            <span class="home-pill">🛰️ Satellite screening</span>
            <span class="home-pill">🌊 Spatial context</span>
            <span class="home-pill">🔥 Potential-risk hotspots</span>
            <span class="home-pill">📍 Coordinate investigation</span>
          </div>
        </div>
        <div class="home-ocean-card">
          <div class="home-ocean-label"><b>From millions of observations to a focused shortlist.</b><span>Use the map to see where the current screening signal deserves a closer look.</span></div>
        </div>
      </div>
      <div class="home-live-row">
        <div class="home-live"><div class="v">{len(latest):,}</div><div class="l">latest processed grid cells</div></div>
        <div class="home-live"><div class="v">{int(risk.shape[0]):,}</div><div class="l">current potential-risk flags</div></div>
        <div class="home-live"><div class="v">{latest_date.strftime('%d %b %Y')}</div><div class="l">latest observation date</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    section("01 · Why this matters","Algal blooms are an environmental signal, not just a colour on a map.","Algae and phytoplankton are normal parts of marine ecosystems. When their concentration changes sharply or becomes unusually high, the area can deserve closer ecological attention. Some blooms are harmless; some can be associated with environmental or health impacts. The difficult part is knowing where to look first across a huge ocean area.")
    purpose=[
        ("🌱","Coastal & environmental teams","Use the screening map to narrow a large ocean field into locations that may deserve additional observation, sampling or environmental investigation."),
        ("🐟","Fisheries & coastal stakeholders","Use spatial screening as an early-warning support layer alongside local observations, because changes in water conditions can matter for marine ecosystems and coastal activity."),
        ("🔬","Researchers & students","Inspect the satellite signal, compare locations, download the latest screening output and use the flagged cells as a starting point for deeper analysis and validation."),
    ]
    cols=st.columns(3,gap="medium")
    for col,(ic,h,p) in zip(cols,purpose):
        with col:
            st.markdown(f'<div class="home-purpose"><div class="icon">{ic}</div><h3>{h}</h3><p>{p}</p></div>',unsafe_allow_html=True)

    section("02 · What BloomDetect adds","It helps answer one practical question: where should we look closer?","Instead of asking a person to inspect tens of thousands of ocean cells manually, BloomDetect uses the latest chlorophyll-a observation together with historical and recent temporal context to produce a shortlist of locations for investigation.")
    st.markdown('<div class="home-action"><h3>Turn a large satellite field into an investigation workflow.</h3><p>The dashboard does not replace field sampling or scientific confirmation. It reduces the first-search problem: <b>which locations deserve attention first?</b></p></div>',unsafe_allow_html=True)
    flow=[("01","MAP","See the full study area","Understand the spatial pattern."),("02","HOTSPOTS","Find concentrations","Group nearby flagged cells."),("03","CHECK","Inspect a coordinate","See the nearest processed observation and signals."),("04","ACT","Download evidence","Take the screening shortlist into further analysis.")]
    st.markdown('<div class="home-flow" style="margin-top:13px">'+''.join([f'<div class="home-flow-step"><b>{a}</b><strong>{b}</strong><span>{c}</span></div>' for a,b,c in flow])+'</div>',unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    action_cols=st.columns(4,gap="small")
    actions=[("🌍 Open Risk Map","map"),("🔥 Find Hotspots","hotspots"),("📍 Check a Location","checker"),("📊 View Insights","insights")]
    for col,(label,target) in zip(action_cols,actions):
        with col:
            if st.button(label,key=f"home_action_{target}",use_container_width=True):
                st.session_state.page=target
                st.rerun()

    section("03 · Use the result correctly","A potential-risk flag is a starting point for investigation.","BloomDetect uses satellite-derived chlorophyll-a behaviour as a screening signal. High chlorophyll-a alone does not prove a harmful algal bloom, and the current dataset does not identify harmful species or toxins. The output is therefore designed as early-warning support that should be combined with field observations and other environmental evidence.")
    st.markdown('<div class="home-boundary"><b>Scientific boundary:</b> BloomDetect reports <b>potential bloom risk</b>, not confirmed HAB detection. It is a screening and prioritisation tool, not a laboratory or real-time monitoring system.</div>',unsafe_allow_html=True)

# -----------------------------
# MAP
# -----------------------------
elif st.session_state.page == "map":
    section("01 · Spatial screening","See where the signal changes.","Blue shows the latest chlorophyll-a field across the project study area. Red marks locations screened as potential bloom risk. The view is intentionally limited to the Indian Ocean, Arabian Sea and Bay of Bengal region used for interpretation.")
    left,right=st.columns([1.45,.75],gap="medium")
    with left:
        show=st.toggle("Show potential-risk overlay",value=True,key="map_risk_toggle")
    with right:
        metric("Visible cells",f"{len(study_area(latest)):,}","latest observation")
    st.markdown('<div class="map-card"><div class="map-study-label"><span>STUDY AREA</span><b>Indian Ocean · Arabian Sea · Bay of Bengal</b></div>',unsafe_allow_html=True)
    st.plotly_chart(map_figure(show),use_container_width=True,config={"scrollZoom":False,"displaylogo":False,"modeBarButtonsToRemove":["lasso2d","select2d"]})
    st.markdown('<div class="map-legend"><span>Normal field</span><span class="legend-blue"></span><span>Blue = latest chlorophyll-a field</span><span class="risk-dot"></span><span>Red = potential bloom-risk screening flag</span></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="note"><b>How to use:</b> identify a red cluster, open <b>Hotspots</b> to see its concentration, then use <b>Risk Checker</b> to inspect individual coordinates.</div>',unsafe_allow_html=True)

# -----------------------------
# RISK CHECKER
# -----------------------------
elif st.session_state.page == "checker":
    section("02 · Coordinate screening","Check a location.","Enter the coordinate you want to investigate. BloomDetect keeps your entered coordinate separate from the nearest processed satellite grid cell, so there is no confusion between the two.")
    # Stable session defaults. The form prevents the whole page from rerunning while the user is editing.
    if "input_lat" not in st.session_state: st.session_state.input_lat=18.00
    if "input_lon" not in st.session_state: st.session_state.input_lon=78.00
    if "checked_lat" not in st.session_state: st.session_state.checked_lat=18.00
    if "checked_lon" not in st.session_state: st.session_state.checked_lon=78.00
    with st.form("checker_form",clear_on_submit=False):
        c1,c2=st.columns(2,gap="large")
        with c1:
            st.markdown('<div class="coord-label">Latitude</div>',unsafe_allow_html=True)
            lat=st.number_input("lat",min_value=-90.0,max_value=90.0,step=0.25,format="%.2f",key="input_lat",label_visibility="collapsed")
        with c2:
            st.markdown('<div class="coord-label">Longitude</div>',unsafe_allow_html=True)
            lon=st.number_input("lon",min_value=-180.0,max_value=180.0,step=0.25,format="%.2f",key="input_lon",label_visibility="collapsed")
        submitted=st.form_submit_button("🔎 Check this coordinate",use_container_width=True,type="primary")
    if submitted:
        st.session_state.checked_lat=float(lat)
        st.session_state.checked_lon=float(lon)
    lat=float(st.session_state.checked_lat); lon=float(st.session_state.checked_lon)
    row=nearest_row(lat,lon)
    risk_yes=bool(row["risk_flag"])
    status="POTENTIAL BLOOM-RISK FLAG" if risk_yes else "NO POTENTIAL-RISK FLAG"
    prob=probability_text(row.get("risk_probability",np.nan))
    anomaly=row.get("chla_anomaly",np.nan); change=row.get("chla_change",np.nan)
    with st.container():
        st.markdown('<div class="checker-grid">',unsafe_allow_html=True)
        st.markdown(f'<div class="checker-form"><div class="mini-label">Entered coordinate</div><div class="card"><h3 style="font-size:1.35rem">{lat:.2f}° · {lon:.2f}°</h3><p>Your exact input is preserved here. The satellite product is on a 0.25° grid, so the nearest available cell can have slightly different coordinates.</p></div></div>',unsafe_allow_html=True)
        status_class="yes" if risk_yes else "no"
        st.markdown(f'<div><div class="card"><div class="mini-label">Nearest processed observation</div><h3 style="font-size:1.85rem">{row.latitude:.2f}° · {row.longitude:.2f}°</h3><div class="nearest-line">Input checked: {lat:.2f}° · {lon:.2f}°</div><div class="result-grid"><div><span>Date</span><b>{row.date.strftime("%d %B %Y")}</b></div><div><span>Chlorophyll-a</span><b>{safe(row.chla)}</b></div><div><span>Model probability</span><b>{prob}</b></div><div><span>Region</span><b>{region_name(row.latitude,row.longitude)}</b></div></div><div class="status-box {status_class}"><strong>{"🔴 " if risk_yes else "🟢 "}{status}</strong><br><span>{"This location is included in the current potential-risk screening shortlist." if risk_yes else "This location is not included in the current potential-risk screening shortlist."}</span></div></div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="mini-label" style="margin-top:20px">Signals behind the screening</div>',unsafe_allow_html=True)
    sg=st.columns(4,gap="small")
    signal_items=[("Current Chl-a",safe(row.chla)),("Historical baseline",safe(row.get("historical_baseline",np.nan))), ("Anomaly",safe(anomaly)), ("Recent change",safe(change))]
    for col,(lab,val) in zip(sg,signal_items):
        with col: st.markdown(f'<div class="signal"><div class="slabel">{lab}</div><div class="sval">{val}</div></div>',unsafe_allow_html=True)
    if risk_yes:
        explanation="The latest cell crosses the two proxy screening conditions used to create the potential-risk label: chlorophyll anomaly and recent change are both at or above their project thresholds."
    else:
        checks=[]
        if pd.notna(anomaly): checks.append("anomaly is above" if anomaly>=ANOMALY_THRESHOLD else "anomaly is below")
        if pd.notna(change): checks.append("recent change is above" if change>=CHANGE_THRESHOLD else "recent change is below")
        explanation="This cell is not flagged by the current proxy rule. " + (" and ".join(checks) + " the project threshold." if checks else "Supporting historical signals are unavailable.")
    st.markdown(f'<div class="note"><b>Why this result?</b> {explanation} This is a screening explanation, not proof of a harmful bloom.</div>',unsafe_allow_html=True)
    report=pd.DataFrame([{"input_latitude":lat,"input_longitude":lon,"nearest_latitude":row.latitude,"nearest_longitude":row.longitude,"date":row.date.strftime("%Y-%m-%d"),"chla":row.chla,"historical_baseline":row.get("historical_baseline",np.nan),"chla_anomaly":anomaly,"chla_change":change,"risk_label":row.risk_label,"risk_probability":row.get("risk_probability",np.nan)}])
    st.download_button("⬇ Download this location report",report.to_csv(index=False).encode(),"bloomdetect_location_report.csv","text/csv",use_container_width=True)

# -----------------------------
# HOTSPOTS
# -----------------------------
elif st.session_state.page == "hotspots":
    section("03 · Spatial concentration","Find the areas worth investigating.","Nearby flagged cells are grouped into 2° × 2° zones. This turns 388 individual screening flags into a smaller investigation list without pretending that the zones represent confirmed HAB severity.")
    study=study_area(latest); risk_study=study[study.risk_flag].copy()
    if risk_study.empty:
        st.markdown('<div class="card"><h3>No potential-risk cells in the study area.</h3><p>The latest processed field has no screening flags in the current study view.</p></div>',unsafe_allow_html=True)
    else:
        risk_study["lat_zone"]=(np.floor(risk_study.latitude/2)*2).round(2)
        risk_study["lon_zone"]=(np.floor(risk_study.plot_lon/2)*2).round(2)
        zones=(risk_study.groupby(["lat_zone","lon_zone"],as_index=False).agg(flagged_cells=("risk_flag","size"),mean_chla=("chla","mean"),max_chla=("chla","max")).sort_values(["flagged_cells","mean_chla"],ascending=False).head(15))
        cols=st.columns(3,gap="medium")
        with cols[0]: metric("Flagged cells",f"{len(risk_study):,}","current study area")
        with cols[1]: metric("Hotspot zones",f"{len(zones):,}","2° × 2° grouping")
        with cols[2]: metric("Largest cluster",f"{int(zones.iloc[0].flagged_cells):,}","flagged cells")
        st.markdown('<div class="section" style="padding-top:30px"><div class="kicker">01 · Priority areas</div><h2>Where the flags concentrate.</h2><p>Larger markers mean more flagged cells in that 2° × 2° zone. They are investigation concentrations, not severity rankings.</p></div>',unsafe_allow_html=True)
        hfig=px.scatter_geo(zones,lat="lat_zone",lon="lon_zone",size="flagged_cells",color="flagged_cells",color_continuous_scale=[[0,"#ffd7dc"],[.45,"#ff6a78"],[1,"#b51f3a"]],projection="equirectangular",hover_data={"lat_zone":":.2f","lon_zone":":.2f","flagged_cells":True,"mean_chla":":.4f","max_chla":":.4f"})
        hfig.update_geos(showland=True,landcolor="#dceae8",showocean=True,oceancolor="#dff7f8",showcoastlines=True,coastlinecolor="#4f9aa4",showcountries=True,countrycolor="#9abdc2",bgcolor="#dff7f8",lataxis_range=[-40,30],lonaxis_range=[20,120],center=dict(lat=-5,lon=70),projection_scale=1.08)
        hfig.update_layout(height=510,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="#dff7f8",plot_bgcolor="#dff7f8",font=dict(color="#174b56"),coloraxis_colorbar=dict(title="Flagged cells",tickfont=dict(color="#174b56"),bgcolor="rgba(255,255,255,.85)",thickness=13,len=.52))
        st.plotly_chart(hfig,use_container_width=True,config={"scrollZoom":False,"displaylogo":False})
        show=zones.copy(); show["Zone"]=show.apply(lambda r:f"{r.lat_zone:.2f}°, {r.lon_zone:.2f}°",axis=1); show["Mean Chl-a"]=show.mean_chla.round(4); show["Max Chl-a"]=show.max_chla.round(4)
        show=show[["Zone","flagged_cells","Mean Chl-a","Max Chl-a"]].rename(columns={"flagged_cells":"Flagged cells"})
        st.markdown('<div class="mini-label" style="margin-top:22px">Investigation list</div>',unsafe_allow_html=True)
        st.dataframe(show,use_container_width=True,hide_index=True)
        st.markdown('<div class="note"><b>Recommended workflow:</b> use a hotspot to identify a concentration, then inspect representative coordinates in Risk Checker and combine the satellite signal with field or environmental evidence.</div>',unsafe_allow_html=True)

# -----------------------------
# INSIGHTS
# -----------------------------
elif st.session_state.page == "insights":
    section("04 · Latest-field insights","What stands out right now?","A compact operational view: how many cells are flagged, where the flags concentrate, and which coordinates are worth inspecting next.")
    normal_count=int((~latest.risk_flag).sum()); risk_count=int(latest.risk_flag.sum()); share=100*risk_count/len(latest) if len(latest) else 0
    cols=st.columns(4,gap="medium")
    for col,(l,v,n) in zip(cols,[("Observation cells",f"{len(latest):,}","latest field"),("Potential-risk cells",f"{risk_count:,}","model screening"),("Normal cells",f"{normal_count:,}","latest field"),("Risk share",f"{share:.2f}%","latest field")]):
        with col: metric(l,v,n)
    # Regional watchlist
    regional=[]
    for name,cond in [("Arabian Sea",(latest.latitude.between(5,30)&latest.plot_lon.between(45,75))), ("Bay of Bengal",(latest.latitude.between(0,25)&latest.plot_lon.between(75.01,100))), ("Southern Indian Ocean",(latest.latitude.between(-30,5)&latest.plot_lon.between(40,100)) )]:
        sub=latest[cond]; regional.append((name,len(sub),int(sub.risk_flag.sum()),100*sub.risk_flag.mean() if len(sub) else 0))
    reg=pd.DataFrame(regional,columns=["Region","Cells","Potential-risk cells","Risk share"])
    st.markdown('<div class="mini-label" style="margin-top:23px">Regional watchlist</div>',unsafe_allow_html=True)
    st.dataframe(reg.style.format({"Risk share":"{:.2f}%"}),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="card chart-card"><div class="mini-label">Screening composition</div><h3>Normal vs potential bloom-risk</h3><p>Count of the two latest-field screening outcomes.</p></div>',unsafe_allow_html=True)
        chart=pd.DataFrame({"Classification":["Normal","Potential bloom risk"],"Cells":[normal_count,risk_count]})
        fig=px.bar(chart,x="Classification",y="Cells",text="Cells",color="Classification",color_discrete_map={"Normal":"#4aaeb8","Potential bloom risk":"#ef5362"})
        fig.update_traces(textposition="outside",cliponaxis=False)
        fig.update_layout(height=320,margin=dict(l=55,r=20,t=30,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#ffffff",showlegend=False,font=dict(color="#174b56"),xaxis=dict(title="Screening outcome",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56"),automargin=True),yaxis=dict(title="Number of cells",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56"),gridcolor="#d6e7e8",automargin=True))
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    with c2:
        st.markdown('<div class="card chart-card"><div class="mini-label">Chlorophyll-a distribution</div><h3>Where values cluster</h3><p>Distribution of the current satellite-derived field, not a direct measure of harmfulness.</p></div>',unsafe_allow_html=True)
        fig=px.histogram(latest,x="chla",nbins=30,color_discrete_sequence=["#159eaa"])
        fig.update_layout(height=320,margin=dict(l=55,r=20,t=30,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#ffffff",font=dict(color="#174b56"),xaxis=dict(title="Chlorophyll-a",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56"),automargin=True),yaxis=dict(title="Number of cells",title_font=dict(size=13,color="#174b56"),tickfont=dict(size=12,color="#174b56"),gridcolor="#d6e7e8",automargin=True))
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.markdown('<div class="section" style="padding-top:28px"><div class="kicker">Investigation queue</div><h2>Which coordinates are currently flagged?</h2><p>This is a follow-up list, not a ranking of confirmed HAB severity.</p></div>',unsafe_allow_html=True)
    q=risk.copy().sort_values(["chla"],ascending=False).head(25)
    q=q[[c for c in ["latitude","longitude","chla","chla_anomaly","chla_change","risk_probability","risk_label"] if c in q.columns]].copy()
    for c in ["chla","chla_anomaly","chla_change","risk_probability"]:
        if c in q.columns: q[c]=q[c].round(4)
    st.dataframe(q,use_container_width=True,hide_index=True)

# -----------------------------
# METHOD
# -----------------------------
elif st.session_state.page == "method":
    section("05 · Project pipeline","From satellite observation to screening support.","The pipeline is deliberately visible: each step explains what happens before a potential-risk flag reaches the map.")
    steps=[("01","Observe","EOS-06 OCM-3 analysed chlorophyll-a observations provide the environmental signal."),("02","Clean","Invalid observations are removed and the spatial-temporal records are organized consistently."),("03","Build context","Previous observation, historical baseline, recent mean, recent maximum, anomaly and change features provide temporal context."),("04","Screen","A Decision Tree model screens the prepared features for the project proxy potential-risk label."),("05","Map","The latest screening output is returned to geographic coordinates so spatial concentrations can be inspected."),("06","Investigate","Flagged cells and hotspot concentrations become a shortlist for additional observations and validation.")]
    for n,h,p in steps:
        st.markdown(f'<div class="step"><div class="step-num">{n}</div><div><h3>{h}</h3><p>{p}</p></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="note"><b>Scientific boundary:</b> the current satellite dataset does not contain confirmed harmful-bloom species or toxin labels. The system therefore reports <b>potential bloom risk</b>, not confirmed HAB detection. Satellite screening should be combined with field observations and other environmental evidence.</div>',unsafe_allow_html=True)

# -----------------------------
# DATA
# -----------------------------
elif st.session_state.page == "data":
    section("06 · Dataset & outputs","Dataset, fields and useful downloads.","This page contains the source product, grid structure, processed fields and downloadable outputs. Other pages focus on analysis rather than repeating the documentation.")
    cols=st.columns(4,gap="medium")
    for col,(l,v,n) in zip(cols,[("Product","E06OCM_L4_AC","EOS-06 / OCM-3"),("Grid","0.25°","latitude × longitude"),("Latest cells",f"{len(latest):,}","processed observation"),("Latest date",latest_date.strftime("%d %b %Y"),"processed dataset")]):
        with col: metric(l,v,n)
    st.markdown('<div class="card" style="margin-top:18px"><div class="mini-label">Source & interpretation</div><h3>What the dashboard is built from.</h3><p><b>Satellite product:</b> EOS-06 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC).</p><p><b>Primary variable:</b> chlorophyll-a (<code>chla</code>).</p><p><b>Spatial structure:</b> 0.25° latitude × 0.25° longitude global-ocean grid.</p><p><b>Interpretation:</b> the product provides an environmental observation signal. It does not contain confirmed harmful-bloom species or toxin labels.</p></div>',unsafe_allow_html=True)
    section("Data dictionary","Fields available in the processed file.","The public dashboard uses these fields to inspect observations and screening context.")
    fields=[("latitude","Grid latitude","degrees"),("longitude","Grid longitude","degrees"),("date","Observation date","date"),("chla","Satellite-derived chlorophyll-a","product value"),("risk_label","Model screening result","Normal / Potential Bloom Risk"),("risk_probability","Model probability when available","stored model probability"),("previous_chla","Previous observation","derived feature"),("historical_baseline","Historical baseline","derived feature"),("recent_mean","Recent mean","derived feature"),("recent_max","Recent maximum","derived feature"),("chla_anomaly","Chlorophyll anomaly","derived feature"),("chla_change","Change from previous observation","derived feature")]
    rows="".join([f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a,b,c in fields])
    st.markdown(f'<div class="table-wrap"><table><thead><tr><th>Field</th><th>Meaning</th><th>Type</th></tr></thead><tbody>{rows}</tbody></table></div>',unsafe_allow_html=True)
    section("Useful outputs","Take the screening result with you.","Downloads are kept here so the dashboard remains easy to audit and reuse.")
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown('<div class="download-panel"><h3>Latest observation table</h3><p>Complete latest processed prediction table used by the dashboard.</p></div>',unsafe_allow_html=True)
        st.download_button("⬇ Download latest observations",latest.to_csv(index=False).encode(),"bloomdetect_latest_observations.csv","text/csv",use_container_width=True)
    with c2:
        st.markdown('<div class="download-panel"><h3>Potential-risk shortlist</h3><p>Only the latest locations currently flagged for potential bloom risk.</p></div>',unsafe_allow_html=True)
        st.download_button("⬇ Download potential-risk locations",risk.to_csv(index=False).encode(),"bloomdetect_potential_risk.csv","text/csv",use_container_width=True)
    st.markdown('<div class="note"><b>Data boundary:</b> this is not a live real-time monitoring feed. Satellite screening should be combined with field observations and other environmental evidence before decisions about a harmful event are made.</div>',unsafe_allow_html=True)

st.markdown(f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Current data through {latest_date.strftime("%d %B %Y")}</div>',unsafe_allow_html=True)