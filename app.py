from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Clean single-app dashboard
# Pages:
#   Home | Risk Map | Location | Insights | How It Works | Data
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
INFOGRAPHIC_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
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
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for col in ["latitude", "longitude", "chla"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

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
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(
        subset=["latitude", "longitude", "date", "chla"]
    ).copy()

    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
    d["risk_flag"] = (
        d["risk_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("potential bloom risk")
    )

    return d


try:
    df = load_data()
except Exception as exc:
    st.error("BloomDetect could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
risk = latest[latest["risk_flag"]].copy()
normal = latest[~latest["risk_flag"]].copy()

# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------
PAGES = ["home", "map", "location", "insights", "method", "data"]
NAV = {
    "home": "⌂ Home",
    "map": "🌊 Risk Map",
    "location": "📍 Location",
    "insights": "📈 Insights",
    "method": "🧠 How It Works",
    "data": "📊 Data",
}

if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"


def go(page):
    st.session_state.page = page
    st.rerun()


# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

:root{
    --ink:#103f49;
    --muted:#64828a;
    --aqua:#0aa3af;
    --deep:#07566a;
    --line:#a9d8dc;
    --pale:#effafa;
    --shadow:0 14px 36px rgba(9,80,94,.09);
}

html, body, [data-testid="stAppViewContainer"]{
    background:#f2fbfb !important;
    color:var(--ink) !important;
    font-family:'DM Sans', sans-serif !important;
}

.stApp{
    background:
        radial-gradient(circle at 5% 5%, rgba(84,220,219,.20), transparent 22%),
        radial-gradient(circle at 95% 20%, rgba(96,202,236,.14), transparent 25%),
        linear-gradient(180deg,#fbffff 0%,#eefafa 55%,#fbffff 100%) !important;
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
    padding:22px 30px 70px !important;
}

*{box-sizing:border-box;}

.stApp::before{
    content:"";
    position:fixed;
    left:-10%;
    right:-10%;
    bottom:-145px;
    height:260px;
    z-index:0;
    pointer-events:none;
    background:
        radial-gradient(ellipse at 15% 55%,rgba(28,193,201,.14) 0 17%,transparent 18%),
        radial-gradient(ellipse at 55% 45%,rgba(69,221,218,.11) 0 20%,transparent 21%),
        radial-gradient(ellipse at 90% 60%,rgba(15,171,191,.11) 0 18%,transparent 19%);
    animation:waterMove 12s ease-in-out infinite alternate;
}

.stApp::after{
    content:"◦   ·     ◦      ·     ◦      ·";
    position:fixed;
    left:5%;
    bottom:-90px;
    z-index:0;
    pointer-events:none;
    color:rgba(8,153,169,.10);
    font-size:24px;
    letter-spacing:55px;
    animation:bubbleRise 20s linear infinite;
}

@keyframes waterMove{
    from{transform:translateX(-2%)}
    to{transform:translateX(2%)}
}
@keyframes bubbleRise{
    from{transform:translateY(80px);opacity:.02}
    50%{opacity:.13}
    to{transform:translateY(-100vh);opacity:.01}
}

/* Header */
.brand-bar{
    position:relative;
    z-index:10;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:15px 20px;
    border:1.5px solid #c5e4e6;
    background:rgba(255,255,255,.90);
    border-radius:22px;
    box-shadow:var(--shadow);
    backdrop-filter:blur(18px);
}
.brand-left{display:flex;align-items:center;gap:13px;}
.logo{
    width:48px;height:48px;border-radius:16px;
    background:linear-gradient(145deg,#28cdd0,#087b8e);
    display:grid;place-items:center;
    color:#fff;font-size:20px;font-weight:800;
    box-shadow:0 10px 24px rgba(9,144,157,.18);
}
.brand-name{
    font:800 1.22rem Manrope,sans-serif;
    color:#103f49;
    letter-spacing:-.03em;
}
.brand-sub{font-size:.73rem;color:#78959b;margin-top:2px;}
.brand-date{text-align:right;color:#6c898f;font-size:.72rem;line-height:1.5;}
.brand-date b{color:#174b55;font-size:.78rem;}

/* Navigation */
.nav-wrap{position:relative;z-index:12;margin:14px 0 8px;}
.stButton>button{
    min-height:46px !important;
    border-radius:14px !important;
    background:#fff !important;
    border:2px solid #164e5b !important;
    color:#123f49 !important;
    font-weight:800 !important;
    font-size:.87rem !important;
    box-shadow:0 6px 16px rgba(15,70,82,.08) !important;
    transition:all .18s ease !important;
}
.stButton>button:hover{
    background:#e2f8f8 !important;
    border-color:#087d8c !important;
    transform:translateY(-1px) !important;
}
.stButton>button[kind="primary"]{
    background:#dff7f7 !important;
    border-color:#079ca9 !important;
    color:#07566a !important;
    box-shadow:0 0 0 3px rgba(16,174,183,.10),0 8px 18px rgba(15,91,101,.10) !important;
}
.stButton>button p,.stButton>button span{color:inherit !important;}

/* General */
.kicker{
    font:800 .69rem Manrope,sans-serif;
    letter-spacing:.18em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:11px;
}
.section{
    position:relative;
    z-index:2;
    padding:38px 0 18px;
}
.section h2{
    font:800 clamp(2.05rem,3.8vw,3.65rem)/1.03 Manrope,sans-serif;
    letter-spacing:-.055em;
    color:#103f49;
    margin:0 0 13px;
}
.section p{
    color:#5f8088;
    line-height:1.65;
    margin:0;
    max-width:1000px;
    font-size:1rem;
}

.card{
    position:relative;
    z-index:2;
    background:rgba(255,255,255,.88);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:var(--shadow);
    padding:22px;
}
.card h3{
    font:800 1.28rem Manrope,sans-serif;
    color:#123f49;
    margin:0 0 9px;
}
.card p{
    color:#66848b;
    line-height:1.62;
    margin:0;
    font-size:.94rem;
}

.metric{
    position:relative;
    z-index:2;
    min-height:116px;
    height:100%;
    padding:18px;
    background:linear-gradient(145deg,#fff,#eaf8f8);
    border:1.5px solid #a6d5da;
    border-radius:19px;
    box-shadow:0 11px 28px rgba(15,91,101,.08);
    display:flex;
    flex-direction:column;
    justify-content:center;
}
.metric .label{
    font:800 .65rem Manrope,sans-serif;
    letter-spacing:.12em;
    text-transform:uppercase;
    color:#6d8c93;
}
.metric .value{
    font:800 1.55rem Manrope,sans-serif;
    color:#123f49;
    margin-top:7px;
    white-space:nowrap;
}
.metric .note{font-size:.74rem;color:#78959b;margin-top:5px;}

.note{
    position:relative;
    z-index:2;
    margin-top:17px;
    padding:13px 16px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 14px 14px 0;
    color:#52757c;
    font-size:.87rem;
    line-height:1.6;
}

/* Home */
.hero{
    position:relative;
    z-index:2;
    overflow:hidden;
    min-height:470px;
    border-radius:31px;
    padding:65px 60px;
    background:
        linear-gradient(135deg,rgba(4,56,72,.95),rgba(5,105,123,.88)),
        url("https://images.unsplash.com/photo-1717292742444-8a7c392da4dd?auto=format&fit=crop&w=1900&q=82")
        center/cover;
    box-shadow:0 28px 72px rgba(6,86,100,.16);
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
    left:-5%;right:-5%;bottom:-130px;height:270px;
    background:rgba(142,244,237,.13);
    border-radius:50%;
    animation:heroWave 8s ease-in-out infinite alternate;
}
.hero-content{position:relative;z-index:2;max-width:850px;}
.hero .kicker{color:#a6fffa;}
.hero h1{
    font:800 clamp(3.1rem,6vw,5.8rem)/.92 Manrope,sans-serif;
    letter-spacing:-.075em;
    color:#e4ffff;
    margin:0 0 22px;
}
.hero p{
    font-size:1.08rem;
    line-height:1.78;
    color:#e0fbfb;
    max-width:760px;
}
.badges{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px;}
.badge{
    padding:9px 13px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.25);
    color:#efffff;
    font-size:.80rem;
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

.bloom-grid{
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:1fr 1.15fr;
    gap:22px;
    align-items:stretch;
}
.bloom-panel,.bloom-visual{
    background:rgba(255,255,255,.82);
    border:1.5px solid #b9dfe2;
    border-radius:23px;
    box-shadow:var(--shadow);
    padding:26px;
}
.bloom-panel{min-height:360px;}
.bloom-panel h3{
    font:800 1.4rem/1.2 Manrope;
    color:#123f49;
    margin:0 0 13px;
}
.bloom-panel p{
    font-size:.97rem;
    line-height:1.75;
    color:#66848b;
    margin:0;
}
.bloom-panel .important{
    margin-top:20px;
    padding:14px 16px;
    background:#e5f9f9;
    border-left:4px solid #13aab2;
    border-radius:0 14px 14px 0;
    color:#3f6971;
    line-height:1.62;
}
.bloom-visual{padding:9px;min-height:360px;}
.bloom-visual img{
    display:block;
    width:100%;
    height:100%;
    min-height:340px;
    object-fit:cover;
    border-radius:17px;
}

.feature-grid{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:16px;
}
.feature{
    position:relative;
    z-index:2;
    padding:23px;
    background:rgba(255,255,255,.82);
    border:1.5px solid #b5dde0;
    border-radius:20px;
    box-shadow:var(--shadow);
    min-height:155px;
}
.feature .icon{font-size:1.45rem;margin-bottom:11px;}
.feature h3{
    font:800 1.1rem Manrope;
    color:#123f49;
    margin:0 0 7px;
}
.feature p{
    font-size:.91rem;
    line-height:1.62;
    color:#66848b;
    margin:0;
}

/* Map */
.map-card{
    position:relative;
    z-index:2;
    background:#e8f8f9;
    border-radius:23px;
    padding:4px;
    border:2px solid #8fcbd1;
    box-shadow:0 16px 42px rgba(15,91,101,.09);
    overflow:hidden;
}
.map-head{
    min-height:48px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:15px;
    padding:5px 13px;
}
.map-head b{color:#174b56;}
.map-head span{font-size:.76rem;color:#67858c;}
.map-legend{
    display:flex;
    align-items:center;
    gap:10px;
    flex-wrap:wrap;
    color:#4f747b;
    font-size:.83rem;
    padding:12px 14px;
}
.legend-blue{
    width:18px;height:11px;border-radius:4px;background:#3ca5bf;
}
.legend-green{
    width:18px;height:11px;border-radius:4px;background:#49b889;
}
.legend-red{
    width:12px;height:12px;border-radius:50%;
    background:#ef4f5e;border:2px solid white;
}

/* Location */
.location-grid{
    display:grid;
    grid-template-columns:.82fr 1.18fr;
    gap:22px;
    align-items:start;
}
.location-form{
    background:rgba(255,255,255,.88);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:var(--shadow);
    padding:22px;
}
.location-form label{
    color:#285761 !important;
    font-weight:800 !important;
}
.stNumberInput input{
    color:#103f49 !important;
    background:#fff !important;
    border:1.5px solid #8dbec4 !important;
    border-radius:12px !important;
}
.result-card{
    background:rgba(255,255,255,.90);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:var(--shadow);
    padding:22px;
}
.result-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:11px;
    margin-top:15px;
}
.result-grid div{
    padding:13px 14px;
    background:linear-gradient(145deg,#f5fdfd,#e9f8f8);
    border:1px solid #c7e5e7;
    border-radius:14px;
    min-height:62px;
}
.result-grid span{
    display:block;
    font-size:.70rem;
    color:#78959b;
    margin-bottom:5px;
}
.result-grid b{font-size:.94rem;color:#194b55;}
.status{
    margin-top:15px;
    padding:16px;
    border-radius:15px;
    border:2px solid;
    line-height:1.55;
}
.status.risk{
    background:#fff0f2;
    border-color:#ef6674;
    color:#9d2d3b;
}
.status.normal{
    background:#eafaf4;
    border-color:#49b995;
    color:#176f58;
}
.status strong{font:800 .96rem Manrope;}

/* Insights */
.chart-card{
    background:rgba(255,255,255,.86);
    border:1.5px solid #a9d6da;
    border-radius:22px;
    box-shadow:var(--shadow);
    padding:17px 17px 8px;
}
.chart-card h3{
    font:800 1.15rem Manrope;
    color:#123f49;
    margin:0 0 5px;
}
.chart-card p{
    color:#66848b;
    font-size:.88rem;
    margin:0 0 8px;
}
.plotly .xtick text,.plotly .ytick text{
    font-size:12px !important;
    fill:#174b56 !important;
}
.plotly .xtitle,.plotly .ytitle{
    fill:#174b56 !important;
}

/* Method */
.step{
    position:relative;
    z-index:2;
    display:grid;
    grid-template-columns:52px 1fr;
    gap:15px;
    align-items:center;
    padding:15px 17px;
    margin:9px 0;
    border:2px solid #174e5b;
    border-radius:17px;
    background:linear-gradient(100deg,rgba(255,255,255,.96),rgba(232,249,249,.90));
    box-shadow:0 9px 23px rgba(15,91,101,.07);
}
.step-num{
    width:43px;height:43px;border-radius:12px;
    background:#0d5264;color:#fff;
    display:grid;place-items:center;
    font:800 .85rem Manrope;
    border:2px solid #082f3b;
}
.step h3{
    font:800 1.12rem Manrope;
    color:#123f49;
    margin:0 0 4px;
}
.step p{
    color:#66848b;
    line-height:1.5;
    margin:0;
    font-size:.92rem;
}

/* Data */
.table-wrap{
    position:relative;
    z-index:2;
    border-radius:18px;
    overflow:hidden;
    border:1px solid #c5e3e5;
    box-shadow:var(--shadow);
    background:#fff;
}
.table-wrap table{
    width:100%;
    border-collapse:collapse;
    font-size:.87rem;
}
.table-wrap th{
    background:#0e5663;
    color:#fff;
    text-align:left;
    padding:12px 14px;
}
.table-wrap td{
    padding:11px 14px;
    border-top:1px solid #e3eeee;
    color:#315b63;
    background:#fff;
}
.table-wrap tr:nth-child(even) td{background:#f7fcfc;}
.download-panel{
    min-height:132px;
    padding:21px;
    border-radius:19px;
    background:linear-gradient(135deg,#087b8b,#13adb3);
    border:2px solid #07576a;
    box-shadow:0 13px 28px rgba(8,91,102,.16);
    color:#fff;
}
.download-panel h3{
    font:800 1.18rem Manrope;
    color:#fff;
    margin:0 0 7px;
}
.download-panel p{
    font-size:.88rem;
    color:#e5ffff;
    line-height:1.45;
    margin:0;
}
.stDownloadButton>button{
    min-height:46px !important;
    border-radius:13px !important;
    background:#fff !important;
    border:2px solid #164e5b !important;
    color:#123f49 !important;
    font-weight:800 !important;
}
.footer{
    position:relative;
    z-index:2;
    border-top:1px solid #d5ebed;
    margin-top:44px;
    padding-top:18px;
    color:#76959b;
    font-size:.72rem;
    text-align:center;
}

@media(max-width:900px){
    .block-container{padding:18px 18px 55px !important;}
    .bloom-grid,.location-grid{grid-template-columns:1fr;}
    .feature-grid{grid-template-columns:1fr 1fr;}
    .hero{padding:48px 35px;}
}
@media(max-width:620px){
    .feature-grid{grid-template-columns:1fr;}
    .hero h1{font-size:3rem;}
    .block-container{padding:15px 12px 45px !important;}
    .brand-date{display:none;}
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HEADER + NAV
# ------------------------------------------------------------
st.markdown(
    f"""
<div class="brand-bar">
    <div class="brand-left">
        <div class="logo">≈</div>
        <div>
            <div class="brand-name">BloomDetect AI</div>
            <div class="brand-sub">Coastal & Ocean Intelligence Platform · EOS-06 OCM-3</div>
        </div>
    </div>
    <div class="brand-date">
        Latest processed field<br>
        <b>{latest_date.strftime("%d %b %Y")}</b>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

nav_cols = st.columns(len(PAGES), gap="small")
for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(
            NAV[page],
            key=f"nav_{page}",
            width="stretch",
            type="primary" if st.session_state.page == page else "secondary",
        ):
            go(page)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
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


def safe(value, digits=4):
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
    if 5 <= lat <= 30 and 45 <= lon <= 78:
        return "Arabian Sea"
    if 0 <= lat <= 25 and 78 < lon <= 100:
        return "Bay of Bengal"
    if -40 <= lat <= 30 and 20 <= lon <= 120:
        return "Indian Ocean study area"
    return "Outside named study window"


def nearest_row(lat, lon, frame=None):
    source = latest if frame is None else frame
    if source.empty:
        return None

    lat_values = source["latitude"].to_numpy(dtype=float)
    lon_values = source["plot_lon"].to_numpy(dtype=float)

    dlat = (lat_values - lat) ** 2
    dlon_raw = np.abs(lon_values - lon)
    dlon = np.minimum(dlon_raw, 360 - dlon_raw) ** 2

    idx = int(np.argmin(dlat + dlon))
    return source.iloc[idx]


def study_area(frame):
    return frame[
        frame["latitude"].between(-40, 30)
        & frame["plot_lon"].between(20, 120)
    ].copy()


def make_map(frame, show_risk=True):
    """
    Clean study-area map:
    - Blue = processed ocean field
    - Green = concentrated potential-risk zones
    - Red = individual potential-risk cells
    """
    data = study_area(frame)

    if data.empty:
        data = frame.copy()

    normal_data = data[~data["risk_flag"]].copy()
    risk_data = data[data["risk_flag"]].copy()

    # Larger square cells create a continuous-looking field instead of
    # the old "thousands of blue dots" appearance.
    if not normal_data.empty:
        normal_data["lat_bin"] = (
            np.floor(normal_data["latitude"]) + 0.5
        ).round(2)
        normal_data["lon_bin"] = (
            np.floor(normal_data["plot_lon"]) + 0.5
        ).round(2)

        blue = (
            normal_data.groupby(["lat_bin", "lon_bin"], as_index=False)
            .agg(chla=("chla", "mean"))
        )
    else:
        blue = pd.DataFrame(columns=["lat_bin", "lon_bin", "chla"])

    fig = go.Figure()

    if not blue.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=blue["lat_bin"],
                lon=blue["lon_bin"],
                mode="markers",
                marker=dict(
                    size=18,
                    color="#3ca5bf",
                    opacity=0.24,
                    symbol="square",
                    line=dict(width=0),
                ),
                customdata=np.column_stack(
                    [
                        blue["lat_bin"].to_numpy(),
                        blue["lon_bin"].to_numpy(),
                        blue["chla"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "<b>Processed ocean field</b><br>"
                    "Latitude: %{customdata[0]:.1f}°<br>"
                    "Longitude: %{customdata[1]:.1f}°<br>"
                    "Mean Chl-a: %{customdata[2]:.4f}"
                    "<extra></extra>"
                ),
                name="Processed ocean field",
            )
        )

    if show_risk and not risk_data.empty:
        risk_data = risk_data.copy()

        risk_data["lat_zone"] = (
            np.floor(risk_data["latitude"] / 2) * 2 + 1
        ).round(2)
        risk_data["lon_zone"] = (
            np.floor(risk_data["plot_lon"] / 2) * 2 + 1
        ).round(2)

        zones = (
            risk_data.groupby(["lat_zone", "lon_zone"], as_index=False)
            .agg(
                flagged_cells=("risk_flag", "size"),
                mean_chla=("chla", "mean"),
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lat=zones["lat_zone"],
                lon=zones["lon_zone"],
                mode="markers",
                marker=dict(
                    size=np.clip(
                        18 + zones["flagged_cells"].to_numpy() * 0.65,
                        18,
                        45,
                    ),
                    color="#48b889",
                    opacity=0.38,
                    line=dict(width=1.2, color="#238260"),
                ),
                customdata=np.column_stack(
                    [
                        zones["lat_zone"].to_numpy(),
                        zones["lon_zone"].to_numpy(),
                        zones["flagged_cells"].to_numpy(),
                        zones["mean_chla"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "<b>Potential-risk concentration</b><br>"
                    "Zone centre: %{customdata[0]:.1f}°, "
                    "%{customdata[1]:.1f}°<br>"
                    "Flagged cells: %{customdata[2]}<br>"
                    "Mean Chl-a: %{customdata[3]:.4f}"
                    "<extra></extra>"
                ),
                name="Risk concentration",
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lat=risk_data["latitude"],
                lon=risk_data["plot_lon"],
                mode="markers",
                marker=dict(
                    size=8.5,
                    color="#ef4f5e",
                    opacity=0.97,
                    line=dict(width=1.2, color="#ffffff"),
                ),
                customdata=np.column_stack(
                    [
                        risk_data["latitude"].to_numpy(),
                        risk_data["plot_lon"].to_numpy(),
                        risk_data["chla"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "<b>Potential bloom-risk flag</b><br>"
                    "Latitude: %{customdata[0]:.2f}°<br>"
                    "Longitude: %{customdata[1]:.2f}°<br>"
                    "Chl-a: %{customdata[2]:.4f}"
                    "<extra></extra>"
                ),
                name="Potential bloom risk",
            )
        )

    fig.update_geos(
        projection="mercator",
        showland=True,
        landcolor="#dce9e8",
        showocean=True,
        oceancolor="#e8f8f9",
        showcoastlines=True,
        coastlinecolor="#4c8992",
        coastlinewidth=1.1,
        showcountries=True,
        countrycolor="#91b6bb",
        showlakes=False,
        bgcolor="#e8f8f9",
        lataxis_range=[-40, 30],
        lonaxis_range=[20, 120],
        center=dict(lat=-5, lon=70),
        projection_scale=1.02,
    )

    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#e8f8f9",
        plot_bgcolor="#e8f8f9",
        font=dict(color="#174b56"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.015,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,.90)",
            bordercolor="#9fcfd4",
            borderwidth=1,
            font=dict(size=12, color="#174b56"),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#4bbbc2",
            font_color="#174b56",
        ),
    )

    return fig


# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------
if st.session_state.page == "home":
    st.markdown(
        """
<div class="hero">
    <div class="hero-content">
        <div class="kicker">EOS-06 OCM-3 · Satellite intelligence</div>
        <h1>BloomDetect<br>AI</h1>
        <p>
            A satellite-based screening platform that turns ocean-colour
            observations into a clear view of <b>potential bloom risk</b>,
            spatial patterns and locations worth investigating further.
        </p>
        <div class="badges">
            <span class="badge">🌊 Ocean colour</span>
            <span class="badge">🛰️ EOS-06 OCM-3</span>
            <span class="badge">🤖 Decision Tree screening</span>
            <span class="badge">📍 Spatial intelligence</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    section(
        "What is BloomDetect AI?",
        "A map-first way to read the signal.",
        "The platform starts with satellite-derived chlorophyll-a and uses the project's screening model to highlight cells that deserve closer observation.",
    )

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.markdown(
            """
<div class="bloom-panel">
    <div class="kicker">Why this matters</div>
    <h3>Chlorophyll-a is a signal, not a verdict.</h3>
    <p>
        Changes in chlorophyll-a can help identify unusual ocean-colour
        patterns and areas worth investigating. BloomDetect uses that signal
        to create a <b>potential bloom-risk screening result</b>.
    </p>
    <div class="important">
        <b>Important:</b> high chlorophyll-a alone does not prove a harmful
        algal bloom. Species, toxins and ecological impacts need additional
        evidence and field validation.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        if INFOGRAPHIC_PATH.exists():
            st.markdown('<div class="bloom-visual">', unsafe_allow_html=True)
            st.image(str(INFOGRAPHIC_PATH), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
<div class="bloom-visual">
    <div class="card" style="height:100%;display:flex;align-items:center;justify-content:center;">
        <div>
            <div class="kicker">Ocean signal</div>
            <h3>Satellite → signal → screening → map</h3>
            <p>
                Keep <b>bloomdetect_bloom_process.png</b> beside app.py
                to display the project illustration here.
            </p>
        </div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

    section(
        "Explore",
        "One feature per job.",
        "The dashboard deliberately avoids separate pages that repeat the same map, counts or dataset description.",
    )

    feature_cols = st.columns(3, gap="medium")
    features = [
        ("🌊", "Risk Map", "See the study area, processed ocean field, risk concentrations and individual flagged cells in one place."),
        ("📍", "Location", "Enter latitude and longitude and get one clear screening result for the nearest processed grid cell."),
        ("📈", "Insights", "Read the latest field through a few useful charts and a compact investigation queue."),
    ]

    for col, (icon, title, copy) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
<div class="feature">
    <div class="icon">{icon}</div>
    <h3>{title}</h3>
    <p>{copy}</p>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="note"><b>Current data:</b> the public CSV bundled with this app contains the latest processed field. It is a screening prototype, not a real-time HAB monitoring service.</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# RISK MAP
# ------------------------------------------------------------
elif st.session_state.page == "map":
    section(
        "01 · Ocean & bloom risk",
        "Where is potential bloom risk showing up?",
        "This is the main spatial view. Blue shows the processed ocean field, green shows concentrations of flagged cells, and red shows the individual cells screened as potential bloom risk.",
    )

    c1, c2, c3 = st.columns([1.1, 1.1, .8], gap="medium")
    with c1:
        metric("Study area", "20°–120°E", "40°S–30°N")
    with c2:
        metric("Flagged cells", f"{len(risk):,}", "latest processed field")
    with c3:
        share = 100 * len(risk) / len(latest) if len(latest) else 0
        metric("Risk share", f"{share:.2f}%", "of processed cells")

    st.markdown(
        '<div class="map-card"><div class="map-head"><b>Indian Ocean study area</b><span>Latest processed field · colour is used for spatial screening, not confirmed HAB extent</span></div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        make_map(latest, show_risk=True),
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    st.markdown(
        """
<div class="map-legend">
    <span class="legend-blue"></span>
    <b>Blue</b> = processed ocean field
    <span class="legend-green"></span>
    <b>Green</b> = spatial concentration of potential-risk cells
    <span class="legend-red"></span>
    <b>Red</b> = individual potential bloom-risk cells
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="note"><b>Reading the map:</b> blue is the normal processed study field. Green does not mean a confirmed bloom. It groups nearby screening flags so the spatial pattern is easier to understand. Red identifies the individual cells stored as potential bloom risk.</div>',
        unsafe_allow_html=True,
    )

    unique_dates = sorted(df["date"].dropna().dt.normalize().unique())

    if len(unique_dates) > 1:
        section(
            "Map timeline",
            "See how the screening pattern changes.",
            "Choose an observation date from the processed dataset. The map is rebuilt for that date.",
        )

        selected_date = st.select_slider(
            "Observation date",
            options=unique_dates,
            value=unique_dates[-1],
            format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"),
            key="map_date_slider",
        )

        date_frame = df[
            df["date"].dt.normalize().eq(pd.Timestamp(selected_date))
        ].copy()

        date_risk = date_frame[date_frame["risk_flag"]]

        d1, d2 = st.columns(2, gap="medium")
        with d1:
            metric(
                "Selected date",
                pd.Timestamp(selected_date).strftime("%d %b %Y"),
                "processed observation",
            )
        with d2:
            metric(
                "Potential-risk cells",
                f"{len(date_risk):,}",
                "selected date",
            )

        st.plotly_chart(
            make_map(date_frame, show_risk=True),
            width="stretch",
            config={"displaylogo": False, "scrollZoom": False},
        )
    else:
        st.markdown(
            '<div class="note"><b>Timeline:</b> the current public CSV contains one processed observation date, so a true date-by-date map slider cannot be shown yet. The app is ready to use the slider automatically when a multi-date processed CSV is supplied.</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# LOCATION
# ------------------------------------------------------------
elif st.session_state.page == "location":
    section(
        "02 · Location check",
        "Check one coordinate.",
        "Enter latitude and longitude, press the button, and BloomDetect checks the nearest processed satellite grid cell. Nothing else is needed.",
    )

    if "checked_lat" not in st.session_state:
        st.session_state.checked_lat = 17.53
    if "checked_lon" not in st.session_state:
        st.session_state.checked_lon = 78.49

    left, right = st.columns([.82, 1.18], gap="large")

    with left:
        st.markdown(
            '<div class="location-form"><div class="kicker">Enter coordinates</div><p style="color:#66848b;margin:0 0 14px;">Use decimal degrees. Example: 17.3850, 78.4867.</p>',
            unsafe_allow_html=True,
        )

        with st.form("location_form", clear_on_submit=False):
            lat_input = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.checked_lat),
                step=0.25,
                format="%.4f",
            )
            lon_input = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.checked_lon),
                step=0.25,
                format="%.4f",
            )

            check = st.form_submit_button(
                "🔎 Check location",
                width="stretch",
                type="primary",
            )

        if check:
            st.session_state.checked_lat = float(lat_input)
            st.session_state.checked_lon = float(lon_input)
            st.rerun()

        st.markdown(
            f'<div class="entered"><b>Your input:</b><br>{st.session_state.checked_lat:.4f}°, {st.session_state.checked_lon:.4f}°</div></div>',
            unsafe_allow_html=True,
        )

    lat = float(st.session_state.checked_lat)
    lon = float(st.session_state.checked_lon)
    row = nearest_row(lat, lon)

    with right:
        if row is None:
            st.error("No processed observation is available.")
        else:
            risk_yes = bool(row["risk_flag"])
            prob = probability_text(row.get("risk_probability", np.nan))
            status_class = "risk" if risk_yes else "normal"

            st.markdown(
                f"""
<div class="result-card">
    <div class="kicker">Nearest processed cell</div>
    <h3 style="font-size:1.65rem;">
        {row["latitude"]:.4f}° · {row["longitude"]:.4f}°
    </h3>
    <p style="color:#66848b;margin:0;">
        Satellite grid cell used for the screening result.
    </p>

    <div class="status {status_class}">
        <strong>{"🔴 POTENTIAL BLOOM-RISK FLAG" if risk_yes else "🟢 NOT FLAGGED"}</strong><br>
        <span>
            {"The nearest processed cell is included in the current potential bloom-risk screening result."
             if risk_yes
             else
             "The nearest processed cell is not included in the current potential bloom-risk screening result."}
        </span>
    </div>

    <div class="result-grid">
        <div><span>Chlorophyll-a</span><b>{safe(row["chla"])}</b></div>
        <div><span>Risk probability</span><b>{prob}</b></div>
        <div><span>Observation date</span><b>{row["date"].strftime("%d %b %Y")}</b></div>
        <div><span>Area</span><b>{region_name(float(row["latitude"]), float(row["longitude"]))}</b></div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------
# INSIGHTS
# ------------------------------------------------------------
elif st.session_state.page == "insights":
    section(
        "03 · Insights",
        "What stands out in the latest field?",
        "Only a few useful views live here. The map remains the place for spatial screening, while Insights explains the latest field through counts, distributions and simple comparisons.",
    )

    risk_count = int(len(risk))
    normal_count = int(len(normal))
    risk_share = 100 * risk_count / len(latest) if len(latest) else 0

    cols = st.columns(4, gap="medium")
    with cols[0]:
        metric("Processed cells", f"{len(latest):,}", "latest field")
    with cols[1]:
        metric("Potential-risk cells", f"{risk_count:,}", "screening output")
    with cols[2]:
        metric("Normal cells", f"{normal_count:,}", "screening output")
    with cols[3]:
        metric("Risk share", f"{risk_share:.2f}%", "latest field")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(
            """
<div class="chart-card">
    <div class="kicker">Screening composition</div>
    <h3>Normal vs potential bloom risk</h3>
    <p>How the latest processed cells are classified by the stored screening output.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        comp = pd.DataFrame(
            {
                "Classification": ["Normal", "Potential bloom risk"],
                "Cells": [normal_count, risk_count],
            }
        )

        fig1 = px.bar(
            comp,
            x="Classification",
            y="Cells",
            text="Cells",
            color="Classification",
            color_discrete_map={
                "Normal": "#4aaeb8",
                "Potential bloom risk": "#ef5362",
            },
        )

        fig1.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            cliponaxis=False,
        )

        fig1.update_layout(
            height=360,
            margin=dict(l=65, r=25, t=25, b=65),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            showlegend=False,
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Screening outcome",
                title_font=dict(size=13, color="#174b56"),
                tickfont=dict(size=12, color="#174b56"),
            ),
            yaxis=dict(
                title="Number of cells",
                title_font=dict(size=13, color="#174b56"),
                tickfont=dict(size=12, color="#174b56"),
                gridcolor="#d6e7e8",
            ),
        )

        st.plotly_chart(
            fig1,
            width="stretch",
            config={"displaylogo": False},
        )

    with c2:
        st.markdown(
            """
<div class="chart-card">
    <div class="kicker">Chlorophyll-a distribution</div>
    <h3>Where the values cluster</h3>
    <p>This is the satellite-derived field distribution, not a direct measure of harmfulness.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        clean_chla = latest["chla"].dropna()

        fig2 = px.histogram(
            clean_chla.to_frame(name="Chlorophyll-a"),
            x="Chlorophyll-a",
            nbins=32,
            color_discrete_sequence=["#159eaa"],
        )

        fig2.update_layout(
            height=360,
            margin=dict(l=65, r=25, t=25, b=65),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font=dict(color="#174b56"),
            xaxis=dict(
                title="Chlorophyll-a",
                title_font=dict(size=13, color="#174b56"),
                tickfont=dict(size=12, color="#174b56"),
                gridcolor="#eef4f4",
            ),
            yaxis=dict(
                title="Number of cells",
                title_font=dict(size=13, color="#174b56"),
                tickfont=dict(size=12, color="#174b56"),
                gridcolor="#d6e7e8",
            ),
        )

        st.plotly_chart(
            fig2,
            width="stretch",
            config={"displaylogo": False},
        )

    section(
        "Spatial summary",
        "Where are the screening flags concentrated?",
        "These three broad study windows summarize the same latest field. They are not separate hotspot features and do not create new risk labels.",
    )

    regional_rows = []
    windows = [
        ("Arabian Sea", 5, 30, 45, 78),
        ("Bay of Bengal", 0, 25, 78, 100),
        ("Southern Indian Ocean", -30, 5, 40, 100),
    ]

    for name, lat1, lat2, lon1, lon2 in windows:
        subset = latest[
            latest["latitude"].between(lat1, lat2)
            & latest["plot_lon"].between(lon1, lon2)
        ]

        flags = int(subset["risk_flag"].sum())
        share = 100 * flags / len(subset) if len(subset) else 0

        regional_rows.append(
            {
                "Region": name,
                "Processed cells": len(subset),
                "Potential-risk cells": flags,
                "Risk share": share,
            }
        )

    regional = pd.DataFrame(regional_rows)

    fig3 = px.bar(
        regional,
        x="Region",
        y="Potential-risk cells",
        text="Potential-risk cells",
        color_discrete_sequence=["#1499a8"],
    )

    fig3.update_traces(textposition="outside", cliponaxis=False)

    fig3.update_layout(
        height=340,
        margin=dict(l=60, r=25, t=25, b=70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#174b56"),
        showlegend=False,
        xaxis=dict(
            title="Study region",
            title_font=dict(size=13, color="#174b56"),
            tickfont=dict(size=12, color="#174b56"),
        ),
        yaxis=dict(
            title="Potential-risk cells",
            title_font=dict(size=13, color="#174b56"),
            tickfont=dict(size=12, color="#174b56"),
            gridcolor="#d6e7e8",
        ),
    )

    st.plotly_chart(
        fig3,
        width="stretch",
        config={"displaylogo": False},
    )

    st.markdown(
        '<div class="kicker" style="margin-top:24px;">Investigation queue</div>',
        unsafe_allow_html=True,
    )

    queue = risk.sort_values("chla", ascending=False).head(15).copy()

    show_cols = [
        c for c in
        ["latitude", "longitude", "chla", "risk_probability"]
        if c in queue.columns
    ]

    if not queue.empty:
        queue_display = queue[show_cols].copy()
        queue_display = queue_display.rename(
            columns={
                "latitude": "Latitude",
                "longitude": "Longitude",
                "chla": "Chlorophyll-a",
                "risk_probability": "Risk probability",
            }
        )

        for col in ["Chlorophyll-a", "Risk probability"]:
            if col in queue_display.columns:
                queue_display[col] = queue_display[col].round(4)

        st.dataframe(
            queue_display,
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No potential-risk cells are present in the latest field.")


# ------------------------------------------------------------
# HOW IT WORKS
# ------------------------------------------------------------
elif st.session_state.page == "method":
    section(
        "04 · How it works",
        "From satellite signal to screening result.",
        "A short, transparent view of what the project actually does.",
    )

    steps = [
        ("01", "Satellite observation", "EOS-06 OCM-3 provides the analysed chlorophyll-a observation used as the environmental signal."),
        ("02", "Data preparation", "Invalid records are removed and the observations are organized into a consistent spatial-temporal structure."),
        ("03", "Temporal context", "Previous chlorophyll-a, historical baseline, recent mean, recent maximum, anomaly and recent change provide context."),
        ("04", "AI screening", "A Decision Tree model screens the prepared features against the project's proxy potential-risk target."),
        ("05", "Spatial result", "The model output is returned to geographic cells so the latest pattern can be viewed on the Risk Map."),
        ("06", "Further investigation", "Flagged cells are candidates for additional oceanographic observations and field validation."),
    ]

    for number, title, copy in steps:
        st.markdown(
            f"""
<div class="step">
    <div class="step-num">{number}</div>
    <div>
        <h3>{title}</h3>
        <p>{copy}</p>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="note">
    <b>Scientific boundary:</b> the current dataset does not contain confirmed
    harmful-algal-bloom species or toxin labels. BloomDetect therefore reports
    <b>potential bloom risk</b>, not confirmed HAB detection. High chlorophyll-a
    alone does not prove a harmful algal bloom.
</div>
""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
elif st.session_state.page == "data":
    section(
        "05 · Data & research",
        "Complete dataset information.",
        "All source, coverage, processing-field and download information is kept here. Other pages do not repeat this documentation.",
    )

    cols = st.columns(4, gap="medium")

    with cols[0]:
        metric("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3")
    with cols[1]:
        metric("Grid", "0.25°", "latitude × longitude")
    with cols[2]:
        metric("Rows in current CSV", f"{len(df):,}", "processed prediction table")
    with cols[3]:
        metric(
            "Date coverage",
            f"{df['date'].nunique():,}",
            "observation dates in current CSV",
        )

    st.markdown(
        f"""
<div class="card" style="margin-top:18px;">
    <div class="kicker">Dataset provenance</div>
    <h3>What is this dataset?</h3>
    <p><b>Source product:</b> EOS-06 / Oceansat-3 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC).</p>
    <p><b>Product ID:</b> E06OCM_L4_AC.</p>
    <p><b>Core variable:</b> chlorophyll-a (<code>chla</code>).</p>
    <p><b>Spatial resolution:</b> 0.25° latitude × 0.25° longitude.</p>
    <p><b>Current website CSV:</b> {len(df):,} processed rows covering {df["date"].nunique():,} observation date(s), from {df["date"].min().strftime("%d %b %Y")} to {df["date"].max().strftime("%d %b %Y")}.</p>
    <p><b>Latest field:</b> {len(latest):,} processed cells on {latest_date.strftime("%d %B %Y")}.</p>
    <p><b>Interpretation:</b> the current dataset contains a satellite-derived environmental signal and stored model screening output. It does not contain confirmed HAB species or toxin measurements.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    section(
        "Project processing record",
        "How the current project dataset was prepared.",
        "These figures describe the project processing stage behind the dashboard. The website itself currently loads the latest processed prediction CSV.",
    )

    record = st.columns(3, gap="medium")

    with record[0]:
        metric("Raw files collected", "324", "EOS-06 OCM-3 files")
    with record[1]:
        metric("Clean files used", "318", "after duplicate/misdated checks")
    with record[2]:
        metric("Clean observations", "22.27M", "processed grid rows")

    record2 = st.columns(3, gap="medium")

    with record2[0]:
        metric("Clean dates", "318", "2025-01-02 to 2026-03-30")
    with record2[1]:
        metric("Grid locations", "70,812", "0.25° cells")
    with record2[2]:
        metric("ML-ready rows", "20.14M", "after history requirement")

    st.markdown(
        """
<div class="card" style="margin-top:18px;">
    <div class="kicker">Processing notes</div>
    <h3>What happened before the website.</h3>
    <p>The raw EOS-06 OCM-3 Level-4 Analysed Chlorophyll files were checked for duplicate or misdated records. Six duplicate/misdated files were excluded from the clean processing set while the raw archive was kept unchanged.</p>
    <p>Temporal features were calculated using only earlier observations: previous chlorophyll-a, historical baseline, recent mean, recent maximum, anomaly and change. The ML-ready table begins after the required historical-record window.</p>
    <p>The stored potential-risk label is a project proxy based on chlorophyll anomaly and recent change thresholds. It is not confirmed HAB ground truth.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    section(
        "Processed fields",
        "What each important column means.",
        "This is the data dictionary for the CSV currently used by the website.",
    )

    fields = [
        ("latitude", "Grid latitude", "degrees"),
        ("longitude", "Grid longitude", "degrees"),
        ("date", "Observation date", "date"),
        ("chla", "Satellite-derived chlorophyll-a", "product value"),
        ("risk_label", "Stored model screening result", "classification"),
        ("risk_probability", "Stored model probability when available", "probability"),
        ("previous_chla", "Previous observation", "derived feature"),
        ("historical_baseline", "Historical baseline", "derived feature"),
        ("recent_mean", "Recent mean", "derived feature"),
        ("recent_max", "Recent maximum", "derived feature"),
        ("chla_anomaly", "Chlorophyll anomaly", "derived feature"),
        ("chla_change", "Change from previous observation", "derived feature"),
    ]

    rows_html = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>"
        for a, b, c in fields
    )

    st.markdown(
        f"""
<div class="table-wrap">
<table>
    <thead>
        <tr><th>Field</th><th>Meaning</th><th>Type</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
</table>
</div>
""",
        unsafe_allow_html=True,
    )

    section(
        "Downloads",
        "Take the processed results with you.",
        "These are the two website downloads. Dataset documentation stays above; downloads stay together here.",
    )

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown(
            """
<div class="download-panel">
    <h3>Latest observation table</h3>
    <p>All rows for the latest processed observation date, including the stored screening output and available derived fields.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download latest observations",
            data=latest.to_csv(index=False).encode("utf-8"),
            file_name=f"bloomdetect_latest_{latest_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            width="stretch",
        )

    with d2:
        st.markdown(
            """
<div class="download-panel">
    <h3>Potential-risk shortlist</h3>
    <p>Only the latest processed cells currently classified as potential bloom risk.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download potential-risk locations",
            data=risk.to_csv(index=False).encode("utf-8"),
            file_name=f"bloomdetect_potential_risk_{latest_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown(
        """
<div class="note">
    <b>Data limitation:</b> the current public CSV is a processed screening
    table, not the full raw EOS-06 archive. It is also not a real-time feed.
    The model output is a proxy screening signal and should be validated with
    additional environmental or field evidence.
</div>
""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown(
    f"""
<div class="footer">
    BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening ·
    Latest processed date: {latest_date.strftime("%d %B %Y")}
</div>
""",
    unsafe_allow_html=True,
)
