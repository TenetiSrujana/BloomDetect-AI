
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# BLOOMDETECT AI - STAGE 1
# Coastal & Ocean Intelligence Platform
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI | Coastal & Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
HERO_IMAGE = BASE_DIR / "bloomdetect_bloom_process.png"
PHYTO_IMAGE = BASE_DIR / "phytoplankton_signal.png"
SAT_IMAGE = BASE_DIR / "satellite_signal_illustration.png"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --navy:#073B4C; --navy2:#0B5267; --aqua:#0EA5B7; --aqua2:#5ED7D7;
    --mint:#DDF6F3; --white:#FFFFFF; --ink:#102A33; --muted:#5C737B;
    --line:#B8DDE0; --green:#2F9E72; --green-soft:#E5F7EF;
    --red:#D64545; --red-soft:#FFF0F0; --amber:#C88619;
    --amber-soft:#FFF7E6; --shadow:0 10px 30px rgba(7,59,76,.08);
}

html,body,[class*="css"] { font-family:"DM Sans",sans-serif; color:var(--ink); }
.stApp {
    background:
      radial-gradient(circle at 8% 4%,rgba(94,215,215,.16),transparent 25%),
      radial-gradient(circle at 92% 10%,rgba(14,165,183,.08),transparent 28%),
      linear-gradient(180deg,#F8FEFD 0%,#EFF9F8 100%);
}
.block-container { max-width:1180px; padding-top:1.2rem!important; padding-bottom:4rem!important; }
#MainMenu,footer,header { visibility:hidden; }
h1,h2,h3 { font-family:"Space Grotesk",sans-serif!important; color:var(--navy)!important; letter-spacing:-.02em; }
h1 { font-size:2.35rem!important; line-height:1.08!important; }
h2 { font-size:1.55rem!important; } h3 { font-size:1.08rem!important; }
p,li,label,.stCaption { color:var(--ink); }

.brand-bar {
    background:rgba(255,255,255,.94); border:1px solid var(--line); border-radius:22px;
    padding:18px 22px; box-shadow:var(--shadow); display:flex; align-items:center;
    justify-content:space-between; gap:18px; margin-bottom:16px;
}
.brand-left { display:flex; align-items:center; gap:14px; }
.logo-mark {
    width:46px;height:46px;border-radius:14px;display:grid;place-items:center;
    background:linear-gradient(145deg,var(--navy),var(--aqua));color:#fff;font-size:23px;
    box-shadow:0 8px 18px rgba(7,59,76,.16);
}
.brand-name { font-family:"Space Grotesk",sans-serif;font-size:1.2rem;font-weight:700;color:var(--navy); }
.brand-sub,.brand-date { font-size:.78rem;color:var(--muted); }
.brand-date { text-align:right; }

.nav-wrap { margin:4px 0 20px; }
div[data-testid="stHorizontalBlock"] { gap:12px!important; }
.stButton>button {
    width:100%; min-height:44px; border-radius:12px; border:1.6px solid var(--navy);
    background:rgba(255,255,255,.96); color:var(--navy); font-size:.82rem;
    font-weight:600; padding:9px 8px; box-shadow:0 4px 12px rgba(7,59,76,.06);
    transition:all .18s ease;
}
.stButton>button:hover { border-color:var(--aqua);background:var(--mint);transform:translateY(-1px); }
.stButton>button:focus { box-shadow:0 0 0 3px rgba(14,165,183,.16);border-color:var(--aqua); }
.nav-active .stButton>button { background:linear-gradient(135deg,var(--navy),var(--aqua));color:#fff;border-color:var(--navy); }

.hero {
    position:relative;overflow:hidden;min-height:350px;border-radius:26px;
    border:1px solid #8FCED3;box-shadow:var(--shadow);
    background:
      linear-gradient(100deg,rgba(4,45,59,.92),rgba(7,81,99,.62)),
      url("https://images.unsplash.com/photo-1530053969600-caed2596d242?auto=format&fit=crop&w=1800&q=82")
      center/cover no-repeat;
    color:#fff;padding:44px;margin:10px 0 24px;
}
.hero:after {
    content:"";position:absolute;inset:auto -10% -48px -10%;height:130px;
    background:rgba(72,205,210,.18);border-radius:50%;animation:wave 5s ease-in-out infinite;
}
@keyframes wave { 0%,100%{transform:translateX(-2%) scaleY(1)} 50%{transform:translateX(3%) scaleY(1.25)} }
.hero-content { position:relative;z-index:2;max-width:720px; }
.kicker {
    display:inline-flex;padding:7px 11px;border:1px solid rgba(255,255,255,.35);
    background:rgba(255,255,255,.12);border-radius:999px;font-size:.74rem;
    font-weight:700;letter-spacing:.08em;text-transform:uppercase;
}
.hero h1 { color:#fff!important;margin:18px 0 12px!important;font-size:2.55rem!important; }
.hero p { color:rgba(255,255,255,.91);font-size:1rem;line-height:1.65;margin:0; }

.section-kicker { color:var(--aqua);font-size:.72rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800;margin-bottom:5px; }
.section-title { font-family:"Space Grotesk",sans-serif;color:var(--navy);font-size:1.55rem;font-weight:700;margin-bottom:7px; }
.section-copy { color:var(--muted);font-size:.9rem;line-height:1.55;margin-bottom:16px; }

.card {
    background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:18px;
    padding:20px;box-shadow:0 7px 24px rgba(7,59,76,.06);height:100%;
}
.card-title { font-family:"Space Grotesk",sans-serif;font-weight:700;color:var(--navy);font-size:1rem;margin-bottom:7px; }
.card-text { color:var(--muted);font-size:.84rem;line-height:1.55; }

.metric-card {
    background:rgba(255,255,255,.97);border:1px solid var(--line);border-radius:17px;
    padding:17px 18px;min-height:132px;box-shadow:0 6px 18px rgba(7,59,76,.05);
}
.metric-label { font-size:.74rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.06em; }
.metric-value { font-family:"Space Grotesk",sans-serif;font-size:1.65rem;font-weight:700;color:var(--navy);margin-top:7px; }
.metric-note { font-size:.72rem;color:var(--muted);margin-top:3px; }

.status { border-radius:16px;padding:16px 18px;border:1.5px solid; }
.status-green { background:var(--green-soft);border-color:#83CDAF;color:#1D6F4E; }
.status-red { background:var(--red-soft);border-color:#E7A1A1;color:#9C3030; }
.status-title { font-family:"Space Grotesk",sans-serif;font-size:1rem;font-weight:700; }
.status-copy { font-size:.8rem;margin-top:4px;line-height:1.5; }

.step-card {
    background:#fff;border:1.5px solid var(--navy);border-radius:16px;padding:17px;
    min-height:150px;box-shadow:0 6px 18px rgba(7,59,76,.05);
}
.step-num {
    width:31px;height:31px;display:grid;place-items:center;border-radius:50%;
    background:var(--navy);color:#fff;font-weight:700;font-size:.78rem;margin-bottom:10px;
}
.step-card h3 { margin:0 0 6px!important;font-size:.98rem!important; }
.step-card p { margin:0;color:var(--muted);font-size:.79rem;line-height:1.5; }

.notice {
    border-left:4px solid var(--aqua);background:#E9F8F7;border-radius:0 13px 13px 0;
    padding:14px 16px;color:var(--ink);font-size:.81rem;line-height:1.55;
}
.download-card {
    background:linear-gradient(135deg,#fff,#ECFAF8);border:1.5px solid var(--navy);
    border-radius:18px;padding:19px;min-height:150px;
}
.download-card h3 { margin-top:0!important; }
.download-note { color:var(--muted);font-size:.8rem;line-height:1.45; }
.footer { border-top:1px solid var(--line);margin-top:38px;padding-top:18px;color:var(--muted);font-size:.72rem;text-align:center; }

.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"] {
    border:1.3px solid #91BEC3!important;border-radius:11px!important;
}
div[data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:14px;overflow:hidden; }

@media (max-width:800px) {
    .block-container { padding-left:1rem!important;padding-right:1rem!important; }
    .hero { padding:28px;min-height:390px; }
    .hero h1 { font-size:2rem!important; }
    .brand-bar { align-items:flex-start;flex-direction:column; }
    .brand-date { text-align:left; }
}
</style>
""",
    unsafe_allow_html=True,
)

def clean_columns(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace(" ","_").replace("-","_") for c in frame.columns]
    return frame

def first_existing(frame, names):
    for name in names:
        if name in frame.columns:
            return name
    return None

@st.cache_data(show_spinner=False)
def load_data(path_string):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv was not found. "
            "Keep it in the same GitHub folder as app.py."
        )

    df = clean_columns(pd.read_csv(path))

    aliases = {
        "date":["date","datetime","time","observation_date"],
        "latitude":["latitude","lat"],
        "longitude":["longitude","lon","lng"],
        "chla":["chla","chlorophyll_a","chlorophyll","chlorophyll_a_mg_m3"],
        "risk":["risk","prediction","predicted_risk","risk_label","bloom_risk"],
        "probability":["risk_probability","probability","risk_prob","predicted_probability","prediction_probability"],
        "previous_chla":["previous_chla","prev_chla"],
        "historical_baseline":["historical_baseline","baseline"],
        "recent_mean":["recent_mean","rolling_mean"],
        "recent_max":["recent_max","rolling_max"],
    }

    rename_map = {}
    for standard, options in aliases.items():
        found = first_existing(df, options)
        if found and found != standard:
            rename_map[found] = standard
    df = df.rename(columns=rename_map)

    missing = [c for c in ["latitude","longitude","chla"] if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    if "date" not in df.columns:
        df["date"] = pd.Timestamp("2026-03-30")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["latitude","longitude","chla","previous_chla","historical_baseline","recent_mean","recent_max","probability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "risk" not in df.columns:
        df["risk"] = 0
    elif df["risk"].dtype == object:
        text = df["risk"].astype(str).str.lower().str.strip()
        df["risk"] = np.where(
            text.str.contains("risk|flag|potential|bloom|yes|true|1", regex=True), 1, 0
        )
    else:
        df["risk"] = (pd.to_numeric(df["risk"], errors="coerce").fillna(0) > 0).astype(int)

    if "probability" not in df.columns:
        df["probability"] = np.where(df["risk"] == 1, 1.0, 0.0)
    else:
        if df["probability"].max(skipna=True) > 1:
            df["probability"] = df["probability"] / 100.0
        df["probability"] = df["probability"].clip(0,1).fillna(0)

    if "previous_chla" in df.columns:
        df["chla_change"] = df["chla"] - df["previous_chla"]
    else:
        df["chla_change"] = np.nan

    if "historical_baseline" in df.columns:
        df["chla_anomaly"] = df["chla"] - df["historical_baseline"]
    else:
        df["chla_anomaly"] = np.nan

    df["risk_label"] = np.where(df["risk"] == 1, "Potential Bloom Risk", "Normal")
    df = df.dropna(subset=["latitude","longitude","chla"]).copy()
    return df.sort_values(["date","latitude","longitude"]).reset_index(drop=True)

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

def section_header(kicker, title, copy=""):
    extra = f"<div class='section-copy'>{copy}</div>" if copy else ""
    st.markdown(
        f"""
        <div class="section-kicker">{kicker}</div>
        <div class="section-title">{title}</div>
        {extra}
        """,
        unsafe_allow_html=True,
    )

def card(title, text, icon=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{icon} {title}</div>
            <div class="card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def go_page(page_name):
    st.session_state["page"] = page_name
    st.rerun()

def nav_button(label, page_name, icon=""):
    active = st.session_state.get("page","Home") == page_name
    if active:
        st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if st.button(f"{icon}{label}", key=f"nav_{page_name}"):
        go_page(page_name)
    if active:
        st.markdown("</div>", unsafe_allow_html=True)

def format_date(value):
    if pd.isna(value):
        return "Not available"
    return pd.Timestamp(value).strftime("%d %b %Y")

def probability_pct(value):
    value = float(value)
    if value <= 1:
        value *= 100
    return f"{value:.1f}%"

def make_map(data, show_risk=True, lat_range=(-40,30), lon_range=(20,120)):
    work = data.copy()
    work["lat_bin"] = np.floor(work["latitude"]).astype(int)
    work["lon_bin"] = np.floor(work["longitude"]).astype(int)

    normal = work.groupby(["lat_bin","lon_bin"],as_index=False).agg(chla=("chla","mean"))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=normal["lon_bin"] + 0.5,
            lat=normal["lat_bin"] + 0.5,
            mode="markers",
            marker=dict(size=8,color="#4B9CC3",opacity=.30),
            name="Normal / processed field",
            customdata=normal["chla"],
            hovertemplate="Lon %{lon:.2f}°<br>Lat %{lat:.2f}°<br>Mean Chl-a %{customdata:.3f}<extra></extra>",
        )
    )

    if show_risk:
        risk = work[work["risk"] == 1].copy()
        if not risk.empty:
            risk["r_lat"] = np.floor(risk["latitude"]/2)*2
            risk["r_lon"] = np.floor(risk["longitude"]/2)*2
            clusters = risk.groupby(["r_lat","r_lon"],as_index=False).agg(count=("risk","size"))

            fig.add_trace(
                go.Scattergeo(
                    lon=clusters["r_lon"],lat=clusters["r_lat"],mode="markers",
                    marker=dict(
                        size=np.clip(10 + clusters["count"]*1.3,12,30),
                        color="#48B47D",opacity=.42,
                        line=dict(color="#2F8E65",width=1),
                    ),
                    name="Risk concentration",
                    customdata=clusters["count"],
                    hovertemplate="Risk concentration<br>Lon %{lon:.1f}°<br>Lat %{lat:.1f}°<br>Flagged cells %{customdata}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scattergeo(
                    lon=risk["longitude"],lat=risk["latitude"],mode="markers",
                    marker=dict(size=5.5,color="#D64545",opacity=.84,line=dict(color="#8E2323",width=.5)),
                    name="Potential bloom-risk cell",
                    customdata=risk[["chla","probability"]].to_numpy(),
                    hovertemplate=(
                        "Potential bloom-risk cell"
                        "<br>Lat %{lat:.3f}°<br>Lon %{lon:.3f}°"
                        "<br>Chl-a %{customdata[0]:.3f}"
                        "<br>Risk probability %{customdata[1]:.1%}<extra></extra>"
                    ),
                )
            )

    fig.update_geos(
        projection_type="mercator",
        lataxis=dict(range=list(lat_range)),
        lonaxis=dict(range=list(lon_range)),
        showland=True,landcolor="#DDE7E9",showocean=True,oceancolor="#D9F1F3",
        showcountries=True,countrycolor="#7A9AA1",coastlinecolor="#5D7D84",
        coastlinewidth=.8,showlakes=True,lakecolor="#D9F1F3",bgcolor="rgba(0,0,0,0)",
        resolution=50,
    )
    fig.update_layout(
        height=590,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",yanchor="bottom",y=.01,xanchor="center",x=.5,
            bgcolor="rgba(255,255,255,.88)",bordercolor="#B8DDE0",borderwidth=1,
            font=dict(size=11),
        ),
        font=dict(family="DM Sans"),
    )
    return fig

# -----------------------------
# Load dataset
# -----------------------------
try:
    df = load_data(str(CSV_PATH))
except Exception as exc:
    st.error("BloomDetect AI could not load its dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()
if latest.empty:
    latest = df.copy()

latest_cells = len(latest)
latest_risk = int(latest["risk"].sum())
latest_risk_share = latest_risk / latest_cells * 100 if latest_cells else 0.0
total_dates = int(df["date"].nunique())
min_date = df["date"].min()
max_date = df["date"].max()

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div class="brand-bar">
        <div class="brand-left">
            <div class="logo-mark">🌊</div>
            <div>
                <div class="brand-name">BloomDetect AI</div>
                <div class="brand-sub">Coastal &amp; Ocean Intelligence Platform · EOS-06 OCM-3</div>
            </div>
        </div>
        <div class="brand-date">
            Latest processed field<br><b>{format_date(latest_date)}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    ("Home","Home","⌂ "),("Risk Map","Risk Map","🌍 "),("Hotspots","Hotspots","🔥 "),
    ("Location","Location","📍 "),("Insights","Insights","📈 "),
    ("Satellite Explorer","Satellite Explorer","🛰️ "),("Environment","Environment","🌿 "),
    ("Fisheries","Fisheries","🐟 "),("Early Warning","Early Warning","🚨 "),
    ("Data & Research","Data & Research","📊 "),
]

st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
row1 = st.columns(5)
for col,item in zip(row1,pages[:5]):
    with col: nav_button(item[0],item[1],item[2])
row2 = st.columns(5)
for col,item in zip(row2,pages[5:]):
    with col: nav_button(item[0],item[1],item[2])
st.markdown("</div>", unsafe_allow_html=True)

page = st.session_state["page"]

# ============================================================
# HOME
# ============================================================
if page == "Home":
    st.markdown(
        """
        <section class="hero">
            <div class="hero-content">
                <div class="kicker">EOS-06 · OCM-3 · STAGE 1</div>
                <h1>See the ocean differently.</h1>
                <p>
                    BloomDetect AI turns satellite-derived chlorophyll-a observations
                    into a practical coastal and ocean screening workspace.
                    Explore the latest field, locate potential bloom-risk cells,
                    inspect hotspots, and study temporal patterns from the available dataset.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Latest cells",f"{latest_cells:,}","processed grid cells")
    with c2: metric_card("Potential-risk cells",f"{latest_risk:,}","latest screening")
    with c3: metric_card("Risk share",f"{latest_risk_share:.2f}%","latest field")
    with c4: metric_card("Grid","0.25°","source product")

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    section_header(
        "01 · OCEAN & BLOOM RISK",
        "A screening platform, not a claim of confirmed HAB.",
        "The current dataset provides satellite-derived chlorophyll-a. Stage 1 therefore reports potential bloom risk and related ocean indicators rather than species or toxin confirmation.",
    )

    a,b,c = st.columns(3)
    with a: card("Risk Map","View the complete study region with the latest processed field in blue, risk concentrations in green, and flagged cells in red.","🌍")
    with b: card("Hotspot Intelligence","Identify where potential-risk cells are concentrated and summarize the most active locations in the latest field.","🔥")
    with c: card("Location Intelligence","Enter a coordinate and inspect the nearest processed observation, chlorophyll-a, and screening status.","📍")

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    section_header("02 · PLATFORM","Stage 1 intelligence modules","Everything here is powered by the dataset already processed for BloomDetect AI.")

    modules = [
        ("Satellite Explorer","Browse observation dates, coverage, chlorophyll-a summaries, and the latest satellite field.","🛰️"),
        ("Environment","Use chlorophyll-a distribution, anomalies and variability as satellite-derived environmental indicators.","🌿"),
        ("Fisheries","Explore a chlorophyll-a based productivity proxy. It is an indicator, not a fish-abundance measurement.","🐟"),
        ("Early Warning","Translate screening results into a compact monitoring view with alert counts and priority areas.","🚨"),
        ("Insights","See the key temporal and spatial patterns without turning the dashboard into a graph museum.","📈"),
        ("Data & Research","Inspect source-product details, dataset coverage, field statistics and downloadable results.","📊"),
    ]
    cols = st.columns(3)
    for i,(title,desc,icon) in enumerate(modules):
        with cols[i%3]: card(title,desc,icon)
        if i in [2,5]: st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    section_header("03 · HOW IT WORKS","From satellite observation to screening","")
    steps = [
        ("01","Satellite field","EOS-06 OCM-3 analysed chlorophyll-a observations are loaded from the processed dataset."),
        ("02","Historical context","Previous and historical values provide temporal context where available in the processed data."),
        ("03","AI screening","The trained BloomDetect model provides the stored potential-risk prediction used by the dashboard."),
        ("04","Action","Flagged cells are surfaced for further observation and field validation."),
    ]
    step_cols = st.columns(4)
    for col,(num,title,desc) in zip(step_cols,steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{num}</div><h3>{title}</h3><p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    if HERO_IMAGE.exists():
        st.image(str(HERO_IMAGE),use_container_width=True)

    if PHYTO_IMAGE.exists():
        with st.expander("View the phytoplankton signal visual"):
            st.image(str(PHYTO_IMAGE),use_container_width=True)

    st.markdown(
        """
        <div class="notice">
            <b>Important:</b> BloomDetect AI is an early-warning support and screening platform.
            Chlorophyll-a alone does not confirm a harmful algal bloom, species, toxin,
            ecological impact, or human-health impact. Satellite screening should be followed
            by appropriate scientific and field validation.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# RISK MAP
# ============================================================
elif page == "Risk Map":
    section_header(
        "02 · OCEAN & BLOOM RISK",
        "Potential bloom-risk map",
        "Blue shows the latest processed chlorophyll-a field. Green shows concentrations of flagged cells. Red shows individual potential bloom-risk cells.",
    )
    map_left,map_right = st.columns([3.2,1])
    with map_right:
        metric_card("Study area","20°–120°E","40°S–30°N")
        st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
        metric_card("Flagged cells",f"{latest_risk:,}","latest field")
        st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
        metric_card("Risk share",f"{latest_risk_share:.2f}%","latest field")
    with map_left:
        show_risk = st.checkbox("Show risk concentrations and flagged cells",value=True)
        st.plotly_chart(make_map(latest,show_risk=show_risk),use_container_width=True,config={"displayModeBar":False})
    st.markdown(
        """
        <div class="notice">
            <b>Reading the map:</b> Blue indicates processed satellite coverage in the
            study region. Green indicates spatial concentrations of flagged cells.
            Red indicates individual cells screened as potential bloom risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HOTSPOTS
# ============================================================
elif page == "Hotspots":
    section_header(
        "03 · HOTSPOT INTELLIGENCE",
        "Where is the latest risk concentrated?",
        "Hotspots are spatial summaries of the latest flagged cells. They are screening locations for further observation, not confirmed bloom boundaries.",
    )
    risk = latest[latest["risk"]==1].copy()
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Flagged cells",f"{len(risk):,}","latest field")
    with c2: metric_card("Risk share",f"{latest_risk_share:.2f}%","of processed cells")
    with c3: metric_card("Highest Chl-a",f"{risk['chla'].max():.3f}" if len(risk) else "—","among flagged cells")

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    if risk.empty:
        st.markdown('<div class="status status-green"><div class="status-title">No potential-risk cells in the latest field.</div><div class="status-copy">The latest processed observation contains no stored flagged cells.</div></div>',unsafe_allow_html=True)
    else:
        risk["lat_zone"] = (np.floor(risk["latitude"]/2)*2).astype(int)
        risk["lon_zone"] = (np.floor(risk["longitude"]/2)*2).astype(int)
        hotspots = (
            risk.groupby(["lat_zone","lon_zone"],as_index=False)
            .agg(flagged_cells=("risk","size"),mean_chla=("chla","mean"),max_chla=("chla","max"),mean_probability=("probability","mean"))
            .sort_values(["flagged_cells","max_chla"],ascending=False).head(12)
        )
        h1,h2 = st.columns([1.4,1])
        with h1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hotspots["flagged_cells"],
                y=[f"{lat}°, {lon}°" for lat,lon in zip(hotspots["lat_zone"],hotspots["lon_zone"])],
                orientation="h",marker_color="#2F9E72",
                hovertemplate="Flagged cells %{x}<extra></extra>",
            ))
            fig.update_layout(
                height=430,margin=dict(l=10,r=10,t=20,b=20),
                xaxis_title="Flagged cells",yaxis_title="2° × 2° zone",
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans"),
            )
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with h2:
            st.markdown('<div class="card"><div class="card-title">Top screening zones</div><div class="card-text">Zones are grouped only to make the spatial pattern easier to read.</div></div>',unsafe_allow_html=True)
            table = hotspots.rename(columns={
                "lat_zone":"Latitude zone","lon_zone":"Longitude zone","flagged_cells":"Flagged cells",
                "mean_chla":"Mean Chl-a","max_chla":"Max Chl-a","mean_probability":"Mean risk probability"
            }).copy()
            table["Mean Chl-a"] = table["Mean Chl-a"].round(3)
            table["Max Chl-a"] = table["Max Chl-a"].round(3)
            table["Mean risk probability"] = (table["Mean risk probability"]*100).round(1).astype(str)+"%"
            st.dataframe(table,use_container_width=True,hide_index=True,height=430)

# ============================================================
# LOCATION
# ============================================================
elif page == "Location":
    section_header(
        "04 · LOCATION INTELLIGENCE",
        "Screen a coordinate",
        "Enter a latitude and longitude. The dashboard finds the nearest processed observation in the latest satellite field.",
    )
    left,right = st.columns([1,1.15])
    with left:
        st.markdown('<div class="card"><div class="card-title">Your coordinate</div><div class="card-text">Use decimal degrees. Example: 17.3850, 78.4867.</div></div>',unsafe_allow_html=True)
        lat = st.number_input("Latitude",min_value=-90.0,max_value=90.0,value=17.3850,step=.01,format="%.4f")
        lon = st.number_input("Longitude",min_value=-180.0,max_value=180.0,value=78.4867,step=.01,format="%.4f")
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:12px;">
                <div class="metric-label">Your input</div>
                <div class="metric-value" style="font-size:1.25rem;">{lat:.4f}°, {lon:.4f}°</div>
                <div class="metric-note">latitude, longitude</div>
            </div>
            """,unsafe_allow_html=True,
        )

    coords = latest[["latitude","longitude"]].to_numpy()
    distances = (coords[:,0]-lat)**2 + (coords[:,1]-lon)**2
    nearest = latest.iloc[int(np.argmin(distances))]

    with right:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Nearest processed observation</div>
                <div class="metric-value" style="font-size:1.25rem;">{nearest['latitude']:.4f}°, {nearest['longitude']:.4f}°</div>
                <div class="metric-note">from the latest processed field</div>
            </div>
            """,unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
        if int(nearest["risk"])==1:
            st.markdown('<div class="status status-red"><div class="status-title">🔴 Potential bloom risk flagged</div><div class="status-copy">The nearest processed cell is classified as potential bloom risk in the stored screening result.</div></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="status status-green"><div class="status-title">🟢 Not flagged in this screening</div><div class="status-copy">The nearest processed cell is not classified as potential bloom risk in the stored screening result.</div></div>',unsafe_allow_html=True)

        m1,m2 = st.columns(2)
        with m1: metric_card("Chl-a",f"{nearest['chla']:.4f}","satellite-derived")
        with m2: metric_card("Risk probability",probability_pct(nearest["probability"]),"stored model output")

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    anom_text = f"{nearest['chla_anomaly']:.4f}" if pd.notna(nearest["chla_anomaly"]) else "Not available"
    change_text = f"{nearest['chla_change']:.4f}" if pd.notna(nearest["chla_change"]) else "Not available"
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Latest Chl-a",f"{nearest['chla']:.4f}","at nearest cell")
    with c2: metric_card("Chl-a anomaly",anom_text,"vs available baseline")
    with c3: metric_card("Chl-a change",change_text,"vs previous observation")

# ============================================================
# INSIGHTS
# ============================================================
elif page == "Insights":
    section_header(
        "05 · OCEAN TRENDS",
        "Useful patterns from the available dataset",
        "Compact indicators only. The point is to see the pattern, not to bury it under charts.",
    )
    daily = df.groupby("date",as_index=False).agg(
        mean_chla=("chla","mean"),max_chla=("chla","max"),
        risk_cells=("risk","sum"),processed_cells=("risk","size")
    )
    daily["risk_share"] = daily["risk_cells"]/daily["processed_cells"]*100

    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Observation dates",f"{total_dates:,}","available in CSV")
    with c2: metric_card("Mean Chl-a, latest",f"{latest['chla'].mean():.4f}","latest field")
    with c3: metric_card("Risk share, latest",f"{latest_risk_share:.2f}%","latest field")

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    trend = go.Figure()
    trend.add_trace(go.Scatter(
        x=daily["date"],y=daily["mean_chla"],mode="lines",name="Mean Chl-a",
        line=dict(color="#0E8FA0",width=2.5),
    ))
    trend.update_layout(
        height=330,margin=dict(l=10,r=10,t=15,b=10),
        xaxis_title="Date",yaxis_title="Mean chlorophyll-a",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",
        font=dict(family="DM Sans"),legend=dict(orientation="h",y=1.08,x=0),
    )
    st.plotly_chart(trend,use_container_width=True,config={"displayModeBar":False})

    risktrend = go.Figure()
    risktrend.add_trace(go.Scatter(
        x=daily["date"],y=daily["risk_share"],mode="lines",name="Potential-risk share",
        line=dict(color="#D64545",width=2.3),fill="tozeroy",fillcolor="rgba(214,69,69,.10)",
    ))
    risktrend.update_layout(
        height=300,margin=dict(l=10,r=10,t=15,b=10),
        xaxis_title="Date",yaxis_title="Potential-risk share (%)",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(risktrend,use_container_width=True,config={"displayModeBar":False})
    st.markdown('<div class="notice"><b>Interpretation:</b> These trends describe the satellite-derived dataset and stored screening output. They do not establish ecological causation or confirmed harmful algal blooms.</div>',unsafe_allow_html=True)

# ============================================================
# SATELLITE EXPLORER
# ============================================================
elif page == "Satellite Explorer":
    section_header(
        "06 · SATELLITE EXPLORER",
        "Explore the EOS-06 OCM-3 observation field",
        "Stage 1 uses the processed chlorophyll-a product already available in the project.",
    )
    dates = sorted(pd.to_datetime(df["date"].dropna().unique()))
    selected_date = st.select_slider(
        "Observation date",options=dates,value=dates[-1],
        format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"),
    )
    selected = df[df["date"]==selected_date].copy()

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Selected date",format_date(selected_date),"observation")
    with c2: metric_card("Cells",f"{len(selected):,}","processed")
    with c3: metric_card("Mean Chl-a",f"{selected['chla'].mean():.4f}","satellite-derived")
    with c4: metric_card("Max Chl-a",f"{selected['chla'].max():.4f}","satellite-derived")

    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    st.plotly_chart(make_map(selected,show_risk=True),use_container_width=True,config={"displayModeBar":False})
    if SAT_IMAGE.exists():
        with st.expander("Satellite processing visual"):
            st.image(str(SAT_IMAGE),use_container_width=True)

# ============================================================
# ENVIRONMENT
# ============================================================
elif page == "Environment":
    section_header(
        "07 · COASTAL ENVIRONMENT",
        "Satellite-derived environmental indicators",
        "This module stays within the evidence available in the current dataset. It does not invent temperature, salinity, turbidity, pollution or dissolved oxygen values that are not present.",
    )
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Mean Chl-a",f"{latest['chla'].mean():.4f}","latest field")
    with c2: metric_card("Median Chl-a",f"{latest['chla'].median():.4f}","latest field")
    with c3: metric_card("Std. deviation",f"{latest['chla'].std():.4f}","latest field")

    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=latest["chla"],nbinsx=45,marker_color="#0E8FA0",opacity=.82,name="Chl-a"))
    fig.update_layout(
        height=360,margin=dict(l=10,r=10,t=20,b=10),
        xaxis_title="Chlorophyll-a",yaxis_title="Number of cells",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",
        font=dict(family="DM Sans"),showlegend=False,
    )
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    e1,e2 = st.columns(2)
    with e1: card("What Chl-a can indicate","Chlorophyll-a is a satellite-derived indicator related to phytoplankton biomass. Spatial and temporal changes can support ocean-colour and bloom-risk screening.","🌿")
    with e2: card("What it cannot confirm alone","Chlorophyll-a alone cannot identify a harmful species, toxin, exact ecological cause, or human-health risk. Those require additional evidence and validation.","🧪")

# ============================================================
# FISHERIES
# ============================================================
elif page == "Fisheries":
    section_header(
        "08 · FISHERIES INTELLIGENCE",
        "Chlorophyll-a based productivity indicator",
        "Stage 1 provides a satellite-derived productivity proxy only. It is not a fish-abundance, catch, stock, or fishing-zone prediction.",
    )
    valid = latest["chla"].replace([np.inf,-np.inf],np.nan).dropna()
    if len(valid):
        q25,q50,q75 = valid.quantile([.25,.50,.75])
        current_mean = valid.mean()
        band = "Lower relative productivity" if current_mean<=q25 else ("Moderate relative productivity" if current_mean<=q75 else "Higher relative productivity")
        c1,c2,c3 = st.columns(3)
        with c1: metric_card("Mean Chl-a",f"{current_mean:.4f}","latest field")
        with c2: metric_card("Relative band",band,"dataset-derived proxy")
        with c3: metric_card("Upper quartile",f"{q75:.4f}","latest field")
        st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Box(x=valid,name="Latest Chl-a distribution",marker_color="#2F9E72",line_color="#1D6F4E",boxmean=True))
        fig.update_layout(
            height=280,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="Chlorophyll-a",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",
            font=dict(family="DM Sans"),showlegend=False,
        )
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.markdown('<div class="notice"><b>Research note:</b> A full fisheries intelligence module should combine chlorophyll-a with sea-surface temperature, fronts, bathymetry, currents, fishery observations and/or catch data. Those variables are not present in the current Stage 1 CSV, so this page deliberately remains a proxy indicator.</div>',unsafe_allow_html=True)

# ============================================================
# EARLY WARNING
# ============================================================
elif page == "Early Warning":
    section_header(
        "09 · EARLY WARNING",
        "Latest screening status",
        "A compact operational view for prioritising cells for further observation.",
    )
    if latest_risk>0:
        overall_class="status-red"
        overall_title="🔴 Potential-risk cells detected"
        overall_copy=f"{latest_risk:,} of {latest_cells:,} latest processed cells are flagged by the stored screening output."
    else:
        overall_class="status-green"
        overall_title="🟢 No potential-risk cells detected"
        overall_copy="No cells are flagged in the latest stored screening output."
    st.markdown(f'<div class="status {overall_class}"><div class="status-title">{overall_title}</div><div class="status-copy">{overall_copy}</div></div>',unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Latest date",format_date(latest_date),"processed")
    with c2: metric_card("Processed cells",f"{latest_cells:,}","latest field")
    with c3: metric_card("Flagged",f"{latest_risk:,}","potential risk")
    with c4: metric_card("Share",f"{latest_risk_share:.2f}%","of latest field")

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    risk = latest[latest["risk"]==1].copy()
    if not risk.empty:
        display = risk.sort_values(["probability","chla"],ascending=False).head(20)[["latitude","longitude","chla","probability"]].copy()
        display["probability"] = display["probability"]*100
        display = display.rename(columns={"latitude":"Latitude","longitude":"Longitude","chla":"Chl-a","probability":"Risk probability (%)"})
        st.dataframe(display.round({"Latitude":4,"Longitude":4,"Chl-a":4,"Risk probability (%)":1}),use_container_width=True,hide_index=True,height=420)

    st.markdown('<div class="notice">The early-warning view is intended to support screening and prioritisation. It is not a real-time alert service and does not independently confirm a harmful algal bloom.</div>',unsafe_allow_html=True)

# ============================================================
# DATA & RESEARCH
# ============================================================
elif page == "Data & Research":
    section_header(
        "10 · DATA & RESEARCH",
        "Dataset, provenance and downloads",
        "The research page keeps the evidence trail visible and puts downloads in one predictable place.",
    )
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Rows",f"{len(df):,}","current CSV")
    with c2: metric_card("Dates",f"{total_dates:,}","observation dates")
    with c3: metric_card("Start",format_date(min_date),"dataset")
    with c4: metric_card("End",format_date(max_date),"dataset")

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    a,b = st.columns(2)
    with a: card("Source product","EOS-06 / Oceansat-3 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC).","🛰️")
    with b: card("Core variable","Chlorophyll-a (Chl-a), used as the central satellite-derived ocean-colour indicator in Stage 1.","🌊")

    st.markdown("<div style='height:16px'></div>",unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>Interpretation boundary:</b> The current Stage 1 dataset does not contain confirmed HAB species labels or toxin measurements. The dashboard therefore uses the term <b>potential bloom risk</b> and treats model output as screening support.</div>',unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    section_header("DOWNLOADS","Research-ready files","Keep the download area simple and predictable.")

    d1,d2 = st.columns(2)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    with d1:
        st.markdown('<div class="download-card"><h3>⬇ Latest screening dataset</h3><div class="download-note">CSV containing the processed observations and stored screening outputs used by this website.</div></div>',unsafe_allow_html=True)
        st.download_button("Download CSV",data=csv_bytes,file_name="bloomdetect_stage1_dataset.csv",mime="text/csv",use_container_width=True,key="download_csv")
    with d2:
        summary = f"""BloomDetect AI - Stage 1 Summary

Source: EOS-06 / OCM-3
Variable: Chlorophyll-a
Study period: {format_date(min_date)} to {format_date(max_date)}
Latest processed date: {format_date(latest_date)}
Rows: {len(df):,}
Latest cells: {latest_cells:,}
Latest potential-risk cells: {latest_risk:,}
Latest risk share: {latest_risk_share:.2f}%

Interpretation:
This is an early-warning support and screening platform.
Chlorophyll-a alone does not confirm a harmful algal bloom, species,
toxin, ecological impact or human-health impact.
"""
        st.markdown('<div class="download-card"><h3>⬇ Project screening summary</h3><div class="download-note">A small text summary suitable for documentation, review meetings and project records.</div></div>',unsafe_allow_html=True)
        st.download_button("Download summary",data=summary.encode("utf-8"),file_name="BloomDetect_AI_Stage1_Summary.txt",mime="text/plain",use_container_width=True,key="download_summary")

    st.markdown("<div style='height:20px'></div>",unsafe_allow_html=True)
    with st.expander("View dataset sample"):
        st.dataframe(df.head(100),use_container_width=True,hide_index=True,height=420)

st.markdown(
    f"""
    <div class="footer">
        BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening ·
        Latest processed data: {format_date(latest_date)}
    </div>
    """,
    unsafe_allow_html=True,
)
