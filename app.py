import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="BloomDetect AI | Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DATA PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"

# Optional full historical file. The current website does not require it.
HISTORY_PATH = BASE_DIR / "bloomdetect_clean_ml_data.csv"

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

html { scroll-behavior: smooth; }
body, [class*="css"] { font-family: "DM Sans", sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 75% 0%, rgba(40, 180, 194, .12), transparent 28%),
      radial-gradient(circle at 10% 18%, rgba(0, 111, 145, .10), transparent 25%),
      #041017;
    color: #edf9fa;
}
.block-container { max-width: 1450px; padding-top: 0.8rem; padding-bottom: 4rem; }
footer { visibility: hidden; }

/* top navigation */
.navbar {
    position: sticky; top: 0; z-index: 99;
    padding: 14px 0;
    background: rgba(4,16,23,.88);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(130,210,218,.10);
}
.nav-inner { display:flex; align-items:center; justify-content:space-between; gap:20px; }
.brand { font-family:Manrope,sans-serif; font-weight:800; font-size:1.05rem; letter-spacing:.02em; color:#f4ffff; }
.brand span { color:#5fd0d6; }
.navlinks { display:flex; gap:18px; flex-wrap:wrap; }
.navlinks a { color:#9ab7bd; text-decoration:none; font-size:.82rem; }
.navlinks a:hover { color:#7ce2e5; }

/* hero */
.hero {
    min-height: 650px;
    display:flex; align-items:center;
    position:relative; overflow:hidden;
    border-radius:0 0 36px 36px;
    padding:70px 7%;
    background:
      radial-gradient(circle at 72% 35%, rgba(49,210,215,.15), transparent 20%),
      radial-gradient(circle at 82% 68%, rgba(0,111,145,.24), transparent 35%),
      linear-gradient(180deg,#061a23 0%,#04232e 52%,#03151d 100%);
}
.hero:before {
    content:""; position:absolute; inset:auto -10% -170px -10%; height:330px;
    border-radius:50% 50% 0 0 / 100% 100% 0 0;
    background:linear-gradient(180deg,rgba(19,153,169,.28),rgba(2,53,67,.05));
    animation: swell 7s ease-in-out infinite;
}
.hero:after {
    content:""; position:absolute; width:650px; height:650px; right:-240px; top:-180px;
    border:1px solid rgba(108,220,224,.12); border-radius:50%;
    box-shadow:0 0 0 70px rgba(108,220,224,.025),0 0 0 150px rgba(108,220,224,.015);
}
@keyframes swell { 0%,100%{transform:translateX(-2%) scaleX(1)} 50%{transform:translateX(2%) scaleX(1.04)} }
.hero-content { position:relative; z-index:2; max-width:900px; }
.eyebrow { color:#65d5da; text-transform:uppercase; letter-spacing:.20em; font-size:.76rem; font-weight:700; margin-bottom:18px; }
.hero h1 { font-family:Manrope,sans-serif; font-size:clamp(3.5rem,8vw,7.2rem); line-height:.92; letter-spacing:-.055em; margin:0 0 26px; color:#f5ffff; }
.hero h1 span { color:#69d9dd; }
.hero-copy { color:#a7c4ca; font-size:1.15rem; line-height:1.75; max-width:690px; }
.hero-pill { display:inline-block; margin-top:26px; padding:9px 14px; border-radius:999px; border:1px solid rgba(105,217,221,.20); background:rgba(105,217,221,.07); color:#bdecef; font-size:.82rem; }

/* section */
.section { padding:90px 0 20px; scroll-margin-top:70px; }
.kicker { color:#5fd0d6; text-transform:uppercase; letter-spacing:.17em; font-size:.73rem; font-weight:700; margin-bottom:10px; }
.section h2 { font-family:Manrope,sans-serif; font-size:clamp(2rem,4vw,3.4rem); letter-spacing:-.04em; margin:0; color:#f2fcfd; }
.section-sub { color:#8eabb2; max-width:760px; line-height:1.7; margin-top:12px; }

/* metrics */
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:28px 0 20px; }
.metric-card { padding:24px; min-height:125px; border:1px solid rgba(130,210,218,.10); border-radius:18px; background:linear-gradient(145deg,rgba(11,39,49,.88),rgba(5,24,31,.82)); }
.metric-label { color:#7e9ca3; font-size:.77rem; text-transform:uppercase; letter-spacing:.10em; }
.metric-value { font-family:Manrope,sans-serif; font-size:2rem; font-weight:800; margin-top:8px; color:#efffff; }
.metric-note { color:#718e95; font-size:.78rem; margin-top:5px; }

/* cards */
.card { padding:26px; border:1px solid rgba(130,210,218,.10); border-radius:22px; background:rgba(7,28,36,.74); box-shadow:0 20px 60px rgba(0,0,0,.16); }
.card-title { font-family:Manrope,sans-serif; font-weight:800; font-size:1.25rem; color:#eafafb; }
.card-copy { color:#8faeb5; line-height:1.65; margin-top:8px; }
.callout { border-left:3px solid #63d4d9; padding:17px 20px; border-radius:0 14px 14px 0; background:rgba(99,212,217,.055); color:#b7d2d7; line-height:1.65; }

/* timeline-like story */
.story {
    padding:25px 0 10px;
    border-left:1px solid rgba(99,212,217,.20);
    margin-left:8px;
}
.story-row { display:flex; gap:20px; margin:0 0 26px -8px; }
.story-dot { flex:0 0 15px; width:15px; height:15px; margin-top:5px; border-radius:50%; background:#5fd0d6; box-shadow:0 0 0 7px rgba(95,208,214,.08); }
.story h3 { margin:0; font-family:Manrope,sans-serif; font-size:1.05rem; }
.story p { margin:6px 0 0; color:#8faeb5; line-height:1.6; }

/* feature cards */
.feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:26px; }
.feature-card { padding:25px; min-height:190px; border-radius:20px; border:1px solid rgba(130,210,218,.09); background:linear-gradient(145deg,rgba(9,37,47,.85),rgba(5,22,29,.85)); transition:transform .25s ease,border-color .25s ease; }
.feature-card:hover { transform:translateY(-4px); border-color:rgba(99,212,217,.28); }
.feature-num { color:#5fd0d6; font-size:.72rem; letter-spacing:.14em; font-weight:800; }
.feature-card h3 { font-family:Manrope,sans-serif; margin:10px 0 8px; font-size:1.18rem; }
.feature-card p { color:#8faeb5; line-height:1.6; margin:0; }

/* footer */
.footer { margin-top:110px; padding:55px 0 20px; border-top:1px solid rgba(130,210,218,.10); color:#708d94; }

@media(max-width:900px){
 .metric-grid{grid-template-columns:repeat(2,1fr)}
 .feature-grid{grid-template-columns:1fr}
 .hero{min-height:580px;padding:55px 7%}
 .navlinks{display:none}
}
@media(max-width:550px){.metric-grid{grid-template-columns:1fr}.hero h1{font-size:3.2rem}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data(show_spinner="Loading EOS-06 observations...")
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = ["date", "latitude", "longitude", "chla", "risk_label"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in ["latitude", "longitude", "chla", "risk_probability"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "latitude", "longitude"]).copy()
    return data

if not DATA_PATH.exists():
    st.error("Dataset file not found.")
    st.code("Place latest_bloom_risk_predictions.csv in the same folder as app.py")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error("BloomDetect AI could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest_df = df[df["date"].eq(latest_date)].copy()

# ============================================================
# HELPERS
# ============================================================
def norm_lon(values):
    x = pd.to_numeric(values, errors="coerce")
    return ((x + 180) % 360) - 180


def norm_lon_one(value):
    return ((float(value) + 180) % 360) - 180


def region_filter(data: pd.DataFrame, region: str) -> pd.DataFrame:
    lon = norm_lon(data["longitude"])
    if region == "Indian Ocean":
        return data[data["latitude"].between(-40, 30) & lon.between(20, 120)].copy()
    if region == "Arabian Sea":
        return data[data["latitude"].between(5, 25) & lon.between(50, 75)].copy()
    if region == "Bay of Bengal":
        return data[data["latitude"].between(5, 25) & lon.between(75, 100)].copy()
    return data.copy()


def ocean_map(data: pd.DataFrame, risk: pd.DataFrame, region: str) -> go.Figure:
    d = data.copy()
    r = risk.copy()
    d["plot_lon"] = norm_lon(d["longitude"])
    r["plot_lon"] = norm_lon(r["longitude"])

    fig = go.Figure()

    if len(d):
        # Keep the full observation layer. The latest CSV is ~70k points, which is manageable.
        fig.add_trace(go.Scattergeo(
            lon=d["plot_lon"], lat=d["latitude"], mode="markers",
            name="Available observations",
            marker=dict(size=3.4, color="#5ca8bb", opacity=.42),
            customdata=np.column_stack([d["chla"], d["risk_label"].astype(str)]),
            hovertemplate=(
                "Lat %{lat:.2f}°<br>Lon %{lon:.2f}°<br>"
                "Chlorophyll-a %{customdata[0]:.4f}<br>"
                "Status %{customdata[1]}<extra></extra>"
            ),
        ))

    if len(r):
        fig.add_trace(go.Scattergeo(
            lon=r["plot_lon"], lat=r["latitude"], mode="markers",
            name="Potential bloom risk",
            marker=dict(size=9, color="#ff5f64", opacity=.97, line=dict(color="#ffd8d9", width=1)),
            customdata=np.column_stack([r["chla"]]),
            hovertemplate=(
                "<b>Potential bloom-risk location</b><br>"
                "Lat %{lat:.2f}°<br>Lon %{lon:.2f}°<br>"
                "Chlorophyll-a %{customdata[0]:.4f}<extra></extra>"
            ),
        ))

    if region == "Global Ocean":
        geo = dict(
            projection_type="equirectangular", showland=True, landcolor="#d7dede",
            showocean=True, oceancolor="#061d27", showcountries=True,
            countrycolor="#718087", coastlinecolor="#83979c", showlakes=True,
            lakecolor="#061d27", lonaxis=dict(range=[-180,180]), lataxis=dict(range=[-60,60]),
        )
    elif region == "Arabian Sea":
        geo = dict(
            projection_type="mercator", showland=True, landcolor="#d7dede",
            showocean=True, oceancolor="#061d27", showcountries=True,
            countrycolor="#718087", coastlinecolor="#83979c",
            lonaxis=dict(range=[48,78]), lataxis=dict(range=[2,28]),
        )
    elif region == "Bay of Bengal":
        geo = dict(
            projection_type="mercator", showland=True, landcolor="#d7dede",
            showocean=True, oceancolor="#061d27", showcountries=True,
            countrycolor="#718087", coastlinecolor="#83979c",
            lonaxis=dict(range=[72,103]), lataxis=dict(range=[2,28]),
        )
    else:
        geo = dict(
            projection_type="mercator", showland=True, landcolor="#d7dede",
            showocean=True, oceancolor="#061d27", showcountries=True,
            countrycolor="#718087", coastlinecolor="#83979c",
            lonaxis=dict(range=[15,125]), lataxis=dict(range=[-43,33]),
        )

    fig.update_geos(**geo)
    fig.update_layout(
        height=670, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d7eef0"),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", bgcolor="rgba(4,16,23,.72)"),
    )
    return fig


def chla_band(value: float, series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if pd.isna(value) or len(clean) < 10:
        return "Not available"
    p75 = clean.quantile(.75)
    p90 = clean.quantile(.90)
    if value >= p90:
        return "Upper observed range"
    if value >= p75:
        return "Above the local dataset median"
    return "Within the lower 75% of this observation field"

# ============================================================
# TOP NAVIGATION
# ============================================================
st.markdown(
    """
<div class="navbar">
  <div class="nav-inner">
    <div class="brand">Bloom<span>Detect</span> AI</div>
    <div class="navlinks">
      <a href="#ocean">Ocean</a>
      <a href="#explorer">Explorer</a>
      <a href="#screening">Screening</a>
      <a href="#method">Method</a>
      <a href="#data">Data</a>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================
st.markdown(
    f"""
<section class="hero">
  <div class="hero-content">
    <div class="eyebrow">EOS-06 · OCM-3 · E06OCM_L4_AC</div>
    <h1>Reading the<br><span>ocean from space.</span></h1>
    <div class="hero-copy">
      BloomDetect AI turns satellite-derived chlorophyll-a observations into a spatial
      screening view of locations showing patterns associated with potential bloom risk.
      Explore the ocean field, inspect individual grid cells and identify places that may deserve further investigation.
    </div>
    <div class="hero-pill">Latest available observation · {latest_date:%d %B %Y}</div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# ============================================================
# OVERVIEW
# ============================================================
all_risk = latest_df[latest_df["risk_label"].astype(str).eq("Potential Bloom Risk")].copy()
risk_share = len(all_risk) / len(latest_df) * 100 if len(latest_df) else 0
mean_chla = pd.to_numeric(latest_df["chla"], errors="coerce").mean()

st.markdown('<section class="section" id="ocean">', unsafe_allow_html=True)
st.markdown('<div class="kicker">01 · The observation field</div><h2>See the ocean we actually measured.</h2>', unsafe_allow_html=True)
st.markdown(
    "<div class='section-sub'>The full latest-date observation grid stays visible. Potential-risk locations are an overlay, not the entire dataset. Regional views below are geographic windows into the same global-ocean source.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="metric-grid">
  <div class="metric-card"><div class="metric-label">Observation cells</div><div class="metric-value">{len(latest_df):,}</div><div class="metric-note">latest available date</div></div>
  <div class="metric-card"><div class="metric-label">Potential-risk cells</div><div class="metric-value">{len(all_risk):,}</div><div class="metric-note">model-derived screening flags</div></div>
  <div class="metric-card"><div class="metric-label">Flagged share</div><div class="metric-value">{risk_share:.2f}%</div><div class="metric-note">of latest observation cells</div></div>
  <div class="metric-card"><div class="metric-label">Mean chlorophyll-a</div><div class="metric-value">{mean_chla:.3f}</div><div class="metric-note">latest observation field</div></div>
</div>
""",
    unsafe_allow_html=True,
)

r1, r2 = st.columns([1.2, 3])
with r1:
    st.markdown('<div class="card"><div class="card-title">Explore a region</div><div class="card-copy">These are geographic filters, not separate datasets.</div></div>', unsafe_allow_html=True)
    region = st.selectbox("Geographic window", ["Global Ocean", "Indian Ocean", "Arabian Sea", "Bay of Bengal"], index=1, label_visibility="collapsed")
with r2:
    focus = region_filter(latest_df, region)
    focus_risk = focus[focus["risk_label"].astype(str).eq("Potential Bloom Risk")].copy()
    st.markdown(f'<div class="callout"><b>{region}</b> · {len(focus):,} available observation cells · {len(focus_risk):,} potential-risk cells. The red layer is the model screening overlay.</div>', unsafe_allow_html=True)

fig = ocean_map(focus, focus_risk, region)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

st.markdown(
    "<div class='callout'>Interpretation: muted points represent available satellite-derived observations. Red points are locations flagged by the project's model as <b>potential bloom risk</b>. A flag is a screening signal, not confirmation of a harmful algal bloom.</div>",
    unsafe_allow_html=True,
)
st.markdown('</section>', unsafe_allow_html=True)

# ============================================================
# OCEAN STORY
# ============================================================
st.markdown('<section class="section">', unsafe_allow_html=True)
st.markdown('<div class="kicker">02 · From pixels to a question</div><h2>What happens between the satellite and the map?</h2>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">The website is designed as a guided exploration rather than a pile of charts. Each step answers one practical question.</div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="story">
  <div class="story-row"><div class="story-dot"></div><div><h3>What did the satellite observe?</h3><p>EOS-06 OCM-3 provides analysed chlorophyll-a observations across the global ocean.</p></div></div>
  <div class="story-row"><div class="story-dot"></div><div><h3>Where are the observations?</h3><p>The latest observation field is returned to geographic grid cells so spatial patterns can be explored directly.</p></div></div>
  <div class="story-row"><div class="story-dot"></div><div><h3>Which cells deserve attention?</h3><p>The trained model marks a subset of locations as potential bloom-risk locations using the project's screening target.</p></div></div>
  <div class="story-row"><div class="story-dot"></div><div><h3>What should happen next?</h3><p>Flagged locations can be prioritized for additional oceanographic analysis and, where appropriate, field validation.</p></div></div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('</section>', unsafe_allow_html=True)

# ============================================================
# LOCATION EXPLORER
# ============================================================
st.markdown('<section class="section" id="explorer">', unsafe_allow_html=True)
st.markdown('<div class="kicker">03 · Location explorer</div><h2>Inspect any available grid cell.</h2>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Enter coordinates and BloomDetect finds the nearest available observation from the latest date.</div>', unsafe_allow_html=True)

x1, x2, x3 = st.columns([1,1,1])
with x1:
    user_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=15.0, step=0.25)
with x2:
    user_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=75.0, step=0.25)
with x3:
    inspect = st.button("Inspect nearest observation", type="primary", use_container_width=True)

if inspect:
    work = latest_df.dropna(subset=["latitude", "longitude"]).copy()
    target_lon = norm_lon_one(user_lon)
    plot_lon = norm_lon(work["longitude"])
    lat_scale = 111.0
    lon_scale = max(111.0 * math.cos(math.radians(user_lat)), 1.0)
    work["distance_km"] = np.sqrt(
        ((work["latitude"] - user_lat) * lat_scale) ** 2
        + ((plot_lon - target_lon) * lon_scale) ** 2
    )
    nearest = work.loc[work["distance_km"].idxmin()]
    status = str(nearest["risk_label"])
    chla = pd.to_numeric(pd.Series([nearest.get("chla")]), errors="coerce").iloc[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Nearest latitude", f"{nearest['latitude']:.2f}°")
    b.metric("Nearest longitude", f"{norm_lon_one(nearest['longitude']):.2f}°")
    c.metric("Chlorophyll-a", f"{chla:.4f}" if pd.notna(chla) else "N/A")
    d.metric("Approx. distance", f"{nearest['distance_km']:.1f} km")
    if status == "Potential Bloom Risk":
        st.warning("This nearest grid cell is flagged for potential bloom risk.")
    else:
        st.success("This nearest grid cell is not flagged for potential bloom risk.")

    band = chla_band(chla, latest_df["chla"])
    st.markdown(f"**Chlorophyll context:** {band} within the latest observation field.")
    st.caption("This comparison describes the latest observation field only. It is not a health, fisheries, or HAB severity assessment.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</section>', unsafe_allow_html=True)

# ============================================================
# SCREENING INSIGHTS
# ============================================================
st.markdown('<section class="section" id="screening">', unsafe_allow_html=True)
st.markdown('<div class="kicker">04 · Screening view</div><h2>Turn 70,000+ observations into a field queue.</h2>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">The following views use only information available in the current prediction file. No artificial probability bands are added.</div>', unsafe_allow_html=True)

s1, s2 = st.columns([1.1, 1.9])
with s1:
    st.markdown('<div class="card"><div class="card-title">Potential-risk locations</div><div class="card-copy">A model-derived screening subset from the latest observation date.</div></div>', unsafe_allow_html=True)
    st.metric("Flagged cells", f"{len(all_risk):,}")
    st.metric("Normal cells", f"{max(len(latest_df)-len(all_risk),0):,}")
    st.metric("Latest date", f"{latest_date:%d %b %Y}")
with s2:
    if len(all_risk):
        chla_risk = pd.to_numeric(all_risk["chla"], errors="coerce").dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=pd.to_numeric(latest_df["chla"], errors="coerce").dropna(), nbinsx=35, name="All observations", marker_color="#397e91", opacity=.55))
        fig2.add_trace(go.Histogram(x=chla_risk, nbinsx=35, name="Potential-risk cells", marker_color="#ff5f64", opacity=.82))
        fig2.update_layout(
            barmode="overlay", height=390, margin=dict(l=10,r=10,t=35,b=10),
            title="Chlorophyll-a distribution: full field vs flagged cells",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cfe5e8"),
            xaxis_title="Chlorophyll-a", yaxis_title="Observation count",
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No potential-risk locations are present in the latest observation.")

st.markdown("### Investigation queue")
if len(all_risk):
    queue = all_risk[["latitude","longitude","chla"]].copy()
    queue["chla"] = pd.to_numeric(queue["chla"], errors="coerce")
    queue = queue.sort_values("chla", ascending=False, na_position="last").head(30)
    queue.columns = ["Latitude", "Longitude", "Chlorophyll-a"]
    st.dataframe(queue, use_container_width=True, hide_index=True)
    st.caption("The queue is sorted by chlorophyll-a for inspection convenience. It is not a ranking of confirmed HAB severity.")
else:
    st.info("No flagged locations to display.")

st.markdown('</section>', unsafe_allow_html=True)

# ============================================================
# WHO USES IT
# ============================================================
st.markdown('<section class="section">', unsafe_allow_html=True)
st.markdown('<div class="kicker">05 · Real-world use</div><h2>Built for people who need to inspect the ocean.</h2>', unsafe_allow_html=True)
st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
features = [
    ("01", "Environmental teams", "Screen large observation areas and create a shortlist of locations for further environmental investigation."),
    ("02", "Marine researchers", "Inspect spatial chlorophyll patterns, compare regions and download the latest observation field."),
    ("03", "Field sampling teams", "Use the screening map as one input when deciding which locations may deserve additional measurements."),
    ("04", "Fisheries monitoring", "Provide an additional environmental indicator for areas that may warrant closer observation. The current system does not predict fish abundance."),
    ("05", "Coastal communities", "Make satellite-derived environmental information easier to explore and understand."),
    ("06", "Students & educators", "Use a real satellite-data and machine-learning workflow to explore ocean colour and geospatial AI."),
]
for n, title, desc in features:
    st.markdown(f'<div class="feature-card"><div class="feature-num">{n}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
st.markdown('</div></section>', unsafe_allow_html=True)

# ============================================================
# METHOD
# ============================================================
st.markdown('<section class="section" id="method">', unsafe_allow_html=True)
st.markdown('<div class="kicker">06 · Method</div><h2>How BloomDetect makes a screening signal.</h2>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">The current prototype uses satellite-derived chlorophyll-a and temporal context to classify locations as Normal or Potential Bloom Risk.</div>', unsafe_allow_html=True)

steps = [
    ("01", "Satellite observation", "EOS-06 OCM-3 analysed chlorophyll-a observations provide the source ocean field."),
    ("02", "Data preparation", "Records are cleaned, organised by location and date, and invalid values are removed."),
    ("03", "Historical context", "Previous observations, historical baseline and recent behaviour provide temporal context during model development."),
    ("04", "Machine learning", "A Decision Tree model is used in the current prototype to classify the constructed screening target."),
    ("05", "Spatial output", "Predictions are returned to their geographic cells and visualised on the ocean map."),
    ("06", "Further investigation", "Flagged locations are intended to support prioritisation for additional oceanographic or field validation."),
]
for n, title, desc in steps:
    st.markdown(f'<div class="feature-card" style="margin-bottom:12px;min-height:auto"><div class="feature-num">{n}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown('<div class="callout"><b>Scientific interpretation:</b> a potential bloom-risk flag is a model-derived screening signal. High chlorophyll-a alone does not prove a harmful algal bloom. The current dataset does not provide confirmed HAB species or toxin labels, and satellite observations do not replace field validation.</div>', unsafe_allow_html=True)
st.markdown('</section>', unsafe_allow_html=True)

# ============================================================
# DATA & DOWNLOADS
# ============================================================
st.markdown('<section class="section" id="data">', unsafe_allow_html=True)
st.markdown('<div class="kicker">07 · Data transparency</div><h2>Know exactly what you are looking at.</h2>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">The application separates what is measured from what the model infers.</div>', unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown('<div class="card"><div class="card-title">Source</div><div class="card-copy">EOS-06 / OCM-3 · E06OCM_L4_AC<br>Analysed chlorophyll-a · Global ocean</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card"><div class="card-title">Latest field</div><div class="card-copy">30 March 2026<br>70,812 observation cells</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><div class="card-title">Resolution</div><div class="card-copy">0.25° × 0.25° product grid<br>Regional controls are geographic filters.</div></div>', unsafe_allow_html=True)

st.markdown("### Export")
export_cols = [c for c in ["date","latitude","longitude","chla","risk_label"] if c in focus.columns]
export_df = focus[export_cols].copy()
st.download_button(
    "Download current map observations",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="bloomdetect_current_observations.csv",
    mime="text/csv",
    type="primary",
)

if len(all_risk):
    summary = (
        f"BloomDetect AI screening summary\n"
        f"Latest observation: {latest_date:%d %B %Y}\n"
        f"Available observation cells: {len(latest_df):,}\n"
        f"Potential bloom-risk cells: {len(all_risk):,}\n\n"
        "Interpretation: model-derived screening signal only. High chlorophyll-a alone does not confirm a harmful algal bloom. Further environmental and/or field validation is recommended.\n"
    )
    st.download_button("Download screening summary", data=summary, file_name="bloomdetect_screening_summary.txt", mime="text/plain")

st.markdown('</section>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<div class="footer">
  <div style="font-family:Manrope,sans-serif;font-size:1.15rem;font-weight:800;color:#eafafb">BloomDetect AI</div>
  <div style="margin-top:8px;max-width:760px;line-height:1.7">
    Satellite-based potential bloom-risk screening using EOS-06 OCM-3 analysed chlorophyll-a observations.
    Research prototype. Not a real-time emergency, public-health, fisheries, toxin or confirmed-HAB warning service.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
