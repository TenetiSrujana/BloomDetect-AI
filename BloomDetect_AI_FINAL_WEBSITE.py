from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI | FINAL WEBSITE
# Home | Risk Map | Location | Insights | Data
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
HISTORY_PATH = BASE_DIR / "bloomdetect_history.csv.gz"
IMAGE_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
    --ink:#103f49;
    --muted:#66858c;
    --aqua:#0aa7b4;
    --deep:#07586a;
    --line:#b8dfe2;
    --pale:#f1fbfb;
    --blue:#2484aa;
    --green:#22a879;
    --red:#ef4f5e;
    --shadow:0 14px 38px rgba(9,80,94,.09);
}

html,body,[data-testid="stAppViewContainer"]{
    background:#f1fbfb !important;
    color:var(--ink) !important;
    font-family:'DM Sans',sans-serif !important;
}
.stApp{
    background:
      radial-gradient(circle at 8% 12%,rgba(55,211,210,.08),transparent 24%),
      radial-gradient(circle at 92% 72%,rgba(19,156,175,.07),transparent 27%),
      linear-gradient(180deg,#fbffff 0%,#effafa 55%,#fbffff 100%) !important;
}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,
[data-testid="stSidebar"]{display:none !important;}
.block-container{max-width:1260px !important;padding:24px 34px 70px !important;}

.brand{
    display:flex;align-items:center;justify-content:space-between;gap:18px;
    padding:14px 18px;border:1.5px solid #c5e4e6;background:rgba(255,255,255,.94);
    border-radius:22px;box-shadow:var(--shadow);
}
.brand-left{display:flex;align-items:center;gap:13px;}
.logo{width:48px;height:48px;border-radius:15px;display:grid;place-items:center;
    background:linear-gradient(145deg,#2bcfd1,#087c8e);color:#fff;font-size:22px;font-weight:800;}
.brand-name{font:800 1.2rem Manrope,sans-serif;letter-spacing:-.03em;color:var(--ink);}
.brand-sub{font-size:.72rem;color:#78959b;margin-top:2px;}
.latest{text-align:right;font-size:.67rem;color:#78959b;line-height:1.4;}
.latest b{color:#174b56;font-size:.78rem;}

.nav-gap{margin:12px 0 7px;}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
    min-height:44px !important;border-radius:14px !important;background:#fff !important;
    border:1.8px solid #164e5b !important;color:#123f49 !important;font-weight:800 !important;
    box-shadow:0 6px 16px rgba(15,70,82,.07) !important;transition:.18s ease !important;
}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{
    background:#e4f8f8 !important;border-color:#087d8c !important;transform:translateY(-1px) !important;
}
.stButton>button[kind="primary"]{background:#d9f7f7 !important;border-color:#078c9b !important;}

.hero{
    position:relative;overflow:hidden;border-radius:30px;padding:55px 56px;
    background:linear-gradient(135deg,#063d51,#087080 58%,#12a9ad);
    box-shadow:0 26px 70px rgba(6,86,100,.16);margin-top:5px;
}
.hero:after{content:"";position:absolute;right:-140px;top:-160px;width:500px;height:500px;border-radius:50%;
    border:1px solid rgba(181,255,251,.17);box-shadow:0 0 0 55px rgba(181,255,251,.05),0 0 0 115px rgba(181,255,251,.025);}
.eyebrow{position:relative;z-index:2;color:#a6fffa;font:800 .68rem Manrope;letter-spacing:.17em;text-transform:uppercase;}
.hero h1{position:relative;z-index:2;font:800 clamp(3rem,6vw,5.5rem)/.92 Manrope;color:#e8ffff;letter-spacing:-.075em;margin:10px 0 18px;}
.hero p{position:relative;z-index:2;max-width:780px;color:#e0fbfb;font-size:1.03rem;line-height:1.75;margin:0;}
.badges{position:relative;z-index:2;display:flex;gap:8px;flex-wrap:wrap;margin-top:22px;}
.badge{padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.76rem;font-weight:700;}

.section-kicker{font:800 .67rem Manrope;color:#0797a5;letter-spacing:.17em;text-transform:uppercase;margin-top:35px;margin-bottom:9px;}
.section-title{font:800 clamp(2rem,3.7vw,3.35rem)/1.03 Manrope;color:var(--ink);letter-spacing:-.06em;margin:0 0 12px;}
.section-copy{color:var(--muted);line-height:1.65;font-size:.94rem;margin:0 0 18px;}

[data-testid="stMetric"]{background:linear-gradient(145deg,#fff,#eaf8f8) !important;border:1.5px solid #a6d5da !important;
    border-radius:18px !important;box-shadow:0 11px 28px rgba(15,91,101,.07) !important;padding:15px 16px !important;}
[data-testid="stMetricLabel"]{color:#6d8c93 !important;font-weight:800 !important;}
[data-testid="stMetricValue"]{color:#123f49 !important;font-family:'Manrope',sans-serif !important;font-weight:800 !important;}

.card-title{font:800 1.15rem Manrope;color:var(--ink);margin-bottom:7px;}
.small-copy{color:var(--muted);font-size:.87rem;line-height:1.6;}
.callout{padding:14px 16px;background:#e5f7f8;border-left:4px solid #11a9b2;border-radius:0 14px 14px 0;color:#52757c;font-size:.86rem;line-height:1.6;margin:16px 0;}
.good{padding:15px 17px;background:#eafaf4;border:1.5px solid #52b996;border-radius:15px;color:#176f58;}
.warn{padding:15px 17px;background:#fff0f2;border:1.5px solid #ef7a87;border-radius:15px;color:#9e2d3c;}

[data-testid="stImage"] img{border-radius:18px !important;box-shadow:0 12px 30px rgba(9,80,94,.10) !important;}
[data-testid="stDataFrame"]{border:1px solid #c7e4e6;border-radius:14px;overflow:hidden;}

.footer{border-top:1px solid #d5ebed;margin-top:42px;padding-top:15px;color:#76959b;font-size:.69rem;}

@media(max-width:800px){.block-container{padding:16px 16px 50px !important}.hero{padding:42px 28px}.latest{display:none}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_latest():
    if not DATA_PATH.exists():
        raise FileNotFoundError("latest_bloom_risk_predictions.csv is missing beside app.py.")
    d = pd.read_csv(DATA_PATH)
    required = {"date","latitude","longitude","chla"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["latitude","longitude","chla","risk","risk_probability"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "risk" not in d.columns:
        if "risk_label" in d.columns:
            d["risk"] = d["risk_label"].astype(str).str.lower().eq("potential bloom risk").astype(int)
        else:
            d["risk"] = 0
    d["risk"] = d["risk"].fillna(0).astype(int)
    d["risk_flag"] = d["risk"].eq(1)
    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
    return d.dropna(subset=["date","latitude","longitude","chla"]).copy()


@st.cache_data(show_spinner=False)
def load_history():
    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size > 150_000_000:
        return pd.DataFrame()
    h = pd.read_csv(HISTORY_PATH, compression="gzip", parse_dates=["date"])
    needed = {"date","lat_bin","lon_bin","chla","risk"}
    if not needed.issubset(h.columns):
        return pd.DataFrame()
    for c in ["lat_bin","lon_bin","chla","risk","cells","max_chla","historical_baseline","chla_anomaly","chla_change"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h.dropna(subset=["date","lat_bin","lon_bin","chla"]).copy()


try:
    df = load_latest()
    history = load_history()
except Exception as exc:
    st.error("BloomDetect AI could not load the project data.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
risk_latest = latest[latest["risk_flag"]].copy()

# ============================================================
# HELPERS
# ============================================================

def navigate(page):
    st.session_state.page = page
    st.rerun()


def fmt(v, n=4):
    return "Not available" if pd.isna(v) else f"{float(v):.{n}f}"


def normalize_lon(v):
    return ((float(v) + 180) % 360) - 180


def nearest_latest(lat, lon):
    work = latest.dropna(subset=["latitude","longitude"]).copy()
    target_lon = normalize_lon(lon)
    work_lon = normalize_lon(work["longitude"])
    lat_scale = 111.0
    lon_scale = max(111.0 * abs(np.cos(np.deg2rad(lat))), 1.0)
    work["distance_km"] = np.sqrt(
        ((work["latitude"] - lat) * lat_scale) ** 2 +
        ((work_lon - target_lon) * lon_scale) ** 2
    )
    return work.loc[work["distance_km"].idxmin()].copy()


def history_context(lat, lon):
    if history.empty:
        return None
    lat_bin = np.floor(float(lat)) + 0.5
    lon_bin = np.floor(float(normalize_lon(lon))) + 0.5
    x = history[(history["lat_bin"].eq(lat_bin)) & (history["lon_bin"].eq(lon_bin))].sort_values("date")
    if x.empty:
        return None
    return x.iloc[-1]


def geo_base():
    return dict(
        projection_type="equirectangular",
        showland=True,
        landcolor="#dce9e7",
        showocean=True,
        oceancolor="#dff7f8",
        showcoastlines=True,
        coastlinecolor="#5b9198",
        coastlinewidth=.8,
        showcountries=True,
        countrycolor="#9ab9be",
        bgcolor="#dff7f8",
        lataxis=dict(range=[LAT_MIN, LAT_MAX]),
        lonaxis=dict(range=[LON_MIN, LON_MAX]),
    )


def latest_map():
    data = latest[
        latest["latitude"].between(LAT_MIN,LAT_MAX) &
        latest["plot_lon"].between(LON_MIN,LON_MAX)
    ].copy()
    data["lat_bin"] = np.floor(data["latitude"]) + .5
    data["lon_bin"] = np.floor(data["plot_lon"]) + .5
    blue = data.groupby(["lat_bin","lon_bin"],as_index=False).agg(chla=("chla","mean"),cells=("chla","size"))
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=blue.lat_bin,lon=blue.lon_bin,mode="markers",name="Processed ocean field",
        marker=dict(size=7,color="#2484aa",opacity=.62),
        customdata=np.column_stack([blue.lat_bin,blue.lon_bin,blue.chla,blue.cells]),
        hovertemplate="<b>Processed ocean field</b><br>Lat: %{customdata[0]:.1f}°<br>Lon: %{customdata[1]:.1f}°<br>Mean Chl-a: %{customdata[2]:.4f}<extra></extra>"
    ))
    r=data[data.risk_flag].copy()
    if not r.empty:
        r["zlat"]=np.floor(r.latitude/2)*2+1;r["zlon"]=np.floor(r.plot_lon/2)*2+1
        z=r.groupby(["zlat","zlon"],as_index=False).agg(flags=("risk_flag","size"),chla=("chla","mean"))
        fig.add_trace(go.Scattergeo(
            lat=z.zlat,lon=z.zlon,mode="markers",name="Flag concentration zone",
            marker=dict(size=np.clip(z.flags*1.2+8,9,28),color="#22a879",opacity=.52,line=dict(width=1.5,color="white")),
            customdata=np.column_stack([z.zlat,z.zlon,z.flags,z.chla]),
            hovertemplate="<b>Flag concentration zone</b><br>Centre: %{customdata[0]:.1f}°, %{customdata[1]:.1f}°<br>Flagged cells: %{customdata[2]}<br>Mean Chl-a: %{customdata[3]:.4f}<extra></extra>"
        ))
        fig.add_trace(go.Scattergeo(
            lat=r.latitude,lon=r.plot_lon,mode="markers",name="Potential bloom-risk cell",
            marker=dict(size=7,color="#ef4f5e",opacity=.96,line=dict(width=.8,color="white")),
            customdata=np.column_stack([r.latitude,r.longitude,r.chla]),
            hovertemplate="<b>Potential bloom-risk screening</b><br>Lat: %{customdata[0]:.2f}°<br>Lon: %{customdata[1]:.2f}°<br>Chl-a: %{customdata[2]:.4f}<extra></extra>"
        ))
    fig.update_geos(**geo_base())
    fig.update_layout(height=620,margin=dict(l=0,r=0,t=8,b=0),paper_bgcolor="#dff7f8",plot_bgcolor="#dff7f8",
        legend=dict(orientation="h",y=.01,x=.01,bgcolor="rgba(255,255,255,.88)",font=dict(size=11,color="#174b56")))
    return fig


def historical_map(selected_date):
    x=history[history.date.eq(selected_date)].copy()
    x=x[x.lat_bin.between(LAT_MIN,LAT_MAX)&x.lon_bin.between(LON_MIN,LON_MAX)]
    fig=go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=x.lat_bin,lon=x.lon_bin,mode="markers",name="Historical processed field",
        marker=dict(size=7,color="#2484aa",opacity=.62),
        customdata=np.column_stack([x.lat_bin,x.lon_bin,x.chla]),
        hovertemplate="<b>Historical processed field</b><br>Lat: %{customdata[0]:.1f}°<br>Lon: %{customdata[1]:.1f}°<br>Mean Chl-a: %{customdata[2]:.4f}<extra></extra>"
    ))
    r=x[x.risk.eq(1)]
    if not r.empty:
        z=r.copy();z["zlat"]=np.floor(z.lat_bin/2)*2+1;z["zlon"]=np.floor(z.lon_bin/2)*2+1
        z=z.groupby(["zlat","zlon"],as_index=False).agg(flags=("risk","sum"),chla=("chla","mean"))
        fig.add_trace(go.Scattergeo(
            lat=z.zlat,lon=z.zlon,mode="markers",name="Flag concentration zone",
            marker=dict(size=np.clip(z.flags*2+8,9,26),color="#22a879",opacity=.55,line=dict(width=1.5,color="white")),
            customdata=np.column_stack([z.zlat,z.zlon,z.flags,z.chla]),
            hovertemplate="<b>Flag concentration zone</b><br>Centre: %{customdata[0]:.1f}°, %{customdata[1]:.1f}°<br>Flagged cells: %{customdata[2]}<extra></extra>"
        ))
        fig.add_trace(go.Scattergeo(
            lat=r.lat_bin,lon=r.lon_bin,mode="markers",name="Potential bloom-risk screening",
            marker=dict(size=7,color="#ef4f5e",opacity=.95,line=dict(width=.8,color="white")),
            customdata=np.column_stack([r.lat_bin,r.lon_bin,r.chla]),
            hovertemplate="<b>Potential bloom-risk screening</b><br>Lat: %{customdata[0]:.1f}°<br>Lon: %{customdata[1]:.1f}°<br>Chl-a: %{customdata[2]:.4f}<extra></extra>"
        ))
    fig.update_geos(**geo_base())
    fig.update_layout(height=620,margin=dict(l=0,r=0,t=8,b=0),paper_bgcolor="#dff7f8",plot_bgcolor="#dff7f8",
        legend=dict(orientation="h",y=.01,x=.01,bgcolor="rgba(255,255,255,.88)",font=dict(size=11,color="#174b56")))
    return fig


def region_table():
    # Non-overlapping broad viewing regions.
    regions=[
        ("Arabian Sea", latest.latitude.between(5,30)&latest.plot_lon.between(45,75)),
        ("Bay of Bengal", latest.latitude.between(0,25)&latest.plot_lon.between(75,100)),
        ("Southern Indian Ocean", latest.latitude.between(-30,0)&latest.plot_lon.between(40,100)),
        ("Eastern Indian Ocean", latest.latitude.between(-30,30)&latest.plot_lon.between(100,120)),
    ]
    rows=[]
    for name,mask in regions:
        x=latest[mask]
        if len(x): rows.append({"Region":name,"Cells":len(x),"Potential-risk cells":int(x.risk_flag.sum()),
            "Risk share":100*x.risk_flag.mean(),"Mean Chl-a":x.chla.mean(),"Max Chl-a":x.chla.max()})
    return pd.DataFrame(rows)

# ============================================================
# HEADER / NAV
# ============================================================

if "page" not in st.session_state:
    st.session_state.page="home"

st.markdown(f"""
<div class="brand">
  <div class="brand-left"><div class="logo">≈</div><div>
    <div class="brand-name">BloomDetect AI</div>
    <div class="brand-sub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div>
  </div></div>
  <div class="latest">Latest processed field<br><b>{latest_date:%d %b %Y}</b></div>
</div>
""",unsafe_allow_html=True)

pages=[("home","⌂ Home"),("map","🗺 Risk Map"),("location","📍 Location"),("insights","📊 Insights"),("data","⇩ Data")]
nav=st.columns(5,gap="small")
for c,(key,label) in zip(nav,pages):
    with c:
        if st.button(label,key="nav_"+key,width="stretch",type="primary" if st.session_state.page==key else "secondary"):
            navigate(key)

# ============================================================
# HOME
# ============================================================

if st.session_state.page=="home":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">EOS-06 · OCM-3 · SATELLITE INTELLIGENCE</div>
      <h1>Read the ocean signal.</h1>
      <p>BloomDetect AI turns satellite-derived chlorophyll-a observations into a practical early-warning support view for identifying locations whose patterns may deserve closer investigation.</p>
      <div class="badges"><span class="badge">🌊 Ocean colour</span><span class="badge">🛰 EOS-06 OCM-3</span><span class="badge">🌱 Potential bloom-risk screening</span><span class="badge">📍 Spatial intelligence</span></div>
    </div>
    """,unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">PROJECT SNAPSHOT</div><div class="section-title">One clear view of the signal.</div>',unsafe_allow_html=True)
    st.markdown('<p class="section-copy">The dashboard starts from the latest processed satellite field and separates observation, screening and investigation so the same information is not repeated everywhere.</p>',unsafe_allow_html=True)

    m=st.columns(4,gap="medium")
    vals=[("Processed cells",f"{len(latest):,}","latest field"),("Potential-risk cells",f"{len(risk_latest):,}","screening output"),
          ("Risk share",f"{100*len(risk_latest)/len(latest):.2f}%","of processed cells"),("Maximum Chl-a",fmt(latest.chla.max()),"latest field")]
    for c,(a,b,d) in zip(m,vals):
        with c: st.metric(a,b,d)

    left,right=st.columns([1,1],gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">From satellite observation to investigation support</div>',unsafe_allow_html=True)
            st.markdown('<p class="small-copy">BloomDetect uses satellite-derived chlorophyll-a and temporal context to screen for unusual patterns. The output is a <b>potential bloom-risk signal</b>, not a confirmed harmful algal bloom.</p>',unsafe_allow_html=True)
            st.info("High chlorophyll-a alone does not prove a harmful algal bloom. Species, toxins and ecological impacts require additional evidence and field validation.")
    with right:
        if IMAGE_PATH.exists():
            st.image(str(IMAGE_PATH),caption="How satellite ocean-colour observations support bloom-risk investigation",use_container_width=True)
        else:
            st.info("Project illustration not included in this deployment.")

    st.markdown('<div class="section-kicker">FOUR FOCUSED VIEWS</div><div class="section-title">Everything has one job.</div>',unsafe_allow_html=True)
    f=st.columns(4,gap="medium")
    cards=[("🗺️","Risk Map","Explore the study area and move through available observation dates."),("📍","Location","Check one coordinate against the nearest processed satellite cell."),("📊","Insights","See hotspot concentration, regional patterns and current Chl-a distributions."),("⇩","Data","Keep the source product and downloadable outputs in one place.")]
    for c,(icon,title,desc) in zip(f,cards):
        with c:
            with st.container(border=True):
                st.markdown(f"### {icon} {title}")
                st.caption(desc)

# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page=="map":
    st.markdown('<div class="section-kicker">01 · SPATIAL INTELLIGENCE</div><div class="section-title">See how the field changes.</div>',unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Use the date slider to move through available observations. The latest date uses the final model screening output; earlier dates use the compact historical Chl-a screening layer.</p>',unsafe_allow_html=True)

    if not history.empty:
        dates=pd.Series(history.date.dt.normalize().drop_duplicates().sort_values().tolist())
        selected=st.slider("Observation date",min_value=dates.min().date(),max_value=dates.max().date(),value=dates.max().date(),format="DD MMM YYYY")
        selected_ts=pd.Timestamp(selected)
    else:
        selected_ts=latest_date
        st.info("The compact history file is not included yet, so the map is showing the latest processed field only.")

    if selected_ts.normalize()==latest_date.normalize():
        fig=latest_map()
        selected_cells=len(latest); selected_risk=len(risk_latest); chla_mean=latest.chla.mean()
        st.caption("30 Mar 2026-style latest-field view: blue = processed field, green = concentration zone, red = model screening output.")
    else:
        x=history[history.date.dt.normalize().eq(selected_ts.normalize())]
        fig=historical_map(selected_ts)
        selected_cells=len(x); selected_risk=int(x.risk.sum()); chla_mean=x.chla.mean()
        st.caption("Historical view: red cells are a screening layer derived from Chl-a temporal behaviour, not final model predictions.")

    a,b,c,d=st.columns(4,gap="medium")
    for col,label,value,note in [(a,"Selected date",selected_ts.strftime("%d %b %Y"),"observation"),(b,"Map cells",f"{selected_cells:,}","selected field"),(c,"Potential-risk",f"{selected_risk:,}","selected field"),(d,"Mean Chl-a",fmt(chla_mean),"selected field")]:
        with col: st.metric(label,value,note)

    st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"scrollZoom":True})
    st.markdown('<div class="callout"><b>Reading the map:</b> green indicates spatial concentration of screening flags. Red indicates potential bloom-risk screening. Neither colour confirms a harmful algal bloom or toxin presence.</div>',unsafe_allow_html=True)

# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page=="location":
    st.markdown('<div class="section-kicker">02 · LOCATION INTELLIGENCE</div><div class="section-title">Check one coordinate across the available evidence.</div>',unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Your input is shown separately from the nearest processed 0.25° satellite cell, so it is always clear which location was actually evaluated.</p>',unsafe_allow_html=True)

    if "lat_input" not in st.session_state: st.session_state.lat_input=17.4
    if "lon_input" not in st.session_state: st.session_state.lon_input=78.5
    left,right=st.columns([.82,1.18],gap="large")
    with left:
        with st.container(border=True):
            st.markdown("### Your input")
            st.caption("Enter decimal degrees. Example: 17.38, 78.49.")
            lat=st.number_input("Latitude",-90.0,90.0,float(st.session_state.lat_input),.25,"%.2f")
            lon=st.number_input("Longitude",-180.0,180.0,float(st.session_state.lon_input),.25,"%.2f")
            if st.button("🔎 Check location",type="primary",width="stretch"):
                st.session_state.lat_input=float(lat);st.session_state.lon_input=float(lon);st.rerun()
    lat=float(st.session_state.lat_input);lon=float(st.session_state.lon_input)
    row=nearest_latest(lat,lon)
    ctx=history_context(row.latitude,row.longitude)

    with right:
        with st.container(border=True):
            st.markdown("### Nearest processed cell")
            st.markdown(f"**{row.latitude:.4f}° · {normalize_lon(row.longitude):.4f}°**")
            st.caption(f"Your input: {lat:.4f}° · {lon:.4f}°  |  Approx. distance: {row.distance_km:.1f} km")
            r1,r2=st.columns(2)
            with r1: st.metric("Observation date",row.date.strftime("%d %b %Y"))
            with r2: st.metric("Chlorophyll-a",fmt(row.chla))
            if row.risk_flag:
                st.markdown('<div class="warn"><b>🔴 POTENTIAL BLOOM-RISK FLAG</b><br>This processed cell is included in the current screening output.</div>',unsafe_allow_html=True)
            else:
                st.markdown('<div class="good"><b>🟢 NOT FLAGGED</b><br>This processed cell is not included in the current potential-risk shortlist.</div>',unsafe_allow_html=True)
            if "risk_probability" in row.index and pd.notna(row.risk_probability):
                score=float(row.risk_probability)
                score=score/100 if score>1 else score
                st.progress(float(np.clip(score,0,1)),text=f"Model score: {score:.3f}  ·  not a calibrated HAB probability")

    st.markdown('<div class="section-kicker">SUPPORTING SIGNALS</div>',unsafe_allow_html=True)
    if ctx is not None:
        vals=[("Current Chl-a",ctx.chla),("Historical baseline",ctx.get("historical_baseline",np.nan)),("Anomaly",ctx.get("chla_anomaly",np.nan)),("Recent change",ctx.get("chla_change",np.nan))]
        st.caption("Historical context from the nearest compact 1° screening bin. The current Chl-a above remains the exact nearest 0.25° processed observation.")
    else:
        vals=[("Current Chl-a",row.chla),("Historical baseline",np.nan),("Anomaly",np.nan),("Recent change",np.nan)]
        st.caption("Historical context is unavailable because the compact history file is not included in this deployment. Values are not replaced with artificial zeros.")
    s=st.columns(4,gap="small")
    for c,(label,value) in zip(s,vals):
        with c: st.metric(label,fmt(value))

# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page=="insights":
    st.markdown('<div class="section-kicker">03 · INSIGHTS</div><div class="section-title">Where should the current signal receive closer attention?</div>',unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Hotspot concentration and current-field statistics are combined here. These are investigation aids, not severity rankings.</p>',unsafe_allow_html=True)

    risk_count=len(risk_latest);normal_count=len(latest)-risk_count;share=100*risk_count/len(latest)
    a,b,c,d=st.columns(4,gap="medium")
    for col,label,value,note in [(a,"Observation cells",f"{len(latest):,}","latest field"),(b,"Potential-risk",f"{risk_count:,}","screening output"),(c,"Risk share",f"{share:.2f}%","latest field"),(d,"Mean Chl-a",fmt(latest.chla.mean()),"latest field")]:
        with col: st.metric(label,value,note)

    # Hotspot concentration
    r=risk_latest.copy()
    if not r.empty:
        r["lat_zone"]=np.floor(r.latitude/2)*2;r["lon_zone"]=np.floor(r.plot_lon/2)*2
        z=r.groupby(["lat_zone","lon_zone"],as_index=False).agg(flagged_cells=("risk_flag","size"),mean_chla=("chla","mean"),max_chla=("chla","max"))
        z=z.sort_values(["flagged_cells","mean_chla"],ascending=False).head(12)
        z["Zone"]=z.apply(lambda q:f"{q.lat_zone:.0f}°–{q.lat_zone+2:.0f}°, {q.lon_zone:.0f}°–{q.lon_zone+2:.0f}°",axis=1)
        z["Flagged cells"]=z.flagged_cells.astype(int)
        z["Mean Chl-a"]=z.mean_chla.round(4);z["Max Chl-a"]=z.max_chla.round(4)
        st.markdown("### Top investigation zones")
        bar=go.Figure(go.Bar(x=z["Flagged cells"][::-1],y=z["Zone"][::-1],orientation="h",text=z["Flagged cells"][::-1],textposition="outside",marker_color="#22a879"))
        bar.update_layout(height=470,margin=dict(l=15,r=55,t=15,b=30),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#174b56"),xaxis_title="Potential-risk cells in zone",yaxis_title="Investigation zone",xaxis=dict(gridcolor="#d6e7e8"))
        st.plotly_chart(bar,width="stretch",config={"displaylogo":False})
        st.dataframe(z[["Zone","Flagged cells","Mean Chl-a","Max Chl-a"]],width="stretch",hide_index=True)
    else:
        st.success("No potential-risk cells are present in the latest field.")

    left,right=st.columns(2,gap="large")
    with left:
        chart=pd.DataFrame({"Screening group":["Normal","Potential bloom risk"],"Cells":[normal_count,risk_count]})
        fig=go.Figure(go.Bar(x=chart["Screening group"],y=chart.Cells,text=chart.Cells,textposition="outside",marker_color=["#2484aa","#ef4f5e"]))
        fig.update_layout(height=360,margin=dict(l=55,r=25,t=25,b=65),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",showlegend=False,font=dict(color="#174b56"),yaxis_title="Number of cells",xaxis_title="Screening group",yaxis=dict(gridcolor="#d6e7e8"))
        st.markdown("### Screening composition")
        st.plotly_chart(fig,width="stretch",config={"displaylogo":False})
    with right:
        fig=go.Figure(go.Histogram(x=latest.chla.dropna(),nbinsx=32,marker_color="#2484aa"))
        fig.update_layout(height=360,margin=dict(l=55,r=25,t=25,b=65),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",showlegend=False,font=dict(color="#174b56"),xaxis_title="Satellite-derived chlorophyll-a",yaxis_title="Number of cells",xaxis=dict(gridcolor="#d6e7e8"),yaxis=dict(gridcolor="#d6e7e8"))
        st.markdown("### Chlorophyll-a distribution")
        st.plotly_chart(fig,width="stretch",config={"displaylogo":False})

    st.markdown("### Broad regional comparison")
    rt=region_table()
    if not rt.empty:
        rt2=rt.copy();rt2["Risk share"]=rt2["Risk share"].map(lambda x:f"{x:.2f}%");rt2["Mean Chl-a"]=rt2["Mean Chl-a"].round(4);rt2["Max Chl-a"]=rt2["Max Chl-a"].round(4)
        st.dataframe(rt2,width="stretch",hide_index=True)

    if not history.empty:
        st.markdown("### How the field has changed over time")
        trend=history.groupby("date",as_index=False).agg(mean_chla=("chla","mean"),risk_cells=("risk","sum"))
        t=go.Figure()
        t.add_trace(go.Scatter(x=trend.date,y=trend.mean_chla,mode="lines",name="Mean Chl-a",line=dict(color="#2484aa",width=2.5)))
        t.update_layout(height=330,margin=dict(l=50,r=20,t=25,b=45),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#174b56"),xaxis_title="Observation date",yaxis_title="Mean Chl-a",yaxis=dict(gridcolor="#d6e7e8"))
        st.plotly_chart(t,width="stretch",config={"displaylogo":False})
        st.caption("Historical trend uses the compact 1° screening layer. It is descriptive context, not a second model evaluation.")

# ============================================================
# DATA
# ============================================================

elif st.session_state.page=="data":
    st.markdown('<div class="section-kicker">04 · DATA & OUTPUTS</div><div class="section-title">The evidence behind the dashboard.</div>',unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Source-product details and downloadable outputs live here. Other pages stay focused on analysis.</p>',unsafe_allow_html=True)

    a,b,c,d=st.columns(4,gap="medium")
    for col,label,value,note in [(a,"Product","E06OCM_L4_AC","EOS-06 / OCM-3"),(b,"Grid","0.25°","latitude × longitude"),(c,"Latest cells",f"{len(latest):,}","processed observation"),(d,"Latest date",latest_date.strftime("%d %b %Y"),"processed dataset")]:
        with col: st.metric(label,value,note)

    with st.container(border=True):
        st.markdown("### EOS-06 OCM-3 analysed chlorophyll-a")
        st.write("The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4 analysed chlorophyll product, E06OCM_L4_AC. The project output is a potential bloom-risk screening layer built from satellite-derived chlorophyll-a and temporal context.")

    st.markdown("### Project outputs")
    x,y=st.columns(2,gap="large")
    with x:
        with st.container(border=True):
            st.markdown("### Latest observation table")
            st.caption("All processed cells from the latest available field, including the screening output.")
            st.download_button("⬇ Download latest observations",latest.to_csv(index=False).encode("utf-8"),"bloomdetect_latest_observations.csv","text/csv",width="stretch")
    with y:
        with st.container(border=True):
            st.markdown("### Potential-risk shortlist")
            st.caption("Only cells currently screened as potential bloom risk in the latest field.")
            st.download_button("⬇ Download potential-risk locations",risk_latest.to_csv(index=False).encode("utf-8"),"bloomdetect_potential_risk.csv","text/csv",width="stretch")

    st.info("Scientific use: potential bloom risk is a screening result, not confirmation of a harmful algal bloom, species identity or toxin presence. Satellite observations should be combined with field observations and additional environmental evidence.")
    if not history.empty:
        st.success("Compact history layer is available for the date slider and historical supporting signals.")
    else:
        st.warning("Compact history layer is not included in this deployment. The latest-field dashboard still works correctly without it.")

# ============================================================
# FOOTER
# ============================================================

st.markdown(f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Latest processed field: {latest_date:%d %B %Y}</div>',unsafe_allow_html=True)
