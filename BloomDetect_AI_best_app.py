from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI | CLEAN FINAL WEBSITE
# 5 pages: Home / Risk Map / Location / Insights / Data
# No Explore page, no separate Hotspots page, no How It Works.
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI | Coastal & Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
LATEST_PATH = BASE / "latest_bloom_risk_predictions.csv"
# Optional lightweight history pack. If present, the timeline becomes active.
HISTORY_PATH = BASE / "bloomdetect_history.csv"
LOCATION_HISTORY_PATH = BASE / "bloomdetect_location_history.csv"
IMAGE_PATH = BASE / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# ============================================================
# THEME
# ============================================================

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--navy:#083f50;--deep:#07596b;--aqua:#10a9b5;--pale:#effafa;--ink:#143f49;--muted:#64838a;--line:#a8d9dd;--blue:#4aa7c0;--green:#2ca777;--red:#df4b58;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;color:var(--ink);}
.stApp{background:radial-gradient(circle at 10% 5%,rgba(20,184,193,.08),transparent 25%),linear-gradient(180deg,#fbffff 0%,#eefafa 55%,#fbffff 100%);overflow-x:hidden;}
#MainMenu,footer,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stSidebar"]{display:none!important;}
.block-container{max-width:1240px!important;padding:22px 30px 65px!important;}
.stApp:after{content:"";position:fixed;left:-8%;right:-8%;bottom:-180px;height:300px;pointer-events:none;z-index:0;background:radial-gradient(ellipse at 20% 50%,rgba(16,169,181,.11) 0 18%,transparent 19%),radial-gradient(ellipse at 55% 45%,rgba(54,208,210,.08) 0 20%,transparent 21%),radial-gradient(ellipse at 88% 58%,rgba(16,169,181,.10) 0 18%,transparent 19%);animation:water 12s ease-in-out infinite alternate;}
@keyframes water{from{transform:translateX(-2%)}to{transform:translateX(2%)}}
.brand{position:relative;z-index:5;display:flex;align-items:center;justify-content:space-between;gap:20px;padding:14px 18px;background:rgba(255,255,255,.92);border:1.5px solid var(--line);border-radius:20px;box-shadow:0 12px 32px rgba(8,80,95,.08);}
.brand-left{display:flex;align-items:center;gap:12px}.logo{width:46px;height:46px;border-radius:15px;display:grid;place-items:center;background:linear-gradient(145deg,#25c8cb,#087d8e);color:white;font-size:21px;font-weight:800}.brand-name{font:800 1.16rem Manrope;color:var(--navy)}.brand-sub,.brand-date{font-size:.70rem;color:var(--muted)}.brand-date{text-align:right}.brand-date b{color:var(--ink);font-size:.78rem}
.nav{position:relative;z-index:6;margin:12px 0 10px}.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:44px!important;border:2px solid #164e5b!important;border-radius:13px!important;background:#fff!important;color:#123f49!important;font-weight:800!important;font-size:.84rem!important;box-shadow:0 6px 16px rgba(15,70,82,.07)!important;transition:.18s!important}.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:#e0f8f8!important;border-color:#087d8c!important;transform:translateY(-1px)!important}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#d9f7f7,#b9eeee)!important;border-color:#078c9b!important;color:#07596b!important}
.section{position:relative;z-index:2;padding:30px 0 15px}.kicker{font:800 .67rem Manrope;letter-spacing:.16em;text-transform:uppercase;color:#0797a5;margin-bottom:8px}.section-title{font:800 clamp(2rem,4vw,3.25rem)/1.05 Manrope;color:var(--navy);letter-spacing:-.055em;margin-bottom:10px}.section-copy{max-width:1000px;color:var(--muted);font-size:.94rem;line-height:1.6}
.hero{position:relative;z-index:2;overflow:hidden;min-height:390px;border-radius:28px;margin-top:8px;padding:54px;background:linear-gradient(110deg,rgba(4,50,64,.95),rgba(5,105,119,.72)),url('https://images.unsplash.com/photo-1530053969600-caed2596d242?auto=format&fit=crop&w=1800&q=80') center/cover no-repeat;box-shadow:0 25px 60px rgba(6,86,100,.15)}.hero:after{content:"";position:absolute;left:-5%;right:-5%;bottom:-95px;height:180px;background:rgba(130,243,237,.14);border-radius:50%;animation:heroWave 7s ease-in-out infinite alternate}.hero-content{position:relative;z-index:3;max-width:820px}.hero h1{font:800 clamp(2.9rem,6vw,5.2rem)/.93 Manrope;color:#e9ffff;letter-spacing:-.07em;margin:15px 0}.hero p{max-width:760px;color:#e3ffff;font-size:1rem;line-height:1.75}.pill{display:inline-block;padding:7px 11px;margin:3px;border-radius:999px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.73rem;font-weight:700}@keyframes heroWave{from{transform:translateX(-2%) rotate(-1deg)}to{transform:translateX(2%) rotate(1deg)}}
.card,.metric,.chart-card,.download-card{position:relative;z-index:2;background:rgba(255,255,255,.90);border:1.5px solid var(--line);border-radius:19px;box-shadow:0 10px 28px rgba(8,80,95,.07);backdrop-filter:blur(8px)}.card{padding:20px;height:100%}.card-title{font:800 1.08rem Manrope;color:var(--navy);margin-bottom:7px}.card-text{color:var(--muted);font-size:.86rem;line-height:1.58}.metric{min-height:105px;padding:16px 17px}.metric-label{font:800 .64rem Manrope;letter-spacing:.10em;text-transform:uppercase;color:#6e8b92}.metric-value{font:800 1.43rem Manrope;color:var(--navy);margin-top:5px;white-space:nowrap}.metric-note{font-size:.70rem;color:var(--muted);margin-top:3px}.notice{position:relative;z-index:2;margin-top:16px;padding:13px 16px;background:#e6f8f8;border-left:4px solid var(--aqua);border-radius:0 13px 13px 0;color:#52757c;font-size:.82rem;line-height:1.55}.image-card{padding:8px;overflow:hidden}.image-card img{display:block;width:100%;height:300px;object-fit:cover;border-radius:13px}
.map-head{position:relative;z-index:2;display:flex;justify-content:space-between;gap:15px;align-items:center;background:#fff;border:1.5px solid var(--line);border-radius:17px 17px 0 0;padding:13px 16px;color:var(--muted);font-size:.74rem}.map-head b{color:var(--navy);font-size:.92rem}.legend{position:relative;z-index:2;display:flex;gap:17px;flex-wrap:wrap;background:#fff;border:1.5px solid var(--line);border-radius:0 0 17px 17px;padding:12px 16px;color:#5b7980;font-size:.76rem}.sw{display:inline-block;width:12px;height:12px;border-radius:4px;margin-right:5px;vertical-align:-2px}.sw.blue{background:var(--blue)}.sw.green{background:var(--green);border-radius:50%}.sw.red{background:var(--red);border-radius:50%}
.status{position:relative;z-index:2;padding:16px 18px;border-radius:16px;border:2px solid}.status.ok{background:#eaf9f2;border-color:#42b589;color:#176f58}.status.bad{background:#fff0f2;border-color:#e96876;color:#9e2d3c}.status-title{font:800 .94rem Manrope}.status-copy{font-size:.80rem;margin-top:4px;line-height:1.45}.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px}.result-item{padding:12px 13px;background:#f6fcfc;border:1px solid #c6e4e7;border-radius:13px}.result-item span{display:block;color:#78959b;font-size:.68rem;margin-bottom:4px}.result-item b{color:#194b55;font-size:.88rem}.signal-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.signal{padding:13px;border:1px solid #c6e4e7;background:#f8fcfc;border-radius:14px}.signal-label{font-size:.67rem;color:#78959b}.signal-value{font:800 .91rem Manrope;color:#174b56;margin-top:4px}
.chart-card{padding:14px}.download-card{padding:20px;min-height:135px;background:linear-gradient(135deg,#087c8d,#13adb3);border:2px solid #07576a;color:#fff}.download-card h3{font:800 1.08rem Manrope;color:#fff;margin:0 0 6px}.download-card p{font-size:.80rem;color:#e6ffff;line-height:1.45;margin:0}.footer{position:relative;z-index:2;border-top:1px solid #d5ebed;margin-top:38px;padding-top:15px;color:#78959b;font-size:.69rem}
.stSlider,.stSelectSlider{position:relative;z-index:2}.stNumberInput input{border:1.3px solid #8ebec4!important;border-radius:10px!important}.stDownloadButton>button{background:#fff!important;color:#123f49!important}.stDownloadButton>button:hover{background:#e2f8f8!important}.stDataFrame{position:relative;z-index:2}.stPlotlyChart{position:relative;z-index:2}
@media(max-width:900px){.block-container{padding:16px 16px 50px!important}.hero{padding:40px 28px}.brand-date{display:none}.signal-grid{grid-template-columns:1fr 1fr}}
@media(max-width:620px){.hero h1{font-size:2.8rem}.hero{min-height:410px;padding:34px 23px}.result-grid,.signal-grid{grid-template-columns:1fr}.map-head{align-items:flex-start;flex-direction:column}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

def normalize_columns(d):
    d=d.copy()
    d.columns=[str(c).strip().lower().replace(" ","_").replace("-","_") for c in d.columns]
    aliases={
        "date":["date","datetime","time","observation_date"],
        "latitude":["latitude","lat"],
        "longitude":["longitude","lon","lng"],
        "chla":["chla","chlorophyll_a","chlorophyll"],
        "risk":["risk","prediction","predicted_risk","risk_label","bloom_risk"],
        "probability":["risk_probability","probability","risk_prob","predicted_probability"],
        "previous_chla":["previous_chla","prev_chla"],
        "historical_baseline":["historical_baseline","baseline"],
        "chla_anomaly":["chla_anomaly","anomaly"],
        "chla_change":["chla_change","change"],
    }
    rename={}
    for std,names in aliases.items():
        for n in names:
            if n in d.columns:
                rename[n]=std; break
    return d.rename(columns=rename)

@st.cache_data(show_spinner=False)
def load_latest():
    if not LATEST_PATH.exists():
        raise FileNotFoundError("latest_bloom_risk_predictions.csv is missing beside app.py.")
    d=normalize_columns(pd.read_csv(LATEST_PATH))
    for c in ["latitude","longitude","chla","probability","previous_chla","historical_baseline","chla_anomaly","chla_change"]:
        if c in d.columns: d[c]=pd.to_numeric(d[c],errors="coerce")
    if "date" not in d.columns: d["date"]=pd.Timestamp("2026-03-30")
    d["date"]=pd.to_datetime(d["date"],errors="coerce")
    if "risk" not in d.columns: d["risk"]=0
    elif d["risk"].dtype==object:
        d["risk"]=d["risk"].astype(str).str.lower().str.contains("risk|flag|potential|bloom|yes|true").astype(int)
    else: d["risk"]=(pd.to_numeric(d["risk"],errors="coerce").fillna(0)>0).astype(int)
    if "probability" not in d.columns: d["probability"]=d["risk"].astype(float)
    else:
        if d["probability"].max(skipna=True)>1: d["probability"]/=100
        d["probability"]=d["probability"].clip(0,1).fillna(0)
    d=d.dropna(subset=["date","latitude","longitude","chla"]).copy()
    d["plot_lon"]=((d["longitude"]+180)%360)-180
    d["risk_label"]=np.where(d["risk"].eq(1),"Potential Bloom Risk","Normal")
    return d

@st.cache_data(show_spinner=False)
def load_history():
    """Loads the lightweight 2-degree map-history file, never the 20M-row ML table."""
    if not HISTORY_PATH.exists(): return pd.DataFrame()
    h=normalize_columns(pd.read_csv(HISTORY_PATH))
    # The history builder stores the map cell centre as lat_bin/lon_bin.
    if "latitude" not in h.columns and "lat_bin" in h.columns: h=h.rename(columns={"lat_bin":"latitude"})
    if "longitude" not in h.columns and "lon_bin" in h.columns: h=h.rename(columns={"lon_bin":"longitude"})
    for c in ["latitude","longitude","chla","probability","risk","cells","max_chla","chla_anomaly","chla_change"]:
        if c in h.columns: h[c]=pd.to_numeric(h[c],errors="coerce")
    if "date" not in h.columns: return pd.DataFrame()
    h["date"]=pd.to_datetime(h["date"],errors="coerce")
    if "risk" not in h.columns: h["risk"]=0
    h["risk"]=(pd.to_numeric(h["risk"],errors="coerce").fillna(0)>0).astype(int)
    h=h.dropna(subset=["date","latitude","longitude","chla"]).copy()
    h["plot_lon"]=((h["longitude"]+180)%360)-180
    # Map-history risk is the number of flagged cells in a 2-degree cell.
    h["risk"]=pd.to_numeric(h["risk"],errors="coerce").fillna(0)
    return h

@st.cache_data(show_spinner=False)
def load_location_history():
    if not LOCATION_HISTORY_PATH.exists(): return pd.DataFrame()
    h=normalize_columns(pd.read_csv(LOCATION_HISTORY_PATH))
    for c in ["latitude","longitude","chla","probability","risk","chla_anomaly","chla_change"]:
        if c in h.columns: h[c]=pd.to_numeric(h[c],errors="coerce")
    if "date" not in h.columns: return pd.DataFrame()
    h["date"]=pd.to_datetime(h["date"],errors="coerce")
    if "risk" not in h.columns: h["risk"]=0
    h["risk"]=(pd.to_numeric(h["risk"],errors="coerce").fillna(0)>0).astype(int)
    h=h.dropna(subset=["date","latitude","longitude","chla"]).copy()
    h["plot_lon"]=((h["longitude"]+180)%360)-180
    return h

try:
    df=load_latest()
except Exception as e:
    st.error("BloomDetect AI could not load the dataset.")
    st.code(str(e))
    st.stop()

history=load_history()
location_history=load_location_history()
latest_date=df["date"].max()
latest=df[df["date"].eq(latest_date)].copy()
latest_risk=int(latest["risk"].sum())
latest_share=(latest_risk/len(latest)*100) if len(latest) else 0

# ============================================================
# HELPERS
# ============================================================

def fmt_date(x): return pd.Timestamp(x).strftime("%d %b %Y")
def pct(x):
    if pd.isna(x): return "Unavailable"
    x=float(x); x=x*100 if x<=1 else x
    return f"{x:.1f}%"
def val(x,d=4): return "Unavailable" if pd.isna(x) else f"{float(x):.{d}f}"
def metric(label,value,note):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
def card(title,text,icon=""):
    st.markdown(f'<div class="card"><div class="card-title">{icon} {title}</div><div class="card-text">{text}</div></div>',unsafe_allow_html=True)
def section(k,title,copy=""):
    st.markdown(f'<div class="section"><div class="kicker">{k}</div><div class="section-title">{title}</div><div class="section-copy">{copy}</div></div>',unsafe_allow_html=True)
def nav_button(label,page,icon):
    if st.button(f"{icon} {label}",key=f"nav_{page}",width="stretch",type="primary" if st.session_state.page==page else "secondary"):
        st.session_state.page=page; st.rerun()
def region_name(lat,lon):
    if 5<=lat<=30 and 45<=lon<=75:return "Arabian Sea"
    if 0<=lat<=25 and 75<lon<=100:return "Bay of Bengal"
    if -30<=lat<5 and 40<=lon<=100:return "Southern Indian Ocean"
    if 5<=lat<=30 and 75<lon<=120:return "Northern Indian Ocean"
    return "Other study area"
def nearest_row(frame,lat,lon):
    a=frame
    dist=(a.latitude.to_numpy()-lat)**2+(a.longitude.to_numpy()-lon)**2
    return a.iloc[int(np.argmin(dist))]
def study(frame):
    return frame[frame.latitude.between(LAT_MIN,LAT_MAX)&frame.plot_lon.between(LON_MIN,LON_MAX)].copy()

def zone_table(field,limit=12):
    r=field[field.risk.gt(0)].copy()
    if r.empty:return pd.DataFrame()
    r["lat_zone"]=np.floor(r.latitude/2)*2
    r["lon_zone"]=np.floor(r.plot_lon/2)*2
    z=r.groupby(["lat_zone","lon_zone"],as_index=False).agg(flagged_cells=("risk","size"),mean_chla=("chla","mean"),max_chla=("chla","max"),mean_probability=("probability","mean")).sort_values(["flagged_cells","mean_chla"],ascending=False).head(limit)
    z["Zone"]=z.apply(lambda x:f"{x.lat_zone:.0f}° to {x.lat_zone+2:.0f}° · {x.lon_zone:.0f}° to {x.lon_zone+2:.0f}°E",axis=1)
    z["Mean Chl-a"]=z.mean_chla.round(4); z["Max Chl-a"]=z.max_chla.round(4); z["Mean probability"]=z.mean_probability.map(pct)
    return z[["Zone","flagged_cells","Mean Chl-a","Max Chl-a","Mean probability"]].rename(columns={"flagged_cells":"Flagged cells"})

def risk_map(field,height=620):
    """Stable Scattergeo-only map. No update_geos(projection=...) misuse and no go-name collision."""
    f=study(field)
    if f.empty:f=field.copy()
    base=f.copy()
    base["lat_bin"]=(np.floor(base.latitude)+.5).round(2)
    base["lon_bin"]=(np.floor(base.plot_lon)+.5).round(2)
    base=base.groupby(["lat_bin","lon_bin"],as_index=False).agg(mean_chla=("chla","mean"),cells=("chla","size"))
    fig=go.Figure()
    fig.add_trace(go.Scattergeo(lon=base.lon_bin,lat=base.lat_bin,mode="markers",name="Processed field",marker=dict(size=7,color="#4AA7C0",opacity=.72,line=dict(width=0)),customdata=base[["mean_chla","cells"]].to_numpy(),hovertemplate="<b>Processed field</b><br>Lat: %{lat:.1f}°<br>Lon: %{lon:.1f}°<br>Mean Chl-a: %{customdata[0]:.4f}<br>Cells: %{customdata[1]}<extra></extra>"))
    r=f[f.risk.gt(0)].copy()
    if not r.empty:
        r["lat_zone"]=np.floor(r.latitude/2)*2+1;r["lon_zone"]=np.floor(r.plot_lon/2)*2+1
        z=r.groupby(["lat_zone","lon_zone"],as_index=False).agg(flagged=("risk","size"),mean_chla=("chla","mean"))
        fig.add_trace(go.Scattergeo(lon=z.lon_zone,lat=z.lat_zone,mode="markers",name="Risk concentration",marker=dict(size=np.clip(12+z.flagged*.8,14,34),color="#2CA777",opacity=.45,line=dict(color="#176B4E",width=1)),customdata=z[["flagged","mean_chla"]].to_numpy(),hovertemplate="<b>Risk concentration</b><br>Flagged cells: %{customdata[0]}<br>Mean Chl-a: %{customdata[1]:.4f}<extra></extra>"))
        fig.add_trace(go.Scattergeo(lon=r.plot_lon,lat=r.latitude,mode="markers",name="Potential bloom-risk cell",marker=dict(size=5.5,color="#DF4B58",opacity=.92,line=dict(color="#8E2833",width=.5)),customdata=r[["longitude","chla","probability"]].to_numpy(),hovertemplate="<b>Potential bloom-risk screening</b><br>Lat: %{lat:.3f}°<br>Lon: %{customdata[0]:.3f}°<br>Chl-a: %{customdata[1]:.4f}<br>Risk probability: %{customdata[2]:.1%}<extra></extra>"))
    fig.update_geos(projection_type="mercator",lataxis_range=[LAT_MIN,LAT_MAX],lonaxis_range=[LON_MIN,LON_MAX],showland=True,landcolor="#DCE7E9",showocean=True,oceancolor="#D9F2F4",showcoastlines=True,coastlinecolor="#52757D",coastlinewidth=1,showcountries=True,countrycolor="#91AEB4",bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=height,margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans",color="#14343D"),legend=dict(orientation="h",y=.01,x=.5,xanchor="center",bgcolor="rgba(255,255,255,.94)",bordercolor="#A8D9DD",borderwidth=1,font=dict(size=10)))
    return fig

def make_daily_history(h):
    if h.empty:return pd.DataFrame()
    return h.groupby("date",as_index=False).agg(mean_chla=("chla","mean"),median_chla=("chla","median"),max_chla=("chla","max"),risk_cells=("risk","sum"),processed_cells=("chla","size"))

# ============================================================
# NAVIGATION
# ============================================================

if "page" not in st.session_state:st.session_state.page="Home"
st.markdown(f'<div class="brand"><div class="brand-left"><div class="logo">≈</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Coastal &amp; Ocean Intelligence · EOS-06 OCM-3</div></div></div><div class="brand-date">Latest processed field<br><b>{fmt_date(latest_date)}</b></div></div>',unsafe_allow_html=True)
nav=st.columns(5,gap="small")
for c,(label,page,icon) in zip(nav,[("Home","Home","⌂"),("Risk Map","Risk Map","🌍"),("Location","Location","📍"),("Insights","Insights","📊"),("Data","Data","⇩")]):
    with c:nav_button(label,page,icon)

# ============================================================
# HOME
# ============================================================
if st.session_state.page=="Home":
    st.markdown(f'<div class="hero"><div class="hero-content"><div><span class="pill">🌊 Ocean colour</span><span class="pill">🛰️ EOS-06 OCM-3</span><span class="pill">🤖 AI screening</span><span class="pill">📍 Spatial intelligence</span></div><h1>Read the ocean signal.</h1><p>BloomDetect AI turns satellite-derived chlorophyll-a observations and the project screening output into a practical view of where unusual patterns may deserve closer investigation.</p></div></div>',unsafe_allow_html=True)
    section("LATEST FIELD","One clean snapshot.","The home page gives only the current status. Detailed analysis lives on the dedicated pages, so the same information is not repeated everywhere.")
    m=st.columns(4,gap="medium")
    for c,(a,b,n) in zip(m,[("Processed cells",f"{len(latest):,}","latest field"),("Potential-risk cells",f"{latest_risk:,}","screening output"),("Risk share",f"{latest_share:.2f}%","of processed cells"),("Maximum Chl-a",val(latest.chla.max()),"latest field")]):
        with c:metric(a,b,n)
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    left,right=st.columns([1,1],gap="large")
    with left:card("What BloomDetect adds","It combines the satellite chlorophyll-a signal with the project’s stored screening result, then exposes the result through a map, coordinate lookup, hotspot analysis and temporal insights.","🌊")
    with right:
        if IMAGE_PATH.exists():st.image(str(IMAGE_PATH),width="stretch")
        else:card("EOS-06 ocean-colour view","Place bloomdetect_bloom_process.png beside app.py if you want the project illustration shown here.","🛰️")
    st.markdown('<div style="height:12px"></div>',unsafe_allow_html=True)
    section("GO DIRECTLY TO THE ANALYSIS","Five pages, five jobs.","No launcher page and no duplicate Hotspots/Insights navigation.")
    a,b,c,d,e=st.columns(5,gap="small")
    for c,icon,title,copy in zip([a,b,c,d,e],["🌍","📍","📊","⇩",""],["Risk Map","Location","Insights","Data",""],["Explore the study area through the available dates and spatial layers.","Check a coordinate against the nearest processed satellite cell.","Study hotspots, persistence, trends and current-field distributions.","Keep provenance, project facts and useful downloads in one place.",""]):
        if title: 
            with c:card(title,copy,icon)
    st.markdown('<div class="notice"><b>Scientific boundary:</b> potential bloom-risk screening is not confirmation of a harmful algal bloom. High chlorophyll-a alone does not establish harmfulness, species identity or toxin presence.</div>',unsafe_allow_html=True)

# ============================================================
# RISK MAP
# ============================================================
elif st.session_state.page=="Risk Map":
    section("01 · SPATIAL INTELLIGENCE","See how the field changes.","Use the date control to inspect an available observation. The map keeps the same study window so movement of the signal is easy to compare.")
    if not history.empty and history.date.nunique()>1:
        dates=sorted(history.date.unique())
        selected=st.select_slider("Observation date",options=dates,value=latest_date if latest_date in dates else dates[-1],format_func=fmt_date,key="map_date")
        field=history[history.date.eq(selected)].copy()
    else:
        selected=latest_date;field=latest.copy()
        st.markdown(f'<div class="notice"><b>Timeline status:</b> the website prediction CSV currently contains the latest field only, so the map is showing <b>{fmt_date(selected)}</b>. Add the lightweight <code>bloomdetect_history.csv</code> to activate the full date slider. The app does not invent historical values.</div>',unsafe_allow_html=True)
    sf=study(field);nr=int(sf.risk.sum()) if len(sf) else int(field.risk.sum());n=len(sf) if len(sf) else len(field);share=nr/n*100 if n else 0
    m=st.columns(4,gap="medium")
    for c,(a,b,nn) in zip(m,[("Selected date",fmt_date(selected),"observation"),("Processed cells",f"{n:,}","study area"),("Potential-risk cells",f"{nr:,}","selected field"),("Risk share",f"{share:.2f}%","of study-area cells")]):
        with c:metric(a,b,nn)
    st.markdown('<div style="height:16px"></div>',unsafe_allow_html=True)
    st.markdown('<div class="map-head"><b>Indian Ocean · Arabian Sea · Bay of Bengal</b><span>Blue field · green concentration · red screening flags</span></div>',unsafe_allow_html=True)
    st.plotly_chart(risk_map(field),width="stretch",config={"displaylogo":False,"scrollZoom":False})
    st.markdown('<div class="legend"><span><i class="sw blue"></i><b>Blue</b> processed ocean field</span><span><i class="sw green"></i><b>Green</b> concentration of flagged cells</span><span><i class="sw red"></i><b>Red</b> individual potential-risk cell</span></div>',unsafe_allow_html=True)
    # One useful map switch, not a pile of buttons.
    st.markdown('<div style="height:16px"></div>',unsafe_allow_html=True)
    layer=st.selectbox("Map detail",["Current screening","Chlorophyll-a","Change from previous observation","Historical anomaly"],key="map_layer")
    if layer!="Current screening":
        if layer=="Chlorophyll-a":
            plot=study(field).copy(); plot["lat_bin"]=np.floor(plot.latitude)+.5;plot["lon_bin"]=np.floor(plot.plot_lon)+.5;plot=plot.groupby(["lat_bin","lon_bin"],as_index=False).agg(value=("chla","mean"))
            fig=go.Figure(go.Scattergeo(lon=plot.lon_bin,lat=plot.lat_bin,mode="markers",marker=dict(size=8,color=plot.value,colorscale="YlGnBu",showscale=True,colorbar=dict(title="Chl-a"),opacity=.78),customdata=plot.value,hovertemplate="Lat: %{lat:.1f}°<br>Lon: %{lon:.1f}°<br>Chl-a: %{customdata:.4f}<extra></extra>"));fig.update_geos(projection_type="mercator",lataxis_range=[LAT_MIN,LAT_MAX],lonaxis_range=[LON_MIN,LON_MAX],showland=True,landcolor="#DCE7E9",showocean=True,oceancolor="#D9F2F4",showcoastlines=True,showcountries=True,countrycolor="#91AEB4",bgcolor="rgba(0,0,0,0)");fig.update_layout(height=570,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)");st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"scrollZoom":False})
        else:
            source=field.copy()
            if layer=="Change from previous observation":
                if "chla_change" not in source.columns and {"chla","previous_chla"}.issubset(source.columns):source["chla_change"]=source.chla-source.previous_chla
                col="chla_change"
            else:
                if "chla_anomaly" not in source.columns and {"chla","historical_baseline"}.issubset(source.columns):source["chla_anomaly"]=source.chla-source.historical_baseline
                col="chla_anomaly"
            if col in source.columns and source[col].notna().any():
                p=study(source).copy();p["lat_bin"]=np.floor(p.latitude)+.5;p["lon_bin"]=np.floor(p.plot_lon)+.5;p=p.groupby(["lat_bin","lon_bin"],as_index=False).agg(value=(col,"mean"));mx=max(abs(p.value).max(),1e-9);fig=go.Figure(go.Scattergeo(lon=p.lon_bin,lat=p.lat_bin,mode="markers",marker=dict(size=8,color=p.value,colorscale="RdBu",cmid=0,cmin=-mx,cmax=mx,showscale=True,colorbar=dict(title="Signal"),opacity=.78),customdata=p.value,hovertemplate="Lat: %{lat:.1f}°<br>Lon: %{lon:.1f}°<br>Signal: %{customdata:.4f}<extra></extra>"));fig.update_geos(projection_type="mercator",lataxis_range=[LAT_MIN,LAT_MAX],lonaxis_range=[LON_MIN,LON_MAX],showland=True,landcolor="#DCE7E9",showocean=True,oceancolor="#D9F2F4",showcoastlines=True,showcountries=True,countrycolor="#91AEB4",bgcolor="rgba(0,0,0,0)");fig.update_layout(height=570,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)");st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"scrollZoom":False})
            else:st.info(f"{layer} needs the corresponding derived column in the selected data.")
    z=zone_table(field,12)
    if not z.empty:
        st.markdown('<div style="height:10px"></div>',unsafe_allow_html=True);section("SPATIAL CONCENTRATIONS","Investigation zones","These zones summarize where flags cluster. They are not a second risk label.")
        left,right=st.columns([1.1,.9],gap="large")
        with left:
            bar=go.Figure(go.Bar(x=z["Flagged cells"],y=z["Zone"],orientation="h",marker_color="#2CA777",text=z["Flagged cells"],textposition="outside",cliponaxis=False));bar.update_layout(height=390,margin=dict(l=70,r=40,t=15,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.60)",font=dict(color="#143F49"),xaxis=dict(title="Flagged cells",showgrid=True,gridcolor="#D7E8EA"),yaxis=dict(title="Investigation zone"),showlegend=False);st.plotly_chart(bar,width="stretch",config={"displaylogo":False})
        with right:st.dataframe(z,width="stretch",hide_index=True,height=390)

# ============================================================
# LOCATION
# ============================================================
elif st.session_state.page=="Location":
    section("02 · LOCATION INTELLIGENCE","Check one coordinate across the available evidence.","The input coordinate and the nearest processed satellite cell are always shown separately. If the historical pack is available, the same location can be explored through time.")
    left,right=st.columns([.82,1.18],gap="large")
    with left:
        card("Your input","Use decimal degrees. Example: 17.3850, 78.4867.","📍")
        lat=st.number_input("Latitude",min_value=-40.0,max_value=30.0,value=float(st.session_state.get("lat",17.385)),step=.01,format="%.4f",key="loc_lat")
        lon=st.number_input("Longitude",min_value=20.0,max_value=120.0,value=float(st.session_state.get("lon",78.4867)),step=.01,format="%.4f",key="loc_lon")
        if st.button("🔎 Check location",width="stretch",type="primary",key="loc_check"):
            st.session_state.lat=float(lat);st.session_state.lon=float(lon);st.session_state.loc_checked=True
    if "loc_checked" not in st.session_state:st.session_state.loc_checked=False
    if st.session_state.loc_checked:
        lat=float(st.session_state.lat);lon=float(st.session_state.lon);row=nearest_row(latest,lat,lon)
        with right:
            flag=int(row.risk)==1
            st.markdown(f'<div class="status {"bad" if flag else "ok"}"><div class="status-title">{"🔴 POTENTIAL BLOOM-RISK FLAG" if flag else "🟢 NOT FLAGGED"}</div><div class="status-copy">{"The nearest processed cell is included in the current screening shortlist." if flag else "The nearest processed cell is not included in the current screening shortlist."}</div></div>',unsafe_allow_html=True)
            a,b=st.columns(2)
            with a:metric("Your input",f"{lat:.4f}° · {lon:.4f}°","query coordinate")
            with b:metric("Nearest processed cell",f"{row.latitude:.4f}° · {row.longitude:.4f}°","actual satellite grid cell")
            st.markdown('<div style="height:10px"></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="card"><div class="card-title">Cell result</div><div class="result-grid"><div class="result-item"><span>Observation date</span><b>{fmt_date(row.date)}</b></div><div class="result-item"><span>Region</span><b>{region_name(row.latitude,row.longitude)}</b></div><div class="result-item"><span>Chlorophyll-a</span><b>{val(row.chla)}</b></div><div class="result-item"><span>Risk probability</span><b>{pct(row.probability)}</b></div></div></div>',unsafe_allow_html=True)
        st.markdown('<div style="height:14px"></div>',unsafe_allow_html=True)
        sg=[("Current Chl-a",val(row.chla)),("Historical baseline",val(row.get("historical_baseline",np.nan))), ("Anomaly",val(row.get("chla_anomaly",np.nan))), ("Recent change",val(row.get("chla_change",np.nan)))]
        for c,(a,b) in zip(st.columns(4),sg):
            with c:st.markdown(f'<div class="signal"><div class="signal-label">{a}</div><div class="signal-value">{b}</div></div>',unsafe_allow_html=True)
        # Historical location chart only when a lightweight history pack is supplied.
        if not location_history.empty:
            h=location_history.copy();dist=(h.latitude.to_numpy()-lat)**2+(h.longitude.to_numpy()-lon)**2;near_lat=float(h.iloc[int(np.argmin(dist))].latitude);near_lon=float(h.iloc[int(np.argmin(dist))].longitude);loc=h[(h.latitude==near_lat)&(h.longitude==near_lon)].sort_values("date").copy()
            if len(loc)>1:
                st.markdown('<div style="height:14px"></div>',unsafe_allow_html=True);section("LOCATION HISTORY","How has this cell changed?","The chart uses the nearest processed grid cell across the dates stored in the lightweight history file.")
                fig=go.Figure();fig.add_trace(go.Scatter(x=loc.date,y=loc.chla,mode="lines+markers",name="Chlorophyll-a",line=dict(color="#10A9B5",width=2.5),marker=dict(size=5)));fig.add_trace(go.Scatter(x=loc.date,y=loc.risk*loc.chla.max(),mode="markers",name="Potential-risk flag",marker=dict(color="#DF4B58",size=6),hovertemplate="Flagged observation<extra></extra>"));fig.update_layout(height=370,margin=dict(l=70,r=25,t=20,b=65),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#143F49"),xaxis=dict(title="Observation date"),yaxis=dict(title="Chlorophyll-a"),legend=dict(orientation="h",y=1.08,x=0));st.plotly_chart(fig,width="stretch",config={"displaylogo":False})
            else:st.info("Only one observation is available for this grid cell in the history pack.")
        else:
            st.markdown('<div class="notice"><b>Location history:</b> the current website CSV contains only the latest field. The location history chart activates when <code>bloomdetect_history.csv</code> is added.</div>',unsafe_allow_html=True)
        report=pd.DataFrame([{"input_latitude":lat,"input_longitude":lon,"nearest_processed_latitude":row.latitude,"nearest_processed_longitude":row.longitude,"date":row.date.strftime("%Y-%m-%d"),"chla":row.chla,"risk_probability":row.probability,"risk_label":row.risk_label}])
        st.download_button("⬇ Download location report",report.to_csv(index=False).encode(),"bloomdetect_location_report.csv","text/csv",width="stretch")
    else:
        with right:card("Result","Enter a coordinate and press Check location. The result will appear here without exposing HTML or internal code.","🔎")

# ============================================================
# INSIGHTS + HOTSPOTS
# ============================================================
elif st.session_state.page=="Insights":
    section("03 · INSIGHTS","Hotspots, persistence and trends in one place.","This page contains the analysis that used to be split between Hotspots and Insights. No duplicated navigation, no repeated charts.")
    # Choose current or historical date for hotspot analysis when history exists.
    if not history.empty and history.date.nunique()>1:
        dates=sorted(history.date.unique());selected=st.select_slider("Analysis date",options=dates,value=latest_date if latest_date in dates else dates[-1],format_func=fmt_date,key="insight_date");field=history[history.date.eq(selected)].copy()
    else:selected=latest_date;field=latest.copy()
    sf=study(field);field=sf if not sf.empty else field;risk=field[field.risk.gt(0)];normal=field[field.risk.eq(0)]
    m=st.columns(4,gap="medium")
    for c,(a,b,n) in zip(m,[("Observation cells",f"{len(field):,}","selected field"),("Potential-risk",f"{len(risk):,}","selected field"),("Mean Chl-a",val(field.chla.mean()),"selected field"),("Risk share",f"{(len(risk)/len(field)*100 if len(field) else 0):.2f}%","selected field")]):
        with c:metric(a,b,n)
    z=zone_table(field,12)
    if not z.empty:
        section("HOTSPOTS","Where are the flags concentrating?","The top zones summarize current spatial concentration. Counts are screening flags, not severity scores.")
        left,right=st.columns([1.1,.9],gap="large")
        with left:
            bar=go.Figure(go.Bar(x=z["Flagged cells"],y=z["Zone"],orientation="h",marker_color="#2CA777",text=z["Flagged cells"],textposition="outside",cliponaxis=False));bar.update_layout(height=390,margin=dict(l=75,r=40,t=15,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#143F49"),xaxis=dict(title="Flagged cells",showgrid=True,gridcolor="#D7E8EA"),yaxis=dict(title="Investigation zone"),showlegend=False);st.plotly_chart(bar,width="stretch",config={"displaylogo":False})
        with right:st.dataframe(z,width="stretch",hide_index=True,height=390)
    if not history.empty and history.date.nunique()>1:
        daily=make_daily_history(history)
        section("TEMPORAL SIGNAL","How has the screening changed?","Daily counts and mean chlorophyll-a show whether the spatial screening pattern is persistent, emerging or declining across the available observations.")
        a,b=st.columns(2,gap="large")
        with a:
            fig=go.Figure(go.Scatter(x=daily.date,y=daily.risk_cells,mode="lines+markers",line=dict(color="#DF4B58",width=2.4),marker=dict(size=4)));fig.update_layout(height=350,margin=dict(l=70,r=20,t=15,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#143F49"),xaxis=dict(title="Observation date"),yaxis=dict(title="Potential-risk cells"),showlegend=False);st.plotly_chart(fig,width="stretch",config={"displaylogo":False})
        with b:
            fig=go.Figure(go.Scatter(x=daily.date,y=daily.mean_chla,mode="lines+markers",line=dict(color="#10A9B5",width=2.4),marker=dict(size=4)));fig.update_layout(height=350,margin=dict(l=70,r=20,t=15,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#143F49"),xaxis=dict(title="Observation date"),yaxis=dict(title="Mean chlorophyll-a"),showlegend=False);st.plotly_chart(fig,width="stretch",config={"displaylogo":False})
        # Peak dates
        peaks=daily.sort_values("risk_cells",ascending=False).head(5)[["date","risk_cells","mean_chla"]].copy();peaks["date"]=peaks.date.map(fmt_date);peaks.columns=["Date","Potential-risk cells","Mean Chl-a"]
        st.markdown('<div style="height:5px"></div>',unsafe_allow_html=True);section("PEAK ACTIVITY","Dates with the highest screening counts.","")
        st.dataframe(peaks,width="stretch",hide_index=True)
        # Persistence classification for current top zones using recent history.
        recent_dates=sorted(history.date.unique())[-30:]
        rh=history[history.date.isin(recent_dates)].copy();rh["lat_zone"]=np.floor(rh.latitude/2)*2;rh["lon_zone"]=np.floor(rh.plot_lon/2)*2
        pers=rh.groupby(["lat_zone","lon_zone"],as_index=False).agg(flagged_observations=("risk","sum"),observations=("risk","size"));pers=pers[pers.flagged_observations>0].copy();pers["persistence"]=pers.flagged_observations/pers.observations;pers=pers.sort_values("persistence",ascending=False).head(12);pers["Status"]=np.select([pers.persistence>=.5,pers.persistence>=.2],["Persistent","Recurring"],default="Recent");pers["Zone"]=pers.apply(lambda x:f"{x.lat_zone:.0f}° to {x.lat_zone+2:.0f}° · {x.lon_zone:.0f}° to {x.lon_zone+2:.0f}°E",axis=1);pers["Flagged observations"]=pers.flagged_observations.astype(int);pers["Persistence"]=pers.persistence.map(lambda x:f"{x*100:.0f}%")
        section("PERSISTENCE","Which zones keep appearing?","A simple recurrence measure over the latest 30 available observations. It describes persistence of the stored screening flag, not bloom severity.")
        st.dataframe(pers[["Zone","Status","Flagged observations","Persistence"]],width="stretch",hide_index=True)
    # Current field charts are always available.
    section("CURRENT FIELD","What stands out right now?","These charts use the selected field and do not repeat the map.")
    a,b=st.columns(2,gap="large")
    with a:
        h=go.Figure(go.Histogram(x=field.chla,nbinsx=40,marker_color="#4AA7C0"));h.update_layout(height=360,margin=dict(l=70,r=20,t=15,b=65),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#143F49"),xaxis=dict(title="Chlorophyll-a"),yaxis=dict(title="Processed cells"),showlegend=False);st.plotly_chart(h,width="stretch",config={"displaylogo":False})
    with b:
        groups=["Normal","Potential-risk"];means=[normal.chla.mean() if len(normal) else 0,risk.chla.mean() if len(risk) else 0];c=go.Figure(go.Bar(x=groups,y=means,marker_color=["#4AA7C0","#DF4B58"],text=[f"{x:.4f}" for x in means],textposition="outside",cliponaxis=False));c.update_layout(height=360,margin=dict(l=70,r=25,t=20,b=65),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#143F49"),xaxis=dict(title="Screening group"),yaxis=dict(title="Mean chlorophyll-a"),showlegend=False);st.plotly_chart(c,width="stretch",config={"displaylogo":False})
    st.markdown('<div class="notice"><b>Reading the charts:</b> they describe satellite-derived chlorophyll-a and the project screening output. They do not independently establish harmfulness, species identity or toxin presence.</div>',unsafe_allow_html=True)

# ============================================================
# DATA
# ============================================================
elif st.session_state.page=="Data":
    section("04 · DATA & OUTPUTS","The evidence behind the website.","Source facts, coverage and downloads stay here. The analysis pages do not repeat the full dataset documentation.")
    m=st.columns(4,gap="medium")
    for c,(a,b,n) in zip(m,[("Source product","E06OCM_L4_AC","EOS-06 / OCM-3"),("Variable","Chlorophyll-a","satellite-derived"),("Grid","0.25°","latitude × longitude"),("Latest field",fmt_date(latest_date),f"{len(latest):,} cells")]):
        with c:metric(a,b,n)
    section("COVERAGE","What is actually in the current website dataset.","")
    a,b=st.columns(2,gap="large")
    with a:card("Current website CSV",f"{len(df):,} packaged rows · {df.date.nunique():,} observation date(s) · {fmt_date(df.date.min())} to {fmt_date(df.date.max())}.","🛰️")
    with b:card("Study window","20°E–120°E and 40°S–30°N, focused on the Indian Ocean, Arabian Sea and Bay of Bengal interpretation region.","🌍")
    section("INTERPRETATION","Use the screening result correctly.","")
    st.markdown('<div class="notice"><b>Potential bloom risk</b> is a project screening output based on the available satellite-derived signal and model pipeline. It is not confirmed HAB ground truth. High chlorophyll-a alone does not prove a harmful algal bloom, and the current satellite data do not identify species or toxins.</div>',unsafe_allow_html=True)
    section("DOWNLOADS","Useful project outputs.","Only the downloads that are actually useful are kept here.")
    csv_bytes=df.to_csv(index=False).encode()
    summary=(f"BloomDetect AI - Project Summary\n\nSource: EOS-06 / Oceansat-3 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC)\nVariable: Chlorophyll-a\nGrid: 0.25 degrees\nCoverage in website CSV: {fmt_date(df.date.min())} to {fmt_date(df.date.max())}\nPackaged rows: {len(df):,}\nLatest field: {fmt_date(latest_date)}\nLatest processed cells: {len(latest):,}\nLatest potential-risk cells: {latest_risk:,}\nLatest risk share: {latest_share:.2f}%\nStudy window: 20E-120E, 40S-30N\n\nScientific boundary: potential bloom-risk screening only; high chlorophyll-a alone does not confirm a harmful algal bloom.\n").encode()
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown('<div class="download-card"><h3>⬇ Latest screening CSV</h3><p>The packaged observations and stored screening output used by the website.</p></div>',unsafe_allow_html=True);st.download_button("Download latest dataset",csv_bytes,"bloomdetect_latest_screening.csv","text/csv",width="stretch",key="dl_latest")
    with b:
        st.markdown('<div class="download-card"><h3>⬇ Project summary</h3><p>Compact source, coverage and interpretation notes for reports and reviews.</p></div>',unsafe_allow_html=True);st.download_button("Download project summary",summary,"BloomDetect_AI_Project_Summary.txt","text/plain",width="stretch",key="dl_summary")
    if not history.empty:
        st.markdown('<div style="height:10px"></div>',unsafe_allow_html=True);card("Timeline pack detected",f"{len(history):,} lightweight map rows across {history.date.nunique():,} dates. Risk Map and Insights timeline features are active.","📅")
    else:
        st.markdown('<div style="height:10px"></div>',unsafe_allow_html=True);card("Timeline pack not detected","The current website CSV contains the latest field only. Add bloomdetect_history.csv for the map timeline and bloomdetect_location_history.csv for full location history. The app does not fabricate historical observations.","📅")

# ============================================================
# FOOTER
# ============================================================
st.markdown(f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Latest processed field: {fmt_date(latest_date)}</div>',unsafe_allow_html=True)
