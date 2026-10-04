from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI · FINAL POLISHED DASHBOARD
# Home | Risk Map | Location | Insights | Data
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
PRED_PATH = BASE / "latest_bloom_risk_predictions.csv"
HISTORY_PATH = BASE / "bloomdetect_history_web.csv.gz"
IMAGE_PATH = BASE / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ---------------- DATA ----------------
@st.cache_data(show_spinner="Loading project data...")
def load_predictions():
    if not PRED_PATH.exists():
        raise FileNotFoundError("latest_bloom_risk_predictions.csv must be beside app.py")
    d = pd.read_csv(PRED_PATH)
    required = {"latitude", "longitude", "date", "chla"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    # Accept both the deployed schema (risk_label) and the compact schema (risk).
    if "risk_label" not in d.columns:
        if "risk" in d.columns:
            d["risk"] = pd.to_numeric(d["risk"], errors="coerce").fillna(0)
            d["risk_label"] = np.where(d["risk"].astype(int).eq(1), "Potential Bloom Risk", "Normal")
        else:
            raise ValueError("Prediction CSV needs either risk_label or risk.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["latitude", "longitude", "chla"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in [
        "risk_probability", "model_score", "previous_chla",
        "historical_baseline", "recent_mean", "recent_max",
        "chla_anomaly", "chla_change",
    ]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
    d["risk_flag"] = d["risk_label"].astype(str).str.strip().str.lower().eq("potential bloom risk")
    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
    return d


@st.cache_data(show_spinner="Loading timeline...")
def load_history():
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    h = pd.read_csv(HISTORY_PATH)
    required = {"date", "lat_bin", "lon_bin", "chla", "risk", "cells", "max_chla"}
    if not required.issubset(h.columns):
        return pd.DataFrame()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    for c in ["lat_bin", "lon_bin", "chla", "risk", "cells", "max_chla"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h.dropna(subset=["date", "lat_bin", "lon_bin", "chla"]).copy()
    return h


try:
    df = load_predictions()
except Exception as e:
    st.error("BloomDetect AI could not load the prediction dataset.")
    st.code(str(e))
    st.stop()

history = load_history()
latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
study_latest = latest[latest["latitude"].between(LAT_MIN, LAT_MAX) & latest["plot_lon"].between(LON_MIN, LON_MAX)].copy()
risk_latest = study_latest[study_latest["risk_flag"]].copy()

# ---------------- STYLE ----------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#103f49;--muted:#66848b;--line:#a9d6da;--blue:#277fa4;--green:#22a879;--red:#ed5260;--shadow:0 12px 32px rgba(9,80,94,.08)}
html,body,[data-testid="stAppViewContainer"]{background:#f2fbfb!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif!important}
.stApp{background:linear-gradient(180deg,#fbffff 0%,#effafa 52%,#fbffff 100%)!important}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}
.block-container{max-width:1240px!important;padding:22px 30px 65px!important}
.brand{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border:1.5px solid #c3e2e5;border-radius:21px;background:rgba(255,255,255,.94);box-shadow:var(--shadow)}
.brand-left{display:flex;align-items:center;gap:12px}.logo{width:47px;height:47px;border-radius:15px;background:linear-gradient(145deg,#28cbd0,#08798b);display:grid;place-items:center;color:#fff;font-weight:800;font-size:20px}.brand-name{font:800 1.2rem Manrope;color:var(--ink)}.brand-sub{font-size:.71rem;color:#78959b}.latest{text-align:right;font-size:.67rem;color:#78959b}.latest b{color:#174b56;font-size:.78rem}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:44px!important;border-radius:14px!important;background:#fff!important;border:2px solid #164e5b!important;color:#123f49!important;font-weight:800!important;box-shadow:0 6px 15px rgba(15,70,82,.08)!important}.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:#e2f8f8!important;border-color:#087d8c!important}.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:#d8f7f7!important;border-color:#078c9b!important}
.section{padding:30px 0 16px}.kicker{font:800 .66rem Manrope;letter-spacing:.18em;text-transform:uppercase;color:#0797a5;margin-bottom:9px}.section h2{font:800 clamp(2rem,4vw,3.5rem)/1.04 Manrope;letter-spacing:-.06em;margin:0 0 11px;color:var(--ink)}.section p{font-size:.94rem;line-height:1.62;color:var(--muted);max-width:1080px;margin:0}
.hero{position:relative;overflow:hidden;min-height:380px;border-radius:29px;padding:52px;background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);box-shadow:0 24px 62px rgba(6,86,100,.15)}.hero:before{content:"";position:absolute;inset:-20%;background:repeating-radial-gradient(ellipse at 20% 115%,transparent 0 55px,rgba(181,255,251,.12) 57px 59px,transparent 61px 105px);transform:rotate(-7deg)}.hero-content{position:relative;z-index:2;max-width:800px}.hero .kicker{color:#a6fffa}.hero h1{font:800 clamp(3rem,6vw,5.4rem)/.92 Manrope;letter-spacing:-.075em;color:#e4ffff;margin:0 0 18px}.hero p{font-size:1.02rem;line-height:1.75;color:#e0fbfb}.badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:20px}.badge{padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.74rem;font-weight:700}
.card,.chart-card,.signal,.metric{background:rgba(255,255,255,.92);border:1.5px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}.card{padding:21px}.card h3{font:800 1.2rem Manrope;margin:0 0 7px}.card p{color:var(--muted);line-height:1.6;font-size:.9rem;margin:0}.metric{padding:16px;min-height:105px}.metric-label{font:800 .62rem Manrope;letter-spacing:.12em;text-transform:uppercase;color:#6d8c93}.metric-value{font:800 1.4rem Manrope;color:var(--ink);margin-top:5px}.metric-note{font-size:.7rem;color:#78959b;margin-top:4px}
.note{margin-top:14px;padding:12px 15px;background:#e6f8f8;border-left:4px solid #11a9b2;border-radius:0 13px 13px 0;color:#52757c;font-size:.82rem;line-height:1.55}.tool-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}.tool{padding:19px;min-height:145px;border:1.5px solid var(--line);border-radius:19px;background:#fff;box-shadow:var(--shadow)}.tool h3{font:800 1rem Manrope;margin:8px 0 6px}.tool p{font-size:.83rem;line-height:1.5;color:var(--muted);margin:0}
.map-shell{border:2px solid #8fcbd1;border-radius:22px;background:#dff7f8;padding:5px;overflow:hidden;box-shadow:var(--shadow)}.map-head{display:flex;justify-content:space-between;padding:10px 12px;color:#174b56;font-size:.82rem}.map-head span{color:#6b8b92;font-size:.68rem}.legend{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:9px 12px;color:#4f747b;font-size:.76rem}.dot{width:11px;height:11px;border-radius:50%;display:inline-block}.blue{background:var(--blue)}.green{background:var(--green)}.red{background:var(--red)}
.result-grid,.signal-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:14px}.signal-grid{grid-template-columns:repeat(4,1fr)}.result{padding:11px 12px;border-radius:12px;background:#f6fcfc;border:1px solid #c8e5e7}.result span,.signal span{display:block;font-size:.65rem;color:#78959b;margin-bottom:4px}.result b,.signal b{font-size:.88rem;color:#194b55}.signal{padding:13px}.status{margin-top:14px;padding:14px;border-radius:14px;border:2px solid;font-size:.85rem}.status b{display:block;font:800 .9rem Manrope;margin-bottom:4px}.status.risk{background:#fff0f2;border-color:#ed6976;color:#9b2d3c}.status.normal{background:#eafaf4;border-color:#49b995;color:#176f58}
.image-wrap{border:1.5px solid var(--line);padding:8px;border-radius:21px;background:#fff;box-shadow:var(--shadow)}
.footer{border-top:1px solid #d5ebed;margin-top:38px;padding-top:15px;color:#76959b;font-size:.68rem}
@media(max-width:900px){.tool-grid{grid-template-columns:1fr 1fr}.signal-grid{grid-template-columns:1fr 1fr}.block-container{padding:18px!important}}
@media(max-width:620px){.tool-grid,.signal-grid,.result-grid{grid-template-columns:1fr}.hero{padding:38px 24px}.latest{display:none}}
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def metric(label, value, note):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def section(k, title, copy):
    st.markdown(f'<div class="section"><div class="kicker">{k}</div><h2>{title}</h2><p>{copy}</p></div>', unsafe_allow_html=True)

def fmt(value, digits=4):
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):.{digits}f}"

def pct(value):
    if value is None or pd.isna(value):
        return "Unavailable"
    x=float(value); x=x*100 if x<=1 else x
    return f"{x:.1f}%"

def region_name(lat, lon):
    lat=float(lat); lon=float(lon)
    if 5<=lat<=30 and 45<=lon<=75: return "Arabian Sea"
    if 0<=lat<=25 and 75<lon<=100: return "Bay of Bengal"
    if -30<=lat<5 and 40<=lon<=100: return "Southern Indian Ocean"
    if 5<=lat<=30 and 75<lon<=120: return "Northern Indian Ocean"
    return "Indian Ocean study area"

def valid_ocean_rows(frame):
    # Zero/non-positive Chl-a values are not useful for a coordinate lookup.
    return frame[frame["chla"].gt(0) & frame["chla"].notna()].copy()

def nearest_valid(lat, lon):
    a=valid_ocean_rows(study_latest)
    if a.empty: return None
    scale=max(np.cos(np.deg2rad(float(lat))),0.25)
    dist=((a["latitude"].to_numpy()-lat)**2 + ((a["longitude"].to_numpy()-lon)*scale)**2)
    return a.iloc[int(np.argmin(dist))]

def history_for_cell(lat, lon):
    if history.empty: return pd.DataFrame()
    # The timeline is intentionally stored on a compact 1° grid.
    lat_bin=np.floor(float(lat))+0.5
    lon_bin=np.floor(float(lon))+0.5
    h=history[(history.lat_bin==lat_bin)&(history.lon_bin==lon_bin)].copy()
    return h.sort_values("date")

def supporting_values(row):
    """Use stored 0.25° model features when present; otherwise use the compact 1° history context."""
    vals={
        "Current Chl-a": row.get("chla", np.nan),
        "Historical baseline": row.get("historical_baseline", np.nan),
        "Anomaly": row.get("chla_anomaly", np.nan),
        "Recent change": row.get("chla_change", np.nan),
    }
    if all(pd.isna(v) for v in vals.values() if v is not None):
        h=history_for_cell(row.latitude,row.longitude)
        if not h.empty:
            latest_h=h[h.date.eq(pd.Timestamp(row.date))]
            if latest_h.empty: latest_h=h.tail(1)
            current=float(latest_h.iloc[0].chla)
            before=h[h.date.lt(pd.Timestamp(row.date))].copy()
            baseline=float(before.chla.mean()) if not before.empty else np.nan
            recent=float(before.tail(1).chla.iloc[0]) if not before.empty else np.nan
            vals={"Current Chl-a":float(row.chla),"Historical baseline":baseline,"Anomaly":current-baseline if pd.notna(baseline) else np.nan,"Recent change":current-recent if pd.notna(recent) else np.nan}
    return vals

def map_figure(data, risk_data, title_suffix=""):
    fig=go.Figure()
    if data.empty: return fig
    blue=data.copy()
    if "lat_bin" not in blue:
        blue["lat_bin"]=np.floor(blue["latitude"])+0.5
        blue["lon_bin"]=np.floor(blue["plot_lon"])+0.5
        blue=blue.groupby(["lat_bin","lon_bin"],as_index=False).agg(mean_chla=("chla","mean"),cells=("chla","size"))
    fig.add_trace(go.Scattergeo(
        lat=blue["lat_bin"],lon=blue["lon_bin"],mode="markers",name="Processed ocean field",
        marker=dict(size=7,color="#277fa4",opacity=.68),
        customdata=np.c_[blue["mean_chla"],blue["cells"]],
        hovertemplate="<b>Processed ocean field</b><br>Mean Chl-a: %{customdata[0]:.4f}<br>Cells: %{customdata[1]}<extra></extra>"))
    if not risk_data.empty:
        r=risk_data.copy()
        if "latitude" not in r: r=r.rename(columns={"lat_bin":"latitude","lon_bin":"plot_lon","chla":"chla"})
        r["zone_lat"]=np.floor(r["latitude"]/2)*2+1
        r["zone_lon"]=np.floor(r["plot_lon"]/2)*2+1
        z=r.groupby(["zone_lat","zone_lon"],as_index=False).size().rename(columns={"size":"flags"})
        fig.add_trace(go.Scattergeo(lat=z.zone_lat,lon=z.zone_lon,mode="markers",name="Flag concentration zone",
            marker=dict(size=np.clip(z.flags*1.4+10,11,38),color="#22a879",opacity=.45,line=dict(color="#fff",width=1.3)),
            text=z.flags,hovertemplate="<b>Flag concentration zone</b><br>Flagged cells: %{text}<extra></extra>"))
        fig.add_trace(go.Scattergeo(lat=r.latitude,lon=r.plot_lon,mode="markers",name="Potential bloom-risk cell",
            marker=dict(size=6,color="#ed5260",opacity=.92,line=dict(color="#fff",width=.5)),
            text=[fmt(x) for x in r.chla],hovertemplate="<b>Potential bloom-risk screening</b><br>Chl-a: %{text}<extra></extra>"))
    fig.update_layout(
        height=610,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="#dff7f8",showlegend=True,
        legend=dict(orientation="h",x=.02,y=.01,bgcolor="rgba(255,255,255,.88)",font=dict(size=10)),
        geo=dict(showland=True,landcolor="#d7e5e3",showocean=True,oceancolor="#dff7f8",showlakes=True,lakecolor="#d5f1f3",
                 showcountries=True,countrycolor="#8db4ba",showcoastlines=True,coastlinecolor="#739da4",coastlinewidth=.8,
                 projection_type="equirectangular",lataxis=dict(range=[LAT_MIN,LAT_MAX],showgrid=True,gridcolor="rgba(90,140,150,.18)"),
                 lonaxis=dict(range=[LON_MIN,LON_MAX],showgrid=True,gridcolor="rgba(90,140,150,.18)"),bgcolor="#dff7f8"))
    return fig

# ---------------- HEADER ----------------
latest_display=latest_date.strftime("%d %b %Y")
st.markdown(f'<div class="brand"><div class="brand-left"><div class="logo">≈</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div></div></div><div class="latest">Latest processed field<br><b>{latest_display}</b></div></div>',unsafe_allow_html=True)

PAGES=["home","map","location","insights","data"]
NAV={"home":"⌂ Home","map":"🗺 Risk Map","location":"📍 Location","insights":"📊 Insights","data":"⇩ Data"}
if st.session_state.get("page") not in PAGES: st.session_state.page="home"
cols=st.columns(5,gap="small")
for c,key in zip(cols,PAGES):
    with c:
        if st.button(NAV[key],key=f"nav_{key}",type="primary" if st.session_state.page==key else "secondary",use_container_width=True):
            st.session_state.page=key; st.rerun()

# ---------------- HOME ----------------
if st.session_state.page=="home":
    st.markdown('<div class="hero"><div class="hero-content"><div class="kicker">EOS-06 · OCM-3 · SATELLITE INTELLIGENCE</div><h1>Read the ocean signal.</h1><p>BloomDetect AI turns satellite-derived chlorophyll-a observations and project screening output into a practical early-warning support view for closer investigation.</p><div class="badges"><span class="badge">🌊 Ocean colour</span><span class="badge">🛰 EOS-06 OCM-3</span><span class="badge">🌱 Potential bloom-risk screening</span><span class="badge">📍 Spatial intelligence</span></div></div></div>',unsafe_allow_html=True)
    n=len(study_latest); nr=len(risk_latest); share=100*nr/n if n else 0; mx=study_latest.chla.max() if n else np.nan
    section("PROJECT SNAPSHOT","One clear view of the latest field.","The home page gives the project context and the current field status. Detailed investigation stays inside Risk Map, Location, Insights and Data.")
    m=st.columns(4,gap="medium")
    for c,(a,b,d) in zip(m,[("Processed cells",f"{n:,}","latest field"),("Potential-risk cells",f"{nr:,}","screening output"),("Risk share",f"{share:.2f}%","of processed cells"),("Maximum Chl-a",fmt(mx),"latest field")]):
        with c: metric(a,b,d)
    st.markdown('<div style="height:18px"></div>',unsafe_allow_html=True)
    a,b=st.columns([1,1],gap="large")
    with a:
        st.markdown('<div class="card"><div class="kicker">SCIENTIFIC CONTEXT</div><h3>Chlorophyll-a is a signal, not a verdict.</h3><p>The platform screens locations using satellite-derived chlorophyll-a and temporal context. A potential-risk flag is an investigation aid, not confirmation of a harmful algal bloom, species or toxin.</p><div class="note"><b>Important:</b> high chlorophyll-a alone does not prove a harmful algal bloom. Field observations and additional environmental evidence are required for confirmation.</div></div>',unsafe_allow_html=True)
    with b:
        if IMAGE_PATH.exists():
            st.markdown('<div class="image-wrap">',unsafe_allow_html=True); st.image(str(IMAGE_PATH),use_container_width=True); st.markdown('</div>',unsafe_allow_html=True)
    section("FOUR FOCUSED VIEWS","Everything has one job.","No duplicate Explore page, no separate Hotspots page, and no repeated scientific explanation across every screen.")
    tools=[("🗺️","Risk Map","Move through available dates and inspect spatial screening."),("📍","Location","Check one coordinate against the nearest valid ocean observation."),("📊","Insights","Study hotspot concentration and current Chl-a patterns."),("⇩","Data","Review source information and download project outputs.")]
    tc=st.columns(4,gap="medium")
    for c,(ico,t,desc) in zip(tc,tools):
        with c: st.markdown(f'<div class="tool"><div style="font-size:1.3rem">{ico}</div><h3>{t}</h3><p>{desc}</p></div>',unsafe_allow_html=True)

# ---------------- MAP ----------------
elif st.session_state.page=="map":
    section("01 · SPATIAL INTELLIGENCE","See how the field changes.","Use the date slider to compare available processed observations. The map keeps one Indian Ocean study window so spatial changes remain easy to compare.")
    if not history.empty:
        dates=sorted(history.date.dropna().unique())
        selected=st.slider("Map timeline",min_value=dates[0].date(),max_value=dates[-1].date(),value=dates[-1].date(),format="DD MMM YYYY")
        selected=pd.Timestamp(selected)
    else:
        selected=latest_date
        st.info("Timeline file is not available. The map is showing the latest processed field.")
    # Latest date uses the actual model output. Earlier dates use the compact historical screening layer.
    if selected==latest_date:
        selected_full=study_latest.copy(); selected_risk=risk_latest.copy(); processed=len(selected_full); risk_count=len(selected_risk); mode="model screening output"
        blue=selected_full.copy(); blue["lat_bin"]=np.floor(blue.latitude)+.5; blue["lon_bin"]=np.floor(blue.plot_lon)+.5
        blue=blue.groupby(["lat_bin","lon_bin"],as_index=False).agg(mean_chla=("chla","mean"),cells=("chla","size"))
    else:
        h=history[history.date.eq(selected)].copy(); h=h[h.lat_bin.between(LAT_MIN,LAT_MAX)&h.lon_bin.between(LON_MIN,LON_MAX)]
        blue=h.rename(columns={"lat_bin":"lat_bin","lon_bin":"lon_bin","chla":"mean_chla"}); blue["cells"]=blue["cells"].fillna(0); selected_risk=blue[blue.risk.gt(0)].rename(columns={"lat_bin":"latitude","lon_bin":"plot_lon","mean_chla":"chla"}); processed=len(blue); risk_count=len(selected_risk); mode="historical screening layer"
    share=100*risk_count/processed if processed else 0
    ms=st.columns(4,gap="medium")
    for c,(a,b,d) in zip(ms,[("Selected date",selected.strftime("%d %b %Y"),"observation"),("Processed cells",f"{processed:,}","selected field"),("Potential-risk",f"{risk_count:,}","screening layer"),("Risk share",f"{share:.2f}%","selected field")]):
        with c: metric(a,b,d)
    st.markdown('<div class="map-shell"><div class="map-head"><b>Indian Ocean · Arabian Sea · Bay of Bengal</b><span>Blue field · green concentration · red screening flags</span></div>',unsafe_allow_html=True)
    fig=map_figure(blue,selected_risk)
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"scrollZoom":True,"responsive":True})
    st.markdown('<div class="legend"><span class="dot blue"></span>Processed ocean field <span class="dot green"></span>Flag concentration zone <span class="dot red"></span>Potential bloom-risk cell</div></div>',unsafe_allow_html=True)
    if mode=="historical screening layer":
        st.markdown('<div class="note"><b>Timeline note:</b> earlier dates use the compact historical screening layer built from the aggregated Chl-a history. The latest date uses the final model screening output. These historical flags are screening proxies, not additional model predictions.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="note"><b>Reading the map:</b> green areas show spatial concentration of screening flags, not a separate severity score. Red cells are potential-risk screening results, not confirmed harmful algal blooms.</div>',unsafe_allow_html=True)

# ---------------- LOCATION ----------------
elif st.session_state.page=="location":
    section("02 · LOCATION INTELLIGENCE","Check one coordinate across the available evidence.","Your input is kept separate from the nearest valid processed ocean cell. Land/zero-value cells are excluded from the lookup so the result does not silently return a meaningless 0.0000 observation.")
    left,right=st.columns([.82,1.18],gap="large")
    with left:
        st.markdown('<div class="card"><div class="kicker">YOUR INPUT</div><h3>Coordinates</h3><p>Enter decimal degrees. Example: 17.38, 78.49.</p></div>',unsafe_allow_html=True)
        with st.form("location_form",clear_on_submit=False):
            lat=st.number_input("Latitude",-90.0,90.0,float(st.session_state.get("lookup_lat",17.38)),step=.01,format="%.2f")
            lon=st.number_input("Longitude",-180.0,180.0,float(st.session_state.get("lookup_lon",78.49)),step=.01,format="%.2f")
            submit=st.form_submit_button("🔎 Check location",type="primary",use_container_width=True)
        if submit:
            st.session_state.lookup_lat=float(lat); st.session_state.lookup_lon=float(lon)
    lat=float(st.session_state.get("lookup_lat",17.38)); lon=float(st.session_state.get("lookup_lon",78.49))
    row=nearest_valid(lat,lon)
    with right:
        if row is None:
            st.error("No valid ocean observation is available for this lookup.")
        else:
            flagged=bool(row.risk_flag); dist=np.sqrt((float(row.latitude)-lat)**2+((float(row.longitude)-lon)*max(np.cos(np.deg2rad(lat)),.25))**2)
            st.markdown(f'<div class="card"><div class="kicker">NEAREST VALID OCEAN CELL</div><h3>{float(row.latitude):.4f}° · {float(row.longitude):.4f}°</h3><p><b>Your input:</b> {lat:.4f}° · {lon:.4f}°<br><b>Approx. angular separation:</b> {dist:.2f}°</p><div class="result-grid"><div class="result"><span>Observation date</span><b>{pd.Timestamp(row.date).strftime("%d %b %Y")}</b></div><div class="result"><span>Chlorophyll-a</span><b>{fmt(row.chla)}</b></div><div class="result"><span>Risk probability</span><b>{pct(row.get("risk_probability",np.nan))}</b></div><div class="result"><span>Region</span><b>{region_name(row.latitude,row.longitude)}</b></div></div></div>',unsafe_allow_html=True)
            if flagged:
                st.markdown('<div class="status risk"><b>🔴 POTENTIAL BLOOM-RISK FLAG</b>This valid ocean cell is included in the latest screening shortlist.</div>',unsafe_allow_html=True)
            else:
                st.markdown('<div class="status normal"><b>🟢 NOT FLAGGED</b>This valid ocean cell is not included in the latest potential-risk shortlist.</div>',unsafe_allow_html=True)
    if row is not None:
        st.markdown('<div class="kicker" style="margin-top:25px">SUPPORTING SIGNALS</div>',unsafe_allow_html=True)
        vals=list(supporting_values(row).items())
        sc=st.columns(4,gap="small")
        for c,(label,val) in zip(sc,vals):
            with c: st.markdown(f'<div class="signal"><span>{label}</span><b>{fmt(val)}</b></div>',unsafe_allow_html=True)
        hist=history_for_cell(row.latitude,row.longitude)
        if not hist.empty:
            st.markdown('<div style="height:15px"></div>',unsafe_allow_html=True)
            hf=px.line(hist,x="date",y="chla",markers=False)
            hf.update_traces(line_color="#0aa8b5",line_width=2.5,hovertemplate="Date: %{x|%d %b %Y}<br>Chl-a: %{y:.4f}<extra></extra>")
            hf.update_layout(height=320,margin=dict(l=55,r=20,t=25,b=55),paper_bgcolor="white",plot_bgcolor="white",showlegend=False,
                             xaxis_title="Observation date",yaxis_title="Mean Chl-a",font=dict(color="#174b56"),xaxis=dict(gridcolor="#e3eeee"),yaxis=dict(gridcolor="#e3eeee"))
            st.markdown('<div class="chart-card"><div class="kicker">LOCATION HISTORY</div><div class="card" style="box-shadow:none;border:0;padding:0"><h3>How the local Chl-a signal changed</h3></div>',unsafe_allow_html=True); st.plotly_chart(hf,use_container_width=True,config={"displaylogo":False}); st.markdown('</div>',unsafe_allow_html=True)
        sv=supporting_values(row)
        report=pd.DataFrame([{"input_latitude":lat,"input_longitude":lon,"nearest_latitude":row.latitude,"nearest_longitude":row.longitude,"date":row.date.strftime("%Y-%m-%d"),"chla":row.chla,"historical_baseline":sv["Historical baseline"],"chla_anomaly":sv["Anomaly"],"chla_change":sv["Recent change"],"risk_label":row.risk_label,"risk_probability":row.get("risk_probability",np.nan),"region":region_name(row.latitude,row.longitude)}])
        st.download_button("⬇ Download location report",report.to_csv(index=False).encode(),"bloomdetect_location_report.csv","text/csv",use_container_width=True)

# ---------------- INSIGHTS ----------------
elif st.session_state.page=="insights":
    section("03 · OCEAN INTELLIGENCE","Where should the current signal receive closer attention?","Hotspot concentration and current-field statistics are combined here because they answer the same investigation question without duplicating the Risk Map.")
    total=len(study_latest); flags=len(risk_latest); share=100*flags/total if total else 0
    mc=st.columns(4,gap="medium")
    for c,(a,b,d) in zip(mc,[("Observation cells",f"{total:,}","latest field"),("Potential-risk",f"{flags:,}","screening output"),("Risk share",f"{share:.2f}%","latest field"),("Mean Chl-a",fmt(study_latest.chla.mean()),"latest field")]):
        with c: metric(a,b,d)
    # hotspots
    r=risk_latest.copy(); r["lat_zone"]=np.floor(r.latitude/2)*2+1; r["lon_zone"]=np.floor(r.plot_lon/2)*2+1
    zones=r.groupby(["lat_zone","lon_zone"],as_index=False).agg(flagged_cells=("risk_flag","size"),mean_chla=("chla","mean"),max_chla=("chla","max")).sort_values("flagged_cells",ascending=False).head(12)
    st.markdown('<div class="section"><div class="kicker">HOTSPOT INTELLIGENCE</div><h2>Where are the flags concentrating?</h2><p>Nearby screening flags are grouped into broad 2° × 2° investigation zones. This is a spatial concentration view, not a severity ranking.</p></div>',unsafe_allow_html=True)
    if zones.empty:
        st.info("No potential-risk cells are available in the latest study field.")
    else:
        zones=zones.copy(); zones["Zone"]=zones.apply(lambda x:f"{x.lat_zone:.0f}°–{x.lat_zone+2:.0f}°, {x.lon_zone:.0f}°–{x.lon_zone+2:.0f}°",axis=1); chart=zones.iloc[::-1]
        fig=px.bar(chart,x="flagged_cells",y="Zone",orientation="h",text="flagged_cells")
        fig.update_traces(marker_color="#22a879",textposition="outside",cliponaxis=False)
        fig.update_layout(height=440,margin=dict(l=170,r=55,t=25,b=60),paper_bgcolor="white",plot_bgcolor="white",showlegend=False,font=dict(color="#174b56"),xaxis_title="Potential-risk cells in zone",yaxis_title="Investigation zone",xaxis=dict(gridcolor="#e3eeee"),yaxis=dict(gridcolor="#fff"))
        st.markdown('<div class="chart-card"><div class="kicker">TOP INVESTIGATION ZONES</div>',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False}); st.markdown('</div>',unsafe_allow_html=True)
        table=zones[["Zone","flagged_cells","mean_chla","max_chla"]].rename(columns={"flagged_cells":"Flagged cells","mean_chla":"Mean Chl-a","max_chla":"Max Chl-a"}).copy(); table["Mean Chl-a"]=table["Mean Chl-a"].round(4); table["Max Chl-a"]=table["Max Chl-a"].round(4)
        st.dataframe(table,hide_index=True,use_container_width=True)
    st.markdown('<div class="section"><div class="kicker">CURRENT FIELD</div><h2>What does the latest observation look like?</h2><p>These charts describe the latest field without repeating the spatial map.</p></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="large")
    with c1:
        f=px.histogram(study_latest,x="chla",nbins=32)
        f.update_traces(marker_color="#277fa4")
        f.update_layout(height=390,margin=dict(l=60,r=30,t=20,b=65),paper_bgcolor="white",plot_bgcolor="white",showlegend=False,font=dict(color="#174b56"),xaxis_title="Satellite-derived chlorophyll-a",yaxis_title="Number of processed cells",xaxis=dict(gridcolor="#e3eeee"),yaxis=dict(gridcolor="#e3eeee"))
        st.markdown('<div class="chart-card"><div class="kicker">DISTRIBUTION</div><h3>Chlorophyll-a distribution</h3>',unsafe_allow_html=True); st.plotly_chart(f,use_container_width=True,config={"displaylogo":False}); st.markdown('</div>',unsafe_allow_html=True)
    with c2:
        comp=pd.DataFrame({"Screening group":["Normal","Potential bloom risk"],"Mean Chl-a":[study_latest.loc[~study_latest.risk_flag,"chla"].mean(),study_latest.loc[study_latest.risk_flag,"chla"].mean()]})
        f=px.bar(comp,x="Screening group",y="Mean Chl-a",text="Mean Chl-a")
        f.update_traces(marker_color=["#277fa4","#ed5260"],texttemplate="%{text:.4f}",textposition="outside",cliponaxis=False)
        f.update_layout(height=390,margin=dict(l=60,r=45,t=20,b=65),paper_bgcolor="white",plot_bgcolor="white",showlegend=False,font=dict(color="#174b56"),xaxis_title="Screening group",yaxis_title="Mean chlorophyll-a",xaxis=dict(gridcolor="#fff"),yaxis=dict(gridcolor="#e3eeee"))
        st.markdown('<div class="chart-card"><div class="kicker">SCREENING GROUPS</div><h3>Mean chlorophyll-a by screening group</h3>',unsafe_allow_html=True); st.plotly_chart(f,use_container_width=True,config={"displaylogo":False}); st.markdown('</div>',unsafe_allow_html=True)
    regional=[]
    for name,cond in [("Arabian Sea",study_latest.latitude.between(5,30)&study_latest.plot_lon.between(45,75)),("Bay of Bengal",study_latest.latitude.between(0,25)&study_latest.plot_lon.between(75.01,100)),("Southern Indian Ocean",study_latest.latitude.between(-30,5)&study_latest.plot_lon.between(40,100)),("Northern Indian Ocean",study_latest.latitude.between(5,30)&study_latest.plot_lon.between(75.01,120))]:
        s=study_latest[cond]
        if len(s): regional.append({"Region":name,"Cells":len(s),"Potential-risk cells":int(s.risk_flag.sum()),"Risk share":100*s.risk_flag.mean(),"Mean Chl-a":s.chla.mean()})
    reg=pd.DataFrame(regional)
    if not reg.empty:
        reg["Risk share"]=reg["Risk share"].map(lambda x:f"{x:.2f}%"); reg["Mean Chl-a"]=reg["Mean Chl-a"].round(4)
        st.markdown('<div class="section"><div class="kicker">REGIONAL SIGNAL</div><h2>Broad-area comparison.</h2><p>Descriptive summaries of the latest processed field.</p></div>',unsafe_allow_html=True); st.dataframe(reg,hide_index=True,use_container_width=True)
    st.markdown('<div class="note"><b>Interpretation:</b> these views describe satellite-derived chlorophyll-a and the project screening output. They do not independently establish harmfulness, species identity or toxin presence.</div>',unsafe_allow_html=True)

# ---------------- DATA ----------------
elif st.session_state.page=="data":
    section("04 · DATA & OUTPUTS","The evidence behind the dashboard.","Source information and downloadable project outputs live here so scientific details are not repeated across the other pages.")
    dc=st.columns(4,gap="medium")
    for c,(a,b,d) in zip(dc,[("Product","E06OCM_L4_AC","EOS-06 / OCM-3"),("Grid","0.25°","latitude × longitude"),("Latest cells",f"{len(study_latest):,}","processed observation"),("Latest date",latest_display,"processed dataset")]):
        with c: metric(a,b,d)
    st.markdown('<div class="card" style="margin-top:18px"><div class="kicker">SOURCE PRODUCT</div><h3>EOS-06 OCM-3 analysed chlorophyll-a</h3><p>The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4 analysed chlorophyll product, E06OCM_L4_AC (E06OCM_L4_AC). The official MOSDAC product is a daily 0.25° × 0.25° analysed chlorophyll field.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="section"><div class="kicker">DOWNLOADS</div><h2>Take the actual project outputs.</h2><p>These files are generated directly from the data used by the dashboard.</p></div>',unsafe_allow_html=True)
    a,b=st.columns(2,gap="large")
    with a:
        st.markdown('<div class="card"><h3>Latest observation table</h3><p>All latest processed cells, including screening and supporting signals.</p></div>',unsafe_allow_html=True); st.download_button("⬇ Download latest observations",latest.to_csv(index=False).encode(),"bloomdetect_latest_observations.csv","text/csv",use_container_width=True)
    with b:
        st.markdown('<div class="card"><h3>Potential-risk shortlist</h3><p>Only the latest cells currently screened as potential bloom risk.</p></div>',unsafe_allow_html=True); st.download_button("⬇ Download potential-risk locations",risk_latest.to_csv(index=False).encode(),"bloomdetect_potential_risk.csv","text/csv",use_container_width=True)
    st.markdown('<div class="note"><b>Scientific use:</b> potential bloom risk is a project screening output. It is not confirmation of a harmful algal bloom, species identity or toxin presence. Satellite observations should be combined with field observations and additional environmental evidence.</div>',unsafe_allow_html=True)

st.markdown(f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Latest processed field: {latest_display}</div>',unsafe_allow_html=True)
