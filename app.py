from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Home | Risk Map | Location | Insights | Data
# Clean final dashboard: no date slider, no history dependency.
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
IMAGE_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# Proxy-screening thresholds used by the project target.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner="Loading satellite field...")
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv must be in the same folder as app.py."
        )

    d = pd.read_csv(DATA_PATH)
    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    numeric_cols = [
        "latitude", "longitude", "chla", "risk_probability", "model_score",
        "previous_chla", "historical_baseline", "recent_mean", "recent_max",
        "chla_anomaly", "chla_change",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
    d["risk_flag"] = (
        d["risk_label"].astype(str).str.strip().str.lower()
        == "potential bloom risk"
    )
    return d


try:
    df = load_data()
except Exception as exc:
    st.error("BloomDetect AI could not load the project data.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
latest_risk = latest[latest["risk_flag"]].copy()

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#103f49;--deep:#07566a;--aqua:#08a7b3;--muted:#64838a;--line:#a8d9dd;--bg:#f2fbfb;--green:#1fa477;--red:#ef4f5e;--blue:#2584a8;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif!important;}
.stApp{background:linear-gradient(180deg,#fbffff 0%,#effafa 55%,#fbffff 100%)!important;overflow-x:hidden!important;}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important;}
.block-container{max-width:1220px!important;padding:34px 30px 72px!important;margin:auto!important;}
.brand{display:flex;justify-content:space-between;align-items:center;gap:18px;padding:15px 19px;border:1.5px solid #c5e5e7;background:#fff;border-radius:21px;box-shadow:0 10px 28px rgba(9,80,94,.08);}
.brand-left{display:flex;align-items:center;gap:12px;min-width:0}.logo{width:49px;height:49px;border-radius:15px;background:linear-gradient(145deg,#2bcdd0,#087b8e);display:grid;place-items:center;color:#fff;font-weight:800;font-size:19px;flex:0 0 49px}.brand-name{font:800 1.2rem Manrope;color:var(--ink);letter-spacing:-.035em}.brand-sub{font-size:.71rem;color:#78959b;margin-top:2px}.latest{font-size:.67rem;color:#78959b;text-align:right;line-height:1.35}.latest b{font-size:.77rem;color:#174b56}
.nav{margin:12px 0 38px}.nav .stButton>button{min-height:42px!important;border-radius:13px!important;font-size:.79rem!important;}
.stButton>button,.stDownloadButton>button{border:2px solid #164e5b!important;background:#fff!important;color:#123f49!important;border-radius:13px!important;font-weight:800!important;min-height:43px!important;box-shadow:0 4px 13px rgba(15,70,82,.07)!important}.stButton>button:hover,.stDownloadButton>button:hover{background:#e4f8f8!important;border-color:#087d8c!important}.stButton>button[kind="primary"]{background:#d8f7f7!important;border-color:#078c9b!important}
.kicker{font:800 .65rem Manrope;letter-spacing:.17em;text-transform:uppercase;color:#0797a5;margin-bottom:9px}.section h1,.section h2{font:800 clamp(2.15rem,4.6vw,3.75rem)/1.03 Manrope;color:var(--ink);letter-spacing:-.06em;margin:0 0 12px}.section p{font-size:.93rem;color:#5f8088;line-height:1.65;margin:0;max-width:1060px}.section{padding:16px 0 8px}
.hero{position:relative;overflow:hidden;min-height:385px;padding:62px 58px;border-radius:30px;background:linear-gradient(135deg,#063d51,#087486 55%,#12a8ad);box-shadow:0 22px 58px rgba(6,86,100,.15);}.hero:after{content:"";position:absolute;width:520px;height:520px;right:-180px;top:-200px;border:80px solid rgba(205,255,253,.10);border-radius:50%;box-shadow:0 0 0 55px rgba(205,255,253,.05),0 0 0 120px rgba(205,255,253,.035)}.hero-content{position:relative;z-index:2;max-width:850px}.hero .kicker{color:#a6fffa}.hero h1{font:800 clamp(3rem,6vw,5.5rem)/.93 Manrope;color:#e5ffff;letter-spacing:-.075em;margin:0 0 20px}.hero p{font-size:1rem;line-height:1.72;color:#e2fbfb;max-width:780px}.badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}.badge{padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.74rem;font-weight:700}
.metric{min-height:116px;padding:17px;border:1.5px solid #a6d5da;border-radius:18px;background:linear-gradient(145deg,#fff,#eaf8f8);box-shadow:0 9px 24px rgba(15,91,101,.06)}.metric-label{font:800 .61rem Manrope;letter-spacing:.11em;text-transform:uppercase;color:#6d8c93}.metric-value{font:800 1.45rem Manrope;color:#123f49;margin-top:5px;overflow-wrap:anywhere}.metric-note{font-size:.69rem;color:#78959b;margin-top:4px}
.card{padding:21px;border:1.5px solid #afd9dc;border-radius:20px;background:#fff;box-shadow:0 11px 29px rgba(9,80,94,.07)}.card h3{font:800 1.18rem Manrope;color:#123f49;margin:0 0 8px}.card p{font-size:.87rem;line-height:1.6;color:#66848b;margin:0}.note{padding:13px 15px;margin-top:14px;background:#e5f8f8;border-left:4px solid #11a9b2;border-radius:0 13px 13px 0;color:#52757c;font-size:.8rem;line-height:1.55}.note b{color:#174b56}
.story{display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:center;margin-top:26px}.story-img{border:1.5px solid #a9d6da;border-radius:21px;overflow:hidden;background:#dff7f8;box-shadow:0 12px 32px rgba(9,80,94,.08)}
.map-wrap{background:#dff7f8;border:1.5px solid #91cbd1;border-radius:21px;padding:5px;box-shadow:0 14px 35px rgba(9,80,94,.08);overflow:hidden}.map-head{display:flex;justify-content:space-between;gap:10px;padding:9px 11px;align-items:center}.map-head b{font-size:.82rem;color:#174b56}.map-head span{font-size:.67rem;color:#6b8b92}.legend{display:flex;gap:12px;flex-wrap:wrap;padding:10px 12px;color:#54757c;font-size:.72rem}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:-1px;margin-right:4px}.square{border-radius:3px}
.signal-title{font:800 .64rem Manrope;letter-spacing:.15em;text-transform:uppercase;color:#6d8c93;margin:20px 0 8px}.signal{padding:15px;border:1.5px solid #b9dfe2;border-radius:17px;background:#fff}.signal-label{font-size:.67rem;color:#78959b}.signal-value{font:800 1.15rem Manrope;color:#164a55;margin-top:4px}.signal-sub{font-size:.7rem;color:#78959b;margin-top:3px}
.status{padding:15px;border-radius:16px;border:2px solid;margin-top:14px}.status-risk{background:#fff0f2;border-color:#f06a78;color:#9e2d3c}.status-ok{background:#eafaf4;border-color:#49b995;color:#176f58}.status-title{font:800 .9rem Manrope;margin-bottom:3px}
[data-testid="stNumberInput"] label{color:#315f68!important;font-weight:800!important;font-size:.78rem!important}
[data-testid="stNumberInput"] input{color:#123f49!important;-webkit-text-fill-color:#123f49!important;background:#fff!important;border:2px solid #164e5b!important;border-radius:11px!important;font-weight:700!important;opacity:1!important}
[data-testid="stNumberInput"] div[data-baseweb="input"]{background:#fff!important;border-radius:11px!important}
.stSelectbox div[data-baseweb="select"]>div{border:2px solid #164e5b!important;border-radius:11px!important;background:#fff!important;color:#123f49!important}.stPlotlyChart{border-radius:18px;overflow:hidden}
.tool{padding:18px;border:1.5px solid #b5dde0;border-radius:18px;background:#fff;min-height:145px;box-shadow:0 8px 22px rgba(9,80,94,.06)}.tool h3{font:800 1rem Manrope;color:#123f49;margin:0 0 6px}.tool p{font-size:.79rem;line-height:1.5;color:#66848b;margin:0}
.footer{border-top:1px solid #d5ebed;margin-top:40px;padding-top:14px;color:#76959b;font-size:.67rem}
@media(max-width:900px){.story{grid-template-columns:1fr}.toolgrid{grid-template-columns:1fr 1fr!important}}@media(max-width:620px){.block-container{padding:15px 12px 45px!important}.brand{padding:12px}.latest{display:none}.hero{padding:35px 24px}.hero h1{font-size:3rem}.toolgrid,.metricgrid,.signalgrid{grid-template-columns:1fr!important}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def fmt_num(x, digits=4):
    try:
        if pd.isna(x):
            return "Unavailable"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "Unavailable"


def fmt_date(x):
    return pd.to_datetime(x).strftime("%d %b %Y")


def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def heading(kicker, title, copy):
    st.markdown(
        f'<div class="section"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{copy}</p></div>',
        unsafe_allow_html=True,
    )


def add_footer():
    st.markdown(
        f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Latest processed field: {fmt_date(latest_date)}</div>',
        unsafe_allow_html=True,
    )


def scattergeo_base():
    return dict(
        projection_type="mercator",
        showland=True,
        landcolor="#dce6e5",
        showocean=True,
        oceancolor="#e6f8fa",
        showcountries=True,
        countrycolor="#94bcc2",
        coastlinecolor="#77aab1",
        showlakes=True,
        lakecolor="#e6f8fa",
        lonaxis=dict(range=[LON_MIN, LON_MAX], showgrid=True, gridcolor="#c5e5e8", dtick=10),
        lataxis=dict(range=[LAT_MIN, LAT_MAX], showgrid=True, gridcolor="#c5e5e8", dtick=10),
        bgcolor="#e6f8fa",
    )


def build_map(view):
    # Keep the background field readable without asking Plotly to draw all
    # 70k points at once. Risk cells are always retained in full.
    field = latest.copy()
    if len(field) > 12000:
        field = field.sample(12000, random_state=42)
    risk = latest_risk.copy()

    fig = go.Figure()

    if view == "Potential bloom-risk screening":
        fig.add_trace(go.Scattergeo(
            lon=field["longitude"], lat=field["latitude"],
            mode="markers", name="Processed ocean field",
            marker=dict(size=3.2, color="#65b8d1", opacity=.34),
            hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Processed cell<extra></extra>",
        ))
        if not risk.empty:
            fig.add_trace(go.Scattergeo(
                lon=risk["longitude"], lat=risk["latitude"],
                mode="markers", name="Potential bloom-risk cell",
                marker=dict(size=7.5, color="#ef4f5e", opacity=.92, line=dict(width=.5,color="#fff")),
                customdata=np.c_[risk["chla"].fillna(np.nan), risk.get("risk_probability", pd.Series(np.nan,index=risk.index)).fillna(np.nan)],
                hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Chl-a %{customdata[0]:.4f}<br>Risk probability %{customdata[1]:.1%}<extra></extra>",
            ))
    else:
        col = {"Current Chl-a":"chla", "Recent change":"chla_change", "Historical anomaly":"chla_anomaly"}[view]
        plot = latest.dropna(subset=[col]).copy()
        if len(plot) > 14000:
            plot = plot.sample(14000, random_state=42)
        if plot.empty:
            return None
        if view == "Current Chl-a":
            colorscale="YlGnBu"; title="Chl-a"; ctitle="Chl-a"
        else:
            colorscale="RdBu_r"; title="Signal"; ctitle="Change"
        fig.add_trace(go.Scattergeo(
            lon=plot["longitude"], lat=plot["latitude"], mode="markers", name=view,
            marker=dict(size=4.1, color=plot[col], colorscale=colorscale, opacity=.76, colorbar=dict(title=ctitle, thickness=13)),
            customdata=np.c_[plot["chla"], plot[col]],
            hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Chl-a %{customdata[0]:.4f}<br>Signal %{customdata[1]:.4f}<extra></extra>",
        ))
        # Make risk cells visible on every scientific signal view.
        if not risk.empty:
            fig.add_trace(go.Scattergeo(
                lon=risk["longitude"], lat=risk["latitude"], mode="markers", name="Potential-risk cells",
                marker=dict(size=6.5, color="#ef4f5e", opacity=.88),
                hovertemplate="Potential bloom-risk screening cell<extra></extra>",
            ))

    fig.update_geos(**scattergeo_base())
    fig.update_layout(
        height=610,
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="#e6f8fa",
        plot_bgcolor="#e6f8fa",
        font=dict(family="DM Sans", color="#174b56"),
        legend=dict(orientation="h", y=.01, x=.02, bgcolor="rgba(255,255,255,.75)", bordercolor="#b9dfe2", borderwidth=1),
        hoverlabel=dict(bgcolor="#103f49", font_color="#fff"),
    )
    return fig


# ============================================================
# HEADER + NAVIGATION
# ============================================================

st.markdown(
    f'''<div class="brand"><div class="brand-left"><div class="logo">≈</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div></div></div><div class="latest">Latest processed field<br><b>{fmt_date(latest_date)}</b></div></div>''',
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state.page = "home"

pages = ["home", "map", "location", "insights", "data"]
labels = {"home":"⌂ Home", "map":"🌊 Risk Map", "location":"📍 Location", "insights":"📊 Insights", "data":"⇩ Data"}

st.markdown('<div class="nav">', unsafe_allow_html=True)
nav_cols = st.columns(5)
for col, page in zip(nav_cols, pages):
    with col:
        if st.button(labels[page], key=f"nav_{page}", use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":
    st.markdown(
        '''<div class="hero"><div class="hero-content"><div class="kicker">EOS-06 · OCM-3 · Satellite intelligence</div><h1>Read the ocean signal.</h1><p>BloomDetect AI turns satellite-derived chlorophyll-a observations and temporal context into practical early-warning support for identifying locations whose patterns may deserve closer investigation.</p><div class="badges"><span class="badge">🌊 Ocean colour</span><span class="badge">🛰 EOS-06 OCM-3</span><span class="badge">🌱 Potential bloom-risk screening</span><span class="badge">📍 Spatial intelligence</span></div></div></div>''',
        unsafe_allow_html=True,
    )

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Processed cells", f"{len(latest):,}", "latest satellite field")
    with c2: metric_card("Potential-risk cells", f"{len(latest_risk):,}", "screening output")
    with c3: metric_card("Risk share", f"{(len(latest_risk)/len(latest)*100 if len(latest) else 0):.2f}%", "of processed cells")
    with c4: metric_card("Maximum Chl-a", fmt_num(latest["chla"].max()), "latest field")

    left,right = st.columns([1,1], gap="large")
    with left:
        st.markdown('<div class="card"><h3>From satellite observation to investigation support</h3><p>BloomDetect uses satellite-derived chlorophyll-a and temporal context to screen for unusual patterns. The output is a <b>potential bloom-risk signal</b>, not a confirmed harmful algal bloom.</p><div class="note"><b>Important:</b> High chlorophyll-a alone does not prove a harmful algal bloom. Species, toxins and ecological impacts require additional evidence and field validation.</div></div>', unsafe_allow_html=True)
    with right:
        if IMAGE_PATH.exists():
            st.image(str(IMAGE_PATH), use_container_width=True, caption="How satellite ocean-colour observations support bloom-risk investigation")
        else:
            st.info("Project illustration not found. The dashboard itself is still fully usable.")

    st.markdown('<div class="section"><div class="kicker">FOUR FOCUSED VIEWS</div><h2>Everything has one job.</h2><p>Move from the spatial field to one location, then to summary insights and project data without repeating the same information everywhere.</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    tools = [
        ("🌊", "Risk Map", "Read the latest spatial field, change signal, anomaly and potential-risk cells."),
        ("📍", "Location", "Check one coordinate against the nearest valid processed ocean cell."),
        ("📊", "Insights", "Understand risk concentration, Chl-a distribution and the strongest current signals."),
        ("⇩", "Data", "Keep the project source, scope, methodology notes and downloadable output together."),
    ]
    for col,(icon,title,copy) in zip(cols,tools):
        with col:
            st.markdown(f'<div class="tool"><div style="font-size:1.25rem;margin-bottom:8px">{icon}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)
    add_footer()

# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":
    heading("01 · SPATIAL INTELLIGENCE", "See the field clearly.", "The latest processed satellite field is shown without a fragile date slider. Switch the map view to inspect concentration, recent change, historical anomaly or the current potential-risk screening output.")

    map_view = st.selectbox(
        "Map detail",
        ["Potential bloom-risk screening", "Current Chl-a", "Recent change", "Historical anomaly"],
        key="map_view",
    )

    # Signal summary before the map.
    mean_chla = latest["chla"].mean()
    mean_change = latest["chla_change"].mean() if "chla_change" in latest else np.nan
    mean_anomaly = latest["chla_anomaly"].mean() if "chla_anomaly" in latest else np.nan
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Processed cells", f"{len(latest):,}", "latest field")
    with c2: metric_card("Mean Chl-a", fmt_num(mean_chla), "current field")
    with c3: metric_card("Mean recent change", fmt_num(mean_change), "vs previous observation")
    with c4: metric_card("Mean anomaly", fmt_num(mean_anomaly), "vs historical baseline")

    fig = build_map(map_view)
    if fig is not None:
        st.markdown('<div class="map-wrap"><div class="map-head"><b>Indian Ocean · Arabian Sea · Bay of Bengal</b><span>Blue = processed field · red = potential-risk screening</span></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False,"scrollZoom":True,"responsive":True})
        st.markdown('<div class="legend"><span><i class="dot" style="background:#2584a8"></i>Processed field</span><span><i class="dot" style="background:#ef4f5e"></i>Potential-risk cell</span><span>Hover points for values · scroll to zoom</span></div></div>', unsafe_allow_html=True)

    # The useful replacement for the old slider: a compact change profile.
    st.markdown('<div class="signal-title">What changed in the latest field</div>', unsafe_allow_html=True)
    positive_change = int((latest.get("chla_change", pd.Series(index=latest.index, dtype=float)) > 0).sum())
    anomaly_cells = int((latest.get("chla_anomaly", pd.Series(index=latest.index, dtype=float)) >= ANOMALY_THRESHOLD).sum())
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(f'<div class="signal"><div class="signal-label">Cells increasing vs previous observation</div><div class="signal-value">{positive_change:,}</div><div class="signal-sub">Chl-a change &gt; 0</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="signal"><div class="signal-label">Cells above anomaly threshold</div><div class="signal-value">{anomaly_cells:,}</div><div class="signal-sub">anomaly ≥ {ANOMALY_THRESHOLD:.4f}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="signal"><div class="signal-label">Potential-risk cells</div><div class="signal-value">{len(latest_risk):,}</div><div class="signal-sub">proxy screening output</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="note"><b>Reading the map:</b> the scientific signal layers show satellite-derived Chl-a patterns. Red cells are potential bloom-risk screening results, not confirmed harmful algal blooms.</div>', unsafe_allow_html=True)
    add_footer()

# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page == "location":
    heading("02 · LOCATION INTELLIGENCE", "Check one coordinate with clean evidence.", "Enter a latitude and longitude. BloomDetect separates your input from the nearest valid processed ocean cell, so a land or zero-value cell is never presented as a meaningful ocean observation.")

    left,right = st.columns([.75,1.25], gap="large")
    with left:
        st.markdown('<div class="card"><h3>Your input</h3><p>Enter decimal degrees. Example: 17.38, 78.49.</p></div>', unsafe_allow_html=True)
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=17.38, step=0.01, format="%.4f", key="input_lat")
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.49, step=0.01, format="%.4f", key="input_lon")
        check = st.button("🔎 Check location", use_container_width=True, type="primary")

    # Use the latest field only. Keep only genuinely valid ocean observations.
    valid = latest[
        latest["chla"].notna()
        & np.isfinite(latest["latitude"])
        & np.isfinite(latest["longitude"])
        & (latest["chla"] > 0)
        & latest["latitude"].between(LAT_MIN, LAT_MAX)
        & latest["longitude"].between(LON_MIN, LON_MAX)
    ].copy()

    if check:
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            with right:
                st.warning(f"The coordinate is outside the BloomDetect study window ({LAT_MIN}° to {LAT_MAX}° latitude, {LON_MIN}° to {LON_MAX}° longitude).")
        elif valid.empty:
            with right:
                st.error("No valid processed ocean cells are available for this lookup.")
        else:
            # Longitude is not wrapped here because the study window is 20°E–120°E.
            dist = ((valid["latitude"] - lat) / 70.0) ** 2 + ((valid["longitude"] - lon) / 100.0) ** 2
            idx = dist.idxmin()
            row = valid.loc[idx]
            distance_deg = float(np.sqrt(((row["latitude"]-lat))**2 + ((row["longitude"]-lon))**2))
            flagged = bool(row["risk_flag"])

            with right:
                st.markdown(f'<div class="card"><div class="kicker">NEAREST VALID PROCESSED OCEAN CELL</div><h3>{row["latitude"]:.4f}° · {row["longitude"]:.4f}°</h3><p>Your input: {lat:.4f}° · {lon:.4f}°<br>Approx. angular separation: {distance_deg:.2f}°</p></div>', unsafe_allow_html=True)

                # Native Streamlit columns. No raw HTML blob can leak into the UI.
                r1,r2 = st.columns(2)
                with r1:
                    st.markdown(f'<div class="signal"><div class="signal-label">Observation date</div><div class="signal-value">{fmt_date(row["date"])}</div></div>', unsafe_allow_html=True)
                with r2:
                    st.markdown(f'<div class="signal"><div class="signal-label">Chlorophyll-a</div><div class="signal-value">{fmt_num(row["chla"])}</div></div>', unsafe_allow_html=True)

                r3,r4 = st.columns(2)
                with r3:
                    prob = row.get("risk_probability", np.nan)
                    prob_text = f"{float(prob):.1%}" if pd.notna(prob) else "Unavailable"
                    st.markdown(f'<div class="signal"><div class="signal-label">Risk probability</div><div class="signal-value">{prob_text}</div></div>', unsafe_allow_html=True)
                with r4:
                    st.markdown(f'<div class="signal"><div class="signal-label">Risk status</div><div class="signal-value">{"Potential risk" if flagged else "Not flagged"}</div></div>', unsafe_allow_html=True)

                if flagged:
                    st.markdown('<div class="status status-risk"><div class="status-title">🔴 Potential bloom-risk screening</div>This processed cell is included in the latest potential-risk screening output.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status status-ok"><div class="status-title">🟢 Not flagged</div>This valid processed ocean cell is not included in the latest potential-risk screening output.</div>', unsafe_allow_html=True)

                st.markdown('<div class="signal-title">Supporting signals</div>', unsafe_allow_html=True)
                s1,s2,s3,s4 = st.columns(4)
                vals = [
                    ("Current Chl-a", row.get("chla",np.nan)),
                    ("Historical baseline", row.get("historical_baseline",np.nan)),
                    ("Anomaly", row.get("chla_anomaly",np.nan)),
                    ("Recent change", row.get("chla_change",np.nan)),
                ]
                for c,(lab,val) in zip([s1,s2,s3,s4],vals):
                    with c:
                        st.markdown(f'<div class="signal"><div class="signal-label">{lab}</div><div class="signal-value">{fmt_num(val)}</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="note"><b>Interpretation:</b> this lookup reports the nearest valid processed satellite cell. It is a screening aid and does not establish a confirmed harmful algal bloom.</div>', unsafe_allow_html=True)
    else:
        with right:
            st.markdown('<div class="card"><h3>Nearest processed ocean cell</h3><p>Press <b>Check location</b> to evaluate the coordinate. Land and zero-value cells are excluded from the result.</p><div class="note">For example, a coordinate on Hyderabad land will not be incorrectly reported as a meaningful ocean observation just because the satellite grid contains a zero there.</div></div>', unsafe_allow_html=True)

    add_footer()

# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":
    heading("03 · SIGNAL INSIGHTS", "Turn the field into a few useful signals.", "This page combines the latest screening output with Chl-a change and anomaly information already stored in the processed dataset. It is descriptive analysis, not a second model.")

    # Risk concentration by coarse geographic bins.
    risk_work = latest_risk.copy()
    if not risk_work.empty:
        risk_work["lat_zone"] = (np.floor(risk_work["latitude"] / 5) * 5).astype(int)
        risk_work["lon_zone"] = (np.floor(risk_work["longitude"] / 5) * 5).astype(int)
        zone = risk_work.groupby(["lat_zone","lon_zone"], as_index=False).size().rename(columns={"size":"risk_cells"})
        zone["label"] = zone.apply(lambda r: f"{r.lat_zone}° to {r.lat_zone+5}°N/S · {r.lon_zone}° to {r.lon_zone+5}°E", axis=1)
        zone = zone.sort_values("risk_cells", ascending=False).head(10)
    else:
        zone = pd.DataFrame(columns=["label","risk_cells"])

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Potential-risk cells", f"{len(latest_risk):,}", "latest screening")
    with c2: metric_card("Risk share", f"{(len(latest_risk)/len(latest)*100 if len(latest) else 0):.2f}%", "of processed cells")
    with c3: metric_card("Positive Chl-a change", f"{int((latest.get('chla_change',pd.Series(dtype=float))>0).sum()):,}", "cells increasing")
    with c4: metric_card("Anomalous cells", f"{int((latest.get('chla_anomaly',pd.Series(dtype=float))>=ANOMALY_THRESHOLD).sum()):,}", "above proxy threshold")

    a,b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="card"><h3>Potential-risk concentration</h3><p>Coarse 5° × 5° zones show where the current screening flags are concentrated.</p></div>', unsafe_allow_html=True)
        if not zone.empty:
            zplot = zone.sort_values("risk_cells")
            fig = px.bar(zplot, x="risk_cells", y="label", orientation="h")
            fig.update_traces(marker_color="#ef4f5e")
            fig.update_layout(height=420, margin=dict(l=10,r=15,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans",color="#234f59",size=13), xaxis_title="Potential-risk cells", yaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
        else:
            st.info("No potential-risk cells are present in the latest field.")

    with b:
        st.markdown('<div class="card"><h3>Chl-a distribution</h3><p>Distribution of valid latest-field chlorophyll-a observations.</p></div>', unsafe_allow_html=True)
        valid_chla = latest.loc[latest["chla"] > 0, "chla"].dropna()
        if not valid_chla.empty:
            fig = px.histogram(valid_chla, nbins=40, labels={"value":"Chlorophyll-a"})
            fig.update_traces(marker_color="#2584a8")
            fig.update_layout(height=420, margin=dict(l=10,r=15,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans",color="#234f59",size=13), xaxis_title="Chlorophyll-a", yaxis_title="Cells", showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})

    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="card"><h3>Current signal summary</h3><p>A compact summary of the latest field, without the old distribution graph.</p></div>',
            unsafe_allow_html=True,
        )
        mean_chla = latest["chla"].mean()
        max_chla = latest["chla"].max()
        mean_change = latest["chla_change"].mean() if "chla_change" in latest.columns else np.nan
        mean_anomaly = latest["chla_anomaly"].mean() if "chla_anomaly" in latest.columns else np.nan
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f'<div class="signal"><div class="signal-label">Mean Chl-a</div><div class="signal-value">{fmt_num(mean_chla)}</div><div class="signal-sub">latest field</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="signal"><div class="signal-label">Maximum Chl-a</div><div class="signal-value">{fmt_num(max_chla)}</div><div class="signal-sub">latest field</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        s3, s4 = st.columns(2)
        with s3:
            st.markdown(f'<div class="signal"><div class="signal-label">Mean recent change</div><div class="signal-value">{fmt_num(mean_change)}</div><div class="signal-sub">vs previous observation</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="signal"><div class="signal-label">Mean anomaly</div><div class="signal-value">{fmt_num(mean_anomaly)}</div><div class="signal-sub">vs historical baseline</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<div class="card"><h3>Highest current-risk cells</h3><p>Locations with the largest stored model screening scores.</p></div>',
            unsafe_allow_html=True,
        )
        if not latest_risk.empty and "risk_probability" in latest_risk.columns:
            top = latest_risk.sort_values("risk_probability", ascending=False).head(8).copy()
            top["Location"] = top.apply(lambda r: f"{r.latitude:.2f}°, {r.longitude:.2f}°", axis=1)
            table = top[["Location","chla","risk_probability"]].copy()
            table.columns = ["Location","Chl-a","Risk probability"]
            table["Chl-a"] = table["Chl-a"].map(lambda x: fmt_num(x,4))
            table["Risk probability"] = table["Risk probability"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "Unavailable")
            st.dataframe(table, use_container_width=True, hide_index=True, height=280)
        else:
            st.info("Risk-probability values are not available for the latest field.")

    st.markdown('<div class="note"><b>Scientific caution:</b> these insights describe satellite-derived signals and the project screening proxy. They do not identify algal species, toxins or confirmed harmful blooms.</div>', unsafe_allow_html=True)
    add_footer()

# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":
    heading("04 · DATA & METHOD", "Everything needed to understand the output.", "The Data view keeps source information, study coverage, project interpretation and downloadable outputs in one place instead of repeating them across the dashboard.")

    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card"><h3>Source product</h3><p><b>EOS-06 / Oceansat-3 OCM-3</b><br>Level-4 Analysed Chlorophyll Product · E06OCM_L4_AC<br>NetCDF satellite-derived ocean-colour product.</p><div class="note"><b>Study window:</b> Indian Ocean region shown in the dashboard, approximately 20°E–120°E and 40°S–30°N.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>What the system screens</h3><p>BloomDetect combines current Chl-a with temporal context such as the previous observation, historical baseline, recent mean and recent maximum. The final classifier produces a potential bloom-risk screening signal.</p><div class="note"><b>Important:</b> the dataset does not contain confirmed HAB species/toxin labels. Therefore the output must be interpreted as potential bloom-risk screening, not confirmed HAB detection.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="kicker">LATEST OUTPUT</div><h2>Download the processed field.</h2></div>', unsafe_allow_html=True)
    csv_bytes = latest.to_csv(index=False).encode("utf-8")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⇩ Download latest field CSV", data=csv_bytes, file_name="bloomdetect_latest_field.csv", mime="text/csv", use_container_width=True)
    with c2:
        risk_bytes = latest_risk.to_csv(index=False).encode("utf-8")
        st.download_button("⇩ Download risk cells CSV", data=risk_bytes, file_name="bloomdetect_potential_risk_cells.csv", mime="text/csv", use_container_width=True)
    with c3:
        st.markdown('<div class="card" style="min-height:43px;padding:11px 14px"><p><b>Latest field:</b> '+fmt_date(latest_date)+"<br><b>Rows:</b> "+f"{len(latest):,}"+"<br><b>Risk cells:</b> "+f"{len(latest_risk):,}"+'</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:18px"><h3>Interpretation rule</h3><p>High chlorophyll-a alone does not establish a harmful algal bloom. Satellite observations are useful for spatial and temporal screening, but species identification, toxin confirmation and ecological impact assessment require additional evidence and field validation.</p></div>', unsafe_allow_html=True)
    add_footer()
