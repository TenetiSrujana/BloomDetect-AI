from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Final clean Streamlit dashboard
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
HISTORY_PATH = BASE_DIR / "bloomdetect_history.csv"
IMAGE_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

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
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
    d["risk_flag"] = (
        d["risk_label"].astype(str).str.strip().str.lower()
        == "potential bloom risk"
    )
    return d


@st.cache_data(show_spinner="Loading compact history...")
def load_history():
    if not HISTORY_PATH.exists():
        return None

    h = pd.read_csv(HISTORY_PATH)
    required = {"date", "lat_bin", "lon_bin", "chla", "risk"}
    if not required.issubset(h.columns):
        return None

    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    for col in ["lat_bin", "lon_bin", "chla", "risk", "cells", "max_chla"]:
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")

    h = h.dropna(subset=["date", "lat_bin", "lon_bin", "chla"]).copy()
    h["risk"] = h["risk"].fillna(0).astype(int)
    h = h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()
    return h


try:
    df = load_data()
except Exception as exc:
    st.error("BloomDetect AI could not load the project data.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
latest_risk = latest[latest["risk_flag"]].copy()

# History is optional. The app never crashes if it is not uploaded yet.
history = load_history()
history_available = history is not None and not history.empty

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
.block-container{max-width:1220px!important;padding:18px 28px 42px!important;margin:auto!important;}
.brand{display:flex;justify-content:space-between;align-items:center;gap:18px;padding:13px 17px;border:1.5px solid #c5e5e7;background:#fff;border-radius:19px;box-shadow:0 8px 24px rgba(9,80,94,.07);}
.brand-left{display:flex;align-items:center;gap:11px;min-width:0}.logo{width:46px;height:46px;border-radius:14px;background:linear-gradient(145deg,#2bcdd0,#087b8e);display:grid;place-items:center;color:#fff;font-weight:800;font-size:18px;flex:0 0 46px}.brand-name{font:800 1.14rem Manrope;color:var(--ink);letter-spacing:-.035em}.brand-sub{font-size:.68rem;color:#78959b;margin-top:2px}.latest{font-size:.63rem;color:#78959b;text-align:right;line-height:1.3}.latest b{font-size:.73rem;color:#174b56}
.nav{margin:8px 0 18px}.nav .stButton>button{min-height:40px!important;border-radius:12px!important;font-size:.76rem!important;}
.stButton>button,.stDownloadButton>button{border:1.7px solid #164e5b!important;background:#fff!important;color:#123f49!important;border-radius:12px!important;font-weight:800!important;min-height:41px!important;box-shadow:0 3px 11px rgba(15,70,82,.06)!important}.stButton>button:hover,.stDownloadButton>button:hover{background:#e4f8f8!important;border-color:#087d8c!important}.stButton>button[kind="primary"]{background:#d8f7f7!important;border-color:#078c9b!important}
.kicker{font:800 .61rem Manrope;letter-spacing:.17em;text-transform:uppercase;color:#0797a5;margin-bottom:8px}.section h1,.section h2{font:800 clamp(2.05rem,4.2vw,3.45rem)/1.04 Manrope;color:var(--ink);letter-spacing:-.06em;margin:0 0 10px}.section p{font-size:.88rem;color:#5f8088;line-height:1.58;margin:0;max-width:1080px}.section{padding:10px 0 6px}
.hero{position:relative;overflow:hidden;min-height:320px;padding:46px 48px;border-radius:27px;background:linear-gradient(135deg,#063d51,#087486 55%,#12a8ad);box-shadow:0 18px 48px rgba(6,86,100,.13);margin-bottom:18px}.hero:after{content:"";position:absolute;width:480px;height:480px;right:-180px;top:-190px;border:72px solid rgba(205,255,253,.10);border-radius:50%;box-shadow:0 0 0 50px rgba(205,255,253,.05),0 0 0 110px rgba(205,255,253,.035)}.hero-content{position:relative;z-index:2;max-width:850px}.hero .kicker{color:#a6fffa}.hero h1{font:800 clamp(2.9rem,5.8vw,5.1rem)/.94 Manrope;color:#e5ffff;letter-spacing:-.075em;margin:0 0 17px}.hero p{font-size:.94rem;line-height:1.65;color:#e2fbfb;max-width:780px}.badges{display:flex;flex-wrap:wrap;gap:7px;margin-top:18px}.badge{padding:7px 11px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.69rem;font-weight:700}
.metric{min-height:103px;padding:15px;border:1.5px solid #a6d5da;border-radius:17px;background:linear-gradient(145deg,#fff,#eaf8f8);box-shadow:0 7px 20px rgba(15,91,101,.05)}.metric-label{font:800 .58rem Manrope;letter-spacing:.11em;text-transform:uppercase;color:#6d8c93}.metric-value{font:800 1.38rem Manrope;color:#123f49;margin-top:4px;overflow-wrap:anywhere}.metric-note{font-size:.65rem;color:#78959b;margin-top:3px}
.card{padding:19px;border:1.5px solid #afd9dc;border-radius:19px;background:#fff;box-shadow:0 9px 25px rgba(9,80,94,.06)}.card h3{font:800 1.12rem Manrope;color:#123f49;margin:0 0 7px}.card p{font-size:.83rem;line-height:1.55;color:#66848b;margin:0}.note{padding:12px 14px;margin-top:12px;background:#e5f8f8;border-left:4px solid #11a9b2;border-radius:0 12px 12px 0;color:#52757c;font-size:.75rem;line-height:1.5}.note b{color:#174b56}
.story{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:center;margin-top:17px}.story-img{border:1.5px solid #a9d6da;border-radius:19px;overflow:hidden;background:#dff7f8;box-shadow:0 10px 28px rgba(9,80,94,.07)}
.map-wrap{background:#dff7f8;border:1.5px solid #91cbd1;border-radius:20px;padding:4px;box-shadow:0 12px 30px rgba(9,80,94,.07);overflow:hidden}.map-head{display:flex;justify-content:space-between;gap:10px;padding:8px 10px;align-items:center}.map-head b{font-size:.79rem;color:#174b56}.map-head span{font-size:.64rem;color:#6b8b92}.legend{display:flex;gap:12px;flex-wrap:wrap;padding:9px 11px;color:#54757c;font-size:.69rem}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:-1px;margin-right:4px}
.signal-title{font:800 .61rem Manrope;letter-spacing:.15em;text-transform:uppercase;color:#6d8c93;margin:17px 0 7px}.signal{padding:13px;border:1.5px solid #b9dfe2;border-radius:15px;background:#fff}.signal-label{font-size:.64rem;color:#78959b}.signal-value{font:800 1.08rem Manrope;color:#164a55;margin-top:3px}.signal-sub{font-size:.66rem;color:#78959b;margin-top:2px}
.status{padding:14px;border-radius:15px;border:2px solid;margin-top:12px}.status-risk{background:#fff0f2;border-color:#f06a78;color:#9e2d3c}.status-ok{background:#eafaf4;border-color:#49b995;color:#176f58}.status-title{font:800 .86rem Manrope;margin-bottom:3px}
.stNumberInput input{border:1.7px solid #164e5b!important;border-radius:10px!important;background:#fff!important}.stSelectbox div[data-baseweb="select"]>div{border:1.7px solid #164e5b!important;border-radius:10px!important;background:#fff!important}.stPlotlyChart{border-radius:16px;overflow:hidden}
.footer{border-top:1px solid #d5ebed;margin-top:28px;padding-top:12px;color:#76959b;font-size:.63rem}
.home-spacer{height:4px}
@media(max-width:900px){.story{grid-template-columns:1fr}}@media(max-width:620px){.block-container{padding:12px 12px 35px!important}.brand{padding:10px}.latest{display:none}.hero{padding:34px 23px;min-height:300px}.hero h1{font-size:2.85rem}}
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


def geo_layout():
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


def make_latest_map(frame, risk_frame, view, height=575):
    fig = go.Figure()

    field = frame.copy()
    if len(field) > 12000:
        field = field.sample(12000, random_state=42)

    if view == "Potential bloom-risk screening":
        fig.add_trace(go.Scattergeo(
            lon=field["longitude"], lat=field["latitude"], mode="markers",
            name="Processed ocean field",
            marker=dict(size=3.1, color="#65b8d1", opacity=.34),
            hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Processed cell<extra></extra>",
        ))
        if not risk_frame.empty:
            fig.add_trace(go.Scattergeo(
                lon=risk_frame["longitude"], lat=risk_frame["latitude"], mode="markers",
                name="Potential bloom-risk cell",
                marker=dict(size=7.5, color="#ef4f5e", opacity=.93, line=dict(width=.5, color="#fff")),
                customdata=np.c_[
                    risk_frame["chla"].fillna(np.nan),
                    risk_frame.get("risk_probability", pd.Series(np.nan, index=risk_frame.index)).fillna(np.nan),
                ],
                hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Chl-a %{customdata[0]:.4f}<br>Risk probability %{customdata[1]:.1%}<extra></extra>",
            ))
    else:
        col = {
            "Current Chl-a": "chla",
            "Recent change": "chla_change",
            "Historical anomaly": "chla_anomaly",
        }[view]
        plot = frame.dropna(subset=[col]).copy() if col in frame.columns else pd.DataFrame()
        if len(plot) > 14000:
            plot = plot.sample(14000, random_state=42)
        if plot.empty:
            return None

        colorscale = "YlGnBu" if view == "Current Chl-a" else "RdBu_r"
        color_title = "Chl-a" if view == "Current Chl-a" else "Signal"
        fig.add_trace(go.Scattergeo(
            lon=plot["longitude"], lat=plot["latitude"], mode="markers", name=view,
            marker=dict(
                size=4.2, color=plot[col], colorscale=colorscale, opacity=.76,
                colorbar=dict(title=color_title, thickness=13, tickfont=dict(color="#244f58"), titlefont=dict(color="#244f58")),
            ),
            customdata=np.c_[plot["chla"], plot[col]],
            hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Chl-a %{customdata[0]:.4f}<br>Signal %{customdata[1]:.4f}<extra></extra>",
        ))
        if not risk_frame.empty:
            fig.add_trace(go.Scattergeo(
                lon=risk_frame["longitude"], lat=risk_frame["latitude"], mode="markers",
                name="Potential-risk cells",
                marker=dict(size=6.5, color="#ef4f5e", opacity=.88),
                hovertemplate="Potential bloom-risk screening cell<extra></extra>",
            ))

    fig.update_geos(**geo_layout())
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#e6f8fa",
        plot_bgcolor="#e6f8fa",
        font=dict(family="DM Sans", color="#174b56"),
        legend=dict(
            orientation="h", y=.01, x=.02,
            bgcolor="rgba(255,255,255,.82)", bordercolor="#b9dfe2", borderwidth=1,
            font=dict(color="#174b56"),
        ),
        hoverlabel=dict(bgcolor="#103f49", font_color="#fff"),
    )
    return fig


def make_history_map(selected_history, selected_date):
    h = selected_history.copy()
    fig = go.Figure()

    # Blue processed field, green concentration zones, red screening cells.
    if len(h) > 18000:
        h = h.sample(18000, random_state=42)

    fig.add_trace(go.Scattergeo(
        lon=h["lon_bin"], lat=h["lat_bin"], mode="markers",
        name="Processed ocean field",
        marker=dict(size=5.5, color="#4d9fc0", opacity=.40),
        customdata=np.c_[h["chla"], h["risk"]],
        hovertemplate="Lat %{lat:.1f}°<br>Lon %{lon:.1f}°<br>Chl-a %{customdata[0]:.4f}<extra></extra>",
    ))

    concentration = h[h["chla"] >= h["chla"].quantile(.90)].copy()
    if not concentration.empty:
        if len(concentration) > 4000:
            concentration = concentration.sample(4000, random_state=42)
        fig.add_trace(go.Scattergeo(
            lon=concentration["lon_bin"], lat=concentration["lat_bin"], mode="markers",
            name="Higher concentration",
            marker=dict(size=7, color="#2eaa78", opacity=.48),
            hovertemplate="Higher Chl-a concentration<extra></extra>",
        ))

    flagged = h[h["risk"] == 1]
    if not flagged.empty:
        fig.add_trace(go.Scattergeo(
            lon=flagged["lon_bin"], lat=flagged["lat_bin"], mode="markers",
            name="Potential bloom-risk cell",
            marker=dict(size=9, color="#ef4f5e", opacity=.93, line=dict(width=.4, color="#fff")),
            hovertemplate="Potential bloom-risk screening cell<extra></extra>",
        ))

    fig.update_geos(**geo_layout())
    fig.update_layout(
        height=575,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#e6f8fa", plot_bgcolor="#e6f8fa",
        font=dict(family="DM Sans", color="#174b56"),
        legend=dict(
            orientation="h", y=.01, x=.02,
            bgcolor="rgba(255,255,255,.84)", bordercolor="#b9dfe2", borderwidth=1,
            font=dict(color="#174b56"),
        ),
        hoverlabel=dict(bgcolor="#103f49", font_color="#fff"),
        title=dict(text=fmt_date(selected_date), font=dict(color="#174b56", size=14), x=.02, xanchor="left"),
    )
    return fig


def nearest_valid(lat, lon):
    valid = latest[
        latest["chla"].notna()
        & np.isfinite(latest["latitude"])
        & np.isfinite(latest["longitude"])
        & (latest["chla"] > 0)
        & latest["latitude"].between(LAT_MIN, LAT_MAX)
        & latest["longitude"].between(LON_MIN, LON_MAX)
    ].copy()
    if valid.empty:
        return None, None

    # Normalized distance avoids longitude/latitude scale dominating the lookup.
    d2 = ((valid["latitude"] - lat) / 70.0) ** 2 + ((valid["longitude"] - lon) / 100.0) ** 2
    idx = d2.idxmin()
    row = valid.loc[idx]
    angular_distance = float(np.hypot(row["latitude"] - lat, row["longitude"] - lon))
    return row, angular_distance

# ============================================================
# HEADER + NAVIGATION
# ============================================================

st.markdown(
    f'''<div class="brand"><div class="brand-left"><div class="logo">≈</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div></div></div><div class="latest">Latest processed field<br><b>{fmt_date(latest_date)}</b></div></div>''',
    unsafe_allow_html=True,
)

if st.session_state.get("page") not in {"home", "map", "location", "insights", "data"}:
    st.session_state.page = "home"

pages = ["home", "map", "location", "insights", "data"]
labels = {"home":"⌂ Home", "map":"🌊 Risk Map", "location":"📍 Location", "insights":"📊 Insights", "data":"⇩ Data"}

st.markdown('<div class="nav">', unsafe_allow_html=True)
nav_cols = st.columns(5)
for col, page in zip(nav_cols, pages):
    with col:
        if st.button(
            labels[page],
            key=f"nav_{page}",
            width="stretch",
            type="primary" if st.session_state.page == page else "secondary",
        ):
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

    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)
    left,right = st.columns([1,1], gap="large")
    with left:
        st.markdown(
            '<div class="card"><h3>From satellite observation to investigation support</h3><p>BloomDetect uses satellite-derived chlorophyll-a and temporal context to screen for unusual patterns. The output is a <b>potential bloom-risk signal</b>, not a confirmed harmful algal bloom.</p><div class="note"><b>Important:</b> High chlorophyll-a alone does not prove a harmful algal bloom. Species, toxins and ecological impacts require additional evidence and field validation.</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        if IMAGE_PATH.exists():
            st.image(
                str(IMAGE_PATH),
                width="stretch",
                caption="How satellite ocean-colour observations support bloom-risk investigation",
            )
        else:
            st.info("Project illustration not found. The dashboard itself is still fully usable.")

    # No duplicate feature cards here. The top navigation already provides the four views.
    st.markdown(
        '<div class="note" style="margin-top:18px"><b>Start with Risk Map</b> to inspect the spatial screening field. Use <b>Location</b> for one coordinate, <b>Insights</b> for current patterns, and <b>Data</b> for project files and interpretation.</div>',
        unsafe_allow_html=True,
    )
    add_footer()

# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":
    if history_available:
        heading(
            "01 · SPATIAL INTELLIGENCE",
            "See how the field changes.",
            "Move through the available observation dates. The latest date uses the final model screening output; earlier dates use the compact historical Chl-a screening layer.",
        )

        hist_dates = sorted(history["date"].dropna().dt.normalize().unique())
        min_hist = pd.Timestamp(hist_dates[0]).date()
        max_hist = pd.Timestamp(hist_dates[-1]).date()
        default_idx = len(hist_dates) - 1
        selected_date = st.select_slider(
            "Map timeline",
            options=[pd.Timestamp(x).date() for x in hist_dates],
            value=pd.Timestamp(hist_dates[default_idx]).date(),
            format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"),
            key="history_date",
        )
        selected_ts = pd.Timestamp(selected_date)

        if selected_ts.normalize() == latest_date.normalize():
            selected = history[history["date"].dt.normalize().eq(selected_ts.normalize())].copy()
            processed_count = len(selected)
            risk_count = int(selected["risk"].sum())
            risk_share = risk_count / processed_count * 100 if processed_count else 0
            map_fig = make_history_map(selected, selected_ts)
            source_note = "latest field + final model screening"
        else:
            selected = history[history["date"].dt.normalize().eq(selected_ts.normalize())].copy()
            processed_count = len(selected)
            risk_count = int(selected["risk"].sum())
            risk_share = risk_count / processed_count * 100 if processed_count else 0
            map_fig = make_history_map(selected, selected_ts)
            source_note = "compact historical Chl-a screening layer"

        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Selected date", fmt_date(selected_ts), "observation")
        with c2: metric_card("Processed cells", f"{processed_count:,}", "selected field")
        with c3: metric_card("Potential-risk", f"{risk_count:,}", source_note)
        with c4: metric_card("Risk share", f"{risk_share:.2f}%", "selected field")

        st.markdown('<div class="map-wrap"><div class="map-head"><b>Indian Ocean · Arabian Sea · Bay of Bengal</b><span>Blue = processed field · green = higher concentration · red = screening flags</span></div>', unsafe_allow_html=True)
        st.plotly_chart(map_fig, width="stretch", config={"displaylogo":False,"scrollZoom":True,"responsive":True})
        st.markdown('<div class="legend"><span><i class="dot" style="background:#2584a8"></i>Processed field</span><span><i class="dot" style="background:#2eaa78"></i>Higher concentration</span><span><i class="dot" style="background:#ef4f5e"></i>Potential-risk cell</span></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="note"><b>Important:</b> earlier dates are a compact historical screening layer derived from aggregated Chl-a history. These flags are screening proxies, not additional model predictions.</div>', unsafe_allow_html=True)
    else:
        heading(
            "01 · SPATIAL INTELLIGENCE",
            "Read the latest field clearly.",
            "The compact history file is not available in this deployment, so the latest processed satellite field remains fully available. Add the small bloomdetect_history.csv file later to enable the date timeline.",
        )

        map_view = st.selectbox(
            "Map detail",
            ["Potential bloom-risk screening", "Current Chl-a", "Recent change", "Historical anomaly"],
            key="map_view_no_history",
        )
        mean_chla = latest["chla"].mean()
        mean_change = latest["chla_change"].mean() if "chla_change" in latest else np.nan
        mean_anomaly = latest["chla_anomaly"].mean() if "chla_anomaly" in latest else np.nan
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("Processed cells", f"{len(latest):,}", "latest field")
        with c2: metric_card("Mean Chl-a", fmt_num(mean_chla), "current field")
        with c3: metric_card("Mean recent change", fmt_num(mean_change), "vs previous observation")
        with c4: metric_card("Mean anomaly", fmt_num(mean_anomaly), "vs historical baseline")

        fig = make_latest_map(latest, latest_risk, map_view)
        if fig is not None:
            st.markdown('<div class="map-wrap"><div class="map-head"><b>Indian Ocean · Arabian Sea · Bay of Bengal</b><span>Blue = processed field · red = potential-risk screening</span></div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch", config={"displaylogo":False,"scrollZoom":True,"responsive":True})
            st.markdown('<div class="legend"><span><i class="dot" style="background:#2584a8"></i>Processed field</span><span><i class="dot" style="background:#ef4f5e"></i>Potential-risk cell</span><span>Hover points for values · scroll to zoom</span></div></div>', unsafe_allow_html=True)

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
    heading(
        "02 · LOCATION INTELLIGENCE",
        "Check one coordinate with clean evidence.",
        "Enter a latitude and longitude. BloomDetect separates your input from the nearest valid processed ocean cell, so land and zero-value observations are not presented as meaningful ocean results.",
    )

    left,right = st.columns([.75,1.25], gap="large")
    with left:
        st.markdown('<div class="card"><h3>Your input</h3><p>Enter decimal degrees. Example: 17.38, 78.49.</p></div>', unsafe_allow_html=True)
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=17.38, step=0.01, format="%.4f", key="input_lat")
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.49, step=0.01, format="%.4f", key="input_lon")
        check = st.button("🔎 Check location", width="stretch", type="primary")

    if check:
        if not (LAT_MIN <= float(lat) <= LAT_MAX and LON_MIN <= float(lon) <= LON_MAX):
            with right:
                st.warning(f"The coordinate is outside the BloomDetect study window ({LAT_MIN}° to {LAT_MAX}° latitude, {LON_MIN}° to {LON_MAX}° longitude).")
        else:
            row, distance_deg = nearest_valid(float(lat), float(lon))
            if row is None:
                with right:
                    st.error("No valid processed ocean cells are available for this lookup.")
            else:
                flagged = bool(row["risk_flag"])
                with right:
                    st.markdown(
                        f'<div class="card"><div class="kicker">NEAREST VALID PROCESSED OCEAN CELL</div><h3>{row["latitude"]:.4f}° · {row["longitude"]:.4f}°</h3><p>Your input: {float(lat):.4f}° · {float(lon):.4f}°<br>Approx. angular separation: {distance_deg:.2f}°</p></div>',
                        unsafe_allow_html=True,
                    )

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
            st.markdown('<div class="card"><h3>Nearest processed ocean cell</h3><p>Press <b>Check location</b> to evaluate the coordinate. Land and zero-value cells are excluded from the result.</p><div class="note">Your input stays separate from the actual processed satellite cell. The dashboard only reports a valid ocean observation after the lookup is run.</div></div>', unsafe_allow_html=True)

    add_footer()

# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":
    heading(
        "03 · SIGNAL INSIGHTS",
        "Turn the field into a few useful signals.",
        "This page combines the latest screening output with Chl-a change and anomaly information already stored in the processed dataset. It is descriptive analysis, not a second model.",
    )

    risk_work = latest_risk.copy()
    if not risk_work.empty:
        risk_work["lat_zone"] = np.floor(risk_work["latitude"] / 5) * 5
        risk_work["lon_zone"] = np.floor(risk_work["longitude"] / 5) * 5
        zone = (
            risk_work.groupby(["lat_zone","lon_zone"], as_index=False)
            .size().rename(columns={"size":"risk_cells"})
        )
        zone["label"] = zone.apply(
            lambda r: f"{int(r.lat_zone)}° to {int(r.lat_zone+5)}° · {int(r.lon_zone)}° to {int(r.lon_zone+5)}°",
            axis=1,
        )
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
            fig.update_layout(
                height=390,
                margin=dict(l=18,r=15,t=10,b=55),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#315b63"),
                xaxis=dict(
                    title="Potential-risk cells",
                    title_font=dict(color="#315b63", size=13),
                    tickfont=dict(color="#315b63", size=11),
                    gridcolor="#cfe4e6",
                    zerolinecolor="#a8c9cd",
                ),
                yaxis=dict(
                    title="",
                    tickfont=dict(color="#315b63", size=10),
                    automargin=True,
                ),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch", config={"displaylogo":False})
        else:
            st.info("No potential-risk cells are present in the latest field.")

    with b:
        st.markdown('<div class="card"><h3>Chl-a distribution</h3><p>Distribution of valid latest-field chlorophyll-a observations.</p></div>', unsafe_allow_html=True)
        valid_chla = latest.loc[latest["chla"] > 0, "chla"].dropna()
        if not valid_chla.empty:
            fig = px.histogram(valid_chla, nbins=40)
            fig.update_traces(marker_color="#2584a8")
            fig.update_layout(
                height=390,
                margin=dict(l=45,r=15,t=10,b=55),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans", color="#315b63"),
                xaxis=dict(
                    title="Chlorophyll-a",
                    title_font=dict(color="#315b63", size=13),
                    tickfont=dict(color="#315b63", size=11),
                    gridcolor="#cfe4e6",
                ),
                yaxis=dict(
                    title="Cells",
                    title_font=dict(color="#315b63", size=13),
                    tickfont=dict(color="#315b63", size=11),
                    gridcolor="#cfe4e6",
                ),
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch", config={"displaylogo":False})

    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card"><h3>Recent-change profile</h3><p>Selected distribution points for Chl-a change relative to the previous observation.</p></div>', unsafe_allow_html=True)
        change = latest["chla_change"].dropna() if "chla_change" in latest else pd.Series(dtype=float)
        if not change.empty:
            q = change.quantile([.01,.25,.5,.75,.99])
            labels_q=["1%","25%","Median","75%","99%"]
            fig=go.Figure(go.Bar(x=labels_q,y=q.values,marker_color="#2584a8"))
            fig.update_layout(
                height=290, margin=dict(l=45,r=10,t=15,b=50),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans",color="#315b63"),
                yaxis=dict(title="Chl-a change", title_font=dict(color="#315b63"), tickfont=dict(color="#315b63"), gridcolor="#cfe4e6"),
                xaxis=dict(title="Distribution point", title_font=dict(color="#315b63"), tickfont=dict(color="#315b63")),
            )
            st.plotly_chart(fig,width="stretch",config={"displaylogo":False})

    with c2:
        st.markdown('<div class="card"><h3>Highest current-risk cells</h3><p>Locations with the largest stored model screening scores.</p></div>', unsafe_allow_html=True)
        if not latest_risk.empty and "risk_probability" in latest_risk.columns:
            top = latest_risk.sort_values("risk_probability", ascending=False).head(8).copy()
            top["Location"] = top.apply(lambda r: f"{r.latitude:.2f}°, {r.longitude:.2f}°", axis=1)
            table = top[["Location","chla","risk_probability"]].copy()
            table.columns=["Location","Chl-a","Risk probability"]
            table["Chl-a"] = table["Chl-a"].map(lambda x: fmt_num(x,4))
            table["Risk probability"] = table["Risk probability"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "Unavailable")
            st.dataframe(table, width="stretch", hide_index=True)
        else:
            st.info("Risk-probability values are not available for the latest field.")

    st.markdown('<div class="note"><b>Scientific caution:</b> these insights describe satellite-derived signals and the project screening proxy. They do not identify algal species, toxins or confirmed harmful blooms.</div>', unsafe_allow_html=True)
    add_footer()

# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":
    heading(
        "04 · DATA",
        "Source, scope and downloadable outputs.",
        "Project data and interpretation notes live here so the other pages can stay focused and avoid repeating the same information.",
    )

    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card"><h3>Source product</h3><p><b>EOS-06 / Oceansat-3 OCM-3</b><br>Level-4 Analysed Chlorophyll Product · E06OCM_L4_AC<br>NetCDF satellite-derived ocean-colour product.</p><div class="note"><b>Study window:</b> approximately 20°E–120°E and 40°S–30°N for the dashboard view.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><h3>What the system screens</h3><p>BloomDetect combines current Chl-a with temporal context such as the previous observation, historical baseline, recent mean and recent maximum. The classifier produces a potential bloom-risk screening signal.</p><div class="note"><b>Important:</b> the dataset does not contain confirmed HAB species/toxin labels. Therefore the output is potential bloom-risk screening, not confirmed HAB detection.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="kicker">LATEST OUTPUT</div><h2>Download the processed field.</h2></div>', unsafe_allow_html=True)
    csv_bytes = latest.to_csv(index=False).encode("utf-8")
    risk_bytes = latest_risk.to_csv(index=False).encode("utf-8")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⇩ Download latest field CSV", data=csv_bytes, file_name="bloomdetect_latest_field.csv", mime="text/csv", width="stretch")
    with c2:
        st.download_button("⇩ Download risk cells CSV", data=risk_bytes, file_name="bloomdetect_potential_risk_cells.csv", mime="text/csv", width="stretch")
    with c3:
        st.markdown(f'<div class="card" style="min-height:41px;padding:10px 13px"><p><b>Latest field:</b> {fmt_date(latest_date)}<br><b>Rows:</b> {len(latest):,}<br><b>Risk cells:</b> {len(latest_risk):,}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:16px"><h3>Interpretation rule</h3><p>High chlorophyll-a alone does not establish a harmful algal bloom. Satellite observations support spatial and temporal screening, while species identification, toxin confirmation and ecological impact assessment require additional evidence and field validation.</p></div>', unsafe_allow_html=True)

    if history_available:
        st.markdown('<div class="card" style="margin-top:14px"><h3>Compact history</h3><p>The optional history file is loaded for the Risk Map timeline. It contains aggregated 1° screening cells rather than the full raw satellite dataset.</p></div>', unsafe_allow_html=True)

    add_footer()
