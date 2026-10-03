import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as pgo
from pathlib import Path

st.set_page_config(page_title="BloomDetect AI | Coastal & Ocean Intelligence", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "latest_bloom_risk_predictions.csv"
IMAGE_PATH = BASE / "bloomdetect_bloom_process.png"

# ----------------------------- THEME -----------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--navy:#073B4C;--aqua:#0EA5B7;--mint:#E9F8F7;--ink:#14343D;--muted:#607B83;--line:#A9D8DC;--green:#2F9E72;--red:#D64545;--blue:#6DBFD2;--white:#fff;}
html,body,[class*="css"]{font-family:"DM Sans",sans-serif;color:var(--ink);}
.stApp{background:linear-gradient(180deg,#F8FEFD 0%,#EFF9F8 100%);}
.block-container{max-width:1180px;padding-top:1rem!important;padding-bottom:3rem!important;}
#MainMenu,footer,header{visibility:hidden;}
h1,h2,h3{font-family:"Space Grotesk",sans-serif!important;color:var(--navy)!important;}
.hero{position:relative;overflow:hidden;min-height:320px;border-radius:26px;border:1px solid #7FCBD1;background:linear-gradient(105deg,rgba(4,48,62,.92),rgba(6,83,101,.48)),url('https://images.unsplash.com/photo-1530053969600-caed2596d242?auto=format&fit=crop&w=1800&q=80') center/cover no-repeat;padding:42px;margin:6px 0 22px;color:#fff;box-shadow:0 12px 32px rgba(7,59,76,.10);}
.hero:after{content:"";position:absolute;left:-10%;right:-10%;bottom:-68px;height:130px;background:rgba(91,224,222,.22);border-radius:50%;animation:wave 5s ease-in-out infinite;}
.hero:before{content:"";position:absolute;left:-5%;right:-5%;bottom:-45px;height:90px;border-top:2px solid rgba(255,255,255,.16);border-radius:50%;animation:wave2 6s ease-in-out infinite;}
@keyframes wave{0%,100%{transform:translateX(-2%) scaleY(1)}50%{transform:translateX(3%) scaleY(1.28)}}
@keyframes wave2{0%,100%{transform:translateX(3%)}50%{transform:translateX(-3%)}}
.hero-content{position:relative;z-index:3;max-width:760px;}
.hero h1{color:#fff!important;font-size:2.65rem!important;line-height:1.05;margin:18px 0 12px!important;}
.hero p{color:rgba(255,255,255,.94);font-size:1rem;line-height:1.65;margin:0;}
.pills{display:flex;gap:8px;flex-wrap:wrap;}
.pill{padding:7px 11px;border-radius:999px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.30);font-size:.72rem;font-weight:700;color:#fff;}
.brand{background:#fff;border:1px solid var(--line);border-radius:20px;padding:14px 18px;display:flex;justify-content:space-between;align-items:center;box-shadow:0 8px 25px rgba(7,59,76,.06);margin-bottom:10px;}
.brand-left{display:flex;align-items:center;gap:12px}.logo{width:45px;height:45px;border-radius:14px;display:grid;place-items:center;background:linear-gradient(145deg,var(--navy),var(--aqua));color:#fff;font-size:21px}.brand-name{font:700 1.15rem "Space Grotesk",sans-serif;color:var(--navy)}.brand-sub,.brand-date{font-size:.72rem;color:var(--muted)}.brand-date{text-align:right}
.navbar{margin:0 0 24px}.stButton>button{width:100%;min-height:43px;border:1.6px solid var(--navy);border-radius:12px;background:#fff;color:var(--navy);font-weight:700;font-size:.82rem;transition:.18s}.stButton>button:hover{background:var(--mint);border-color:var(--aqua);transform:translateY(-1px)}.nav-active .stButton>button{background:linear-gradient(135deg,var(--navy),var(--aqua));color:#fff;border-color:var(--navy)}
.kicker{font-size:.70rem;color:var(--aqua);font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:5px}.section-title{font:700 2rem "Space Grotesk",sans-serif;color:var(--navy);letter-spacing:-.03em;margin-bottom:7px}.section-copy{color:var(--muted);font-size:.88rem;line-height:1.55;margin-bottom:15px}
.card,.metric,.download,.image-card,.note{background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:17px;box-shadow:0 7px 22px rgba(7,59,76,.05)}.card{padding:18px;height:100%}.card-title{font:700 1rem "Space Grotesk",sans-serif;color:var(--navy);margin-bottom:6px}.card-text{font-size:.82rem;color:var(--muted);line-height:1.55}.metric{padding:15px 17px;min-height:108px}.metric-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;font-weight:800;color:var(--muted)}.metric-value{font:700 1.42rem "Space Grotesk",sans-serif;color:var(--navy);margin-top:7px}.metric-note{font-size:.70rem;color:var(--muted);margin-top:3px}.step{background:#fff;border:1.6px solid var(--navy);border-radius:16px;padding:16px;min-height:145px;box-shadow:0 6px 18px rgba(7,59,76,.05)}.step-no{width:31px;height:31px;border-radius:50%;display:grid;place-items:center;background:var(--navy);color:#fff;font-weight:700;font-size:.76rem;margin-bottom:9px}.step h3{font-size:.95rem!important;margin:0 0 6px!important}.step p{font-size:.77rem;color:var(--muted);line-height:1.5;margin:0}.notice{border-left:4px solid var(--aqua);background:#E9F8F7;border-radius:0 13px 13px 0;padding:13px 15px;font-size:.79rem;line-height:1.55}.status{padding:16px 18px;border-radius:16px;border:1.5px solid}.ok{background:#EAF8F0;border-color:#82CDAE;color:#1D6F4E}.bad{background:#FFF0F0;border-color:#E6A0A0;color:#9B3030}.status-title{font:700 1rem "Space Grotesk",sans-serif}.status-copy{font-size:.78rem;margin-top:4px;line-height:1.45}.image-card{padding:8px;overflow:hidden}.image-card img{display:block;width:100%;border-radius:12px}.download{padding:18px;border:1.5px solid var(--navy);background:linear-gradient(135deg,#fff,#ECFAF8);min-height:128px}.download h3{margin:0 0 6px!important;font-size:1rem!important}.download p{font-size:.78rem;color:var(--muted);line-height:1.45;margin:0 0 12px}.map-head{display:flex;justify-content:space-between;gap:12px;align-items:center;background:#fff;border:1px solid var(--line);border-radius:15px 15px 0 0;padding:12px 16px;font-size:.78rem;color:var(--muted)}.map-head b{color:var(--navy);font-size:.9rem}.legend{display:flex;gap:18px;flex-wrap:wrap;align-items:center;background:#fff;border:1px solid var(--line);border-radius:0 0 15px 15px;padding:12px 16px;font-size:.77rem;color:var(--muted)}.sw{width:13px;height:13px;border-radius:4px;display:inline-block;margin-right:5px;vertical-align:-2px}.blue{background:#73BED1}.green{background:#43B581}.red{background:#D64545}.spacer{height:18px}.small{font-size:.72rem;color:var(--muted)}
.stNumberInput input,.stTextInput input,.stSelectbox [data-baseweb="select"]{border:1.3px solid #8EBEC4!important;border-radius:10px!important}.stDownloadButton>button{width:100%;border:1.5px solid var(--navy);border-radius:11px;background:var(--navy);color:#fff;font-weight:700}.stDownloadButton>button:hover{background:var(--aqua);border-color:var(--aqua)}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:13px;overflow:hidden}.footer{border-top:1px solid var(--line);margin-top:34px;padding-top:15px;text-align:center;font-size:.70rem;color:var(--muted)}
@media(max-width:800px){.block-container{padding-left:1rem!important;padding-right:1rem!important}.hero{padding:28px;min-height:390px}.hero h1{font-size:2.05rem!important}.brand{align-items:flex-start;flex-direction:column}.brand-date{text-align:left}.map-head{align-items:flex-start;flex-direction:column}}
</style>
""", unsafe_allow_html=True)

# ----------------------------- DATA -----------------------------
def clean_cols(df):
    df=df.copy(); df.columns=[str(c).strip().lower().replace(" ","_").replace("-","_") for c in df.columns]; return df

@st.cache_data(show_spinner=False)
def load_data(path):
    df=clean_cols(pd.read_csv(path))
    aliases={"date":["date","datetime","time","observation_date"],"latitude":["latitude","lat"],"longitude":["longitude","lon","lng"],"chla":["chla","chlorophyll_a","chlorophyll"],"risk":["risk","prediction","predicted_risk","risk_label","bloom_risk"],"probability":["risk_probability","probability","risk_prob","predicted_probability"] ,"previous_chla":["previous_chla","prev_chla"],"historical_baseline":["historical_baseline","baseline"]}
    ren={}
    for std,names in aliases.items():
        for n in names:
            if n in df.columns: ren[n]=std; break
    df=df.rename(columns=ren)
    need=["latitude","longitude","chla"]
    missing=[c for c in need if c not in df.columns]
    if missing: raise ValueError("CSV missing required columns: "+", ".join(missing))
    if "date" not in df.columns: df["date"]=pd.Timestamp("2026-03-30")
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    for c in ["latitude","longitude","chla","probability","previous_chla","historical_baseline"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
    if "risk" not in df.columns: df["risk"]=0
    elif df["risk"].dtype==object:
        s=df["risk"].astype(str).str.lower().str.strip(); df["risk"]=s.str.contains("risk|flag|potential|bloom|yes|true").astype(int)
    else: df["risk"]=(pd.to_numeric(df["risk"],errors="coerce").fillna(0)>0).astype(int)
    if "probability" not in df.columns: df["probability"]=df["risk"].astype(float)
    else:
        if df["probability"].max(skipna=True)>1: df["probability"]/=100
        df["probability"]=df["probability"].clip(0,1).fillna(0)
    df["risk_label"]=np.where(df["risk"].eq(1),"Potential Bloom Risk","Normal")
    df=df.dropna(subset=["date","latitude","longitude","chla"]).sort_values(["date","latitude","longitude"]).reset_index(drop=True)
    return df

if not CSV_PATH.exists(): st.error("latest_bloom_risk_predictions.csv is missing beside app.py."); st.stop()
try: df=load_data(str(CSV_PATH))
except Exception as e: st.error("BloomDetect AI could not load the dataset."); st.code(str(e)); st.stop()

dates=sorted(df["date"].unique()); latest_date=dates[-1]; latest=df[df["date"].eq(latest_date)].copy(); latest_risk=int(latest["risk"].sum()); latest_share=(latest_risk/len(latest)*100) if len(latest) else 0

# ----------------------------- HELPERS -----------------------------
def nav(label,page,icon):
    active=st.session_state.page==page
    if active: st.markdown('<div class="nav-active">',unsafe_allow_html=True)
    if st.button(f"{icon} {label}",key="nav_"+page): st.session_state.page=page; st.rerun()
    if active: st.markdown('</div>',unsafe_allow_html=True)

def metric(label,value,note): st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
def card(title,text,icon=""): st.markdown(f'<div class="card"><div class="card-title">{icon} {title}</div><div class="card-text">{text}</div></div>',unsafe_allow_html=True)
def section(kicker,title,copy=""):
    st.markdown(f'<div class="kicker">{kicker}</div><div class="section-title">{title}</div>{("<div class=\"section-copy\">"+copy+"</div>") if copy else ""}',unsafe_allow_html=True)
def pct(x): return f"{float(x)*100:.1f}%"
def fmt_date(x): return pd.Timestamp(x).strftime("%d %b %Y")

def make_map(field):
    fig=pgo.Figure()
    # One GeoJSON/Choropleth layer keeps the full study area fast and filled blue.
    w=field[["latitude","longitude","chla"]].copy()
    step=1.0
    w["lat0"]=np.floor(w["latitude"]/step)*step
    w["lon0"]=np.floor(w["longitude"]/step)*step
    grid=w.groupby(["lat0","lon0"],as_index=False).agg(chla=("chla","mean"))
    features=[]
    values=[]
    for idx,r in grid.iterrows():
        lat=float(r["lat0"]); lon=float(r["lon0"])
        features.append({"type":"Feature","properties":{"cell_id":int(idx)},"geometry":{"type":"Polygon","coordinates":[[[lon,lat],[lon+step,lat],[lon+step,lat+step],[lon,lat+step],[lon,lat]]]}})
        values.append(float(r["chla"]))
    geojson={"type":"FeatureCollection","features":features}
    fig.add_trace(pgo.Choropleth(
        geojson=geojson,
        locations=list(range(len(values))),
        z=values,
        featureidkey="properties.cell_id",
        colorscale=[[0,"#E7F6F8"],[0.35,"#BFE5EC"],[0.7,"#91D1DE"],[1,"#55AFC5"]],
        marker_line_width=0,
        showscale=False,
        name="Processed ocean field",
        hovertemplate="Processed ocean field<br>Mean Chl-a: %{z:.4f}<extra></extra>",
    ))
    risk=field[field["risk"].eq(1)].copy()
    if not risk.empty:
        risk["lat_zone"]=np.floor(risk["latitude"]/2)*2
        risk["lon_zone"]=np.floor(risk["longitude"]/2)*2
        zones=risk.groupby(["lat_zone","lon_zone"],as_index=False).agg(count=("risk","size"),mean_chla=("chla","mean"))
        fig.add_trace(pgo.Scattergeo(
            lon=zones["lon_zone"]+1,lat=zones["lat_zone"]+1,mode="markers",
            marker=dict(size=np.clip(12+zones["count"]*1.0,14,34),color="#43B581",opacity=.42,line=dict(color="#247C58",width=1)),
            name="Risk concentration",customdata=zones[["count","mean_chla"]].to_numpy(),
            hovertemplate="Risk concentration<br>Flagged cells: %{customdata[0]}<br>Mean Chl-a: %{customdata[1]:.4f}<extra></extra>"
        ))
        fig.add_trace(pgo.Scattergeo(
            lon=risk["longitude"],lat=risk["latitude"],mode="markers",
            marker=dict(size=6,symbol="square",color="#D64545",opacity=.92,line=dict(color="#8E2323",width=.5)),
            name="Potential bloom-risk cell",customdata=risk[["chla","probability"]].to_numpy(),
            hovertemplate="Potential bloom-risk cell<br>Lat: %{lat:.4f}°<br>Lon: %{lon:.4f}°<br>Chl-a: %{customdata[0]:.4f}<br>Risk probability: %{customdata[1]:.1%}<extra></extra>"
        ))
    else:
        fig.add_trace(pgo.Scattergeo(lon=[],lat=[],mode="markers",name="Potential bloom-risk cell"))
    fig.update_geos(
        projection_type="mercator",lataxis=dict(range=[-40,30]),lonaxis=dict(range=[20,120]),
        showland=True,landcolor="#DCE7E9",showocean=True,oceancolor="#D9F2F4",
        showcountries=True,countrycolor="#78939A",coastlinecolor="#52757D",showlakes=True,lakecolor="#D9F2F4",
        bgcolor="rgba(0,0,0,0)"
    )
    fig.update_layout(
        height=610,margin=dict(l=0,r=0,t=8,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans",color="#14343D"),
        legend=dict(orientation="h",y=.01,x=.5,xanchor="center",bgcolor="rgba(255,255,255,.94)",bordercolor="#A9D8DC",borderwidth=1)
    )
    return fig

def zones_table(field):
    r=field[field.risk.eq(1)].copy()
    if r.empty: return pd.DataFrame()
    r["lat_zone"]=np.floor(r.latitude/2)*2; r["lon_zone"]=np.floor(r.longitude/2)*2
    z=r.groupby(["lat_zone","lon_zone"],as_index=False).agg(flagged_cells=("risk","size"),mean_chla=("chla","mean"),max_chla=("chla","max"),mean_probability=("probability","mean")).sort_values("flagged_cells",ascending=False).head(10)
    z["Zone"]=z.apply(lambda x:f"{x.lat_zone:.0f}° to {x.lat_zone+2:.0f}° · {x.lon_zone:.0f}° to {x.lon_zone+2:.0f}°E",axis=1)
    z["Mean Chl-a"]=z.mean_chla.round(4); z["Max Chl-a"]=z.max_chla.round(4); z["Risk probability"]=z.mean_probability.map(pct)
    return z[["Zone","flagged_cells","Mean Chl-a","Max Chl-a","Risk probability"]].rename(columns={"flagged_cells":"Flagged cells"})

# ----------------------------- NAV -----------------------------
if "page" not in st.session_state: st.session_state.page="Home"
st.markdown(f'<div class="brand"><div class="brand-left"><div class="logo">🌊</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Coastal &amp; Ocean Intelligence Platform · EOS-06 OCM-3</div></div></div><div class="brand-date">Latest processed field<br><b>{fmt_date(latest_date)}</b></div></div>',unsafe_allow_html=True)
cols=st.columns(5)
for c,(label,page,icon) in zip(cols,[("Home","Home","⌂"),("Risk Map","Risk Map","🌍"),("Location","Location","📍"),("Insights","Insights","📈"),("Data & Research","Data & Research","📊")]):
    with c: nav(label,page,icon)

# ----------------------------- HOME -----------------------------
if st.session_state.page=="Home":
    st.markdown(f'''<section class="hero"><div class="hero-content"><div class="pills"><span class="pill">🌊 Ocean colour</span><span class="pill">🛰️ EOS-06 OCM-3</span><span class="pill">🤖 Decision Tree screening</span><span class="pill">📍 Spatial intelligence</span></div><h1>Read the ocean signal before it becomes a bigger question.</h1><p>BloomDetect AI uses satellite-derived chlorophyll-a and a stored machine-learning screening result to highlight cells that deserve closer observation. It is an early-warning support tool, not a confirmation of harmful algal bloom occurrence.</p></div></section>''',unsafe_allow_html=True)
    section("WHAT IS BLOOMDETECT AI?","A map-first way to read the signal.","The platform starts with satellite-derived chlorophyll-a, adds temporal context where available, and turns the stored screening output into a practical spatial view.")
    left,right=st.columns([1,1],gap="large")
    with left:
        st.markdown('<div class="card"><div class="kicker">WHY THIS MATTERS</div><div class="card-title" style="font-size:1.15rem">Chlorophyll-a is a signal, not a verdict.</div><div class="card-text" style="font-size:.88rem">Changes in chlorophyll-a can help identify unusual ocean-colour patterns and areas worth investigating. BloomDetect uses this signal to create a <b>potential bloom-risk screening</b> result.</div><div class="notice" style="margin-top:15px"><b>Important:</b> high chlorophyll-a alone does not prove a harmful algal bloom. Species, toxins and ecological impacts need additional evidence and field validation.</div></div>',unsafe_allow_html=True)
    with right:
        if IMAGE_PATH.exists():
            st.markdown('<div class="image-card">',unsafe_allow_html=True); st.image(str(IMAGE_PATH),width="stretch"); st.markdown('</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">Satellite-to-bloom context</div><div class="card-text">Add <b>bloomdetect_bloom_process.png</b> beside app.py in GitHub to show the project illustration here.</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Latest cells",f"{len(latest):,}","processed grid cells")
    with c2: metric("Potential-risk cells",f"{latest_risk:,}","latest screening")
    with c3: metric("Risk share",f"{latest_share:.2f}%","latest field")
    with c4: metric("Study grid","0.25°","source product")
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    section("EXPLORE","Every feature has one job.","Use the navigation above. No mysterious maze of fifteen pages pretending to be a product.")
    a,b,c,d=st.columns(4)
    with a: card("Risk Map","See the full Indian Ocean study region, normal processed field, risk concentrations and individual flagged cells.","🌍")
    with b: card("Location Check","Enter coordinates and get the nearest processed cell with a clear flagged / not flagged result.","📍")
    with c: card("Insights","Read only the useful latest-field charts, with visible axis labels and values.","📈")
    with d: card("Data & Research","Keep provenance, coverage, scientific boundaries and downloads together.","📊")
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    section("HOW IT WORKS","Four clear steps")
    steps=[("01","Satellite observation","EOS-06 OCM-3 supplies the analysed chlorophyll-a field used in this project."),("02","Temporal context","Previous and historical values provide context where those columns are present in the packaged data."),("03","AI screening","The stored Decision Tree model output identifies cells for potential bloom-risk screening."),("04","Investigation","Flagged cells become priorities for closer observation and scientific validation.")]
    sc=st.columns(4)
    for col,(n,t,desc) in zip(sc,steps):
        with col: st.markdown(f'<div class="step"><div class="step-no">{n}</div><h3>{t}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div><div class="notice"><b>Scientific boundary:</b> BloomDetect reports <b>potential bloom risk</b>, not confirmed HAB detection. The current satellite dataset does not identify harmful species or toxins.</div>',unsafe_allow_html=True)

# ----------------------------- MAP -----------------------------
elif st.session_state.page=="Risk Map":
    section("02 · OCEAN & BLOOM RISK","Where is potential bloom risk showing up?","Blue is the processed ocean field. Green shows concentrations of flagged cells. Red marks individual cells screened as potential bloom risk.")
    if len(dates)>1: selected=st.select_slider("Observation date",options=dates,value=latest_date,format_func=fmt_date)
    else: selected=latest_date; st.markdown(f'<div class="small"><b>Observation date:</b> {fmt_date(selected)} · The current website CSV contains one processed date.</div>',unsafe_allow_html=True)
    field=df[df.date.eq(selected)].copy(); n=len(field); nr=int(field.risk.sum()); share=nr/n*100 if n else 0
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Study area","20°–120°E","40°S–30°N")
    with c2: metric("Processed cells",f"{n:,}","selected field")
    with c3: metric("Potential-risk cells",f"{nr:,}","selected field")
    with c4: metric("Risk share",f"{share:.2f}%","of processed cells")
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    st.markdown('<div class="map-head"><b>Indian Ocean study area</b><span>Spatial screening · not confirmed HAB extent</span></div>',unsafe_allow_html=True)
    st.plotly_chart(make_map(field),width="stretch",config={"displayModeBar":False})
    st.markdown('<div class="legend"><span><i class="sw blue"></i><b>Blue</b> processed ocean field</span><span><i class="sw green"></i><b>Green</b> risk concentration</span><span><i class="sw red"></i><b>Red</b> potential bloom-risk cell</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    section("HOTSPOT INTELLIGENCE","Where are the flags concentrating?","Nearby flagged cells are grouped into broad zones so the spatial pattern is easier to inspect. A zone is a concentration summary, not a new risk label.")
    z=zones_table(field)
    if z.empty: st.markdown('<div class="status ok"><div class="status-title">🟢 No potential-risk cells in this field</div><div class="status-copy">There are no stored flagged cells to group into concentrations.</div></div>',unsafe_allow_html=True)
    else:
        l,r=st.columns([1.05,.95],gap="large")
        with l:
            bar=pgo.Figure(pgo.Bar(x=z["Flagged cells"],y=z["Zone"],orientation="h",marker_color="#43B581",text=z["Flagged cells"],textposition="outside",cliponaxis=False))
            bar.update_layout(height=390,margin=dict(l=70,r=35,t=20,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",font=dict(family="DM Sans",color="#14343D"),xaxis=dict(title="Flagged cells",title_font=dict(color="#14343D",size=13),tickfont=dict(color="#14343D"),showgrid=True,gridcolor="#D7E8EA"),yaxis=dict(title="Screening zone",title_font=dict(color="#14343D",size=13),tickfont=dict(color="#14343D")),showlegend=False)
            st.plotly_chart(bar,width="stretch",config={"displayModeBar":False})
        with r: st.dataframe(z,width="stretch",hide_index=True,height=390)

# ----------------------------- LOCATION -----------------------------
elif st.session_state.page=="Location":
    section("03 · LOCATION INTELLIGENCE","Check one coordinate.","Enter latitude and longitude. BloomDetect returns the nearest processed satellite grid cell and its stored screening result.")
    left,right=st.columns([.9,1.1],gap="large")
    with left:
        st.markdown('<div class="card"><div class="card-title">Enter coordinates</div><div class="card-text">Use decimal degrees. Example: 17.3850, 78.4867.</div></div>',unsafe_allow_html=True)
        lat=st.number_input("Latitude",min_value=-40.0,max_value=30.0,value=17.3850,step=.01,format="%.4f",key="input_lat")
        lon=st.number_input("Longitude",min_value=20.0,max_value=120.0,value=78.4867,step=.01,format="%.4f",key="input_lon")
        check=st.button("🔎 Check location",width="stretch",key="check_btn")
    if "location_result" not in st.session_state: st.session_state.location_result=None
    if check:
        d2=(latest.latitude.to_numpy()-lat)**2+(latest.longitude.to_numpy()-lon)**2
        st.session_state.location_result=latest.iloc[int(np.argmin(d2))]
        st.session_state.location_input=(lat,lon)
    with right:
        row=st.session_state.location_result
        if row is None: st.markdown('<div class="card"><div class="card-title">Waiting for a coordinate</div><div class="card-text">Enter the location on the left and press <b>Check location</b>. The result will appear here.</div></div>',unsafe_allow_html=True)
        else:
            flagged=int(row.risk)==1
            if flagged: st.markdown('<div class="status bad"><div class="status-title">🔴 POTENTIAL BLOOM RISK FLAGGED</div><div class="status-copy">The nearest processed cell is flagged in the stored screening output.</div></div>',unsafe_allow_html=True)
            else: st.markdown('<div class="status ok"><div class="status-title">🟢 NOT FLAGGED</div><div class="status-copy">The nearest processed cell is not flagged in the stored screening output.</div></div>',unsafe_allow_html=True)
            inp=st.session_state.get("location_input",(lat,lon))
            st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
            a,b=st.columns(2)
            with a: metric("Your input",f"{inp[0]:.4f}°, {inp[1]:.4f}°","latitude, longitude")
            with b: metric("Nearest processed cell",f"{row.latitude:.4f}°, {row.longitude:.4f}°","actual lookup cell")
            a,b=st.columns(2)
            with a: metric("Chlorophyll-a",f"{row.chla:.4f}","satellite-derived")
            with b: metric("Risk probability",pct(row.probability),"stored model output")
            st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
            card("Interpretation","The coordinate you entered is a query point. The screening result belongs to the nearest processed satellite grid cell, not necessarily the exact coordinate.","📍")
    st.markdown('<div class="spacer"></div><div class="notice"><b>Important:</b> A flag is a screening result. It does not confirm a harmful algal bloom at the exact coordinate.</div>',unsafe_allow_html=True)

# ----------------------------- INSIGHTS -----------------------------
elif st.session_state.page=="Insights":
    section("04 · INSIGHTS","What stands out in the latest field?","Compact charts with readable labels. No decorative graphs whose only purpose is to make the page look employed.")
    risk=latest[latest.risk.eq(1)]; normal=latest[latest.risk.eq(0)]
    mx=latest.loc[latest.chla.idxmax()]
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Mean Chl-a",f"{latest.chla.mean():.4f}","latest field")
    with c2: metric("Median Chl-a",f"{latest.chla.median():.4f}","latest field")
    with c3: metric("Maximum Chl-a",f"{latest.chla.max():.4f}",f"{mx.latitude:.2f}°, {mx.longitude:.2f}°")
    with c4: metric("Risk share",f"{latest_share:.2f}%",f"{latest_risk:,} flagged cells")
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    a,b=st.columns(2,gap="large")
    with a:
        h=pgo.Figure(pgo.Histogram(x=latest.chla,nbinsx=42,marker_color="#4DAABD"))
        h.update_layout(height=360,margin=dict(l=75,r=25,t=25,b=70),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",font=dict(family="DM Sans",color="#14343D"),xaxis=dict(title="Chlorophyll-a",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D"),showgrid=True,gridcolor="#D7E8EA"),yaxis=dict(title="Processed cells",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D"),showgrid=True,gridcolor="#D7E8EA"),showlegend=False)
        st.plotly_chart(h,width="stretch",config={"displayModeBar":False})
    with b:
        comp=pgo.Figure(pgo.Bar(x=["Normal cells","Potential-risk cells"],y=[normal.chla.mean() if len(normal) else 0,risk.chla.mean() if len(risk) else 0],marker_color=["#73BED1","#D64545"],text=[f"{normal.chla.mean():.4f}" if len(normal) else "0",f"{risk.chla.mean():.4f}" if len(risk) else "0"],textposition="outside",cliponaxis=False))
        comp.update_layout(height=360,margin=dict(l=75,r=25,t=35,b=75),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",font=dict(family="DM Sans",color="#14343D"),xaxis=dict(title="Cell group",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D")),yaxis=dict(title="Mean chlorophyll-a",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D"),showgrid=True,gridcolor="#D7E8EA",rangemode="tozero"),showlegend=False)
        st.plotly_chart(comp,width="stretch",config={"displayModeBar":False})
    if len(risk):
        r=risk.copy(); r["band"]=np.floor(r.latitude/5)*5; band=r.groupby("band",as_index=False).size().rename(columns={"size":"Flagged cells"}).sort_values("Flagged cells",ascending=False).head(10); band["Latitude band"]=band.band.map(lambda x:f"{x:.0f}° to {x+5:.0f}°")
        st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
        bfig=pgo.Figure(pgo.Bar(x=band["Flagged cells"],y=band["Latitude band"],orientation="h",marker_color="#43B581",text=band["Flagged cells"],textposition="outside",cliponaxis=False))
        bfig.update_layout(height=350,margin=dict(l=75,r=50,t=25,b=60),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",font=dict(family="DM Sans",color="#14343D"),xaxis=dict(title="Flagged cells",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D"),showgrid=True,gridcolor="#D7E8EA"),yaxis=dict(title="Latitude band",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D")),showlegend=False)
        st.plotly_chart(bfig,width="stretch",config={"displayModeBar":False})
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    x,y,z=st.columns(3)
    with x: card("Spatial signal",f"{latest_risk:,} cells are flagged in the latest field, representing {latest_share:.2f}% of processed cells.","🌍")
    with y: card("Chl-a signal",f"The maximum stored chlorophyll-a is {latest.chla.max():.4f}. This is an ocean-colour signal, not proof of harmfulness.","🌿")
    with z: card("Interpretation", "Screening flags are investigation priorities and should be checked with additional environmental evidence and field validation.","🔬")
    if len(dates)>1:
        daily=df.groupby("date",as_index=False).agg(mean_chla=("chla","mean"),risk_cells=("risk","sum"));
        st.markdown('<div class="spacer"></div>',unsafe_allow_html=True); section("TIME SERIES","Mean chlorophyll-a across available dates","Shown only when the packaged CSV contains multiple observation dates.")
        tr=pgo.Figure(pgo.Scatter(x=daily.date,y=daily.mean_chla,mode="lines+markers",line=dict(color="#0EA5B7",width=2.5),marker=dict(size=5)))
        tr.update_layout(height=350,margin=dict(l=75,r=25,t=20,b=70),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.55)",font=dict(family="DM Sans",color="#14343D"),xaxis=dict(title="Observation date",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D")),yaxis=dict(title="Mean chlorophyll-a",title_font=dict(color="#14343D",size=14),tickfont=dict(color="#14343D"),showgrid=True,gridcolor="#D7E8EA"),showlegend=False)
        st.plotly_chart(tr,width="stretch",config={"displayModeBar":False})

# ----------------------------- DATA -----------------------------
elif st.session_state.page=="Data & Research":
    section("05 · DATA & RESEARCH","Complete dataset information","Source, coverage, variables, interpretation limits and downloads stay here so the other pages remain focused.")
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Source product","E06OCM_L4_AC","EOS-06 / Oceansat-3 OCM-3")
    with c2: metric("Variable","Chlorophyll-a","satellite-derived")
    with c3: metric("Grid","0.25°","source product")
    with c4: metric("Packaged rows",f"{len(df):,}","current website CSV")
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a: card("Dataset coverage",f"{fmt_date(dates[0])} to {fmt_date(dates[-1])} · {len(dates)} observation date(s) · {len(df):,} rows.","🛰️")
    with b: card("Study region","The map is focused on 20°E–120°E and 40°S–30°N for the Indian Ocean, Arabian Sea and Bay of Bengal interpretation region.","🌍")
    a,b=st.columns(2)
    with a: card("Stored AI output","The packaged CSV contains the screening result and risk probability used by the dashboard. It is screening support, not confirmed HAB ground truth.","🤖")
    with b: card("Scientific limitation","The current dataset has no confirmed harmful-species labels or toxin measurements. High chlorophyll-a alone does not prove a harmful algal bloom.","🔬")
    st.markdown('<div class="spacer"></div><div class="notice"><b>Interpretation boundary:</b> Satellite-derived chlorophyll-a can support ocean-colour and potential bloom-risk screening. It cannot independently identify a harmful species, toxin, exact ecological cause, or human-health impact.</div>',unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    section("DOWNLOADS","Research-ready files","The buttons below are intentionally prominent and equal in size.")
    csv_bytes=df.to_csv(index=False).encode("utf-8")
    summary=(f"BloomDetect AI - Stage 1 Data Summary\n\nSource product: EOS-06 / Oceansat-3 OCM-3 Level-4 Analysed Chlorophyll Product (E06OCM_L4_AC)\nVariable: Chlorophyll-a\nGrid: 0.25 degrees\nCoverage: {fmt_date(dates[0])} to {fmt_date(dates[-1])}\nRows: {len(df):,}\nLatest processed date: {fmt_date(latest_date)}\nLatest processed cells: {len(latest):,}\nLatest potential bloom-risk cells: {latest_risk:,}\nLatest risk share: {latest_share:.2f}%\nStudy region: 20E-120E, 40S-30N\n\nScientific boundary: potential bloom-risk screening only; high chlorophyll-a alone does not confirm a harmful algal bloom. Additional evidence and field validation are required.\n").encode("utf-8")
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="download"><h3>⬇ Latest screening dataset</h3><p>Processed observations and stored screening outputs used by the platform.</p></div>',unsafe_allow_html=True)
        st.download_button("Download CSV",csv_bytes,"bloomdetect_stage1_dataset.csv","text/csv",width="stretch",key="dl_csv")
    with b:
        st.markdown('<div class="download"><h3>⬇ Project data summary</h3><p>Compact provenance, coverage and interpretation notes for reports and reviews.</p></div>',unsafe_allow_html=True)
        st.download_button("Download summary",summary,"BloomDetect_AI_Data_Summary.txt","text/plain",width="stretch",key="dl_summary")
    st.markdown('<div class="spacer"></div>',unsafe_allow_html=True)
    with st.expander("View a small sample of the packaged CSV"):
        st.dataframe(df.head(80),width="stretch",hide_index=True,height=390)

st.markdown(f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Latest processed field: {fmt_date(latest_date)}</div>',unsafe_allow_html=True)
