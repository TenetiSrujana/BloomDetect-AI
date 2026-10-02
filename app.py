from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='BloomDetect AI',page_icon='🌊',layout='wide',initial_sidebar_state='collapsed')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root{--navy:#073B4C;--aqua:#1FA7A0;--mint:#E9F8F5;--blue:#4B86B4;--red:#D94A54;--amber:#D99427;--ink:#16343D;--muted:#587078;--line:#B9D5D9}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:radial-gradient(circle at 8% 8%,rgba(31,167,160,.08),transparent 24%),radial-gradient(circle at 92% 12%,rgba(29,111,165,.07),transparent 25%),#fff}
.stApp:after{content:'';position:fixed;left:-10%;right:-10%;bottom:-90px;height:190px;background:radial-gradient(ellipse at 20% 50%,rgba(31,167,160,.12),transparent 38%),radial-gradient(ellipse at 75% 50%,rgba(29,111,165,.1),transparent 40%);animation:drift 12s ease-in-out infinite alternate;z-index:-1}@keyframes drift{to{transform:translateX(25px) scale(1.04)}}
.block-container{max-width:1380px;padding-top:1.4rem;padding-bottom:4rem}
.brand{display:flex;align-items:center;gap:14px;margin-bottom:18px}.brand-mark{width:52px;height:52px;border-radius:17px;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,#0B5266,#1FA7A0);color:#fff;font-size:26px;box-shadow:0 10px 24px rgba(7,59,76,.16)}.brand-title{font-family:'Plus Jakarta Sans';font-size:28px;font-weight:800;color:var(--navy)}.brand-sub{color:var(--muted);font-size:14px}
.hero,.card,.metric,.download{background:rgba(255,255,255,.96);border:1.5px solid var(--line);border-radius:20px;box-shadow:0 8px 24px rgba(7,59,76,.055)}
.hero{padding:30px 32px;margin:8px 0 22px;background:linear-gradient(115deg,#F3FBFA,#fff);position:relative;overflow:hidden}.hero:after{content:'◌';position:absolute;right:35px;bottom:-35px;font-size:170px;color:rgba(31,167,160,.08)}.eyebrow{color:var(--aqua);font-size:13px;font-weight:800;letter-spacing:1.4px;text-transform:uppercase}.hero h1{font-family:'Plus Jakarta Sans';color:var(--navy);font-size:38px;line-height:1.12;margin:8px 0 10px}.hero p{color:var(--muted);font-size:17px;line-height:1.6;max-width:900px}
div.stButton>button{width:100%;min-height:48px;border-radius:13px;border:1.5px solid #0A5266;background:#F7FCFC;color:#073B4C;font-weight:700;font-size:14px;box-shadow:0 4px 12px rgba(7,59,76,.05)}div.stButton>button:hover{background:#E8F8F5;border-color:#1FA7A0;transform:translateY(-1px)}
.section-title{font-family:'Plus Jakarta Sans';color:var(--navy);font-size:25px;font-weight:800;margin:18px 0 5px}.section-note{color:var(--muted);font-size:15px;margin-bottom:16px}.card{padding:21px;height:100%}.card-title{font-family:'Plus Jakarta Sans';color:var(--navy);font-size:17px;font-weight:800;margin-bottom:7px}.card-text{color:var(--muted);font-size:14px;line-height:1.55}.metric{padding:18px;min-height:118px}.metric-label{color:var(--muted);font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.5px}.metric-value{color:var(--navy);font-family:'Plus Jakarta Sans';font-size:28px;font-weight:800;margin-top:6px}.metric-small{color:var(--muted);font-size:12px;margin-top:4px}.badge{display:inline-block;border-radius:999px;padding:6px 11px;font-size:12px;font-weight:800}.red{background:#FBE3E5;color:#A92C38}.green{background:#DDF5EC;color:#0C7656}.amber{background:#FFF0D2;color:#8A5B09}.blue{background:#E1F1FA;color:#155B83}.callout{border-left:5px solid var(--aqua);border-radius:14px;background:#F1FBFA;padding:15px 17px;color:var(--ink);font-size:14px;line-height:1.55;margin:12px 0}.danger{border-left-color:var(--red);background:#FFF5F5}.download{padding:20px;background:linear-gradient(135deg,#F2FBFA,#fff);border:2px solid #0A5266}.footer{margin-top:42px;padding-top:18px;border-top:1px solid #D7E5E7;color:var(--muted);font-size:12px;text-align:center}
</style>''',unsafe_allow_html=True)

DATA=Path(__file__).resolve().parent/'latest_bloom_risk_predictions.csv'
if not DATA.exists(): st.error('Keep latest_bloom_risk_predictions.csv beside app.py.'); st.stop()

@st.cache_data(show_spinner=False)
def load(path):
    d=pd.read_csv(path);d.columns=[str(c).strip().lower().replace(' ','_') for c in d.columns]
    ren={'lat':'latitude','lon':'longitude','prediction':'risk_prediction','predicted_label':'risk_prediction'};d=d.rename(columns={k:v for k,v in ren.items() if k in d})
    for c in ['latitude','longitude','chla','previous_chla','historical_baseline','recent_mean','recent_max','chla_anomaly','chla_change']:
        if c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
    if 'date' in d:d['date']=pd.to_datetime(d['date'],errors='coerce')
    if 'risk_prediction' not in d:d['risk_prediction']=0
    if d['risk_prediction'].dtype==object:d['risk_flag']=d['risk_prediction'].astype(str).str.lower().str.contains('risk|bloom|potential|1|true',regex=True).astype(int)
    else:d['risk_flag']=pd.to_numeric(d['risk_prediction'],errors='coerce').fillna(0).astype(int)
    if 'chla_anomaly' not in d and {'chla','historical_baseline'}<=set(d):d['chla_anomaly']=d.chla-d.historical_baseline
    if 'chla_change' not in d and {'chla','previous_chla'}<=set(d):d['chla_change']=d.chla-d.previous_chla
    return d.dropna(subset=['latitude','longitude']).copy()

df=load(str(DATA)); latest_date=df.date.max() if 'date' in df and df.date.notna().any() else None
latest=df[df.date.dt.normalize()==latest_date.normalize()].copy() if latest_date is not None else df.copy()
if latest.empty:latest=df.copy()

def score(r):
    parts=[]
    if pd.notna(r.get('chla_anomaly',np.nan)):parts.append(np.clip(float(r.chla_anomaly)/.15,0,1)*35)
    if pd.notna(r.get('chla_change',np.nan)):parts.append(np.clip(float(r.chla_change)/.06,0,1)*30)
    if pd.notna(r.get('chla',np.nan)) and pd.notna(r.get('recent_mean',np.nan)) and r.recent_mean!=0:parts.append(np.clip((r.chla-r.recent_mean)/max(abs(r.recent_mean),1e-6),0,1)*20)
    if pd.notna(r.get('chla',np.nan)) and pd.notna(r.get('recent_max',np.nan)) and r.recent_max!=0:parts.append(np.clip(r.chla/max(abs(r.recent_max),1e-6),0,1)*15)
    return float(np.clip(sum(parts),0,100)) if parts else (100. if int(r.risk_flag) else 0.)
latest['risk_score']=latest.apply(score,axis=1); risk=latest[latest.risk_flag==1].copy()

def region(lat,lon):
    if 5<=lat<=30 and 40<=lon<=80:return 'Arabian Sea / West Indian Ocean'
    if 5<=lat<=30 and 80<lon<=110:return 'Bay of Bengal / East Indian Ocean'
    if -40<=lat<5 and 40<=lon<=120:return 'Southern Indian Ocean'
    return 'Wider study region'

def hotspots(r):
    if r.empty:return pd.DataFrame()
    p=r[['latitude','longitude']].drop_duplicates().copy();p['a']=np.round(p.latitude*4).astype(int);p['b']=np.round(p.longitude*4).astype(int);cells=set(zip(p.a,p.b));seen=set();out=[]
    for s in cells:
        if s in seen:continue
        stack=[s];seen.add(s);group=[]
        while stack:
            a,b=stack.pop();group.append((a,b))
            for n in [(a+1,b),(a-1,b),(a,b+1),(a,b-1),(a+1,b+1),(a+1,b-1),(a-1,b+1),(a-1,b-1)]:
                if n in cells and n not in seen:seen.add(n);stack.append(n)
        z=p.set_index(['a','b']).loc[group];out.append({'hotspot_id':f'HS-{len(out)+1:02d}','cells':len(group),'latitude':z.latitude.mean(),'longitude':z.longitude.mean()})
    h=pd.DataFrame(out).sort_values('cells',ascending=False).reset_index(drop=True);h['approx_area_km2']=h.cells*30*27.5;h['priority']=np.select([h.cells>=20,h.cells>=8],['High','Moderate'],default='Watch');return h
hs=hotspots(risk)

st.markdown('''<div class="brand"><div class="brand-mark">🌊</div><div><div class="brand-title">BloomDetect AI</div><div class="brand-sub">Potential bloom-risk intelligence from EOS-06 OCM-3 Chlorophyll-a data</div></div></div>''',unsafe_allow_html=True)

pages=['Command Center','Event Explorer','Spatial Intelligence','Location Intelligence','Analytics','Response Center','Report Center']
if 'page' not in st.session_state:st.session_state.page=pages[0]
nav=st.columns(len(pages),gap='small')
for c,p in zip(nav,pages):
    with c:
        if st.button(p,key='nav_'+p,type='primary' if st.session_state.page==p else 'secondary'):st.session_state.page=p;st.rerun()

def geo_base(fig):
    fig.update_geos(lataxis_range=[-40,30],lonaxis_range=[20,120],showland=True,landcolor='#F4F7F7',showocean=True,oceancolor='#EAF7FA',showcountries=True,countrycolor='#A9C5CB');fig.update_layout(height=560,margin=dict(l=0,r=0,t=10,b=0),font=dict(family='DM Sans',size=13,color='#16343D'));return fig

# COMMAND CENTER
if st.session_state.page=='Command Center':
    date=latest_date.strftime('%d %b %Y') if latest_date is not None else 'Latest processed observation'
    st.markdown(f'''<div class="hero"><div class="eyebrow">EOS-06 OCM-3 • Indian Ocean study region</div><h1>Find where potential bloom-risk conditions need attention.</h1><p>BloomDetect AI combines Chlorophyll-a behaviour with machine-learning screening and turns the result into spatial intelligence for monitoring. It is a screening system, not confirmed HAB ground truth.</p><span class="badge blue">Latest processed date: {date}</span></div>''',unsafe_allow_html=True)
    vals=[('Processed locations',f'{len(latest):,}','latest grid'),('Potential-risk cells',f'{len(risk):,}',f'{len(risk)/len(latest)*100:.2f}% of cells'),('Hotspot clusters',f'{len(hs):,}','connected risk cells'),('Highest screening score',f'{latest.risk_score.max():.0f}/100','transparent index')]
    cs=st.columns(4)
    for c,(a,b,e) in zip(cs,vals):
        with c:st.markdown(f'<div class="metric"><div class="metric-label">{a}</div><div class="metric-value">{b}</div><div class="metric-small">{e}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Current spatial situation</div><div class="section-note">Blue = normal processed cells. Red = potential-risk cells.</div>',unsafe_allow_html=True)
    m=latest.copy();m['status']=np.where(m.risk_flag==1,'Potential bloom risk','Normal');fig=px.scatter_geo(m,lat='latitude',lon='longitude',color='status',color_discrete_map={'Normal':'#4B86B4','Potential bloom risk':'#D94A54'},projection='mercator',scope='world',hover_data=['latitude','longitude','risk_score']);st.plotly_chart(geo_base(fig),use_container_width=True)
