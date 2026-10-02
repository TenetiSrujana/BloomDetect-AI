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
    a,b=st.columns([1.2,1])
    with a:
        st.markdown('<div class="card"><div class="card-title">Priority hotspots</div><div class="card-text">Connected flagged cells are grouped into spatial monitoring areas.</div></div>',unsafe_allow_html=True)
        if hs.empty:st.info('No current hotspot clusters.')
        else:
            q=hs.head(8).copy();q['region']=q.apply(lambda x:region(x.latitude,x.longitude),axis=1);q['area_km2']=q.approx_area_km2.round(0);st.dataframe(q[['hotspot_id','region','cells','area_km2','priority']],hide_index=True,use_container_width=True)
    with b:
        st.markdown('''<div class="card"><div class="card-title">Scientific interpretation</div><div class="card-text">A flag indicates a proxy Chlorophyll-a risk condition. It does not establish harmful species, toxin presence or a confirmed HAB. Field/laboratory validation remains necessary.</div></div>''',unsafe_allow_html=True)

# EVENT EXPLORER
elif st.session_state.page=='Event Explorer':
    st.markdown('<div class="section-title">Event Explorer</div><div class="section-note">Investigate a spatial risk cluster as one event instead of reading hundreds of dots.</div>',unsafe_allow_html=True)
    if hs.empty:st.info('No current events available.')
    else:
        selected=st.selectbox('Select a potential-risk event',hs.hotspot_id.tolist());e=hs[hs.hotspot_id==selected].iloc[0]
        members=risk[(risk.latitude.between(e.latitude-.8,e.latitude+.8))&(risk.longitude.between(e.longitude-.8,e.longitude+.8))].copy()
        a,b,c,d=st.columns(4)
        data=[('Event cells',len(members),'connected risk cells'),('Mean score',f'{members.risk_score.mean():.0f}/100','current screening signal'),('Peak score',f'{members.risk_score.max():.0f}/100','strongest cell'),('Area',f'{e.approx_area_km2/1000:.1f}k km²','approximate grid footprint')]
        for c,(x,y,z) in zip([a,b,c,d],data):
            with c:st.markdown(f'<div class="metric"><div class="metric-label">{x}</div><div class="metric-value" style="font-size:22px">{y}</div><div class="metric-small">{z}</div></div>',unsafe_allow_html=True)
        fig=go.Figure();fig.add_trace(go.Scattergeo(lat=latest.latitude,lon=latest.longitude,mode='markers',marker=dict(size=3,color='#4B86B4',opacity=.22),name='Processed'));fig.add_trace(go.Scattergeo(lat=members.latitude,lon=members.longitude,mode='markers',marker=dict(size=9,color='#D94A54'),name='Potential risk'));st.plotly_chart(geo_base(fig),use_container_width=True)
        st.markdown('<div class="callout"><b>Event status:</b> this is a current spatial cluster. A true Emerging → Growing → Peak → Declining lifecycle requires a multi-date event-history dataset, so the current website does not invent that history.</div>',unsafe_allow_html=True)
        cols=[c for c in ['latitude','longitude','chla','historical_baseline','chla_anomaly','chla_change','risk_score'] if c in members];st.dataframe(members[cols].sort_values('risk_score',ascending=False).head(30),hide_index=True,use_container_width=True)

# SPATIAL
elif st.session_state.page=='Spatial Intelligence':
    st.markdown('<div class="section-title">Spatial Intelligence</div><div class="section-note">Explore risk intensity, connected hotspots and approximate potential-risk extent.</div>',unsafe_allow_html=True)
    mode=st.radio('Map layer',['Normal + potential risk','Potential risk only','Screening score'],horizontal=True)
    if mode=='Normal + potential risk':
        x=latest.copy();x['status']=np.where(x.risk_flag==1,'Potential bloom risk','Normal');fig=px.scatter_geo(x,lat='latitude',lon='longitude',color='status',color_discrete_map={'Normal':'#4B86B4','Potential bloom risk':'#D94A54'},projection='mercator',scope='world',hover_data=['latitude','longitude','risk_score'])
    elif mode=='Potential risk only':fig=px.scatter_geo(risk,lat='latitude',lon='longitude',color_discrete_sequence=['#D94A54'],projection='mercator',scope='world',hover_data=['latitude','longitude','risk_score'])
    else:fig=px.scatter_geo(latest,lat='latitude',lon='longitude',color='risk_score',color_continuous_scale=['#EAF7FA','#65C9BF','#F1B95E','#D94A54'],projection='mercator',scope='world',hover_data=['latitude','longitude','risk_flag'])
    st.plotly_chart(geo_base(fig),use_container_width=True)
    a,b,c=st.columns(3)
    with a:st.markdown(f'<div class="metric"><div class="metric-label">Potential-risk grid extent</div><div class="metric-value">{len(risk)*30*27.5/1e6:.2f} M km²</div><div class="metric-small">approximate grid footprint, not confirmed bloom area</div></div>',unsafe_allow_html=True)
    with b:st.markdown(f'<div class="metric"><div class="metric-label">Largest hotspot</div><div class="metric-value">{(hs.iloc[0].approx_area_km2/1000 if len(hs) else 0):.1f}k km²</div><div class="metric-small">connected flagged cells</div></div>',unsafe_allow_html=True)
    with c:st.markdown(f'<div class="metric"><div class="metric-label">Study cells</div><div class="metric-value">{len(latest):,}</div><div class="metric-small">latest processed grid</div></div>',unsafe_allow_html=True)
    if not hs.empty:
        q=hs.copy();q['region']=q.apply(lambda x:region(x.latitude,x.longitude),axis=1);st.dataframe(q[['hotspot_id','region','latitude','longitude','cells','approx_area_km2','priority']],hide_index=True,use_container_width=True)

# LOCATION
elif st.session_state.page=='Location Intelligence':
    st.markdown('<div class="section-title">Location Intelligence</div><div class="section-note">Your coordinate and the nearest processed satellite cell are always shown separately.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:lat=st.number_input('Your latitude',-90.,90.,17.35,.25,format='%.2f')
    with b:lon=st.number_input('Your longitude',-180.,180.,97.,.25,format='%.2f')
    ls=111*math.cos(math.radians(lat));d2=((latest.latitude-lat)*111)**2+((latest.longitude-lon)*ls)**2;i=d2.idxmin();r=latest.loc[i];dist=math.sqrt(d2.loc[i]);flag=int(r.risk_flag)==1
    a,b=st.columns(2)
    with a:st.markdown(f'<div class="card"><div class="card-title">Your input</div><div class="metric-value" style="font-size:24px">{lat:.2f}° · {lon:.2f}°</div><div class="card-text">The coordinate you entered.</div></div>',unsafe_allow_html=True)
    with b:st.markdown(f'<div class="card"><div class="card-title">Nearest processed cell</div><div class="metric-value" style="font-size:24px">{r.latitude:.2f}° · {r.longitude:.2f}°</div><div class="card-text">Approx. distance: <b>{dist:.1f} km</b></div></div>',unsafe_allow_html=True)
    if flag:st.markdown('<div class="card" style="border-color:#E29AA0;background:#FFF6F6"><div class="card-title">🔴 Potential bloom-risk condition flagged</div><span class="badge red">HIGH-PRIORITY SCREENING SIGNAL</span></div>',unsafe_allow_html=True)
    else:st.markdown('<div class="card" style="border-color:#9ED6C3;background:#F3FCF8"><div class="card-title">🟢 No potential bloom-risk flag</div><span class="badge green">NOT FLAGGED</span></div>',unsafe_allow_html=True)
    fields=[('Current Chl-a','chla'),('Previous Chl-a','previous_chla'),('Historical baseline','historical_baseline'),('Recent mean','recent_mean'),('Recent maximum','recent_max')];cs=st.columns(5)
    for c,(title,key) in zip(cs,fields):
        with c:st.markdown(f'<div class="metric"><div class="metric-label">{title}</div><div class="metric-value" style="font-size:21px">{r[key]:.4f}</div></div>',unsafe_allow_html=True) if key in r else st.markdown(f'<div class="metric"><div class="metric-label">{title}</div><div class="metric-value">—</div></div>',unsafe_allow_html=True)
    why=[]
    if 'chla_anomaly' in r:why.append(f"anomaly <b>{r.chla_anomaly:.4f}</b>")
    if 'chla_change' in r:why.append(f"change <b>{r.chla_change:.4f}</b>")
    why.append(f"screening score <b>{r.risk_score:.0f}/100</b>");st.markdown('<div class="callout"><b>Why this result?</b> '+' • '.join(why)+'</div>',unsafe_allow_html=True)
    st.markdown('<div class="callout danger"><b>Limit:</b> the signal cannot identify species or toxins and is not confirmation of HAB.</div>',unsafe_allow_html=True)

# ANALYTICS
elif st.session_state.page=='Analytics':
    st.markdown('<div class="section-title">Analytics</div><div class="section-note">Only decision-useful charts. No decorative graphs pretending to be science.</div>',unsafe_allow_html=True)
    t1,t2,t3=st.tabs(['Risk structure','Chlorophyll-a','Hotspot structure'])
    with t1:
        q=pd.DataFrame({'Status':['Normal','Potential bloom risk'],'Cells':[(latest.risk_flag==0).sum(),(latest.risk_flag==1).sum()]});f=px.bar(q,x='Status',y='Cells',color='Status',color_discrete_map={'Normal':'#4B86B4','Potential bloom risk':'#D94A54'},text='Cells');f.update_traces(texttemplate='%{text:,}',textposition='outside');f.update_layout(height=390,showlegend=False,xaxis_title='',yaxis_title='Processed cells');st.plotly_chart(f,use_container_width=True)
    with t2:
        if 'chla' in latest:
            f=px.histogram(latest,x='chla',nbins=45,color_discrete_sequence=['#1FA7A0']);f.update_layout(height=390,xaxis_title='Chlorophyll-a',yaxis_title='Number of processed cells');st.plotly_chart(f,use_container_width=True)
        else:st.info('Chlorophyll-a is not present in the website file.')
    with t3:
        if not hs.empty:
            q=hs.head(12).sort_values('cells');f=px.bar(q,x='cells',y='hotspot_id',orientation='h',text='cells',color_discrete_sequence=['#D94A54']);f.update_layout(height=430,xaxis_title='Potential-risk grid cells',yaxis_title='Hotspot');st.plotly_chart(f,use_container_width=True)
        else:st.info('No hotspots.')

# RESPONSE
elif st.session_state.page=='Response Center':
    st.markdown('<div class="section-title">Response Center</div><div class="section-note">The system supports verification and prioritisation. It does not pretend there is a universal “cure”.</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4);steps=[('01','DETECT','Locate potential-risk areas.'),('02','VERIFY','Prioritize appropriate field or water-quality validation.'),('03','MONITOR','Check persistence and subsequent observations.'),('04','ASSESS','Use confirmed evidence and authority guidance for environmental, fisheries or public-health action.')]
    for col,(n,t,x) in zip([a,b,c,d],steps):
        with col:st.markdown(f'<div class="card"><span class="badge blue">{n}</span><div class="card-title" style="margin-top:10px">{t}</div><div class="card-text">{x}</div></div>',unsafe_allow_html=True)
    st.markdown('### What this satellite signal cannot establish')
    for x in ['confirmed harmful algal bloom','exact algal species','toxin presence','a closure, health or fisheries decision by itself']:st.markdown(f'<div class="callout">• It cannot establish <b>{x}</b>.</div>',unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Real-world value</div><div class="card-text">The system reduces the search area, highlights unusual spatial patterns and helps monitoring teams decide where confirmation and closer observation may be useful.</div></div>',unsafe_allow_html=True)

# REPORT
else:
    st.markdown('<div class="section-title">Report Center</div><div class="section-note">Download concise, evidence-focused outputs from the current processed observation.</div>',unsafe_allow_html=True)
    summary=pd.DataFrame([{'processed_date':latest_date.strftime('%Y-%m-%d') if latest_date is not None else 'Latest','processed_locations':len(latest),'potential_risk_cells':len(risk),'risk_percent':round(len(risk)/len(latest)*100,3),'hotspot_clusters':len(hs),'largest_hotspot_cells':int(hs.iloc[0].cells) if len(hs) else 0,'largest_hotspot_area_km2':round(float(hs.iloc[0].approx_area_km2),2) if len(hs) else 0,'highest_screening_score':round(float(latest.risk_score.max()),2)}])
    st.markdown('<div class="download"><div class="card-title">Latest screening report</div><div class="card-text">Compact summary for project review and sharing.</div></div>',unsafe_allow_html=True);st.dataframe(summary,hide_index=True,use_container_width=True)
    st.download_button('Download screening summary (CSV)',summary.to_csv(index=False).encode(),file_name='bloomdetect_screening_report.csv',mime='text/csv',use_container_width=True)
    cols=[c for c in ['date','latitude','longitude','chla','previous_chla','historical_baseline','recent_mean','recent_max','chla_anomaly','chla_change','risk_score','risk_prediction'] if c in risk];ex=risk[cols].copy();ex['interpretation']='Potential bloom risk; not confirmed HAB';st.download_button('Download potential-risk observations (CSV)',ex.to_csv(index=False).encode(),file_name='bloomdetect_potential_risk_observations.csv',mime='text/csv',use_container_width=True)

st.markdown('<div class="footer">BloomDetect AI • EOS-06 OCM-3 Chlorophyll-a screening • Potential bloom-risk support, not confirmed HAB diagnosis.</div>',unsafe_allow_html=True)
