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
