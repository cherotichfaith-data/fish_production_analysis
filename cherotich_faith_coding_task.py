# ==========================
# Fish Cage Production Analysis (Cage 2 with Transfers)
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------
# 1. Load Data
# --------------------------
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding = pd.read_excel(feeding_file, parse_dates=['DATE'], dayfirst=True)
    harvest = pd.read_excel(harvest_file, parse_dates=['DATE'], dayfirst=True)
    sampling = pd.read_excel(sampling_file, parse_dates=['DATE'], dayfirst=True)
    if transfer_file:
        transfers = pd.read_excel(transfer_file, parse_dates=['DATE'], dayfirst=True)
    else:
        transfers = pd.DataFrame()
    return feeding, harvest, sampling, transfers

# --------------------------
# 2. Preprocess Cage 2
# --------------------------
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")

    feeding_c2 = feeding[(feeding['CAGE_NUMBER']==cage_number) & 
                         (feeding['DATE']>=start_date) & (feeding['DATE']<=end_date)].copy()
    harvest_c2 = harvest[(harvest['CAGE']==cage_number) & 
                         (harvest['DATE']>=start_date) & (harvest['DATE']<=end_date)].copy()
    sampling_c2 = sampling[(sampling['CAGE_NUMBER']==cage_number) & 
                           (sampling['DATE']>=start_date) & (sampling['DATE']<=end_date)].copy()

    # Ensure numeric columns
    for col in ['NUMBER_OF_FISH', 'ABW_G']:
        if col in sampling_c2.columns:
            sampling_c2[col] = pd.to_numeric(sampling_c2[col], errors='coerce').fillna(0)

    # Initial stocking row
    stocking_row = pd.DataFrame([{
        'DATE': start_date,
        'CAGE_NUMBER': cage_number,
        'NUMBER_OF_FISH': 7290,
        'ABW_G': 11.9
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE').reset_index(drop=True)

    # Add transfer columns
    sampling_c2['TRANSFER_IN_FISH'] = 0
    sampling_c2['TRANSFER_OUT_FISH'] = 0
    if transfers is not None and not transfers.empty:
        transfers_c2 = transfers.copy()
        in_mask = transfers_c2.get('DESTINATION_CAGE', pd.Series()) == cage_number
        out_mask = transfers_c2.get('ORIGIN_CAGE', pd.Series()) == cage_number

        if out_mask.any():
            out_transfers = transfers_c2[out_mask].groupby('DATE')['NUMBER_OF_FISH'].sum().cumsum().reset_index()
            out_transfers.rename(columns={'NUMBER_OF_FISH':'TRANSFER_OUT_FISH'}, inplace=True)
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('DATE'), 
                                        out_transfers.sort_values('DATE'), on='DATE', direction='backward')
        if in_mask.any():
            in_transfers = transfers_c2[in_mask].groupby('DATE')['NUMBER_OF_FISH'].sum().cumsum().reset_index()
            in_transfers.rename(columns={'NUMBER_OF_FISH':'TRANSFER_IN_FISH'}, inplace=True)
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('DATE'), 
                                        in_transfers.sort_values('DATE'), on='DATE', direction='backward')

        sampling_c2[['TRANSFER_IN_FISH','TRANSFER_OUT_FISH']] = sampling_c2[['TRANSFER_IN_FISH','TRANSFER_OUT_FISH']].fillna(0)

    # Compute fish alive & biomass
    sampling_c2['FISH_ALIVE'] = (sampling_c2['NUMBER_OF_FISH'] + sampling_c2['TRANSFER_IN_FISH'] - sampling_c2['TRANSFER_OUT_FISH']).clip(lower=0)
    sampling_c2['BIOMASS_KG'] = sampling_c2['FISH_ALIVE'] * sampling_c2['ABW_G'] / 1000

    return feeding_c2, harvest_c2, sampling_c2

# --------------------------
# 3. Compute Production Summary
# --------------------------
def compute_summary(feeding_c2, sampling_c2, harvest_c2=None):
    df = sampling_c2.sort_values('DATE').reset_index(drop=True)
    df['FEED_PERIOD_KG'] = 0
    df['FEED_AGG_KG'] = 0
    df['GROWTH_KG'] = 0
    df['HARVEST_FISH'] = 0
    df['HARVEST_KG'] = 0
    df['FISH_COUNT_DISCREPANCY'] = 0

    feeding_c2 = feeding_c2.sort_values('DATE')
    if harvest_c2 is not None:
        harvest_c2 = harvest_c2.sort_values('DATE')

    for i in range(1, len(df)):
        start_date = df.loc[i-1, 'DATE']
        end_date = df.loc[i, 'DATE']
        # Period feed
        mask = (feeding_c2['DATE']>start_date) & (feeding_c2['DATE']<=end_date)
        df.loc[i, 'FEED_PERIOD_KG'] = feeding_c2.loc[mask,'FEED_AMOUNT_KG'].sum()
        df.loc[i, 'FEED_AGG_KG'] = df['FEED_PERIOD_KG'][:i+1].sum()
        # Growth
        df.loc[i, 'GROWTH_KG'] = df.loc[i,'BIOMASS_KG'] - df.loc[i-1,'BIOMASS_KG']
        # Harvest
        if harvest_c2 is not None and not harvest_c2.empty:
            hmask = (harvest_c2['DATE']>start_date) & (harvest_c2['DATE']<=end_date)
            df.loc[i,'HARVEST_FISH'] = harvest_c2.loc[hmask,'NUMBER_OF_FISH'].sum()
            df.loc[i,'HARVEST_KG'] = harvest_c2.loc[hmask,'TOTAL_WEIGHT_KG'].sum()
        # Fish discrepancy
        df.loc[i,'FISH_COUNT_DISCREPANCY'] = df.loc[i,'NUMBER_OF_FISH'] + df.loc[i,'TRANSFER_IN_FISH'] - df.loc[i,'TRANSFER_OUT_FISH'] - df.loc[i,'FISH_ALIVE'] - df.loc[i,'HARVEST_FISH']

    # First row
    df.loc[0,'FEED_PERIOD_KG'] = 0
    df.loc[0,'FEED_AGG_KG'] = 0
    df.loc[0,'GROWTH_KG'] = df.loc[0,'BIOMASS_KG']
    df.loc[0,'HARVEST_FISH'] = 0
    df.loc[0,'HARVEST_KG'] = 0
    df.loc[0,'FISH_COUNT_DISCREPANCY'] = df.loc[0,'NUMBER_OF_FISH'] + df.loc[0,'TRANSFER_IN_FISH'] - df.loc[0,'TRANSFER_OUT_FISH'] - df.loc[0,'FISH_ALIVE']

    # eFCR
    df['PERIOD_eFCR'] = df['FEED_PERIOD_KG'] / df['GROWTH_KG'].replace(0,np.nan)
    df['AGGREGATED_eFCR'] = df['FEED_AGG_KG'] / df['BIOMASS_KG'].replace(0,np.nan)

    return df

# --------------------------
# 4. Streamlit App
# --------------------------
st.title("Fish Cage Production Analysis (with Transfers)")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file  = st.sidebar.file_uploader("Feeding Record",    type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest",      type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling",     type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2, harvest_c2)

    st.subheader("Cage 2 – Production Summary (period-based)")
    show_cols = [
        "DATE","NUMBER_OF_FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH","HARVEST_FISH",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]
    # Fill transfer KG if ABW_G exists
    if 'ABW_G' in summary_c2.columns:
        summary_c2['TRANSFER_IN_KG'] = summary_c2['TRANSFER_IN_FISH'] * summary_c2['ABW_G']/1000
        summary_c2['TRANSFER_OUT_KG'] = summary_c2['TRANSFER_OUT_FISH'] * summary_c2['ABW_G']/1000

    st.dataframe(summary_c2[[c for c in show_cols if c in summary_c2.columns]].round(2))

    # KPI selection
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])
    if selected_kpi == "Biomass":
        fig = px.line(summary_c2.dropna(subset=["BIOMASS_KG"]), x="DATE", y="BIOMASS_KG", markers=True,
                      title="Cage 2: Biomass Over Time", labels={"BIOMASS_KG":"Total Biomass (kg)"})
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi == "ABW":
        fig = px.line(summary_c2.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                      title="Cage 2: Average Body Weight Over Time", labels={"ABW_G":"ABW (g)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
                      title="Cage 2: eFCR Over Time", labels={"AGGREGATED_eFCR":"Aggregated eFCR"})
        fig.update_traces(showlegend=True, name="Aggregated eFCR")
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", 
                        name="Period eFCR", showlegend=True, line=dict(dash="dash"))
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
