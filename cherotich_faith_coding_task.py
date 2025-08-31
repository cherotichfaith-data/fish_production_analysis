# app.py — Fish Cage Production Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# ===============================
# 1. Load Data
# ===============================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfers = pd.read_excel(transfer_file) if transfer_file else None
    # Normalize column names
    for df in [feeding, harvest, sampling, transfers] if transfers is not None else [feeding, harvest, sampling]:
        if df is not None:
            df.columns = [c.strip().upper() for c in df.columns]
    return feeding, harvest, sampling, transfers

# ===============================
# 2. Preprocess Cage 2 (with transfers)
# ===============================
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    
    # Filter to cage 2 and timeframe
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy() if 'CAGE NUMBER' in feeding.columns else feeding.copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy() if 'CAGE' in harvest.columns else harvest.copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy() if 'CAGE NUMBER' in sampling.columns else sampling.copy()
    
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]
    harvest_c2 = harvest_c2[(harvest_c2['DATE'] >= start_date) & (harvest_c2['DATE'] <= end_date)]
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    
    # --- Stocking
    stocked_fish = 7290
    initial_abw = 11.9
    stocking_row = pd.DataFrame([{
        'DATE': start_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': stocked_fish,
        'AVERAGE BODY WEIGHT (g)': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE').reset_index(drop=True)
    
    # --- Handle transfers
    sampling_c2['IN_FISH_CUM'] = 0
    sampling_c2['IN_KG_CUM'] = 0
    sampling_c2['OUT_FISH_CUM'] = 0
    sampling_c2['OUT_KG_CUM'] = 0
    
    if transfers is not None:
        transfers = transfers.copy()
        # Ensure numeric
        for col in ['ORIGIN CAGE', 'DESTINATION CAGE', 'NUMBER OF FISH', 'TOTAL WEIGHT (KG)']:
            if col in transfers.columns:
                transfers[col] = pd.to_numeric(transfers[col], errors='coerce')
        # Convert g → kg if weight > 500 (assume g)
        if 'TOTAL WEIGHT (KG)' in transfers.columns:
            transfers.loc[transfers['TOTAL WEIGHT (KG)'] > 500, 'TOTAL WEIGHT (KG)'] /= 1000.0
        # Outgoing
        out = transfers[transfers['ORIGIN CAGE'] == cage_number].sort_values('DATE')
        out['OUT_FISH_CUM'] = out['NUMBER OF FISH'].cumsum()
        out['OUT_KG_CUM'] = out['TOTAL WEIGHT (KG)'].cumsum()
        sampling_c2 = pd.merge_asof(sampling_c2.sort_values('DATE'), 
                                    out[['DATE','OUT_FISH_CUM','OUT_KG_CUM']], on='DATE', direction='backward')
        sampling_c2['OUT_FISH_CUM'] = sampling_c2['OUT_FISH_CUM'].fillna(0)
        sampling_c2['OUT_KG_CUM'] = sampling_c2['OUT_KG_CUM'].fillna(0)
        # Incoming
        inc = transfers[transfers['DESTINATION CAGE'] == cage_number].sort_values('DATE')
        inc['IN_FISH_CUM'] = inc['NUMBER OF FISH'].cumsum()
        inc['IN_KG_CUM'] = inc['TOTAL WEIGHT (KG)'].cumsum()
        sampling_c2 = pd.merge_asof(sampling_c2.sort_values('DATE'), 
                                    inc[['DATE','IN_FISH_CUM','IN_KG_CUM']], on='DATE', direction='backward')
        sampling_c2['IN_FISH_CUM'] = sampling_c2['IN_FISH_CUM'].fillna(0)
        sampling_c2['IN_KG_CUM'] = sampling_c2['IN_KG_CUM'].fillna(0)
    
    # --- Compute standing fish
    sampling_c2['STOCKED'] = stocked_fish
    sampling_c2['HARV_FISH_CUM'] = harvest_c2['NUMBER OF FISH'].cumsum() if not harvest_c2.empty else 0
    sampling_c2['HARV_KG_CUM'] = harvest_c2['TOTAL WEIGHT (KG)'].cumsum() if not harvest_c2.empty else 0
    sampling_c2['FISH_ALIVE'] = (sampling_c2['STOCKED'] - sampling_c2['HARV_FISH_CUM'] + 
                                 sampling_c2['IN_FISH_CUM'] - sampling_c2['OUT_FISH_CUM']).clip(lower=0)
    sampling_c2['NUMBER OF FISH'] = sampling_c2['FISH_ALIVE'].astype(int)
    
    return feeding_c2, harvest_c2, sampling_c2

# ===============================
# 3. Compute summary
# ===============================
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    sampling_c2 = sampling_c2.copy()
    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (KG)'].cumsum()
    
    summary = pd.merge_asof(sampling_c2.sort_values('DATE'),
                            feeding_c2[['DATE','CUM_FEED']].sort_values('DATE'),
                            on='DATE', direction='backward')
    
    # Total weight
    summary['TOTAL_WEIGHT_KG'] = summary['NUMBER OF FISH'] * summary['AVERAGE BODY WEIGHT (g)'] / 1000
    
    # Period metrics
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
    
    return summary

# ===============================
# 4. Streamlit UI
# ===============================
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file  = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)
    
    st.subheader("Cage 2 – Production Summary")
    st.dataframe(summary_c2[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']])
    
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth","eFCR"])
    
    if selected_kpi == "Growth":
        fig = px.line(summary_c2, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title="Cage 2: Growth Over Time", labels={"TOTAL_WEIGHT_KG":"Total Weight (Kg)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(summary_c2, x='DATE', y='AGGREGATED_eFCR', markers=True, name="Aggregated eFCR")
        fig.add_scatter(x=summary_c2['DATE'], y=summary_c2['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title="Cage 2: eFCR Over Time", yaxis_title="eFCR")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
