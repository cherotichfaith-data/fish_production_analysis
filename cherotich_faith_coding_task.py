import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===============================
# 1. Load Data
# ===============================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfers = pd.read_excel(transfer_file)
    
    # Correct transfer weights if given in grams
    if 'WEIGHT' in transfers.columns:
        transfers['WEIGHT_KG'] = transfers['WEIGHT']
        if transfers['WEIGHT'].max() > 1000:  # likely grams
            transfers['WEIGHT_KG'] = transfers['WEIGHT'] / 1000

    return feeding, harvest, sampling, transfers

# ===============================
# 2. Preprocess Cage 2 with transfers
# ===============================
def preprocess_cage2(feeding, harvest, sampling, transfers):
    cage_number = 2
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Add stocking manually
    stocking_date = pd.to_datetime("2024-08-26")
    stocked_fish = 7290
    initial_abw = 11.9
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': stocked_fish,
        'AVERAGE BODY WEIGHT (g)': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE')

    # Limit timeframe
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]

    # Apply transfers out of Cage 2
    if transfers is not None and not transfers.empty:
        transfers_out = transfers[transfers['ORIGIN CAGE'] == cage_number].copy()
        for idx, row in transfers_out.iterrows():
            date_mask = sampling_c2['DATE'] >= pd.to_datetime(row['DATE'])
            sampling_c2.loc[date_mask, 'NUMBER OF FISH'] -= row['NUMBER_FISH']

    return feeding_c2, harvest_c2, sampling_c2

# ===============================
# 3. Compute Production Summary
# ===============================
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'])
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'])

    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2.sort_values('DATE')[['DATE', 'CUM_FEED']],
        on='DATE'
    )

    # Handle NaN CUM_FEED
    summary['CUM_FEED'] = summary['CUM_FEED'].fillna(method='ffill').fillna(0)
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

    return summary

# ===============================
# 4. Streamlit Interface
# ===============================
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers", type=["xlsx"])

if feeding_file and harvest_file and sampling_file and transfer_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Display production summary table
    st.subheader("Cage 2 Production Summary")
    st.dataframe(summary_c2[['DATE', 'NUMBER OF FISH', 'TOTAL_WEIGHT_KG', 'AGGREGATED_eFCR', 'PERIOD_eFCR']])

    # Plot Growth and eFCR
    kpi = st.selectbox("Select KPI", ["Growth", "eFCR"])
    df = summary_c2.copy()
    if kpi == "Growth":
        df = df.dropna(subset=['TOTAL_WEIGHT_KG'])
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title='Cage 2 Growth Over Time', labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    else:
        df = df.dropna(subset=['AGGREGATED_eFCR', 'PERIOD_eFCR'])
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True)
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title='Cage 2 eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)
