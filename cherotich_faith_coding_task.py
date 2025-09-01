# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# 1. Load data
# -------------------------------
@st.cache_data
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)

    transfers = None
    if transfer_file:
        transfers = pd.read_excel(transfer_file)
        transfers['DATE'] = pd.to_datetime(transfers['DATE'])
        # Convert weights from g -> kg
        transfers['TOTAL_WEIGHT_KG'] = transfers['NUMBER_OF_FISH'] * transfers['AVERAGE_BODY_WEIGHT (g)'] / 1000
    return feeding, harvest, sampling, transfers

# -------------------------------
# 2. Preprocess Cage 2 with Transfers
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2

    # Filter cage 2
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Stocking event
    stocking_date = pd.to_datetime("2024-08-26")
    stocked_fish = 7290
    initial_abw = 11.9
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': stocked_fish,
        'AVERAGE BODY WEIGHT (g)': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE').reset_index(drop=True)

    # Limit analysis window
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)].reset_index(drop=True)
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)].reset_index(drop=True)

    # Initialize biomass column
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF_FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    # Apply transfers
    if transfers is not None and not transfers.empty:
        cage_transfers = transfers[(transfers['CAGE_FROM'] == cage_number) | (transfers['CAGE_TO'] == cage_number)].copy()
        cage_transfers.sort_values('DATE', inplace=True)

        for _, row in cage_transfers.iterrows():
            date = row['DATE']
            if row['CAGE_FROM'] == cage_number:
                mask = sampling_c2['DATE'] >= date
                sampling_c2.loc[mask, 'NUMBER OF FISH'] = (sampling_c2.loc[mask, 'NUMBER OF FISH'] - row['NUMBER_OF_FISH']).clip(lower=0)
            if row['CAGE_TO'] == cage_number:
                mask = sampling_c2['DATE'] >= date
                sampling_c2.loc[mask, 'NUMBER OF FISH'] += row['NUMBER_OF_FISH']

        # Recalculate TOTAL_WEIGHT_KG after transfers
        sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF_FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    return feeding_c2, harvest_c2, sampling_c2

# -------------------------------
# 3. Compute summary (eFCR + Growth + Biomass)
# -------------------------------
def compute_summary(feeding_c2, sampling_c2):
    # Ensure datetime
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'])
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'])

    feeding_c2 = feeding_c2.sort_values('DATE')
    sampling_c2 = sampling_c2.sort_values('DATE').reset_index(drop=True)

    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()

    # Merge cumulative feed
    summary = pd.merge_asof(sampling_c2, feeding_c2[['DATE','CUM_FEED']], on='DATE', direction='backward')
    summary['CUM_FEED'] = summary['CUM_FEED'].fillna(0)

    # eFCR
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG'].replace(0,np.nan)
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN'].replace(0,np.nan)

    return summary

# -------------------------------
# 4. Create mock cages (3-7)
# -------------------------------
def create_mock_cages(summary_c2, feeding_c2, sampling_c2):
    mock_summaries = {}
    cage_ids = range(3,8)
    sampling_dates = sampling_c2['DATE'].tolist()
    n_dates = len(sampling_dates)

    start_date = feeding_c2['DATE'].min() if not feeding_c2.empty else sampling_c2['DATE'].min()
    end_date = feeding_c2['DATE'].max() if not feeding_c2.empty else sampling_c2['DATE'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for cage_id in cage_ids:
        daily_feed = pd.DataFrame({
            'DATE': date_range,
            'CAGE NUMBER': cage_id,
            'FEED AMOUNT (Kg)': np.random.normal(10,1,len(date_range))
        })
        daily_feed['CUM_FEED'] = daily_feed['FEED AMOUNT (Kg)'].cumsum()

        mock_sampling = pd.DataFrame({
            'DATE': sampling_dates,
            'CAGE NUMBER': cage_id,
            'NUMBER OF FISH': (summary_c2['NUMBER OF FISH'].values[:n_dates] + np.random.randint(-50,50,n_dates)).clip(min=1),
            'AVERAGE BODY WEIGHT (g)': (summary_c2['AVERAGE BODY WEIGHT (g)'].values[:n_dates] * np.random.normal(1,0.05,n_dates)).clip(min=0.1)
        })
        mock_sampling['TOTAL_WEIGHT_KG'] = mock_sampling['NUMBER OF FISH'] * mock_sampling['AVERAGE BODY WEIGHT (g)'] / 1000

        summary = pd.merge_asof(mock_sampling.sort_values('DATE'), daily_feed[['DATE','CUM_FEED']], on='DATE', direction='backward')
        summary['CUM_FEED'] = summary['CUM_FEED'].fillna(0)
        summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG'].replace(0,np.nan)
        summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
        summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
        summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN'].replace(0,np.nan)

        mock_summaries[cage_id] = summary

    return mock_summaries

# -------------------------------
# 5. Streamlit interface
# -------------------------------
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    mock_cages = create_mock_cages(summary_c2, feeding_c2, sampling_c2)
    all_cages = {2: summary_c2, **mock_cages}

    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth","eFCR","Biomass"])

    df = all_cages[selected_cage]

    # Display summary table
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']].reset_index(drop=True))

    # Plot KPI
    if selected_kpi == "Growth":
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG':'Total Weight (Kg)'} )
        st.plotly_chart(fig)
    elif selected_kpi == "eFCR":
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True)
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)
    else:  # Biomass
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Biomass Over Time',
                      labels={'TOTAL_WEIGHT_KG':'Biomass (Kg)'})
        st.plotly_chart(fig)
