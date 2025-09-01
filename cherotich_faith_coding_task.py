# =========================================
# Fish Cage Production Analysis - Streamlit
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================
# 1. Load Data Function
# =========================================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    """Load Excel files into pandas DataFrames."""
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfer = pd.read_excel(transfer_file) if transfer_file else pd.DataFrame()
    return feeding, harvest, sampling, transfer

# =========================================
# 2. Clean and Prepare Data
# =========================================
def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df=None):
    """Cleans and prepares fish production datasets."""
    transfer_df = pd.DataFrame() if transfer_df is None else transfer_df

    # Drop empty columns
    for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
        if df is not None and not df.empty:
            df.dropna(axis=1, how='all', inplace=True)

    # Standardize column names
    def standardize_columns(df):
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
        )
        return df

    feeding_df = standardize_columns(feeding_df)
    harvest_df = standardize_columns(harvest_df)
    sampling_df = standardize_columns(sampling_df)
    transfer_df = standardize_columns(transfer_df)

    # Convert numeric columns
    for col in ['feed_amount_kg', 'total_weight_kg', 'abw_g', 'number_of_fish']:
        for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert dates
    for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Fill missing text values
    if 'feeding_type' in feeding_df.columns:
        feeding_df['feeding_type'] = feeding_df['feeding_type'].fillna('Unknown').str.strip().str.title()

    return feeding_df, harvest_df, sampling_df, transfer_df

# =========================================
# 3. Preprocess Cage 2
# =========================================
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    feeding_c2  = feeding[(feeding.get('cage_number', 0) == cage_number) &
                          (feeding['date'] >= start_date) & (feeding['date'] <= end_date)].copy()
    harvest_c2  = harvest[(harvest.get('cage', 0) == cage_number) &
                          (harvest['date'] >= start_date) & (harvest['date'] <= end_date)].copy()
    sampling_c2 = sampling[(sampling.get('cage_number', 0) == cage_number) &
                           (sampling['date'] >= start_date) & (sampling['date'] <= end_date)].copy()

    # Ensure numeric
    for col in ['number_of_fish', 'average_body_weight_g']:
        if col in sampling_c2.columns:
            sampling_c2[col] = pd.to_numeric(sampling_c2[col], errors='coerce').fillna(0)

    # Add manual stocking
    stocking_row = pd.DataFrame([{
        'date': start_date,
        'cage_number': cage_number,
        'number_of_fish': 7290,
        'average_body_weight_g': 11.9
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('date').reset_index(drop=True)

    # Initialize transfer columns
    sampling_c2['IN_FISH'] = 0
    sampling_c2['OUT_FISH'] = 0

    # Apply transfers
    if transfers is not None and not transfers.empty:
        transfers_c2 = transfers[(transfers['date'] >= start_date) & (transfers['date'] <= end_date)]

        # Outgoing
        out_mask = transfers_c2.get('origin_cage', 0) == cage_number
        if out_mask.any():
            out_transfers = (transfers_c2[out_mask]
                             .groupby('date')['number_of_fish']
                             .sum()
                             .cumsum()
                             .reset_index()
                             .rename(columns={'number_of_fish':'OUT_FISH'}))
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'),
                                        out_transfers.sort_values('date'),
                                        on='date', direction='backward')

        # Incoming
        in_mask = transfers_c2.get('destination_cage', 0) == cage_number
        if in_mask.any():
            in_transfers = (transfers_c2[in_mask]
                            .groupby('date')['number_of_fish']
                            .sum()
                            .cumsum()
                            .reset_index()
                            .rename(columns={'number_of_fish':'IN_FISH'}))
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'),
                                        in_transfers.sort_values('date'),
                                        on='date', direction='backward')

        sampling_c2[['IN_FISH','OUT_FISH']] = sampling_c2[['IN_FISH','OUT_FISH']].fillna(0)

    # Compute standing fish & biomass
    sampling_c2['FISH_ALIVE'] = (sampling_c2['number_of_fish'] + sampling_c2['IN_FISH'] - sampling_c2['OUT_FISH']).clip(lower=0)
    sampling_c2['BIOMASS_KG'] = sampling_c2['FISH_ALIVE'] * sampling_c2['average_body_weight_g'] / 1000

    return feeding_c2.sort_values('date').reset_index(drop=True), \
           harvest_c2.sort_values('date').reset_index(drop=True), \
           sampling_c2.sort_values('date').reset_index(drop=True)

# =========================================
# 4. Cage 2 Production Summary
# =========================================
def cage2_production_summary(feeding_c2, sampling_c2, harvest_c2):
    df = sampling_c2.copy().sort_values('date').reset_index(drop=True)
    feeding_c2 = feeding_c2.sort_values('date')

    # Align feeding
    df['PERIOD_FEED_KG'] = 0
    for i in range(1, len(df)):
        start, end = df.loc[i-1, 'date'], df.loc[i, 'date']
        mask = (feeding_c2['date'] > start) & (feeding_c2['date'] <= end)
        df.loc[i, 'PERIOD_FEED_KG'] = feeding_c2.loc[mask, 'feed_amount_kg'].sum()

    # Cumulative feed
    df['CUM_FEED_KG'] = df['PERIOD_FEED_KG'].cumsum()

    # Biomass changes
    df['PREV_BIOMASS_KG'] = df['BIOMASS_KG'].shift(1).fillna(0)
    df['DELTA_BIOMASS_KG'] = df['BIOMASS_KG'] - df['PREV_BIOMASS_KG']

    # eFCR
    df['PERIOD_eFCR'] = df.apply(lambda r: r['PERIOD_FEED_KG']/r['DELTA_BIOMASS_KG'] if r['DELTA_BIOMASS_KG']>0 else np.nan, axis=1)
    df['AGGREGATED_eFCR'] = df['CUM_FEED_KG'] / df['BIOMASS_KG'].replace(0,np.nan)

    # Growth
    df['AVERAGE_BODY_WEIGHT_G'] = df['average_body_weight_g']
    df['FISH_ALIVE'] = pd.to_numeric(df['FISH_ALIVE'], errors='coerce').fillna(0)

    return df[['date','FISH_ALIVE','AVERAGE_BODY_WEIGHT_G','BIOMASS_KG','PERIOD_FEED_KG','CUM_FEED_KG','DELTA_BIOMASS_KG','PERIOD_eFCR','AGGREGATED_eFCR']]

# =========================================
# 5. Generate Mock Cages (3-7)
# =========================================
def generate_mock_cages(feeding_c2, sampling_c2, harvest_c2, num_cages=5):
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = [], [], [], {}

    for cage in range(3, 3 + num_cages):
        # Feeding
        f = feeding_c2.copy()
        f['cage_number'] = cage
        if 'feed_amount_kg' in f.columns:
            f['feed_amount_kg'] *= np.random.uniform(0.9,1.1,size=len(f))
        mock_feeding.append(f)

        # Sampling
        s = sampling_c2.copy()
        s['cage_number'] = cage
        if 'average_body_weight_g' in s.columns:
            s['average_body_weight_g'] *= np.random.uniform(0.95,1.05,size=len(s))
        s['FISH_ALIVE'] = pd.to_numeric(s['FISH_ALIVE'], errors='coerce').fillna(0)
        s['BIOMASS_KG'] = s['FISH_ALIVE']*s['average_body_weight_g']/1000
        mock_sampling.append(s)

        # Harvest
        h = harvest_c2.copy()
        h['cage'] = cage
        if 'total_weight_kg' in h.columns:
            h['total_weight_kg'] *= np.random.uniform(0.95,1.05,size=len(h))
        if 'number_of_fish' in h.columns:
            h['number_of_fish'] = pd.to_numeric(h['number_of_fish'], errors='coerce').fillna(0)
        mock_harvest.append(h)

        # Summary
        summary = cage2_production_summary(f, s, h)
        mock_summaries[cage] = summary

    return mock_feeding, mock_sampling, mock_harvest, mock_summaries

# =========================================
# 6. Streamlit UI
# =========================================
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfer = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding, harvest, sampling, transfer = clean_and_prepare(feeding, harvest, sampling, transfer)

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfer)
    summary_c2 = cage2_production_summary(feeding_c2, sampling_c2, harvest_c2)
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = generate_mock_cages(feeding_c2, sampling_c2, harvest_c2)

    all_cages = {2: summary_c2, **mock_summaries}

    st.sidebar.header("Options")
    selected_cage = st.sidebar.selectbox("Select Cage", sorted(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage].copy()
    df.columns = [c.upper() for c in df.columns]

    # Table
    display_cols = ['DATE','FISH_ALIVE','BIOMASS_KG','PERIOD_FEED_KG','CUM_FEED_KG','PERIOD_EFCR','AGGREGATED_EFCR']
    df_display = df[[col for col in display_cols if col in df.columns]].round(2)
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df_display)

    # Graph
    if selected_kpi == "Growth":
        fig = px.line(df, x='DATE', y='BIOMASS_KG', markers=True,
                      title=f'Cage {selected_cage}: Biomass Growth Over Time',
                      labels={'BIOMASS_KG':'Biomass (Kg)', 'DATE':'Date'})
        st.plotly_chart(fig)
    else:
        fig = px.line(df, x='DATE', y='AGGREGATED_EFCR', markers=True, name='Aggregated eFCR')
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_EFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR', xaxis_title='Date')
        st.plotly_chart(fig)
else:
    st.info("Please upload all required Excel files to start the analysis.")
