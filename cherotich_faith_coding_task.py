# ==========================
# Fish Cage Production Analysis App
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------
# 1. Load Data Function
# --------------------------
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    if transfer_file is not None:
        transfer = pd.read_excel(transfer_file)
    else:
        transfer = pd.DataFrame()
    return feeding, harvest, sampling, transfer

# --------------------------
# 2. Clean & Prepare Data
# --------------------------
def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df):
    # Drop fully empty columns
    for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
        df.dropna(axis=1, how='all', inplace=True)

    # Standardize column names
    def standardize_columns(df):
        df.columns = (
            df.columns
            .astype(str)
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

    # Numeric conversion
    for col in ['feed_amount_kg', 'total_weight_kg', 'abw_g', 'number_of_fish', 'average_body_weight_g']:
        for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Date conversion
    for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return feeding_df, harvest_df, sampling_df, transfer_df

# --------------------------
# 3. Preprocess Cage 2
# --------------------------
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")

    feeding_c2 = feeding[(feeding['cage_number']==cage_number) &
                         (feeding['date']>=start_date) &
                         (feeding['date']<=end_date)].copy()

    harvest_c2 = harvest[(harvest['cage']==cage_number) &
                         (harvest['date']>=start_date) &
                         (harvest['date']<=end_date)].copy()

    sampling_c2 = sampling[(sampling['cage_number']==cage_number) &
                           (sampling['date']>=start_date) &
                           (sampling['date']<=end_date)].copy()

    # Ensure numeric
    for col in ['number_of_fish','average_body_weight_g']:
        if col in sampling_c2.columns:
            sampling_c2[col] = pd.to_numeric(sampling_c2[col], errors='coerce').fillna(0)

    # Manual stocking
    stocking_row = pd.DataFrame([{
        'date': start_date,
        'cage_number': cage_number,
        'number_of_fish': 7290,
        'average_body_weight_g': 11.9
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('date').reset_index(drop=True)

    # Transfers
    sampling_c2['IN_FISH'] = 0
    sampling_c2['OUT_FISH'] = 0
    if transfers is not None and not transfers.empty:
        transfers_c2 = transfers[(transfers['date']>=start_date) & (transfers['date']<=end_date)]
        # Outgoing
        out_mask = transfers_c2.get('origin_cage', pd.Series()) == cage_number
        if out_mask.any():
            out_transfers = transfers_c2[out_mask].groupby('date')['number_of_fish'].sum().cumsum().reset_index()
            out_transfers.rename(columns={'number_of_fish':'OUT_FISH'}, inplace=True)
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'), out_transfers.sort_values('date'),
                                        on='date', direction='backward')
        # Incoming
        in_mask = transfers_c2.get('destination_cage', pd.Series()) == cage_number
        if in_mask.any():
            in_transfers = transfers_c2[in_mask].groupby('date')['number_of_fish'].sum().cumsum().reset_index()
            in_transfers.rename(columns={'number_of_fish':'IN_FISH'}, inplace=True)
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'), in_transfers.sort_values('date'),
                                        on='date', direction='backward')

        sampling_c2[['IN_FISH','OUT_FISH']] = sampling_c2[['IN_FISH','OUT_FISH']].fillna(0)

    # Compute fish alive & biomass
    sampling_c2['FISH_ALIVE'] = (sampling_c2['number_of_fish'] + sampling_c2['IN_FISH'] - sampling_c2['OUT_FISH']).clip(lower=0)
    sampling_c2['BIOMASS_KG'] = sampling_c2['FISH_ALIVE'] * sampling_c2['average_body_weight_g'] / 1000

    return feeding_c2, harvest_c2, sampling_c2

# --------------------------
# 4. Production Summary
# --------------------------
def cage2_production_summary(feeding_c2, sampling_c2, harvest_c2):
    df = sampling_c2.copy()
    df = df.sort_values('date').reset_index(drop=True)
    feeding_c2 = feeding_c2.sort_values('date')

    # Period feed
    df['PERIOD_FEED_KG'] = 0
    for i in range(1,len(df)):
        start,end = df.loc[i-1,'date'], df.loc[i,'date']
        mask = (feeding_c2['date']>start) & (feeding_c2['date']<=end)
        df.loc[i,'PERIOD_FEED_KG'] = feeding_c2.loc[mask,'feed_amount_kg'].sum()

    df['CUM_FEED_KG'] = df['PERIOD_FEED_KG'].cumsum()
    df['PREV_BIOMASS_KG'] = df['BIOMASS_KG'].shift(1).fillna(0)
    df['DELTA_BIOMASS_KG'] = df['BIOMASS_KG'] - df['PREV_BIOMASS_KG']

    df['PERIOD_eFCR'] = df.apply(lambda row: row['PERIOD_FEED_KG']/row['DELTA_BIOMASS_KG']
                                 if row['DELTA_BIOMASS_KG']>0 else np.nan, axis=1)
    df['AGGREGATED_eFCR'] = df['CUM_FEED_KG']/df['BIOMASS_KG'].replace(0,np.nan)
    df['AVERAGE_BODY_WEIGHT_G'] = df['average_body_weight_g']
    df['FISH_ALIVE'] = pd.to_numeric(df['FISH_ALIVE'], errors='coerce').fillna(0)

    summary = df[['date','FISH_ALIVE','AVERAGE_BODY_WEIGHT_G','BIOMASS_KG',
                  'PERIOD_FEED_KG','CUM_FEED_KG','DELTA_BIOMASS_KG','PERIOD_eFCR','AGGREGATED_eFCR']]
    return summary

# --------------------------
# 5. Generate Mock Cages
# --------------------------
def generate_mock_cages(feeding_c2, sampling_c2, harvest_c2, num_cages=5):
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = [],[],[],{}
    for cage in range(3,3+num_cages):
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

        # Production summary
        summary = cage2_production_summary(f,s,h)
        mock_summaries[cage] = summary
    return mock_feeding,mock_sampling,mock_harvest,mock_summaries

# --------------------------
# 6. Streamlit Interface
# --------------------------
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfer = clean_and_prepare(
        pd.read_excel(feeding_file),
        pd.read_excel(harvest_file),
        pd.read_excel(sampling_file),
        pd.read_excel(transfer_file) if transfer_file else pd.DataFrame()
    )

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfer)
    summary_c2 = cage2_production_summary(feeding_c2, sampling_c2, harvest_c2)
    _, _, _, mock_summaries = generate_mock_cages(feeding_c2, sampling_c2, harvest_c2)

    all_cages = {2: summary_c2, **mock_summaries}

    # Sidebar selectors
    st.sidebar.header("Options")
    selected_cage = st.sidebar.selectbox("Select Cage", sorted(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage].copy()
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[['date','FISH_ALIVE','BIOMASS_KG','PERIOD_FEED_KG',
                     'CUM_FEED_KG','PERIOD_eFCR','AGGREGATED_eFCR']].round(2))

    # Plot KPI
    if selected_kpi=="Growth":
        fig = px.line(df, x='date', y='BIOMASS_KG', markers=True,
                      title=f'Cage {selected_cage} Biomass Growth',
                      labels={'BIOMASS_KG':'Biomass (Kg)','date':'Date'})
        st.plotly_chart(fig)
    else:
        # Ensure numeric & drop NaNs
        df_plot = df.copy()
        df_plot['AGGREGATED_eFCR'] = pd.to_numeric(df_plot['AGGREGATED_eFCR'], errors='coerce')
        df_plot['PERIOD_eFCR'] = pd.to_numeric(df_plot['PERIOD_eFCR'], errors='coerce')
        df_plot = df_plot.dropna(subset=['AGGREGATED_eFCR','PERIOD_eFCR'], how='all')

        if not df_plot.empty:
            fig = px.line(df_plot, x='date', y='AGGREGATED_eFCR', markers=True, name='Aggregated eFCR')
            fig.add_scatter(x=df_plot['date'], y=df_plot['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
            fig.update_layout(title=f'Cage {selected_cage} eFCR Over Time',
                              yaxis_title='eFCR', xaxis_title='Date')
            st.plotly_chart(fig)
        else:
            st.warning("No eFCR data available to plot for this cage.")
else:
    st.info("Please upload all required Excel files to start the analysis.")
