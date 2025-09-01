# ==========================
# Fish Cage Production Analysis (Cage 2, with Transfers)
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------
# 1. Load and clean data safely
# --------------------------
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    def safe_read(file):
        df = pd.read_excel(file)
        # Normalize column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
        )
        # Convert any 'date' column
        for c in df.columns:
            if 'date' in c:
                df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)
        return df

    feeding = safe_read(feeding_file)
    harvest = safe_read(harvest_file)
    sampling = safe_read(sampling_file)
    transfers = safe_read(transfer_file) if transfer_file else pd.DataFrame()
    return feeding, harvest, sampling, transfers

# --------------------------
# 2. Preprocess Cage 2
# --------------------------
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # Filter cage 2
    feeding_c2 = feeding[(feeding.get('cage_number',0)==cage_number) & 
                         (feeding['date']>=start_date) & (feeding['date']<=end_date)].copy()
    harvest_c2 = harvest[(harvest.get('cage',0)==cage_number) & 
                         (harvest['date']>=start_date) & (harvest['date']<=end_date)].copy()
    sampling_c2 = sampling[(sampling.get('cage_number',0)==cage_number) & 
                           (sampling['date']>=start_date) & (sampling['date']<=end_date)].copy()

    # Ensure numeric
    for col in ['number_of_fish','abw_g','total_weight_kg']:
        if col in sampling_c2.columns:
            sampling_c2[col] = pd.to_numeric(sampling_c2[col], errors='coerce').fillna(0)

    # Initial stocking
    stocking_row = pd.DataFrame([{
        'date': start_date,
        'cage_number': cage_number,
        'number_of_fish': 7290,
        'abw_g': 11.9
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('date').reset_index(drop=True)

    # Transfers
    sampling_c2['transfer_in_fish'] = 0
    sampling_c2['transfer_out_fish'] = 0
    if transfers is not None and not transfers.empty:
        transfers_c2 = transfers.copy()
        in_mask  = transfers_c2.get('destination_cage', pd.Series()) == cage_number
        out_mask = transfers_c2.get('origin_cage', pd.Series()) == cage_number

        if out_mask.any():
            out_transfers = transfers_c2[out_mask].groupby('date')['number_of_fish'].sum().cumsum().reset_index()
            out_transfers.rename(columns={'number_of_fish':'transfer_out_fish'}, inplace=True)
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'), out_transfers.sort_values('date'),
                                        on='date', direction='backward')
        if in_mask.any():
            in_transfers = transfers_c2[in_mask].groupby('date')['number_of_fish'].sum().cumsum().reset_index()
            in_transfers.rename(columns={'number_of_fish':'transfer_in_fish'}, inplace=True)
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'), in_transfers.sort_values('date'),
                                        on='date', direction='backward')
        sampling_c2[['transfer_in_fish','transfer_out_fish']] = sampling_c2[['transfer_in_fish','transfer_out_fish']].fillna(0)

    # Fish alive & biomass
    sampling_c2['fish_alive'] = (sampling_c2['number_of_fish'] + sampling_c2['transfer_in_fish'] - sampling_c2['transfer_out_fish']).clip(lower=0)
    sampling_c2['biomass_kg'] = sampling_c2['fish_alive'] * sampling_c2['abw_g'] / 1000

    return feeding_c2, harvest_c2, sampling_c2

# --------------------------
# 3. Compute Production Summary
# --------------------------
def compute_summary(feeding_c2, sampling_c2, harvest_c2=None):
    df = sampling_c2.sort_values('date').reset_index(drop=True)
    df['feed_period_kg'] = 0
    df['feed_agg_kg'] = 0
    df['growth_kg'] = 0
    df['harvest_fish'] = 0
    df['harvest_kg'] = 0
    df['fish_count_discrepancy'] = 0

    feeding_c2 = feeding_c2.sort_values('date')
    if harvest_c2 is not None:
        harvest_c2 = harvest_c2.sort_values('date')

    for i in range(1,len(df)):
        start,end = df.loc[i-1,'date'], df.loc[i,'date']
        mask = (feeding_c2['date']>start) & (feeding_c2['date']<=end)
        df.loc[i,'feed_period_kg'] = feeding_c2.loc[mask,'feed_amount_kg'].sum()
        df.loc[i,'feed_agg_kg'] = df['feed_period_kg'][:i+1].sum()
        df.loc[i,'growth_kg'] = df.loc[i,'biomass_kg'] - df.loc[i-1,'biomass_kg']

        if harvest_c2 is not None and not harvest_c2.empty:
            hmask = (harvest_c2['date']>start) & (harvest_c2['date']<=end)
            df.loc[i,'harvest_fish'] = harvest_c2.loc[hmask,'number_of_fish'].sum()
            df.loc[i,'harvest_kg'] = harvest_c2.loc[hmask,'total_weight_kg'].sum()

        df.loc[i,'fish_count_discrepancy'] = df.loc[i,'number_of_fish'] + df.loc[i,'transfer_in_fish'] - df.loc[i,'transfer_out_fish'] - df.loc[i,'fish_alive'] - df.loc[i,'harvest_fish']

    df.loc[0,'feed_agg_kg'] = df.loc[0,'feed_period_kg'] = df.loc[0,'growth_kg'] = df.loc[0,'biomass_kg']
    df['period_efcr'] = df['feed_period_kg'] / df['growth_kg'].replace(0,np.nan)
    df['aggregated_efcr'] = df['feed_agg_kg'] / df['biomass_kg'].replace(0,np.nan)

    return df

# --------------------------
# 4. Streamlit Interface
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
        "date","number_of_fish","abw_g","biomass_kg",
        "feed_period_kg","feed_agg_kg","growth_kg",
        "transfer_in_fish","transfer_out_fish","harvest_kg",
        "harvest_fish","fish_count_discrepancy",
        "period_efcr","aggregated_efcr",
    ]
    st.dataframe(summary_c2[[c for c in show_cols if c in summary_c2.columns]].round(2))

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])
    if selected_kpi=="Biomass":
        fig = px.line(summary_c2, x='date', y='biomass_kg', markers=True,
                      title="Cage 2: Biomass Over Time", labels={"biomass_kg":"Biomass (kg)"})
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi=="ABW":
        fig = px.line(summary_c2, x='date', y='abw_g', markers=True,
                      title="Cage 2: Average Body Weight Over Time", labels={"abw_g":"ABW (g)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        dff = summary_c2.dropna(subset=["aggregated_efcr","period_efcr"])
        fig = px.line(dff, x='date', y='aggregated_efcr', markers=True,
                      title="Cage 2: eFCR Over Time", labels={"aggregated_efcr":"Aggregated eFCR"})
        fig.add_scatter(x=dff["date"], y=dff["period_efcr"], mode="lines+markers",
                        name="Period eFCR", line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
