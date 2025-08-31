# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# 1. Load data
# -------------------------------
def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfers = pd.read_excel(transfer_file)

    # Handle column name differences in transfers
    weight_col_candidates = ["WEIGHT", "WEIGHT_KG", "Total weight [kg]", "AMOUNT", "FEED AMOUNT (Kg)"]
    weight_col = None
    for col in weight_col_candidates:
        if col in transfers.columns:
            weight_col = col
            break
    if weight_col is None:
        raise KeyError(f"No weight column found in transfers. Columns: {transfers.columns.tolist()}")
    
    # Convert g -> kg if necessary
    if transfers[weight_col].max() > 200:  # assume >200 means grams
        transfers[weight_col] = transfers[weight_col] / 1000
    transfers = transfers.rename(columns={weight_col: "WEIGHT"})

    return feeding, harvest, sampling, transfers

# -------------------------------
# 2. Handle missing values
# -------------------------------
def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df):
    # Drop fully empty columns
    feeding_df = feeding_df.dropna(axis=1, how="all")
    harvest_df = harvest_df.dropna(axis=1, how="all")
    sampling_df = sampling_df.dropna(axis=1, how="all")
    transfer_df = transfer_df.dropna(axis=1, how="all")

    # Fill non-critical missing values
    feeding_df["FEED AMOUNT (Kg)"] = feeding_df["FEED AMOUNT (Kg)"].fillna(0)
    feeding_df["FEEDING TYPE"] = feeding_df.get("FEEDING TYPE", "Unknown").fillna("Unknown")
    feeding_df["feeding_response [very good, good, bad]"] = feeding_df.get("feeding_response [very good, good, bad]", "Unknown").fillna("Unknown")

    transfer_df["NUMBER OF FISH"] = transfer_df.get("NUMBER OF FISH", 0).fillna(0)
    transfer_df["ABW [g]"] = transfer_df.get("ABW [g]", 0).fillna(0)

    return feeding_df, harvest_df, sampling_df, transfer_df

# -------------------------------
# 3. Preprocess Cage 2
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling, transfers):
    cage_number = 2

    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()
    transfers_c2 = transfers[transfers['FROM CAGE'] == cage_number].copy()

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

    # Apply transfers (outgoing)
    for _, row in transfers_c2.iterrows():
        transfer_date = pd.to_datetime(row['DATE'])
        transferred_fish = row['NUMBER OF FISH']
        sampling_c2.loc[sampling_c2['DATE'] >= transfer_date, 'NUMBER OF FISH'] -= transferred_fish

    # Limit timeframe (final harvest = 09 July 2025)
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]

    return feeding_c2, harvest_c2, sampling_c2

# -------------------------------
# 4. Compute production summary
# -------------------------------
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

    summary['CUM_FEED'] = summary['CUM_FEED'].fillna(method='ffill').fillna(0)
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

    return summary

# -------------------------------
# 5. Streamlit interface
# -------------------------------
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers", type=["xlsx"])

if feeding_file and harvest_file and sampling_file and transfer_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    
    # Check for missing values
    st.write("### Missing Values")
    st.write("Feeding dataframe:", feeding.isnull().sum())
    st.write("Harvest dataframe:", harvest.isnull().sum())
    st.write("Sampling dataframe:", sampling.isnull().sum())
    st.write("Transfer dataframe:", transfers.isnull().sum())

    # Handle missing values
    feeding, harvest, sampling, transfers = clean_and_prepare(feeding, harvest, sampling, transfers)

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Sidebar selector
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])
    df = summary_c2

    # Display production summary
    st.subheader("Cage 2 Production Summary")
    st.dataframe(df[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']])

    # Plot graphs
    if selected_kpi == "Growth":
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title='Cage 2: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    else:
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True, name='Cumulative eFCR')
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title='Cage 2: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)
