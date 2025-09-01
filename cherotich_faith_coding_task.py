# -*- coding: utf-8 -*- 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Data Cleaning and Preparation
# -------------------------------
def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df=None, scale_numeric=False, show_eda=False):
    for df in [feeding_df, harvest_df, sampling_df] + ([transfer_df] if transfer_df is not None else []):
        df.dropna(axis=1, how='all', inplace=True)

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
    if transfer_df is not None:
        transfer_df = standardize_columns(transfer_df)

    # Fill missing numeric/text values
    if 'feed_amount_kg' in feeding_df.columns:
        feeding_df['feed_amount_kg'].fillna(0, inplace=True)
    for col in ['number_of_fish','average_body_weight_g']:
        if col in sampling_df.columns:
            sampling_df[col].fillna(0, inplace=True)
    if transfer_df is not None:
        for col in ['total_weight_kg','abw_g']:
            if col in transfer_df.columns:
                transfer_df[col].fillna(0, inplace=True)

    # Convert numeric columns
    for df, numeric_cols in zip([feeding_df, sampling_df, transfer_df] if transfer_df is not None else [feeding_df, sampling_df],
                                [['feed_amount_kg'], ['number_of_fish','average_body_weight_g'], ['total_weight_kg','abw_g'] if transfer_df is not None else []]):
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert dates
    for df in [feeding_df, harvest_df, sampling_df] + ([transfer_df] if transfer_df is not None else []):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Remove duplicates
    for df in [feeding_df, harvest_df, sampling_df] + ([transfer_df] if transfer_df is not None else []):
        df.drop_duplicates(inplace=True)

    report = {
        'feeding_rows': len(feeding_df),
        'harvest_rows': len(harvest_df),
        'sampling_rows': len(sampling_df),
        'transfer_rows': len(transfer_df) if transfer_df is not None else 0
    }

    return feeding_df, harvest_df, sampling_df, transfer_df, report

# -------------------------------
# Preprocess Cage 2
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling):
    cage_number = 2

    feeding_c2 = feeding[feeding['cage_number'] == cage_number].copy()
    harvest_c2 = harvest[harvest['cage'] == cage_number].copy()
    sampling_c2 = sampling[sampling['cage_number'] == cage_number].copy()

    # Add manual stocking
    stocking_date = pd.to_datetime("2024-08-26")
    stocked_fish = 7290
    initial_abw = 11.9
    stocking_row = pd.DataFrame([{
        'date': stocking_date,
        'cage_number': cage_number,
        'number_of_fish': stocked_fish,
        'average_body_weight_g': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('date')

    # Limit timeframe
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    sampling_c2 = sampling_c2[(sampling_c2['date'] >= start_date) & (sampling_c2['date'] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2['date'] >= start_date) & (feeding_c2['date'] <= end_date)]

    return feeding_c2, harvest_c2, sampling_c2

# -------------------------------
# Compute summary
# -------------------------------
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2['date'] = pd.to_datetime(feeding_c2['date'])
    sampling_c2['date'] = pd.to_datetime(sampling_c2['date'])

    feeding_c2 = feeding_c2.sort_values('date')
    feeding_c2['cum_feed'] = feeding_c2['feed_amount_kg'].cumsum()

    sampling_c2 = sampling_c2.sort_values('date')
    sampling_c2['total_weight_kg'] = sampling_c2['number_of_fish'] * sampling_c2['average_body_weight_g'] / 1000

    summary = pd.merge_asof(
        sampling_c2,
        feeding_c2[['date','cum_feed']],
        on='date'
    )
    summary['cum_feed'] = summary['cum_feed'].fillna(method='ffill').fillna(0)
    summary['aggregated_efcr'] = summary['cum_feed'] / summary['total_weight_kg']
    summary['period_weight_gain'] = summary['total_weight_kg'].diff().fillna(summary['total_weight_kg'])
    summary['period_feed'] = summary['cum_feed'].diff().fillna(summary['cum_feed'])
    summary['period_efcr'] = summary['period_feed'] / summary['period_weight_gain']
    summary.replace([np.inf, -np.inf], np.nan, inplace=True)
    summary.fillna(0, inplace=True)

    return summary

# -------------------------------
# Streamlit App
# -------------------------------
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers (Optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding_df = pd.read_excel(feeding_file)
    harvest_df = pd.read_excel(harvest_file)
    sampling_df = pd.read_excel(sampling_file)
    transfer_df = pd.read_excel(transfer_file) if transfer_file else None

    feeding_clean, harvest_clean, sampling_clean, transfer_clean, report = clean_and_prepare(
        feeding_df, harvest_df, sampling_df, transfer_df, scale_numeric=False, show_eda=False
    )

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding_clean, harvest_clean, sampling_clean)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 Production Summary")
    st.dataframe(summary_c2[['date','number_of_fish','total_weight_kg','aggregated_efcr','period_efcr']])
