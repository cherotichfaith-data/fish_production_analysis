# app.py — Fish Cage Production Analysis

# ==========================
# 0. Import Libraries
# ==========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ==========================
# 1. Load Data
# ==========================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfer = pd.read_excel(transfer_file)
    return feeding, harvest, sampling, transfer


# ==========================
# 2. Check & Handle Missing Values
# ==========================
def inspect_missing_values(df_dict):
    """Print missing values summary for each dataframe."""
    for name, df in df_dict.items():
        st.write(f"### Missing values in {name} dataframe")
        st.write(df.isnull().sum())


def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df):
    """Basic cleaning: drop empty cols, fill important fields."""
    # Drop empty columns
    feeding_df = feeding_df.dropna(axis=1, how="all")
    harvest_df = harvest_df.dropna(axis=1, how="all")
    sampling_df = sampling_df.dropna(axis=1, how="all")
    transfer_df = transfer_df.dropna(axis=1, how="all")

    # Fill missing feeding values
    if "FEED AMOUNT (Kg)" in feeding_df.columns:
        feeding_df["FEED AMOUNT (Kg)"] = feeding_df["FEED AMOUNT (Kg)"].fillna(0)
    if "FEEDING TYPE" in feeding_df.columns:
        feeding_df["FEEDING TYPE"] = feeding_df["FEEDING TYPE"].fillna("Unknown")
    if "feeding_response [very good, good, bad]" in feeding_df.columns:
        feeding_df["feeding_response [very good, good, bad]"] = feeding_df[
            "feeding_response [very good, good, bad]"
        ].fillna("Unknown")

    # Fill missing transfer values
    if "Total weight [kg]" in transfer_df.columns:
        transfer_df["Total weight [kg]"] = transfer_df["Total weight [kg]"].fillna(0)
    if "ABW [g]" in transfer_df.columns:
        transfer_df["ABW [g]"] = transfer_df["ABW [g]"].fillna(0)

    return feeding_df, harvest_df, sampling_df, transfer_df


# ==========================
# 3. Compute Production Summary (UNCHANGED)
# ==========================
def compute_production_summary(feeding_df, harvest_df, sampling_df, transfer_df):
    summary = {}

    # Feeding aggregates
    feed_summary = feeding_df.groupby("CAGE NUMBER")["FEED AMOUNT (Kg)"].sum().reset_index()
    summary["feed"] = feed_summary

    # Harvest aggregates
    harvest_summary = harvest_df.groupby("CAGE")[["NUMBER OF FISH", "TOTAL WEIGHT  [kg]"]].sum().reset_index()
    summary["harvest"] = harvest_summary

    # Sampling aggregates
    if sampling_df.empty:
        st.write("⚠️ No sampling data available. Skipping growth analysis.")
    else:
        sampling_summary = sampling_df.groupby("CAGE NUMBER")["AVERAGE BODY WEIGHT(g)"].mean().reset_index()
        summary["sampling"] = sampling_summary

    # Transfer aggregates
    transfer_summary = transfer_df.groupby("DESTINATION CAGE")["NUMBER OF FISH"].sum().reset_index()
    summary["transfer"] = transfer_summary

    # Compute EFCR: Feed given / Harvested weight
    efcr_df = pd.merge(feed_summary, harvest_summary, left_on="CAGE NUMBER", right_on="CAGE", how="left")
    efcr_df["EFCR"] = efcr_df["FEED AMOUNT (Kg)"] / efcr_df["TOTAL WEIGHT  [kg]"].replace(0, pd.NA)
    summary["efcr"] = efcr_df

    return summary


# ==========================
# 4. Streamlit Interface
# ==========================
st.title("🐟 Fish Cage Production Analysis")

# Sidebar uploaders
st.sidebar.header("Upload Excel Files")
feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer", type=["xlsx"])

if feeding_file and harvest_file and sampling_file and transfer_file:
    # Load raw data
    feeding, harvest, sampling, transfer = load_data(feeding_file, harvest_file, sampling_file, transfer_file)

    # Inspect missing values
    st.subheader("🔍 Missing Values Check")
    inspect_missing_values({
        "Feeding": feeding,
        "Harvest": harvest,
        "Sampling": sampling,
        "Transfer": transfer,
    })

    # Clean & prepare data
    feeding, harvest, sampling, transfer = clean_and_prepare(feeding, harvest, sampling, transfer)

    # Compute production summary
    st.subheader("📊 Production Summary")
    summary = compute_production_summary(feeding, harvest, sampling, transfer)

    # Display results
    for key, df in summary.items():
        st.write(f"#### {key.capitalize()} Summary")
        st.dataframe(df)
