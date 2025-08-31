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
def standardize_columns(df):
    """Clean column names: strip spaces, uppercase, normalize brackets."""
    df.columns = (
        df.columns.str.strip()       # remove leading/trailing spaces
        .str.upper()                 # make uppercase
        .str.replace("  ", " ", regex=False)  # remove double spaces
        .str.replace("[", "", regex=False)    # remove [
        .str.replace("]", "", regex=False)    # remove ]
    )
    return df


def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfer = pd.read_excel(transfer_file)

    # Standardize column names
    feeding = standardize_columns(feeding)
    harvest = standardize_columns(harvest)
    sampling = standardize_columns(sampling)
    transfer = standardize_columns(transfer)

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
    if "FEED AMOUNT (KG)" in feeding_df.columns:
        feeding_df["FEED AMOUNT (KG)"] = feeding_df["FEED AMOUNT (KG)"].fillna(0)
    if "FEEDING TYPE" in feeding_df.columns:
        feeding_df["FEEDING TYPE"] = feeding_df["FEEDING TYPE"].fillna("UNKNOWN")
    if "FEEDING_RESPONSE VERY GOOD, GOOD, BAD" in feeding_df.columns:
        feeding_df["FEEDING_RESPONSE VERY GOOD, GOOD, BAD"] = feeding_df[
            "FEEDING_RESPONSE VERY GOOD, GOOD, BAD"
        ].fillna("UNKNOWN")

    # Fill missing transfer values
    if "TOTAL WEIGHT KG" in transfer_df.columns:
        transfer_df["TOTAL WEIGHT KG"] = transfer_df["TOTAL WEIGHT KG"].fillna(0)
    if "ABW G" in transfer_df.columns:
        transfer_df["ABW G"] = transfer_df["ABW G"].fillna(0)

    return feeding_df, harvest_df, sampling_df, transfer_df


# ==========================
# 3. Compute Production Summary
# ==========================
def compute_production_summary(feeding_df, harvest_df, sampling_df, transfer_df):
    summary = {}

    # Feeding aggregates
    if "CAGE NUMBER" in feeding_df.columns and "FEED AMOUNT (KG)" in feeding_df.columns:
        feed_summary = feeding_df.groupby("CAGE NUMBER")["FEED AMOUNT (KG)"].sum().reset_index()
        summary["feed"] = feed_summary

    # Harvest aggregates
    if {"CAGE", "NUMBER OF FISH", "TOTAL WEIGHT KG"}.issubset(harvest_df.columns):
        harvest_summary = harvest_df.groupby("CAGE")[["NUMBER OF FISH", "TOTAL WEIGHT KG"]].sum().reset_index()
        summary["harvest"] = harvest_summary
    else:
        st.warning("⚠️ Harvest file missing expected columns.")

    # Sampling aggregates
    if {"CAGE NUMBER", "AVERAGE BODY WEIGHTG"}.issubset(sampling_df.columns):
        sampling_summary = sampling_df.groupby("CAGE NUMBER")["AVERAGE BODY WEIGHTG"].mean().reset_index()
        summary["sampling"] = sampling_summary
    else:
        st.warning("⚠️ Sampling file missing expected columns.")

    # Transfer aggregates
    if {"DESTINATION CAGE", "NUMBER OF FISH"}.issubset(transfer_df.columns):
        transfer_summary = transfer_df.groupby("DESTINATION CAGE")["NUMBER OF FISH"].sum().reset_index()
        summary["transfer"] = transfer_summary

    # Compute EFCR if both feed & harvest available
    if "feed" in summary and "harvest" in summary:
        efcr_df = pd.merge(
            summary["feed"],
            summary["harvest"],
            left_on="CAGE NUMBER",
            right_on="CAGE",
            how="left"
        )
        efcr_df["EFCR"] = efcr_df["FEED AMOUNT (KG)"] / efcr_df["TOTAL WEIGHT KG"].replace(0, pd.NA)
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
