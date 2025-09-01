# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# -------------------------------
# Helpers
# -------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().upper()) for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates, fuzzy_hint: str | None = None):
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U:
                return orig
    return None

def to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# -------------------------------
# 1. Load and clean data
# -------------------------------
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Convert DATE columns
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df.dropna(subset=["DATE"], inplace=True)

    # Convert cage numbers
    for df, col in [(feeding,"CAGE_NUMBER"), (sampling,"CAGE_NUMBER"), (harvest,"CAGE_NUMBER")]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if transfers is not None:
        for col in ["ORIGIN_CAGE","DESTINATION_CAGE"]:
            if col in transfers.columns:
                transfers[col] = pd.to_numeric(transfers[col], errors="coerce").astype("Int64")
        wcol = find_col(transfers, ["TOTAL_WEIGHT_KG","WEIGHT_KG"], "WEIGHT")
        if wcol and wcol != "TOTAL_WEIGHT_KG":
            transfers.rename(columns={wcol:"TOTAL_WEIGHT_KG"}, inplace=True)

    return feeding, harvest, sampling, transfers

# -------------------------------
# 2. Preprocess Cage 2
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # Helper to find cage column dynamically
    def get_cage_col(df):
        for col in df.columns:
            if "CAGE" in col:
                return col
        raise KeyError("No cage column found in dataframe.")

    feeding_cage_col  = get_cage_col(feeding)
    harvest_cage_col  = get_cage_col(harvest)
    sampling_cage_col = get_cage_col(sampling)

    # Filter datasets safely
    def _clip(df, cage_col):
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        df = df[(df[cage_col]==cage_number) & (df["DATE"] >= start_date) & (df["DATE"] <= end_date)]
        return df

    feeding_c2  = _clip(feeding, feeding_cage_col)
    harvest_c2  = _clip(harvest, harvest_cage_col)
    sampling_c2 = _clip(sampling, sampling_cage_col)

    # Stocking row (same as before)
    stocked_fish = 7290
    initial_abw = 11.9
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        sampling_cage_col: cage_number,
        "NUMBER_OF_FISH": stocked_fish,
        "AVERAGE_BODY_WEIGHT_G": initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE")

    # Initialize cumulative columns
    for col in ["STOCKED","HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        sampling_c2[col] = 0.0
    sampling_c2["STOCKED"] = stocked_fish

    # Harvest cumulatives
    if not harvest_c2.empty:
        fish_col = find_col(harvest_c2, ["NUMBER_OF_FISH"], "FISH")
        kg_col   = find_col(harvest_c2, ["TOTAL_WEIGHT_KG","TOTAL_WEIGHT_(KG)"], "WEIGHT")
        h = harvest_c2.copy()
        h["H_FISH"] = pd.to_numeric(h[fish_col], errors="coerce").fillna(0) if fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[kg_col], errors="coerce").fillna(0) if kg_col else 0
        h["HARV_FISH_CUM"], h["HARV_KG_CUM"] = h["H_FISH"].cumsum(), h["H_KG"].cumsum()
        mh = pd.merge_asof(sampling_c2[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        sampling_c2["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        sampling_c2["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # Transfer cumulatives (optional)
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        # Outgoing
        tout = t[t.get("ORIGIN_CAGE")==cage_number].copy()
        if not tout.empty:
            fish_col = find_col(tout, ["NUMBER_OF_FISH","N_FISH"], "FISH")
            kg_col = find_col(tout, ["TOTAL_WEIGHT_KG","WEIGHT_KG"], "WEIGHT")
            tout["OUT_FISH_CUM"] = pd.to_numeric(tout[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tout["OUT_KG_CUM"]   = pd.to_numeric(tout[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mo = pd.merge_asof(sampling_c2[["DATE"]], tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            sampling_c2["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            sampling_c2["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)
        # Incoming
        tin = t[t.get("DESTINATION_CAGE")==cage_number].copy()
        if not tin.empty:
            fish_col = find_col(tin, ["NUMBER_OF_FISH","N_FISH"], "FISH")
            kg_col = find_col(tin, ["TOTAL_WEIGHT_KG","WEIGHT_KG"], "WEIGHT")
            tin["IN_FISH_CUM"] = pd.to_numeric(tin[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tin["IN_KG_CUM"]   = pd.to_numeric(tin[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mi = pd.merge_asof(sampling_c2[["DATE"]], tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
            sampling_c2["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            sampling_c2["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # Standing fish
    sampling_c2["FISH_ALIVE"] = (sampling_c2["STOCKED"] - sampling_c2["HARV_FISH_CUM"] +
                                 sampling_c2["IN_FISH_CUM"] - sampling_c2["OUT_FISH_CUM"]).clip(lower=0)
    sampling_c2["NUMBER_OF_FISH"] = sampling_c2["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, sampling_c2

# -------------------------------
# 3. Compute production summary
# -------------------------------
def compute_summary(feeding_c2, sampling_c2):
    s = sampling_c2.copy().sort_values("DATE")
    feed_col = find_col(feeding_c2, ["FEED_AMOUNT_KG","FEED_AMT_KG"], "FEED")
    abw_col = find_col(s, ["AVERAGE_BODY_WEIGHT_G","ABW_G","ABW"], "ABW")
    if not feed_col or not abw_col:
        return s

    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    summary = pd.merge_asof(s, feeding_c2[["DATE","CUM_FEED"]], on="DATE", direction="backward")
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = summary["NUMBER_OF_FISH"] * summary["ABW_G"] / 1000.0

    # Period metrics
    summary["FEED_PERIOD_KG"] = summary["CUM_FEED"].diff()
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()
    summary["GROWTH_KG"] = summary["ΔBIOMASS_STANDING"].fillna(summary["BIOMASS_KG"])
    summary["PERIOD_eFCR"] = summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"].replace(0, np.nan)
    summary["AGGREGATED_eFCR"] = summary["CUM_FEED"] / summary["BIOMASS_KG"].replace(0, np.nan)

    # First row NaN for period metrics
    summary.loc[0, ["FEED_PERIOD_KG","ΔBIOMASS_STANDING","GROWTH_KG","PERIOD_eFCR"]] = np.nan

    return summary

# -------------------------------
# 4. Create mock cages (3-7)
# -------------------------------
def create_mock_cages(summary_c2, feeding_c2, sampling_c2):
    mock_summaries = {}
    cage_ids = range(3, 8)
    dates = sampling_c2['DATE'].tolist()
    for cage in cage_ids:
        mock = summary_c2.copy()
        mock['CAGE_NUMBER'] = cage
        mock['NUMBER_OF_FISH'] = (mock['NUMBER_OF_FISH'] + np.random.randint(-50,50,len(mock))).clip(lower=0)
        mock['ABW_G'] = mock['ABW_G'] * np.random.normal(1,0.05,len(mock))
        mock['BIOMASS_KG'] = mock['NUMBER_OF_FISH'] * mock['ABW_G'] / 1000
        mock['CUM_FEED'] = mock['CUM_FEED'] * np.random.normal(1,0.05,len(mock))
        mock['GROWTH_KG'] = mock['ΔBIOMASS_STANDING'].fillna(mock['BIOMASS_KG'])
        mock['PERIOD_eFCR'] = mock['PERIOD_eFCR'] * np.random.normal(1,0.05,len(mock))
        mock['AGGREGATED_eFCR'] = mock['AGGREGATED_eFCR'] * np.random.normal(1,0.05,len(mock))
        mock_summaries[cage] = mock
    return mock_summaries

# -------------------------------
# 5. Streamlit Interface
# -------------------------------
st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")
st.title("Fish Cage Production Analysis")

st.sidebar.header("Upload Excel Files (Cage 2)")
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
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth","Biomass","ABW","eFCR"])

    df = all_cages[selected_cage].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["DATE"])

    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[["DATE","NUMBER_OF_FISH","ABW_G","BIOMASS_KG","AGGREGATED_eFCR","PERIOD_eFCR"]].style.format({
        "ABW_G":"{:.2f}",
        "BIOMASS_KG":"{:.2f}",
        "AGGREGATED_eFCR":"{:.2f}",
        "PERIOD_eFCR":"{:.2f}"
    }))

    # Plotting
    if selected_kpi=="Growth":
        fig = px.line(df, x="DATE", y="GROWTH_KG", markers=True, title=f"Cage {selected_cage} Growth Over Time")
    elif selected_kpi=="Biomass":
        fig = px.line(df, x="DATE", y="BIOMASS_KG", markers=True, title=f"Cage {selected_cage} Biomass Over Time")
    elif selected_kpi=="ABW":
        fig = px.line(df, x="DATE", y="ABW_G", markers=True, title=f"Cage {selected_cage} Average Body Weight Over Time")
    else:  # eFCR
        fig = px.line(df, x="DATE", y="AGGREGATED_eFCR", markers=True, name="Aggregated eFCR")
        fig.add_scatter(x=df["DATE"], y=df["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR")
        fig.update_layout(yaxis_title="eFCR", title=f"Cage {selected_cage} eFCR Over Time")

    st.plotly_chart(fig, use_container_width=True)
