# ==========================
# Fish Cage Production Analysis App
# ==========================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------
# Utility Functions
# --------------------------
def to_number(x):
    try:
        return float(str(x).replace(",","").strip())
    except:
        return np.nan

def find_col(df, possible_cols, default_name=None):
    for col in possible_cols:
        if col in df.columns:
            return col
    return default_name

# --------------------------
# Load Data Function
# --------------------------
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding = pd.read_excel(feeding_file, parse_dates=True)
    harvest = pd.read_excel(harvest_file, parse_dates=True)
    sampling = pd.read_excel(sampling_file, parse_dates=True)
    if transfer_file:
        transfers = pd.read_excel(transfer_file, parse_dates=True)
    else:
        transfers = pd.DataFrame()
    return feeding, harvest, sampling, transfers

# --------------------------
# Preprocess Cage 2
# --------------------------
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")

    # Filter cage
    feeding_c2  = feeding[(feeding.get("CAGE NUMBER",0)==cage_number)]
    harvest_c2  = harvest[(harvest.get("CAGE",0)==cage_number)]
    sampling_c2 = sampling[(sampling.get("CAGE NUMBER",0)==cage_number)].sort_values("DATE").reset_index(drop=True)

    # Ensure numeric
    for col in ["NUMBER OF FISH","ABW_G"]:
        if col in sampling_c2.columns:
            sampling_c2[col] = pd.to_numeric(sampling_c2[col], errors="coerce").fillna(0)

    # Add transfer columns if missing
    if transfers is not None and not transfers.empty:
        transfers_c2 = transfers
        for col in ["IN_FISH_CUM","OUT_FISH_CUM","IN_KG_CUM","OUT_KG_CUM","HARV_FISH_CUM","HARV_KG_CUM"]:
            if col not in sampling_c2.columns:
                sampling_c2[col] = 0
    return feeding_c2, harvest_c2, sampling_c2

# --------------------------
# Compute Summary
# --------------------------
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2  = feeding_c2.copy()
    s           = sampling_c2.copy().sort_values("DATE")

    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)","FEED AMOUNT [KG]","FEED (KG)","FEED KG","FEED_AMOUNT","FEED"], "FEED")
    abw_col  = find_col(s, ["AVERAGE BODY WEIGHT(G)","AVERAGE BODY WEIGHT (G)","ABW(G)","ABW [G]","ABW"], "ABW")
    if not feed_col or not abw_col:
        return s

    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    summary = pd.merge_asof(s, feeding_c2[["DATE","CUM_FEED"]], on="DATE", direction="backward")

    # Standing biomass
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = (pd.to_numeric(summary.get("FISH_ALIVE", summary.get("NUMBER OF FISH",0)), errors="coerce").fillna(0) * summary["ABW_G"].fillna(0) / 1000.0)

    # Period deltas
    summary["FEED_PERIOD_KG"]    = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"]       = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Logistics per period (kg)
    for cum_col, per_col in [
        ("IN_KG_CUM","TRANSFER_IN_KG"),
        ("OUT_KG_CUM","TRANSFER_OUT_KG"),
        ("HARV_KG_CUM","HARVEST_KG"),
    ]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan
    
    # Logistics per period (fish)
    summary["TRANSFER_IN_FISH"]  = summary["IN_FISH_CUM"].diff()   if "IN_FISH_CUM"   in summary.columns else np.nan
    summary["TRANSFER_OUT_FISH"] = summary["OUT_FISH_CUM"].diff()  if "OUT_FISH_CUM"  in summary.columns else np.nan
    summary["HARVEST_FISH"]      = summary["HARV_FISH_CUM"].diff() if "HARV_FISH_CUM" in summary.columns else np.nan

    # Produced growth in the period
    summary["GROWTH_KG"] = (
        summary["ΔBIOMASS_STANDING"]
        + summary["HARVEST_KG"].fillna(0)
        + summary["TRANSFER_OUT_KG"].fillna(0)
        - summary["TRANSFER_IN_KG"].fillna(0)
    )

    # Fish count discrepancy
    summary["EXPECTED_FISH_ALIVE"] = (
        summary.get("STOCKED", 0)
        - summary.get("HARV_FISH_CUM", 0)
        + summary.get("IN_FISH_CUM", 0)
        - summary.get("OUT_FISH_CUM", 0)
    )
    actual_fish = pd.to_numeric(summary.get("NUMBER OF FISH", summary.get("FISH_ALIVE",0)), errors="coerce")
    summary["FISH_COUNT_DISCREPANCY"] = pd.to_numeric(summary["EXPECTED_FISH_ALIVE"], errors="coerce").fillna(0) - actual_fish.fillna(0)

    # Period & aggregated eFCR
    growth_cum = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["PERIOD_eFCR"]     = np.where(summary["GROWTH_KG"] > 0, summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum > 0, summary["FEED_AGG_KG"] / growth_cum, np.nan)

    # First row (stocking) → NA
    first_idx = summary.index.min()
    summary.loc[first_idx, [
        "FEED_PERIOD_KG","ΔBIOMASS_STANDING",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH",
        "GROWTH_KG","PERIOD_eFCR","FISH_COUNT_DISCREPANCY"
    ]] = np.nan

    cols = [
        "DATE","CAGE NUMBER","NUMBER OF FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH","HARVEST_FISH",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]

    return summary[[c for c in cols if c in summary.columns]]

# --------------------------
# Streamlit UI
# --------------------------
st.title("Fish Cage Production Analysis (Cage 2, Transfer-aware)")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file  = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 – Production Summary (Wide Table)")
    st.dataframe(summary_c2, width=1800)

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])

    if selected_kpi == "Biomass":
        fig = px.line(summary_c2.dropna(subset=["BIOMASS_KG"]), x="DATE", y="BIOMASS_KG", markers=True,
                      title="Cage 2: Biomass Over Time", labels={"BIOMASS_KG":"Total Biomass (kg)"})
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi == "ABW":
        fig = px.line(summary_c2.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                      title="Cage 2: Average Body Weight Over Time", labels={"ABW_G":"ABW (g)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
                      title="Cage 2: eFCR Over Time", labels={"AGGREGATED_eFCR":"Aggregated eFCR"})
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", line=dict(dash="dash"))
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
