# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from typing import List, Optional

# =====================
# Helpers
# =====================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    def _coerce(val) -> Optional[int]:
        if pd.isna(val):
            return np.nan
        try:
            return int(re.search(r"(\d+)", str(val)).group(1))
        except (AttributeError, ValueError):
            return np.nan
    return series.apply(_coerce)

def to_number(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

def find_col(df: pd.DataFrame, candidates: List[str], fuzzy_hint: Optional[str] = None) -> Optional[str]:
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U:
                return orig
    return None

# =====================
# Load data
# =====================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Coerce cage numbers
    for df, col_names in [(feeding, ["CAGE NUMBER"]),
                          (sampling, ["CAGE NUMBER"]),
                          (harvest, ["CAGE NUMBER","CAGE"])]:
        for col in col_names:
            if col in df.columns:
                df["CAGE NUMBER"] = to_int_cage(df[col])
                break
        else:
            df["CAGE NUMBER"] = np.nan

    if transfers is not None:
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])

    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

# =====================
# Preprocess Cage 2
# =====================
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)].copy()

    feeding_c2  = _clip(feeding[feeding.get("CAGE NUMBER", feeding.columns[0]) == cage_number])
    harvest_c2  = _clip(harvest[harvest.get("CAGE NUMBER", harvest.get("CAGE", None)) == cage_number])
    sampling_c2 = _clip(sampling[sampling.get("CAGE NUMBER", sampling.columns[0]) == cage_number])

    # --- Manual Stocking row
    stocked_fish = 7290
    initial_abw_g = 11.9
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g,
        "STOCKED": stocked_fish
    }])

    # Combine sampling with stocking
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # Ensure final harvest date exists
    final_h_date = harvest_c2["DATE"].max() if harvest_c2 is not None and not harvest_c2.empty else end_date
    if final_h_date not in base["DATE"].values:
        last_abw = base["AVERAGE BODY WEIGHT(G)"].iloc[-1] if not base.empty else initial_abw_g
        final_row = pd.DataFrame([{
            "DATE": final_h_date,
            "CAGE NUMBER": cage_number,
            "AVERAGE BODY WEIGHT(G)": last_abw,
            "STOCKED": stocked_fish
        }])
        base = pd.concat([base, final_row], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # --- Initialize placeholder columns
    for col in ["HARVEST_FISH","HARVEST_KG","TRANSFER_IN_FISH","TRANSFER_OUT_FISH","TRANSFER_IN_KG","TRANSFER_OUT_KG"]:
        base[col] = 0.0

    # --- Standing fish placeholder
    base["FISH_ALIVE"] = base["STOCKED"]

    return feeding_c2, harvest_c2, base
# =====================
# Compute summary
# =====================
def compute_summary(feeding_c2, base, harvest_c2=None, transfers=None):
    df = base.copy().sort_values("DATE").reset_index(drop=True)

    # --- Cumulative Harvest
    df["HARVEST_FISH"] = 0
    df["HARVEST_KG"] = 0
    if harvest_c2 is not None and not harvest_c2.empty:
        harvest_c2 = harvest_c2.sort_values("DATE")
        h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
        h_kg_col = find_col(harvest_c2, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)"], "WEIGHT")
        harvest_c2["HARVEST_FISH"] = pd.to_numeric(harvest_c2.get(h_fish_col,0), errors="coerce").fillna(0)
        harvest_c2["HARVEST_KG"] = pd.to_numeric(harvest_c2.get(h_kg_col,0), errors="coerce").fillna(0)
        harvest_c2["HARV_CUM_FISH"] = harvest_c2["HARVEST_FISH"].cumsum()
        harvest_c2["HARV_CUM_KG"] = harvest_c2["HARVEST_KG"].cumsum()
        df = pd.merge_asof(df, harvest_c2[["DATE","HARV_CUM_FISH","HARV_CUM_KG","HARVEST_FISH","HARVEST_KG"]],
                           on="DATE", direction="backward")
        df[["HARVEST_FISH","HARVEST_KG","HARV_CUM_FISH","HARV_CUM_KG"]] = df[["HARVEST_FISH","HARVEST_KG","HARV_CUM_FISH","HARV_CUM_KG"]].fillna(0)

    # --- Cumulative Transfers
    df["TRANSFER_IN_FISH"] = 0
    df["TRANSFER_OUT_FISH"] = 0
    df["TRANSFER_IN_KG"] = 0
    df["TRANSFER_OUT_KG"] = 0
    if transfers is not None and not transfers.empty:
        transfers = transfers.sort_values("DATE")
        # In
        tin = transfers[transfers["DESTINATION CAGE"]==2].copy()
        if not tin.empty:
            tin["TRANSFER_IN_FISH"] = pd.to_numeric(tin.get("NUMBER OF FISH",0), errors="coerce").fillna(0)
            tin["TRANSFER_IN_KG"] = pd.to_numeric(tin.get("TOTAL WEIGHT [KG]",0), errors="coerce").fillna(0)
            tin["IN_CUM_FISH"] = tin["TRANSFER_IN_FISH"].cumsum()
            tin["IN_CUM_KG"] = tin["TRANSFER_IN_KG"].cumsum()
            df = pd.merge_asof(df, tin[["DATE","TRANSFER_IN_FISH","TRANSFER_IN_KG","IN_CUM_FISH","IN_CUM_KG"]],
                               on="DATE", direction="backward")
            df[["TRANSFER_IN_FISH","TRANSFER_IN_KG","IN_CUM_FISH","IN_CUM_KG"]] = df[["TRANSFER_IN_FISH","TRANSFER_IN_KG","IN_CUM_FISH","IN_CUM_KG"]].fillna(0)

        # Out
        tout = transfers[transfers["ORIGIN CAGE"]==2].copy()
        if not tout.empty:
            tout["TRANSFER_OUT_FISH"] = pd.to_numeric(tout.get("NUMBER OF FISH",0), errors="coerce").fillna(0)
            tout["TRANSFER_OUT_KG"] = pd.to_numeric(tout.get("TOTAL WEIGHT [KG]",0), errors="coerce").fillna(0)
            tout["OUT_CUM_FISH"] = tout["TRANSFER_OUT_FISH"].cumsum()
            tout["OUT_CUM_KG"] = tout["TRANSFER_OUT_KG"].cumsum()
            df = pd.merge_asof(df, tout[["DATE","TRANSFER_OUT_FISH","TRANSFER_OUT_KG","OUT_CUM_FISH","OUT_CUM_KG"]],
                               on="DATE", direction="backward")
            df[["TRANSFER_OUT_FISH","TRANSFER_OUT_KG","OUT_CUM_FISH","OUT_CUM_KG"]] = df[["TRANSFER_OUT_FISH","TRANSFER_OUT_KG","OUT_CUM_FISH","OUT_CUM_KG"]].fillna(0)

    # --- Standing fish
    df["NUMBER_OF_FISH"] = (df["STOCKED"] 
                            + df.get("IN_CUM_FISH",0) 
                            - df.get("OUT_CUM_FISH",0) 
                            - df.get("HARV_CUM_FISH",0)).clip(lower=0).astype(int)

    # --- Biomass & ABW
    df["ABW_G"] = pd.to_numeric(df["AVERAGE BODY WEIGHT(G)"], errors="coerce")
    df["BIOMASS_KG"] = df["ABW_G"] * df["NUMBER_OF_FISH"] / 1000

    # --- Feed
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED_KG"])
    if feed_col and feed_col in feeding_c2.columns:
        feeding_c2 = feeding_c2.sort_values("DATE").copy()
        feeding_c2["FEED_AMOUNT_KG"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0)
        feeding_c2["FEED_CUM"] = feeding_c2["FEED_AMOUNT_KG"].cumsum()
        df = pd.merge_asof(df, feeding_c2[["DATE","FEED_CUM"]], on="DATE", direction="backward")
        df["FEED_AGG_KG"] = df["FEED_CUM"].fillna(0)
        df["FEED_PERIOD_KG"] = df["FEED_AGG_KG"].diff().fillna(df["FEED_AGG_KG"])

    # --- Growth
    df["GROWTH_KG"] = df["BIOMASS_KG"].diff().fillna(df["BIOMASS_KG"].iloc[0])

    # --- eFCR
    df["PERIOD_eFCR"] = (df["FEED_PERIOD_KG"] / df["GROWTH_KG"].replace(0,np.nan)).fillna(0)
    df["AGGREGATED_eFCR"] = (df["FEED_AGG_KG"] / df["BIOMASS_KG"].diff().cumsum().replace(0,np.nan)).fillna(0)

    # --- Keep only needed columns
    df = df[[
        "DATE","NUMBER_OF_FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "PERIOD_eFCR","AGGREGATED_eFCR",
        "HARVEST_FISH","HARVEST_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG"
    ]]

    return df

# =====================
# Streamlit UI
# =====================
st.title("Fish Cage Production Analysis Dashboard")

st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file = st.sidebar.file_uploader("Feeding Records", type=["xls","xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xls","xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xls","xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers", type=["xls","xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding_c2, harvest_c2, sampling_c2, transfers_c2 = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, summary_c2 = preprocess_cage2(feeding_c2, harvest_c2, sampling_c2, transfers_c2)
    summary_c2 = compute_summary(feeding_c2, summary_c2, harvest_c2, transfers_c2)

    selected_kpi = st.sidebar.selectbox(
        "Select KPI to Plot",
        ["Biomass", "ABW", "eFCR"]
    )

    if selected_kpi=="Biomass":
        y_col="BIOMASS_KG"; title="Cage 2: Biomass Over Time"
    elif selected_kpi=="ABW":
        y_col="ABW_G"; title="Cage 2: Average Body Weight Over Time"
    else:
        y_col="AGGREGATED_eFCR"; title="Cage 2: eFCR Over Time"

    if selected_kpi!="eFCR":
        fig = px.line(summary_c2.dropna(subset=[y_col]), x="DATE", y=y_col, markers=True, title=title)
    else:
        df = summary_c2.dropna(subset=["PERIOD_eFCR","AGGREGATED_eFCR"])
        fig = px.line(df, x="DATE", y="AGGREGATED_eFCR", markers=True, title=title, labels={"AGGREGATED_eFCR":"Aggregated eFCR"})
        fig.add_scatter(x=df["DATE"], y=df["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", line=dict(dash="dash"))
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cage 2 Production Summary")
    st.dataframe(
        summary_c2[[
            "DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG",
            "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
            "PERIOD_eFCR","AGGREGATED_eFCR",
            "HARVEST_FISH","HARVEST_KG",
            "TRANSFER_IN_FISH","TRANSFER_OUT_FISH",
            "TRANSFER_IN_KG","TRANSFER_OUT_KG"
        ]].round(2)
    )
else:
    st.warning("Upload the Excel files to begin.")
