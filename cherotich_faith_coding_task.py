# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# =====================
# Helpers
# =====================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def find_col(df: pd.DataFrame, candidates, fuzzy_hint: str | None = None) -> str | None:
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

# =====================
# Load data
# =====================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Coerce cage numbers
    for df, col_names in [(feeding, ["CAGE NUMBER"]), (sampling, ["CAGE NUMBER"]), (harvest, ["CAGE NUMBER","CAGE"])]:
        for col in col_names:
            if col in df.columns:
                df["CAGE NUMBER"] = to_int_cage(df[col])
                break

    # Parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers


def to_int_cage(val):
    """Safely convert a value to integer cage number."""
    if pd.isna(val):
        return np.nan
    try:
        return int(re.search(r"(\d+)", str(val)).group(1))
    except (AttributeError, ValueError):
        return np.nan

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """
    Preprocess Cage 2 timeline with actual stocking, harvests, and optional transfers.
    Returns feeding_c2, harvest_c2, and base sampling DataFrame.
    """
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    # Filter by cage
    feeding_c2  = _clip(feeding[feeding.get("CAGE NUMBER", feeding.columns[0]) == cage_number])
    harvest_c2  = _clip(harvest[harvest.get("CAGE NUMBER", harvest.get("CAGE", None)) == cage_number])
    sampling_c2 = _clip(sampling[sampling.get("CAGE NUMBER", sampling.columns[0]) == cage_number])

    # Stocking row
    stocked_fish = 7290
    initial_abw_g = 11.9
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        t["DEST_INT"] = t["DESTINATION CAGE"].apply(to_int_cage) if "DESTINATION CAGE" in t.columns else np.nan
        inbound = t[t["DEST_INT"] == cage_number].sort_values("DATE")
        if not inbound.empty:
            first = inbound.iloc[0]
            if "NUMBER OF FISH" in inbound.columns and pd.notna(first.get("NUMBER OF FISH")):
                stocked_fish = int(first["NUMBER OF FISH"])
            if "TOTAL WEIGHT [KG]" in inbound.columns and pd.notna(first.get("TOTAL WEIGHT [KG]")):
                initial_abw_g = float(first["TOTAL WEIGHT [KG]"]) * 1000.0 / stocked_fish

    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g,
        "STOCKED": stocked_fish
    }])

    base = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    base = base.sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # Initialize cumulative columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # Compute harvest cumulatives
    if not harvest_c2.empty:
        h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
        h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
        h = harvest_c2.sort_values("DATE").copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"], h["HARV_KG_CUM"] = h["H_FISH"].cumsum(), h["H_KG"].cumsum()
        mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # Transfers cumulatives
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        # Convert origin/dest to int
        t["ORIGIN_INT"] = t["ORIGIN CAGE"].apply(to_int_cage) if "ORIGIN CAGE" in t.columns else np.nan
        t["DEST_INT"]   = t["DESTINATION CAGE"].apply(to_int_cage) if "DESTINATION CAGE" in t.columns else np.nan

        # Outgoing
        tout = t[t["ORIGIN_INT"] == cage_number].sort_values("DATE")
        if not tout.empty:
            tout["OUT_FISH_CUM"] = tout["NUMBER OF FISH"].cumsum() if "NUMBER OF FISH" in tout.columns else 0
            tout["OUT_KG_CUM"]   = tout["TOTAL WEIGHT [KG]"].cumsum() if "TOTAL WEIGHT [KG]" in tout.columns else 0
            mo = pd.merge_asof(base[["DATE"]].sort_values("DATE"), tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

        # Incoming
        tin = t[t["DEST_INT"] == cage_number].sort_values("DATE")
        if not tin.empty:
            tin["IN_FISH_CUM"] = tin["NUMBER OF FISH"].cumsum() if "NUMBER OF FISH" in tin.columns else 0
            tin["IN_KG_CUM"]   = tin["TOTAL WEIGHT [KG]"].cumsum() if "TOTAL WEIGHT [KG]" in tin.columns else 0
            mi = pd.merge_asof(base[["DATE"]].sort_values("DATE"), tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
            base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # Standing fish
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# =====================
# Compute summary
# =====================
def compute_summary(feeding_c2, sampling_c2):
    s = sampling_c2.copy().sort_values("DATE")
    s["ABW_G"] = pd.to_numeric(s["AVERAGE BODY WEIGHT(G)"], errors="coerce")
    s["BIOMASS_KG"] = s["ABW_G"] * s["NUMBER OF FISH"] / 1000.0

    # --------------------------
    # Handle feeding column safely
    # --------------------------
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED_KG"])
    if feed_col and feed_col in feeding_c2.columns:
        feeding_c2["FEED_AMOUNT_KG"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0)
    else:
        feeding_c2["FEED_AMOUNT_KG"] = 0

    # eFCR calculations
    s["FEED_CUM_KG"] = feeding_c2["FEED_AMOUNT_KG"].cumsum()
    s["PERIOD_FEED"] = s["FEED_CUM_KG"].diff().fillna(s["FEED_CUM_KG"])
    s["PERIOD_WEIGHT_GAIN"] = s["BIOMASS_KG"].diff().fillna(s["BIOMASS_KG"])
    s["PERIOD_eFCR"] = (s["PERIOD_FEED"] / s["PERIOD_WEIGHT_GAIN"]).replace([np.inf, -np.inf], np.nan)
    s["AGGREGATED_eFCR"] = (s["FEED_CUM_KG"] / s["BIOMASS_KG"]).replace([np.inf, -np.inf], np.nan)

    return s

# =====================
# App UI
# =====================
st.title("Fish Cage Production Analysis Dashboard")

# Upload files
st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file = st.sidebar.file_uploader("Feeding Records", type=["xls","xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xls","xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xls","xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers", type=["xls","xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding_c2, harvest_c2, sampling_c2, transfers_c2 = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, summary_c2 = preprocess_cage2(feeding_c2, harvest_c2, sampling_c2, transfers_c2)
    summary_c2 = compute_summary(feeding_c2, summary_c2)

    # -------------------------------
    # KPI Selection
    # -------------------------------
    selected_kpi = st.sidebar.selectbox(
        "Select KPI to Plot",
        ["Biomass", "ABW", "eFCR"]
    )

    if selected_kpi == "Biomass":
        y_col = "BIOMASS_KG"
        title = "Cage 2: Biomass Over Time"
    elif selected_kpi == "ABW":
        y_col = "ABW_G"
        title = "Cage 2: Average Body Weight Over Time"
    else:  # eFCR
        y_col = "AGGREGATED_eFCR"
        title = "Cage 2: eFCR Over Time"

    # Plot
    if selected_kpi != "eFCR":
        fig = px.line(summary_c2.dropna(subset=[y_col]), x="DATE", y=y_col, markers=True, title=title)
    else:
        df = summary_c2.dropna(subset=["PERIOD_eFCR", "AGGREGATED_eFCR"])
        fig = px.line(df, x="DATE", y="AGGREGATED_eFCR", markers=True, title=title, labels={"AGGREGATED_eFCR":"Aggregated eFCR"})
        fig.add_scatter(x=df["DATE"], y=df["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", line=dict(dash="dash"))
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")

    st.plotly_chart(fig, use_container_width=True)

    # Production Summary Table
    st.subheader("Cage 2 Production Summary")
    st.dataframe(summary_c2[["DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG","PERIOD_FEED","PERIOD_WEIGHT_GAIN","PERIOD_eFCR","AGGREGATED_eFCR"]].round(2))

else:
    st.warning("Upload the Excel files to begin.")
