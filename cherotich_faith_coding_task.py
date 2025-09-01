# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

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
# 1. Load and clean data
# =====================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Coerce cage numbers
    for df, col_names in [(feeding, ["CAGE NUMBER"]), (sampling, ["CAGE NUMBER"]), (harvest, ["CAGE NUMBER", "CAGE"])]:
        for col in col_names:
            if col in df.columns:
                df["CAGE NUMBER"] = to_int_cage(df[col])
                break

    # Parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

# =====================
# 2. Preprocess Cage 2
# =====================
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number])
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number])
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number])

    # Stocking from first inbound transfer (fallback if missing)
    stocked_fish = 7290
    initial_abw_g = 11.9
    first_inbound_idx = None
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        t_in = t[t["DESTINATION CAGE"] == cage_number].sort_values("DATE") if "DESTINATION CAGE" in t.columns else pd.DataFrame()
        if not t_in.empty:
            first = t_in.iloc[0]
            first_inbound_idx = first.name
            if "NUMBER OF FISH" in t_in.columns and pd.notna(first.get("NUMBER OF FISH")):
                stocked_fish = int(float(first["NUMBER OF FISH"]))
            if "TOTAL WEIGHT [KG]" in t_in.columns and pd.notna(first.get("TOTAL WEIGHT [KG]")) and stocked_fish:
                initial_abw_g = float(first["TOTAL WEIGHT [KG"])*1000.0/stocked_fish

    # Inject stocking row
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g
    }])
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # Init cumulative columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # Harvest cumulatives
    h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
    h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)"], "WEIGHT")
    if (h_fish_col or h_kg_col) and not harvest_c2.empty:
        h = harvest_c2.sort_values("DATE").copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"], h["HARV_KG_CUM"] = h["H_FISH"].cumsum(), h["H_KG"].cumsum()
        mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # Transfers cumulatives (exclude stocking inbound)
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if first_inbound_idx in t.index:
            t = t.drop(index=first_inbound_idx)

        origin_col = find_col(t, ["ORIGIN CAGE", "ORIGIN"], "ORIGIN")
        dest_col   = find_col(t, ["DESTINATION CAGE", "DESTINATION"], "DEST")
        fish_col   = find_col(t, ["NUMBER OF FISH", "N_FISH"], "FISH")
        kg_col     = find_col(t, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]"], "WEIGHT")

        t["T_FISH"] = pd.to_numeric(t[fish_col], errors="coerce").fillna(0) if fish_col else 0
        t["T_KG"]   = pd.to_numeric(t[kg_col], errors="coerce").fillna(0) if kg_col else 0

        def _cage_to_int(x):
            m = re.search(r"(\d+)", str(x)) if pd.notna(x) else None
            return int(m.group(1)) if m else None

        t["ORIGIN_INT"] = t[origin_col].apply(_cage_to_int) if origin_col in t.columns else np.nan
        t["DEST_INT"]   = t[dest_col].apply(_cage_to_int)   if dest_col   in t.columns else np.nan

        # Outgoing
        tout = t[t["ORIGIN_INT"] == cage_number].sort_values("DATE").copy()
        if not tout.empty:
            tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
            tout["OUT_KG_CUM"]   = tout["T_KG"].cumsum()
            mo = pd.merge_asof(base[["DATE"]], tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

        # Incoming
        tin = t[t["DEST_INT"] == cage_number].sort_values("DATE").copy()
        if not tin.empty:
            tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
            tin["IN_KG_CUM"]   = tin["T_KG"].cumsum()
            mi = pd.merge_asof(base[["DATE"]], tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
            base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # Standing fish
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# =====================
# 3. Compute summary (period metrics)
# =====================
def compute_summary(feeding_c2, sampling_c2):
    s = sampling_c2.copy().sort_values("DATE")
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED (KG)"], "FEED")
    s["BIOMASS_KG"] = (s["AVERAGE BODY WEIGHT(G)"]*s["FISH_ALIVE"]/1000).fillna(0)
    s["FEED_PERIOD_KG"] = 0.0
    if feed_col and not feeding_c2.empty:
        f = feeding_c2.sort_values("DATE")
        for i in range(len(s)):
            start, end = (s["DATE"].iloc[i-1] if i>0 else pd.Timestamp.min), s["DATE"].iloc[i]
            s.loc[i,"FEED_PERIOD_KG"] = f[(f["DATE"]>start)&(f["DATE"]<=end)][feed_col].sum()
    # Growth
    s["ΔBIOMASS_STANDING"] = s["BIOMASS_KG"].diff().fillna(s["BIOMASS_KG"])
    s["PERIOD_eFCR"] = (s["FEED_PERIOD_KG"]/s["ΔBIOMASS_STANDING"]).replace([np.inf,-np.inf],np.nan)
    s["AGGREGATED_eFCR"] = (s["FEED_PERIOD_KG"].cumsum()/s["ΔBIOMASS_STANDING"].cumsum()).replace([np.inf,-np.inf],np.nan)
    return s

# =====================
# Streamlit App
# =====================
st.title("Fish Cage 2 Production Dashboard (Updated)")

st.sidebar.header("Upload Data")
feeding_file  = st.sidebar.file_uploader("Feeding Excel", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Harvest Excel", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Sampling Excel", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Transfer Excel (Optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding_c2, harvest_c2, sampling_c2, transfers_c2 = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, base_c2 = preprocess_cage2(feeding_c2, harvest_c2, sampling_c2, transfers_c2)
    summary_c2 = compute_summary(feeding_c2, base_c2)

    st.subheader("Production Summary (Cage 2)")
    st.dataframe(summary_c2)

    kpi = st.selectbox("Select KPI to Plot", ["BIOMASS_KG","FEED_PERIOD_KG","ΔBIOMASS_STANDING","PERIOD_eFCR","AGGREGATED_eFCR"])
    fig = px.line(summary_c2, x="DATE", y=kpi, markers=True, title=f"{kpi} over Time")
    st.plotly_chart(fig)
