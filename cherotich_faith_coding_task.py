# app.py — Corrected Fish Cage Production Analysis (with Transfers & Correct Stocking)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# ===============================
# 1. Helper Functions
# ===============================
def normalize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series):
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def to_number(x):
    if pd.isna(x): return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

def find_col(df, candidates, fuzzy_hint=None):
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut: return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U: return orig
    return None

# ===============================
# 2. Load Data
# ===============================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding = normalize_columns(pd.read_excel(feeding_file))
    harvest = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Coerce cage numbers
    for df in [feeding, harvest, sampling]:
        if "CAGE NUMBER" in df.columns:
            df["CAGE NUMBER"] = to_int_cage(df["CAGE NUMBER"])
    if transfers is not None:
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # Normalize weight to kg if in grams
        wcol = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)","WEIGHT [KG]"], "WEIGHT")
        if wcol:
            transfers[wcol] = pd.to_numeric(transfers[wcol], errors="coerce")
            transfers.loc[transfers[wcol] > 1000, wcol] /= 1000  # assume >1000 means grams → kg
    # Parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return feeding, harvest, sampling, transfers

# ===============================
# 3. Preprocess Cage 2 (Stocking + Transfers + Final Harvest)
# ===============================
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")
    stocked_fish = 7290
    initial_abw = 11.9

    # Clip data
    def _clip(df):
        if df is None or df.empty: return pd.DataFrame()
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)].sort_values("DATE")

    feeding_c2 = _clip(feeding[feeding["CAGE NUMBER"]==cage_number])
    harvest_c2 = _clip(harvest[harvest["CAGE NUMBER"]==cage_number])
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"]==cage_number])

    # Inject stocking row
    stocking_row = pd.DataFrame([{
        "DATE": start_date, "CAGE NUMBER": cage_number, "STOCKED": stocked_fish,
        "AVERAGE BODY WEIGHT(G)": initial_abw
    }])
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)
    if "STOCKED" not in base.columns: base["STOCKED"] = stocked_fish

    # Ensure final harvest date row exists
    final_h_date = harvest_c2["DATE"].max() if not harvest_c2.empty else end_date
    if not (base["DATE"] == final_h_date).any():
        base = pd.concat([base, pd.DataFrame([{
            "DATE": final_h_date, "CAGE NUMBER": cage_number, "STOCKED": stocked_fish
        }])], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # Initialize cumulative columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # Harvest cumulatives
    h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
    h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)"], "WEIGHT")
    if h_fish_col or h_kg_col:
        h = harvest_c2.copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"]   = h["H_KG"].cumsum()
        mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # Transfers cumulatives
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        # Outgoing
        tout = t[t["ORIGIN CAGE"]==cage_number].copy()
        if not tout.empty:
            fish_col = find_col(tout, ["NUMBER OF FISH","N_FISH"], "FISH")
            kg_col   = find_col(tout, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)"], "WEIGHT")
            tout["OUT_FISH_CUM"] = pd.to_numeric(tout[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tout["OUT_KG_CUM"]   = pd.to_numeric(tout[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mo = pd.merge_asof(base[["DATE"]], tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)
        # Incoming
        tin = t[t["DESTINATION CAGE"]==cage_number].copy()
        if not tin.empty:
            fish_col = find_col(tin, ["NUMBER OF FISH","N_FISH"], "FISH")
            kg_col   = find_col(tin, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)"], "WEIGHT")
            tin["IN_FISH_CUM"] = pd.to_numeric(tin[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tin["IN_KG_CUM"]   = pd.to_numeric(tin[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mi = pd.merge_asof(base[["DATE"]], tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
            base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # Standing fish
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# ===============================
# 4. Compute Summary (eFCR + Growth)
# ===============================
def compute_summary(feeding_c2, sampling_c2):
    s = sampling_c2.copy()
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)"], "FEED")
    abw_col  = find_col(s, ["AVERAGE BODY WEIGHT(G)","AVERAGE BODY WEIGHT (G)","ABW(G)","ABW [G]"], "ABW")
    if not feed_col or not abw_col: return s

    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()
    summary = pd.merge_asof(s.sort_values("DATE"), feeding_c2[["DATE","CUM_FEED"]].sort_values("DATE"), on="DATE", direction="backward")

    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = summary["FISH_ALIVE"] * summary["ABW_G"] / 1000

    summary["FEED_PERIOD_KG"] = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"]    = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Growth period (kg) including harvest & transfers
    summary["GROWTH_KG"] = summary["ΔBIOMASS_STANDING"] + summary.get("HARV_KG",0).fillna(0) + summary.get("OUT_KG_CUM",0).diff().fillna(0) - summary.get("IN_KG_CUM",0).diff().fillna(0)

    # eFCR
    growth_cum = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["PERIOD_eFCR"]     = np.where(summary["GROWTH_KG"]>0, summary["FEED_PERIOD_KG"]/summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum>0, summary["FEED_AGG_KG"]/growth_cum, np.nan)

    first_idx = summary.index.min()
    summary.loc[first_idx, ["FEED_PERIOD_KG","ΔBIOMASS_STANDING","GROWTH_KG","PERIOD_eFCR"]] = np.nan

    return summary

# ===============================
# 5. Streamlit Interface
# ===============================
st.title("Fish Cage Production Analysis (Corrected)")

st.sidebar.header("Upload Excel Files (Cage 2)")
feeding_file  = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 – Production Summary")
    show_cols = ["DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG","FEED_PERIOD_KG","FEED_AGG_KG",
                 "GROWTH_KG","PERIOD_eFCR","AGGREGATED_eFCR"]
    st.dataframe(summary_c2[[c for c in show_cols if c in summary_c2.columns]])

    # Biomass plot
    fig = px.line(summary_c2, x="DATE", y="BIOMASS_KG", title="Cage 2 Biomass Over Time")
    st.plotly_chart(fig, use_container_width=True)
