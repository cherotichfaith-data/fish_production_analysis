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

    # Coerce cage columns
    for df in [feeding, sampling, harvest]:
        if "CAGE NUMBER" in df.columns:
            df["CAGE NUMBER"] = to_int_cage(df["CAGE NUMBER"])
    if "CAGE" in harvest.columns and "CAGE NUMBER" not in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    if transfers is not None:
        for col in ["ORIGIN CAGE", "DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        wcol = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], fuzzy_hint="WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers else []):
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

    # Filter by cage & date
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        out = df.dropna(subset=["DATE"]).sort_values("DATE")
        return out[(out["DATE"] >= start_date) & (out["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number])
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number])
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number])

    # Stocking row
    stocked_fish = 7290
    initial_abw_g = 11.9
    if transfers is not None and not transfers.empty:
        t_in = _clip(transfers[transfers["DESTINATION CAGE"] == cage_number])
        if not t_in.empty and "NUMBER" in t_in.columns and pd.notna(t_in.iloc[0]["NUMBER"]):
            stocked_fish = int(t_in.iloc[0]["NUMBER"])
        if not t_in.empty and "TOTAL WEIGHT [KG]" in t_in.columns and pd.notna(t_in.iloc[0]["TOTAL WEIGHT [KG]"]):
            initial_abw_g = t_in.iloc[0]["TOTAL WEIGHT [KG]"]*1000/stocked_fish

    stocking_row = pd.DataFrame([{
        "DATE": start_date, "CAGE NUMBER": cage_number, "AVERAGE BODY WEIGHT(G)": initial_abw_g, "STOCKED": stocked_fish
    }])
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # Initialize cumulatives
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # Harvest cumulatives
    if not harvest_c2.empty:
        h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
        h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
        if h_fish_col or h_kg_col:
            h = harvest_c2.copy()
            h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
            h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
            h["HARV_FISH_CUM"], h["HARV_KG_CUM"] = h["H_FISH"].cumsum(), h["H_KG"].cumsum()
            mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
            base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
            base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # Transfers cumulatives
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        # Outgoing
        tout = t[t["ORIGIN CAGE"] == cage_number]
        if not tout.empty:
            tout["OUT_FISH_CUM"] = tout["NUMBER"].cumsum() if "NUMBER" in tout.columns else 0
            tout["OUT_KG_CUM"]   = tout["TOTAL WEIGHT [KG]"].cumsum() if "TOTAL WEIGHT [KG]" in tout.columns else 0
            mo = pd.merge_asof(base[["DATE"]], tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)
        # Incoming
        tin = t[t["DESTINATION CAGE"] == cage_number]
        if not tin.empty:
            tin["IN_FISH_CUM"] = tin["NUMBER"].cumsum() if "NUMBER" in tin.columns else 0
            tin["IN_KG_CUM"]   = tin["TOTAL WEIGHT [KG]"].cumsum() if "TOTAL WEIGHT [KG]" in tin.columns else 0
            mi = pd.merge_asof(base[["DATE"]], tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
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
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)"], "FEED")
    abw_col  = find_col(s, ["AVERAGE BODY WEIGHT(G)","ABW(G)"], "ABW")
    if not feed_col or not abw_col: return s

    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()
    summary = pd.merge_asof(s, feeding_c2[["DATE","CUM_FEED"]], on="DATE", direction="backward")
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = summary["NUMBER OF FISH"] * summary["ABW_G"]/1000

    summary["FEED_PERIOD_KG"]    = summary["CUM_FEED"].diff()
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()
    summary["GROWTH_KG"]         = summary["ΔBIOMASS_STANDING"]  # simplified

    summary["PERIOD_eFCR"] = np.where(summary["GROWTH_KG"]>0, summary["FEED_PERIOD_KG"]/summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(summary["GROWTH_KG"].cumsum()>0, summary["CUM_FEED"]/summary["GROWTH_KG"].cumsum(), np.nan)

    # First row → NaN for period metrics
    first_idx = summary.index.min()
    for col in ["FEED_PERIOD_KG","ΔBIOMASS_STANDING","GROWTH_KG","PERIOD_eFCR"]:
        if col in summary.columns: summary.loc[first_idx,col] = np.nan

    return summary

# =====================
# UI
# =====================

st.title("Fish Cage Production Analysis (Cage 2)")
st.sidebar.header("Upload Excel Files")
feeding_file  = st.sidebar.file_uploader("Feeding Record",    type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest",      type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling",     type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 – Production Summary")
    st.dataframe(summary_c2)

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])
    if selected_kpi == "Biomass":
        fig = px.line(summary_c2, x="DATE", y="BIOMASS_KG", markers=True, title="Cage 2 Biomass Over Time")
    elif selected_kpi == "ABW":
        fig = px.line(summary_c2, x="DATE", y="ABW_G", markers=True, title="Cage 2 ABW Over Time")
    else:
        fig = px.line(summary_c2, x="DATE", y="AGGREGATED_eFCR", markers=True, title="Cage 2 eFCR Over Time")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
