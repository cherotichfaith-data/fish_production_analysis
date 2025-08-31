# app.py — Fish Cage Production Analysis
# Robust version: Biomass, ABW, eFCR, transfers included, dynamic column handling

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
    for df in [feeding, harvest, sampling]:
        if "CAGE NUMBER" in df.columns:
            df["CAGE NUMBER"] = to_int_cage(df["CAGE NUMBER"])
    if transfers is not None:
        for col in ["ORIGIN CAGE", "DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # Standardize transfer weight
        wcol = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], "WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Parse dates
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
        out = df.dropna(subset=["DATE"]).sort_values("DATE")
        return out[(out["DATE"] >= start_date) & (out["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number]) if "CAGE NUMBER" in feeding.columns else _clip(feeding)
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number]) if "CAGE NUMBER" in harvest.columns else _clip(harvest)
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number]) if "CAGE NUMBER" in sampling.columns else _clip(sampling)

    # Initial stocking
    stocked_fish = 7290
    initial_abw_g = 11.9
    first_inbound_idx = None
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if "DESTINATION CAGE" in t.columns:
            t_in = t[t["DESTINATION CAGE"] == cage_number].sort_values("DATE")
            if not t_in.empty:
                first = t_in.iloc[0]
                first_inbound_idx = first.name
                if "NUMBER OF FISH" in t_in.columns and pd.notna(first.get("NUMBER OF FISH")):
                    stocked_fish = int(float(first["NUMBER OF FISH"]))
                if "TOTAL WEIGHT [KG]" in t_in.columns and pd.notna(first.get("TOTAL WEIGHT [KG]")):
                    initial_abw_g = float(first["TOTAL WEIGHT [KG]"]) * 1000.0 / stocked_fish

    # Inject stocking row
    stocking_row = pd.DataFrame([{
        "DATE": start_date, "CAGE NUMBER": cage_number, "ABW_G": initial_abw_g, "STOCKED": stocked_fish
    }])
    sampling_c2 = sampling_c2.rename(columns={"AVERAGE BODY WEIGHT(G)":"ABW_G"}) if "AVERAGE BODY WEIGHT(G)" in sampling_c2.columns else sampling_c2
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    base = base.sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = base.get("STOCKED", stocked_fish)

    # Add cumulative columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # Compute harvest cumulatives
    h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
    h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
    if not harvest_c2.empty and (h_fish_col or h_kg_col):
        h = harvest_c2.sort_values("DATE").copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"], h["HARV_KG_CUM"] = h["H_FISH"].cumsum(), h["H_KG"].cumsum()
        mh = pd.merge_asof(base[["DATE"]].sort_values("DATE"), h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]].sort_values("DATE"), on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # Transfers
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if first_inbound_idx is not None and first_inbound_idx in t.index:
            t = t.drop(index=first_inbound_idx)  # remove stocking

        origin_col = find_col(t, ["ORIGIN CAGE"], "ORIGIN")
        dest_col   = find_col(t, ["DESTINATION CAGE"], "DEST")
        fish_col   = find_col(t, ["NUMBER OF FISH","N_FISH"], "FISH")
        kg_col     = find_col(t, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)"], "WEIGHT")

        t["T_FISH"] = pd.to_numeric(t[fish_col], errors="coerce").fillna(0) if fish_col else 0
        t["T_KG"]   = pd.to_numeric(t[kg_col], errors="coerce").fillna(0) if kg_col else 0
        t["ORIGIN_INT"] = t[origin_col].apply(to_int_cage) if origin_col else np.nan
        t["DEST_INT"]   = t[dest_col].apply(to_int_cage)   if dest_col else np.nan

        # Outgoing
        tout = t[t["ORIGIN_INT"] == cage_number].sort_values("DATE")
        if not tout.empty:
            tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
            tout["OUT_KG_CUM"]   = tout["T_KG"].cumsum()
            mo = pd.merge_asof(base[["DATE"]].sort_values("DATE"), tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

        # Incoming
        tin = t[t["DEST_INT"] == cage_number].sort_values("DATE")
        if not tin.empty:
            tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
            tin["IN_KG_CUM"]   = tin["T_KG"].cumsum()
            mi = pd.merge_asof(base[["DATE"]].sort_values("DATE"), tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
            base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # Standing fish
    base["FISH_ALIVE"]     = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# =====================
# Compute summary
# =====================

def compute_summary(feeding_c2, sampling_c2):
    s = sampling_c2.copy().sort_values("DATE")
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)","FEED (KG)","FEED KG","FEED_AMOUNT","FEED"], "FEED")
    abw_col  = find_col(s, ["ABW_G","AVERAGE BODY WEIGHT(G)","ABW(G)"], "ABW")

    if feed_col is None:
        feeding_c2["FEED_AMOUNT_KG"] = 0
        feed_col = "FEED_AMOUNT_KG"
    feeding_c2[feed_col] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0)
    feeding_c2["CUM_FEED"] = feeding_c2[feed_col].cumsum()

    # merge feeding cumulative
    summary = pd.merge_asof(
        s.sort_values("DATE"),
        feeding_c2[["DATE","CUM_FEED"]].sort_values("DATE"),
        on="DATE", direction="backward"
    )

    # ABW & biomass
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = summary["FISH_ALIVE"] * summary["ABW_G"] / 1000.0

    # Period deltas
    summary["FEED_PERIOD_KG"] = summary["CUM_FEED"].diff()
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()
    summary["FEED_AGG_KG"] = summary["CUM_FEED"]

    # Logistics per period
    for cum_col, per_col in [("IN_KG_CUM","TRANSFER_IN_KG"), ("OUT_KG_CUM","TRANSFER_OUT_KG"), ("HARV_KG_CUM","HARVEST_KG")]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan
    summary["TRANSFER_IN_FISH"]  = summary["IN_FISH_CUM"].diff()   if "IN_FISH_CUM"   in summary.columns else np.nan
    summary["TRANSFER_OUT_FISH"] = summary["OUT_FISH_CUM"].diff()  if "OUT_FISH_CUM"  in summary.columns else np.nan
    summary["HARVEST_FISH"]      = summary["HARV_FISH_CUM"].diff() if "HARV_FISH_CUM" in summary.columns else np.nan

    # Growth & eFCR
    summary["GROWTH_KG"] = summary["ΔBIOMASS_STANDING"] + summary["HARVEST_KG"].fillna(0) + summary["TRANSFER_OUT_KG"].fillna(0) - summary["TRANSFER_IN_KG"].fillna(0)
    growth_cum = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["PERIOD_eFCR"] = np.where(summary["GROWTH_KG"]>0, summary["FEED_PERIOD_KG"]/summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum>0, summary["FEED_AGG_KG"]/growth_cum, np.nan)

    # First row → NA
    first_idx = summary.index.min()
    summary.loc[first_idx, ["FEED_PERIOD_KG","ΔBIOMASS_STANDING","TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG","TRANSFER_IN_FISH","TRANSFER_OUT_FISH","GROWTH_KG","PERIOD_eFCR"]] = np.nan

    return summary

# =====================
# UI
# =====================

st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file  = st.sidebar.file_uploader("Feeding Record",    type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest",      type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling",     type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 – Production Summary")
    show_cols = ["DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG","FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG","TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG","TRANSFER_IN_FISH","TRANSFER_OUT_FISH","HARVEST_FISH","PERIOD_eFCR","AGGREGATED_eFCR"]
    st.dataframe(summary_c2[[c for c in show_cols if c in summary_c2.columns]])

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])
    if selected_kpi=="Biomass":
        fig = px.line(summary_c2, x="DATE", y="BIOMASS_KG", markers=True, title="Cage 2: Biomass Over Time")
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi=="ABW":
        fig = px.line(summary_c2, x="DATE", y="ABW_G", markers=True, title="Cage 2: Average Body Weight Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True, title="Cage 2: eFCR Over Time")
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload the Excel files to begin.")
