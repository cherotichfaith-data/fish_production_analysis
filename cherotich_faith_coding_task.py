# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# =====================
# Helper Functions (From Enhanced Version)
# =====================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistent processing"""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    """Extract cage numbers from mixed data types"""
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def find_col(df: pd.DataFrame, candidates, fuzzy_hint: str | None = None) -> str | None:
    """Find column by exact match or fuzzy matching"""
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
    """Convert string to number, handling commas and extracting numeric values"""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# =====================
# Data Loading (Enhanced)
# =====================

def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    """Load and normalize all data files with robust column handling"""
    feeding = normalize_columns(pd.read_excel(feeding_file))
    harvest = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file is not None else None

    # Coerce cage columns to integers
    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    
    # Handle harvest cage column variations
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    # Handle transfer data
    if transfers is not None:
        for col in ["ORIGIN CAGE", "DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        
        # Standardize weight column
        wcol = find_col(
            transfers,
            ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"],
            fuzzy_hint="WEIGHT",
        )
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

# =====================
# Enhanced Cage 2 Preprocessing
# =====================

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """Enhanced preprocessing with robust transfer handling and stocking detection"""
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")

    # Filter and clip data to timeframe
    def _clip(df, cage_col=None):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        if cage_col and cage_col in df.columns:
            df = df[df[cage_col] == cage_number]
        out = df.dropna(subset=["DATE"]).sort_values("DATE")
        return out[(out["DATE"] >= start_date) & (out["DATE"] <= end_date)]

    feeding_c2 = _clip(feeding, "CAGE NUMBER")
    harvest_c2 = _clip(harvest, "CAGE NUMBER") 
    sampling_c2 = _clip(sampling, "CAGE NUMBER")

    # Handle stocking from transfers or use defaults
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
                if "TOTAL WEIGHT [KG]" in t_in.columns and pd.notna(first.get("TOTAL WEIGHT [KG]")) and stocked_fish:
                    initial_abw_g = float(first["TOTAL WEIGHT [KG]"]) * 1000.0 / stocked_fish

    # Create base sampling timeline
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g
    }])
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    base = base[(base["DATE"] >= start_date) & (base["DATE"] <= end_date)].sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # Ensure final harvest date is included with proper ABW
    final_h_date = harvest_c2["DATE"].max() if not harvest_c2.empty else pd.NaT
    if pd.notna(final_h_date) and not (base["DATE"] == final_h_date).any():
        hh = harvest_c2[harvest_c2["DATE"] == final_h_date].copy()
        fish_col = find_col(hh, ["NUMBER OF FISH"], "FISH")
        kg_col = find_col(hh, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
        abw_colh = find_col(hh, ["ABW(G)", "ABW [G]", "ABW"], "ABW")
        
        abw_final = np.nan
        if fish_col and kg_col and hh[fish_col].notna().any() and hh[kg_col].notna().any():
            tot_fish = pd.to_numeric(hh[fish_col], errors="coerce").fillna(0).sum()
            tot_kg = pd.to_numeric(hh[kg_col], errors="coerce").fillna(0).sum()
            if tot_fish > 0 and tot_kg > 0:
                abw_final = (tot_kg * 1000.0) / tot_fish
        
        if np.isnan(abw_final) and abw_colh and hh[abw_colh].notna().any():
            abw_final = pd.to_numeric(hh[abw_colh].map(to_number), errors="coerce").mean()
        
        if pd.notna(abw_final):
            base = pd.concat([
                base,
                pd.DataFrame([{
                    "DATE": final_h_date,
                    "CAGE NUMBER": cage_number,
                    "AVERAGE BODY WEIGHT(G)": abw_final,
                    "STOCKED": stocked_fish
                }])
            ], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # Initialize cumulative tracking columns
    for col in ["HARV_FISH_CUM", "HARV_KG_CUM", "IN_FISH_CUM", "IN_KG_CUM", "OUT_FISH_CUM", "OUT_KG_CUM"]:
        base[col] = 0.0

    # Calculate harvest cumulatives
    h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
    h_kg_col = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
    
    if (h_fish_col or h_kg_col) and not harvest_c2.empty:
        h = harvest_c2.sort_values("DATE").copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"] = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"], h["HARV_KG_CUM"] = h["H_FISH"].cumsum(), h["H_KG"].cumsum()
        
        mh = pd.merge_asof(base[["DATE"]], h[["DATE", "HARV_FISH_CUM", "HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"] = mh["HARV_KG_CUM"].fillna(0)

    # Calculate transfer cumulatives (excluding stocking event)
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if not t.empty:
            if first_inbound_idx is not None and first_inbound_idx in t.index:
                t = t.drop(index=first_inbound_idx)  # Remove stocking event

            origin_col = find_col(t, ["ORIGIN CAGE", "ORIGIN"], "ORIGIN")
            dest_col = find_col(t, ["DESTINATION CAGE", "DESTINATION"], "DEST")
            fish_col = find_col(t, ["NUMBER OF FISH", "N_FISH"], "FISH")
            kg_col = find_col(t, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")

            t["T_FISH"] = pd.to_numeric(t[fish_col], errors="coerce").fillna(0) if fish_col else 0
            t["T_KG"] = pd.to_numeric(t[kg_col], errors="coerce").fillna(0) if kg_col else 0

            def _cage_to_int(x):
                m = re.search(r"(\d+)", str(x)) if pd.notna(x) else None
                return int(m.group(1)) if m else None

            t["ORIGIN_INT"] = t[origin_col].apply(_cage_to_int) if origin_col else np.nan
            t["DEST_INT"] = t[dest_col].apply(_cage_to_int) if dest_col else np.nan

            # Outgoing transfers
            tout = t[t["ORIGIN_INT"] == cage_number].sort_values("DATE").copy()
            if not tout.empty:
                tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
                tout["OUT_KG_CUM"] = tout["T_KG"].cumsum()
                mo = pd.merge_asof(
                    base[["DATE"]].sort_values("DATE"),
                    tout[["DATE", "OUT_FISH_CUM", "OUT_KG_CUM"]].sort_values("DATE"),
                    on="DATE", direction="backward"
                )
                base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
                base["OUT_KG_CUM"] = mo["OUT_KG_CUM"].fillna(0)

            # Incoming transfers
            tin = t[t["DEST_INT"] == cage_number].sort_values("DATE").copy()
            if not tin.empty:
                tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
                tin["IN_KG_CUM"] = tin["T_KG"].cumsum()
                mi = pd.merge_asof(
                    base[["DATE"]].sort_values("DATE"),
                    tin[["DATE", "IN_FISH_CUM", "IN_KG_CUM"]].sort_values("DATE"),
                    on="DATE", direction="backward"
                )
                base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
                base["IN_KG_CUM"] = mi["IN_KG_CUM"].fillna(0)

    # Calculate standing fish count
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

# =====================
# Enhanced Summary Computation
# =====================

def compute_summary(feeding_c2, sampling_c2):
    """Compute comprehensive production metrics with period-based calculations"""
    feeding_c2 = feeding_c2.copy()
    s = sampling_c2.copy().sort_values("DATE")

    # Find relevant columns flexibly
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)", "FEED AMOUNT (Kg)", "FEED AMOUNT [KG]", "FEED (KG)", "FEED KG"], "FEED")
    abw_col = find_col(s, ["AVERAGE BODY WEIGHT(G)", "AVERAGE BODY WEIGHT (G)", "ABW(G)", "ABW [G]", "ABW"], "ABW")
    
    if not feed_col or not abw_col:
        return s

    # Calculate cumulative feed
    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    # Merge feed data to sampling timeline
    summary = pd.merge_asof(s, feeding_c2[["DATE", "CUM_FEED"]], on="DATE", direction="backward")

    # Calculate biomass metrics
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = (pd.to_numeric(summary["FISH_ALIVE"], errors="coerce").fillna(0) * summary["ABW_G"].fillna(0) / 1000.0)

    # Period-based calculations
    summary["FEED_PERIOD_KG"] = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"] = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Logistics per period (kg and fish)
    for cum_col, per_col in [
        ("IN_KG_CUM", "TRANSFER_IN_KG"),
        ("OUT_KG_CUM", "TRANSFER_OUT_KG"),
        ("HARV_KG_CUM", "HARVEST_KG"),
    ]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan

    summary["TRANSFER_IN_FISH"] = summary["IN_FISH_CUM"].diff() if "IN_FISH_CUM" in summary.columns else np.nan
    summary["TRANSFER_OUT_FISH"] = summary["OUT_FISH_CUM"].diff() if "OUT_FISH_CUM" in summary.columns else np.nan
    summary["HARVEST_FISH"] = summary["HARV_FISH_CUM"].diff() if "HARV_FISH_CUM" in summary.columns else np.nan

    # Growth calculations (including transfers and harvest)
    summary["GROWTH_KG"] = (
        summary["ΔBIOMASS_STANDING"]
        + summary["HARVEST_KG"].fillna(0)
        + summary["TRANSFER_OUT_KG"].fillna(0)
        - summary["TRANSFER_IN_KG"].fillna(0)
    )

    # Fish count discrepancy tracking for data quality
    summary["EXPECTED_FISH_ALIVE"] = (
        summary.get("STOCKED", 0)
        - summary.get("HARV_FISH_CUM", 0)
        + summary.get("IN_FISH_CUM", 0)
        - summary.get("OUT_FISH_CUM", 0)
    )
    actual_fish = pd.to_numeric(summary.get("NUMBER OF FISH"), errors="coerce")
    summary["FISH_COUNT_DISCREPANCY"] = pd.to_numeric(summary["EXPECTED_FISH_ALIVE"], errors="coerce").fillna(0) - actual_fish.fillna(0)

    # eFCR calculations
    growth_cum = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["PERIOD_eFCR"] = np.where(summary["GROWTH_KG"] > 0, summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum > 0, summary["FEED_AGG_KG"] / growth_cum, np.nan)

    # Set first row (stocking) metrics to NaN for period calculations
    first_idx = summary.index.min()
    summary.loc[first_idx, [
        "FEED_PERIOD_KG", "ΔBIOMASS_STANDING",
        "TRANSFER_IN_KG", "TRANSFER_OUT_KG", "HARVEST_KG",
        "TRANSFER_IN_FISH", "TRANSFER_OUT_FISH", "HARVEST_FISH",
        "GROWTH_KG", "PERIOD_eFCR", "FISH_COUNT_DISCREPANCY"
    ]] = np.nan

    return summary

# 4. Create mock cages (3-7)
def create_mock_cage_data(summary_c2):
    mock_summaries = {}
    for cage_id in range(3, 8):
        mock = summary_c2.copy()
        mock['CAGE NUMBER'] = cage_id

        # Randomize weights ±5%, number of fish ±50, feed ±10%
        mock['TOTAL_WEIGHT_KG'] *= np.random.normal(1, 0.05, size=len(mock))
        mock['NUMBER OF FISH'] = mock['NUMBER OF FISH'] + np.random.randint(-50, 50, size=len(mock))
        mock['CUM_FEED'] *= np.random.normal(1, 0.1, size=len(mock))

        # recompute eFCR
        mock['AGGREGATED_eFCR'] = mock['CUM_FEED'] / mock['TOTAL_WEIGHT_KG']
        mock['PERIOD_WEIGHT_GAIN'] = mock['TOTAL_WEIGHT_KG'].diff().fillna(mock['TOTAL_WEIGHT_KG'])
        mock['PERIOD_FEED'] = mock['CUM_FEED'].diff().fillna(mock['CUM_FEED'])
        mock['PERIOD_eFCR'] = mock['PERIOD_FEED'] / mock['PERIOD_WEIGHT_GAIN']

        mock_summaries[cage_id] = mock
    return mock_summaries

# 5. Streamlit Interface
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling = load_data(feeding_file, harvest_file, sampling_file)

    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Generate mock cages
    mock_cages = create_mock_cage_data(summary_c2)
    all_cages = {2: summary_c2, **mock_cages}

    # Sidebar selectors
    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage]

    # Display production summary table
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']])

    # Plot graphs
    if selected_kpi == "Growth":
        df['TOTAL_WEIGHT_KG'] = pd.to_numeric(df['TOTAL_WEIGHT_KG'], errors='coerce')
        df = df.dropna(subset=['TOTAL_WEIGHT_KG'])
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    else:
        df['AGGREGATED_eFCR'] = pd.to_numeric(df['AGGREGATED_eFCR'], errors='coerce')
        df['PERIOD_eFCR'] = pd.to_numeric(df['PERIOD_eFCR'], errors='coerce')
        df = df.dropna(subset=['AGGREGATED_eFCR','PERIOD_eFCR'])
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True)
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)
