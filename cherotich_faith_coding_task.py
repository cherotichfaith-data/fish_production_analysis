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
# Utitlity/Helper Functions 
# =====================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """standardizes column names for consistent processing"""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    """Extract integer cage numbers from mixed data types(strings)"""
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

from typing import Optional
def find_col(df: pd.DataFrame, candidates, fuzzy_hint: Optional[str] = None) -> Optional[str]:
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
    """Convert string to number, handling commas and extracting numeric values from strings"""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# =====================
# =====================
# Data Loading (Corrected, suppress warnings)
# =====================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    """Load and normalize all Excel data files without showing warnings"""
    
    # Load and normalize column names
    feeding = normalize_columns(pd.read_excel(feeding_file))
    harvest = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    
    # Coerce cage columns to integers   
    cage_col = find_col(feeding, ["CAGE NUMBER", "CAGE"])
    if cage_col:
        feeding["CAGE NUMBER"] = to_int_cage(feeding[cage_col])
    cage_col = find_col(sampling, ["CAGE NUMBER", "CAGE"])
    if cage_col:
        sampling["CAGE NUMBER"] = to_int_cage(sampling[cage_col])
    
    cage_col = find_col(harvest, ["CAGE NUMBER", "CAGE"])
    if cage_col:
        harvest["CAGE NUMBER"] = to_int_cage(harvest[cage_col])

    # Handle transfer data      
    transfers = None
    if transfer_file:
        try:
            transfers = normalize_columns(pd.read_excel(transfer_file))
        except Exception:
            transfers = None

        if transfers is not None:
            wcol = find_col(
                transfers,
                ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"],
                fuzzy_hint="WEIGHT",
            )
            if wcol and wcol != "TOTAL WEIGHT [KG]":
                transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Parse dates safely (suppress warnings)
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", dayfirst=True)
            # silently replace unparseable dates with NaT

    # Replace inf/-inf with NaN and infer objects quietly
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df[:] = df.infer_objects()  # avoids dtype warnings

    return feeding, harvest, sampling, transfers

# Enhanced Cage 2 Preprocessing 
# ==============================
# Cage 2 Preprocessing
# ==============================
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """Preprocess Cage 2 data with robust transfer handling, stocking detection, and safe date/numeric handling"""
    
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")

    # Clip data to timeframe and cage
    def _clip(df, cage_col=None):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        if cage_col and cage_col in df.columns:
            df = df[df[cage_col] == cage_number]
        df = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]
        return df.reset_index(drop=True)

    feeding_c2 = _clip(feeding, "CAGE NUMBER")
    harvest_c2 = _clip(harvest, "CAGE NUMBER")
    sampling_c2 = _clip(sampling, "CAGE NUMBER")

    # Stocking defaults
    stocked_fish = 7290
    initial_abw_g = 11.9
    first_inbound_idx = None

    # Handle transfers for stocking
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

    # Base sampling timeline
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g
    }])
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    base = base.sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # Ensure final harvest date is included
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
            if tot_fish > 0:
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

    # Initialize cumulative columns
    for col in ["HARV_FISH_CUM", "HARV_KG_CUM", "IN_FISH_CUM", "IN_KG_CUM", "OUT_FISH_CUM", "OUT_KG_CUM"]:
        base[col] = 0.0

    # Harvest cumulatives
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

    # Transfer cumulatives
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if not t.empty:
            if first_inbound_idx is not None and first_inbound_idx in t.index:
                t = t.drop(index=first_inbound_idx)

            origin_col = find_col(t, ["ORIGIN CAGE", "ORIGIN"], "ORIGIN")
            dest_col = find_col(t, ["DESTINATION CAGE", "DESTINATION"], "DEST")
            fish_col = find_col(t, ["NUMBER OF FISH", "N_FISH"], "FISH")
            kg_col = find_col(t, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")

            t["T_FISH"] = pd.to_numeric(t[fish_col], errors="coerce").fillna(0) if fish_col else 0
            t["T_KG"] = pd.to_numeric(t[kg_col], errors="coerce").fillna(0) if kg_col else 0

            t["ORIGIN_INT"] = t[origin_col].apply(lambda x: int(re.search(r"(\d+)", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d+)", str(x)) else None) if origin_col else np.nan
            t["DEST_INT"] = t[dest_col].apply(lambda x: int(re.search(r"(\d+)", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d+)", str(x)) else None) if dest_col else np.nan

            # Outgoing
            tout = t[t["ORIGIN_INT"] == cage_number].sort_values("DATE")
            if not tout.empty:
                tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
                tout["OUT_KG_CUM"] = tout["T_KG"].cumsum()
                mo = pd.merge_asof(base[["DATE"]], tout[["DATE", "OUT_FISH_CUM", "OUT_KG_CUM"]], on="DATE", direction="backward")
                base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
                base["OUT_KG_CUM"] = mo["OUT_KG_CUM"].fillna(0)

            # Incoming
            tin = t[t["DEST_INT"] == cage_number].sort_values("DATE")
            if not tin.empty:
                tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
                tin["IN_KG_CUM"] = tin["T_KG"].cumsum()
                mi = pd.merge_asof(base[["DATE"]], tin[["DATE", "IN_FISH_CUM", "IN_KG_CUM"]], on="DATE", direction="backward")
                base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
                base["IN_KG_CUM"] = mi["IN_KG_CUM"].fillna(0)

    # Calculate standing fish count
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    # Infer object types to avoid future warnings
    base = base.infer_objects()

    return feeding_c2, harvest_c2, base

# =====================
# =====================
# Enhanced Summary Computation
# =====================
def compute_summary(feeding_c2, sampling_c2):
    """Compute Cage 2 production metrics: biomass, feed, and eFCR (period and aggregated)."""
    
    feeding_c2 = feeding_c2.copy()
    s = sampling_c2.copy().sort_values("DATE")

    # Find relevant columns
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)", "FEED AMOUNT (Kg)", "FEED AMOUNT [KG]", "FEED (KG)", "FEED KG"], "FEED")
    abw_col = find_col(s, ["AVERAGE BODY WEIGHT(G)", "AVERAGE BODY WEIGHT (G)", "ABW(G)", "ABW [G]", "ABW"], "ABW")
    
    if not feed_col or not abw_col:
        return s

    # Cumulative feed
    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    # Merge feed with sampling
    summary = pd.merge_asof(s, feeding_c2[["DATE", "CUM_FEED"]], on="DATE", direction="backward")

    # Biomass calculation
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce").fillna(0)
    summary["BIOMASS_KG"] = summary["FISH_ALIVE"].fillna(0) * summary["ABW_G"] / 1000.0

    # Period-based feed and biomass changes
    summary["FEED_PERIOD_KG"] = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"] = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Period logistics
    for cum_col, per_col in [
        ("IN_KG_CUM", "TRANSFER_IN_KG"),
        ("OUT_KG_CUM", "TRANSFER_OUT_KG"),
        ("HARV_KG_CUM", "HARVEST_KG")
    ]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan

    summary["TRANSFER_IN_FISH"] = summary["IN_FISH_CUM"].diff() if "IN_FISH_CUM" in summary.columns else np.nan
    summary["TRANSFER_OUT_FISH"] = summary["OUT_FISH_CUM"].diff() if "OUT_FISH_CUM" in summary.columns else np.nan
    summary["HARVEST_FISH"] = summary["HARV_FISH_CUM"].diff() if "HARV_FISH_CUM" in summary.columns else np.nan

    # Growth per period (exclude transfers)
    summary["GROWTH_KG"] = summary["ΔBIOMASS_STANDING"].fillna(0) + summary["HARVEST_KG"].fillna(0)

    # Fish count discrepancy
    summary["EXPECTED_FISH_ALIVE"] = (
        summary.get("STOCKED", 0)
        - summary.get("HARV_FISH_CUM", 0)
        + summary.get("IN_FISH_CUM", 0)
        - summary.get("OUT_FISH_CUM", 0)
    )
    actual_fish = pd.to_numeric(summary.get("NUMBER OF FISH"), errors="coerce").fillna(0)
    summary["FISH_COUNT_DISCREPANCY"] = summary["EXPECTED_FISH_ALIVE"].fillna(0) - actual_fish

    # eFCR calculations
    growth_cum = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["PERIOD_eFCR"] = np.where(summary["GROWTH_KG"] > 0, summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum > 0, summary["FEED_AGG_KG"] / growth_cum, np.nan)

    # Set first row metrics to NaN
    first_idx = summary.index.min()
    summary.loc[first_idx, [
        "FEED_PERIOD_KG", "ΔBIOMASS_STANDING",
        "TRANSFER_IN_KG", "TRANSFER_OUT_KG", "HARVEST_KG",
        "TRANSFER_IN_FISH", "TRANSFER_OUT_FISH", "HARVEST_FISH",
        "GROWTH_KG", "PERIOD_eFCR", "FISH_COUNT_DISCREPANCY"
    ]] = np.nan

    return summary


# =====================
# 4. Create mock cages (3-7)
# =====================
def create_mock_cage_data(summary_c2, num_cages=5):
    """Generate realistic mock production data for cages 3–7 based on Cage 2 trends."""
    
    mock_summaries = {}
    
    # Copy first stocking row from Cage 2
    first_row = summary_c2.iloc[0:1].copy()

    for cage_id in range(3, 3 + num_cages):
        mock = summary_c2.copy()
        mock['CAGE NUMBER'] = cage_id
        
        # Replace first row with Cage 2's stocking data
        mock.iloc[0] = first_row.iloc[0]

        # Performance variation factors
        base_perf = np.random.normal(1.0, 0.12)
        growth_eff = np.random.normal(1.0, 0.08)
        feed_eff = np.random.normal(1.0, 0.10)

        for idx in range(1, len(mock)):
            # Biomass variation
            if 'BIOMASS_KG' in mock.columns:
                factor = np.random.normal(base_perf * growth_eff, 0.05)
                mock.at[idx, 'BIOMASS_KG'] *= max(factor, 0.5)

            # ABW variation
            if 'ABW_G' in mock.columns:
                factor = np.random.normal(base_perf, 0.04)
                mock.at[idx, 'ABW_G'] *= max(factor, 0.7)

            # Fish count variation
            if 'NUMBER OF FISH' in mock.columns:
                mortality = np.random.uniform(0.88, 0.98)
                noise = np.random.randint(-30, 15)
                mock.at[idx, 'NUMBER OF FISH'] = max(
                    int(mock.at[idx, 'NUMBER OF FISH'] * mortality) + noise, 100
                )

            # Feed variations
            for feed_col in ['FEED_AGG_KG', 'FEED_PERIOD_KG']:
                if feed_col in mock.columns:
                    factor = np.random.normal(feed_eff, 0.06)
                    mock.at[idx, feed_col] *= max(factor, 0.4)

        # Transfer variations
        transfer_strategy = np.random.choice(['minimal', 'moderate', 'active'], p=[0.4, 0.4, 0.2])
        for col in ['TRANSFER_OUT_KG', 'TRANSFER_IN_KG']:
            if col in mock.columns:
                if transfer_strategy == 'minimal':
                    mock[col] *= np.random.uniform(0.05, 0.2)
                elif transfer_strategy == 'moderate':
                    mock[col] *= np.random.uniform(0.3, 0.8)
                else:
                    mock[col] *= np.random.uniform(0.8, 1.3)

        # Harvest variations
        if 'HARVEST_KG' in mock.columns:
            harvest_strategy = np.random.choice(['early', 'standard', 'late'], p=[0.2, 0.6, 0.2])
            if harvest_strategy == 'early':
                mock['HARVEST_KG'] *= np.random.uniform(1.2, 1.8)
            elif harvest_strategy == 'late':
                mock['HARVEST_KG'] *= np.random.uniform(0.3, 0.7)

        # Recompute growth and eFCR
        if 'GROWTH_KG' in mock.columns:
            mock['GROWTH_KG'] = (
                mock.get('ΔBIOMASS_STANDING', 0)
                + mock.get('HARVEST_KG', 0).fillna(0)
                + mock.get('TRANSFER_OUT_KG', 0).fillna(0)
                - mock.get('TRANSFER_IN_KG', 0).fillna(0)
            )

        if 'GROWTH_KG' in mock.columns and 'FEED_PERIOD_KG' in mock.columns:
            growth_cum = mock['GROWTH_KG'].cumsum(skipna=True)
            mock['PERIOD_eFCR'] = np.where(mock['GROWTH_KG'] > 0, mock['FEED_PERIOD_KG'] / mock['GROWTH_KG'], np.nan)
            mock['AGGREGATED_eFCR'] = np.where(growth_cum > 0, mock['FEED_AGG_KG'] / growth_cum, np.nan)

        # Fish count discrepancy
        if 'FISH_COUNT_DISCREPANCY' in mock.columns:
            quality = np.random.choice(['excellent', 'good', 'fair'], p=[0.3, 0.5, 0.2])
            std_dev = {'excellent': 5, 'good': 15, 'fair': 35}[quality]
            mock['FISH_COUNT_DISCREPANCY'] = np.random.normal(0, std_dev, size=len(mock))

        # Clean extreme values and clip eFCR
        mock = mock.replace([np.inf, -np.inf], np.nan)
        for fcr_col in ['PERIOD_eFCR', 'AGGREGATED_eFCR']:
            if fcr_col in mock.columns:
                mock[fcr_col] = np.clip(mock[fcr_col], 0.5, 5.0)

        mock_summaries[cage_id] = mock

    return mock_summaries

# 5. Streamlit Interface
# Page setup
st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")
st.title("🐟 Fish Cage Production Analysis Dashboard")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

# File upload
feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers (Optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    
    # Load data
    feeding, harvest, sampling, transfers = load_data(
        feeding_file, harvest_file, sampling_file, transfer_file
    )

    # Preprocess cage 2
    feeding_c2, harvest_c2, summary_c2 = preprocess_cage2(
        feeding, harvest, sampling, transfers
    )

    # Compute summary
    summary_c2 = compute_summary(feeding_c2, summary_c2)

    # Generate mock cages 3–7
    mock_cages = create_mock_cage_data(summary_c2)
    all_cages = {2: summary_c2, **mock_cages}

    # Sidebar options
    st.sidebar.header("Visualization Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage].copy()

    # Ensure key display columns exist
    if 'BIOMASS_KG' not in df.columns:
        df['BIOMASS_KG'] = np.nan
    if 'AGGREGATED_eFCR' not in df.columns:
        df['AGGREGATED_eFCR'] = np.nan
    if 'PERIOD_eFCR' not in df.columns:
        df['PERIOD_eFCR'] = np.nan

    # Production summary table
    st.subheader(f"Cage {selected_cage} Production Summary")
    display_cols = ['DATE', 'NUMBER OF FISH', 'BIOMASS_KG', 'FEED_AGG_KG', 'AGGREGATED_eFCR', 'PERIOD_eFCR']
    st.dataframe(df[display_cols].sort_values("DATE").reset_index(drop=True))

    # KPI plots
    if selected_kpi == "Growth":
        fig = px.line(df, x='DATE', y='BIOMASS_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'BIOMASS_KG': 'Biomass (Kg)'})
        st.plotly_chart(fig, use_container_width=True)
    else:  # eFCR
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['DATE'], y=df['AGGREGATED_eFCR'],
            mode='lines+markers', name='Aggregated eFCR'
        ))
        fig.add_trace(go.Scatter(
            x=df['DATE'], y=df['PERIOD_eFCR'],
            mode='lines+markers', name='Period eFCR'
        ))
        fig.update_layout(
            title=f'Cage {selected_cage}: eFCR Over Time',
            xaxis_title='Date', yaxis_title='eFCR'
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload Excel files for Analysis.")
