# Fish Cage Production Analysis - Enhanced Version
# Combines robust data processing with comprehensive mock cage generation and interactive UI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# =====================
# Helper Functions
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
# Data Loading
# =====================

def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    """Load and normalize all data files"""
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file is not None else None

    # Coerce cage columns to integers
    cage_columns = {
        feeding: "CAGE NUMBER",
        sampling: "CAGE NUMBER",
        harvest: ["CAGE NUMBER", "CAGE"]
    }
    
    for df, col_names in cage_columns.items():
        if isinstance(col_names, list):
            for col in col_names:
                if col in df.columns:
                    if col == "CAGE":
                        df["CAGE NUMBER"] = to_int_cage(df[col])
                    else:
                        df[col] = to_int_cage(df[col])
                    break
        else:
            if col_names in df.columns:
                df[col_names] = to_int_cage(df[col_names])

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
# Cage 2 Preprocessing (Enhanced)
# =====================

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """
    Enhanced preprocessing for Cage 2 with robust transfer handling
    """
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

    # Ensure final harvest date is included
    final_h_date = harvest_c2["DATE"].max() if not harvest_c2.empty else pd.NaT
    if pd.notna(final_h_date) and not (base["DATE"] == final_h_date).any():
        # Calculate ABW from harvest data
        hh = harvest_c2[harvest_c2["DATE"] == final_h_date].copy()
        fish_col = find_col(hh, ["NUMBER OF FISH", "NUMBER OF FISH "], "FISH")
        kg_col = find_col(hh, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "TOTAL WEIGHT  [KG]"], "WEIGHT")
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

    # Initialize cumulative columns
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

    # Calculate transfer cumulatives (excluding stocking)
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if not t.empty:
            if first_inbound_idx is not None and first_inbound_idx in t.index:
                t = t.drop(index=first_inbound_idx)  # Remove stocking event

            origin_col = find_col(t, ["ORIGIN CAGE", "ORIGIN", "ORIGIN CAGE NUMBER"], "ORIGIN")
            dest_col = find_col(t, ["DESTINATION CAGE", "DESTINATION", "DESTINATION CAGE NUMBER"], "DEST")
            fish_col = find_col(t, ["NUMBER OF FISH", "N_FISH"], "FISH")
            kg_col = find_col(t, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], "WEIGHT")

            if fish_col:
                t["T_FISH"] = pd.to_numeric(t[fish_col], errors="coerce").fillna(0)
            else:
                t["T_FISH"] = pd.Series(0, index=t.index, dtype="float64")
            
            if kg_col:
                t["T_KG"] = pd.to_numeric(t[kg_col], errors="coerce").fillna(0)
            else:
                t["T_KG"] = pd.Series(0, index=t.index, dtype="float64")

            def _cage_to_int(x):
                m = re.search(r"(\d+)", str(x)) if pd.notna(x) else None
                return int(m.group(1)) if m else None

            t["ORIGIN_INT"] = t[origin_col].apply(_cage_to_int) if origin_col in t.columns else np.nan
            t["DEST_INT"] = t[dest_col].apply(_cage_to_int) if dest_col in t.columns else np.nan

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

    # Find relevant columns
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)", "FEED AMOUNT (Kg)", "FEED AMOUNT [KG]", "FEED (KG)", "FEED KG", "FEED_AMOUNT", "FEED"], "FEED")
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

    # Logistics per period (kg)
    for cum_col, per_col in [
        ("IN_KG_CUM", "TRANSFER_IN_KG"),
        ("OUT_KG_CUM", "TRANSFER_OUT_KG"),
        ("HARV_KG_CUM", "HARVEST_KG"),
    ]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan

    # Logistics per period (fish)
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

    # Fish count discrepancy tracking
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

    # Clean and organize columns
    cols = [
        "DATE", "CAGE NUMBER", "NUMBER OF FISH", "ABW_G", "BIOMASS_KG",
        "FEED_PERIOD_KG", "FEED_AGG_KG", "GROWTH_KG",
        "TRANSFER_IN_KG", "TRANSFER_OUT_KG", "HARVEST_KG",
        "TRANSFER_IN_FISH", "TRANSFER_OUT_FISH", "HARVEST_FISH",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR", "AGGREGATED_eFCR",
    ]

    return summary[[c for c in cols if c in summary.columns]]

# =====================
# Mock Cage Generation (Enhanced)
# =====================

def create_mock_cage_data(summary_c2, num_cages=5):
    """Create realistic mock cage data based on Cage 2 performance"""
    mock_summaries = {}
    
    for cage_id in range(3, 3 + num_cages):
        mock = summary_c2.copy()
        mock['CAGE NUMBER'] = cage_id
        
        # Create realistic variations
        base_variance = np.random.normal(1, 0.08, size=len(mock))  # 8% variance
        
        # Vary key metrics with realistic correlations
        growth_factor = np.random.normal(0.95, 0.15)  # Some cages perform better/worse
        feed_efficiency = np.random.normal(1.0, 0.1)  # Feed efficiency variation
        
        # Apply variations to metrics
        if 'BIOMASS_KG' in mock.columns:
            mock['BIOMASS_KG'] *= base_variance * growth_factor
        
        if 'ABW_G' in mock.columns:
            abw_variance = np.random.normal(1, 0.05, size=len(mock))
            mock['ABW_G'] *= abw_variance
        
        if 'NUMBER OF FISH' in mock.columns:
            # Vary fish count with some mortality variation
            mortality_factor = np.random.uniform(0.92, 0.98)  # 2-8% mortality variation
            fish_variance = np.random.randint(-25, 25, size=len(mock))
            mock['NUMBER OF FISH'] = np.maximum(
                (mock['NUMBER OF FISH'] * mortality_factor).astype(int) + fish_variance,
                50  # Minimum fish count
            )
        
        # Feed variations
        if 'FEED_AGG_KG' in mock.columns:
            feed_variance = np.random.normal(feed_efficiency, 0.08, size=len(mock))
            mock['FEED_AGG_KG'] *= feed_variance
            mock['FEED_PERIOD_KG'] *= feed_variance
        
        # Transfer and harvest variations (some cages have different patterns)
        transfer_probability = np.random.random()
        if transfer_probability > 0.7:  # 30% chance of having transfers
            transfer_factor = np.random.uniform(0.3, 1.5)
            for col in ['TRANSFER_OUT_KG', 'TRANSFER_IN_KG', 'HARVEST_KG']:
                if col in mock.columns:
                    mock[col] *= transfer_factor
        else:
            # Minimal transfers for this cage
            for col in ['TRANSFER_OUT_KG', 'TRANSFER_IN_KG']:
                if col in mock.columns:
                    mock[col] *= 0.1
        
        # Recalculate derived metrics
        if 'GROWTH_KG' in mock.columns:
            mock['GROWTH_KG'] = (
                mock.get('ΔBIOMASS_STANDING', 0)
                + mock.get('HARVEST_KG', 0).fillna(0)
                + mock.get('TRANSFER_OUT_KG', 0).fillna(0)
                - mock.get('TRANSFER_IN_KG', 0).fillna(0)
            )
        
        # Recalculate eFCR
        if 'GROWTH_KG' in mock.columns and 'FEED_PERIOD_KG' in mock.columns:
            growth_cum = mock['GROWTH_KG'].cumsum(skipna=True)
            mock['PERIOD_eFCR'] = np.where(
                mock['GROWTH_KG'] > 0,
                mock['FEED_PERIOD_KG'] / mock['GROWTH_KG'],
                np.nan
            )
            mock['AGGREGATED_eFCR'] = np.where(
                growth_cum > 0,
                mock['FEED_AGG_KG'] / growth_cum,
                np.nan
            )
        
        # Add some realistic fish count discrepancies
        if 'FISH_COUNT_DISCREPANCY' in mock.columns:
            discrepancy_noise = np.random.normal(0, 10, size=len(mock))
            mock['FISH_COUNT_DISCREPANCY'] += discrepancy_noise
        
        # Clean up infinite values
        mock = mock.replace([np.inf, -np.inf], np.nan)
        
        mock_summaries[cage_id] = mock
    
    return mock_summaries

# =====================
# Enhanced Streamlit Interface
# =====================

def main():
    st.title("🐟 Advanced Fish Cage Production Analysis")
    st.markdown("""
    ### Comprehensive aquaculture performance tracking with transfer integration
    *Upload your production data to analyze cage performance, growth metrics, and feed conversion efficiency*
    """)

    # Sidebar for file uploads
    st.sidebar.header("📁 Data Upload")
    feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"], help="Daily feeding data by cage")
    harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"], help="Harvest events and weights")
    sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"], help="Periodic sampling for growth tracking")
    transfer_file = st.sidebar.file_uploader("Fish Transfer (Optional)", type=["xlsx"], help="Fish movements between cages")

    if feeding_file and harvest_file and sampling_file:
        # Load and process data
        with st.spinner("🔄 Loading and processing data..."):
            feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
            feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
            summary_c2 = compute_summary(feeding_c2, sampling_c2)

            # Generate mock cages
            mock_cages = create_mock_cage_data(summary_c2, num_cages=5)
            all_cages = {2: summary_c2, **mock_cages}

        # Sidebar controls
        st.sidebar.header("📊 Analysis Controls")
        selected_cage = st.sidebar.selectbox("Select Cage", sorted(all_cages.keys()), help="Choose cage to analyze")
        selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass", "ABW", "eFCR", "Growth"], help="Key performance indicator")
        
        # Show transfer option for Cage 2 only
        show_transfers = False
        if selected_cage == 2 and transfer_file is not None:
            show_transfers = st.sidebar.checkbox("Show Transfer Details", value=False)

        # Get selected cage data
        df = all_cages[selected_cage].copy()

        # Main dashboard layout
        col1, col2, col3 = st.columns([1, 1, 2])

        # Key metrics display
        with col1:
            st.metric("🏷️ Cage", f"#{selected_cage}")
            cage_type = "Real Data" if selected_cage == 2 else "Mock Data"
            st.caption(cage_type)

        with col2:
            if not df.empty and 'BIOMASS_KG' in df.columns:
                final_biomass = df['BIOMASS_KG'].dropna().iloc[-1] if not df['BIOMASS_KG'].dropna().empty else 0
                st.metric("🐟 Final Biomass", f"{final_biomass:.1f} kg")

        with col3:
            if not df.empty and 'AGGREGATED_eFCR' in df.columns:
                final_fcr = df['AGGREGATED_eFCR'].dropna().iloc[-1] if not df['AGGREGATED_eFCR'].dropna().empty else 0
                if not np.isnan(final_fcr) and final_fcr > 0:
                    st.metric("📈 Final eFCR", f"{final_fcr:.2f}")
                    fcr_status = "Excellent" if final_fcr < 1.2 else "Good" if final_fcr < 1.5 else "Needs Improvement"
                    color = "🟢" if final_fcr < 1.2 else "🟡" if final_fcr < 1.5 else "🔴"
                    st.caption(f"{color} {fcr_status}")

        # Production summary table
        st.subheader(f"📋 Cage {selected_cage} Production Summary")
        
        # Select columns for display
        display_cols = [
            "DATE", "NUMBER OF FISH", "ABW_G", "BIOMASS_KG",
            "FEED_PERIOD_KG", "FEED_AGG_KG", "GROWTH_KG",
            "PERIOD_eFCR", "AGGREGATED_eFCR"
        ]
        
        if show_transfers:
            display_cols.extend(["TRANSFER_IN_KG", "TRANSFER_OUT_KG", "HARVEST_KG", "FISH_COUNT_DISCREPANCY"])

        # Format and display table
        display_df = df[[c for c in display_cols if c in df.columns]].copy()
        
        # Rename columns for better display
        column_renames = {
            "DATE": "Date",
            "NUMBER OF FISH": "Fish Count",
            "ABW_G": "ABW (g)",
            "BIOMASS_KG": "Biomass (kg)",
            "FEED_PERIOD_KG": "Period Feed (kg)",
            "FEED_AGG_KG": "Total Feed (kg)",
            "GROWTH_KG": "Growth (kg)",
            "PERIOD_eFCR": "Period eFCR",
            "AGGREGATED_eFCR": "Cumulative eFCR",
            "TRANSFER_IN_KG": "Transfers In (kg)",
            "TRANSFER_OUT_KG": "Transfers Out (kg)",
            "HARVEST_KG": "Harvest (kg)",
            "FISH_COUNT_DISCREPANCY": "Count Discrepancy"
        }
        
        display_df.rename(columns=column_renames, inplace=True)
        
        # Format numeric columns
        numeric_cols = [col for col in display_df.columns if col not in ["Date"]]
        for col in numeric_cols:
            if col in display_df.columns:
                if "eFCR" in col:
                    display_df[col] = display_df[col].round(3)
                else:
                    display_df[col] = display_df[col].round(1)

        st.dataframe(display_df, use_container_width=True, height=300)

        # Visualization section
        st.subheader(f"📊 Cage {selected_cage}: {selected_kpi} Analysis")

        # Create visualizations based on selected KPI
        if selected_kpi == "Biomass":
            df_clean = df.dropna(subset=["BIOMASS_KG"])
            if not df_clean.empty:
                fig = go.Figure()
                
                # Main biomass line
                fig.add_trace(go.Scatter(
                    x=df_clean['DATE'],
                    y=df_clean['BIOMASS_KG'],
                    mode='lines+markers',
                    name='Standing Biomass',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                # Add growth trend line if available
                if 'GROWTH_KG' in df_clean.columns:
                    cumulative_growth = df_clean['GROWTH_KG'].cumsum()
                    fig.add_trace(go.Scatter(
                        x=df_clean['DATE'],
                        y=cumulative_growth,
                        mode='lines',
                        name='Cumulative Growth',
                        line=dict(color='#2ca02c', width=2, dash='dash'),
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title=f'Cage {selected_cage}: Biomass Development Over Time',
                    xaxis_title='Date',
                    yaxis_title='Biomass (kg)',
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Growth rate analysis
                if len(df_clean) > 1:
                    total_days = (df_clean['DATE'].max() - df_clean['DATE'].min()).days
                    initial_biomass = df_clean['BIOMASS_KG'].iloc[0]
                    final_biomass = df_clean['BIOMASS_KG'].iloc[-1]
                    growth_rate = ((final_biomass / initial_biomass) ** (1 / (total_days / 30))) - 1  # Monthly growth rate
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📈 Total Growth", f"{final_biomass - initial_biomass:.1f} kg")
                    with col2:
                        st.metric("📅 Production Days", f"{total_days} days")
                    with col3:
                        st.metric("📊 Monthly Growth Rate", f"{growth_rate:.1%}")

        elif selected_kpi == "ABW":
            df_clean = df.dropna(subset=["ABW_G"])
            if not df_clean.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_clean['DATE'],
                    y=df_clean['ABW_G'],
                    mode='lines+markers',
                    name='Average Body Weight',
                    line=dict(color='#ff7f0e', width=3),
                    marker=dict(size=8)
                ))
                
                # Add growth target line if realistic
                if len(df_clean) > 1:
                    target_abw = df_clean['ABW_G'].iloc[0] * 2  # Example: double the weight target
                    fig.add_hline(
                        y=target_abw,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"Target: {target_abw:.0f}g"
                    )
                
                fig.update_layout(
                    title=f'Cage {selected_cage}: Average Body Weight Growth',
                    xaxis_title='Date',
                    yaxis_title='Average Body Weight (g)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ABW growth analysis
                if len(df_clean) > 1:
                    initial_abw = df_clean['ABW_G'].iloc[0]
                    final_abw = df_clean['ABW_G'].iloc[-1]
                    abw_gain = final_abw - initial_abw
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏁 Final ABW", f"{final_abw:.1f} g")
                    with col2:
                        st.metric("📈 Weight Gain", f"{abw_gain:.1f} g")
                    with col3:
                        growth_multiple = final_abw / initial_abw if initial_abw > 0 else 0
                        st.metric("✨ Growth Multiple", f"{growth_multiple:.1f}x")

        elif selected_kpi == "eFCR":
            df_clean = df.dropna(subset=["AGGREGATED_eFCR", "PERIOD_eFCR"])
            if not df_clean.empty:
                fig = go.Figure()
                
                # Aggregated eFCR
                fig.add_trace(go.Scatter(
                    x=df_clean['DATE'],
                    y=df_clean['AGGREGATED_eFCR'],
                    mode='lines+markers',
                    name='Cumulative eFCR',
                    line=dict(color='#d62728', width=3),
                    marker=dict(size=8)
                ))
                
                # Period eFCR
                fig.add_trace(go.Scatter(
                    x=df_clean['DATE'],
                    y=df_clean['PERIOD_eFCR'],
                    mode='lines+markers',
                    name='Period eFCR',
                    line=dict(color='#ff7f0e', width=2, dash='dot'),
                    marker=dict(size=6)
                ))
                
                # Add efficiency benchmarks
                fig.add_hline(y=1.2, line_dash="dash", line_color="green", 
                            annotation_text="Excellent (1.2)", annotation_position="right")
                fig.add_hline(y=1.5, line_dash="dash", line_color="orange", 
                            annotation_text="Good (1.5)", annotation_position="right")
                
                fig.update_layout(
                    title=f'Cage {selected_cage}: Feed Conversion Ratio Analysis',
                    xaxis_title='Date',
                    yaxis_title='eFCR',
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # FCR performance metrics
                final_fcr = df_clean['AGGREGATED_eFCR'].iloc[-1] if not df_clean.empty else 0
                avg_period_fcr = df_clean['PERIOD_eFCR'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Final eFCR", f"{final_fcr:.3f}")
                with col2:
                    st.metric("📊 Avg Period eFCR", f"{avg_period_fcr:.3f}")
                with col3:
                    efficiency = "Excellent" if final_fcr < 1.2 else "Good" if final_fcr < 1.5 else "Needs Improvement"
                    color = "🟢" if final_fcr < 1.2 else "🟡" if final_fcr < 1.5 else "🔴"
                    st.metric("⭐ Performance", f"{color} {efficiency}")

        else:  # Growth KPI
            df_clean = df.dropna(subset=["GROWTH_KG"])
            if not df_clean.empty:
                fig = go.Figure()
                
                # Cumulative growth
                cumulative_growth = df_clean['GROWTH_KG'].cumsum()
                fig.add_trace(go.Scatter(
                    x=df_clean['DATE'],
                    y=cumulative_growth,
                    mode='lines+markers',
                    name='Cumulative Growth',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=8),
                    fill='tonexty'
                ))
                
                # Period growth as bars
                fig.add_trace(go.Bar(
                    x=df_clean['DATE'],
                    y=df_clean['GROWTH_KG'],
                    name='Period Growth',
                    opacity=0.6,
                    marker_color='#17becf'
                ))
                
                fig.update_layout(
                    title=f'Cage {selected_cage}: Growth Performance',
                    xaxis_title='Date',
                    yaxis_title='Growth (kg)',
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)

        # Transfer details for Cage 2
        if show_transfers and selected_cage == 2:
            st.subheader("🔄 Transfer Activity Analysis")
            
            # Transfer summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_out = df['TRANSFER_OUT_KG'].sum() if 'TRANSFER_OUT_KG' in df.columns else 0
                st.metric("📤 Total Transfers Out", f"{total_out:.1f} kg")
                
            with col2:
                total_in = df['TRANSFER_IN_KG'].sum() if 'TRANSFER_IN_KG' in df.columns else 0
                st.metric("📥 Total Transfers In", f"{total_in:.1f} kg")
                
            with col3:
                total_harvest = df['HARVEST_KG'].sum() if 'HARVEST_KG' in df.columns else 0
                st.metric("🎣 Total Harvest", f"{total_harvest:.1f} kg")
                
            with col4:
                net_change = total_in - total_out
                st.metric("⚖️ Net Transfer", f"{net_change:+.1f} kg")
            
            # Fish count discrepancy analysis
            if 'FISH_COUNT_DISCREPANCY' in df.columns:
                avg_discrepancy = df['FISH_COUNT_DISCREPANCY'].abs().mean()
                max_discrepancy = df['FISH_COUNT_DISCREPANCY'].abs().max()
                
                st.info(f"📊 **Data Quality**: Average fish count discrepancy: {avg_discrepancy:.1f} fish, Maximum: {max_discrepancy:.1f} fish")
                
                if max_discrepancy > 100:
                    st.warning("⚠️ Large fish count discrepancies detected. Review transfer and sampling records for accuracy.")

        # Comparative analysis (if mock cages available)
        if len(all_cages) > 1:
            st.subheader("📈 Comparative Performance Analysis")
            
            # Create comparison dataframe
            comparison_data = []
            for cage_num, cage_data in all_cages.items():
                if not cage_data.empty and 'AGGREGATED_eFCR' in cage_data.columns:
                    final_fcr = cage_data['AGGREGATED_eFCR'].dropna().iloc[-1] if not cage_data['AGGREGATED_eFCR'].dropna().empty else np.nan
                    final_biomass = cage_data['BIOMASS_KG'].dropna().iloc[-1] if 'BIOMASS_KG' in cage_data.columns and not cage_data['BIOMASS_KG'].dropna().empty else np.nan
                    final_abw = cage_data['ABW_G'].dropna().iloc[-1] if 'ABW_G' in cage_data.columns and not cage_data['ABW_G'].dropna().empty else np.nan
                    
                    comparison_data.append({
                        'Cage': cage_num,
                        'Final eFCR': final_fcr,
                        'Final Biomass (kg)': final_biomass,
                        'Final ABW (g)': final_abw,
                        'Type': 'Real Data' if cage_num == 2 else 'Mock Data'
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                # Performance ranking
                comp_df = comp_df.sort_values('Final eFCR')
                comp_df['eFCR Rank'] = range(1, len(comp_df) + 1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🏆 Performance Ranking (by eFCR)**")
                    display_comp = comp_df[['Cage', 'Final eFCR', 'eFCR Rank', 'Type']].copy()
                    display_comp['Final eFCR'] = display_comp['Final eFCR'].round(3)
                    st.dataframe(display_comp, hide_index=True)
                
                with col2:
                    # Performance distribution
                    fig_comp = px.bar(
                        comp_df,
                        x='Cage',
                        y='Final eFCR',
                        color='Type',
                        title='Final eFCR Comparison Across Cages',
                        color_discrete_map={'Real Data': '#1f77b4', 'Mock Data': '#ff7f0e'}
                    )
                    fig_comp.add_hline(y=1.2, line_dash="dash", line_color="green", annotation_text="Excellent")
                    fig_comp.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="Good")
                    fig_comp.update_layout(height=400)
                    st.plotly_chart(fig_comp, use_container_width=True)

        # Data insights and recommendations
        with st.expander("💡 Insights & Recommendations"):
            if selected_cage == 2:
                st.write("### 🔍 Cage 2 Analysis (Real Data)")
                
                # Performance assessment
                if not df.empty and 'AGGREGATED_eFCR' in df.columns:
                    final_fcr = df['AGGREGATED_eFCR'].dropna().iloc[-1] if not df['AGGREGATED_eFCR'].dropna().empty else 0
                    
                    if final_fcr < 1.2:
                        st.success("✅ **Excellent Performance**: Your eFCR is below 1.2, indicating highly efficient feed conversion.")
                    elif final_fcr < 1.5:
                        st.info("ℹ️ **Good Performance**: Your eFCR is between 1.2-1.5, which is acceptable but has room for improvement.")
                        st.write("**Recommendations:**")
                        st.write("- Monitor feeding schedules and adjust based on fish behavior")
                        st.write("- Consider optimizing feed composition or feeding frequency")
                    else:
                        st.warning("⚠️ **Performance Needs Improvement**: eFCR above 1.5 suggests inefficient feed conversion.")
                        st.write("**Action Items:**")
                        st.write("- Review feeding practices and feed quality")
                        st.write("- Check for overfeeding or feed wastage")
                        st.write("- Monitor water quality parameters")
                        st.write("- Consider fish health assessment")
                
                # Growth pattern analysis
                if 'ABW_G' in df.columns and len(df) > 1:
                    growth_periods = df['ABW_G'].diff().dropna()
                    if not growth_periods.empty:
                        consistent_growth = (growth_periods > 0).mean()
                        if consistent_growth > 0.8:
                            st.success("📈 **Consistent Growth**: Fish show steady weight gain throughout the production cycle.")
                        else:
                            st.warning("📉 **Inconsistent Growth**: Consider reviewing environmental conditions and feeding practices.")
            
            else:
                st.write(f"### 🎲 Cage {selected_cage} Analysis (Mock Data)")
                st.write("This cage uses simulated data based on Cage 2 performance with realistic variations.")
                st.write("Mock data helps demonstrate system capabilities and comparative analysis features.")
            
            st.write("### 🔧 Data Processing Notes")
            st.write("""
            **Key Calculations:**
            - **Growth (kg)** = Change in biomass + harvest + transfers out - transfers in
            - **Period eFCR** = Feed consumed in period ÷ Growth in period
            - **Aggregated eFCR** = Total feed consumed ÷ Total growth from stocking
            - **Fish Count Discrepancy** = Expected count - Actual count (for data quality monitoring)
            """)

    else:
        # Landing page when no files uploaded
        st.info("👆 **Please upload the required Excel files to begin analysis**")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("### 📋 Required Files:")
            st.write("""
            1. **Feeding Record** - Daily feed amounts by cage
            2. **Fish Harvest** - Harvest events with weights and counts
            3. **Fish Sampling** - Periodic sampling for growth tracking
            4. **Fish Transfer** (Optional) - Fish movements between cages
            """)
        
        with col2:
            st.write("### 🎯 What You'll Get:")
            st.write("""
            - **Performance Metrics** - eFCR, growth rates, biomass tracking
            - **Interactive Visualizations** - Growth curves, eFCR analysis
            - **Comparative Analysis** - Performance across multiple cages
            - **Data Quality Monitoring** - Fish count discrepancy tracking
            - **Mock Data Generation** - Additional cages for comparison
            """)

        with st.expander("📝 File Format Requirements"):
            st.write("""
            ### Feeding Record
            **Required columns:** DATE, CAGE NUMBER, FEED AMOUNT (Kg)
            
            ### Fish Harvest  
            **Required columns:** DATE, CAGE, NUMBER OF FISH, TOTAL WEIGHT [kg], ABW[g]
            
            ### Fish Sampling
            **Required columns:** DATE, CAGE NUMBER, NUMBER OF FISH, AVERAGE BODY WEIGHT (g)
            
            ### Fish Transfer (Optional)
            **Required columns:** DATE, ORIGIN CAGE, DESTINATION CAGE, NUMBER OF FISH, Total weight [kg]
            
            *Note: Column names are flexible - the system will attempt to match variations in naming and spacing.*
            """)

    # Footer
    st.markdown("---")
    st.markdown("*Enhanced Fish Cage Production Analysis Tool - Built with Streamlit & Plotly*")
    st.caption("🔬 Features comprehensive transfer tracking, period-based metrics, and data quality monitoring")

if __name__ == "__main__":
    main()
