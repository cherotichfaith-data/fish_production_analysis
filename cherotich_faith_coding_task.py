# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

#Column Normalization; to handle messy Excel files, column variants, and cage numbers
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

def to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

def find_col(df: pd.DataFrame, candidates, fuzzy_hint: str | None = None) -> str | None:
    """Enhanced column finder with fuzzy matching"""
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U:
                return orig
    return None

# 1. Load data
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Coerce cage columns
    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    # Transfers
    if transfers is not None:
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # Standardize weight column using enhanced find_col
        wcol = find_col(transfers, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]","WEIGHT (KG)"], fuzzy_hint="WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers


# 2. Preprocess Cage 2 
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # ----------------------
    # Clip function for timeframe
    # ----------------------
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        out = df.dropna(subset=["DATE"]).sort_values("DATE")
        return out[(out["DATE"] >= start_date) & (out["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number])   if "CAGE NUMBER" in feeding.columns else _clip(feeding)
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number])   if "CAGE NUMBER" in harvest.columns else _clip(harvest)
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number]) if "CAGE NUMBER" in sampling.columns else _clip(sampling)

    # ----------------------
    # Dynamic stocking extraction from transfers
    # ----------------------
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

    # ----------------------
    # Stocking row with proper column name
    # ----------------------
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g,  # Use original column name
        "STOCKED": stocked_fish
    }])

    # ----------------------
    # Combine sampling + stocking and ensure date boundaries
    # ----------------------
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True)
    base = base[(base["DATE"] >= start_date) & (base["DATE"] <= end_date)].sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish
    
    # Ensure start date is always present
    if base.empty or base["DATE"].min() > start_date:
        base = pd.concat([stocking_row, base], ignore_index=True).drop_duplicates(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    
    # Ensure end date is present in the timeline
    if base.empty or base["DATE"].max() < end_date:
        # Get last known ABW for end date
        last_abw = base["AVERAGE BODY WEIGHT(G)"].dropna().iloc[-1] if not base.empty and base["AVERAGE BODY WEIGHT(G)"].notna().any() else initial_abw_g
        end_row = pd.DataFrame([{
            "DATE": end_date,
            "CAGE NUMBER": cage_number,
            "AVERAGE BODY WEIGHT(G)": last_abw,
            "STOCKED": stocked_fish
        }])
        base = pd.concat([base, end_row], ignore_index=True).drop_duplicates(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)

    # ----------------------
    # Ensure final harvest date appears with ABW
    # ----------------------
    final_h_date = harvest_c2["DATE"].max() if not harvest_c2.empty else pd.NaT
    if pd.notna(final_h_date):
        hh = harvest_c2[harvest_c2["DATE"] == final_h_date].copy()
        fish_col = find_col(hh, ["NUMBER OF FISH", "NUMBER OF FISH "], "FISH")
        kg_col   = find_col(hh, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "TOTAL WEIGHT  [KG]"], "WEIGHT")
        abw_colh = find_col(hh, ["ABW(G)", "ABW [G]", "ABW"], "ABW")
        abw_final = np.nan
        
        if fish_col and kg_col and hh[fish_col].notna().any() and hh[kg_col].notna().any():
            tot_fish = pd.to_numeric(hh[fish_col], errors="coerce").fillna(0).sum()
            tot_kg   = pd.to_numeric(hh[kg_col],   errors="coerce").fillna(0).sum()
            if tot_fish > 0 and tot_kg > 0:
                abw_final = (tot_kg * 1000.0) / tot_fish
        if np.isnan(abw_final) and abw_colh and hh[abw_colh].notna().any():
            abw_final = pd.to_numeric(hh[abw_colh].map(to_number), errors="coerce").mean()
            
        if pd.notna(abw_final):
            if (base["DATE"] == final_h_date).any():
                base.loc[base["DATE"] == final_h_date, "AVERAGE BODY WEIGHT(G)"] = abw_final
            else:
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
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # ----------------------
    # Harvest cumulatives using enhanced column finding
    # ----------------------
    if not harvest_c2.empty:
        h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
        h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
        
        h = harvest_c2.copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"]   = h["H_KG"].cumsum()
        
        mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # ----------------------
    # Transfers cumulatives (exclude stocking event)
    # ----------------------
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if not t.empty:
            # Exclude stocking inbound transfer
            if first_inbound_idx is not None and first_inbound_idx in t.index:
                t = t.drop(index=first_inbound_idx)

            # Enhanced column finding for transfers
            origin_col = find_col(t, ["ORIGIN CAGE", "ORIGIN", "ORIGIN CAGE NUMBER"], "ORIGIN")
            dest_col   = find_col(t, ["DESTINATION CAGE", "DESTINATION", "DESTINATION CAGE NUMBER"], "DEST")
            fish_col   = find_col(t, ["NUMBER OF FISH", "N_FISH"], "FISH")
            kg_col     = find_col(t, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], "WEIGHT")

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
            t["DEST_INT"]   = t[dest_col].apply(_cage_to_int)   if dest_col   in t.columns else np.nan

            # Outgoing transfers
            tout = t[t["ORIGIN_INT"] == cage_number].sort_values("DATE").copy()
            if not tout.empty:
                tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
                tout["OUT_KG_CUM"]   = tout["T_KG"].cumsum()
                mo = pd.merge_asof(
                    base[["DATE"]].sort_values("DATE"),
                    tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]].sort_values("DATE"),
                    on="DATE", direction="backward"
                )
                base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
                base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

            # Incoming transfers
            tin = t[t["DEST_INT"] == cage_number].sort_values("DATE").copy()
            if not tin.empty:
                tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
                tin["IN_KG_CUM"]   = tin["T_KG"].cumsum()
                mi = pd.merge_asof(
                    base[["DATE"]].sort_values("DATE"),
                    tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]].sort_values("DATE"),
                    on="DATE", direction="backward"
                )
                base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
                base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # ----------------------
    # Standing fish calculation
    # ----------------------
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    s = sampling_c2.copy().sort_values("DATE")

    # Enhanced column finding for feed and ABW
    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)","FEED AMOUNT [KG]","FEED (KG)","FEED KG","FEED_AMOUNT","FEED"], "FEED")
    abw_col  = find_col(s, ["AVERAGE BODY WEIGHT(G)","AVERAGE BODY WEIGHT (G)","ABW(G)","ABW [G]","ABW"], "ABW")
    
    # Return early if essential columns missing
    if not feed_col or not abw_col:
        return s

    # Cumulative feed
    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    # Merge cumulative feed to sampling
    summary = pd.merge_asof(s, feeding_c2[["DATE", "CUM_FEED"]], on="DATE", direction="backward")

    # Enhanced ABW and biomass calculation
    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = (pd.to_numeric(summary["FISH_ALIVE"], errors="coerce").fillna(0) * summary["ABW_G"].fillna(0) / 1000.0)

    # Period deltas
    summary["FEED_PERIOD_KG"]    = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"]       = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Period logistics (kg)
    for cum_col, per_col in [
        ("IN_KG_CUM","TRANSFER_IN_KG"),
        ("OUT_KG_CUM","TRANSFER_OUT_KG"),
        ("HARV_KG_CUM","HARVEST_KG")
    ]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan

    # Period logistics (fish)
    summary["TRANSFER_IN_FISH"]  = summary["IN_FISH_CUM"].diff()   if "IN_FISH_CUM"   in summary.columns else np.nan
    summary["TRANSFER_OUT_FISH"] = summary["OUT_FISH_CUM"].diff()  if "OUT_FISH_CUM"  in summary.columns else np.nan
    summary["HARVEST_FISH"]      = summary["HARV_FISH_CUM"].diff() if "HARV_FISH_CUM" in summary.columns else np.nan

    # Enhanced growth calculation
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
    summary["PERIOD_eFCR"]     = np.where(summary["GROWTH_KG"] > 0, summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum > 0, summary["FEED_AGG_KG"] / growth_cum, np.nan)

    # First row → NA for period metrics
    if not summary.empty:
        first_idx = summary.index.min()
        summary.loc[first_idx, [
            "FEED_PERIOD_KG","ΔBIOMASS_STANDING",
            "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
            "TRANSFER_IN_FISH","TRANSFER_OUT_FISH",
            "GROWTH_KG","PERIOD_eFCR","FISH_COUNT_DISCREPANCY"
        ]] = np.nan

    # Enhanced column selection
    cols = [
        "DATE","CAGE NUMBER","NUMBER OF FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH","HARVEST_FISH",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]
    return summary[[c for c in cols if c in summary.columns]]

#Mock Cages
def generate_mock_cages_full(feeding_c2, sampling_c2, harvest_c2, num_cages=5):
    """
    Generate mock cages with full production summary logic from Cage 2.
    - Daily feeding: +/-10% variation
    - Sampling ABW: +/-5%
    - Harvest: +/-5%
    - Computes growth, ΔBIOMASS, eFCR like Cage 2
    """
    mock_summaries = {}

    for cage in range(3, 3 + num_cages):  # Cage numbers 3,4,5,6,7
        # ----- Feeding -----
        f = feeding_c2.copy()
        f['CAGE NUMBER'] = cage
        if 'FEED_PERIOD_KG' in f.columns:
            f['FEED_PERIOD_KG'] = f['FEED_PERIOD_KG'] * np.random.uniform(0.9, 1.1, size=len(f))
            f['FEED_AGG_KG'] = f['FEED_PERIOD_KG'].cumsum()

        # ----- Sampling -----
        s = sampling_c2.copy()
        s['CAGE NUMBER'] = cage
        if 'ABW_G' in s.columns:
            s['ABW_G'] = s['ABW_G'] * np.random.uniform(0.95, 1.05, size=len(s))
        if 'FISH_ALIVE' in s.columns:
            s['FISH_ALIVE'] = pd.to_numeric(s['FISH_ALIVE'], errors='coerce').fillna(0)
        s['BIOMASS_KG'] = s['FISH_ALIVE'] * s['ABW_G'] / 1000.0

        # ----- Harvest -----
        h = harvest_c2.copy()
        h['CAGE NUMBER'] = cage
        if 'HARVEST_KG' in h.columns:
            h['HARVEST_KG'] = h['HARVEST_KG'] * np.random.uniform(0.95, 1.05, size=len(h))
        if 'HARVEST_FISH' in h.columns:
            h['HARVEST_FISH'] = h['HARVEST_FISH'] * np.random.uniform(0.95, 1.05, size=len(h))

        # ----- Compute full summary like Cage 2 -----
        summary = compute_summary(f, s)

        # Manually adjust GROWTH_KG and eFCR for mock cage using harvest
        summary["GROWTH_KG"] = (
            summary["BIOMASS_KG"].diff().fillna(0)
            + summary["HARVEST_KG"].fillna(0)
        )
        summary["PERIOD_eFCR"] = np.where(summary["GROWTH_KG"] > 0, summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"], np.nan)
        summary["AGGREGATED_eFCR"] = np.where(summary["GROWTH_KG"].cumsum() > 0, summary["FEED_AGG_KG"] / summary["GROWTH_KG"].cumsum(), np.nan)

        # First row: period metrics NA
        first_idx = summary.index.min()
        for col in ["FEED_PERIOD_KG","ΔBIOMASS_STANDING","GROWTH_KG","PERIOD_eFCR"]:
            if col in summary.columns:
                summary.loc[first_idx, col] = np.nan

        mock_summaries[cage] = summary

    return mock_summaries

# ===========================
# Streamlit UI – Cage Selection + KPI
# ===========================
st.title("Fish Cage Production Analysis Dashboard")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file  = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    # Load and preprocess Cage 2
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Generate mock cages (Cages 3–7)
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = generate_mock_cages(feeding_c2, sampling_c2, harvest_c2)
    
    # Combine all summaries into a dictionary
    all_summaries = {2: summary_c2, **mock_summaries}

    # Sidebar: select cage and KPI
    selected_cage = st.sidebar.selectbox("Select Cage", sorted(all_summaries.keys()))
    selected_kpi   = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])
    
    summary_df = all_summaries[selected_cage]

    # Display summary table
    st.subheader(f"Cage {selected_cage} – Production Summary (period-based)")
    show_cols = [
        "DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH","HARVEST_FISH",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]
    display_summary = summary_df[[c for c in show_cols if c in summary_df.columns]]
    st.dataframe(display_summary)
    st.write(f"**Analysis Period:** 26 Aug 2024 to 09 Jul 2025")
    st.write(f"**Data Points:** {len(display_summary)} records from {display_summary['DATE'].min().strftime('%d %b %Y')} to {display_summary['DATE'].max().strftime('%d %b %Y')}")

    # KPI Plots
    if selected_kpi == "Biomass":
        fig = px.line(summary_df.dropna(subset=["BIOMASS_KG"]), x="DATE", y="BIOMASS_KG", markers=True,
                      title=f"Cage {selected_cage}: Biomass Over Time", labels={"BIOMASS_KG":"Total Biomass (kg)"})
        fig.update_xaxes(range=[pd.to_datetime("2024-08-26"), pd.to_datetime("2025-07-09")])
        st.plotly_chart(fig, use_container_width=True)
        
    elif selected_kpi == "ABW":
        fig = px.line(summary_df.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                      title=f"Cage {selected_cage}: Average Body Weight Over Time", labels={"ABW_G":"ABW (g)"})
        fig.update_xaxes(range=[pd.to_datetime("2024-08-26"), pd.to_datetime("2025-07-09")])
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # eFCR
        dff = summary_df.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
                      title=f"Cage {selected_cage}: eFCR Over Time", labels={"AGGREGATED_eFCR":"Aggregated eFCR"})
        fig.update_traces(showlegend=True, name="Aggregated eFCR")
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", showlegend=True, line=dict(dash="dash"))
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        fig.update_xaxes(range=[pd.to_datetime("2024-08-26"), pd.to_datetime("2025-07-09")])
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
