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
    """
    Preprocess Cage 2 timeline (or any cage) with consistent columns:
    ABW_G, FISH_ALIVE, STOCKED, HARV_FISH_CUM, HARV_KG_CUM, IN/OUT cumulatives.
    """
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # ----------------------
    # Helper: clip dates and sort
    # ----------------------
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number] if "CAGE NUMBER" in feeding.columns else feeding)
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number] if "CAGE NUMBER" in harvest.columns else harvest)
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number] if "CAGE NUMBER" in sampling.columns else sampling)

    # ----------------------
    # Standardize ABW column
    # ----------------------
    abw_col_candidates = ["AVERAGE_BODY_WEIGHT(G)", "ABW(G)", "ABW_G", "ABW"]
    for col in abw_col_candidates:
        if col in sampling_c2.columns:
            sampling_c2["ABW_G"] = pd.to_numeric(sampling_c2[col].map(to_number), errors="coerce")
            break
    if "ABW_G" not in sampling_c2.columns:
        sampling_c2["ABW_G"] = np.nan

    # ----------------------
    # Stocking info (from transfers if available)
    # ----------------------
    stocked_fish = 7290
    initial_abw_g = 11.9
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        t_in = t[t.get("DESTINATION CAGE", t.get("DEST_CAGE", -1)) == cage_number].sort_values("DATE")
        if not t_in.empty:
            first = t_in.iloc[0]
            if "NUMBER OF FISH" in t_in.columns and pd.notna(first.get("NUMBER OF FISH")):
                stocked_fish = int(float(first["NUMBER OF FISH"]))
            if "TOTAL WEIGHT [KG]" in t_in.columns and pd.notna(first.get("TOTAL WEIGHT [KG]")):
                initial_abw_g = float(first["TOTAL WEIGHT [KG]"]) * 1000 / stocked_fish

    # ----------------------
    # Create stocking row
    # ----------------------
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "ABW_G": initial_abw_g,
        "STOCKED": stocked_fish
    }])

    # ----------------------
    # Combine sampling + stocking
    # ----------------------
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # ----------------------
    # Ensure end date row exists
    # ----------------------
    if base["DATE"].max() < end_date:
        last_abw = base["ABW_G"].dropna().iloc[-1] if base["ABW_G"].notna().any() else initial_abw_g
        base = pd.concat([base, pd.DataFrame([{
            "DATE": end_date,
            "CAGE NUMBER": cage_number,
            "ABW_G": last_abw,
            "STOCKED": stocked_fish
        }])], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # ----------------------
    # Initialize cumulative columns
    # ----------------------
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # ----------------------
    # Harvest cumulatives
    # ----------------------
    if not harvest_c2.empty:
        h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
        h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
        h = harvest_c2.copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"]   = h["H_KG"].cumsum()

        mh = pd.merge_asof(base[["DATE"]].sort_values("DATE"), h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]].sort_values("DATE"),
                           on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # ----------------------
    # Transfers cumulatives (optional)
    # ----------------------
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        # Convert cage numbers to int
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in t.columns:
                t[col] = t[col].apply(lambda x: int(str(x).strip()) if pd.notna(x) and str(x).strip().isdigit() else np.nan)
        # Outgoing
        tout = t[t.get("ORIGIN CAGE",-1) == cage_number].sort_values("DATE").copy()
        if not tout.empty:
            fish_col = find_col(tout, ["NUMBER OF FISH","N_FISH"], "FISH")
            kg_col   = find_col(tout, ["TOTAL WEIGHT [KG]","WEIGHT"], "WEIGHT")
            tout["OUT_FISH_CUM"] = pd.to_numeric(tout[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tout["OUT_KG_CUM"]   = pd.to_numeric(tout[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mo = pd.merge_asof(base[["DATE"]], tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)
        # Incoming
        tin = t[t.get("DESTINATION CAGE",-1) == cage_number].sort_values("DATE").copy()
        if not tin.empty:
            fish_col = find_col(tin, ["NUMBER OF FISH","N_FISH"], "FISH")
            kg_col   = find_col(tin, ["TOTAL WEIGHT [KG]","WEIGHT"], "WEIGHT")
            tin["IN_FISH_CUM"] = pd.to_numeric(tin[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tin["IN_KG_CUM"]   = pd.to_numeric(tin[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mi = pd.merge_asof(base[["DATE"]], tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
            base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # ----------------------
    # Standing fish alive
    # ----------------------
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base

#compute summary
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

# Generate mock cages 
def generate_mock_cages(feeding_c2, sampling_c2, harvest_c2, num_cages=5):
    """
    Generate mock cages (default 3–7) based on Cage 2 data.
    Adds random variations to simulate different cages.
    Returns feeding, sampling, harvest lists and summaries dictionary.
    """
    mock_feeding = []
    mock_sampling = []
    mock_harvest = []
    mock_summaries = {}

    # Ensure ABW column exists
    if "ABW_G" not in sampling_c2.columns:
        raise KeyError("Sampling data must have 'ABW_G' column from preprocess_cage2!")

    for cage in range(3, 3 + num_cages):  # Cage numbers 3,4,5,...
        # ----- Feeding -----
        f = feeding_c2.copy()
        f["CAGE NUMBER"] = cage
        feed_col = find_col(f, ["FEED AMOUNT (KG)","FEED KG","FEED_PERIOD_KG","FEED"], "FEED")
        if feed_col:
            f[feed_col] = f[feed_col] * np.random.uniform(0.9, 1.1, size=len(f))
        mock_feeding.append(f)

        # ----- Sampling -----
        s = sampling_c2.copy()
        s["CAGE NUMBER"] = cage
        s["ABW_G"] = s["ABW_G"] * np.random.uniform(0.95, 1.05, size=len(s))
        if "FISH_ALIVE" in s.columns:
            s["FISH_ALIVE"] = pd.to_numeric(s["FISH_ALIVE"], errors="coerce").fillna(0)
        s["BIOMASS_KG"] = s["FISH_ALIVE"] * s["ABW_G"] / 1000
        mock_sampling.append(s)

        # ----- Harvest -----
        h = harvest_c2.copy()
        h["CAGE NUMBER"] = cage
        fish_col = find_col(h, ["NUMBER OF FISH","N_FISH"], "FISH")
        kg_col   = find_col(h, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]"], "WEIGHT")
        if kg_col:
            h[kg_col] = h[kg_col] * np.random.uniform(0.95, 1.05, size=len(h))
        if fish_col:
            h[fish_col] = pd.to_numeric(h[fish_col], errors="coerce").fillna(0)
        mock_harvest.append(h)

        # ----- Summary -----
        summary = compute_summary(f, s)
        mock_summaries[cage] = summary

    return mock_feeding, mock_sampling, mock_harvest, mock_summaries

# ===========================
# Streamlit UI – Cage Selection + KPI (Styled)
# ===========================
import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Fish Cage Production Dashboard",
    layout="wide",
    page_icon="🐟"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Sidebar headers */
    .sidebar .stHeader {
        color: #1E90FF;
        font-weight: bold;
    }

    /* KPI summary cards */
    .kpi-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }

    /* Table styling */
    .dataframe th {
        background-color: #1E90FF;
        color: white;
    }

    .dataframe td {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-title">Fish Cage Production Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar upload & selections
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

    # KPI cards
    st.subheader(f"Cage {selected_cage} – Production Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-card"><h3>Total Biomass</h3><p>{summary_df["BIOMASS_KG"].sum():,.2f} kg</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><h3>Average ABW</h3><p>{summary_df["ABW_G"].mean():,.2f} g</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><h3>Average eFCR</h3><p>{summary_df["AGGREGATED_eFCR"].mean():.2f}</p></div>', unsafe_allow_html=True)

    # Display summary table
    show_cols = [
        "DATE","NUMBER OF FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH","HARVEST_FISH",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]
    display_summary = summary_df[[c for c in show_cols if c in summary_df.columns]]
    st.dataframe(display_summary, use_container_width=True)
    
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
