# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Set page config 
st.set_page_config( 
    page_title="Fish Production Analysis", 
    page_icon="🐟", layout="wide", 
    initial_sidebar_state="expanded"
)

#start by defining the utility functions
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip, collapse spaces, upper-case"""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    """Extract cage number (int) from mixed labels like 'CAGE 3 A' or 'C3A'"""
    def _coerce(x):
        if pd.isna(x): return None
        if isinstance(x, (int, np.integer)): return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def find_col(df: pd.DataFrame, candidates, fuzzy_hint: str | None = None) -> str | None:
    """Find a column in df matching one of candidates"""
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
    """Convert messy numeric strings (with commas, text) into floats"""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# Load data
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None, verbose=True):
    """
    Load + normalize the four input files and coerce key columns.
    Returns dict: feeding, harvest, sampling, transfers
    """
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Feeding
    c = find_col(feeding, ["CAGE NUMBER", "CAGE"], "CAGE")
    if c: feeding["CAGE NUMBER"] = to_int_cage(feeding[c])
    fa = find_col(feeding, ["FEED AMOUNT (KG)", "FEED AMOUNT [KG]", "FEED (KG)", "FEED"], "FEED")
    if fa: feeding["FEED AMOUNT (KG)"] = feeding[fa].apply(to_number)
    # parse dates
    if "DATE" in feeding.columns:
        feeding["DATE"] = pd.to_datetime(feeding["DATE"], errors="coerce")

    # Harvest
    c = find_col(harvest, ["CAGE NUMBER", "CAGE"], "CAGE")
    if c: harvest["CAGE NUMBER"] = to_int_cage(harvest[c])
    hfish = find_col(harvest, ["NUMBER OF FISH"], "FISH")
    if hfish: harvest["NUMBER OF FISH"] = pd.to_numeric(harvest[hfish].map(to_number), errors="coerce")
    hkg = find_col(harvest, ["TOTAL WEIGHT (KG)", "TOTAL WEIGHT [KG]"], "WEIGHT")
    if hkg: harvest["TOTAL WEIGHT [KG]"] = pd.to_numeric(harvest[hkg].map(to_number), errors="coerce")
    habw = find_col(harvest, ["ABW (G)", "ABW [G]", "ABW(G)", "ABW"], "ABW")
    if habw: harvest["ABW (G)"] = pd.to_numeric(harvest[habw].map(to_number), errors="coerce")
    if "DATE" in harvest.columns:
        harvest["DATE"] = pd.to_datetime(harvest["DATE"], errors="coerce")

    # Sampling
    c = find_col(sampling, ["CAGE NUMBER", "CAGE"], "CAGE")
    if c: sampling["CAGE NUMBER"] = to_int_cage(sampling[c])
    sfish = find_col(sampling, ["NUMBER OF FISH"], "FISH")
    if sfish: sampling["NUMBER OF FISH"] = pd.to_numeric(sampling[sfish].map(to_number), errors="coerce")
    sabw = find_col(sampling, ["AVERAGE BODY WEIGHT (G)", "ABW (G)", "ABW [G]", "ABW(G)", "ABW"], "WEIGHT")
    if sabw: sampling["AVERAGE BODY WEIGHT (G)"] = pd.to_numeric(sampling[sabw].map(to_number), errors="coerce")
    if "DATE" in sampling.columns:
        sampling["DATE"] = pd.to_datetime(sampling["DATE"], errors="coerce")

    # Transfers
    if transfers is not None:
        oc = find_col(transfers, ["ORIGIN CAGE", "ORIGIN", "ORIGIN CAGE NUMBER"], "ORIGIN")
        dc = find_col(transfers, ["DESTINATION CAGE", "DESTINATION", "DESTINATION CAGE NUMBER"], "DEST")
        if oc: transfers["ORIGIN CAGE"] = to_int_cage(transfers[oc])
        if dc: transfers["DESTINATION CAGE"] = to_int_cage(transfers[dc])
        tfish = find_col(transfers, ["NUMBER OF FISH", "N_FISH"], "FISH")
        if tfish: transfers["NUMBER OF FISH"] = pd.to_numeric(transfers[tfish].map(to_number), errors="coerce")
        tkg = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], "WEIGHT")
        if tkg and tkg != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={tkg: "TOTAL WEIGHT [KG]"}, inplace=True)
        if "TOTAL WEIGHT [KG]" in transfers.columns:
            transfers["TOTAL WEIGHT [KG]"] = pd.to_numeric(transfers["TOTAL WEIGHT [KG]"].map(to_number), errors="coerce")
        tabw = find_col(transfers, ["ABW (G)", "ABW [G]", "ABW(G)"], "ABW")
        if tabw: transfers["ABW (G)"] = pd.to_numeric(transfers[tabw].map(to_number), errors="coerce")
        if "DATE" in transfers.columns:
            transfers["DATE"] = pd.to_datetime(transfers["DATE"], errors="coerce")

    data = {"feeding": feeding, "harvest": harvest, "sampling": sampling, "transfers": transfers}

    if verbose:
        print("=== Data Summary ===")
        for k, df in data.items():
            if df is not None and not df.empty:
                dmin = df["DATE"].min() if "DATE" in df.columns else None
                dmax = df["DATE"].max() if "DATE" in df.columns else None
                print(f"{k:<10} rows={len(df):>5} | {dmin} → {dmax}")
        print("====================")

    return data
def apply_transfers(sampling_df, transfers, cage_number):
    """
    Adjust NUMBER OF FISH and TOTAL_WEIGHT_KG in sampling_df 
    based on transfers involving this cage.
    """
    if transfers is None or transfers.empty:
        return sampling_df

    sampling_df = sampling_df.copy()

    for _, t in transfers.iterrows():
        date = pd.to_datetime(t["DATE"], errors="coerce")
        fish = t.get("NUMBER OF FISH", 0) or 0
        weight = t.get("TOTAL WEIGHT [KG]", 0) or 0

        if pd.isna(date):
            continue

        # Origin cage (fish leave)
        if t.get("ORIGIN CAGE") == cage_number:
            mask = sampling_df["DATE"] >= date
            sampling_df.loc[mask, "NUMBER OF FISH"] -= fish
            sampling_df.loc[mask, "TOTAL_WEIGHT_KG"] -= weight

        # Destination cage (fish arrive)
        if t.get("DESTINATION CAGE") == cage_number:
            mask = sampling_df["DATE"] >= date
            sampling_df.loc[mask, "NUMBER OF FISH"] += fish
            sampling_df.loc[mask, "TOTAL_WEIGHT_KG"] += weight

    return sampling_df


# 2. Preprocess Cage 2
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2

    # Filter Cage 2
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE NUMBER'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Add stocking manually
    stocking_date = pd.to_datetime("2024-08-26")
    stocked_fish = 7290
    initial_abw = 11.9
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': stocked_fish,
        'AVERAGE BODY WEIGHT (G)': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE')

    # Limit timeframe
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]

    # If feeding is empty, create synthetic daily feed
    if feeding_c2.empty:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        feeding_c2 = pd.DataFrame({
            'DATE': date_range,
            'CAGE NUMBER': cage_number,
            'FEED AMOUNT (Kg)': np.random.uniform(5, 15, size=len(date_range))
        })

    # Compute biomass before transfers
    sampling_c2["TOTAL_WEIGHT_KG"] = sampling_c2["NUMBER OF FISH"] * sampling_c2["AVERAGE BODY WEIGHT (G)"] / 1000

    # Apply transfers
    sampling_c2 = apply_transfers(sampling_c2, transfers, cage_number)

    return feeding_c2, harvest_c2, sampling_c2

# 3. Compute production summary
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'])
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'])

    # cumulative feed
    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()

    # biomass in kg
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (G)'] / 1000

    # merge cumulative feed into sampling
    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2.sort_values('DATE')[['DATE', 'CUM_FEED']],
        on='DATE'
    )

    # aggregated eFCR = total feed used ÷ biomass at sampling
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']

    # period-based metrics
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff()
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff()

    # handle first row (stocking)
    summary.loc[0, 'PERIOD_WEIGHT_GAIN'] = summary.loc[0, 'TOTAL_WEIGHT_KG']
    summary.loc[0, 'PERIOD_FEED'] = summary.loc[0, 'CUM_FEED']

    # period eFCR = feed during the period ÷ weight gain during the period
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

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
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    # Load all datasets into a dict
    data = load_data(
        feeding_file,
        harvest_file,
        sampling_file,
        transfer_file=transfer_file,
        verbose=True
    )

    # Extract individual DataFrames
    feeding = data["feeding"]
    harvest = data["harvest"]
    sampling = data["sampling"]
    transfers = data["transfers"]

    # Preprocess Cage 2 only
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(
        feeding, harvest, sampling, transfers
    )

    # Compute production summary
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Mock cages for comparison
    mock_cages = create_mock_cage_data(summary_c2)
    all_cages = {2: summary_c2, **mock_cages}

    # Sidebar selectors
    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage]

    # Show production summary table
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(
        df[["DATE", "NUMBER OF FISH", "TOTAL_WEIGHT_KG", "AGGREGATED_eFCR", "PERIOD_eFCR"]]
    )

    # Plot graphs
    if selected_kpi == "Growth":
        df["TOTAL_WEIGHT_KG"] = pd.to_numeric(df["TOTAL_WEIGHT_KG"], errors="coerce")
        df = df.dropna(subset=["TOTAL_WEIGHT_KG"])
        fig = px.line(
            df, x="DATE", y="TOTAL_WEIGHT_KG", markers=True,
            title=f"Cage {selected_cage}: Growth Over Time",
            labels={"TOTAL_WEIGHT_KG": "Total Weight (Kg)"}
        )
        st.plotly_chart(fig)

    else:  # eFCR
        df["AGGREGATED_eFCR"] = pd.to_numeric(df["AGGREGATED_eFCR"], errors="coerce")
        df["PERIOD_eFCR"] = pd.to_numeric(df["PERIOD_eFCR"], errors="coerce")
        df = df.dropna(subset=["AGGREGATED_eFCR", "PERIOD_eFCR"])
        fig = px.line(df, x="DATE", y="AGGREGATED_eFCR", markers=True)
        fig.add_scatter(x=df["DATE"], y=df["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR")
        fig.update_layout(title=f"Cage {selected_cage}: eFCR Over Time", yaxis_title="eFCR")
        st.plotly_chart(fig)
