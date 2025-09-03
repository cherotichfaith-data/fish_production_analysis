# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Page config
st.set_page_config(
    page_title="Fish Production Analysis – Cage 2",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Utility functions
# ==============================
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
    if pd.isna(x): return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

# ==============================
# Load & preprocess data
# ==============================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file))

    # Coerce cages
    for df in [feeding, harvest, sampling]:
        if "CAGE NUMBER" in df.columns:
            df["CAGE NUMBER"] = to_int_cage(df["CAGE NUMBER"])

    if "ORIGIN CAGE" in transfers.columns:
        transfers["ORIGIN CAGE"] = to_int_cage(transfers["ORIGIN CAGE"])
    if "DESTINATION CAGE" in transfers.columns:
        transfers["DESTINATION CAGE"] = to_int_cage(transfers["DESTINATION CAGE"])

    # Parse dates
    for df in [feeding, harvest, sampling, transfers]:
        if "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers


def preprocess_cage2(feeding, harvest, sampling, transfers):
    cage_number = 2

    # Filter Cage 2
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE NUMBER'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Add stocking row manually
    stocking_date = pd.to_datetime("2024-08-26")
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': 7290,
        'AVERAGE BODY WEIGHT (g)': 11.9
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE')

    # Apply transfers to Cage 2
    transfers_c2 = transfers[(transfers['ORIGIN CAGE'] == cage_number) |
                              (transfers['DESTINATION CAGE'] == cage_number)].copy()

    transfer_rows = []
    for _, row in transfers_c2.iterrows():
        if row['ORIGIN CAGE'] == cage_number:
            # fish leaving Cage 2
            transfer_rows.append({
                'DATE': row['DATE'],
                'CAGE NUMBER': cage_number,
                'NUMBER OF FISH': -row['NUMBER OF FISH'],
                'AVERAGE BODY WEIGHT (g)': row.get('ABW (G)', np.nan)
            })
        if row['DESTINATION CAGE'] == cage_number:
            # fish entering Cage 2
            transfer_rows.append({
                'DATE': row['DATE'],
                'CAGE NUMBER': cage_number,
                'NUMBER OF FISH': row['NUMBER OF FISH'],
                'AVERAGE BODY WEIGHT (g)': row.get('ABW (G)', np.nan)
            })

    if transfer_rows:
        transfers_df = pd.DataFrame(transfer_rows)
        sampling_c2 = pd.concat([sampling_c2, transfers_df]).sort_values('DATE')

    # Limit timeframe
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]

    return feeding_c2, harvest_c2, sampling_c2


def compute_summary(feeding_c2, sampling_c2):
    # cumulative feed
    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (KG)'].cumsum()

    # calculate fish count over time
    sampling_c2['CUM_FISH'] = sampling_c2['NUMBER OF FISH'].cumsum()
    sampling_c2['CUM_FISH'] = sampling_c2['CUM_FISH'].clip(lower=0)  # avoid negatives

    # biomass
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['CUM_FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    # merge feed to sampling
    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2.sort_values('DATE')[['DATE', 'CUM_FEED']],
        on='DATE'
    )

    # eFCR
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

    return summary

# ==============================
# Streamlit Interface
# ==============================
st.title("🐟 Fish Production Analysis – Cage 2 (with Transfers)")

st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers", type=["xlsx"])

if feeding_file and harvest_file and sampling_file and transfer_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    # Sidebar selector
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    # Production summary table
    st.subheader("📊 Production Summary (Cage 2, 26-Aug-2024 → 09-Jul-2025)")
    st.dataframe(summary_c2[['DATE', 'CUM_FISH', 'TOTAL_WEIGHT_KG',
                             'AGGREGATED_eFCR', 'PERIOD_eFCR']])

    # Graphs
    if selected_kpi == "Growth":
        fig = px.line(summary_c2, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title='Cage 2: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Biomass (Kg)'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(summary_c2, x='DATE', y='AGGREGATED_eFCR',
                      markers=True, labels={'AGGREGATED_eFCR': 'eFCR'},
                      title='Cage 2: eFCR Over Time')
        fig.add_scatter(x=summary_c2['DATE'], y=summary_c2['PERIOD_eFCR'],
                        mode='lines+markers', name='Period eFCR')
        st.plotly_chart(fig, use_container_width=True)
