# app.py — Fish Cage Production Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# ===============================
# Helpers
# ===============================
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
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U:
                return orig
    return None

# ===============================
# Load data
# ===============================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file is not None else None

    # Coerce cage columns
    for df, colname in zip([feeding, harvest, sampling], ["CAGE NUMBER", "CAGE NUMBER", "CAGE NUMBER"]):
        if colname in df.columns:
            df[colname] = to_int_cage(df[colname])
    if "CAGE" in harvest.columns and "CAGE NUMBER" not in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    # Transfers
    if transfers is not None:
        for col in ["ORIGIN CAGE", "DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        wcol = find_col(transfers, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT (KG)"], "WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Ensure DATE columns
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

# ===============================
# Preprocess Cage 2 with stocking & transfers
# ===============================
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # Filter
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number])
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number])
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number])

    # Stocking row (from first inbound transfer or default)
    stocked_fish = 7290
    initial_abw_g = 11.9
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        t_in = t[t.get("DESTINATION CAGE", -1) == cage_number].sort_values("DATE")
        if not t_in.empty:
            first = t_in.iloc[0]
            if "NUMBER OF FISH" in t_in.columns: stocked_fish = int(first["NUMBER OF FISH"])
            if "TOTAL WEIGHT [KG]" in t_in.columns and stocked_fish:
                initial_abw_g = float(first["TOTAL WEIGHT [KG]"]) * 1000 / stocked_fish

    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT (g)": initial_abw_g,
        "NUMBER OF FISH": stocked_fish
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values("DATE").reset_index(drop=True)

    return feeding_c2, harvest_c2, sampling_c2

# ===============================
# Compute summary
# ===============================
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.copy()
    sampling_c2 = sampling_c2.copy()

    # Ensure datetime
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'])
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'])

    # Compute cumulative feed
    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()

    # Drop duplicate dates in feeding
    feeding_c2 = feeding_c2.sort_values('DATE').drop_duplicates(subset='DATE', keep='last')

    # Merge asof
    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2[['DATE','CUM_FEED']].sort_values('DATE'),
        on='DATE',
        direction='backward'
    )

    # Metrics
    summary['TOTAL_WEIGHT_KG'] = summary['NUMBER OF FISH'] * summary['AVERAGE BODY WEIGHT (g)'] / 1000
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']

    return summary

# ===============================
# Streamlit UI
# ===============================
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file  = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    st.subheader("Cage 2 – Production Summary (period-based)")
    st.dataframe(summary_c2[['DATE','NUMBER OF FISH','AVERAGE BODY WEIGHT (g)','TOTAL_WEIGHT_KG','PERIOD_eFCR','AGGREGATED_eFCR']])

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth","eFCR"])

    if selected_kpi == "Growth":
        fig = px.line(summary_c2, x='DATE', y='TOTAL_WEIGHT_KG', markers=True, title="Cage 2: Growth Over Time", labels={'TOTAL_WEIGHT_KG':'Total Weight (Kg)'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.line(summary_c2, x='DATE', y='AGGREGATED_eFCR', markers=True, title="Cage 2: eFCR Over Time")
        fig.add_scatter(x=summary_c2['DATE'], y=summary_c2['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(yaxis_title='eFCR')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload the Excel files to begin.")
