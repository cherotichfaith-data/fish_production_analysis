#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# 1. Load data
# -------------------------------
def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfer = pd.read_excel(transfer_file)
    return feeding, harvest, sampling, transfer


# -------------------------------
# 2. Clean & prepare
# -------------------------------
def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df):
    # Drop empty columns
    feeding_df = feeding_df.dropna(axis=1, how="all")
    harvest_df = harvest_df.dropna(axis=1, how="all")
    sampling_df = sampling_df.dropna(axis=1, how="all")
    transfer_df = transfer_df.dropna(axis=1, how="all")

    # Fill missing values
    if "FEED AMOUNT (Kg)" in feeding_df:
        feeding_df["FEED AMOUNT (Kg)"] = feeding_df["FEED AMOUNT (Kg)"].fillna(0)
    if "FEEDING TYPE" in feeding_df:
        feeding_df["FEEDING TYPE"] = feeding_df["FEEDING TYPE"].fillna("Unknown")
    if "feeding_response [very good, good, bad]" in feeding_df:
        feeding_df["feeding_response [very good, good, bad]"] = feeding_df[
            "feeding_response [very good, good, bad]"
        ].fillna("Unknown")

    if "Total weight [kg]" in transfer_df:
        transfer_df["Total weight [kg]"] = transfer_df["Total weight [kg]"].fillna(0)
    if "ABW [g]" in transfer_df:
        transfer_df["ABW [g]"] = transfer_df["ABW [g]"].fillna(0)

    return feeding_df, harvest_df, sampling_df, transfer_df


# -------------------------------
# 3. Preprocess Cage 2
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling):
    cage_number = 2

    # Filter Cage 2
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Ensure numeric columns
    sampling_c2['NUMBER OF FISH'] = pd.to_numeric(sampling_c2['NUMBER OF FISH'], errors='coerce')
    sampling_c2['AVERAGE BODY WEIGHT (g)'] = pd.to_numeric(
        sampling_c2['AVERAGE BODY WEIGHT (g)'], errors='coerce'
    )

    # Add stocking manually
    stocking_date = pd.to_datetime("2024-08-26")
    stocked_fish = 7290
    initial_abw = 11.9
    stocking_row = pd.DataFrame([{
        'DATE': stocking_date,
        'CAGE NUMBER': cage_number,
        'NUMBER OF FISH': stocked_fish,
        'AVERAGE BODY WEIGHT (g)': initial_abw
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('DATE')

    # Limit timeframe
    start_date = pd.to_datetime("2024-07-16")
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
        feeding_c2['FEED AMOUNT (Kg)'] = feeding_c2['FEED AMOUNT (Kg)'].rolling(3, min_periods=1).mean()

    return feeding_c2.sort_values('DATE'), harvest_c2.sort_values('DATE'), sampling_c2.sort_values('DATE')


# -------------------------------
# 4. Compute production summary
# -------------------------------
def compute_production_summary(feeding_df, harvest_df, sampling_df, transfer_df):
    summary = {}

    # Feeding aggregates
    feed_summary = feeding_df.groupby("CAGE NUMBER")["FEED AMOUNT (Kg)"].sum().reset_index()
    summary["feed"] = feed_summary

    # Harvest aggregates
    harvest_summary = harvest_df.groupby("CAGE")[["NUMBER OF FISH", "TOTAL WEIGHT  [kg]"]].sum().reset_index()
    summary["harvest"] = harvest_summary

    # Sampling aggregates
    if not sampling_df.empty:
        sampling_summary = sampling_df.groupby("CAGE NUMBER")["AVERAGE BODY WEIGHT (g)"].mean().reset_index()
        summary["sampling"] = sampling_summary

    # Transfer aggregates
    if not transfer_df.empty:
        transfer_summary = transfer_df.groupby("DESTINATION CAGE")["NUMBER OF FISH"].sum().reset_index()
        summary["transfer"] = transfer_summary

    # EFCR
    efcr_df = pd.merge(feed_summary, harvest_summary, left_on="CAGE NUMBER", right_on="CAGE", how="left")
    efcr_df["EFCR"] = efcr_df["FEED AMOUNT (Kg)"] / efcr_df["TOTAL WEIGHT  [kg]"].replace(0, pd.NA)
    summary["efcr"] = efcr_df

    return summary


# -------------------------------
# 5. Mock cage data
# -------------------------------
def create_mock_cage_data(summary_c2):
    mock_summaries = {}
    for cage_id in range(3, 8):
        mock = summary_c2.copy()
        mock['CAGE NUMBER'] = cage_id

        # Randomize
        if "TOTAL_WEIGHT_KG" in mock:
            mock['TOTAL_WEIGHT_KG'] = np.maximum(
                0.1, mock['TOTAL_WEIGHT_KG'] * np.random.normal(1, 0.05, size=len(mock))
            )
        if "NUMBER OF FISH" in mock:
            mock['NUMBER OF FISH'] = np.maximum(
                0, mock['NUMBER OF FISH'] + np.random.randint(-50, 50, size=len(mock))
            )
        if "CUM_FEED" in mock:
            mock['CUM_FEED'] = np.maximum(
                0.1, mock['CUM_FEED'] * np.random.normal(1, 0.1, size=len(mock))
            )

        # recompute eFCR
        epsilon = 1e-6
        if "CUM_FEED" in mock and "TOTAL_WEIGHT_KG" in mock:
            mock['AGGREGATED_eFCR'] = mock['CUM_FEED'] / (mock['TOTAL_WEIGHT_KG'] + epsilon)
            mock['PERIOD_WEIGHT_GAIN'] = mock['TOTAL_WEIGHT_KG'].diff().fillna(mock['TOTAL_WEIGHT_KG'])
            mock['PERIOD_FEED'] = mock['CUM_FEED'].diff().fillna(mock['CUM_FEED'])
            mock['PERIOD_eFCR'] = mock['PERIOD_FEED'] / (mock['PERIOD_WEIGHT_GAIN'] + epsilon)

        mock_summaries[cage_id] = mock
    return mock_summaries


# -------------------------------
# 6. Streamlit UI
# -------------------------------
st.title("🐟 Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer", type=["xlsx"])

if feeding_file and harvest_file and sampling_file and transfer_file:
    feeding, harvest, sampling, transfer = load_data(
        feeding_file, harvest_file, sampling_file, transfer_file
    )

    # Clean & check missing values
    feeding, harvest, sampling, transfer = clean_and_prepare(feeding, harvest, sampling, transfer)

    st.subheader("🔍 Missing Values Check")
    with st.expander("Show Missing Value Summary"):
        st.write("Feeding:", feeding.isnull().sum())
        st.write("Harvest:", harvest.isnull().sum())
        st.write("Sampling:", sampling.isnull().sum())
        st.write("Transfer:", transfer.isnull().sum())

    # Preprocess Cage 2
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling)

    # Compute summary
    summary_c2 = compute_production_summary(feeding_c2, harvest_c2, sampling_c2, transfer)

    # Mock cages
    mock_cages = create_mock_cage_data(summary_c2["efcr"])
    all_cages = {2: summary_c2["efcr"], **mock_cages}

    # Sidebar options
    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage]

    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df)

    # Plots
    if selected_kpi == "Growth" and "TOTAL_WEIGHT_KG" in df:
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    elif selected_kpi == "eFCR" and "AGGREGATED_eFCR" in df:
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True, title=f'Cage {selected_cage}: eFCR Over Time')
        if "PERIOD_eFCR" in df:
            fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        st.plotly_chart(fig)
