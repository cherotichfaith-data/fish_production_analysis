# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Load data
def load_data(feeding_file, harvest_file, sampling_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    return feeding, harvest, sampling

# ===========================
# 2. Preprocess Cage 2 Robustly
# ===========================
def preprocess_cage2(feeding, harvest, sampling):
    cage_number = 2

    # Normalize column names: strip spaces and uppercase
    feeding.columns = feeding.columns.str.strip().str.upper()
    harvest.columns = harvest.columns.str.strip().str.upper()
    sampling.columns = sampling.columns.str.strip().str.upper()

    # Ensure essential columns exist
    required_sampling_cols = ['CAGE NUMBER', 'DATE', 'NUMBER OF FISH', 'AVERAGE BODY WEIGHT (G)']
    for col in required_sampling_cols:
        if col not in sampling.columns:
            sampling[col] = np.nan

    required_feeding_cols = ['CAGE NUMBER', 'DATE', 'FEED AMOUNT (KG)']
    for col in required_feeding_cols:
        if col not in feeding.columns:
            feeding[col] = np.nan

    # Filter Cage 2
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy() if 'CAGE' in harvest.columns else pd.DataFrame()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

    # Ensure numeric columns
    sampling_c2['NUMBER OF FISH'] = pd.to_numeric(sampling_c2['NUMBER OF FISH'], errors='coerce').fillna(0)
    sampling_c2['AVERAGE BODY WEIGHT (G)'] = pd.to_numeric(sampling_c2['AVERAGE BODY WEIGHT (G)'], errors='coerce').fillna(0)
    feeding_c2['FEED AMOUNT (KG)'] = pd.to_numeric(feeding_c2['FEED AMOUNT (KG)'], errors='coerce').fillna(0)

    # Add initial stocking manually if not present
    stocking_date = pd.to_datetime("2024-08-26")
    stocked_fish = 7290
    initial_abw = 11.9
    if not ((sampling_c2['DATE'] == stocking_date) & (sampling_c2['NUMBER OF FISH'] == stocked_fish)).any():
        stocking_row = pd.DataFrame([{
            'DATE': stocking_date,
            'CAGE NUMBER': cage_number,
            'NUMBER OF FISH': stocked_fish,
            'AVERAGE BODY WEIGHT (G)': initial_abw
        }])
        sampling_c2 = pd.concat([stocking_row, sampling_c2], ignore_index=True)

    # Convert DATE column to datetime
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'], errors='coerce')
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'], errors='coerce')

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
            'FEED AMOUNT (KG)': np.random.uniform(5, 15, size=len(date_range))
        })
        feeding_c2['FEED AMOUNT (KG)'] = feeding_c2['FEED AMOUNT (KG)'].rolling(3, min_periods=1).mean()

    # Sort by date
    feeding_c2 = feeding_c2.sort_values('DATE').reset_index(drop=True)
    sampling_c2 = sampling_c2.sort_values('DATE').reset_index(drop=True)

    return feeding_c2, harvest_c2, sampling_c2


# ===========================
# 3. Compute Summary Robustly
# ===========================
def compute_summary(feeding_c2, sampling_c2):
    # Ensure DATE columns are datetime
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'], errors='coerce')
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'], errors='coerce')

    # Compute cumulative feed
    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (KG)'].cumsum()

    # Compute total biomass
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (G)'] / 1000

    # Merge cumulative feed to sampling
    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2[['DATE', 'CUM_FEED']].sort_values('DATE'),
        on='DATE',
        direction='backward'
    )

    # eFCR calculations
    epsilon = 1e-6  # prevent division by zero
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / (summary['TOTAL_WEIGHT_KG'] + epsilon)
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / (summary['PERIOD_WEIGHT_GAIN'] + epsilon)

    return summary

# 4. Create mock cages (3-7)
def create_mock_cage_data(summary_c2):
    mock_summaries = {}
    for cage_id in range(3, 8):
        mock = summary_c2.copy()
        mock['CAGE NUMBER'] = cage_id

        # Randomize weights ±5%, number of fish ±50, feed ±10%
        mock['TOTAL_WEIGHT_KG'] = np.maximum(0.1, mock['TOTAL_WEIGHT_KG'] * np.random.normal(1, 0.05, size=len(mock)))
        mock['NUMBER OF FISH'] = np.maximum(0, mock['NUMBER OF FISH'] + np.random.randint(-50, 50, size=len(mock)))
        mock['CUM_FEED'] = np.maximum(0.1, mock['CUM_FEED'] * np.random.normal(1, 0.1, size=len(mock)))

        # recompute eFCR
        epsilon = 1e-6
        mock['AGGREGATED_eFCR'] = mock['CUM_FEED'] / (mock['TOTAL_WEIGHT_KG'] + epsilon)
        mock['PERIOD_WEIGHT_GAIN'] = mock['TOTAL_WEIGHT_KG'].diff().fillna(mock['TOTAL_WEIGHT_KG'])
        mock['PERIOD_FEED'] = mock['CUM_FEED'].diff().fillna(mock['CUM_FEED'])
        mock['PERIOD_eFCR'] = mock['PERIOD_FEED'] / (mock['PERIOD_WEIGHT_GAIN'] + epsilon)

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
    st.dataframe(df[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']].round(2))

    # Plot graphs
    if selected_kpi == "Growth":
        df_plot = df.dropna(subset=['TOTAL_WEIGHT_KG'])
        fig = px.line(df_plot, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    else:
        df_plot = df.dropna(subset=['AGGREGATED_eFCR','PERIOD_eFCR'])
        fig = px.line(df_plot, x='DATE', y='AGGREGATED_eFCR', markers=True, name='Aggregated eFCR')
        fig.add_scatter(x=df_plot['DATE'], y=df_plot['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)
