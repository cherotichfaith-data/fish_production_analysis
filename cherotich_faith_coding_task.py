import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===============================
# 1. Load Data
# ===============================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)
    transfers = pd.read_excel(transfer_file)
    
    # Handle missing values
    feeding.fillna({'FEED AMOUNT (Kg)': 0}, inplace=True)
    sampling.fillna({'NUMBER OF FISH': 0, 'AVERAGE BODY WEIGHT (g)': 0}, inplace=True)
    harvest.fillna({'NUMBER OF FISH': 0, 'AVERAGE BODY WEIGHT (g)': 0}, inplace=True)
    transfers.fillna({'NUMBER_FISH': 0, 'WEIGHT': 0}, inplace=True)

    # Correct transfer weights if given in grams
    if 'WEIGHT' in transfers.columns:
        transfers['WEIGHT_KG'] = transfers['WEIGHT']
        if transfers['WEIGHT'].max() > 1000:  # likely grams
            transfers['WEIGHT_KG'] = transfers['WEIGHT'] / 1000

    return feeding, harvest, sampling, transfers

# ===============================
# 2. Preprocess Cage 2 with transfers
# ===============================
def preprocess_cage2(feeding, harvest, sampling, transfers):
    cage_number = 2
    feeding_c2 = feeding[feeding['CAGE NUMBER'] == cage_number].copy()
    harvest_c2 = harvest[harvest['CAGE'] == cage_number].copy()
    sampling_c2 = sampling[sampling['CAGE NUMBER'] == cage_number].copy()

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
    start_date = pd.to_datetime("2024-08-26")
    end_date = pd.to_datetime("2025-07-09")
    sampling_c2 = sampling_c2[(sampling_c2['DATE'] >= start_date) & (sampling_c2['DATE'] <= end_date)]
    feeding_c2 = feeding_c2[(feeding_c2['DATE'] >= start_date) & (feeding_c2['DATE'] <= end_date)]

    # Apply transfers out of Cage 2
    if transfers is not None and not transfers.empty:
        transfers_out = transfers[transfers['ORIGIN CAGE'] == cage_number].copy()
        transfers_out['NUMBER_FISH'].fillna(0, inplace=True)
        for idx, row in transfers_out.iterrows():
            date_mask = sampling_c2['DATE'] >= pd.to_datetime(row['DATE'])
            sampling_c2.loc[date_mask, 'NUMBER OF FISH'] -= row['NUMBER_FISH']

    # Ensure no negative numbers
    sampling_c2['NUMBER OF FISH'] = sampling_c2['NUMBER OF FISH'].clip(lower=0)
    return feeding_c2, harvest_c2, sampling_c2

# ===============================
# 3. Compute Production Summary
# ===============================
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2['DATE'] = pd.to_datetime(feeding_c2['DATE'])
    sampling_c2['DATE'] = pd.to_datetime(sampling_c2['DATE'])

    feeding_c2['FEED AMOUNT (Kg)'].fillna(0, inplace=True)
    sampling_c2['NUMBER OF FISH'].fillna(0, inplace=True)
    sampling_c2['AVERAGE BODY WEIGHT (g)'].fillna(0, inplace=True)

    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    summary = pd.merge_asof(
        sampling_c2.sort_values('DATE'),
        feeding_c2.sort_values('DATE')[['DATE', 'CUM_FEED']],
        on='DATE'
    )

    summary['CUM_FEED'] = summary['CUM_FEED'].fillna(method='ffill').fillna(0)
    summary['TOTAL_WEIGHT_KG'] = summary['TOTAL_WEIGHT_KG'].fillna(0)
    summary['AGGREGATED_eFCR'] = np.where(summary['TOTAL_WEIGHT_KG']>0, summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG'], np.nan)
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = np.where(summary['PERIOD_WEIGHT_GAIN']>0, summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN'], np.nan)

    return summary

# ===============================
# 4. Create Mock Cages (3-7)
# ===============================
def create_mock_cages(summary_c2, feeding_c2, sampling_c2):
    mock_summaries = {}
    cage_ids = range(3, 8)
    sampling_dates = sampling_c2['DATE'].tolist()

    start_date = feeding_c2['DATE'].min() if not feeding_c2.empty else sampling_c2['DATE'].min()
    end_date = feeding_c2['DATE'].max() if not feeding_c2.empty else sampling_c2['DATE'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for cage_id in cage_ids:
        daily_feed = pd.DataFrame({
            'DATE': date_range,
            'CAGE NUMBER': cage_id,
            'FEED AMOUNT (Kg)': np.random.normal(10, 1, size=len(date_range))
        })
        daily_feed['FEED AMOUNT (Kg)'] = daily_feed['FEED AMOUNT (Kg)'].clip(lower=0)
        daily_feed['CUM_FEED'] = daily_feed['FEED AMOUNT (Kg)'].cumsum()

        mock_sampling = pd.DataFrame({
            'DATE': sampling_dates,
            'CAGE NUMBER': cage_id,
            'NUMBER OF FISH': (summary_c2['NUMBER OF FISH'].values + np.random.randint(-50, 50, len(sampling_dates))).clip(min=0),
            'AVERAGE BODY WEIGHT (g)': (summary_c2['AVERAGE BODY WEIGHT (g)'].values * np.random.normal(1, 0.05, len(sampling_dates))).clip(min=0)
        })
        mock_sampling['TOTAL_WEIGHT_KG'] = mock_sampling['NUMBER OF FISH'] * mock_sampling['AVERAGE BODY WEIGHT (g)'] / 1000

        summary = pd.merge_asof(
            mock_sampling.sort_values('DATE'),
            daily_feed.sort_values('DATE')[['DATE', 'CUM_FEED']],
            on='DATE'
        )
        summary['CUM_FEED'] = summary['CUM_FEED'].fillna(method='ffill').fillna(0)
        summary['TOTAL_WEIGHT_KG'] = summary['TOTAL_WEIGHT_KG'].fillna(0)
        summary['AGGREGATED_eFCR'] = np.where(summary['TOTAL_WEIGHT_KG']>0, summary['CUM_FEED']/summary['TOTAL_WEIGHT_KG'], np.nan)
        summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
        summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
        summary['PERIOD_eFCR'] = np.where(summary['PERIOD_WEIGHT_GAIN']>0, summary['PERIOD_FEED']/summary['PERIOD_WEIGHT_GAIN'], np.nan)

        mock_summaries[cage_id] = summary

    return mock_summaries

# ===============================
# 5. Streamlit Interface
# ===============================
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfers", type=["xlsx"])

if feeding_file and harvest_file and sampling_file and transfer_file:
    feeding, harvest, sampling, transfers = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)

    mock_cages = create_mock_cages(summary_c2, feeding_c2, sampling_c2)
    all_cages = {2: summary_c2, **mock_cages}

    # Sidebar selectors
    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    # Display table
    df = all_cages[selected_cage]
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[['DATE', 'NUMBER OF FISH', 'TOTAL_WEIGHT_KG', 'AGGREGATED_eFCR', 'PERIOD_eFCR']])

    # Plot KPI
    if selected_kpi == "Growth":
        df_plot = df.dropna(subset=['TOTAL_WEIGHT_KG'])
        fig = px.line(df_plot, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time', labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig)
    else:
        df_plot = df.dropna(subset=['AGGREGATED_eFCR', 'PERIOD_eFCR'])
        fig = px.line(df_plot, x='DATE', y='AGGREGATED_eFCR', markers=True)
        fig.add_scatter(x=df_plot['DATE'], y=df_plot['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig)
