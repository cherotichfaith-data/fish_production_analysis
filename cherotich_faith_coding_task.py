# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# 1. Load and clean data
# -------------------------------
def load_data(feeding_file, harvest_file, sampling_file):
    feeding = pd.read_excel(feeding_file)
    harvest = pd.read_excel(harvest_file)
    sampling = pd.read_excel(sampling_file)

    # Strip columns to prevent KeyErrors
    feeding.columns = feeding.columns.str.strip()
    harvest.columns = harvest.columns.str.strip()
    sampling.columns = sampling.columns.str.strip()

    # Convert DATE columns to datetime
    feeding['DATE'] = pd.to_datetime(feeding['DATE'])
    harvest['DATE'] = pd.to_datetime(harvest['DATE'])
    sampling['DATE'] = pd.to_datetime(sampling['DATE'])

    return feeding, harvest, sampling

# -------------------------------
# 2. Preprocess Cage 2
# -------------------------------
def preprocess_cage2(feeding, harvest, sampling):
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

    return feeding_c2, harvest_c2, sampling_c2

# -------------------------------
# 3. Compute production summary
# -------------------------------
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2 = feeding_c2.sort_values('DATE')
    sampling_c2 = sampling_c2.sort_values('DATE')

    feeding_c2['CUM_FEED'] = feeding_c2['FEED AMOUNT (Kg)'].cumsum()
    sampling_c2['TOTAL_WEIGHT_KG'] = sampling_c2['NUMBER OF FISH'] * sampling_c2['AVERAGE BODY WEIGHT (g)'] / 1000

    summary = pd.merge_asof(
        sampling_c2,
        feeding_c2[['DATE','CUM_FEED']],
        on='DATE'
    )

    summary['CUM_FEED'] = summary['CUM_FEED'].fillna(method='ffill').fillna(0)
    summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
    summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
    summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
    summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

    return summary

# -------------------------------
# 4. Create mock cages (3-7)
# -------------------------------
def create_mock_cages(summary_c2, feeding_c2, sampling_c2):
    mock_summaries = {}
    cage_ids = range(3, 8)
    sampling_dates = sampling_c2['DATE'].tolist()

    start_date = feeding_c2['DATE'].min()
    end_date = feeding_c2['DATE'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for cage_id in cage_ids:
        # Daily feeding (no negative values)
        daily_feed = pd.DataFrame({
            'DATE': date_range,
            'CAGE NUMBER': cage_id,
            'FEED AMOUNT (Kg)': np.clip(np.random.normal(10, 1, len(date_range)), 0.1, None)
        })
        daily_feed['CUM_FEED'] = daily_feed['FEED AMOUNT (Kg)'].cumsum()

        # Sampling
        mock_sampling = pd.DataFrame({
            'DATE': sampling_dates,
            'CAGE NUMBER': cage_id,
            'NUMBER OF FISH': summary_c2['NUMBER OF FISH'].values + np.random.randint(-50,50,len(sampling_dates)),
            'AVERAGE BODY WEIGHT (g)': summary_c2['AVERAGE BODY WEIGHT (g)'].values * np.random.normal(1,0.05,len(sampling_dates))
        })
        mock_sampling['TOTAL_WEIGHT_KG'] = mock_sampling['NUMBER OF FISH']*mock_sampling['AVERAGE BODY WEIGHT (g)']/1000

        # Merge cumulative feed
        summary = pd.merge_asof(
            mock_sampling.sort_values('DATE'),
            daily_feed[['DATE','CUM_FEED']].sort_values('DATE'),
            on='DATE'
        )
        summary['CUM_FEED'] = summary['CUM_FEED'].fillna(method='ffill').fillna(0)

        # eFCR calculations
        summary['AGGREGATED_eFCR'] = summary['CUM_FEED'] / summary['TOTAL_WEIGHT_KG']
        summary['PERIOD_WEIGHT_GAIN'] = summary['TOTAL_WEIGHT_KG'].diff().fillna(summary['TOTAL_WEIGHT_KG'])
        summary['PERIOD_FEED'] = summary['CUM_FEED'].diff().fillna(summary['CUM_FEED'])
        summary['PERIOD_eFCR'] = summary['PERIOD_FEED'] / summary['PERIOD_WEIGHT_GAIN']

        mock_summaries[cage_id] = summary

    return mock_summaries

# -------------------------------
# 5. Streamlit Interface
# -------------------------------
st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")
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
    mock_cages = create_mock_cages(summary_c2, feeding_c2, sampling_c2)
    all_cages = {2: summary_c2, **mock_cages}

    # Sidebar selectors
    st.sidebar.header("Select Options")
    selected_cage = st.sidebar.selectbox("Select Cage", list(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    df = all_cages[selected_cage].copy()

    # Display production summary table
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[['DATE','NUMBER OF FISH','TOTAL_WEIGHT_KG','AGGREGATED_eFCR','PERIOD_eFCR']].style.format({
        'TOTAL_WEIGHT_KG': '{:.2f}',
        'AGGREGATED_eFCR': '{:.2f}',
        'PERIOD_eFCR': '{:.2f}'
    }))

    # Plot graphs
    if selected_kpi == "Growth":
        df = df.dropna(subset=['TOTAL_WEIGHT_KG'])
        fig = px.line(df, x='DATE', y='TOTAL_WEIGHT_KG', markers=True,
                      title=f'Cage {selected_cage}: Growth Over Time',
                      labels={'TOTAL_WEIGHT_KG': 'Total Weight (Kg)'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = df.dropna(subset=['AGGREGATED_eFCR','PERIOD_eFCR'])
        fig = px.line(df, x='DATE', y='AGGREGATED_eFCR', markers=True, name='Agg eFCR')
        fig.add_scatter(x=df['DATE'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR')
        st.plotly_chart(fig, use_container_width=True)
