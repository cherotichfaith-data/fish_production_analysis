# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. Load Data Function
def load_data(feeding_file, harvest_file, sampling_file, transfer_file):
    """Load fish production Excel files into pandas DataFrames."""
    try:
        feeding = pd.read_excel(feeding_file)
        harvest = pd.read_excel(harvest_file)
        sampling = pd.read_excel(sampling_file)
        transfer = pd.read_excel(transfer_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None, None
    return feeding, harvest, sampling, transfer

# 2. File Paths
feeding_file = 'feeding record.xlsx'
harvest_file = 'fish harvest.xlsx'
sampling_file = 'fish sampling.xlsx'
transfer_file = 'fish transfer.xlsx'


# 3. Load Data
feeding, harvest, sampling, transfer = load_data(feeding_file, harvest_file, sampling_file, transfer_file)

# 4. Check for missing values
def check_missing(df, name):
    """Print missing values for a DataFrame."""
    if df is not None:
        print(f"Missing values in {name} DataFrame:")
        print(df.isnull().sum())
        print("\n")
    else:
        print(f"{name} DataFrame could not be loaded.\n")

check_missing(feeding, "Feeding")
check_missing(harvest, "Harvest")
check_missing(sampling, "Sampling")
check_missing(transfer, "Transfer")

"""*   The most important features: date, cage, number of fish, weights do not have missing values
*   Secondary columns like comments, feeding response, unnamed columns and weight estimates have missing values.
"""

# Inspect DataFrame Column Data Types
def show_dtypes(df, name):
    """
    Display column names and their data types in a readable format.
    """
    if df is not None:
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str)
        })
        print(f"=== {name} Data Types ===")
        print(dtypes_df)
        print("\n")
    else:
        print(f"{name} DataFrame is None. Cannot show data types.\n")

# Display Data Types for Each DataFrame
show_dtypes(feeding, "Feeding")
show_dtypes(harvest, "Harvest")
show_dtypes(sampling, "Sampling")
show_dtypes(transfer, "Transfer")

# Data Cleaning and Preparation
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df, scale_numeric=False, show_eda=True):
    """
    Cleans and prepares fish production datasets for EDA.
    Returns cleaned dataframes and a summary report.
    Optionally shows basic EDA visualizations.
    """

    # 1. Drop fully empty columns
    for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
        df.dropna(axis=1, how='all', inplace=True)

    # 2. Standardize column names
    def standardize_columns(df):
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace(r'[^0-9a-zA-Z_]', '', regex=True)
        )
        return df

    feeding_df = standardize_columns(feeding_df)
    harvest_df = standardize_columns(harvest_df)
    sampling_df = standardize_columns(sampling_df)
    transfer_df = standardize_columns(transfer_df)

    # 3. Handle Missing Values
    for col, default in [('feed_amount_kg', 0),
                         ('feeding_type', 'Unknown'),
                         ('feeding_response_very_good_good_bad', 'Unknown')]:
        if col in feeding_df.columns:
            feeding_df[col].fillna(default, inplace=True)

    for col, default in [('total_weight_kg', 0), ('abw_g', 0)]:
        if col in transfer_df.columns:
            transfer_df[col].fillna(default, inplace=True)

    # 4. Convert Data Types
    numeric_cols = ['feed_amount_kg', 'total_weight_kg', 'abw_g']
    for col in numeric_cols:
        for df in [feeding_df, transfer_df]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 5. Remove duplicates
    for df in [feeding_df, harvest_df, sampling_df, transfer_df]:
        df.drop_duplicates(inplace=True)

    # 6. Standardize Text Data
    if 'feeding_type' in feeding_df.columns:
        feeding_df['feeding_type'] = feeding_df['feeding_type'].str.strip().str.title()
    if 'feeding_response_very_good_good_bad' in feeding_df.columns:
        feeding_df['feeding_response_very_good_good_bad'] = feeding_df['feeding_response_very_good_good_bad'].str.lower().str.strip()

    # 7. Remove Outliers
    if 'feed_amount_kg' in feeding_df.columns:
        feeding_df = feeding_df[feeding_df['feed_amount_kg'] >= 0]
    if 'total_weight_kg' in transfer_df.columns and 'abw_g' in transfer_df.columns:
        transfer_df = transfer_df[(transfer_df['total_weight_kg'] >= 0) & (transfer_df['abw_g'] >= 0)]

    # 8. Encode Categorical Variables
    if 'feeding_type' in feeding_df.columns:
        feeding_df = pd.get_dummies(feeding_df, columns=['feeding_type'], drop_first=True)

    # 9. Optional scaling
    if scale_numeric:
        scaler = StandardScaler()
        for df in [feeding_df, transfer_df]:
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = scaler.fit_transform(df[[col]])

    # 10. Summary Report for EDA
    report = {
        'feeding_rows': len(feeding_df),
        'harvest_rows': len(harvest_df),
        'sampling_rows': len(sampling_df),
        'transfer_rows': len(transfer_df),
        'feeding_missing_values': feeding_df.isnull().sum().to_dict(),
        'transfer_missing_values': transfer_df.isnull().sum().to_dict(),
    }

    # 11. Basic EDA Visualizations
    if show_eda:
        for df, name in zip([feeding_df, harvest_df, sampling_df, transfer_df],
                            ['Feeding', 'Harvest', 'Sampling', 'Transfer']):
            print(f"\n===== {name} DataFrame Summary =====")
            print(df.describe(include='all'))

            numeric = df.select_dtypes(include=np.number).columns.tolist()
            if numeric:
                for col in numeric:
                    plt.figure(figsize=(6,3))
                    sns.histplot(df[col], kde=True, bins=20)
                    plt.title(f"{name} - Distribution of {col}")
                    plt.show()

                for col in numeric:
                    plt.figure(figsize=(6,3))
                    sns.boxplot(x=df[col])
                    plt.title(f"{name} - Boxplot of {col}")
                    plt.show()

    return feeding_df, harvest_df, sampling_df, transfer_df, report

"""## Preprocessing Cage2"""

def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """
    Preprocess Cage 2 for production analysis.
    - Adds manual stocking (26/08/2024, 7290 fish, ABW 11.9 g)
    - Clips data between 26/08/2024 and 09/07/2025
    - Incorporates transfers to adjust fish alive and biomass
    """

    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # Filter cage and timeframe
    feeding_c2  = feeding[(feeding['cage_number'] == cage_number) &
                          (feeding['date'] >= start_date) &
                          (feeding['date'] <= end_date)].copy()

    harvest_c2  = harvest[(harvest['cage'] == cage_number) &
                          (harvest['date'] >= start_date) &
                          (harvest['date'] <= end_date)].copy()

    sampling_c2 = sampling[(sampling['cage_number'] == cage_number) &
                           (sampling['date'] >= start_date) &
                           (sampling['date'] <= end_date)].copy()

    # Ensure numeric columns
    for col in ['number_of_fish', 'average_body_weight_g']:
        if col in sampling_c2.columns:
            sampling_c2[col] = pd.to_numeric(sampling_c2[col], errors='coerce').fillna(0)

    # Add manual stocking row
    stocking_row = pd.DataFrame([{
        'date': start_date,
        'cage_number': cage_number,
        'number_of_fish': 7290,
        'average_body_weight_g': 11.9
    }])
    sampling_c2 = pd.concat([stocking_row, sampling_c2]).sort_values('date').reset_index(drop=True)

    # Initialize transfer columns
    sampling_c2['IN_FISH'] = 0
    sampling_c2['OUT_FISH'] = 0

    if transfers is not None and not transfers.empty:
        transfers_c2 = transfers.copy()
        transfers_c2 = transfers_c2[(transfers_c2['date'] >= start_date) &
                                    (transfers_c2['date'] <= end_date)]

        # Outgoing transfers
        out_mask = transfers_c2['origin_cage'] == cage_number
        if out_mask.any():
            out_transfers = (transfers_c2[out_mask]
                             .groupby('date')['number_of_fish']
                             .sum()
                             .cumsum()
                             .reset_index()
                             .rename(columns={'number_of_fish':'OUT_FISH'}))
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'),
                                        out_transfers.sort_values('date'),
                                        on='date', direction='backward')

        # Incoming transfers
        in_mask = transfers_c2['destination_cage'] == cage_number
        if in_mask.any():
            in_transfers = (transfers_c2[in_mask]
                            .groupby('date')['number_of_fish']
                            .sum()
                            .cumsum()
                            .reset_index()
                            .rename(columns={'number_of_fish':'IN_FISH'}))
            sampling_c2 = pd.merge_asof(sampling_c2.sort_values('date'),
                                        in_transfers.sort_values('date'),
                                        on='date', direction='backward')

        # Fill NaN values after merge
        sampling_c2[['IN_FISH','OUT_FISH']] = sampling_c2[['IN_FISH','OUT_FISH']].fillna(0)

    # Compute standing fish and biomass
    sampling_c2['FISH_ALIVE'] = (sampling_c2['number_of_fish'] + sampling_c2['IN_FISH'] - sampling_c2['OUT_FISH']).clip(lower=0)
    sampling_c2['BIOMASS_KG'] = sampling_c2['FISH_ALIVE'] * sampling_c2['average_body_weight_g'] / 1000

    # Sort final output
    sampling_c2 = sampling_c2.sort_values('date').reset_index(drop=True)
    feeding_c2 = feeding_c2.sort_values('date').reset_index(drop=True)
    harvest_c2 = harvest_c2.sort_values('date').reset_index(drop=True)

    return feeding_c2, harvest_c2, sampling_c2

"""# Computing Production summary"""

def cage2_production_summary(feeding_c2, sampling_c2, harvest_c2):
    """
    Generates a production summary for Cage 2:
    - Growth (average body weight, biomass)
    - Period eFCR and aggregated eFCR
    - Period feed and cumulative feed
    """

    df = sampling_c2.copy()

    # Ensure proper sorting by date
    df = df.sort_values('date').reset_index(drop=True)
    feeding_c2 = feeding_c2.sort_values('date')

    # 1. Align feeding to sampling periods
    df['PERIOD_FEED_KG'] = 0
    for i in range(1, len(df)):
        start, end = df.loc[i-1, 'date'], df.loc[i, 'date']
        mask = (feeding_c2['date'] > start) & (feeding_c2['date'] <= end)
        df.loc[i, 'PERIOD_FEED_KG'] = feeding_c2.loc[mask, 'feed_amount_kg'].sum()

    # 2. Cumulative feed
    df['CUM_FEED_KG'] = df['PERIOD_FEED_KG'].cumsum()

    # 3. Compute biomass changes
    df['PREV_BIOMASS_KG'] = df['BIOMASS_KG'].shift(1).fillna(0)
    df['DELTA_BIOMASS_KG'] = df['BIOMASS_KG'] - df['PREV_BIOMASS_KG']

    # 4. Period eFCR = feed / biomass gained in that period
    df['PERIOD_eFCR'] = df.apply(
        lambda row: row['PERIOD_FEED_KG'] / row['DELTA_BIOMASS_KG']
        if row['DELTA_BIOMASS_KG'] > 0 else np.nan,
        axis=1
    )

    # 5. Aggregated eFCR = cumulative feed / cumulative biomass gained
    df['AGGREGATED_eFCR'] = df['CUM_FEED_KG'] / df['BIOMASS_KG'].replace(0, np.nan)

    # 6. Growth tracking
    df['AVERAGE_BODY_WEIGHT_G'] = df['average_body_weight_g']
    # FISH_ALIVE already exists in sampling_c2; ensure numeric
    df['FISH_ALIVE'] = pd.to_numeric(df['FISH_ALIVE'], errors='coerce').fillna(0)

    # 7. Keep relevant columns
    summary = df[['date','FISH_ALIVE','AVERAGE_BODY_WEIGHT_G','BIOMASS_KG',
                  'PERIOD_FEED_KG','CUM_FEED_KG','DELTA_BIOMASS_KG','PERIOD_eFCR','AGGREGATED_eFCR']]

    return summary

# Function to generate mock cages
def generate_mock_cages(feeding_c2, sampling_c2, harvest_c2, num_cages=5):
    """
    Generate mock data for additional cages based on Cage 2 with randomness.
    - Daily feeding: +/- 10% variation
    - Sampling: +/- small variation in ABW (average body weight)
    - Harvest: +/- 5% variation in total weight/fish
    """
    mock_feeding = []
    mock_sampling = []
    mock_harvest = []
    mock_summaries = {}

    for cage in range(3, 3 + num_cages):  # Cage numbers 3,4,5,6,7
        # ----- Feeding -----
        f = feeding_c2.copy()
        f['cage_number'] = cage
        if 'feed_amount_kg' in f.columns:
            f['feed_amount_kg'] = f['feed_amount_kg'] * np.random.uniform(0.9, 1.1, size=len(f))
        mock_feeding.append(f)

        # ----- Sampling -----
        s = sampling_c2.copy()
        s['cage_number'] = cage
        if 'average_body_weight_g' in s.columns:
            s['average_body_weight_g'] = s['average_body_weight_g'] * np.random.uniform(0.95, 1.05, size=len(s))
        if 'FISH_ALIVE' in s.columns:
            s['FISH_ALIVE'] = pd.to_numeric(s['FISH_ALIVE'], errors='coerce').fillna(0)
        s['BIOMASS_KG'] = s['FISH_ALIVE'] * s['average_body_weight_g'] / 1000
        mock_sampling.append(s)

        # ----- Harvest -----
        h = harvest_c2.copy()
        h['cage'] = cage
        if 'total_weight_kg' in h.columns:
            h['total_weight_kg'] = h['total_weight_kg'] * np.random.uniform(0.95, 1.05, size=len(h))
        if 'number_of_fish' in h.columns:
            h['number_of_fish'] = pd.to_numeric(h['number_of_fish'], errors='coerce').fillna(0)
        mock_harvest.append(h)

        # ----- Production summary -----
        summary = cage2_production_summary(f, s, h)
        mock_summaries[cage] = summary

    return mock_feeding, mock_sampling, mock_harvest, mock_summaries


# ===== Example workflow =====

# Load and clean the data
feeding, harvest, sampling, transfer, report = clean_and_prepare(feeding, harvest, sampling, transfer)

# Preprocess Cage 2
feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfer)

# Generate mock cages (Cages 3–7)
mock_feeding, mock_sampling, mock_harvest, mock_summaries = generate_mock_cages(feeding_c2, sampling_c2, harvest_c2)

# Example: view Cage 3 production summary
print(mock_summaries[3].head())

# Streamlit Interface for Fish Cage Production Analysis
import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Title and Sidebar
st.title("Fish Cage Production Analysis")
st.sidebar.header("Upload Excel Files (Cage 2)")

feeding_file = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

# 2. Load and preprocess
if feeding_file and harvest_file and sampling_file:
    # Load data
    if transfer_file:
        transfer_df = pd.read_excel(transfer_file)
    else:
        transfer_df = pd.DataFrame()  # empty if not uploaded

    feeding, harvest, sampling, transfer, report = clean_and_prepare(
        pd.read_excel(feeding_file),
        pd.read_excel(harvest_file),
        pd.read_excel(sampling_file),
        transfer_df,
        scale_numeric=False,
        show_eda=False
    )

    # Preprocess Cage 2
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfer)

    # Cage 2 summary
    summary_c2 = cage2_production_summary(feeding_c2, sampling_c2, harvest_c2)

    # Generate mock cages (3–7)
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = generate_mock_cages(
        feeding_c2, sampling_c2, harvest_c2
    )

    # Combine Cage 2 + mock cages for selection
    all_cages = {2: summary_c2}
    all_cages.update({cage: mock_summaries[cage] for cage in mock_summaries})

    # 3. Sidebar selectors
    st.sidebar.header("Options")
    selected_cage = st.sidebar.selectbox("Select Cage", sorted(all_cages.keys()))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])


    # 4. Display table
    df = all_cages[selected_cage].copy()
    st.subheader(f"Cage {selected_cage} Production Summary")
    st.dataframe(df[['date','FISH_ALIVE','BIOMASS_KG','PERIOD_FEED_KG','CUM_FEED_KG','PERIOD_eFCR','AGGREGATED_eFCR']].round(2))

    # 5. Plot KPI graph
    if selected_kpi == "Growth":
        fig = px.line(df, x='date', y='BIOMASS_KG', markers=True,
                      title=f'Cage {selected_cage}: Biomass Growth Over Time',
                      labels={'BIOMASS_KG':'Biomass (Kg)', 'date':'Date'})
        st.plotly_chart(fig)
    else:  # eFCR
        fig = px.line(df, x='date', y='AGGREGATED_eFCR', markers=True, name='Aggregated eFCR')
        fig.add_scatter(x=df['date'], y=df['PERIOD_eFCR'], mode='lines+markers', name='Period eFCR')
        fig.update_layout(title=f'Cage {selected_cage}: eFCR Over Time', yaxis_title='eFCR', xaxis_title='Date')
        st.plotly_chart(fig)

else:
    st.info("Please upload all required Excel files to start the analysis.")
