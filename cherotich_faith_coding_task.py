# main.py
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

# ---------------------------
# 1. Load Excel data
# ---------------------------
feeding_df = pd.read_excel(r"C:\Users\user\Downloads\feeding record.xlsx")
harvest_df = pd.read_excel(r"C:\Users\user\Downloads\fish harvest.xlsx")
transfer_df = pd.read_excel(r"C:\Users\user\Downloads\feeding record.xlsx")

# ---------------------------
# 2. Filter Cage 2 and timeframe
# ---------------------------
start_date = pd.to_datetime("2024-07-16")
end_date = pd.to_datetime("2025-06-30")
cage_number = 2

feeding_df['DATE'] = pd.to_datetime(feeding_df['DATE'], errors='coerce')
harvest_df['DATE'] = pd.to_datetime(harvest_df['DATE'], errors='coerce')
transfer_df['DATE'] = pd.to_datetime(transfer_df['DATE'], errors='coerce')

feeding_df_filtered = feeding_df[(feeding_df['CAGE NUMBER'] == f'C{cage_number}') &
                                 (feeding_df['DATE'] >= start_date) &
                                 (feeding_df['DATE'] <= end_date)].copy()

harvest_df_filtered = harvest_df[(harvest_df['CAGE'] == f'CAGE {cage_number}') &
                                 (harvest_df['DATE'] >= start_date) &
                                 (harvest_df['DATE'] <= end_date)].copy()

transfer_df_filtered = transfer_df[((transfer_df['ORIGIN CAGE'] == f'CAGE {cage_number}') |
                                     (transfer_df['DESTINATION CAGE'] == f'CAGE {cage_number}')) &
                                    (transfer_df['DATE'] >= start_date) &
                                    (transfer_df['DATE'] <= end_date)].copy()
# Filter cage 2
feeding_c2 = feeding_df[(feeding_df['CAGE NUMBER'] == cage_number) & 
                        (feeding_df['DATE'] >= start_date) & 
                        (feeding_df['DATE'] <= end_date)].copy()

harvest_c2 = harvest_df[(harvest_df['CAGE'] == cage_number) & 
                        (harvest_df['DATE'] >= start_date) & 
                        (harvest_df['DATE'] <= end_date)].copy()

transfer_c2 = transfer_df[(transfer_df['ORIGIN CAGE'] == cage_number) | 
                          (transfer_df['DESTINATION CAGE'] == cage_number)]
# ---------------------------
# 3. Add stocking manually
# ---------------------------
stocking = pd.DataFrame({
    'DATE': [start_date],
    'NUMBER_OF_FISH': [7902],
    'ABW': [0.7]  # g
})

# ---------------------------
# 4. Mock sampling data for Cage 2
# ---------------------------
sampling_dates = pd.date_range(start=start_date + pd.Timedelta(days=20), end=end_date, freq='20D')
np.random.seed(42)
sampling_c2 = pd.DataFrame({
    'DATE': sampling_dates,
    'CAGE NUMBER': cage_number,
    'NUMBER OF FISH': np.linspace(7900, harvest_c2['NUMBER OF_FISH'].sum() if not harvest_c2.empty else 7800, len(sampling_dates)).astype(int),
    'AVERAGE BODY WEIGHT (g)': np.linspace(0.7, harvest_c2['ABW'].mean() if not harvest_c2.empty else 500, len(sampling_dates)) + np.random.normal(0, 5, len(sampling_dates))
})
# Include stocking
sampling_c2 = pd.concat([stocking.rename(columns={'NUMBER_OF_FISH':'NUMBER OF FISH', 'ABW':'AVERAGE BODY WEIGHT (g)'}), sampling_c2], ignore_index=True)

# ---------------------------
# 5. Simulate 5 additional cages
# ---------------------------
cages = {}
for i in range(3, 8):
    df = sampling_c2.copy()
    df['CAGE NUMBER'] = i
    df['AVERAGE BODY WEIGHT (g)'] *= np.random.uniform(0.95, 1.05, len(df))
    df['NUMBER OF FISH'] = (df['NUMBER OF FISH'] * np.random.uniform(0.98, 1.02, len(df))).astype(int)
    cages[i] = df
cages[cage_number] = sampling_c2

# ---------------------------
# 6. Compute FCR
# ---------------------------
def compute_fcr(sampling_df, feeding_df):
    sampling_df = sampling_df.sort_values('DATE').reset_index(drop=True)
    period_fcr = []
    agg_fcr = []
    total_feed = 0
    start_weight = sampling_df.loc[0, 'NUMBER OF FISH'] * sampling_df.loc[0, 'AVERAGE BODY WEIGHT (g)'] / 1000
    for i in range(1, len(sampling_df)):
        period_feed = feeding_df[(feeding_df['DATE'] > sampling_df.loc[i-1, 'DATE']) &
                                 (feeding_df['DATE'] <= sampling_df.loc[i, 'DATE'])]['FEED AMOUNT (Kg)'].sum()
        total_feed += period_feed
        prev_total_weight = sampling_df.loc[i-1, 'NUMBER OF FISH'] * sampling_df.loc[i-1, 'AVERAGE BODY WEIGHT (g)'] / 1000
        curr_total_weight = sampling_df.loc[i, 'NUMBER OF FISH'] * sampling_df.loc[i, 'AVERAGE BODY WEIGHT (g)'] / 1000
        weight_gain = curr_total_weight - prev_total_weight
        period_fcr.append(period_feed / weight_gain if weight_gain>0 else np.nan)
        agg_fcr.append(total_feed / (curr_total_weight - start_weight) if (curr_total_weight - start_weight)>0 else np.nan)
    sampling_df = sampling_df.iloc[1:].copy()
    sampling_df['Period FCR'] = period_fcr
    sampling_df['Aggregated FCR'] = agg_fcr
    return sampling_df

summary_tables = {}
for cage_id, df in cages.items():
    feeding = feeding_df[feeding_df['CAGE NUMBER'] == cage_id]
    summary_tables[cage_id] = compute_fcr(df, feeding)

# ---------------------------
# 7. Streamlit App
# ---------------------------
st.title("🐟 Fish Cage Production Analysis")

cage_options = list(summary_tables.keys())
selected_cage = st.selectbox("Select Cage", cage_options)
kpi = st.selectbox("Select KPI", ['Growth', 'FCR'])

df = summary_tables[selected_cage]

# Plot
fig, ax = plt.subplots()
if kpi == 'Growth':
    ax.plot(df['DATE'], df['AVERAGE BODY WEIGHT (g)'], marker='o')
    ax.set_ylabel("Average Body Weight (g)")
    ax.set_title(f"Cage {selected_cage} Growth Over Time")
else:
    ax.plot(df['DATE'], df['Aggregated FCR'], marker='o', label='Aggregated FCR')
    ax.plot(df['DATE'], df['Period FCR'], marker='x', label='Period FCR')
    ax.set_ylabel("FCR")
    ax.set_title(f"Cage {selected_cage} Feed Conversion Ratio")
    ax.legend()
ax.set_xlabel("Date")
plt.xticks(rotation=45)
st.pyplot(fig)

# Show table
st.subheader(f"Cage {selected_cage} Production Summary Table")
st.dataframe(df)

# ---------------------------
# 8. Download buttons
# ---------------------------
# Excel download
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name=f"Cage_{selected_cage}")
st.download_button(
    label="📥 Download Summary as Excel",
    data=excel_buffer.getvalue(),
    file_name=f"Cage_{selected_cage}_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Plot download
img_buffer = BytesIO()
fig.savefig(img_buffer, format='png', bbox_inches='tight')
st.download_button(
    label="📥 Download Graph as PNG",
    data=img_buffer.getvalue(),
    file_name=f"Cage_{selected_cage}_{kpi}.png",
    mime="image/png"
)
