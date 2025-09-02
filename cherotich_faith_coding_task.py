# ==============================
# Fish Cage Production Analysis Dashboard
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Dashboard",
                   layout="wide", page_icon="🐟")

# ==============================
# Column Normalization utilities
# ==============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    def _coerce(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
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

# ==============================
# 1. Load and clean data
# ==============================
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    if transfers is not None:
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        wcol = find_col(transfers, 
                        ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]","WEIGHT (KG)"], 
                        fuzzy_hint="WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

# ==============================
# Compute metrics
# ==============================
def compute_metrics(stocking_date, end_date, initial_stock, sampling_data, feeding_data, transfer_data, harvest_data):
    dates = pd.date_range(stocking_date, end_date, freq="D")
    df = pd.DataFrame({"DATE": dates})
    df["ABW_G"] = np.nan
    df["FISH_ALIVE"] = initial_stock
    df["BIOMASS_KG"] = np.nan
    df["FEED_KG"] = 0.0
    df["HARV_KG"] = 0.0
    df["IN_KG"] = 0.0
    df["OUT_KG"] = 0.0
    df["GROWTH_KG"] = np.nan
    df["PERIOD_eFCR"] = np.nan
    df["AGGREGATED_eFCR"] = np.nan

    # Feeding
    if feeding_data is not None and not feeding_data.empty:
        feed_col = find_col(feeding_data, ["FEED (KG)", "FEED_KG", "FEED"], "FEED")
        for _, row in feeding_data.iterrows():
            if pd.notna(row["DATE"]):
                df.loc[df["DATE"] == row["DATE"], "FEED_KG"] += row.get(feed_col, 0)

    # Transfers
    if transfer_data is not None and not transfer_data.empty:
        for _, row in transfer_data.iterrows():
            d = row["DATE"]
            fish = row.get("NUMBER OF FISH", np.nan)
            kg   = row.get("TOTAL WEIGHT [KG]", np.nan)

            if "DESTINATION CAGE" in row and row["DESTINATION CAGE"] == 2:
                df.loc[df["DATE"] == d, "IN_KG"] += kg if pd.notna(kg) else 0
                df.loc[df["DATE"] >= d, "FISH_ALIVE"] += fish if pd.notna(fish) else 0

            if "ORIGIN CAGE" in row and row["ORIGIN CAGE"] == 2:
                df.loc[df["DATE"] == d, "OUT_KG"] += kg if pd.notna(kg) else 0
                df.loc[df["DATE"] >= d, "FISH_ALIVE"] -= fish if pd.notna(fish) else 0

    # Harvest
    if harvest_data is not None and not harvest_data.empty:
        for _, row in harvest_data.iterrows():
            d = row["DATE"]
            fish = row.get("NUMBER OF FISH", np.nan)
            kg   = row.get("TOTAL WEIGHT [KG]", np.nan)
            df.loc[df["DATE"] == d, "HARV_KG"] += kg if pd.notna(kg) else 0
            if pd.notna(fish):
                df.loc[df["DATE"] >= d, "FISH_ALIVE"] -= fish

    # Sampling → ABW
    if sampling_data is not None and not sampling_data.empty:
        for _, row in sampling_data.iterrows():
            df.loc[df["DATE"] == row["DATE"], "ABW_G"] = row.get("ABW_G", np.nan)

    df["ABW_G"] = df["ABW_G"].ffill()
    df["FISH_ALIVE"] = df["FISH_ALIVE"].ffill()

    # Biomass
    df["BIOMASS_KG"] = df["FISH_ALIVE"] * df["ABW_G"] / 1000

    # Growth & eFCR
    if sampling_data is not None and not sampling_data.empty:
        sampling_dates = sampling_data["DATE"].sort_values().tolist()
        growth_cum = 0.0
        feed_cum = 0.0

        for i in range(1, len(sampling_dates)):
            d0, d1 = sampling_dates[i-1], sampling_dates[i]
            biomass_gain = df.loc[df["DATE"] == d1, "BIOMASS_KG"].values[0] - df.loc[df["DATE"] == d0, "BIOMASS_KG"].values[0]
            feed_used = df.loc[(df["DATE"] > d0) & (df["DATE"] <= d1), "FEED_KG"].sum()
            in_kg  = df.loc[(df["DATE"] > d0) & (df["DATE"] <= d1), "IN_KG"].sum()
            out_kg = df.loc[(df["DATE"] > d0) & (df["DATE"] <= d1), "OUT_KG"].sum()
            harv_kg = df.loc[(df["DATE"] > d0) & (df["DATE"] <= d1), "HARV_KG"].sum()

            growth = biomass_gain + harv_kg + out_kg - in_kg
            idx = df.index[df["DATE"] == d1][0]
            df.loc[idx, "GROWTH_KG"] = growth

            if growth > 0:
                df.loc[idx, "PERIOD_eFCR"] = feed_used / growth
                feed_cum += feed_used
                growth_cum += growth
                df.loc[idx, "AGGREGATED_eFCR"] = feed_cum / growth_cum

    return df

# Mock cages
def generate_mock_cages(feeding_c2, sampling_c2, harvest_c2, num_cages=5, start_id=3):
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = [], [], [], {}

    # Ensure ABW_G exists (auto-generate if missing)
    if "ABW_G" not in sampling_c2.columns:
        sampling_c2 = sampling_c2.copy()
        sampling_c2["ABW_G"] = np.linspace(50, 800, len(sampling_c2))

    if "FISH_ALIVE" not in sampling_c2.columns:
        sampling_c2["FISH_ALIVE"] = np.linspace(5000, 1000, len(sampling_c2)).astype(int)

    for cage in range(start_id, start_id + num_cages):
        # Feeding
        f = feeding_c2.copy()
        f["CAGE NUMBER"] = cage
        feed_col = find_col(f, ["FEED_KG", "FEED (KG)", "FEED"], "FEED_KG")
        if not feed_col:
            f["FEED_KG"] = 0
        else:
            f["FEED_KG"] = pd.to_numeric(f[feed_col], errors="coerce").fillna(0)
            f["FEED_KG"] *= np.random.uniform(0.9, 1.1, size=len(f))
        mock_feeding.append(f)

        # Sampling
        s = sampling_c2.copy()
        s["CAGE NUMBER"] = cage
        s["ABW_G"] *= np.random.uniform(0.95, 1.05, size=len(s))
        s["FISH_ALIVE"] = pd.to_numeric(s["FISH_ALIVE"], errors="coerce").ffill().fillna(0)
        s["BIOMASS_KG"] = s["FISH_ALIVE"] * s["ABW_G"] / 1000
        mock_sampling.append(s)

        # Harvest
        h = harvest_c2.copy()
        h["CAGE NUMBER"] = cage
        fish_col = find_col(h, ["NUMBER OF FISH", "N_FISH"], "FISH")
        kg_col   = find_col(h, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]"], "WEIGHT")
        if kg_col:
            h[kg_col] = pd.to_numeric(h[kg_col], errors="coerce").fillna(0)
            h[kg_col] *= np.random.uniform(0.95, 1.05, size=len(h))
        if fish_col:
            h[fish_col] = pd.to_numeric(h[fish_col], errors="coerce").ffill().fillna(0)
        mock_harvest.append(h)

        # Ensure feeding + harvest DataFrames have required columns
        feeding_safe = f[["DATE", "FEED_KG"]] if "DATE" in f and "FEED_KG" in f else pd.DataFrame(columns=["DATE", "FEED_KG"])
        harvest_safe = h[["DATE", fish_col, kg_col]] if (fish_col and kg_col) else h[["DATE"]] if "DATE" in h else pd.DataFrame(columns=["DATE"])

        # Summary
        summary = compute_metrics(
            stocking_date=s["DATE"].min(),
            end_date=s["DATE"].max(),
            initial_stock=s["FISH_ALIVE"].iloc[0],
            sampling_data=s[["DATE", "ABW_G"]],
            feeding_data=feeding_safe,
            transfer_data=None,
            harvest_data=harvest_safe
        )
        mock_summaries[cage] = summary

    return mock_feeding, mock_sampling, mock_harvest, mock_summaries


# ==============================
# Streamlit UI
# ==============================
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Fish Cage Production Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar uploads
st.sidebar.header("Upload Excel Files (Cage 2 only)")
feeding_file  = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

if feeding_file and harvest_file and sampling_file:
    feeding, harvest, sampling, transfers = load_data(
        feeding_file, harvest_file, sampling_file, transfer_file
    )

    # Cage 2 baseline
    feeding_c2, harvest_c2, sampling_c2 = feeding, harvest, sampling
    summary_c2 = compute_metrics(
        stocking_date=sampling_c2["DATE"].min(),
        end_date=sampling_c2["DATE"].max(),
        initial_stock=7290,
        sampling_data=sampling_c2[["DATE", "ABW_G"]] if "ABW_G" in sampling_c2 else pd.DataFrame(),
        feeding_data=feeding_c2,
        transfer_data=transfers,
        harvest_data=harvest_c2
    )

    # Generate mock cages
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = generate_mock_cages(
        feeding_c2.copy(), sampling_c2.copy(), harvest_c2.copy()
    )

    # Combine Cage 2 with mock cages
    all_summaries = {2: summary_c2, **mock_summaries}

    # Sidebar selection
    selected_cage = st.sidebar.selectbox("Select Cage", sorted(all_summaries.keys()))
    selected_kpi  = st.sidebar.selectbox("Select KPI", ["Biomass", "ABW", "eFCR"])
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

    # Plot
    if selected_kpi == "Biomass":
        fig = px.line(summary_df.dropna(subset=["BIOMASS_KG"]), x="DATE", y="BIOMASS_KG", markers=True,
                      title=f"Cage {selected_cage}: Biomass Over Time", labels={"BIOMASS_KG": "Biomass (kg)"})
    elif selected_kpi == "ABW":
        fig = px.line(summary_df.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                      title=f"Cage {selected_cage}: ABW Over Time", labels={"ABW_G": "ABW (g)"})
    else:
        dff = summary_df.dropna(subset=["AGGREGATED_eFCR", "PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
                      title=f"Cage {selected_cage}: eFCR Over Time", labels={"AGGREGATED_eFCR": "Aggregated eFCR"})
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬅️ Upload the Excel files to begin.")
