# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide")

#Column Normalization; to handle messy Excel files, column variants, and cage numbers
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
    """Enhanced column finder with fuzzy matching"""
    lut = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    if fuzzy_hint:
        for U, orig in lut.items():
            if fuzzy_hint.upper() in U:
                return orig
    return None

# 1. Load data
def load_data(feeding_file, harvest_file, sampling_file, transfer_file=None):
    feeding  = normalize_columns(pd.read_excel(feeding_file))
    harvest  = normalize_columns(pd.read_excel(harvest_file))
    sampling = normalize_columns(pd.read_excel(sampling_file))
    transfers = normalize_columns(pd.read_excel(transfer_file)) if transfer_file else None

    # Coerce cage columns
    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    # Transfers
    if transfers is not None:
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # Standardize weight column using enhanced find_col
        wcol = find_col(transfers, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]","WEIGHT (KG)"], fuzzy_hint="WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Parse dates
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers


# 2. Preprocess Cage 2 
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """
    Preprocess Cage 2 timeline (or any cage) with consistent columns:
    ABW_G, FISH_ALIVE, STOCKED, HARV_FISH_CUM, HARV_KG_CUM, IN/OUT cumulatives.
    Ensures timeline starts with actual stocking (fallback 7290 fish @ 11.9g)
    and includes final harvest date.
    """
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # ----------------------
    # Helper: clip dates and sort
    # ----------------------
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding.get("CAGE NUMBER", -1) == cage_number] if "CAGE NUMBER" in feeding.columns else feeding)
    harvest_c2  = _clip(harvest[harvest.get("CAGE NUMBER", -1) == cage_number] if "CAGE NUMBER" in harvest.columns else harvest)
    sampling_c2 = _clip(sampling[sampling.get("CAGE NUMBER", -1) == cage_number] if "CAGE NUMBER" in sampling.columns else sampling)

    # ----------------------
    # Standardize ABW column
    # ----------------------
    abw_candidates = ["AVERAGE_BODY_WEIGHT(G)", "ABW(G)", "ABW_G", "ABW"]
    abw_set = False
    for col in abw_candidates:
        if col in sampling_c2.columns:
            sampling_c2["ABW_G"] = pd.to_numeric(sampling_c2[col].map(to_number), errors="coerce")
            abw_set = True
            break
    if not abw_set:
        sampling_c2["ABW_G"] = np.nan

    # ----------------------
    # Stocking info (from transfers if available)
    # ----------------------
    stocked_fish = 7290
    initial_abw_g = 11.9
    first_inbound_idx = None
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        t_in = t[t.get("DESTINATION CAGE", t.get("DEST_CAGE", -1)) == cage_number].sort_values("DATE")
        if not t_in.empty:
            first = t_in.iloc[0]
            first_inbound_idx = first.name
            if "NUMBER OF FISH" in t_in.columns and pd.notna(first.get("NUMBER OF FISH")):
                stocked_fish = int(float(first["NUMBER OF FISH"]))
            if "TOTAL WEIGHT [KG]" in t_in.columns and pd.notna(first.get("TOTAL WEIGHT [KG]")):
                initial_abw_g = float(first["TOTAL WEIGHT [KG]"]) * 1000 / stocked_fish

    # ----------------------
    # Create stocking row
    # ----------------------
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "ABW_G": initial_abw_g,
        "STOCKED": stocked_fish
    }])

    # ----------------------
    # Combine sampling + stocking
    # ----------------------
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # ----------------------
    # Ensure end date row exists
    # ----------------------
    if base["DATE"].max() < end_date:
        last_abw = base["ABW_G"].dropna().iloc[-1] if base["ABW_G"].notna().any() else initial_abw_g
        base = pd.concat([base, pd.DataFrame([{
            "DATE": end_date,
            "CAGE NUMBER": cage_number,
            "ABW_G": last_abw,
            "STOCKED": stocked_fish
        }])], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # ----------------------
    # Initialize cumulative columns
    # ----------------------
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # ----------------------
    # Harvest cumulatives
    # ----------------------
    if not harvest_c2.empty:
        h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH"], "FISH")
        h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)"], "WEIGHT")
        h = harvest_c2.copy()
        h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
        h["H_KG"]   = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0) if h_kg_col else 0
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"]   = h["H_KG"].cumsum()
        mh = pd.merge_asof(base[["DATE"]], h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]], on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # ----------------------
    # Transfers cumulatives
    # ----------------------
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if first_inbound_idx is not None and first_inbound_idx in t.index:
            t = t.drop(index=first_inbound_idx)  # remove initial stocking transfer

        # Convert cage numbers to int
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in t.columns:
                t[col] = t[col].apply(lambda x: int(str(x).strip()) if pd.notna(x) and str(x).strip().isdigit() else np.nan)

        # Outgoing
        tout = t[t.get("ORIGIN CAGE",-1) == cage_number].sort_values("DATE").copy()
        if not tout.empty:
            fish_col = find_col(tout, ["NUMBER OF FISH","N_FISH"], "FISH")
            kg_col   = find_col(tout, ["TOTAL WEIGHT [KG]","WEIGHT"], "WEIGHT")
            tout["OUT_FISH_CUM"] = pd.to_numeric(tout[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tout["OUT_KG_CUM"]   = pd.to_numeric(tout[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mo = pd.merge_asof(base[["DATE"]], tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]], on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

        # Incoming
        tin = t[t.get("DESTINATION CAGE",-1) == cage_number].sort_values("DATE").copy()
        if not tin.empty:
            fish_col = find_col(tin, ["NUMBER OF FISH","N_FISH"], "FISH")
            kg_col   = find_col(tin, ["TOTAL WEIGHT [KG]","WEIGHT"], "WEIGHT")
            tin["IN_FISH_CUM"] = pd.to_numeric(tin[fish_col], errors="coerce").fillna(0).cumsum() if fish_col else 0
            tin["IN_KG_CUM"]   = pd.to_numeric(tin[kg_col], errors="coerce").fillna(0).cumsum() if kg_col else 0
            mi = pd.merge_asof(base[["DATE"]], tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]], on="DATE", direction="backward")
            base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # ----------------------
    # Standing fish alive
    # ----------------------
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base


#Compute summary
def compute_metrics(stocking_date, end_date, initial_stock, sampling_data, feeding_data, transfer_data, harvest_data):
    # ----------------------
    # Prepare full date range
    # ----------------------
    dates = pd.date_range(stocking_date, end_date, freq="D")
    df = pd.DataFrame({"Date": dates})
    df["ABW"] = np.nan
    df["Stock"] = initial_stock
    df["Biomass"] = np.nan
    df["Feed"] = 0.0
    df["Harvest"] = 0.0
    df["Transfer_In"] = 0.0
    df["Transfer_Out"] = 0.0
    df["GROWTH_KG"] = np.nan
    df["PERIOD_eFCR"] = np.nan
    df["AGGREGATED_eFCR"] = np.nan

    # ----------------------
    # Populate feed
    # ----------------------
    if feeding_data is not None and not feeding_data.empty:
        for _, row in feeding_data.iterrows():
            df.loc[df["Date"] == row["Date"], "Feed"] += row.get("Feed", 0)

    # ----------------------
    # Populate transfers
    # ----------------------
    if transfer_data is not None and not transfer_data.empty:
        for _, row in transfer_data.iterrows():
            d = row["Date"]
            change = row.get("Change", 0)
            in_out = row.get("Type", "IN")  # assume "IN" or "OUT"
            if in_out.upper() == "IN":
                df.loc[df["Date"] >= d, "Stock"] += change
                df.loc[df["Date"] == d, "Transfer_In"] += change
            else:
                df.loc[df["Date"] >= d, "Stock"] -= change
                df.loc[df["Date"] == d, "Transfer_Out"] += change

    # ----------------------
    # Populate harvests
    # ----------------------
    if harvest_data is not None and not harvest_data.empty:
        for _, row in harvest_data.iterrows():
            d = row["Date"]
            h = row.get("Harvest", 0)
            df.loc[df["Date"] == d, "Harvest"] += h
            df.loc[df["Date"] >= d, "Stock"] -= h

    # ----------------------
    # Update ABW from sampling
    # ----------------------
    if sampling_data is not None and not sampling_data.empty:
        for _, row in sampling_data.iterrows():
            df.loc[df["Date"] == row["Date"], "ABW"] = row.get("ABW", np.nan)

    # ----------------------
    # Forward-fill ABW and Stock
    # ----------------------
    df["ABW"] = df["ABW"].ffill()
    df["Stock"] = df["Stock"].ffill()

    # ----------------------
    # Compute Biomass
    # ----------------------
    df["Biomass"] = df["Stock"] * df["ABW"] / 1000  # kg

    # ----------------------
    # Compute period growth and eFCR (transfer-aware)
    # ----------------------
    if sampling_data is not None and not sampling_data.empty:
        sampling_dates = sampling_data["Date"].sort_values().tolist()
        growth_cum = 0.0
        df["GROWTH_KG"] = np.nan
        df["PERIOD_eFCR"] = np.nan
        df["AGGREGATED_eFCR"] = np.nan

        for i in range(1, len(sampling_dates)):
            d0, d1 = sampling_dates[i-1], sampling_dates[i]

            biomass_gain = df.loc[df["Date"] == d1, "Biomass"].values[0] - df.loc[df["Date"] == d0, "Biomass"].values[0]
            feed_used = df.loc[(df["Date"] > d0) & (df["Date"] <= d1), "Feed"].sum()
            transfer_in_kg  = df.loc[(df["Date"] > d0) & (df["Date"] <= d1), "Transfer_In"].sum()
            transfer_out_kg = df.loc[(df["Date"] > d0) & (df["Date"] <= d1), "Transfer_Out"].sum()
            harvest_kg      = df.loc[(df["Date"] > d0) & (df["Date"] <= d1), "Harvest"].sum()

            growth = biomass_gain + harvest_kg + transfer_out_kg - transfer_in_kg
            idx = df.index[df["Date"] == d1][0]
            df.loc[idx, "GROWTH_KG"] = growth

            if growth > 0:
                df.loc[idx, "PERIOD_eFCR"] = feed_used / growth
                growth_cum += growth
                df.loc[idx, "AGGREGATED_eFCR"] = df.loc[idx, "Feed"].cumsum() / growth_cum

    return df

def generate_mock_cages(feeding_c2, sampling_c2, harvest_c2, num_cages=5, start_id=3):
    """
    Generate mock cages based on Cage 2 data.
    Adds random variations to simulate different cages.
    Returns feeding, sampling, harvest lists and summaries dictionary.
    """
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = [], [], [], {}

    if "ABW_G" not in sampling_c2.columns:
        raise KeyError("Sampling data must have 'ABW_G' column from preprocess_cage2!")

    for cage in range(start_id, start_id + num_cages):
        # ----- Feeding -----
        f = feeding_c2.copy()
        f["CAGE NUMBER"] = cage
        feed_col = find_col(f, ["FEED_PERIOD_KG","FEED AMOUNT (KG)","FEED KG","FEED"], "FEED")
        if feed_col:
            f[feed_col] *= np.random.uniform(0.9, 1.1, size=len(f))
        mock_feeding.append(f)

        # ----- Sampling -----
        s = sampling_c2.copy()
        s["CAGE NUMBER"] = cage
        s["ABW_G"] *= np.random.uniform(0.95, 1.05, size=len(s))
        if "FISH_ALIVE" in s.columns:
            s["FISH_ALIVE"] = pd.to_numeric(s["FISH_ALIVE"], errors="coerce").ffill().fillna(0)
        s["BIOMASS_KG"] = s["FISH_ALIVE"] * s["ABW_G"] / 1000
        mock_sampling.append(s)

        # ----- Harvest -----
        h = harvest_c2.copy()
        h["CAGE NUMBER"] = cage
        fish_col = find_col(h, ["NUMBER OF FISH","N_FISH"], "FISH")
        kg_col   = find_col(h, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]"], "WEIGHT")
        if kg_col:
            h[kg_col] *= np.random.uniform(0.95, 1.05, size=len(h))
        if fish_col:
            h[fish_col] = pd.to_numeric(h[fish_col], errors="coerce").ffill().fillna(0)
        mock_harvest.append(h)

        # ----- Summary -----
        summary = compute_summary(f, s)   # or replace with compute_metrics pipeline
        mock_summaries[cage] = summary

    return mock_feeding, mock_sampling, mock_harvest, mock_summaries

# ===========================
# Streamlit UI – Cage Selection + KPI (Styled)
# ===========================
import streamlit as st
import pandas as pd
import plotly.express as px

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="Fish Cage Production Dashboard",
    layout="wide",
    page_icon="🐟"
)

# ===========================
# Custom CSS Styling
# ===========================
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
    .dataframe th {
        background-color: #1E90FF;
        color: white;
    }
    .dataframe td {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# Main Title
# ===========================
st.markdown('<div class="main-title">Fish Cage Production Analysis Dashboard</div>', unsafe_allow_html=True)

# ===========================
# Sidebar Uploads
# ===========================
st.sidebar.header("Upload Excel Files (Cage 2 only)")

feeding_file  = st.sidebar.file_uploader("Feeding Record", type=["xlsx"])
harvest_file  = st.sidebar.file_uploader("Fish Harvest", type=["xlsx"])
sampling_file = st.sidebar.file_uploader("Fish Sampling", type=["xlsx"])
transfer_file = st.sidebar.file_uploader("Fish Transfer (optional)", type=["xlsx"])

# ===========================
# Main Logic
# ===========================
if feeding_file and harvest_file and sampling_file:
    # Load and preprocess Cage 2
    feeding, harvest, sampling, transfers = load_data(
        feeding_file, harvest_file, sampling_file, transfer_file
    )
    feeding_c2, harvest_c2, sampling_c2 = preprocess_cage2(feeding, harvest, sampling, transfers)
    summary_c2 = compute_summary(feeding_c2, sampling_c2)  # uses corrected logic

    # Generate mock cages (3–7)
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = generate_mock_cages(
        feeding_c2, sampling_c2, harvest_c2
    )

    # Combine all summaries into dictionary
    all_summaries = {2: summary_c2, **mock_summaries}

    # Sidebar: Cage & KPI selection
    selected_cage = st.sidebar.selectbox("Select Cage", sorted(all_summaries.keys()))
    selected_kpi  = st.sidebar.selectbox("Select KPI", ["Biomass", "ABW", "eFCR"])

    summary_df = all_summaries[selected_cage]

    # ===========================
    # KPI Summary Cards
    # ===========================
    st.subheader(f"Cage {selected_cage} – Production Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_biomass = summary_df["BIOMASS_KG"].sum()
        st.markdown(
            f'<div class="kpi-card"><h3>Total Biomass</h3><p>{total_biomass:,.2f} kg</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        avg_abw = summary_df["ABW_G"].mean()
        st.markdown(
            f'<div class="kpi-card"><h3>Average ABW</h3><p>{avg_abw:,.2f} g</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        avg_efcr = summary_df["AGGREGATED_eFCR"].mean()
        st.markdown(
            f'<div class="kpi-card"><h3>Average eFCR</h3><p>{avg_efcr:.2f}</p></div>',
            unsafe_allow_html=True,
        )

    # ===========================
    # Data Table
    # ===========================
    show_cols = [
        "DATE", "NUMBER OF FISH", "ABW_G", "BIOMASS_KG",
        "FEED_PERIOD_KG", "FEED_AGG_KG", "GROWTH_KG",
        "TRANSFER_IN_KG", "TRANSFER_OUT_KG", "HARVEST_KG",
        "TRANSFER_IN_FISH", "TRANSFER_OUT_FISH", "HARVEST_FISH",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR", "AGGREGATED_eFCR",
    ]
    display_summary = summary_df[[c for c in show_cols if c in summary_df.columns]]
    st.dataframe(display_summary, use_container_width=True)

    # Period info
    st.write(f"**Records:** {len(display_summary)} rows")
    st.write(f"**From:** {display_summary['DATE'].min().strftime('%d %b %Y')} "
             f"**To:** {display_summary['DATE'].max().strftime('%d %b %Y')}")

    # ===========================
    # KPI Plots
    # ===========================
    if selected_kpi == "Biomass":
        fig = px.line(
            summary_df.dropna(subset=["BIOMASS_KG"]),
            x="DATE", y="BIOMASS_KG", markers=True,
            title=f"Cage {selected_cage}: Biomass Over Time",
            labels={"BIOMASS_KG": "Total Biomass (kg)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_kpi == "ABW":
        fig = px.line(
            summary_df.dropna(subset=["ABW_G"]),
            x="DATE", y="ABW_G", markers=True,
            title=f"Cage {selected_cage}: Average Body Weight (ABW) Over Time",
            labels={"ABW_G": "ABW (g)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # eFCR
        dff = summary_df.dropna(subset=["AGGREGATED_eFCR", "PERIOD_eFCR"])
        fig = px.line(
            dff, x="DATE", y="AGGREGATED_eFCR", markers=True,
            title=f"Cage {selected_cage}: eFCR Over Time",
            labels={"AGGREGATED_eFCR": "Aggregated eFCR"}
        )
        fig.update_traces(name="Aggregated eFCR")
        fig.add_scatter(
            x=dff["DATE"], y=dff["PERIOD_eFCR"],
            mode="lines+markers", name="Period eFCR",
            line=dict(dash="dash")
        )
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬅️ Upload the Excel files to begin.")

