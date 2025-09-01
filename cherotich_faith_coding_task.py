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
        # Standardize weight column
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
    Preprocess Cage 2 data robustly:
    - Aligns dates to sampling
    - Adds initial stocking (2024-08-26, 7290 fish @ 11.9 g)
    - Computes harvest and transfer cumulatives
    - Calculates standing fish per day
    """
    import pandas as pd
    import numpy as np
    import re

    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # 1. Clip function
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number])
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number])
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number])

    # 2. Initial stocking
    stocked_fish = 7290
    initial_abw_g = 11.9
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "AVERAGE BODY WEIGHT(G)": initial_abw_g
    }])
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # 3. Initialize cumulative columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # 4. Compute harvest cumulatives
    if not harvest_c2.empty:
        h_fish_col = "NUMBER OF FISH" if "NUMBER OF FISH" in harvest_c2.columns else None
        h_kg_col   = "TOTAL WEIGHT (KG)" if "TOTAL WEIGHT (KG)" in harvest_c2.columns else None

        if h_fish_col or h_kg_col:
            h = harvest_c2.copy().sort_values("DATE")
            h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0) if h_fish_col else 0
            h["H_KG"]   = pd.to_numeric(h[h_kg_col],   errors="coerce").fillna(0) if h_kg_col else 0
            h["HARV_FISH_CUM"], h["HARV_KG_CUM"] = h["H_FISH"].cumsum(), h["H_KG"].cumsum()
            # Merge to base using closest previous date
            base = pd.merge_asof(base.sort_values("DATE"), 
                                 h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]].sort_values("DATE"), 
                                 on="DATE", direction="backward", suffixes=("","_h"))
            base["HARV_FISH_CUM"].fillna(0, inplace=True)
            base["HARV_KG_CUM"].fillna(0, inplace=True)

    # 5. Compute transfer cumulatives
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        if not t.empty:
            # Convert cage columns to int
            for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
                if col in t.columns:
                    t[col] = t[col].apply(lambda x: int(re.search(r"(\d+)", str(x)).group(1)) if pd.notna(x) and re.search(r"(\d+)", str(x)) else np.nan)
            # Fish and weight columns
            t["T_FISH"] = pd.to_numeric(t["NUMBER OF FISH"], errors="coerce").fillna(0) if "NUMBER OF FISH" in t.columns else 0
            t["T_KG"]   = pd.to_numeric(t["TOTAL WEIGHT (KG)"], errors="coerce").fillna(0) if "TOTAL WEIGHT (KG)" in t.columns else 0

            # Outgoing
            tout = t[t["ORIGIN CAGE"] == cage_number].sort_values("DATE")
            if not tout.empty:
                tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
                tout["OUT_KG_CUM"]   = tout["T_KG"].cumsum()
                base = pd.merge_asof(base.sort_values("DATE"), 
                                     tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]].sort_values("DATE"),
                                     on="DATE", direction="backward").fillna(0)

            # Incoming
            tin = t[t["DESTINATION CAGE"] == cage_number].sort_values("DATE")
            if not tin.empty:
                tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
                tin["IN_KG_CUM"]   = tin["T_KG"].cumsum()
                base = pd.merge_asof(base.sort_values("DATE"), 
                                     tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]].sort_values("DATE"),
                                     on="DATE", direction="backward").fillna(0)

  
    # 6. Standing fish
    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    return feeding_c2, harvest_c2, base
    
# 3. Compute Summary
def compute_summary(feeding_c2, sampling_c2):
    feeding_c2  = feeding_c2.copy()
    s = sampling_c2.copy().sort_values("DATE")

    feed_col = find_col(feeding_c2, ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)"], "FEED")
    abw_col  = find_col(s, ["AVERAGE BODY WEIGHT(G)","AVERAGE BODY WEIGHT (G)","ABW"], "ABW")
    if not feed_col or not abw_col:
        return s

    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()
    summary = pd.merge_asof(s, feeding_c2[["DATE","CUM_FEED"]], on="DATE", direction="backward")

    summary["ABW_G"] = pd.to_numeric(summary[abw_col].map(to_number), errors="coerce")
    summary["BIOMASS_KG"] = pd.to_numeric(summary["FISH_ALIVE"], errors="coerce") * summary["ABW_G"] / 1000.0

    # Period deltas
    summary["FEED_PERIOD_KG"] = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"]    = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Logistics per period (kg & fish)
    for cum_col, per_col in [
        ("IN_KG_CUM","TRANSFER_IN_KG"),("OUT_KG_CUM","TRANSFER_OUT_KG"),("HARV_KG_CUM","HARVEST_KG")]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan
    summary["TRANSFER_IN_FISH"]  = summary["IN_FISH_CUM"].diff()   if "IN_FISH_CUM"   in summary.columns else np.nan
    summary["TRANSFER_OUT_FISH"] = summary["OUT_FISH_CUM"].diff()  if "OUT_FISH_CUM"  in summary.columns else np.nan
    summary["HARVEST_FISH"]      = summary["HARV_FISH_CUM"].diff() if "HARV_FISH_CUM" in summary.columns else np.nan

    # Growth & eFCR
    summary["GROWTH_KG"] = summary["ΔBIOMASS_STANDING"] + summary["HARVEST_KG"].fillna(0) \
                            + summary["TRANSFER_OUT_KG"].fillna(0) - summary["TRANSFER_IN_KG"].fillna(0)
    growth_cum = summary["GROWTH_KG"].cumsum(skipna=True)
    summary["PERIOD_eFCR"] = np.where(summary["GROWTH_KG"]>0, summary["FEED_PERIOD_KG"]/summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum>0, summary["FEED_AGG_KG"]/growth_cum, np.nan)

    # First row → NA
    first_idx = summary.index.min()
    summary.loc[first_idx, ["FEED_PERIOD_KG","ΔBIOMASS_STANDING","TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
                            "TRANSFER_IN_FISH","TRANSFER_OUT_FISH","GROWTH_KG","PERIOD_eFCR"]] = np.nan

    return summary


# ===========================
# 5. Streamlit Interface
# ===========================
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

    st.subheader("Cage 2 – Production Summary")
    st.dataframe(summary_c2)

    selected_kpi = st.sidebar.selectbox("Select KPI", ["Biomass","ABW","eFCR"])
    if selected_kpi == "Biomass":
        fig = px.line(summary_c2.dropna(subset=["BIOMASS_KG"]), x="DATE", y="BIOMASS_KG", markers=True,
                      title="Cage 2: Biomass Over Time", labels={"BIOMASS_KG":"Total Biomass (kg)"})
        st.plotly_chart(fig, use_container_width=True)
    elif selected_kpi == "ABW":
        fig = px.line(summary_c2.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                      title="Cage 2: Average Body Weight Over Time", labels={"ABW_G":"ABW (g)"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
        fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True, title="Cage 2: eFCR Over Time")
        fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", line=dict(dash="dash"))
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)

