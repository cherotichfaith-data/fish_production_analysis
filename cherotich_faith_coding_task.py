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
    import pandas as pd
    import numpy as np

    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # ----------------------
    # Clip function for timeframe
    # ----------------------
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        out = df.dropna(subset=["DATE"]).sort_values("DATE")
        return out[(out["DATE"] >= start_date) & (out["DATE"] <= end_date)]

    feeding_c2  = _clip(feeding[feeding["CAGE NUMBER"] == cage_number])   if "CAGE NUMBER" in feeding.columns else _clip(feeding)
    harvest_c2  = _clip(harvest[harvest["CAGE NUMBER"] == cage_number])   if "CAGE NUMBER" in harvest.columns else _clip(harvest)
    sampling_c2 = _clip(sampling[sampling["CAGE NUMBER"] == cage_number]) if "CAGE NUMBER" in sampling.columns else _clip(sampling)

    # ----------------------
    # Stocking row
    # ----------------------
    stocked_fish = 7290
    initial_abw_g = 11.9
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "ABW_G": initial_abw_g,
        "STOCKED": stocked_fish,
        "FISH_ALIVE": stocked_fish
    }])

    # ----------------------
    # Combine sampling + stocking
    # ----------------------
    if "AVERAGE BODY WEIGHT(G)" in sampling_c2.columns:
        sampling_c2.rename(columns={"AVERAGE BODY WEIGHT(G)": "ABW_G"}, inplace=True)

    base = pd.concat([stocking_row, sampling_c2], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # Initialize cumulative columns
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        if col not in base.columns:
            base[col] = 0.0

    # ----------------------
    # Harvest cumulatives
    # ----------------------
    if not harvest_c2.empty:
        h_fish_col = next((c for c in harvest_c2.columns if "FISH" in c.upper()), None)
        h_kg_col   = next((c for c in harvest_c2.columns if "WEIGHT" in c.upper()), None)
        h = harvest_c2.copy()
        if h_fish_col:
            h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0)
        else:
            h["H_FISH"] = 0
        if h_kg_col:
            h["H_KG"] = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0)
        else:
            h["H_KG"] = 0
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
        if not t.empty:
            for flow, col_prefix in [("ORIGIN", "OUT"), ("DESTINATION", "IN")]:
                cage_col = f"{flow} CAGE"
                fish_col = next((c for c in t.columns if "FISH" in c.upper()), None)
                kg_col   = next((c for c in t.columns if "WEIGHT" in c.upper()), None)
                mask = (t[cage_col] == cage_number) if cage_col in t.columns else pd.Series(False, index=t.index)
                sub = t[mask].copy()
                if not sub.empty:
                    sub["T_FISH"] = pd.to_numeric(sub[fish_col], errors="coerce").fillna(0) if fish_col else 0
                    sub["T_KG"]   = pd.to_numeric(sub[kg_col], errors="coerce").fillna(0) if kg_col else 0
                    # Convert grams → kg if needed
                    sub["T_KG"] = sub["T_KG"].apply(lambda x: x/1000 if x > 100 else x)
                    sub[f"{col_prefix}_FISH_CUM"] = sub["T_FISH"].cumsum()
                    sub[f"{col_prefix}_KG_CUM"]   = sub["T_KG"].cumsum()
                    merged = pd.merge_asof(base[["DATE"]], sub[["DATE", f"{col_prefix}_FISH_CUM", f"{col_prefix}_KG_CUM"]], on="DATE", direction="backward")
                    base[f"{col_prefix}_FISH_CUM"] = merged[f"{col_prefix}_FISH_CUM"].fillna(0)
                    base[f"{col_prefix}_KG_CUM"]   = merged[f"{col_prefix}_KG_CUM"].fillna(0)

    # ----------------------
    # Standing fish
    # ----------------------
    for col in ["STOCKED","HARV_FISH_CUM","IN_FISH_CUM","OUT_FISH_CUM"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)

    base["FISH_ALIVE"] = (base["STOCKED"] - base["HARV_FISH_CUM"] + base["IN_FISH_CUM"] - base["OUT_FISH_CUM"]).clip(lower=0)
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].astype(int)

    # ----------------------
    # Sort final
    # ----------------------
    base = base.sort_values("DATE").reset_index(drop=True)

    return feeding_c2, harvest_c2, base

def compute_summary(feeding_c2, sampling_c2):
    import pandas as pd
    import numpy as np

    feeding_c2 = feeding_c2.copy()
    s = sampling_c2.copy().sort_values("DATE")

    # Standardize ABW column
    abw_candidates = ["AVERAGE BODY WEIGHT(G)","AVERAGE BODY WEIGHT (G)","ABW(G)","ABW [G]","ABW"]
    abw_col = next((c for c in abw_candidates if c in s.columns), None)
    if abw_col is None:
        s["ABW_G"] = np.nan
    else:
        s.rename(columns={abw_col: "ABW_G"}, inplace=True)

    # Standardize feed column
    feed_candidates = ["FEED AMOUNT (KG)","FEED AMOUNT (Kg)","FEED (KG)","FEED KG","FEED_AMOUNT","FEED"]
    feed_col = next((c for c in feed_candidates if c in feeding_c2.columns), None)
    if feed_col is None:
        feeding_c2["FEED"] = 0
        feed_col = "FEED"

    # Cumulative feed
    feeding_c2 = feeding_c2.sort_values("DATE")
    feeding_c2["CUM_FEED"] = pd.to_numeric(feeding_c2[feed_col], errors="coerce").fillna(0).cumsum()

    # Merge cumulative feed to sampling
    summary = pd.merge_asof(
        s,
        feeding_c2[["DATE", "CUM_FEED"]],
        on="DATE",
        direction="backward"
    )

    # Ensure FISH_ALIVE exists
    if "FISH_ALIVE" not in summary.columns:
        summary["FISH_ALIVE"] = summary.get("NUMBER OF FISH", np.nan).fillna(0)

    # Compute biomass safely
    summary["BIOMASS_KG"] = summary["FISH_ALIVE"] * summary["ABW_G"].fillna(0) / 1000.0

    # Period deltas
    summary["FEED_PERIOD_KG"]    = summary["CUM_FEED"].diff()
    summary["FEED_AGG_KG"]       = summary["CUM_FEED"]
    summary["ΔBIOMASS_STANDING"] = summary["BIOMASS_KG"].diff()

    # Period logistics (kg)
    for cum_col, per_col in [
        ("IN_KG_CUM","TRANSFER_IN_KG"),
        ("OUT_KG_CUM","TRANSFER_OUT_KG"),
        ("HARV_KG_CUM","HARVEST_KG")
    ]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan

    # Period logistics (fish)
    for cum_col, per_col in [
        ("IN_FISH_CUM","TRANSFER_IN_FISH"),
        ("OUT_FISH_CUM","TRANSFER_OUT_FISH"),
        ("HARV_FISH_CUM","HARVEST_FISH")
    ]:
        summary[per_col] = summary[cum_col].diff() if cum_col in summary.columns else np.nan

    # Period growth
    summary["GROWTH_KG"] = (
        summary["ΔBIOMASS_STANDING"].fillna(0)
        + summary["HARVEST_KG"].fillna(0)
        + summary["TRANSFER_OUT_KG"].fillna(0)
        - summary["TRANSFER_IN_KG"].fillna(0)
    )

    # Fish count discrepancy
    summary["EXPECTED_FISH_ALIVE"] = (
        summary.get("STOCKED", 0)
        - summary.get("HARV_FISH_CUM", 0)
        + summary.get("IN_FISH_CUM", 0)
        - summary.get("OUT_FISH_CUM", 0)
    )
    actual_fish = pd.to_numeric(summary.get("NUMBER OF FISH", 0), errors="coerce").fillna(0)
    summary["FISH_COUNT_DISCREPANCY"] = summary["EXPECTED_FISH_ALIVE"] - actual_fish

    # eFCR
    growth_cum = summary["GROWTH_KG"].cumsum()
    summary["PERIOD_eFCR"]     = np.where(summary["GROWTH_KG"] > 0, summary["FEED_PERIOD_KG"] / summary["GROWTH_KG"], np.nan)
    summary["AGGREGATED_eFCR"] = np.where(growth_cum > 0, summary["FEED_AGG_KG"] / growth_cum, np.nan)

    # First row → NA for period metrics
    if not summary.empty:
        first_idx = summary.index.min()
        for col in [
            "FEED_PERIOD_KG","ΔBIOMASS_STANDING",
            "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
            "TRANSFER_IN_FISH","TRANSFER_OUT_FISH",
            "GROWTH_KG","PERIOD_eFCR","FISH_COUNT_DISCREPANCY"
        ]:
            summary.loc[first_idx, col] = np.nan

    # Column order
    cols = [
        "DATE","CAGE NUMBER","NUMBER OF FISH","ABW_G","BIOMASS_KG",
        "FEED_PERIOD_KG","FEED_AGG_KG","GROWTH_KG",
        "TRANSFER_IN_KG","TRANSFER_OUT_KG","HARVEST_KG",
        "TRANSFER_IN_FISH","TRANSFER_OUT_FISH","HARVEST_FISH",
        "FISH_COUNT_DISCREPANCY",
        "PERIOD_eFCR","AGGREGATED_eFCR",
    ]
    return summary[[c for c in cols if c in summary.columns]]

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

    # Helper to create empty chart
    def empty_chart(y_label):
        fig = px.line(x=[], y=[], title="No data available")
        fig.update_layout(yaxis_title=y_label, xaxis_title="Date")
        return fig

    # --- Robust plotting
    if selected_kpi == "Biomass":
        if "BIOMASS_KG" in summary_c2.columns and not summary_c2["BIOMASS_KG"].isna().all():
            fig = px.line(summary_c2.dropna(subset=["BIOMASS_KG"]), x="DATE", y="BIOMASS_KG", markers=True,
                          title="Cage 2: Biomass Over Time", labels={"BIOMASS_KG":"Total Biomass (kg)"})
        else:
            fig = empty_chart("Total Biomass (kg)")
        st.plotly_chart(fig, use_container_width=True)
        
    elif selected_kpi == "ABW":
        if "ABW_G" in summary_c2.columns and not summary_c2["ABW_G"].isna().all():
            fig = px.line(summary_c2.dropna(subset=["ABW_G"]), x="DATE", y="ABW_G", markers=True,
                          title="Cage 2: Average Body Weight Over Time", labels={"ABW_G":"ABW (g)"})
        else:
            fig = empty_chart("ABW (g)")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        if ("AGGREGATED_eFCR" in summary_c2.columns and 
            "PERIOD_eFCR" in summary_c2.columns and
            not summary_c2[["AGGREGATED_eFCR","PERIOD_eFCR"]].isna().all().all()):
            
            dff = summary_c2.dropna(subset=["AGGREGATED_eFCR","PERIOD_eFCR"])
            fig = px.line(dff, x="DATE", y="AGGREGATED_eFCR", markers=True, title="Cage 2: eFCR Over Time")
            fig.add_scatter(x=dff["DATE"], y=dff["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR", line=dict(dash="dash"))
            fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        else:
            fig = empty_chart("eFCR")
        st.plotly_chart(fig, use_container_width=True)
