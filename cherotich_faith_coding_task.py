# app.py — Fish Cage Production Analysis (updated per your corrections)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Fish Cage Production Analysis")

# -------------------------
# Helpers for robust column retrieval & unit fixes
# -------------------------
def find_col(df, candidates):
    """Return the first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_datetime(df, colnames):
    for c in colnames:
        if c and c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def convert_transfer_weights(transfer_df):
    """
    Convert transfer weights to kg when unit is ambiguous.
    Heuristic: if the median weight is > 100, assume grams and divide by 1000.
    Accepts columns like 'Total weight [kg]' or 'TOTAL_WEIGHT' etc.
    """
    # try several candidates
    weight_col = find_col(transfer_df, [
        "Total weight [kg]",
        "Total weight [g]",
        "TOTAL WEIGHT [kg]",
        "TotalWeight",
        "TOTAL_WEIGHT",
        "TOTAL_WEIGHT_G",
        "WEIGHT"
    ])
    if weight_col is None:
        return transfer_df, None

    # coerce numeric
    transfer_df[weight_col] = pd.to_numeric(transfer_df[weight_col], errors="coerce")

    # median heuristic: if median > 100 -> probably grams
    med = transfer_df[weight_col].median(skipna=True)
    if pd.notna(med) and med > 100:
        transfer_df[weight_col] = transfer_df[weight_col] / 1000.0  # convert g -> kg
    # rename to canonical
    transfer_df = transfer_df.rename(columns={weight_col: "TRANSFER_WEIGHT_KG"})
    return transfer_df, "TRANSFER_WEIGHT_KG"

# -------------------------
# 1. Load Data
# -------------------------
def load_data(feeding_f, harvest_f, sampling_f, transfer_f):
    feeding = pd.read_excel(feeding_f)
    harvest = pd.read_excel(harvest_f)
    sampling = pd.read_excel(sampling_f)
    transfer = pd.read_excel(transfer_f)
    return feeding, harvest, sampling, transfer

# -------------------------
# 2. Inspect & Clean (missing values)
# -------------------------
def inspect_missing_values(dfs):
    """Return a dict name -> missing counts (Series) for display."""
    miss = {}
    for name, df in dfs.items():
        miss[name] = df.isnull().sum()
    return miss

def clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df):
    # Drop fully-empty columns
    feeding_df = feeding_df.dropna(axis=1, how="all")
    harvest_df = harvest_df.dropna(axis=1, how="all")
    sampling_df = sampling_df.dropna(axis=1, how="all")
    transfer_df = transfer_df.dropna(axis=1, how="all")

    # Standardize date columns
    feeding_date_col = find_col(feeding_df, ["DATE", "Date", "date"])
    harvest_date_col = find_col(harvest_df, ["DATE", "Date", "date"])
    sampling_date_col = find_col(sampling_df, ["DATE", "Date", "date"])
    transfer_date_col = find_col(transfer_df, ["DATE", "Date", "date", "Transfer Date"])

    feeding_df = ensure_datetime(feeding_df, [feeding_date_col])
    harvest_df = ensure_datetime(harvest_df, [harvest_date_col])
    sampling_df = ensure_datetime(sampling_df, [sampling_date_col])
    transfer_df = ensure_datetime(transfer_df, [transfer_date_col])

    # Canonicalize key columns (defensive)
    # Feeding: FEED AMOUNT (Kg), CAGE NUMBER
    fa = find_col(feeding_df, ["FEED AMOUNT (Kg)", "FEED AMOUNT KG", "Feed Amount (Kg)", "Feed Amount", "FEED_AMT_KG", "FEED AMOUNT"])
    cn_f = find_col(feeding_df, ["CAGE NUMBER", "CAGE", "Cage", "Cage Number"])
    feeding_df = feeding_df.rename(columns={fa: "FEED_AMT_KG"} if fa else {}).rename(columns={cn_f: "CAGE_NUMBER"} if cn_f else {})

    # Harvest: CAGE, NUMBER OF FISH, TOTAL WEIGHT [kg]
    cn_h = find_col(harvest_df, ["CAGE", "Cage", "CAGE NUMBER"])
    num_h = find_col(harvest_df, ["NUMBER OF FISH", "Number of Fish", "NUM_FISH", "NUMBER_OF_FISH"])
    tw_h = find_col(harvest_df, ["TOTAL WEIGHT  [kg]", "TOTAL_WEIGHT_KG", "Total weight [kg]", "TotalWeightKg", "WEIGHT_KG"])
    harvest_df = harvest_df.rename(columns={cn_h: "CAGE", num_h: "HARV_NUM_FISH"} if cn_h or num_h else {})\
                           .rename(columns={tw_h: "HARV_WEIGHT_KG"} if tw_h else {})

    # Sampling: CAGE NUMBER, NUMBER OF FISH, AVERAGE BODY WEIGHT (g)
    cn_s = find_col(sampling_df, ["CAGE NUMBER", "CAGE", "Cage"])
    num_s = find_col(sampling_df, ["NUMBER OF FISH", "NUM_FISH", "Number of Fish"])
    abw = find_col(sampling_df, ["AVERAGE BODY WEIGHT (g)", "AVERAGE BODY WEIGHT(g)", "ABW [g]", "ABW_g", "AVERAGE_BODY_WEIGHT_G"])
    sampling_df = sampling_df.rename(columns={cn_s: "CAGE_NUMBER", num_s: "SAMP_NUM_FISH"} if cn_s or num_s else {})\
                             .rename(columns={abw: "ABW_G"} if abw else {})

    # Transfer cleaning + units correction
    # canonicalize source/dest cage and number of fish
    src = find_col(transfer_df, ["SOURCE CAGE", "Source Cage", "SOURCE", "FROM_CAGE"])
    dest = find_col(transfer_df, ["DESTINATION CAGE", "DESTINATION", "DEST CAGE", "TO_CAGE"])
    num_t = find_col(transfer_df, ["NUMBER OF FISH", "NUM_FISH", "Number of Fish"])
    transfer_df = transfer_df.rename(columns={src: "SOURCE_CAGE"} if src else {}).rename(columns={dest: "DESTINATION_CAGE"} if dest else {})\
                             .rename(columns={num_t: "TRANSFER_NUM_FISH"} if num_t else {})

    # Convert transfer weights to kg and rename column
    transfer_df, transfer_weight_col = convert_transfer_weights(transfer_df)

    # Ensure numeric columns
    if "FEED_AMT_KG" in feeding_df.columns:
        feeding_df["FEED_AMT_KG"] = pd.to_numeric(feeding_df["FEED_AMT_KG"], errors="coerce").fillna(0)
    if "HARV_WEIGHT_KG" in harvest_df.columns:
        harvest_df["HARV_WEIGHT_KG"] = pd.to_numeric(harvest_df["HARV_WEIGHT_KG"], errors="coerce").fillna(0)
    if "HARV_NUM_FISH" in harvest_df.columns:
        harvest_df["HARV_NUM_FISH"] = pd.to_numeric(harvest_df["HARV_NUM_FISH"], errors="coerce").fillna(0)
    if "SAMP_NUM_FISH" in sampling_df.columns:
        sampling_df["SAMP_NUM_FISH"] = pd.to_numeric(sampling_df["SAMP_NUM_FISH"], errors="coerce")
    if "ABW_G" in sampling_df.columns:
        sampling_df["ABW_G"] = pd.to_numeric(sampling_df["ABW_G"], errors="coerce")

    return feeding_df, harvest_df, sampling_df, transfer_df

# -------------------------
# 3. Compute production summary (keeps your logic, expanded to per-sample)
# -------------------------
def compute_production_summary_for_cage(feeding_df, harvest_df, sampling_df, transfer_df,
                                       cage_number,
                                       stocking_date=pd.to_datetime("2024-08-26"),
                                       stocked_fish=7290,
                                       initial_abw_g=11.9,
                                       start_date=pd.to_datetime("2024-07-16"),
                                       end_date=pd.to_datetime("2025-07-09")):
    """
    Returns a detailed production summary dataframe indexed by sampling DATE with:
    biomass at sample, cumulative feed (kg), cumulative removed (harvest+transfers),
    aggregated_eFCR, and period_eFCR (since previous sample).
    """

    # Filter cage-specific records
    fcol_date = find_col(feeding_df, ["DATE", "Date", "date"])
    feeding_df = feeding_df.rename(columns={fcol_date: "DATE"}) if fcol_date else feeding_df
    feeding = feeding_df[feeding_df.get("CAGE_NUMBER") == cage_number].copy()
    feeding["DATE"] = pd.to_datetime(feeding["DATE"], errors="coerce")

    # Harvest for cage
    harvest_date_col = find_col(harvest_df, ["DATE", "Date", "date"])
    harvest_df = harvest_df.rename(columns={harvest_date_col: "DATE"}) if harvest_date_col else harvest_df
    harvest = harvest_df[harvest_df.get("CAGE") == cage_number].copy()
    harvest["DATE"] = pd.to_datetime(harvest["DATE"], errors="coerce")

    # Sampling for cage
    samp_date_col = find_col(sampling_df, ["DATE", "Date", "date"])
    sampling_df = sampling_df.rename(columns={samp_date_col: "DATE"}) if samp_date_col else sampling_df
    sampling = sampling_df[sampling_df.get("CAGE_NUMBER") == cage_number].copy()
    sampling["DATE"] = pd.to_datetime(sampling["DATE"], errors="coerce")

    # Transfers where source is this cage (outgoing transfers reduce cage biomass)
    tr_date_col = find_col(transfer_df, ["DATE", "Date", "date"])
    transfer_df = transfer_df.rename(columns={tr_date_col: "DATE"}) if tr_date_col else transfer_df
    transfers_out = transfer_df[transfer_df.get("SOURCE_CAGE") == cage_number].copy()
    transfers_out["DATE"] = pd.to_datetime(transfers_out["DATE"], errors="coerce")
    # Ensure TRANSFER_WEIGHT_KG present
    if "TRANSFER_WEIGHT_KG" not in transfers_out.columns:
        transfers_out["TRANSFER_WEIGHT_KG"] = 0

    # Add stocking row to sampling (explicit start)
    stocking_row = pd.DataFrame([{
        "DATE": pd.to_datetime(stocking_date),
        "CAGE_NUMBER": cage_number,
        "SAMP_NUM_FISH": stocked_fish,
        "ABW_G": initial_abw_g
    }])
    # Combine and limit timeframe
    sampling = pd.concat([stocking_row, sampling], ignore_index=True).drop_duplicates(subset=["DATE"], keep="first")
    sampling = sampling.sort_values("DATE")
    sampling = sampling[(sampling["DATE"] >= start_date) & (sampling["DATE"] <= end_date)].reset_index(drop=True)
    feeding = feeding[(feeding["DATE"] >= start_date) & (feeding["DATE"] <= end_date)].sort_values("DATE").reset_index(drop=True)
    harvest = harvest[(harvest["DATE"] >= start_date) & (harvest["DATE"] <= end_date)].sort_values("DATE").reset_index(drop=True)
    transfers_out = transfers_out[(transfers_out["DATE"] >= start_date) & (transfers_out["DATE"] <= end_date)].sort_values("DATE").reset_index(drop=True)

    # Prepare cumulative feed series by date
    # If feeding has no DATE or FEED_AMT_KG, create daily zeros
    if "DATE" not in feeding.columns or "FEED_AMT_KG" not in feeding.columns or feeding.empty:
        # create synthetic daily feed (small daily default) — but for cage2 we expect real data
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        feeding = pd.DataFrame({"DATE": date_range, "CAGE_NUMBER": cage_number, "FEED_AMT_KG": np.random.uniform(5, 12, len(date_range))})
    # Aggregate feed by date
    daily_feed = feeding.groupby("DATE", as_index=False)["FEED_AMT_KG"].sum().sort_values("DATE")
    daily_feed["CUM_FEED_KG"] = daily_feed["FEED_AMT_KG"].cumsum()

    # Helper to get cumulative feed up to a date
    def cum_feed_up_to(d):
        # last CUM_FEED_KG where DATE <= d
        tmp = daily_feed[daily_feed["DATE"] <= d]
        return float(tmp["CUM_FEED_KG"].iloc[-1]) if not tmp.empty else 0.0

    # Helper for cumulative removed biomass (harvest + transfers out)
    harvest_by_date = harvest.groupby("DATE", as_index=False)["HARV_WEIGHT_KG"].sum().sort_values("DATE") if not harvest.empty else pd.DataFrame(columns=["DATE", "HARV_WEIGHT_KG"])
    transfers_by_date = transfers_out.groupby("DATE", as_index=False)["TRANSFER_WEIGHT_KG"].sum().sort_values("DATE") if not transfers_out.empty else pd.DataFrame(columns=["DATE", "TRANSFER_WEIGHT_KG"])
    # compute cumulative sums
    if not harvest_by_date.empty:
        harvest_by_date["CUM_HARV_KG"] = harvest_by_date["HARV_WEIGHT_KG"].cumsum()
    else:
        harvest_by_date = pd.DataFrame(columns=["DATE", "HARV_WEIGHT_KG", "CUM_HARV_KG"])
    if not transfers_by_date.empty:
        transfers_by_date["CUM_TR_KG"] = transfers_by_date["TRANSFER_WEIGHT_KG"].cumsum()
    else:
        transfers_by_date = pd.DataFrame(columns=["DATE", "TRANSFER_WEIGHT_KG", "CUM_TR_KG"])

    def cum_removed_up_to(d):
        h = harvest_by_date[harvest_by_date["DATE"] <= d]
        t = transfers_by_date[transfers_by_date["DATE"] <= d]
        hsum = float(h["HARV_WEIGHT_KG"].sum()) if not h.empty else 0.0
        tsum = float(t["TRANSFER_WEIGHT_KG"].sum()) if not t.empty else 0.0
        return hsum + tsum

    # Initial stocking biomass (kg)
    initial_biomass_kg = stocked_fish * (initial_abw_g / 1000.0)

    # Build summary rows at each sampling event
    rows = []
    prev_total_biomass_produced = 0.0
    prev_cum_feed = 0.0
    prev_date = None

    for idx, s in sampling.iterrows():
        d = s["DATE"]
        samp_num = float(s.get("SAMP_NUM_FISH", np.nan))
        abw_g = float(s.get("ABW_G", np.nan))
        biomass_kg = samp_num * (abw_g / 1000.0) if (pd.notna(samp_num) and pd.notna(abw_g)) else np.nan

        cum_feed = cum_feed_up_to(d)
        cum_removed = cum_removed_up_to(d)

        # Total biomass produced since stocking = (current biomass + cumulative removed) - initial biomass
        total_biomass_produced = 0.0
        if pd.notna(biomass_kg):
            total_biomass_produced = (biomass_kg + cum_removed) - initial_biomass_kg
        else:
            total_biomass_produced = (cum_removed) - initial_biomass_kg  # fallback

        aggregated_eFCR = np.nan
        if total_biomass_produced > 0:
            aggregated_eFCR = cum_feed / total_biomass_produced

        # Period computations since previous sampling point (or stocking for first row)
        if prev_date is None:
            period_feed = cum_feed  # since stocking
            # removed in period = cum_removed - 0
            period_removed = cum_removed
            prev_total_biomass_produced = 0.0
        else:
            period_feed = cum_feed - prev_cum_feed
            # biomass produced in period = (biomass_kg - biomass_prev) + removed_in_period
            removed_in_period = cum_removed - cum_removed_up_to(prev_date)
            biomass_prev = prev_row.get("BIOMASS_KG", 0.0) if prev_row.get("BIOMASS_KG") is not None else 0.0
            if pd.notna(biomass_kg):
                period_biomass_gain = (biomass_kg - biomass_prev) + removed_in_period
            else:
                period_biomass_gain = removed_in_period
            period_removed = removed_in_period

        period_biomass_gain = None
        if prev_date is None:
            period_biomass_gain = total_biomass_produced
        else:
            # compute biomass gain in period explicitly
            prev_biomass = prev_row.get("BIOMASS_KG", 0.0)
            prev_cum_removed = cum_removed_up_to(prev_date)
            prev_total_produced = (prev_biomass + prev_cum_removed) - initial_biomass_kg
            period_biomass_gain = total_biomass_produced - prev_total_produced

        if period_biomass_gain is None:
            period_biomass_gain = np.nan

        period_eFCR = np.nan
        if period_biomass_gain is not None and period_biomass_gain > 0:
            # careful: period_feed might be slightly negative due to rounding; guard it
            pf = max(0.0, float(period_feed))
            period_eFCR = pf / period_biomass_gain

        row = {
            "DATE": d,
            "NUMBER_OF_FISH": samp_num,
            "ABW_G": abw_g,
            "BIOMASS_KG": biomass_kg,
            "CUM_FEED_KG": cum_feed,
            "CUM_REMOVED_KG": cum_removed,
            "TOTAL_BIOMASS_PRODUCED_KG": total_biomass_produced,
            "AGGREGATED_eFCR": aggregated_eFCR,
            "PERIOD_FEED_KG": float(period_feed),
            "PERIOD_BIOMASS_PRODUCED_KG": float(period_biomass_gain),
            "PERIOD_eFCR": period_eFCR
        }
        rows.append(row)
        prev_date = d
        prev_cum_feed = cum_feed
        prev_row = row.copy()

    summary_df = pd.DataFrame(rows).sort_values("DATE").reset_index(drop=True)
    return summary_df, daily_feed, harvest_by_date, transfers_by_date

# -------------------------
# 4. Mock cages generator (3-7) based on cage2 summary
# -------------------------
def create_mock_cages_from_cage2(summary_c2, daily_feed_c2, harvest_c2, num_mocks=5, base_cage_id=3):
    """
    Create mock cages (IDs base_cage_id ... base_cage_id+num_mocks-1)
    - daily feeding: scaled and noisy version of cage2 daily_feed
    - sampling: same dates as summary_c2 but perturbed numbers/ABW
    - harvest: scaled copies of harvest_c2 by date
    - no transfers
    Returns dict: cage_id -> (summary_df, daily_feed_df, harvest_df)
    """
    mock_results = {}
    for i in range(num_mocks):
        cage_id = base_cage_id + i
        # daily feed: scale by random factor 0.8..1.2 and add small noise
        df_feed = daily_feed_c2.copy().reset_index(drop=True)
        scale_feed = np.random.normal(1.0, 0.08)
        df_feed["FEED_AMT_KG"] = (df_feed["FEED_AMT_KG"] * scale_feed).clip(lower=0.1)
        df_feed["CUM_FEED_KG"] = df_feed["FEED_AMT_KG"].cumsum()
        df_feed["CAGE_NUMBER"] = cage_id

        # sampling: copy summary_c2 dates, perturb fish counts and ABW slightly
        samp = summary_c2.copy()
        samp = samp.sort_values("DATE").reset_index(drop=True)
        samp["NUMBER_OF_FISH"] = (samp["NUMBER_OF_FISH"].fillna(0).astype(float) + np.random.randint(-50, 50, size=len(samp))).clip(lower=0)
        samp["ABW_G"] = (samp["ABW_G"].fillna(0).astype(float) * np.random.normal(1.0, 0.03, size=len(samp))).clip(lower=0.1)
        # recompute biomass
        samp["BIOMASS_KG"] = samp["NUMBER_OF_FISH"] * (samp["ABW_G"] / 1000.0)

        # harvest: scale original harvest pattern by a random factor
        if not harvest_c2.empty:
            h = harvest_c2.copy()
            scale_h = np.random.normal(1.0, 0.1)
            h["HARV_WEIGHT_KG"] = (h["HARV_WEIGHT_KG"].fillna(0).astype(float) * scale_h).clip(lower=0)
            h["CAGE"] = cage_id
        else:
            h = pd.DataFrame(columns=["DATE", "HARV_WEIGHT_KG", "CAGE"])

        # Recompute aggregated and period eFCR for mock cage using the same routines:
        # Build feeding and harvest dfs with canonical columns
        feeding_mock = pd.DataFrame({"DATE": df_feed["DATE"], "CAGE_NUMBER": df_feed["CAGE_NUMBER"], "FEED_AMT_KG": df_feed["FEED_AMT_KG"]})
        harvest_mock = pd.DataFrame({"DATE": h["DATE"], "CAGE": h["CAGE"], "HARV_WEIGHT_KG": h["HARV_WEIGHT_KG"]})
        sampling_mock = pd.DataFrame({"DATE": samp["DATE"], "CAGE_NUMBER": cage_id, "SAMP_NUM_FISH": samp["NUMBER_OF_FISH"], "ABW_G": samp["ABW_G"]})
        transfer_mock = pd.DataFrame(columns=["DATE", "SOURCE_CAGE", "DESTINATION_CAGE", "TRANSFER_WEIGHT_KG"])  # empty

        summary_mock, _, _, _ = compute_production_summary_for_cage(feeding_mock, harvest_mock, sampling_mock, transfer_mock,
                                                                    cage_number=cage_id,
                                                                    stocking_date=samp["DATE"].min(),
                                                                    stocked_fish=int(samp["NUMBER_OF_FISH"].iloc[0]) if len(samp)>0 else 1000,
                                                                    initial_abw_g=float(samp["ABW_G"].iloc[0]) if len(samp)>0 else 10.0,
                                                                    start_date=df_feed["DATE"].min(),
                                                                    end_date=df_feed["DATE"].max())
        mock_results[cage_id] = {"summary": summary_mock, "daily_feed": df_feed, "harvest": h}
    return mock_results

# -------------------------
# 5. Streamlit UI
# -------------------------
st.title("Fish Cage Production Analysis — Cage 2 baseline + mocks")
st.markdown("Upload your Excel files (if you already uploaded a cleaned transfer file, the app will handle it).")

with st.sidebar:
    st.header("Upload files (Excel)")
    feeding_file = st.file_uploader("Feeding record (xlsx)", type=["xlsx", "xls"])
    harvest_file = st.file_uploader("Harvest record (xlsx)", type=["xlsx", "xls"])
    sampling_file = st.file_uploader("Sampling record (xlsx)", type=["xlsx", "xls"])
    transfer_file = st.file_uploader("Transfer record (xlsx). If transfer weights were in g, the app will convert.", type=["xlsx", "xls"])

# If any file not supplied, allow user to run with synthetic example (useful for quick testing)
use_example = False
if feeding_file and harvest_file and sampling_file and transfer_file:
    feeding_df, harvest_df, sampling_df, transfer_df = load_data(feeding_file, harvest_file, sampling_file, transfer_file)
else:
    st.warning("Not all files uploaded. You can either upload all four files OR click 'Use example data' to run a demonstration.")
    if st.button("Use example data (demo)"):
        use_example = True

if use_example:
    # Create very small example datasets (toy) to demonstrate the flow
    dates = pd.date_range("2024-08-26", "2025-07-09", freq="7D")
    sampling_df = pd.DataFrame({"DATE": dates, "CAGE_NUMBER": 2, "SAMP_NUM_FISH": np.linspace(7290, 2000, len(dates)).astype(int), "ABW_G": np.linspace(11.9, 350, len(dates))})
    daily = pd.date_range("2024-08-26", "2025-07-09", freq="D")
    feeding_df = pd.DataFrame({"DATE": daily, "CAGE_NUMBER": 2, "FEED AMT KG": np.random.uniform(5, 12, len(daily))})
    feeding_df = feeding_df.rename(columns={"FEED AMT KG": "FEED AMT KG"})  # keep awkward name to demonstrate robust parsing
    harvest_df = pd.DataFrame({"DATE": [pd.to_datetime("2025-07-09")], "CAGE": [2], "TOTAL WEIGHT  [kg]": [500.0], "NUMBER OF FISH": [2000]})
    transfer_df = pd.DataFrame(columns=["DATE", "SOURCE CAGE", "DESTINATION CAGE", "Total weight [g]", "NUMBER OF FISH"])
    # no transfers in demo

# If real files or example are ready, proceed
if (feeding_file and harvest_file and sampling_file and transfer_file) or use_example:
    # When read from real files we already loaded; if example, variables exist above
    if not use_example:
        feeding_df, harvest_df, sampling_df, transfer_df = clean_and_prepare(*load_data(feeding_file, harvest_file, sampling_file, transfer_file))
    else:
        # For demo we already created the dfs above; run clean to canonicalize names
        feeding_df, harvest_df, sampling_df, transfer_df = clean_and_prepare(feeding_df, harvest_df, sampling_df, transfer_df)

    # Show missing-values inspection
    st.subheader("Missing values check")
    miss = inspect_missing_values({
        "Feeding": feeding_df,
        "Harvest": harvest_df,
        "Sampling": sampling_df,
        "Transfer": transfer_df
    })
    for k, v in miss.items():
        st.write(f"**{k}**")
        st.dataframe(v.to_frame(name="missing_count"))

    # Compute cage 2 summary (the main cage)
    st.subheader("Computing Cage 2 production summary (includes stocking & transfers)")
    summary_c2, daily_feed_c2, harvest_by_date_c2, transfers_by_date_c2 = compute_production_summary_for_cage(
        feeding_df, harvest_df, sampling_df, transfer_df,
        cage_number=2,
        stocking_date=pd.to_datetime("2024-08-26"),
        stocked_fish=7290,
        initial_abw_g=11.9,
        start_date=pd.to_datetime("2024-07-16"),
        end_date=pd.to_datetime("2025-07-09")
    )

    st.write("### Cage 2 summary (per sampling point)")
    # Format numeric columns for display
    display_df = summary_c2.copy()
    num_cols = ["BIOMASS_KG", "CUM_FEED_KG", "CUM_REMOVED_KG", "TOTAL_BIOMASS_PRODUCED_KG", "AGGREGATED_eFCR", "PERIOD_FEED_KG", "PERIOD_BIOMASS_PRODUCED_KG", "PERIOD_eFCR"]
    for c in num_cols:
        if c in display_df.columns:
            display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round(3)
    st.dataframe(display_df)

    # Create mock cages (3..7) using cage2 baseline
    st.subheader("Generating mock cages (3 → 7) based on Cage 2")
    mock_results = create_mock_cages_from_cage2(summary_c2, daily_feed_c2, harvest_by_date_c2, num_mocks=5, base_cage_id=3)

    # Build a dict of all cages for selection
    all_cages = {}
    all_cages[2] = {"summary": summary_c2, "daily_feed": daily_feed_c2, "harvest": harvest_by_date_c2}
    for cid, v in mock_results.items():
        all_cages[cid] = v

    # Sidebar selectors
    st.sidebar.header("Visualization options")
    selected_cage = st.sidebar.selectbox("Select Cage", sorted(list(all_cages.keys())))
    selected_kpi = st.sidebar.selectbox("Select KPI", ["Growth", "eFCR"])

    # Get data for selected cage
    sel = all_cages[selected_cage]
    df_summary = sel["summary"].copy().sort_values("DATE")
    df_daily_feed = sel["daily_feed"].copy().sort_values("DATE")

    # Growth graph
    st.subheader(f"Cage {selected_cage} — KPI: {selected_kpi}")
    if selected_kpi == "Growth":
        df_plot = df_summary.dropna(subset=["BIOMASS_KG"])
        if df_plot.empty:
            st.warning("No biomass/growth data available for this cage.")
        else:
            fig = px.line(df_plot, x="DATE", y="BIOMASS_KG", markers=True,
                          title=f"Cage {selected_cage} — Growth (Total biomass at sampling points)",
                          labels={"BIOMASS_KG": "Biomass (kg)"})
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Combined aggregated + period eFCR
        df_plot = df_summary.copy()
        if df_plot.empty or ("AGGREGATED_eFCR" not in df_plot.columns and "PERIOD_eFCR" not in df_plot.columns):
            st.warning("No eFCR data available for this cage.")
        else:
            fig = px.line(df_plot, x="DATE", y="AGGREGATED_eFCR", markers=True, title=f"Cage {selected_cage} — eFCR (aggregated & period)", labels={"AGGREGATED_eFCR": "eFCR"})
            fig.add_scatter(x=df_plot["DATE"], y=df_plot["PERIOD_eFCR"], mode="lines+markers", name="Period eFCR")
            fig.update_layout(yaxis_title="eFCR")
            st.plotly_chart(fig, use_container_width=True)

    # Provide downloadable summary CSVs
    st.subheader("Download data")
    st.download_button(label=f"Download Cage {selected_cage} production summary (CSV)", data=df_summary.to_csv(index=False).encode("utf-8"), file_name=f"cage_{selected_cage}_production_summary.csv", mime="text/csv")
    st.download_button(label=f"Download Cage {selected_cage} daily feed (CSV)", data=df_daily_feed.to_csv(index=False).encode("utf-8"), file_name=f"cage_{selected_cage}_daily_feed.csv", mime="text/csv")

    st.info("Notes: aggregated eFCR = cumulative_feed / total_biomass_produced_since_stocking. "
            "Period eFCR = feed in period / biomass produced in period (including fish removed via harvest/transfer). "
            "If denominators are zero or negative, eFCR is left as NaN.")
