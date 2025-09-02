# ==============================
# Import libraries
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Fish Cage Production Analysis", layout="wide", page_icon="🐟")

# ==============================
# Column Normalization utilities
# ==============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize messy Excel column headers"""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip()).upper() for c in df.columns]
    return df

def to_int_cage(series: pd.Series) -> pd.Series:
    """Extract integer cage numbers from mixed inputs"""
    def _coerce(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None
    return series.apply(_coerce)

def to_number(x):
    """Convert messy numeric strings to float"""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group()) if m else np.nan

def find_col(df: pd.DataFrame, candidates, fuzzy_hint: str | None = None) -> str | None:
    """Find the right column among messy headers"""
    lut = {c.upper(): c for c in df.columns}
    # direct match first
    for name in candidates:
        if name.upper() in lut:
            return lut[name.upper()]
    # fuzzy fallback
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

    # Coerce cage numbers where present
    if "CAGE NUMBER" in feeding.columns:
        feeding["CAGE NUMBER"] = to_int_cage(feeding["CAGE NUMBER"])
    if "CAGE NUMBER" in sampling.columns:
        sampling["CAGE NUMBER"] = to_int_cage(sampling["CAGE NUMBER"])
    if "CAGE NUMBER" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE NUMBER"])
    elif "CAGE" in harvest.columns:
        harvest["CAGE NUMBER"] = to_int_cage(harvest["CAGE"])

    # Clean transfers file
    if transfers is not None:
        for col in ["ORIGIN CAGE","DESTINATION CAGE"]:
            if col in transfers.columns:
                transfers[col] = to_int_cage(transfers[col])
        # standardize weight column
        wcol = find_col(transfers, 
                        ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]","WEIGHT (KG)"], 
                        fuzzy_hint="WEIGHT")
        if wcol and wcol != "TOTAL WEIGHT [KG]":
            transfers.rename(columns={wcol: "TOTAL WEIGHT [KG]"}, inplace=True)

    # Parse date columns safely
    for df in [feeding, harvest, sampling] + ([transfers] if transfers is not None else []):
        if df is not None and "DATE" in df.columns:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    return feeding, harvest, sampling, transfers

#preprocess_cage2
def preprocess_cage2(feeding, harvest, sampling, transfers=None):
    """
    Preprocess Cage 2 timeline (or any cage) with consistent columns:
    ABW_G, FISH_ALIVE, STOCKED, HARV_FISH_CUM, HARV_KG_CUM, IN/OUT cumulatives.
    Ensures timeline starts with actual stocking (fallback 7290 fish @ 11.9g)
    and includes final harvest date.
    Returns (feeding_c2, harvest_c2, base) where `base` is the sampling timeline
    augmented with stocking + cumulatives.
    """
    cage_number = 2
    start_date = pd.to_datetime("2024-08-26")
    end_date   = pd.to_datetime("2025-07-09")

    # local helper to clip and sort by DATE
    def _clip(df):
        if df is None or df.empty or "DATE" not in df.columns:
            return pd.DataFrame()
        out = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
        return out[(out["DATE"] >= start_date) & (out["DATE"] <= end_date)].reset_index(drop=True)

    # Filter files to cage 2 (if cage column exists) and clip to window
    feeding_c2  = _clip(feeding[feeding.get("CAGE NUMBER", -1) == cage_number])   if ("CAGE NUMBER" in feeding.columns) else _clip(feeding)
    harvest_c2  = _clip(harvest[harvest.get("CAGE NUMBER", -1) == cage_number])   if ("CAGE NUMBER" in harvest.columns) else _clip(harvest)
    sampling_c2 = _clip(sampling[sampling.get("CAGE NUMBER", -1) == cage_number]) if ("CAGE NUMBER" in sampling.columns) else _clip(sampling)

    # Ensure sampling_c2 is a DataFrame with DATE column (may be empty)
    if sampling_c2 is None:
        sampling_c2 = pd.DataFrame(columns=["DATE"])

    # --- Standardize ABW column in sampling_c2 to "ABW_G" ---
    abw_candidates = ["AVERAGE_BODY_WEIGHT(G)", "AVERAGE BODY WEIGHT(G)", "AVERAGE BODY WEIGHT (G)",
                      "ABW(G)", "ABW_G", "ABW", "ABW [G]"]
    abw_col_found = None
    for col in abw_candidates:
        if col in sampling_c2.columns:
            abw_col_found = col
            break

    if abw_col_found:
        sampling_c2 = sampling_c2.copy()
        sampling_c2["ABW_G"] = pd.to_numeric(sampling_c2[abw_col_found].map(to_number), errors="coerce")
    else:
        # create ABW_G column (will be filled by stocking row / future merges)
        sampling_c2 = sampling_c2.copy()
        sampling_c2["ABW_G"] = np.nan

    # ----------------------
    # Stocking info (from transfers if available)
    # ----------------------
    stocked_fish = 7290
    initial_abw_g = 11.9
    first_inbound_idx = None
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        # some files may call destination "DESTINATION CAGE" or "DEST_CAGE" etc
        dest_col = None
        for c in ["DESTINATION CAGE", "DEST_CAGE", "DESTINATION", "TO CAGE"]:
            if c in t.columns:
                dest_col = c
                break
        if dest_col is not None:
            t_in = t[t[dest_col] == cage_number].sort_values("DATE").reset_index(drop=False)
            if not t_in.empty:
                # keep original dataframe index so we can identify and drop the exact transfer row later
                first = t_in.iloc[0]
                first_inbound_idx = first.name  # original index in t
                # number of fish column candidates
                if "NUMBER OF FISH" in t_in.columns and pd.notna(first.get("NUMBER OF FISH")):
                    try:
                        stocked_fish = int(float(first["NUMBER OF FISH"]))
                    except Exception:
                        pass
                # total weight col candidates
                wcol = find_col(t_in, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]", "WEIGHT"], fuzzy_hint="WEIGHT")
                if wcol and pd.notna(first.get(wcol)) and stocked_fish:
                    try:
                        initial_abw_g = float(first[wcol]) * 1000.0 / stocked_fish
                    except Exception:
                        pass

    # ----------------------
    # Build stocking row and combine with sampling timeline
    # ----------------------
    stocking_row = pd.DataFrame([{
        "DATE": start_date,
        "CAGE NUMBER": cage_number,
        "ABW_G": initial_abw_g,
        "STOCKED": stocked_fish
    }])

    # combine; ensure DATE ordering and remove duplicates on DATE by keeping first occurrence from sampling after stocking
    base = pd.concat([stocking_row, sampling_c2], ignore_index=True, sort=False)
    base = base.drop_duplicates(subset=["DATE"], keep="first").sort_values("DATE").reset_index(drop=True)
    base["STOCKED"] = stocked_fish

    # If sampling had no rows, ensure there is at least stocking + end_date rows
    if base.empty:
        base = stocking_row.copy()

    # Ensure final harvest/date exists in base (so plots show the endpoint)
    if not harvest_c2.empty:
        final_h_date = harvest_c2["DATE"].max()
    else:
        final_h_date = pd.NaT

    # If we have a final harvest date within window, try to compute ABW from harvest if possible and insert/update row
    if pd.notna(final_h_date):
        # Try to compute ABW from harvest totals on that date
        hh = harvest_c2[harvest_c2["DATE"] == final_h_date].copy()
        h_fish_col = find_col(hh, ["NUMBER OF FISH", "N_FISH"], "FISH")
        h_kg_col   = find_col(hh, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT [KG]"], "WEIGHT")
        abw_final = np.nan
        if h_fish_col and h_kg_col and hh[h_fish_col].notna().any() and hh[h_kg_col].notna().any():
            tot_fish = pd.to_numeric(hh[h_fish_col], errors="coerce").fillna(0).sum()
            tot_kg   = pd.to_numeric(hh[h_kg_col], errors="coerce").fillna(0).sum()
            if tot_fish > 0 and tot_kg > 0:
                abw_final = (tot_kg * 1000.0) / tot_fish
        # fallback: maybe harvest rows already have ABW column
        if np.isnan(abw_final):
            abw_col_h = find_col(hh, ["ABW(G)", "ABW_G", "ABW", "AVERAGE_BODY_WEIGHT(G)"], "ABW")
            if abw_col_h and hh[abw_col_h].notna().any():
                abw_final = pd.to_numeric(hh[abw_col_h].map(to_number), errors="coerce").mean()

        if pd.notna(abw_final):
            # if base already has final_h_date row, set ABW_G there; else append a row
            if (base["DATE"] == final_h_date).any():
                base.loc[base["DATE"] == final_h_date, "ABW_G"] = abw_final
            else:
                add = pd.DataFrame([{
                    "DATE": final_h_date,
                    "CAGE NUMBER": cage_number,
                    "ABW_G": abw_final,
                    "STOCKED": stocked_fish
                }])
                base = pd.concat([base, add], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # Ensure end_date is present (if not, append using last known ABW)
    if base["DATE"].max() < end_date:
        last_abw = base["ABW_G"].dropna().iloc[-1] if base["ABW_G"].notna().any() else initial_abw_g
        add_end = pd.DataFrame([{
            "DATE": end_date,
            "CAGE NUMBER": cage_number,
            "ABW_G": last_abw,
            "STOCKED": stocked_fish
        }])
        base = pd.concat([base, add_end], ignore_index=True).sort_values("DATE").reset_index(drop=True)

    # ----------------------
    # Initialize cumulative columns on base
    # ----------------------
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        base[col] = 0.0

    # ----------------------
    # Harvest cumulatives aligned to sampling (base)
    # ----------------------
    if not harvest_c2.empty:
        h_fish_col = find_col(harvest_c2, ["NUMBER OF FISH", "N_FISH"], "FISH")
        h_kg_col   = find_col(harvest_c2, ["TOTAL WEIGHT [KG]", "TOTAL WEIGHT (KG)", "WEIGHT"], "WEIGHT")
        h = harvest_c2.copy().sort_values("DATE").reset_index(drop=True)
        if h_fish_col:
            h["H_FISH"] = pd.to_numeric(h[h_fish_col], errors="coerce").fillna(0)
        else:
            h["H_FISH"] = 0.0
        if h_kg_col:
            h["H_KG"] = pd.to_numeric(h[h_kg_col], errors="coerce").fillna(0)
        else:
            h["H_KG"] = 0.0
        h["HARV_FISH_CUM"] = h["H_FISH"].cumsum()
        h["HARV_KG_CUM"] = h["H_KG"].cumsum()
        # align to base dates
        mh = pd.merge_asof(base[["DATE"]].sort_values("DATE"), h[["DATE","HARV_FISH_CUM","HARV_KG_CUM"]].sort_values("DATE"),
                           on="DATE", direction="backward")
        base["HARV_FISH_CUM"] = mh["HARV_FISH_CUM"].fillna(0)
        base["HARV_KG_CUM"]   = mh["HARV_KG_CUM"].fillna(0)

    # ----------------------
    # Transfers cumulatives aligned to sampling (base)
    # ----------------------
    if transfers is not None and not transfers.empty:
        t = _clip(transfers)
        # If we detected an inbound stocking transfer, drop that exact row so it's not double-counted
        if first_inbound_idx is not None and first_inbound_idx in t.index:
            t = t.drop(index=first_inbound_idx)

        # Normalize origin/dest cage text columns if present
        origin_col = find_col(t, ["ORIGIN CAGE", "ORIGIN", "FROM CAGE"], "ORIGIN") if not t.empty else None
        dest_col   = find_col(t, ["DESTINATION CAGE", "DESTINATION", "TO CAGE"], "DEST")   if not t.empty else None
        fish_col   = find_col(t, ["NUMBER OF FISH", "N_FISH"], "FISH") if not t.empty else None
        kg_col     = find_col(t, ["TOTAL WEIGHT [KG]","TOTAL WEIGHT (KG)","WEIGHT [KG]","WEIGHT"], "WEIGHT") if not t.empty else None

        # Ensure numeric fields exist
        t = t.copy()
        t["T_FISH"] = pd.to_numeric(t[fish_col], errors="coerce").fillna(0) if fish_col else 0.0
        t["T_KG"]   = pd.to_numeric(t[kg_col], errors="coerce").fillna(0) if kg_col else 0.0

        # convert origin/dest to integers (best-effort)
        def _cage_to_int(val):
            try:
                if pd.isna(val):
                    return np.nan
                m = re.search(r"(\d+)", str(val))
                return int(m.group(1)) if m else np.nan
            except Exception:
                return np.nan

        if origin_col in t.columns:
            t["ORIGIN_INT"] = t[origin_col].apply(_cage_to_int)
        else:
            t["ORIGIN_INT"] = np.nan
        if dest_col in t.columns:
            t["DEST_INT"] = t[dest_col].apply(_cage_to_int)
        else:
            t["DEST_INT"] = np.nan

        # Outgoing cumulative (from this cage)
        tout = t[t["ORIGIN_INT"] == cage_number].sort_values("DATE").reset_index(drop=True)
        if not tout.empty:
            tout["OUT_FISH_CUM"] = tout["T_FISH"].cumsum()
            tout["OUT_KG_CUM"]   = tout["T_KG"].cumsum()
            mo = pd.merge_asof(base[["DATE"]].sort_values("DATE"), tout[["DATE","OUT_FISH_CUM","OUT_KG_CUM"]].sort_values("DATE"),
                               on="DATE", direction="backward")
            base["OUT_FISH_CUM"] = mo["OUT_FISH_CUM"].fillna(0)
            base["OUT_KG_CUM"]   = mo["OUT_KG_CUM"].fillna(0)

        # Incoming cumulative (into this cage)
        tin = t[t["DEST_INT"] == cage_number].sort_values("DATE").reset_index(drop=True)
        if not tin.empty:
            tin["IN_FISH_CUM"] = tin["T_FISH"].cumsum()
            tin["IN_KG_CUM"]   = tin["T_KG"].cumsum()
            mi = pd.merge_asof(base[["DATE"]].sort_values("DATE"), tin[["DATE","IN_FISH_CUM","IN_KG_CUM"]].sort_values("DATE"),
                               on="DATE", direction="backward")
            base["IN_FISH_CUM"] = mi["IN_FISH_CUM"].fillna(0)
            base["IN_KG_CUM"]   = mi["IN_KG_CUM"].fillna(0)

    # ----------------------
    # Standing fish alive & integer number column
    # ----------------------
    # expected stock at each date (stocked minus cum harvest plus incoming minus outgoing)
    base["FISH_ALIVE"] = (base.get("STOCKED", 0)
                         - base.get("HARV_FISH_CUM", 0)
                         + base.get("IN_FISH_CUM", 0)
                         - base.get("OUT_FISH_CUM", 0)).clip(lower=0)
    # create integer column
    base["NUMBER OF FISH"] = base["FISH_ALIVE"].fillna(0).astype(int)

    # final housekeeping: ensure ABW_G column present and numeric
    if "ABW_G" not in base.columns:
        base["ABW_G"] = np.nan
    base["ABW_G"] = pd.to_numeric(base["ABW_G"], errors="coerce")

    # Ensure all cumulatives are numeric and have zeros where missing
    for col in ["HARV_FISH_CUM","HARV_KG_CUM","IN_FISH_CUM","IN_KG_CUM","OUT_FISH_CUM","OUT_KG_CUM"]:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    # reset index and return
    base = base.sort_values("DATE").reset_index(drop=True)

    return feeding_c2, harvest_c2, base

#Compute summary
def compute_metrics(stocking_date, end_date, initial_stock, sampling_data, feeding_data, transfer_data, harvest_data):
    """
    Compute cage production metrics:
    - ABW, Stock, Biomass
    - Feed, Harvest, Transfers
    - Growth, Period eFCR, Aggregated eFCR
    Consistent with preprocess_cage2 output.
    """

    # ----------------------
    # Prepare full timeline
    # ----------------------
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

    # ----------------------
    # Populate feeding
    # ----------------------
    if feeding_data is not None and not feeding_data.empty:
        feed_col = find_col(feeding_data, ["FEED (KG)", "FEED_KG", "FEED"], "FEED")
        for _, row in feeding_data.iterrows():
            if pd.notna(row["DATE"]):
                df.loc[df["DATE"] == row["DATE"], "FEED_KG"] += row.get(feed_col, 0)

    # ----------------------
    # Populate transfers (biomass basis)
    # ----------------------
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

    # ----------------------
    # Populate harvests
    # ----------------------
    if harvest_data is not None and not harvest_data.empty:
        for _, row in harvest_data.iterrows():
            d = row["DATE"]
            fish = row.get("NUMBER OF FISH", np.nan)
            kg   = row.get("TOTAL WEIGHT [KG]", np.nan)
            df.loc[df["DATE"] == d, "HARV_KG"] += kg if pd.notna(kg) else 0
            if pd.notna(fish):
                df.loc[df["DATE"] >= d, "FISH_ALIVE"] -= fish

    # ----------------------
    # Sampling → ABW
    # ----------------------
    if sampling_data is not None and not sampling_data.empty:
        for _, row in sampling_data.iterrows():
            df.loc[df["DATE"] == row["DATE"], "ABW_G"] = row.get("ABW_G", np.nan)

    df["ABW_G"] = df["ABW_G"].ffill()
    df["FISH_ALIVE"] = df["FISH_ALIVE"].ffill()

    # ----------------------
    # Biomass
    # ----------------------
    df["BIOMASS_KG"] = df["FISH_ALIVE"] * df["ABW_G"] / 1000

    # ----------------------
    # Growth & eFCR
    # ----------------------
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

        # ----------------------
        # Feeding
        # ----------------------
        f = feeding_c2.copy()
        f["CAGE NUMBER"] = cage
        feed_col = find_col(f, ["FEED_KG", "FEED (KG)", "FEED"], "FEED_KG")
        if feed_col:
            f["FEED_KG"] = pd.to_numeric(f[feed_col], errors="coerce").fillna(0)
            f["FEED_KG"] *= np.random.uniform(0.9, 1.1, size=len(f))
        mock_feeding.append(f)

        # ----------------------
        # Sampling
        # ----------------------
        s = sampling_c2.copy()
        s["CAGE NUMBER"] = cage
        s["ABW_G"] *= np.random.uniform(0.95, 1.05, size=len(s))

        if "FISH_ALIVE" in s.columns:
            s["FISH_ALIVE"] = pd.to_numeric(s["FISH_ALIVE"], errors="coerce").ffill().fillna(0)

        s["BIOMASS_KG"] = s["FISH_ALIVE"] * s["ABW_G"] / 1000
        mock_sampling.append(s)

        # ----------------------
        # Harvest
        # ----------------------
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

        # ----------------------
        # Summary using compute_metrics
        # ----------------------
        summary = compute_metrics(
            stocking_date=s["DATE"].min(),
            end_date=s["DATE"].max(),
            initial_stock=s["FISH_ALIVE"].iloc[0],
            sampling_data=s[["DATE", "ABW_G"]],
            feeding_data=f[["DATE", "FEED_KG"]],
            transfer_data=None,   # mock cages don’t have transfers by default
            harvest_data=h[["DATE", fish_col, kg_col]] if (fish_col and kg_col) else h[["DATE"]]
        )

        mock_summaries[cage] = summary

    return mock_feeding, mock_sampling, mock_harvest, mock_summaries

# ===========================
# Streamlit UI – Cage Selection + KPI (Aligned with compute_metrics)
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

    # Run metrics for Cage 2
    summary_c2 = compute_metrics(
        stocking_date=sampling_c2["DATE"].min(),
        end_date=sampling_c2["DATE"].max(),
        initial_stock=sampling_c2["FISH_ALIVE"].iloc[0],
        sampling_data=sampling_c2[["DATE", "ABW_G"]].rename(columns={"DATE": "Date", "ABW_G": "ABW"}),
        feeding_data=feeding_c2[["DATE", "FEED_KG"]].rename(columns={"DATE": "Date", "FEED_KG": "Feed"}),
        transfer_data=transfers if transfers is not None else None,
        harvest_data=harvest_c2.rename(columns={"DATE": "Date"})
    )

    # Generate mock cages (3–7)
    mock_feeding, mock_sampling, mock_harvest, mock_summaries = generate_mock_cages(
        feeding_c2, sampling_c2, harvest_c2
    )

    # Combine all summaries
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
        total_biomass = summary_df["Biomass"].max()
        st.markdown(
            f'<div class="kpi-card"><h3>Total Biomass</h3><p>{total_biomass:,.2f} kg</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        avg_abw = summary_df["ABW"].mean()
        st.markdown(
            f'<div class="kpi-card"><h3>Average ABW</h3><p>{avg_abw:,.2f} g</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        avg_efcr = summary_df["AGGREGATED_eFCR"].dropna().mean()
        st.markdown(
            f'<div class="kpi-card"><h3>Average eFCR</h3><p>{avg_efcr:.2f}</p></div>',
            unsafe_allow_html=True,
        )

    # ===========================
    # Data Table
    # ===========================
    show_cols = [
        "Date", "Stock", "ABW", "Biomass", "Feed",
        "Transfer_In", "Transfer_Out", "Harvest",
        "GROWTH_KG", "PERIOD_eFCR", "AGGREGATED_eFCR",
    ]
    display_summary = summary_df[[c for c in show_cols if c in summary_df.columns]]
    st.dataframe(display_summary, use_container_width=True)

    # Period info
    st.write(f"**Records:** {len(display_summary)} rows")
    st.write(f"**From:** {display_summary['Date'].min().strftime('%d %b %Y')} "
             f"**To:** {display_summary['Date'].max().strftime('%d %b %Y')}")

    # ===========================
    # KPI Plots
    # ===========================
    if selected_kpi == "Biomass":
        fig = px.line(
            summary_df.dropna(subset=["Biomass"]),
            x="Date", y="Biomass", markers=True,
            title=f"Cage {selected_cage}: Biomass Over Time",
            labels={"Biomass": "Total Biomass (kg)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif selected_kpi == "ABW":
        fig = px.line(
            summary_df.dropna(subset=["ABW"]),
            x="Date", y="ABW", markers=True,
            title=f"Cage {selected_cage}: Average Body Weight (ABW) Over Time",
            labels={"ABW": "ABW (g)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # eFCR
        dff = summary_df.dropna(subset=["AGGREGATED_eFCR", "PERIOD_eFCR"])
        fig = px.line(
            dff, x="Date", y="AGGREGATED_eFCR", markers=True,
            title=f"Cage {selected_cage}: eFCR Over Time",
            labels={"AGGREGATED_eFCR": "Aggregated eFCR"}
        )
        fig.update_traces(name="Aggregated eFCR")
        fig.add_scatter(
            x=dff["Date"], y=dff["PERIOD_eFCR"],
            mode="lines+markers", name="Period eFCR",
            line=dict(dash="dash")
        )
        fig.update_layout(yaxis_title="eFCR", legend_title_text="Legend")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("⬅️ Upload the Excel files to begin.")


