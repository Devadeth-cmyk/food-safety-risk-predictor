"""Build leakage-aware labels and point-in-time features for next routine inspection failure.

Target: failure result at the next Canvass inspection after the current inspection event,
including intervening complaint/re-inspection events. Rows with a same-day target or a
next-routine-inspection gap greater than MAX_TARGET_GAP_DAYS are excluded from training.
The latest event for every restaurant is separately exported for risk scoring.
"""
from pathlib import Path
import re
from collections import Counter
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CANDIDATE_PATHS = [
    BASE_DIR / "restaurant_feature_engineering_base.csv",
    BASE_DIR / "data" / "restaurant_feature_engineering_base.csv",
]
RAW_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])
TARGET_PATH = OUTPUT_DIR / "restaurant_target_prepared.csv"
SCORING_PATH = OUTPUT_DIR / "restaurant_scoring_latest.csv"

# 18 months, as recommended for a meaningful next-routine prediction horizon.
MIN_TARGET_GAP_DAYS = 1
MAX_TARGET_GAP_DAYS = 548

required = [
    "Restaurant_ID", "Inspection Date", "Inspection_Sequence",
    "Inspection Type", "Results",
]

if not RAW_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {RAW_PATH}\nPut restaurant_feature_engineering_base.csv "
        "next to target.py or inside a data/ subfolder."
    )

print(f"Loading: {RAW_PATH}")
df = pd.read_csv(RAW_PATH, low_memory=False)
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse fields and validate the known 0/1 result encoding (1 = Fail).
df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], errors="coerce")
df["Inspection_Sequence"] = pd.to_numeric(df["Inspection_Sequence"], errors="coerce")
df["Results"] = pd.to_numeric(df["Results"], errors="coerce")
df = df.dropna(subset=required).copy()
df["Inspection_Sequence"] = df["Inspection_Sequence"].astype(int)
df["Results"] = df["Results"].astype(int)
if not set(df["Results"].unique()).issubset({0, 1}):
    raise ValueError("Results must be binary, with 1 = Fail and 0 = non-failure.")

# Remove exact duplicate records only. Do not collapse distinct same-day events.
rows_before_dedup = len(df)
df = df.drop_duplicates().copy()
print(f"Exact duplicate rows removed: {rows_before_dedup - len(df):,}")

# Keep event order. Sequence is the main within-restaurant ordering key.
df = df.sort_values(
    ["Restaurant_ID", "Inspection_Sequence", "Inspection Date"], kind="mergesort"
).reset_index(drop=True)

# Audit repeated restaurant/date pairs. They are reported, not blindly removed.
duplicate_pairs = int(df.duplicated(["Restaurant_ID", "Inspection Date"], keep=False).sum())
print(f"Rows participating in repeated restaurant/date pairs: {duplicate_pairs:,}")

# Normalize inspection type without changing the source value.
type_norm = df["Inspection Type"].astype("string").str.strip().str.casefold()
is_canvass = type_norm.eq("canvass").fillna(False).to_numpy()

# ------------------------- Point-in-time features -------------------------
g = df.groupby("Restaurant_ID", sort=False)
df["Past_Inspection_Count"] = g.cumcount()
df["Prev_Inspection_Result"] = g["Results"].shift(1)
df["Past_Failure_Count"] = g["Results"].cumsum() - df["Results"]
df["Past_Failure_Rate"] = (
    df["Past_Failure_Count"] / df["Past_Inspection_Count"].replace(0, np.nan)
)

def prior_rolling_mean(series, window=3):
    return series.groupby(df["Restaurant_ID"], sort=False).transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

def prior_rolling_sum(series, window=3):
    return series.groupby(df["Restaurant_ID"], sort=False).transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    )

for col, out_col in [
    ("Results", "Failures_Last_3_Available"),
    ("Violation_Count", "Mean_Violations_Last_3"),
    ("Hazard_Count", "Mean_Hazards_Last_3"),
    ("Average_Violation_Score", "Mean_Violation_Score_Last_3"),
]:
    if col in df.columns:
        if col == "Results":
            df[out_col] = prior_rolling_sum(df[col], 3).fillna(0)
        else:
            df[out_col] = prior_rolling_mean(df[col], 3)

# Days since most recent failure, including the current result because it is known
# after the current inspection event.
last_failure_date = df["Inspection Date"].where(df["Results"].eq(1))
last_failure_date = last_failure_date.groupby(df["Restaurant_ID"], sort=False).ffill()
df["Days_Since_Last_Failure"] = (df["Inspection Date"] - last_failure_date).dt.days

# Canvass-only history through the current event, including the current event if it is Canvass.
canvass_result = df["Results"].where(is_canvass)
canvass_seen = pd.Series(is_canvass.astype(int), index=df.index)
canvass_fail = pd.Series((is_canvass & df["Results"].eq(1).to_numpy()).astype(int), index=df.index)
canvass_cum_count = canvass_seen.groupby(df["Restaurant_ID"], sort=False).cumsum()
canvass_cum_fail = canvass_fail.groupby(df["Restaurant_ID"], sort=False).cumsum()
df["Canvass_Failure_Rate_To_Date"] = (
    canvass_cum_fail / canvass_cum_count.replace(0, np.nan)
)
df["Last_Canvass_Failed"] = canvass_result.groupby(df["Restaurant_ID"], sort=False).ffill()
last_canvass_date = df["Inspection Date"].where(is_canvass)
last_canvass_date = last_canvass_date.groupby(df["Restaurant_ID"], sort=False).ffill()
df["Days_Since_Last_Canvass"] = (df["Inspection Date"] - last_canvass_date).dt.days

df["Month_Sin"] = np.sin(2 * np.pi * df["Inspection Date"].dt.month / 12)
df["Month_Cos"] = np.cos(2 * np.pi * df["Inspection Date"].dt.month / 12)

# Parse current inspection violation codes from the text. Codes are features of the
# current inspection, which is already known at the time this row is scored.
# The code-count flags are generic code indicators, not a claim that every code has
# identical severity across all historical rule eras.
if "Violations" in df.columns:
    code_pattern = re.compile(r"(?<!\d)(\d{1,2})\.\s+[A-Z]")
    def extract_codes(value):
        if not isinstance(value, str) or not value.strip():
            return set()
        return {int(x) for x in code_pattern.findall(value) if 1 <= int(x) <= 70}
    codes = df["Violations"].map(extract_codes)
    code_cols = [f"Current_Violation_Code_{code:02d}" for code in range(1, 71)]
    # Build all current-code flags in one pass rather than 70 separate Python maps.
    flag_matrix = np.zeros((len(df), 70), dtype=np.int8)
    for row_idx, row_codes in enumerate(codes.array):
        for code in row_codes:
            flag_matrix[row_idx, code - 1] = 1
    df[code_cols] = pd.DataFrame(flag_matrix, index=df.index, columns=code_cols)
    df["Current_Violation_Code_Count"] = codes.map(len).astype("int16")

    # Choose common code histories using only pre-test-era inspection text. The
    # current flags for all 70 codes are retained; history is limited to the 20
    # most frequent pre-test codes to keep memory use practical on an 8 GB laptop.
    cutoff_for_code_selection = pd.Timestamp("2022-08-04")
    pretest_mask = df["Inspection Date"] < cutoff_for_code_selection
    code_frequency = Counter(
        code for row_codes in codes.loc[pretest_mask].array for code in row_codes
    )
    common_codes = [code for code, _ in code_frequency.most_common(20)]
    print("Historical code features for common codes:", common_codes)

    # Build historical frequency features only for selected common codes. For every
    # row, both cumulative counts and last-3 counts exclude the current inspection.
    hist_width = len(common_codes)
    past_code_counts = np.zeros((len(df), hist_width), dtype=np.int32)
    code_counts_last3 = np.zeros((len(df), hist_width), dtype=np.int8)
    selected_indices = [code - 1 for code in common_codes]
    for positions in df.groupby("Restaurant_ID", sort=False).indices.values():
        positions = np.asarray(positions, dtype=np.int64)
        values = flag_matrix[positions][:, selected_indices].astype(np.int32, copy=False)
        prefix = np.vstack([np.zeros((1, hist_width), dtype=np.int32), np.cumsum(values, axis=0)])
        local_rows = np.arange(len(positions))
        past_code_counts[positions] = prefix[local_rows]
        starts = np.maximum(local_rows - 3, 0)
        code_counts_last3[positions] = (prefix[local_rows] - prefix[starts]).astype(np.int8)

    past_cols = [f"Past_Violation_Code_{code:02d}_Count" for code in common_codes]
    recent_cols = [f"Violation_Code_{code:02d}_Last_3" for code in common_codes]
    history_features = pd.concat([
        pd.DataFrame(past_code_counts, index=df.index, columns=past_cols),
        pd.DataFrame(code_counts_last3, index=df.index, columns=recent_cols),
    ], axis=1)
    df = pd.concat([df.copy(), history_features], axis=1)

# ---------------------------- Next Canvass target ----------------------------
# For each row, find the next later row (by event order) whose type is Canvass.
# This allows complaint/re-inspection events between the prediction point and target.
n = len(df)
next_result = np.full(n, np.nan, dtype=float)
next_sequence = np.full(n, np.nan, dtype=float)
next_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
restaurant_groups = df.groupby("Restaurant_ID", sort=False).indices
results_array = df["Results"].to_numpy()
sequence_array = df["Inspection_Sequence"].to_numpy(dtype=float)
date_array = df["Inspection Date"].to_numpy(dtype="datetime64[ns]")

for positions in restaurant_groups.values():
    positions = np.asarray(positions, dtype=np.int64)
    routine_positions = positions[is_canvass[positions]]
    if len(routine_positions) == 0:
        continue
    # side='right' excludes the current event when it is itself a Canvass.
    next_slot = np.searchsorted(routine_positions, positions, side="right")
    valid = next_slot < len(routine_positions)
    source_positions = positions[valid]
    target_positions = routine_positions[next_slot[valid]]
    next_result[source_positions] = results_array[target_positions]
    next_sequence[source_positions] = sequence_array[target_positions]
    next_date[source_positions] = date_array[target_positions]

df["Next_Routine_Result"] = next_result
df["Next_Routine_Sequence"] = next_sequence
df["Next_Routine_Inspection_Date"] = pd.to_datetime(next_date)
df["Target_Gap_Days"] = (
    df["Next_Routine_Inspection_Date"] - df["Inspection Date"]
).dt.days

eligible = (
    df["Next_Routine_Result"].notna()
    & df["Target_Gap_Days"].between(MIN_TARGET_GAP_DAYS, MAX_TARGET_GAP_DAYS)
)
model_df = df.loc[eligible].copy()
model_df["Target_Next_Routine_Failure"] = model_df["Next_Routine_Result"].astype(int)
# Raw free text is preserved in the original source file; engineered code flags are
# used by the model, so omit the large text column from processed outputs.
model_df.drop(columns=["Violations"], errors="ignore", inplace=True)

# Keep all latest restaurant events separately for deployment scoring. These rows do
# not need a future target and must not be discarded just because no next Canvass exists.
scoring_df = df.groupby("Restaurant_ID", sort=False).tail(1).copy()
future_cols = [
    "Next_Routine_Result", "Next_Routine_Sequence",
    "Next_Routine_Inspection_Date", "Target_Gap_Days",
    "Target_Next_Routine_Failure",
]
scoring_df.drop(columns=future_cols + ["Violations"], errors="ignore", inplace=True)

model_df.to_csv(TARGET_PATH, index=False)
scoring_df.to_csv(SCORING_PATH, index=False)

print("\nTARGET BUILD SUMMARY")
print("Source rows:", f"{len(df):,}")
print("Eligible labelled rows:", f"{len(model_df):,}")
print("Restaurants in labelled data:", f"{model_df['Restaurant_ID'].nunique():,}")
print("Latest scoring rows:", f"{len(scoring_df):,}")
print("Same-day/zero-day target rows excluded by horizon:", int((df["Target_Gap_Days"] == 0).sum()))
print("Targets beyond 548 days excluded:", int((df["Target_Gap_Days"] > MAX_TARGET_GAP_DAYS).sum()))
print("Target failure rate:", round(model_df["Target_Next_Routine_Failure"].mean(), 4))
print("Target counts:\n", model_df["Target_Next_Routine_Failure"].value_counts().sort_index())
print("Target gap days (median / 90th percentile):", model_df["Target_Gap_Days"].median(), model_df["Target_Gap_Days"].quantile(0.90))
print("Saved labelled data:", TARGET_PATH)
print("Saved latest scoring set:", SCORING_PATH)
