"""Rebuild restaurant_feature_engineering_base.csv from the RAW Chicago Food Inspections export.

Why: the old base file had every inspection with zero violations removed (Violation_Count
minimum was 1), which biased the target, the history features and the time gaps.
This script keeps every real restaurant inspection and recomputes everything on the full data.

Usage:
    python build_base.py [raw_csv_or_csv.gz] [output_csv]
Defaults: chicago_inspections.csv(.gz) / Food_Inspections*.csv next to this file ->
          restaurant_feature_engineering_base.csv next to this file.

Then run, unchanged apart from the small edits listed in the notes:
    python target.py ; python split_data.py ; python train.py
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "restaurant_feature_engineering_base.csv"

if len(sys.argv) > 1:
    RAW_PATH = Path(sys.argv[1])
else:
    candidates = (
        [BASE_DIR / n for n in ["Food_Inspections_20261015.csv", "chicago_inspections.csv.gz",
                                "chicago_inspections_csv.gz"]]
        + sorted(BASE_DIR.glob("Food_Inspections*.csv"))
    )
    RAW_PATH = next((p for p in candidates if p.exists()), candidates[0])
if not RAW_PATH.exists():
    raise FileNotFoundError(f"Raw Chicago file not found: {RAW_PATH}")

SHRINK = 20  # pseudo-count pulling small-sample rates toward the global rate

# ------------------------------- load & filter -------------------------------
print("Loading:", RAW_PATH)
raw = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw rows: {len(raw):,}")

df = raw[raw["Facility Type"].astype("string").str.strip().eq("Restaurant")].copy()
print(f"Restaurant rows: {len(df):,}")

# Results -> binary. Fail = 1. Pass and Pass w/ Conditions = 0.
# No Entry / Not Ready / Business Not Located: no real inspection happened -> dropped.
# Out of Business: not an inspection outcome we can predict -> dropped.
result_map = {"Fail": 1, "Pass": 0, "Pass w/ Conditions": 0}
df["Results"] = df["Results"].astype("string").str.strip().map(result_map)
dropped_results = int(df["Results"].isna().sum())
df = df.dropna(subset=["Results"]).copy()
df["Results"] = df["Results"].astype(int)
print(f"Rows dropped for non-binary results (No Entry/Not Ready/Out of Business/...): {dropped_results:,}")

df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["Inspection Date"]).copy()

# KEY FIX: keep inspections with no violations (empty Violations text).
df["Violations"] = df["Violations"].fillna("").astype(str)
n_clean = int((df["Violations"].str.strip() == "").sum())
print(f"Inspections with NO violations kept (previously dropped): {n_clean:,}")

# --------------------------------- identifiers ---------------------------------
# Restaurant_ID: License # (matches the old file's ~22.3k restaurants). Rows with a missing
# or zero licence fall back to name + address.
lic = pd.to_numeric(df["License #"], errors="coerce")
fallback = "ADDR:" + df["DBA Name"].astype(str).str.upper().str.strip() + "|" + df["Address"].astype(str).str.upper().str.strip()
key = np.where(lic.fillna(0) > 0, "LIC:" + lic.fillna(0).astype(int).astype(str), fallback)
df["Restaurant_ID"] = pd.factorize(pd.Series(key, index=df.index))[0] + 1

df = df.sort_values(["Restaurant_ID", "Inspection Date", "Inspection ID"], kind="mergesort").reset_index(drop=True)
g = df.groupby("Restaurant_ID", sort=False)
df["Inspection_Sequence"] = g.cumcount() + 1
df["Days_Since_Previous_Inspection"] = g["Inspection Date"].diff().dt.days  # NaN for first inspection
df["Inspection_Year"] = df["Inspection Date"].dt.year
df["Inspection_Month"] = df["Inspection Date"].dt.month
df["Inspection_DayOfWeek"] = df["Inspection Date"].dt.day_name()

# ------------------------------ violation features ------------------------------
# Codes are parsed only at the start of the text or after a "|" separator, so numbers inside
# the free-text comments are never mistaken for codes (validated: 100% match with the old
# Violation_Count on 147k matched rows).

code_pat = re.compile(r"(?:^|\|\s*)(\d{1,3})\.\s")

def parse_codes(text):
    return [int(c) for c in code_pat.findall(text)] if text else []

codes = df["Violations"].map(parse_codes)
df["Violation_Count"] = codes.map(len).astype(int)
# Chicago severity tiers by code: 1-14 priority (critical), 15-29 priority foundation (serious),
# 30+ core (minor).
df["Critical_Count"] = codes.map(lambda c: sum(1 <= x <= 14 for x in c)).astype(int)
df["Serious_Count"] = codes.map(lambda c: sum(15 <= x <= 29 for x in c)).astype(int)
df["Minor_Count"] = codes.map(lambda c: sum(x >= 30 for x in c)).astype(int)
df["Hazard_Count"] = df["Critical_Count"] + df["Serious_Count"]   # critical + serious violations

# ------------------------------ location / risk fields ------------------------------
df["Geo_Grid_ID"] = np.where(
    df["Latitude"].notna() & df["Longitude"].notna(),
    df["Latitude"].round(1).astype(str) + "_" + df["Longitude"].round(1).astype(str),
    "unknown",
)
df["Risk"] = df["Risk"].where(df["Risk"].astype("string").str.startswith("Risk", na=False), "Unknown")

# ---------------- time-aware group failure rates (no future information) ----------------
# For each row, a group's rate uses only inspections on STRICTLY EARLIER dates, shrunk toward
# the global rate. Same-day inspections never see each other.
df["_d"] = df["Inspection Date"]
df["_y"] = df["Results"]
GLOBAL_RATE = float(df["_y"].mean())

def prior_counts(frame, key_col):
    """Return (n, f) = inspections and failures of each row's group on strictly earlier dates."""
    agg = (frame.groupby([key_col, "_d"], sort=True)["_y"]
                .agg(n="size", f="sum").reset_index().sort_values([key_col, "_d"]))
    agg["pn"] = agg.groupby(key_col)["n"].cumsum() - agg["n"]
    agg["pf"] = agg.groupby(key_col)["f"].cumsum() - agg["f"]
    merged = frame[[key_col, "_d"]].merge(agg[[key_col, "_d", "pn", "pf"]], on=[key_col, "_d"], how="left")
    return merged["pn"].to_numpy(dtype=float), merged["pf"].to_numpy(dtype=float)

def shrunk(f, n):
    return (f + SHRINK * GLOBAL_RATE) / (n + SHRINK)

# Zip-code failure rate
df["_zip"] = df["Zip"].astype("Int64").astype(str)
zn, zf = prior_counts(df, "_zip")
df["Zip_Fail_Rate"] = pd.Series(shrunk(zf, zn), index=df.index).fillna(GLOBAL_RATE)  # missing zip -> global rate

# Chain signal: other outlets sharing a normalised business name.
def name_key(s):
    s = str(s).upper()
    s = re.sub(r"#.*$", "", s)                 # drop store numbers such as "#1234"
    s = re.sub(r"[^A-Z ]", " ", s)             # letters only
    s = re.sub(r"\b(INC|LLC|CORP|CO|LTD|THE)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip() or "UNKNOWN"

df["_name"] = df["DBA Name"].map(name_key)
cn, cf = prior_counts(df, "_name")
# the restaurant's own earlier history UNDER THE SAME NAME KEY (names can change over time)
df["_name_rid"] = df["_name"] + "||" + df["Restaurant_ID"].astype(str)
on, of = prior_counts(df, "_name_rid")
other_n, other_f = cn - on, cf - of            # OTHER outlets of the same name, earlier dates only
df["Chain_Other_Past_Inspections"] = other_n
df["Chain_Other_Fail_Rate"] = shrunk(other_f, other_n)

# Outlets of the same name already open (first inspection strictly earlier), excluding itself.
first = df.groupby(["_name", "Restaurant_ID"], sort=False)["_d"].min().reset_index()
first_by_name = {k: np.sort(v["_d"].to_numpy("datetime64[D]")) for k, v in first.groupby("_name")}
outlets = np.zeros(len(df))
for name, idx in df.groupby("_name", sort=False).indices.items():
    dates = df["_d"].to_numpy("datetime64[D]")[idx]
    outlets[idx] = np.searchsorted(first_by_name[name], dates, side="left")
df["Chain_Other_Outlets_Seen"] = np.maximum(outlets - 1, 0)   # minus the restaurant itself

# ----------------------------------- save -----------------------------------
out_cols = [
    "Risk", "Inspection Date", "Inspection Type", "Results", "Violations", "Restaurant_ID",
    "Days_Since_Previous_Inspection", "Inspection_Year", "Inspection_Month", "Inspection_DayOfWeek",
    "Violation_Count", "Critical_Count", "Serious_Count", "Minor_Count", "Hazard_Count",
    "Latitude", "Longitude", "Geo_Grid_ID", "Inspection_Sequence",
    "Zip_Fail_Rate", "Chain_Other_Past_Inspections", "Chain_Other_Fail_Rate", "Chain_Other_Outlets_Seen",
]
out = df[out_cols].copy()
out.to_csv(OUT_PATH, index=False)

print("\nBASE BUILD SUMMARY")
print(f"Rows: {len(out):,}  | restaurants: {out['Restaurant_ID'].nunique():,}")
print(f"Date range: {out['Inspection Date'].min().date()} to {out['Inspection Date'].max().date()}")
print(f"Failure rate: {out['Results'].mean():.4f}")
print(f"Zero-violation inspections: {(out['Violation_Count'] == 0).sum():,} "
      f"({(out['Violation_Count'] == 0).mean():.1%}); their failure rate: "
      f"{out.loc[out['Violation_Count'] == 0, 'Results'].mean():.4f}")
print("Saved:", OUT_PATH)