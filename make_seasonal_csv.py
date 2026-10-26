"""
# TEAM NOTE:
# Builds seasonal_failure_by_month.csv for the Streamlit Seasonal Trends tab.
# Uses restaurant_feature_engineering_base.csv (Fail=1 / Pass=0 encoding).
# Writes to both outputs/ and project root so app.py path search finds it.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

BASE_CANDIDATES = [
    ROOT / "restaurant_feature_engineering_base.csv",
    OUTPUTS / "restaurant_feature_engineering_base.csv",
]

base_path = next((p for p in BASE_CANDIDATES if p.exists()), None)
if base_path is None:
    raise FileNotFoundError(
        "restaurant_feature_engineering_base.csv not found. "
        "Put it in the project root or outputs/."
    )

print(f"Loading: {base_path}")
df = pd.read_csv(base_path, low_memory=False)

# --- find a date column ---
date_col = None
for c in ["Inspection_Date", "Inspection Date", "inspection_date"]:
    if c in df.columns:
        date_col = c
        break
if date_col is None:
    raise KeyError(f"No inspection date column found. Columns: {list(df.columns)[:30]}")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).copy()
df["Month"] = df[date_col].dt.month

# --- find fail indicator (Results already 1=Fail, 0=Pass in this project) ---
fail_col = None
for c in ["Results", "Result", "Failed", "Is_Fail", "Failure"]:
    if c in df.columns:
        fail_col = c
        break
if fail_col is None:
    raise KeyError("No Results/Fail column found.")

# Normalize to 0/1 if still text
if df[fail_col].dtype == object:
    s = df[fail_col].astype(str).str.strip().str.lower()
    df["_fail"] = s.isin(["fail", "1", "failed"]).astype(int)
else:
    df["_fail"] = (df[fail_col] == 1).astype(int)

# Monthly failure rate (%)
monthly = (
    df.groupby("Month", as_index=False)
    .agg(N_Inspections=(" _fail".strip(), "size"), N_Fails=("_fail", "sum"))
)
# fix accidental space in column name above
monthly = (
    df.groupby("Month", as_index=False)
    .agg(N_Inspections=("_fail", "size"), N_Fails=("_fail", "sum"))
)
monthly["Failure_Rate_Percent"] = (
    100.0 * monthly["N_Fails"] / monthly["N_Inspections"]
).round(2)

# Optional nice month labels
month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
monthly["Month_Name"] = monthly["Month"].map(month_names)

# Column order expected by app
out = monthly[["Month", "Month_Name", "Failure_Rate_Percent", "N_Inspections", "N_Fails"]]

path_out = OUTPUTS / "seasonal_failure_by_month.csv"
path_root = ROOT / "seasonal_failure_by_month.csv"
out.to_csv(path_out, index=False)
out.to_csv(path_root, index=False)

print(f"Wrote: {path_out}")
print(f"Wrote: {path_root}")
print(out.to_string(index=False))