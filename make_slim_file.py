"""Shrink restaurant_feature_engineering_base.csv so it can be uploaded.

Put this script in the same folder as the CSV and run:
    python make_slim_file.py
Output: restaurant_slim.csv.gz (attach this file).
"""
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parent / "restaurant_feature_engineering_base.csv"
OUT = SRC.parent / "restaurant_slim.csv.gz"

if not SRC.exists():
    raise FileNotFoundError(f"Put this script next to {SRC.name}")

header = pd.read_csv(SRC, nrows=0).columns.tolist()
print("Columns in your file:", header)

# Drop only the heavy free-text / duplicate columns; keep everything the model can use.
drop = [c for c in ["Violations", "Location"] if c in header]
keep = [c for c in header if c not in drop]

# Violation codes are tiny and useful for the Pareto chart, so keep them if the text exists.
codes = None
if "Violations" in header:
    chunks = []
    for ch in pd.read_csv(SRC, usecols=["Violations"], chunksize=100_000, low_memory=False):
        codes_ch = (ch["Violations"].fillna("").astype(str)
                    .str.findall(r"(?:^|\|)\s*(\d{1,2})\.\s")
                    .map(lambda xs: ",".join(xs)))
        chunks.append(codes_ch)
    codes = pd.concat(chunks, ignore_index=True)

df = pd.read_csv(SRC, usecols=keep, low_memory=False)
if codes is not None:
    df["Violation_Codes"] = codes.values

df.to_csv(OUT, index=False, compression="gzip")
print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
print(f"Saved {OUT.name}: {OUT.stat().st_size/1e6:.1f} MB  (from {SRC.stat().st_size/1e6:.1f} MB)")