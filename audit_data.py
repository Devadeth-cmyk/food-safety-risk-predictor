
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "restaurant_feature_engineering_base.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset shape:", df.shape)
print("Unique restaurants:", df["Restaurant_ID"].nunique())

print("\nInspection type counts:")
print(df["Inspection Type"].value_counts().head(15))

print("\nResults counts:")
print(df["Results"].value_counts(dropna=False))

print("\nMean violation and hazard counts by result:")
print(
    df.groupby("Results")[
        ["Violation_Count", "Hazard_Count"]
    ].mean().round(3)
)

print("\nDate range:")
print(df["Inspection Date"].min(), "to", df["Inspection Date"].max())
