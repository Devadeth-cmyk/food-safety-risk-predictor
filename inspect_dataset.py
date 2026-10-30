import pandas as pd
import numpy as np

FEATURES_FILE = "X_full_encoded.csv"
TARGET_FILE = "y_full.csv"
COMBINED_FILE = "restaurant_feature_engineered_encoded_full.csv"

X = pd.read_csv(FEATURES_FILE)
y = pd.read_csv(TARGET_FILE)
combined = pd.read_csv(COMBINED_FILE)

print("=" * 60)
print("1. DATASET SHAPES")
print("=" * 60)

print("X:", X.shape)
print("y:", y.shape)
print("Combined:", combined.shape)

print("\n" + "=" * 60)
print("2. TARGET DISTRIBUTION")
print("=" * 60)

target = y["Target_Next_Results"]

print(target.value_counts(dropna=False).sort_index())
print("\nTarget proportions:")
print(target.value_counts(normalize=True, dropna=False).sort_index())

print("\n" + "=" * 60)
print("3. MISSING VALUES")
print("=" * 60)

print("Missing feature values:", int(X.isna().sum().sum()))
print("Missing target values:", int(y["Target_Next_Results"].isna().sum()))

print("\nFeatures containing missing values:")
print(X.columns[X.isna().any()].tolist())

print("\n" + "=" * 60)
print("4. TARGET ALIGNMENT")
print("=" * 60)

if len(X) == len(y) == len(combined):
    print("Row counts match.")

    if "source_row_index" in y.columns:
        print(
            "Target source_row_index matches combined:",
            y["source_row_index"].equals(combined["source_row_index"])
        )

    if "Restaurant_ID" in y.columns:
        print(
            "Target Restaurant_ID matches combined:",
            y["Restaurant_ID"].equals(combined["Restaurant_ID"])
        )

    print(
        "Target values match combined:",
        target.reset_index(drop=True).equals(
            combined["Target_Next_Results"].reset_index(drop=True)
        )
    )
else:
    print("WARNING: Dataset row counts do not match.")

print("\n" + "=" * 60)
print("5. FEATURE CHECKS")
print("=" * 60)

for col in ["Results", "Risk", "Inspection_Year",
            "Inspection_Sequence", "Historical_Failure_Rate"]:
    if col in X.columns:
        print(f"\n{col}:")
        print(X[col].describe().to_string())

print("\n" + "=" * 60)
print("6. DATE AND RESTAURANT CHECKS")
print("=" * 60)

if "Inspection Date" in combined.columns:
    dates = pd.to_datetime(combined["Inspection Date"], errors="coerce")
    print("Invalid dates:", int(dates.isna().sum()))
    print("Earliest inspection:", dates.min())
    print("Latest inspection:", dates.max())

if "Restaurant_ID" in combined.columns:
    print("Unique restaurants:", combined["Restaurant_ID"].nunique())

print("\nDataset inspection completed.")
