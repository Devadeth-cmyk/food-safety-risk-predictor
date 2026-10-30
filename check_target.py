
import pandas as pd

FILE = "restaurant_feature_engineered_encoded_full.csv"

df = pd.read_csv(FILE)

df["Inspection Date"] = pd.to_datetime(
    df["Inspection Date"], errors="coerce"
)

print("=" * 65)
print("1. TARGET DISTRIBUTION")
print("=" * 65)

print(df["Target_Next_Results"].value_counts(dropna=False).sort_index())

print("\nTarget distribution by current inspection result:")
print(
    pd.crosstab(
        df["Results"],
        df["Target_Next_Results"],
        margins=True,
        dropna=False
    )
)

print("\n" + "=" * 65)
print("2. MISSING TARGETS")
print("=" * 65)

missing = df[df["Target_Next_Results"].isna()]

print("Rows with missing targets:", len(missing))
print("Unique restaurants among missing targets:",
      missing["Restaurant_ID"].nunique())

missing_per_restaurant = missing.groupby("Restaurant_ID").size()

print("Maximum missing-target rows for one restaurant:",
      missing_per_restaurant.max())

print("\nMissing targets per restaurant:")
print(missing_per_restaurant.value_counts().sort_index())

print("\n" + "=" * 65)
print("3. INSPECTION ORDER CHECK")
print("=" * 65)

# Sort inspections by restaurant and date.
df = df.sort_values(
    ["Restaurant_ID", "Inspection Date", "source_row_index"]
).copy()

df["Next_Inspection_Date"] = (
    df.groupby("Restaurant_ID")["Inspection Date"].shift(-1)
)

df["Next_Actual_Result"] = (
    df.groupby("Restaurant_ID")["Results"].shift(-1)
)

print("Rows with no next inspection:",
      int(df["Next_Inspection_Date"].isna().sum()))

print("\nRows with missing target but an available next inspection:")
print(
    int(
        (
            df["Target_Next_Results"].isna()
            & df["Next_Inspection_Date"].notna()
        ).sum()
    )
)

print("\nRows with a target but no next inspection:")
print(
    int(
        (
            df["Target_Next_Results"].notna()
            & df["Next_Inspection_Date"].isna()
        ).sum()
    )
)

print("\n" + "=" * 65)
print("4. COMPARE TARGET WITH NEXT ACTUAL RESULT")
print("=" * 65)

valid = df[
    df["Target_Next_Results"].notna()
    & df["Next_Actual_Result"].notna()
].copy()

print("Comparable rows:", len(valid))

if len(valid) > 0:
    print("\nTarget versus next inspection result:")
    print(
        pd.crosstab(
            valid["Target_Next_Results"],
            valid["Next_Actual_Result"],
            margins=True,
            dropna=False
        )
    )

    agreement = (
        valid["Target_Next_Results"]
        == valid["Next_Actual_Result"]
    ).mean()

    print(f"\nExact numeric agreement: {agreement:.4%}")

print("\n" + "=" * 65)
print("5. SAMPLE RECORDS")
print("=" * 65)

columns = [
    "Restaurant_ID",
    "Inspection Date",
    "Results",
    "Target_Next_Results",
    "Next_Inspection_Date",
    "Next_Actual_Result",
]

print(df[columns].head(20).to_string(index=False))

print("\nTarget logic inspection completed.")
