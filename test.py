import pandas as pd

files = [
    "X_full_encoded.csv",
    "y_full.csv",
    "restaurant_feature_engineered_encoded_full.csv",
]

for filename in files:
    df = pd.read_csv(filename)

    print(f"\n{'=' * 60}")
    print(f"File: {filename}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string())