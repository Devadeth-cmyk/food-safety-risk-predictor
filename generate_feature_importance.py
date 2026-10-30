
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "artifacts" / "best_restaurant_failure_model.joblib"
DATASET_PATH = ROOT / "restaurant_feature_engineered_fixed.csv"
OUTPUT_DIR = ROOT / "artifacts" / "dashboard_exports"
OUTPUT_PATH = OUTPUT_DIR / "shap_feature_importance.csv"

TARGET = "Target_Next_Results"
MAX_ROWS = 3000
N_REPEATS = 5
RANDOM_STATE = 42


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def reconstruct_features(df, expected_features):
    """Reconstruct the features expected by the saved model."""

    data = df.copy()

    # Numeric features expected by the model.
    numeric_features = [
        "Inspection_Sequence",
        "Days_Since_Previous_Inspection",
        "Inspection_Year",
        "Violation_Count",
        "Latitude",
        "Longitude",
        "Administrative_Count",
        "Average_Violation_Score",
        "Compliant_Positive_Count",
        "Hazard_Count",
        "Previous_Failure_Count",
        "Historical_Failure_Rate",
        "Previous_Violation_Count",
        "Violation_Count_Change",
    ]

    for col in numeric_features:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Previous inspection availability.
    if "Previous_Inspection_Count" in data.columns:
        data["Has_Previous_Inspection"] = (
            pd.to_numeric(
                data["Previous_Inspection_Count"],
                errors="coerce",
            ).fillna(0) > 0
        ).astype(int)
    else:
        data["Has_Previous_Inspection"] = (
            data.get(
                "Previous_Failure_Count",
                pd.Series(0, index=data.index),
            ).fillna(0) > 0
        ).astype(int)

    # Cyclical inspection day and month features.
    if "Inspection Date" in data.columns:
        dates = pd.to_datetime(
            data["Inspection Date"],
            errors="coerce",
        )
        day = dates.dt.dayofweek
        month = dates.dt.month

        data["Inspection_Day_sin"] = np.sin(
            2 * np.pi * day / 7
        )
        data["Inspection_Day_cos"] = np.cos(
            2 * np.pi * day / 7
        )

        data["Inspection_Month_sin"] = np.sin(
            2 * np.pi * (month - 1) / 12
        )
        data["Inspection_Month_cos"] = np.cos(
            2 * np.pi * (month - 1) / 12
        )
    else:
        for col in [
            "Inspection_Day_sin",
            "Inspection_Day_cos",
            "Inspection_Month_sin",
            "Inspection_Month_cos",
        ]:
            data[col] = np.nan

    # One-hot encode previous inspection result.
    previous = normalize_text(
        data.get(
            "Previous_Result",
            pd.Series("", index=data.index),
        )
    )

    previous_categories = {
        "Previous_Result_fail": {
            "fail",
            "failed",
            "failure",
        },
        "Previous_Result_no_previous_inspection": {
            "",
            "nan",
            "none",
            "no previous inspection",
            "no_previous_inspection",
        },
        "Previous_Result_pass": {
            "pass",
            "passed",
            "pass w/ conditions",
            "pass with conditions",
        },
    }

    for col, categories in previous_categories.items():
        data[col] = previous.isin(categories).astype(int)

    # One-hot encode inspection types.
    inspection_type = normalize_text(
        data.get(
            "Inspection Type",
            pd.Series("", index=data.index),
        )
    )

    inspection_type_features = [
        "canvass",
        "canvass re-inspection",
        "complaint",
        "complaint re-inspection",
        "complaint-fire",
        "consultation",
        "license",
        "license re-inspection",
        "license-task force",
        "other/unknown",
        "recent inspection",
        "short form complaint",
        "short form fire-complaint",
        "suspected food poisoning",
        "suspected food poisoning re-inspection",
        "tag removal",
        "task force liquor 1475",
    ]

    for category in inspection_type_features:
        col = f"Inspection Type_{category}"
        data[col] = (inspection_type == category).astype(int)

    # Reconstruct geographic one-hot columns.
    # Use the same grid values encoded in the trained model.
    grid_values = [
        "41.6_-87.5",
        "41.6_-87.6",
        "41.7_-87.5",
        "41.7_-87.6",
        "41.7_-87.7",
        "41.8_-87.5",
        "41.8_-87.6",
        "41.8_-87.7",
        "41.8_-87.8",
        "41.9_-87.6",
        "41.9_-87.7",
        "41.9_-87.8",
        "42.0_-87.6",
        "42.0_-87.7",
        "42.0_-87.8",
        "42.0_-87.9",
    ]

    if "Geo_Grid_ID" in data.columns:
        grid = (
            data["Geo_Grid_ID"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        # The original preprocessing appears to use latitude and
        # longitude rounded to one decimal place.
        lat = data["Latitude"].round(1)
        lon = data["Longitude"].round(1)
        grid = (
            lat.map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
            + "_"
            + lon.map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        )

    for value in grid_values:
        col = f"Geo_Grid_ID_{value}"
        data[col] = (grid == value).astype(int)

    # Preserve the exact feature order expected by the model.
    missing = [
        col for col in expected_features
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Unable to reconstruct model features: "
            + ", ".join(missing)
        )

    X = data[expected_features].copy()

    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        elif pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].replace(
                [np.inf, -np.inf],
                np.nan,
            )

    return X


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("Loading saved model...")
    model = joblib.load(MODEL_PATH)

    expected_features = list(
        getattr(model, "feature_names_in_", [])
    )

    if not expected_features:
        raise ValueError(
            "The model does not expose feature_names_in_. "
            "Provide the original training feature list."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column {TARGET!r} not found in dataset."
        )

    print("Reconstructing the model's 55 input features...")
    X = reconstruct_features(df, expected_features)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if len(X) > MAX_ROWS:
        sampled = X.sample(
            n=MAX_ROWS,
            random_state=RANDOM_STATE,
        )
        X = sampled
        y = y.loc[X.index]

    print(f"Rows: {len(X):,}")
    print(f"Features: {len(expected_features)}")
    print("Calculating permutation importance...")

    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": expected_features,
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
        "importance_method": (
            "Permutation importance (average precision)"
        ),
    }).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(OUTPUT_PATH, index=False)

    print("\nTop 15 features:")
    print(importance.head(15).to_string(index=False))

    print(f"\nSaved feature importance to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
