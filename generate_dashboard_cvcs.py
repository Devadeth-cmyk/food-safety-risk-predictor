from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS / "dashboard_exports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Target_Next_Results"
HIGH_RISK_THRESHOLD = 0.15


# ============================================================
# FILE LOADING
# ============================================================

def find_file(*paths):
    for path in paths:
        if path and path.exists():
            return path
    return None


def find_column(df, candidates):
    columns = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    return None


def load_project():
    dataset_path = find_file(
        ROOT / "restaurant_feature_engineered_encoded_full.csv",
        ROOT / "restaurant_feature_engineered_fixed.csv",
        ROOT / "restaurant_feature_engineered_final.csv",
    )

    if dataset_path is None:
        raise FileNotFoundError(
            "Dataset not found. Place one of these files beside "
            "this script:\n"
            "- restaurant_feature_engineered_encoded_full.csv\n"
            "- restaurant_feature_engineered_fixed.csv\n"
            "- restaurant_feature_engineered_final.csv"
        )

    df = pd.read_csv(dataset_path, low_memory=False)

    model_path = find_file(
        ARTIFACTS / "best_restaurant_failure_model.joblib",
        ROOT / "best_restaurant_failure_model.joblib",
    )

    metadata_path = find_file(
        ARTIFACTS / "model_metadata.json",
        ROOT / "model_metadata.json",
    )

    model = None
    metadata = {}

    if model_path:
        model = joblib.load(model_path)
    else:
        warnings.warn(
            "Saved prediction model not found. "
            "Prediction probabilities will be blank."
        )

    if metadata_path:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return df, model, metadata


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def predict_probabilities(df, model, metadata):
    if model is None:
        return pd.Series(np.nan, index=df.index)

    features = (
        metadata.get("feature_columns")
        or metadata.get("features")
    )

    if not features and hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)

    excluded = [
        TARGET,
        "Restaurant_ID",
        "Inspection Date",
        "Results",
        "Risk",
        "Next_Inspection_Date",
        "Restaurant Name",
        "DBA Name",
    ]

    X = df.drop(
        columns=[
            c for c in excluded
            if c in df.columns
        ],
        errors="ignore",
    )

    X = X.select_dtypes(
        include=["number", "bool"]
    )

    if features:
        missing = [
            feature for feature in features
            if feature not in X.columns
        ]

        if missing:
            warnings.warn(
                "Model scoring skipped because these features "
                f"are missing: {missing[:10]}"
            )
            return pd.Series(np.nan, index=df.index)

        X = X[features]

    if not hasattr(model, "predict_proba"):
        warnings.warn(
            "The loaded model does not support predict_proba()."
        )
        return pd.Series(np.nan, index=df.index)

    try:
        probabilities = model.predict_proba(X)

        classes = list(
            getattr(model, "classes_", [0, 1])
        )

        positive_index = (
            classes.index(1)
            if 1 in classes
            else len(classes) - 1
        )

        return pd.Series(
            probabilities[:, positive_index],
            index=df.index,
        )

    except Exception as exc:
        warnings.warn(f"Prediction failed: {exc}")
        return pd.Series(np.nan, index=df.index)


# ============================================================
# 1. INSPECTION PRIORITY LISTS
# ============================================================

def generate_priority_lists(df, probabilities):
    name_col = find_column(
        df,
        ["Restaurant Name", "DBA Name", "Restaurant_Name", "Name"],
    )

    id_col = find_column(
        df,
        ["Restaurant_ID", "Restaurant ID", "License #", "ID"],
    )

    date_col = find_column(
        df,
        ["Inspection Date", "Inspection_Date", "Date"],
    )

    result_col = find_column(
        df,
        ["Results", "Result", "Inspection Result"],
    )

    violations_col = find_column(
        df,
        ["Violations", "Violation Description", "Violation_Text"],
    )

    output = df.copy()
    output["_probability"] = probabilities

    # Keep the latest available inspection per restaurant.
    identity_col = id_col or name_col

    if date_col:
        output[date_col] = pd.to_datetime(
            output[date_col],
            errors="coerce",
        )

        if identity_col:
            output = (
                output.sort_values(
                    date_col,
                    na_position="first",
                )
                .drop_duplicates(
                    subset=identity_col,
                    keep="last",
                )
            )

    rename_map = {}

    if name_col:
        rename_map[name_col] = "restaurant_name"

    if id_col:
        rename_map[id_col] = "restaurant_id"

    if date_col:
        rename_map[date_col] = "latest_inspection_date"

    if result_col:
        rename_map[result_col] = "latest_inspection_result"

    if violations_col:
        rename_map[violations_col] = "violations"

    output = output.rename(columns=rename_map)

    output["predicted_next_failure_probability"] = (
        output.pop("_probability")
    )

    output["predicted_next_failure_probability_pct"] = (
        output["predicted_next_failure_probability"] * 100
    )

    output["priority_score_0_100"] = (
        output["predicted_next_failure_probability_pct"]
    )

    output = output.sort_values(
        "predicted_next_failure_probability",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    output.insert(
        0,
        "priority_rank",
        np.arange(1, len(output) + 1),
    )

    n = len(output)

    if n:
        rank = output["priority_rank"]

        top_5 = max(1, int(np.ceil(n * 0.05)))
        top_10 = max(top_5, int(np.ceil(n * 0.10)))
        top_25 = max(top_10, int(np.ceil(n * 0.25)))

        output["priority_tier"] = np.select(
            [
                rank <= top_5,
                rank <= top_10,
                rank <= top_25,
            ],
            [
                "P1 - Very High (top 5%)",
                "P2 - High (next 5%)",
                "P3 - Elevated (next 15%)",
            ],
            default="P4 - Standard (remaining 75%)",
        )

    if "latest_inspection_date" in output.columns:
        dates = pd.to_datetime(
            output["latest_inspection_date"],
            errors="coerce",
        )

        dataset_end = dates.max()

        if pd.notna(dataset_end):
            output["days_since_inspection_at_dataset_end"] = (
                dataset_end - dates
            ).dt.days

    if "latest_inspection_result" in output.columns:
        numeric_results = pd.to_numeric(
            output["latest_inspection_result"],
            errors="coerce",
        )

        output["current_result_label"] = (
            numeric_results.map({
                0: "Pass",
                1: "Fail",
            })
            .fillna(
                output["latest_inspection_result"].astype(str)
            )
        )

    if "violations" in output.columns:
        output["violation_excerpt"] = (
            output["violations"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.slice(0, 300)
        )

    # Full queue.
    output.to_csv(
        OUTPUT_DIR / "inspection_priority_list_full.csv",
        index=False,
    )

    # First 100 rows of the ranked queue.
    output.head(100).to_csv(
        OUTPUT_DIR / "inspection_priority_list.csv",
        index=False,
    )

    print("Generated inspection priority lists.")


# ============================================================
# 2. VIOLATION PARETO SUMMARY
# ============================================================

def generate_violation_pareto(df):
    violations_col = find_column(
        df,
        [
            "Violations",
            "Violation Description",
            "Violation_Text",
            "Violation",
        ],
    )

    if violations_col:
        values = (
            df[violations_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[values != ""]

        entries = (
            values.str.split(r"\s*;\s*|\s*\|\s*", regex=True)
            .explode()
            .str.strip()
        )

        entries = entries[entries != ""]

        summary = (
            entries.value_counts()
            .rename_axis("violation")
            .reset_index(name="occurrences")
        )

    else:
        count_col = find_column(
            df,
            [
                "Violation_Count",
                "Violation Count",
                "Number_of_Violations",
            ],
        )

        if not count_col:
            print(
                "Violation descriptions/counts unavailable. "
                "Creating an empty summary."
            )

            summary = pd.DataFrame(
                columns=["violation", "occurrences"]
            )

        else:
            counts = pd.to_numeric(
                df[count_col],
                errors="coerce",
            ).dropna()

            summary = pd.DataFrame([
                {
                    "violation": (
                        "Aggregate row-level violation count; "
                        "individual descriptions unavailable"
                    ),
                    "occurrences": int(counts.sum()),
                }
            ])

    if not summary.empty:
        total = summary["occurrences"].sum()

        summary["percentage"] = (
            summary["occurrences"] / total * 100
            if total
            else 0
        )

        summary["cumulative_percentage"] = (
            summary["percentage"].cumsum()
        )
    else:
        summary["percentage"] = pd.Series(dtype=float)
        summary["cumulative_percentage"] = pd.Series(dtype=float)

    summary.to_csv(
        OUTPUT_DIR / "violation_pareto_summary.csv",
        index=False,
    )

    print("Generated violation Pareto summary.")


# ============================================================
# 3. NEIGHBORHOOD RISK SUMMARY
# ============================================================

def generate_neighborhood_summary(df, probabilities):
    """
    Generate neighborhood/geographical risk summaries using
    Geo_Region_ID, Geo_Grid_ID, or coordinates.
    """

    df = df.copy()
    df["_predicted_probability"] = np.asarray(probabilities)

    # Prefer the broad geographical region.
    group_col = find_column(
        df,
        [
            "Geo_Region_ID",
            "Geo_Grid_ID",
            "Neighborhood",
            "Neighbourhood",
            "Borough",
            "Community Area",
            "ZIP",
            "Zip Code",
            "Postal Code",
        ],
    )

    # If no geographic identifier exists, construct approximate
    # grid cells from latitude and longitude.
    if group_col is None:
        lat_col = find_column(
            df, ["Latitude", "latitude", "lat"]
        )
        lon_col = find_column(
            df, ["Longitude", "longitude", "lon", "lng"]
        )

        if lat_col and lon_col:
            df[lat_col] = pd.to_numeric(
                df[lat_col], errors="coerce"
            )
            df[lon_col] = pd.to_numeric(
                df[lon_col], errors="coerce"
            )

            valid = (
                df[lat_col].between(-90, 90)
                & df[lon_col].between(-180, 180)
            )

            df = df.loc[valid].copy()

            df["_geo_cell"] = (
                df[lat_col].round(2).astype(str)
                + ", "
                + df[lon_col].round(2).astype(str)
            )

            group_col = "_geo_cell"

    if group_col is None:
        raise ValueError(
            "No usable geographic identifiers, latitude, "
            "or longitude columns were found."
        )

    # Use the target for historical failure rate when available.
    target_col = find_column(
        df,
        ["Target_Next_Results", "Target", "Failure"],
    )

    result_col = find_column(
        df,
        ["Results", "Result", "Inspection Result"],
    )

    actual_col = target_col or result_col

    if actual_col:
        df["_actual_failure"] = pd.to_numeric(
            df[actual_col], errors="coerce"
        )
    else:
        df["_actual_failure"] = np.nan

    # Count distinct restaurants when an ID is available.
    id_col = find_column(
        df,
        ["Restaurant_ID", "Restaurant ID", "License #", "ID"],
    )

    summary = df.groupby(
        group_col, dropna=False
    ).agg(
        inspection_records=("_predicted_probability", "size"),
        historical_failure_rate=(
            "_actual_failure",
            lambda values: (
                (values == 1).mean()
                if values.notna().any()
                else np.nan
            ),
        ),
        mean_predicted_probability=(
            "_predicted_probability", "mean"
        ),
        high_risk_restaurants=(
            "_predicted_probability",
            lambda values: int(
                (values >= HIGH_RISK_THRESHOLD).sum()
            ),
        ),
    ).reset_index()

    summary.rename(
        columns={group_col: "neighborhood"},
        inplace=True,
    )

    if id_col:
        counts = (
            df.groupby(group_col, dropna=False)[id_col]
            .nunique()
            .rename("restaurants")
            .reset_index()
            .rename(columns={group_col: "neighborhood"})
        )

        summary = summary.merge(
            counts,
            on="neighborhood",
            how="left",
        )
    else:
        summary["restaurants"] = summary["inspection_records"]

    summary["historical_failure_rate_pct"] = (
        summary.pop("historical_failure_rate") * 100
    )

    summary["mean_predicted_failure_probability_pct"] = (
        summary.pop("mean_predicted_probability") * 100
    )

    summary.sort_values(
        "mean_predicted_failure_probability_pct",
        ascending=False,
        na_position="last",
        inplace=True,
    )

    output_path = (
        OUTPUT_DIR / "neighborhood_risk_summary.csv"
    )

    summary.to_csv(output_path, index=False)

    print(
        f"Neighborhood summary generated: "
        f"{len(summary)} geographic groups"
    )
    print(f"Grouping column: {group_col}")
    print(f"Saved to: {output_path}")


# ============================================================
# 4. FEATURE IMPORTANCE / SHAP
# ============================================================

def generate_feature_importance(df, model, metadata):
    if model is None:
        print(
            "No model available. SHAP feature importance "
            "cannot be calculated."
        )

        pd.DataFrame(
            columns=["feature", "importance", "importance_method"]
        ).to_csv(
            OUTPUT_DIR / "shap_feature_importance.csv",
            index=False,
        )

        return

    features = (
        metadata.get("feature_columns")
        or metadata.get("features")
    )

    if not features and hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)

    # Use native feature importance if the model supports it.
    # This is not SHAP and is labelled accurately.
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

        if not features or len(features) != len(importance):
            features = list(
                getattr(
                    model,
                    "feature_names_in_",
                    [
                        f"feature_{i}"
                        for i in range(len(importance))
                    ],
                )
            )

        summary = pd.DataFrame({
            "feature": features,
            "importance": importance,
            "importance_method": (
                "Native model feature importance; not SHAP"
            ),
        })

        summary = summary.sort_values(
            "importance",
            ascending=False,
        )

        summary.to_csv(
            OUTPUT_DIR / "shap_feature_importance.csv",
            index=False,
        )

        print(
            "Generated feature importance using the model's "
            "native importance values, not SHAP."
        )

        return

    # If native feature importance is unavailable, try SHAP.
    try:
        import shap

        X = df.select_dtypes(
            include=["number", "bool"]
        ).drop(
            columns=[TARGET],
            errors="ignore",
        )

        if features:
            missing = [
                f for f in features
                if f not in X.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing model features: {missing[:10]}"
                )

            X = X[features]

        if X.empty:
            raise ValueError("No numeric feature columns available.")

        sample = X.sample(
            min(200, len(X)),
            random_state=42,
        )

        explainer = shap.Explainer(model, sample)
        shap_values = explainer(sample).values

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        importance = np.abs(shap_values).mean(axis=0)

        summary = pd.DataFrame({
            "feature": sample.columns,
            "importance": importance,
            "importance_method": "Mean absolute SHAP value",
        }).sort_values(
            "importance",
            ascending=False,
        )

        summary.to_csv(
            OUTPUT_DIR / "shap_feature_importance.csv",
            index=False,
        )

        print("Generated SHAP feature importance.")

    except Exception as exc:
        warnings.warn(
            f"SHAP calculation failed: {exc}"
        )

        pd.DataFrame(
            columns=["feature", "importance", "importance_method"]
        ).to_csv(
            OUTPUT_DIR / "shap_feature_importance.csv",
            index=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    df, model, metadata = load_project()

    print(f"Loaded {len(df):,} dataset rows.")

    probabilities = predict_probabilities(
        df,
        model,
        metadata,
    )

    generate_priority_lists(df, probabilities)
    generate_violation_pareto(df)
    generate_neighborhood_summary(df, probabilities)
    generate_feature_importance(df, model, metadata)

    print("\nFinished generating dashboard exports.")
    print(f"Output folder: {OUTPUT_DIR}")

    for filename in [
        "inspection_priority_list.csv",
        "inspection_priority_list_full.csv",
        "violation_pareto_summary.csv",
        "neighborhood_risk_summary.csv",
        "shap_feature_importance.csv",
    ]:
        path = OUTPUT_DIR / filename
        print(
            f"{filename}: "
            f"{'Created' if path.exists() else 'MISSING'}"
        )


if __name__ == "__main__":
    main()