
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "best_restaurant_failure_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

X_PATH = BASE_DIR / "X_full_encoded.csv"
Y_PATH = BASE_DIR / "y_full.csv"
DATA_PATH = BASE_DIR / "restaurant_feature_engineered_encoded_full.csv"

OUTPUT_DIR = ARTIFACTS_DIR / "shap_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BACKGROUND_SIZE = 20
EXPLAIN_SIZE = 60
MAX_DISPLAY = 20


# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

def load_inputs():
    required_files = [
        MODEL_PATH,
        METADATA_PATH,
        X_PATH,
        Y_PATH,
        DATA_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = joblib.load(MODEL_PATH)

    feature_columns = metadata.get("feature_columns")
    target = metadata.get("target", "Target_Next_Results")

    if not feature_columns:
        raise ValueError(
            "feature_columns are missing from model_metadata.json"
        )

    X_all = pd.read_csv(X_PATH)
    y_all = pd.read_csv(Y_PATH)
    combined = pd.read_csv(DATA_PATH)

    if not (len(X_all) == len(y_all) == len(combined)):
        raise ValueError(
            "Row counts differ between the feature, target, "
            "and combined datasets."
        )

    if target not in y_all.columns:
        raise KeyError(f"Target column not found: {target}")

    if "Inspection Date" not in combined.columns:
        raise KeyError("Inspection Date is missing from the dataset.")

    # Verify target alignment.
    if not y_all[target].reset_index(drop=True).equals(
        combined[target].reset_index(drop=True)
    ):
        raise ValueError(
            "Target alignment failed between y_full.csv "
            "and the combined dataset."
        )

    # Keep only rows with known binary outcomes.
    valid_mask = y_all[target].isin([0, 1])

    X = X_all.loc[valid_mask].reset_index(drop=True)
    y = (
        y_all.loc[valid_mask, target]
        .astype(int)
        .reset_index(drop=True)
    )

    dates = pd.to_datetime(
        combined.loc[valid_mask, "Inspection Date"],
        errors="coerce",
    ).reset_index(drop=True)

    if dates.isna().any():
        raise ValueError("Invalid inspection dates were found.")

    missing_features = [
        col for col in feature_columns
        if col not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing training features: {missing_features}"
        )

    # Preserve exactly the training feature order.
    X = X[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # The saved training pipeline includes a median imputer.
    # Check that the remaining missing values can be handled.
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(
            "Infinite values found in features. "
            "Clean them before running SHAP."
        )

    # Reconstruct approximate chronological partitions.
    # Dates tied at a boundary are kept together.
    sorted_dates = dates.sort_values().reset_index(drop=True)
    n = len(sorted_dates)

    train_cutoff = sorted_dates.iloc[
        max(0, int(n * 0.70) - 1)
    ]

    validation_cutoff = sorted_dates.iloc[
        max(0, int(n * 0.85) - 1)
    ]

    train_positions = np.flatnonzero(
        (dates <= train_cutoff).to_numpy()
    )

    test_positions = np.flatnonzero(
        (dates > validation_cutoff).to_numpy()
    )

    if len(train_positions) == 0 or len(test_positions) == 0:
        raise ValueError(
            "Unable to reconstruct chronological data partitions."
        )

    rng = np.random.default_rng(RANDOM_STATE)

    background_positions = rng.choice(
        train_positions,
        size=min(BACKGROUND_SIZE, len(train_positions)),
        replace=False,
    )

    explain_positions = rng.choice(
        test_positions,
        size=min(EXPLAIN_SIZE, len(test_positions)),
        replace=False,
    )

    background = X.iloc[background_positions].reset_index(drop=True)
    X_explain = X.iloc[explain_positions].reset_index(drop=True)
    y_explain = y.iloc[explain_positions].reset_index(drop=True)

    print("=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)
    print("Saved model:", metadata.get("best_model"))
    print("Estimator type:", type(model).__name__)
    print("Number of features:", len(feature_columns))
    print("Labeled records:", len(X))
    print("Background sample:", len(background))
    print("Explanation sample:", len(X_explain))
    print("Output folder:", OUTPUT_DIR)

    print(
        "\nNote: the explanation sample is drawn from a reconstructed "
        "chronological test-period proxy. Verify the exact original "
        "split before reporting these results as test-set explanations."
    )

    return (
        model,
        feature_columns,
        background,
        X_explain,
        y_explain,
        metadata,
    )


# ============================================================
# 3. FAILURE PROBABILITY
# ============================================================

def predict_failure_probability(model, data, feature_columns):
    data = pd.DataFrame(data, columns=feature_columns)

    probabilities = model.predict_proba(data)

    # Class 1 represents a predicted next-inspection failure.
    return np.asarray(probabilities)[:, 1]


# ============================================================
# 4. RUN SHAP
# ============================================================

def main():
    (
        model,
        feature_columns,
        background,
        X_explain,
        y_explain,
        metadata,
    ) = load_inputs()

    print("\nInitializing SHAP PermutationExplainer...")

    # A model-agnostic explainer is used for compatibility
    # with HistGradientBoosting and sklearn pipelines.
    masker = shap.maskers.Independent(background)

    explainer = shap.Explainer(
        lambda data: predict_failure_probability(
            model,
            data,
            feature_columns,
        ),
        masker,
        algorithm="permutation",
        feature_names=feature_columns,
    )

    max_evals = 2 * len(feature_columns) + 1

    print("Calculating SHAP values. Please wait...")

    explanation = explainer(
        X_explain,
        max_evals=max_evals,
    )

    shap_values = np.asarray(explanation.values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Create a SHAP Explanation object for plotting.
    base_values = np.asarray(explanation.base_values)

    if base_values.ndim == 0:
        base_values = np.repeat(
            float(base_values),
            len(X_explain),
        )
    else:
        base_values = base_values.reshape(-1)

    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_explain.to_numpy(),
        feature_names=feature_columns,
    )

    # ========================================================
    # 5. GLOBAL FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0),
        "Mean_SHAP": shap_values.mean(axis=0),
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTop 15 influential features:")
    print(importance.head(15).to_string(index=False))

    # Bar plot.
    shap.plots.bar(
        shap_explanation,
        max_display=MAX_DISPLAY,
        show=False,
    )

    plt.gcf().set_size_inches(11, 7)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_summary_bar.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    # ========================================================
    # 6. BEESWARM SUMMARY PLOT
    # ========================================================

    shap.plots.beeswarm(
        shap_explanation,
        max_display=MAX_DISPLAY,
        show=False,
    )

    plt.gcf().set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_beeswarm.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    # ========================================================
    # 7. DEPENDENCE PLOTS FOR TOP FEATURES
    # ========================================================

    top_features = importance["Feature"].head(5).tolist()

    for feature in top_features:
        shap.plots.scatter(
            shap_explanation[:, feature],
            show=False,
        )

        plt.gcf().set_size_inches(9, 6)
        plt.tight_layout()

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in feature
        )

        plt.savefig(
            OUTPUT_DIR / f"shap_dependence_{safe_name}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close("all")

    # ========================================================
    # 8. INDIVIDUAL PREDICTION EXPLANATION
    # ========================================================

    probabilities = predict_failure_probability(
        model,
        X_explain,
        feature_columns,
    )

    # Choose the sample with the highest predicted failure probability.
    example_idx = int(np.argmax(probabilities))

    example_explanation = shap_explanation[example_idx]

    shap.plots.waterfall(
        example_explanation,
        max_display=MAX_DISPLAY,
        show=False,
    )

    plt.gcf().set_size_inches(11, 8)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "shap_waterfall.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    # Save values for every explained sample.
    shap_values_df = pd.DataFrame(
        shap_values,
        columns=feature_columns,
    )

    shap_values_df.insert(
        0,
        "Sample_Index",
        np.arange(len(shap_values_df)),
    )

    shap_values_df.to_csv(
        OUTPUT_DIR / "shap_values_sample.csv",
        index=False,
    )

    # Save the individual example's feature contributions.
    individual = pd.DataFrame({
        "Feature": feature_columns,
        "Feature_Value": X_explain.iloc[example_idx].to_numpy(),
        "SHAP_Value": shap_values[example_idx],
    })

    individual["Absolute_SHAP"] = individual["SHAP_Value"].abs()

    individual = individual.sort_values(
        "Absolute_SHAP",
        ascending=False,
    )

    individual.to_csv(
        OUTPUT_DIR / "high_risk_example_explanation.csv",
        index=False,
    )

    # Save a compact summary report.
    report = [
        "RESTAURANT INSPECTION FAILURE - SHAP REPORT",
        "=" * 50,
        f"Model in metadata: {metadata.get('best_model')}",
        f"Estimator type: {type(model).__name__}",
        f"Features explained: {len(feature_columns)}",
        f"Explanation sample size: {len(X_explain)}",
        f"Highest predicted failure probability: "
        f"{probabilities[example_idx]:.4f}",
        "",
        "TOP 15 FEATURES BY MEAN ABSOLUTE SHAP VALUE",
        "-" * 50,
    ]

    for _, row in importance.head(15).iterrows():
        report.append(
            f"{row['Feature']}: "
            f"{row['Mean_Absolute_SHAP']:.6f}"
        )

    report.extend([
        "",
        "Interpretation:",
        "- Positive SHAP values push predicted failure probability upward.",
        "- Negative SHAP values push predicted failure probability downward.",
        "- Mean absolute SHAP measures average contribution magnitude.",
        "- SHAP describes model behavior; it does not prove causation.",
        "- Feature leakage and prediction-time availability require separate checks.",
    ])

    (OUTPUT_DIR / "shap_analysis_summary.txt").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 60)
    print("Results saved in:", OUTPUT_DIR.resolve())
    print("\nGenerated files:")

    for file_path in sorted(OUTPUT_DIR.iterdir()):
        print("-", file_path.name)


if __name__ == "__main__":
    main()
