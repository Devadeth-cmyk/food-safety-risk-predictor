
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

X_FILE = Path("X_full_encoded.csv")
Y_FILE = Path("y_full.csv")
COMBINED_FILE = Path(
    "restaurant_feature_engineered_encoded_full.csv"
)

OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Target_Next_Results"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

RANDOM_STATE = 42

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
]

EXCLUDE_FEATURES = {
    "Restaurant_ID",
    "source_row_index",
    "Inspection Date",
    "Results",
    "Risk",
    "Target_Next_Results",
}


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    for file_path in [X_FILE, Y_FILE, COMBINED_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path.resolve()}"
            )

    X_all = pd.read_csv(X_FILE)
    y_frame = pd.read_csv(Y_FILE)
    combined = pd.read_csv(COMBINED_FILE)

    if not (
        len(X_all) == len(y_frame) == len(combined)
    ):
        raise ValueError(
            "Row counts differ between X, y, and combined CSV files."
        )

    if TARGET not in y_frame.columns:
        raise KeyError(f"{TARGET} missing from y_full.csv")

    if TARGET not in combined.columns:
        raise KeyError(f"{TARGET} missing from combined CSV")

    if "Inspection Date" not in combined.columns:
        raise KeyError(
            "Inspection Date missing from combined CSV"
        )

    if not y_frame[TARGET].reset_index(drop=True).equals(
        combined[TARGET].reset_index(drop=True)
    ):
        raise ValueError("Target alignment verification failed.")

    dates_all = pd.to_datetime(
        combined["Inspection Date"],
        errors="coerce",
    )

    if dates_all.isna().any():
        raise ValueError(
            f"{dates_all.isna().sum()} inspection dates are invalid."
        )

    # Unknown future outcomes cannot be used for supervised learning.
    valid_mask = y_frame[TARGET].isin([0, 1])

    unknown_count = int((~valid_mask).sum())

    X = X_all.loc[valid_mask].reset_index(drop=True)

    y = (
        y_frame.loc[valid_mask, TARGET]
        .astype(int)
        .reset_index(drop=True)
    )

    dates = (
        dates_all.loc[valid_mask]
        .reset_index(drop=True)
    )

    feature_columns = [
        col for col in X.columns
        if col not in EXCLUDE_FEATURES
    ]

    X = X[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_columns = X.columns[
        X.isna().any()
    ].tolist()

    if invalid_columns:
        raise ValueError(
            "Missing or non-numeric feature values detected: "
            f"{invalid_columns}"
        )

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    print("=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)

    print("Original records:", len(X_all))
    print("Known targets:", len(X))
    print("Unknown targets excluded:", unknown_count)
    print("Number of features:", len(feature_columns))

    print(
        "Inspection date range:",
        dates.min().date(),
        "to",
        dates.max().date(),
    )

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    print(f"\nFailure rate: {y.mean():.2%}")

    return X, y, dates, feature_columns


# ============================================================
# 3. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def chronological_split(X, y, dates):

    # Sort by inspection date.
    order = np.argsort(
        dates.to_numpy(),
        kind="stable",
    )

    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    dates = dates.iloc[order].reset_index(drop=True)

    n = len(X)

    train_end = int(n * TRAIN_FRACTION)

    validation_end = int(
        n * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    # Date cutoffs avoid splitting records with the same date
    # across different datasets.
    train_cutoff = dates.iloc[train_end]
    validation_cutoff = dates.iloc[validation_end]

    train_mask = dates < train_cutoff

    validation_mask = (
        (dates >= train_cutoff)
        & (dates < validation_cutoff)
    )

    test_mask = dates >= validation_cutoff

    if not (
        train_mask.any()
        and validation_mask.any()
        and test_mask.any()
    ):
        raise ValueError(
            "One of the chronological datasets is empty."
        )

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)

    X_val = X.loc[validation_mask].reset_index(drop=True)
    y_val = y.loc[validation_mask].reset_index(drop=True)

    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    test_dates = dates.loc[test_mask].reset_index(drop=True)

    print("\n" + "=" * 65)
    print("CHRONOLOGICAL DATA SPLIT")
    print("=" * 65)

    for name, mask in [
        ("Training", train_mask),
        ("Validation", validation_mask),
        ("Test", test_mask),
    ]:
        split_dates = dates.loc[mask]
        split_y = y.loc[mask]

        print(
            f"{name}: {len(split_dates):,} records "
            f"({len(split_dates) / n:.1%}) | "
            f"{split_dates.min().date()} to "
            f"{split_dates.max().date()} | "
            f"Failure rate: {split_y.mean():.2%}"
        )

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        test_dates,
    )


# ============================================================
# 4. DEFINE MODELS
# ============================================================

def create_models():

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# 5. TRAIN MODELS AND SELECT USING VALIDATION PR-AUC
# ============================================================

def train_models(X_train, y_train, X_val, y_val):

    results = []
    trained_models = {}
    validation_probabilities = {}

    for name, estimator in create_models().items():

        print("\n" + "=" * 65)
        print("TRAINING:", name)
        print("=" * 65)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ])

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        metrics = {
            "model": name,

            "validation_accuracy": accuracy_score(
                y_val, predictions
            ),

            "validation_precision": precision_score(
                y_val, predictions, zero_division=0
            ),

            "validation_recall": recall_score(
                y_val, predictions, zero_division=0
            ),

            "validation_f1": f1_score(
                y_val, predictions, zero_division=0
            ),

            "validation_pr_auc": average_precision_score(
                y_val, probabilities
            ),

            "validation_roc_auc": (
                roc_auc_score(y_val, probabilities)
                if y_val.nunique() > 1
                else float("nan")
            ),
        }

        results.append(metrics)
        trained_models[name] = model
        validation_probabilities[name] = probabilities

        for metric, value in metrics.items():
            if metric != "model":
                print(f"{metric}: {value:.4f}")

    comparison = pd.DataFrame(results).sort_values(
        "validation_pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    best_name = comparison.iloc[0]["model"]

    best_model = trained_models[best_name]

    best_probabilities = validation_probabilities[best_name]

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))

    print(
        f"\nBest model by validation PR-AUC: {best_name}"
    )

    return best_name, best_model, best_probabilities, comparison


# ============================================================
# 6. SELECT THRESHOLD USING VALIDATION DATA
# ============================================================

def tune_thresholds(y_val, probabilities):

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

        results.append({
            "Threshold": threshold,

            "Accuracy": accuracy_score(
                y_val, predictions
            ),

            "Precision": precision_score(
                y_val, predictions, zero_division=0
            ),

            "Recall": recall_score(
                y_val, predictions, zero_division=0
            ),

            "F1": f1_score(
                y_val, predictions, zero_division=0
            ),

            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),

            "Predicted_Failures": int(
                predictions.sum()
            ),

            "PR_AUC": average_precision_score(
                y_val, probabilities
            ),
        })

    threshold_table = pd.DataFrame(results)

    threshold_table.to_csv(
        OUTPUT_DIR / "validation_threshold_comparison.csv",
        index=False,
    )

    # Choose the threshold with the highest validation F1.
    # Ties prefer higher recall, then higher precision.
    ranked = threshold_table.sort_values(
        ["F1", "Recall", "Precision"],
        ascending=[False, False, False],
    )

    best_row = ranked.iloc[0]

    selected_threshold = float(
        best_row["Threshold"]
    )

    print("\n" + "=" * 65)
    print("VALIDATION THRESHOLD COMPARISON")
    print("=" * 65)

    print(
        threshold_table.round(4).to_string(index=False)
    )

    print(
        f"\nSelected threshold: {selected_threshold:.2f}"
    )

    print(
        f"Validation precision: {best_row['Precision']:.4f}"
    )

    print(
        f"Validation recall: {best_row['Recall']:.4f}"
    )

    print(
        f"Validation F1: {best_row['F1']:.4f}"
    )

    return selected_threshold, threshold_table


# ============================================================
# 7. FINAL TEST EVALUATION
# ============================================================

def evaluate_test(
    model,
    X_test,
    y_test,
    test_dates,
    threshold,
):

    # Test data is not used for model or threshold selection.
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "selected_threshold": float(threshold),

        "test_accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "test_precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),

        "test_recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),

        "test_f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),

        "test_pr_auc": float(
            average_precision_score(
                y_test, probabilities
            )
        ),

        "test_roc_auc": float(
            roc_auc_score(y_test, probabilities)
            if y_test.nunique() > 1
            else float("nan")
        ),

        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),

        "predicted_failures": int(
            predictions.sum()
        ),
    }

    test_predictions = pd.DataFrame({
        "Inspection_Date": test_dates.to_numpy(),
        "Actual_Next_Results": y_test.to_numpy(),
        "Failure_Probability": probabilities,
        "Predicted_Next_Results": predictions,
        "Threshold_Used": threshold,
    })

    test_predictions.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    return metrics


# ============================================================
# 8. SAVE MODEL AND METADATA
# ============================================================

def save_artifacts(
    best_name,
    best_model,
    selected_threshold,
    comparison,
    threshold_table,
    feature_columns,
    test_metrics,
):

    joblib.dump(
        best_model,
        OUTPUT_DIR / "best_restaurant_failure_model.joblib",
    )

    selected_row = threshold_table.loc[
        threshold_table["Threshold"] == selected_threshold
    ].iloc[0].to_dict()

    metadata = {
        "best_model": str(best_name),

        "selection_metric": "Validation PR-AUC",

        "threshold_selection_metric": "Validation F1",

        "selected_threshold": selected_threshold,

        "target": TARGET,

        "feature_columns": feature_columns,

        "split": {
            "method": "chronological",
            "train_fraction_target": TRAIN_FRACTION,
            "validation_fraction_target": VALIDATION_FRACTION,
            "test_fraction_target": TEST_FRACTION,
        },

        "selected_validation_threshold_metrics": selected_row,

        "final_test_metrics": test_metrics,

        "notes": [
            "Unknown targets were excluded from supervised modeling.",
            "Model selection used validation PR-AUC.",
            "Threshold selection used validation F1.",
            "Test data was reserved for final evaluation.",
            "The saved model is fitted on the training split only.",
            "Historical feature engineering must be audited for leakage.",
        ],
    }

    with open(
        OUTPUT_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2, default=float)

    print("\nSaved artifacts:")

    for filename in [
        "model_comparison.csv",
        "validation_threshold_comparison.csv",
        "test_predictions.csv",
        "test_metrics.json",
        "model_metadata.json",
        "best_restaurant_failure_model.joblib",
    ]:
        print(OUTPUT_DIR / filename)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    X, y, dates, feature_columns = load_data()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_dates,
    ) = chronological_split(X, y, dates)

    (
        best_name,
        best_model,
        validation_probabilities,
        comparison,
    ) = train_models(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    selected_threshold, threshold_table = tune_thresholds(
        y_val,
        validation_probabilities,
    )

    test_metrics = evaluate_test(
        best_model,
        X_test,
        y_test,
        test_dates,
        selected_threshold,
    )

    save_artifacts(
        best_name,
        best_model,
        selected_threshold,
        comparison,
        threshold_table,
        feature_columns,
        test_metrics,
    )

    print("\nModel-building pipeline completed successfully.")


if __name__ == "__main__":
    main()
