
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50
]


def find_column(df, candidates):
    """Find a column without depending on capitalization."""
    lookup = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}\n"
            "Place your saved test_predictions.csv inside the artifacts folder."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    print("\nPrediction file columns:")
    print(df.columns.tolist())

    # Identify actual labels and predicted probabilities.
    actual_col = find_column(
        df,
        [
            "Actual_Target",
            "Actual_Next_Routine_Failure",
            "Actual_Next_Results",
            "Target_Next_Results",
            "y_true",
            "actual",
        ],
    )

    probability_col = find_column(
        df,
        [
            "Predicted_Failure_Probability",
            "Predicted_Probability",
            "Failure_Probability",
            "y_probability",
            "probability",
        ],
    )

    if actual_col is None or probability_col is None:
        raise ValueError(
            "\nCould not identify the actual-label and probability columns.\n"
            f"Actual-label column found: {actual_col}\n"
            f"Probability column found: {probability_col}\n"
            "Check the printed column names and update the candidate lists."
        )

    df[actual_col] = pd.to_numeric(df[actual_col], errors="coerce")
    df[probability_col] = pd.to_numeric(
        df[probability_col], errors="coerce"
    )

    df = df.dropna(subset=[actual_col, probability_col]).copy()
    df = df[df[actual_col].isin([0, 1])]

    if df.empty:
        raise ValueError("No valid labeled predictions were found.")

    y_true = df[actual_col].astype(int).to_numpy()
    probabilities = df[probability_col].to_numpy()

    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if len(np.unique(y_true)) < 2:
        raise ValueError("Actual labels must contain both classes: 0 and 1.")

    pr_auc = average_precision_score(y_true, probabilities)

    print("\nDataset summary")
    print("-" * 50)
    print(f"Rows evaluated:          {len(y_true):,}")
    print(f"Actual failures:         {int(y_true.sum()):,}")
    print(f"Actual non-failures:     {int((y_true == 0).sum()):,}")
    print(f"Actual failure rate:     {y_true.mean():.2%}")
    print(f"Average precision:       {pr_auc:.4f}")

    rows = []

    for threshold in THRESHOLDS:
        y_pred = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "F1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Predicted_Failures": int(tp + fp),
            "PR_AUC": pr_auc,
        })

    results = pd.DataFrame(rows)

    output_path = ARTIFACTS_DIR / "threshold_comparison.csv"
    results.to_csv(output_path, index=False)

    print("\nThreshold comparison")
    print("-" * 100)
    print(
        results[
            [
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "True_Positives",
                "False_Positives",
                "False_Negatives",
                "Predicted_Failures",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    # These are candidates for investigation, not automatic deployment choices.
    best_f1 = results.loc[results["F1"].idxmax()]
    best_recall = results.loc[results["Recall"].idxmax()]

    print("\nCandidate thresholds")
    print("-" * 50)
    print(
        f"Best F1 threshold: {best_f1['Threshold']:.2f} "
        f"(F1={best_f1['F1']:.3f}, "
        f"Precision={best_f1['Precision']:.3f}, "
        f"Recall={best_f1['Recall']:.3f})"
    )
    print(
        f"Best recall threshold: {best_recall['Threshold']:.2f} "
        f"(Recall={best_recall['Recall']:.3f}, "
        f"Precision={best_recall['Precision']:.3f})"
    )

    print(f"\nSaved results to: {output_path}")

    print(
        "\nIMPORTANT: If these are final test-set predictions, use this "
        "comparison for preliminary analysis only. Select the final "
        "threshold using a separate validation set."
    )


if __name__ == "__main__":
    main()
