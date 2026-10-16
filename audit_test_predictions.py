
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
PRED_PATH = OUTPUT_DIR / "test_predictions.csv"
if not PRED_PATH.exists(): PRED_PATH = BASE_DIR / PRED_PATH.name
OUTPUT_PATH = OUTPUT_DIR / "deduplicated_test_audit.csv"

df = pd.read_csv(
    PRED_PATH,
    parse_dates=["Inspection Date", "Next_Routine_Inspection_Date"],
)

required = [
    "Restaurant_ID",
    "Inspection Date",
    "Next_Routine_Inspection_Date",
    "Actual_Next_Routine_Failure",
    "Predicted_Failure_Probability",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isna().any().any():
    raise ValueError("Required prediction fields contain missing values.")

if not df["Predicted_Failure_Probability"].between(0, 1).all():
    raise ValueError("Predicted probabilities must be between 0 and 1.")

# One prediction per restaurant and target routine inspection date:
# retain the latest available prediction point before that target.
keys = ["Restaurant_ID", "Next_Routine_Inspection_Date"]

df = df.sort_values(
    ["Inspection Date", "Next_Routine_Inspection_Date"]
)

dedup = df.drop_duplicates(keys, keep="last").copy()

y = dedup["Actual_Next_Routine_Failure"].astype(int).to_numpy()
p = dedup["Predicted_Failure_Probability"].to_numpy()

if len(np.unique(y)) < 2:
    raise ValueError("Evaluation requires both target classes.")

k = max(1, int(np.ceil(0.05 * len(dedup))))
top_idx = np.argsort(p)[::-1][:k]

report = pd.DataFrame([{
    "Rows": len(dedup),
    "Unique_Restaurants": dedup["Restaurant_ID"].nunique(),
    "Failure_Rate": y.mean(),
    "PR_AUC": average_precision_score(y, p),
    "ROC_AUC": roc_auc_score(y, p),
    "Brier_Score": brier_score_loss(y, p),
    "Top_5pct_Count": k,
    "Precision_At_Top_5pct": y[top_idx].mean(),
    "Recall_At_Top_5pct": y[top_idx].sum() / max(1, y.sum()),
}])

print(report.to_string(index=False))
report.to_csv(OUTPUT_PATH, index=False)
print("\nSaved:", OUTPUT_PATH)
