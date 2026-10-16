"""Create SHAP explanations for the fitted deployment base estimator.

Requires the model_building.py run to have produced:
- restaurant_failure_base_estimator.joblib
- restaurant_failure_model_metadata.joblib
- restaurant_scoring_latest.csv
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

try:
    import shap
except ImportError as exc:
    raise SystemExit("SHAP is not installed. Run: python -m pip install shap") from exc

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
BASE_MODEL_PATH = OUTPUT_DIR / "restaurant_failure_base_estimator.joblib"
META_PATH = OUTPUT_DIR / "restaurant_failure_model_metadata.joblib"
SCORING_PATH = OUTPUT_DIR / "restaurant_scoring_latest.csv"
for _name in ("restaurant_failure_base_estimator.joblib", "restaurant_failure_model_metadata.joblib", "restaurant_scoring_latest.csv"):
    if not (OUTPUT_DIR / _name).exists() and (BASE_DIR / _name).exists():
        if _name.endswith("base_estimator.joblib"): BASE_MODEL_PATH = BASE_DIR / _name
        elif _name.endswith("metadata.joblib"): META_PATH = BASE_DIR / _name
        else: SCORING_PATH = BASE_DIR / _name

for path in [BASE_MODEL_PATH, META_PATH, SCORING_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run model_building.py first.")

base_pipeline = joblib.load(BASE_MODEL_PATH)
metadata = joblib.load(META_PATH)
scoring_df = pd.read_csv(SCORING_PATH, low_memory=False)

exclude = {
    "Restaurant_ID", "Inspection Date", "Inspection_Sequence",
    "Next_Result", "Next_Sequence", "Next_Inspection_Date",
    "Next_Routine_Result", "Next_Routine_Sequence",
    "Next_Routine_Inspection_Date", "Target_Gap_Days",
    "Target_Next_Routine_Failure", "Violations", "Inspection_Year", "Geo_Grid_ID",
}
X = scoring_df.drop(columns=list(exclude), errors="ignore").copy()
X = X.drop(columns=[c for c in X.columns if c.startswith("Next_") or c.startswith("Target_")], errors="ignore")
feature_cols = metadata["feature_columns"]
X = X.reindex(columns=feature_cols)

# A few hundred rows are enough for a readable global explanation and avoid high RAM use.
X_sample = X.sample(n=min(500, len(X)), random_state=42)
preprocessor = base_pipeline.named_steps["preprocess"]
estimator = base_pipeline.named_steps["model"]
X_transformed = preprocessor.transform(X_sample)
if sparse.issparse(X_transformed):
    X_transformed = X_transformed.toarray()
feature_names = preprocessor.get_feature_names_out()

if hasattr(estimator, "feature_importances_"):
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X_transformed)
else:
    # Logistic Regression fallback if validation selects it.
    background = X_transformed[: min(100, len(X_transformed))]
    explainer = shap.LinearExplainer(estimator, background)
    values = explainer.shap_values(X_transformed)

if isinstance(values, list):
    values = values[-1]
values = np.asarray(values)
if values.ndim == 3:
    values = values[:, :, 1] if values.shape[-1] > 1 else values[:, :, 0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": np.abs(values).mean(axis=0),
}).sort_values("Mean_Absolute_SHAP", ascending=False)
importance.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)

plt.figure()
shap.summary_plot(
    values,
    X_transformed,
    feature_names=feature_names,
    max_display=25,
    show=False,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
plt.close("all")

print("Saved:", OUTPUT_DIR / "shap_feature_importance.csv")
print("Saved:", OUTPUT_DIR / "shap_summary.png")
print("Top features:\n", importance.head(15).to_string(index=False))
