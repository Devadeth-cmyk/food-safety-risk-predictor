"""Train, compare, calibrate and score next-routine-inspection failure models.

Validation: choose among unweighted Logistic Regression, Random Forest and XGBoost
using average rank of PR-AUC (higher is better) and Brier score (lower is better).
Test: evaluate once using a model fitted before validation and calibrated on validation.
Deployment: after the test report is written, fit a deployment model on older labelled
rows and calibrate on a recent temporal holdout; score latest rows for a priority queue.
"""
from pathlib import Path
import time  # Added time module to track model training duration
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6
except ImportError:
    FrozenEstimator = None
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TRAIN_PATH = OUTPUT_DIR / "restaurant_train.csv"
TEST_PATH = OUTPUT_DIR / "restaurant_test.csv"
LABELLED_PATH = OUTPUT_DIR / "restaurant_target_prepared.csv"
SCORING_PATH = OUTPUT_DIR / "restaurant_scoring_latest.csv"
for _path_name in ("restaurant_train.csv", "restaurant_test.csv", "restaurant_target_prepared.csv", "restaurant_scoring_latest.csv"):
    if not (OUTPUT_DIR / _path_name).exists() and (BASE_DIR / _path_name).exists():
        if _path_name == "restaurant_train.csv": TRAIN_PATH = BASE_DIR / _path_name
        elif _path_name == "restaurant_test.csv": TEST_PATH = BASE_DIR / _path_name
        elif _path_name == "restaurant_target_prepared.csv": LABELLED_PATH = BASE_DIR / _path_name
        else: SCORING_PATH = BASE_DIR / _path_name

TARGET = "Target_Next_Routine_Failure"
VALIDATION_START = pd.Timestamp("2021-01-01")
TEST_START = pd.Timestamp("2022-08-04")
RANDOM_STATE = 42
MAX_CALIBRATION_DAYS = 548

# These columns are identifiers, future/label fields, raw free text, or features
# explicitly excluded to reduce leakage/high-cardinality time/geography shortcuts.
EXCLUDE_COLUMNS = {
    "Restaurant_ID", "Inspection Date", "Inspection_Sequence",
    "Next_Result", "Next_Sequence", "Next_Inspection_Date",
    "Next_Routine_Result", "Next_Routine_Sequence",
    "Next_Routine_Inspection_Date", "Target_Gap_Days", TARGET,
    "Violations", "Inspection_Year", "Geo_Grid_ID",
}


def make_X(frame):
    X = frame.drop(columns=list(EXCLUDE_COLUMNS), errors="ignore").copy()
    # Defensive removal of future-looking columns by name.
    X = X.drop(columns=[c for c in X.columns if c.startswith("Next_") or c.startswith("Target_")], errors="ignore")
    return X


def make_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], remainder="drop")


def make_pipeline(estimator, X_fit):
    return Pipeline([
        ("preprocess", make_preprocessor(X_fit)),
        ("model", estimator),
    ])

# TEAM NOTE: Updated to include Top 10% metrics for better inspector capacity planning
def safe_metrics(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    
    def _at_k(frac):
        k = max(1, int(np.ceil(frac * len(y_true))))
        top_idx = np.argsort(scores)[::-1][:k]
        positives = int(y_true.sum())
        return {
            "precision": float(y_true[top_idx].mean()),
            "recall": float(y_true[top_idx].sum() / positives) if positives else np.nan,
            "count": int(k),
        }
        
    top5 = _at_k(0.05)
    top10 = _at_k(0.10)
    roc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan
    
    return {
        "PR_AUC": average_precision_score(y_true, scores),
        "ROC_AUC": roc,
        "Brier_Score": brier_score_loss(y_true, np.clip(scores, 0, 1)),
        "Precision_At_Top_5pct": top5["precision"],
        "Recall_At_Top_5pct": top5["recall"],
        "Precision_At_Top_10pct": top10["precision"],
        "Recall_At_Top_10pct": top10["recall"],
        "Base_Failure_Rate": float(y_true.mean()),
        "Top_5pct_Count": top5["count"],
        "Top_10pct_Count": top10["count"],
    }

def calibrate_prefit(estimator, X_cal, y_cal):
    """Fit sigmoid calibration on a held-out chronological calibration period."""
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(estimator), method="sigmoid"
        )
    else:
        # Compatible with scikit-learn 1.5; cv='prefit' is deprecated in newer releases.
        try:
            calibrated = CalibratedClassifierCV(
                estimator=estimator, method="sigmoid", cv="prefit"
            )
        except TypeError:
            calibrated = CalibratedClassifierCV(
                base_estimator=estimator, method="sigmoid", cv="prefit"
            )
    calibrated.fit(X_cal, y_cal)
    return calibrated

def read_dates(path):
    return pd.read_csv(
        path,
        parse_dates=["Inspection Date", "Next_Routine_Inspection_Date"],
        low_memory=False,
    )

def align_features(X, columns):
    return X.reindex(columns=columns)


for path in [TRAIN_PATH, TEST_PATH, LABELLED_PATH, SCORING_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run target.py and split_data.py first.")

train_df = read_dates(TRAIN_PATH)
test_df = read_dates(TEST_PATH)
all_df = read_dates(LABELLED_PATH)
scoring_df = pd.read_csv(SCORING_PATH, parse_dates=["Inspection Date"], low_memory=False)

for name, frame in [("train", train_df), ("test", test_df), ("all labelled", all_df)]:
    if TARGET not in frame:
        raise ValueError(f"{name} dataset does not contain {TARGET}; rerun target.py.")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{name} target is missing or not binary.")

if train_df["Next_Routine_Inspection_Date"].max() >= TEST_START:
    raise ValueError("Train targets cross into the test period.")
if test_df["Next_Routine_Inspection_Date"].min() < TEST_START:
    raise ValueError("Test targets begin before the test boundary.")

# Chronological fit/validation split inside the pre-test data.
fit_df = train_df[train_df["Next_Routine_Inspection_Date"] < VALIDATION_START].copy()
val_df = train_df[
    (train_df["Next_Routine_Inspection_Date"] >= VALIDATION_START)
    & (train_df["Next_Routine_Inspection_Date"] < TEST_START)
].copy()
if fit_df.empty or val_df.empty:
    raise ValueError("Fit/validation split is empty; check date boundaries.")

X_fit, y_fit = make_X(fit_df), fit_df[TARGET].astype(int)
X_val, y_val = make_X(val_df), val_df[TARGET].astype(int)
X_test, y_test = make_X(test_df), test_df[TARGET].astype(int)
feature_cols = X_fit.columns.tolist()
for name, X in [("validation", X_val), ("test", X_test)]:
    if set(X.columns) != set(feature_cols):
        missing = sorted(set(feature_cols) - set(X.columns))
        extra = sorted(set(X.columns) - set(feature_cols))
        raise ValueError(f"Feature mismatch in {name}. Missing={missing}; extra={extra}")
X_val = align_features(X_val, feature_cols)
X_test = align_features(X_test, feature_cols)

print(f"Rows: fit={len(fit_df):,}; validation={len(val_df):,}; test={len(test_df):,}")
print("Failure rates:", round(y_fit.mean(), 4), round(y_val.mean(), 4), round(y_test.mean(), 4))
print("Features before encoding:", len(feature_cols))

# No class_weight: balanced weights often distort probability estimates.
# Upgraded hyperparameters! Increased tree depth and estimators 
# so XGBoost & Random Forest can capture deeper, more complex risk patterns.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500, tol=1e-3, solver="liblinear", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=16, min_samples_leaf=5, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_STATE,
    ),
    # TEAM NOTE: scale_pos_weight = neg/pos so XGBoost pays more attention to the rarer Fail class
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.80, reg_lambda=1.0,
        min_child_weight=3, eval_metric="logloss", tree_method="hist",
        scale_pos_weight=float((y_fit == 0).sum() / max(1, (y_fit == 1).sum())),
        n_jobs=-1, random_state=RANDOM_STATE,
    ),
}

validation_rows = []
fitted = {}

# Ranking baselines. Scores are rankings, not calibrated probabilities.
if "Results" in X_val:
    row = {"Model": "Current inspection result baseline"}
    row.update(safe_metrics(y_val, pd.to_numeric(X_val["Results"], errors="coerce").fillna(0)))
    validation_rows.append(row)
if "Last_Canvass_Failed" in X_val:
    row = {"Model": "Last Canvass result baseline"}
    row.update(safe_metrics(y_val, pd.to_numeric(X_val["Last_Canvass_Failed"], errors="coerce").fillna(0)))
    validation_rows.append(row)

# Constant training prevalence baseline for probability calibration context.
constant_val = np.full(len(y_val), float(y_fit.mean()))
row = {"Model": "Constant training prevalence baseline"}
row.update(safe_metrics(y_val, constant_val))
validation_rows.append(row)

for name, estimator in models.items():
    print(f"\nFitting validation model: {name}")
    pipe = make_pipeline(estimator, X_fit)
    pipe.fit(X_fit, y_fit)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    row = {"Model": name}
    row.update(safe_metrics(y_val, val_prob))
    validation_rows.append(row)
    fitted[name] = pipe

validation_report = pd.DataFrame(validation_rows)
model_mask = validation_report["Model"].isin(models.keys())
validation_report.loc[model_mask, "PR_AUC_Rank"] = validation_report.loc[model_mask, "PR_AUC"].rank(ascending=False, method="average")
validation_report.loc[model_mask, "Brier_Rank"] = validation_report.loc[model_mask, "Brier_Score"].rank(ascending=True, method="average")
validation_report.loc[model_mask, "Combined_Rank"] = validation_report.loc[model_mask, ["PR_AUC_Rank", "Brier_Rank"]].mean(axis=1)
validation_report.to_csv(OUTPUT_DIR / "model_validation_comparison.csv", index=False)
print("\nValidation comparison (models selected by mean PR-AUC/Brier rank):")
print(validation_report.sort_values("Combined_Rank", na_position="last").to_string(index=False))

ml_report = validation_report[model_mask].sort_values("Combined_Rank")
best_name = str(ml_report.iloc[0]["Model"])
print("\nSelected model:", best_name)

# Honest test evaluation: base estimator fits only on early fit data; sigmoid
# calibration uses the validation period; the test set is untouched until now.
# TEAM NOTE: Calibrate EVERY candidate on validation, score once on untouched test.
# Deployment still uses validation winner only (best_name). This makes the Streamlit
# Model & SHAP table show LR / RF / XGB side-by-side for the professor demo.
test_rows = []
calibrated_models = {}
for name, pipe in fitted.items():
    cal_model = calibrate_prefit(pipe, X_val, y_val)
    calibrated_models[name] = cal_model
    probs = cal_model.predict_proba(X_test)[:, 1]
    row = {"Model": f"{name} (sigmoid-calibrated)"}
    row.update(safe_metrics(y_test, probs))
    test_rows.append(row)

# Keep a handle on the validation-selected model for test_predictions export
calibrated_eval_model = calibrated_models[best_name]
test_prob = calibrated_eval_model.predict_proba(X_test)[:, 1]

# Compare against ranking baselines and a constant-probability Brier baseline.
if "Results" in X_test:
    row = {"Model": "Current inspection result baseline"}
    row.update(safe_metrics(y_test, pd.to_numeric(X_test["Results"], errors="coerce").fillna(0)))
    test_rows.append(row)
if "Last_Canvass_Failed" in X_test:
    row = {"Model": "Last Canvass result baseline"}
    row.update(safe_metrics(y_test, pd.to_numeric(X_test["Last_Canvass_Failed"], errors="coerce").fillna(0)))
    test_rows.append(row)
row = {"Model": "Constant training prevalence baseline"}
row.update(safe_metrics(y_test, np.full(len(y_test), float(train_df[TARGET].mean()))))
test_rows.append(row)

test_report = pd.DataFrame(test_rows)
# Sort to make the dashboard table look clean
test_report = test_report.sort_values(by="PR_AUC", ascending=False)
test_report.to_csv(OUTPUT_DIR / "model_test_metrics.csv", index=False)

prediction_cols = [c for c in ["Restaurant_ID", "Inspection Date", "Next_Routine_Inspection_Date"] if c in test_df]
test_predictions = test_df[prediction_cols].copy()
test_predictions["Actual_Next_Routine_Failure"] = y_test.to_numpy()
test_predictions["Predicted_Failure_Probability"] = test_prob
test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
joblib.dump(calibrated_eval_model, OUTPUT_DIR / "restaurant_failure_evaluation_model.joblib")

print("\nHeld-out test comparison:")
print(test_report.to_string(index=False))
print("Saved evaluation model and test predictions.")

# -------------------------- Deployment model --------------------------
# After the honest test evaluation, train a separate model for current scoring.
# The most recent 18 months of target outcomes are reserved for calibration.
max_target_date = all_df["Next_Routine_Inspection_Date"].max()
deploy_cal_start = max_target_date - pd.Timedelta(days=MAX_CALIBRATION_DAYS)
deploy_fit_df = all_df[all_df["Next_Routine_Inspection_Date"] < deploy_cal_start].copy()
deploy_cal_df = all_df[all_df["Next_Routine_Inspection_Date"] >= deploy_cal_start].copy()
if deploy_fit_df.empty or deploy_cal_df.empty:
    raise ValueError("Deployment fit/calibration split is empty. Check target date range.")
if deploy_fit_df[TARGET].nunique() < 2 or deploy_cal_df[TARGET].nunique() < 2:
    raise ValueError("Deployment fit and calibration sets must each contain both target classes.")

X_deploy_fit, y_deploy_fit = make_X(deploy_fit_df), deploy_fit_df[TARGET].astype(int)
X_deploy_cal, y_deploy_cal = make_X(deploy_cal_df), deploy_cal_df[TARGET].astype(int)
X_score = make_X(scoring_df)
deploy_features = X_deploy_fit.columns.tolist()
for name, X in [("deployment calibration", X_deploy_cal), ("scoring", X_score)]:
    X = X.reindex(columns=deploy_features)
    if name == "deployment calibration":
        X_deploy_cal = X
    else:
        X_score = X

# Clone a fresh estimator of the selected family; do not reuse the evaluation fit.
from sklearn.base import clone
deploy_estimator = clone(models[best_name])
deploy_base = make_pipeline(deploy_estimator, X_deploy_fit)
deploy_base.fit(X_deploy_fit, y_deploy_fit)
deployment_model = calibrate_prefit(deploy_base, X_deploy_cal, y_deploy_cal)

score_prob = deployment_model.predict_proba(X_score)[:, 1]
queue = scoring_df[[c for c in ["Restaurant_ID", "Inspection Date", "Risk", "Latitude", "Longitude", "Results", "Last_Canvass_Failed", "Days_Since_Last_Failure", "Violation_Count", "Hazard_Count"] if c in scoring_df]].copy()
queue["Predicted_Failure_Probability"] = score_prob
queue["Priority_Rank"] = queue["Predicted_Failure_Probability"].rank(method="first", ascending=False).astype(int)
rank_percentile = queue["Predicted_Failure_Probability"].rank(method="first", ascending=False, pct=True)
queue["Relative_Risk_Tier"] = np.select(
    [rank_percentile <= 0.10, rank_percentile <= 0.30],
    ["Red - highest 10%", "Yellow - next 20%"],
    default="Green - remaining 70%",
)
queue = queue.sort_values("Predicted_Failure_Probability", ascending=False)
queue.to_csv(OUTPUT_DIR / "restaurant_risk_priority_queue.csv", index=False)
joblib.dump(deployment_model, OUTPUT_DIR / "restaurant_failure_model.joblib")
# Save the uncalibrated base pipeline separately for feature attribution / SHAP.
joblib.dump(deploy_base, OUTPUT_DIR / "restaurant_failure_base_estimator.joblib")

# Extracting clean feature names post-preprocessing so SHAP plots 
# in the next script have accurate labels instead of raw array indices.
try:
    transformed_feature_names = deploy_base.named_steps["preprocess"].get_feature_names_out().tolist()
except Exception:
    transformed_feature_names = deploy_features

metadata = {
    "selected_model": best_name,
    "target": TARGET,
    "target_definition": "failure at next Canvass inspection after current event; intervening event types allowed",
    "minimum_target_gap_days": 1,
    "maximum_target_gap_days": MAX_CALIBRATION_DAYS,
    "deployment_calibration_start": str(deploy_cal_start.date()),
    "deployment_calibration_end": str(max_target_date.date()),
    "feature_columns": deploy_features,
    "transformed_feature_names": transformed_feature_names,  # Added for SHAP!
    "risk_tiers": "relative ranking: top 10% red, next 20% yellow, remaining 70% green",
}
joblib.dump(metadata, OUTPUT_DIR / "restaurant_failure_model_metadata.joblib")
print("\nDeployment model saved:", OUTPUT_DIR / "restaurant_failure_model.joblib")
print("Priority queue saved:", OUTPUT_DIR / "restaurant_risk_priority_queue.csv")
print("Deployment fit rows:", len(deploy_fit_df), "calibration rows:", len(deploy_cal_df))
print("Scored latest restaurants:", len(queue))
print("Risk tier counts:\n", queue["Relative_Risk_Tier"].value_counts())