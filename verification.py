"""
Verify SHAP artifacts and create a geographic-grid risk summary.

Run from the food_inspections project folder:
    python verify_shap_and_grid_risk.py

Inputs expected beside this script:
- explain_model.py
- restaurant_failure_base_estimator.joblib
- restaurant_failure_model_metadata.joblib
- restaurant_scoring_latest.csv
- restaurant_risk_priority_queue_corrected.csv
"""
from pathlib import Path
import subprocess
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def first_existing(*names):
    for name in names:
        for p in (OUTPUT_DIR / name, BASE_DIR / name):
            if p.exists(): return p
    return None

# 1. Regenerate SHAP from the saved base estimator and metadata.
explain_script = first_existing("explain_model.py")
if explain_script is None:
    raise FileNotFoundError("explain_model.py is missing from the project folder.")

print("Step 1/3: Regenerating SHAP outputs using the saved base estimator...")
subprocess.run([sys.executable, str(explain_script)], cwd=BASE_DIR, check=True)

shap_csv = OUTPUT_DIR / "shap_feature_importance.csv"
shap_png = OUTPUT_DIR / "shap_summary.png"
if not shap_csv.exists() or not shap_png.exists():
    raise FileNotFoundError("SHAP script finished but expected output files were not created.")

shap_df = pd.read_csv(shap_csv)
if not {"Feature", "Mean_Absolute_SHAP"}.issubset(shap_df.columns):
    raise ValueError("Unexpected SHAP CSV schema.")
if shap_df.empty or shap_df["Mean_Absolute_SHAP"].isna().all():
    raise ValueError("SHAP importance output is empty or invalid.")

print(f"SHAP verification: {len(shap_df)} features; top feature = {shap_df.iloc[0]['Feature']}")
print(f"SHAP CSV: {shap_csv}")
print(f"SHAP plot: {shap_png}")
print("Note: these SHAP values explain the saved base estimator, not necessarily calibrated probabilities.")

# 2. Load queue and latest features, then join the geographic grid identifier.
queue_path = first_existing(
    "restaurant_risk_priority_queue_corrected.csv",
    "restaurant_risk_priority_queue.csv",
)
scoring_path = first_existing("restaurant_scoring_latest.csv")
if queue_path is None:
    raise FileNotFoundError("Could not find corrected or original restaurant risk queue CSV.")
if scoring_path is None:
    raise FileNotFoundError("restaurant_scoring_latest.csv is missing.")

queue = pd.read_csv(queue_path, low_memory=False)
scoring = pd.read_csv(scoring_path, low_memory=False)

required_queue = {"Restaurant_ID", "Latitude", "Longitude", "Predicted_Failure_Probability"}
required_scoring = {"Restaurant_ID", "Geo_Grid_ID"}
if missing := required_queue - set(queue.columns):
    raise ValueError(f"Queue is missing columns: {sorted(missing)}")
if missing := required_scoring - set(scoring.columns):
    raise ValueError(f"Scoring file is missing columns: {sorted(missing)}")

grid_lookup = scoring[["Restaurant_ID", "Geo_Grid_ID"]].drop_duplicates("Restaurant_ID")
joined = queue.merge(grid_lookup, on="Restaurant_ID", how="left", validate="many_to_one")
joined["Predicted_Failure_Probability"] = pd.to_numeric(
    joined["Predicted_Failure_Probability"], errors="coerce"
)
joined["Latitude"] = pd.to_numeric(joined["Latitude"], errors="coerce")
joined["Longitude"] = pd.to_numeric(joined["Longitude"], errors="coerce")
joined = joined.dropna(subset=[
    "Geo_Grid_ID", "Latitude", "Longitude", "Predicted_Failure_Probability"
]).copy()

# 3. Aggregate by geographic grid. This is a grid-level proxy, not an official neighborhood.
grid = (
    joined.groupby("Geo_Grid_ID", dropna=False)
    .agg(
        Restaurant_Count=("Restaurant_ID", "nunique"),
        Mean_Predicted_Failure_Probability=("Predicted_Failure_Probability", "mean"),
        Median_Predicted_Failure_Probability=("Predicted_Failure_Probability", "median"),
        High_Risk_Restaurants=("Relative_Risk_Tier", lambda s: (s == "Red - highest 10%").sum())
            if "Relative_Risk_Tier" in joined.columns
            else ("Restaurant_ID", "size"),
        Latitude=("Latitude", "mean"),
        Longitude=("Longitude", "mean"),
    )
    .reset_index()
    .sort_values("Mean_Predicted_Failure_Probability", ascending=False)
)
grid.to_csv(OUTPUT_DIR / "geographic_grid_risk_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    grid["Longitude"],
    grid["Latitude"],
    c=grid["Mean_Predicted_Failure_Probability"],
    s=(grid["Restaurant_Count"].clip(lower=1) ** 0.5) * 14,
    alpha=0.8,
)
ax.set_title("Chicago Food Safety Risk by Geographic Grid")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
cb = fig.colorbar(scatter, ax=ax)
cb.set_label("Mean predicted failure probability")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "geographic_grid_risk_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Grid summary rows: {len(grid):,}")
print("Saved:", OUTPUT_DIR / "geographic_grid_risk_summary.csv")
print("Saved:", OUTPUT_DIR / "geographic_grid_risk_heatmap.png")
print("IMPORTANT: Geo_Grid_ID is an internal grid, not an official Chicago neighborhood/community-area label.")
