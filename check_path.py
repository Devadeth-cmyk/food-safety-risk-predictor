
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SHAP_DIR = ARTIFACTS / "shap_analysis"

checks = {
    "Primary dataset": [
        ROOT / "restaurant_feature_engineered_encoded_full.csv",
        ROOT / "restaurant_feature_engineered_fixed.csv",
    ],
    "Encoded features": [
        ROOT / "X_full_encoded.csv",
    ],
    "Target labels": [
        ROOT / "y_full.csv",
    ],
    "Model": [
        ARTIFACTS / "best_restaurant_failure_model.joblib",
        ROOT / "best_restaurant_failure_model.joblib",
    ],
    "Metadata": [
        ARTIFACTS / "model_metadata.json",
        ROOT / "model_metadata.json",
    ],
    "Test metrics": [
        ARTIFACTS / "test_metrics.json",
        ROOT / "test_metrics.json",
    ],
    "Model comparison": [
        ARTIFACTS / "model_comparison.csv",
        ROOT / "model_comparison.csv",
    ],
    "Test predictions": [
        ARTIFACTS / "test_predictions.csv",
        ROOT / "test_predictions.csv",
    ],
}

print(f"\nProject root: {ROOT}\n")

for label, paths in checks.items():
    found = next((path for path in paths if path.exists()), None)

    if found:
        print(f"[FOUND]   {label}: {found.relative_to(ROOT)}")
    else:
        print(f"[MISSING] {label}")
        for path in paths:
            print(f"          Expected: {path.relative_to(ROOT)}")

print("\n--- SHAP files ---")

expected_images = [
    "shap_summary_bar.png",
    "shap_beeswarm.png",
    "shap_waterfall.png",
]

for filename in expected_images:
    path = SHAP_DIR / filename
    print(f"[{'FOUND' if path.exists() else 'MISSING'}] {path.relative_to(ROOT)}")

print("\n--- Dashboard exports ---")

exports_dir = ARTIFACTS / "dashboard_exports"

if exports_dir.exists():
    for path in sorted(exports_dir.iterdir()):
        if path.is_file():
            print(f"[FOUND] {path.relative_to(ROOT)}")
else:
    print(f"[MISSING] {exports_dir.relative_to(ROOT)}")

print("\n--- SHAP directory contents ---")

if SHAP_DIR.exists():
    for path in sorted(SHAP_DIR.iterdir()):
        print(path.name)
else:
    print("SHAP directory does not exist.")
