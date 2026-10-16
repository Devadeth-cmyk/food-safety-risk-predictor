"""Week 2 analytics for Restaurant Health Inspection Failure Predictor.

Place this script in the same folder as:
  - restaurant_feature_engineering_base.csv
  - restaurant_risk_priority_queue.csv

Outputs:
  - seasonal_failure_by_month.csv
  - seasonal_failure_by_month.png
  - geographic_risk_heatmap.png
  - risk_queue_audit.csv

This creates a coordinate-based risk heatmap, not a neighborhood-boundary map.
"""
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
BASE_CANDIDATES = [
    BASE_DIR / 'restaurant_feature_engineering_base.csv',
    BASE_DIR / 'data' / 'restaurant_feature_engineering_base.csv',
]
QUEUE_CANDIDATES = [
    BASE_DIR / 'outputs' / 'restaurant_risk_priority_queue_corrected.csv',
    OUTPUT_DIR / 'restaurant_risk_priority_queue.csv',
    BASE_DIR / 'restaurant_risk_priority_queue.csv',
    BASE_DIR / 'restaurant_risk_priority_queue(1).csv',
    BASE_DIR / 'data' / 'restaurant_risk_priority_queue.csv',
]

def find_existing(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label}. Expected one of:\n" + '\n'.join(map(str, candidates))
    )

base_path = find_existing(BASE_CANDIDATES, 'restaurant_feature_engineering_base.csv')
queue_path = find_existing(QUEUE_CANDIDATES, 'restaurant_risk_priority_queue.csv')
print(f'Loading inspection data: {base_path}')
print(f'Loading risk queue: {queue_path}')

# ---- 1) Seasonal/monthly failure analysis ----
# Read only columns needed, keeping memory usage lower than loading the full dataset.
header = pd.read_csv(base_path, nrows=0).columns.tolist()
needed = [c for c in ['Inspection Date', 'Results', 'Inspection Type'] if c in header]
if not {'Inspection Date', 'Results'}.issubset(needed):
    raise ValueError('Base dataset must contain Inspection Date and Results columns.')
inspections = pd.read_csv(base_path, usecols=needed, low_memory=False)
inspections['Inspection Date'] = pd.to_datetime(inspections['Inspection Date'], errors='coerce')
inspections['Results'] = pd.to_numeric(inspections['Results'], errors='coerce')
inspections = inspections.dropna(subset=['Inspection Date', 'Results'])
inspections = inspections[inspections['Results'].isin([0, 1])].copy()

# Align seasonal analysis with the project's target where possible: routine Canvass events.
if 'Inspection Type' in inspections.columns:
    canvass_mask = inspections['Inspection Type'].astype(str).str.strip().str.lower().eq('canvass')
    if canvass_mask.any():
        seasonal = inspections.loc[canvass_mask].copy()
        analysis_scope = 'Canvass inspections only'
    else:
        warnings.warn('No Inspection Type values matched exactly to Canvass; using all valid inspections.')
        seasonal = inspections.copy()
        analysis_scope = 'All valid inspection types (no exact Canvass matches found)'
else:
    seasonal = inspections.copy()
    analysis_scope = 'All valid inspection types (Inspection Type column unavailable)'

seasonal['Month_Number'] = seasonal['Inspection Date'].dt.month
seasonal['Month'] = seasonal['Inspection Date'].dt.month_name()
monthly = (
    seasonal.groupby('Month_Number', as_index=False)
    .agg(Inspection_Count=('Results', 'size'), Failure_Count=('Results', 'sum'), Failure_Rate=('Results', 'mean'))
    .sort_values('Month_Number')
)
month_names = {i: pd.Timestamp(2000, i, 1).strftime('%B') for i in range(1, 13)}
monthly['Month'] = monthly['Month_Number'].map(month_names)
monthly['Failure_Rate_Percent'] = monthly['Failure_Rate'] * 100
monthly['Analysis_Scope'] = analysis_scope
monthly.to_csv(OUTPUT_DIR / 'seasonal_failure_by_month.csv', index=False)

plt.figure(figsize=(11, 5.5))
plt.plot(monthly['Month'], monthly['Failure_Rate_Percent'], marker='o')
plt.title(f'Monthly Inspection Failure Rate\n{analysis_scope}')
plt.xlabel('Month')
plt.ylabel('Failure rate (%)')
plt.xticks(rotation=35, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'seasonal_failure_by_month.png', dpi=180, bbox_inches='tight')
plt.close()

# ---- 2) Audit risk queue labels/tiers and rank ordering ----
queue = pd.read_csv(queue_path, low_memory=False)
required = {'Restaurant_ID', 'Latitude', 'Longitude', 'Predicted_Failure_Probability', 'Priority_Rank', 'Relative_Risk_Tier'}
missing = required - set(queue.columns)
if missing:
    raise ValueError(f'Risk queue missing required columns: {sorted(missing)}')
queue['Predicted_Failure_Probability'] = pd.to_numeric(queue['Predicted_Failure_Probability'], errors='coerce')
queue['Latitude'] = pd.to_numeric(queue['Latitude'], errors='coerce')
queue['Longitude'] = pd.to_numeric(queue['Longitude'], errors='coerce')
queue['Priority_Rank'] = pd.to_numeric(queue['Priority_Rank'], errors='coerce')
queue = queue.dropna(subset=['Predicted_Failure_Probability', 'Latitude', 'Longitude']).copy()
queue = queue.sort_values('Predicted_Failure_Probability', ascending=False).reset_index(drop=True)

n = len(queue)
red_end = int(np.ceil(n * 0.10))
yellow_end = int(np.ceil(n * 0.30))
queue['Expected_Relative_Risk_Tier'] = np.where(
    queue.index < red_end, 'Red - highest 10%',
    np.where(queue.index < yellow_end, 'Yellow - next 20%', 'Green - remaining 70%')
)
queue['Tier_Matches_Expected'] = queue['Relative_Risk_Tier'].eq(queue['Expected_Relative_Risk_Tier'])
queue['Probability_Rank_Matches'] = queue['Priority_Rank'].eq(np.arange(1, n + 1))

# Save concise audit, retaining the source Risk label if present for comparison.
audit_cols = [c for c in [
    'Restaurant_ID', 'Predicted_Failure_Probability', 'Priority_Rank',
    'Relative_Risk_Tier', 'Expected_Relative_Risk_Tier', 'Tier_Matches_Expected',
    'Probability_Rank_Matches', 'Risk'
] if c in queue.columns]
queue[audit_cols].to_csv(OUTPUT_DIR / 'risk_queue_audit.csv', index=False)
print('\n--- Seasonal analysis ---')
print('Scope:', analysis_scope)
print(monthly[['Month', 'Inspection_Count', 'Failure_Count', 'Failure_Rate_Percent']].to_string(index=False, formatters={'Failure_Rate_Percent': '{:.2f}%'.format}))
print('\nSaved seasonal_failure_by_month.csv and seasonal_failure_by_month.png')
print('\n--- Risk queue audit ---')
print(f'Rows with valid coordinates/probabilities: {n:,}')
print(f'Tier labels matching top-10/next-20/rest-70 rule: {queue["Tier_Matches_Expected"].mean():.1%}')
print(f'Priority ranks matching probability order: {queue["Probability_Rank_Matches"].mean():.1%}')
if 'Risk' in queue.columns:
    print('\nRisk label vs relative tier (these may be different classification rules):')
    print(pd.crosstab(queue['Risk'], queue['Relative_Risk_Tier']).to_string())

# ---- 3) Coordinate-based geographic risk heatmap ----
# The map is a density-style scatter of restaurant locations; no neighborhood boundaries are inferred.
plt.figure(figsize=(9, 8))
scatter = plt.scatter(
    queue['Longitude'], queue['Latitude'],
    c=queue['Predicted_Failure_Probability'],
    s=7, alpha=0.55, cmap='YlOrRd', vmin=0, vmax=1, linewidths=0
)
plt.colorbar(scatter, label='Predicted next-Canvass failure probability')
plt.title('Chicago Restaurant Predicted Risk by Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'geographic_risk_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print('\nSaved geographic_risk_heatmap.png and risk_queue_audit.csv')
print('Note: this is a coordinate-based risk map, not a neighborhood-boundary heatmap.')
