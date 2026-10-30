# 🍔 Chicago Restaurant Health Inspection Risk Prediction System

An AI-powered decision-support system that predicts the likelihood of Chicago restaurants failing their next routine health inspection. The application helps inspection teams prioritize restaurants using machine learning, historical inspection data, geographic risk analysis, and explainable AI.

## 📌 Project Overview

Restaurant health inspections are essential for maintaining food safety and public health. However, inspection resources are limited, making it important to identify restaurants that may require closer attention.

This project uses historical restaurant inspection data to estimate the probability of failure at a future routine (Canvass) inspection.

The system presents these predictions through an interactive Streamlit dashboard, allowing users to explore restaurant-level risk, identify high-priority inspections, analyze neighborhood-level trends, and understand important model features.

**The model provides risk estimates to support inspection planning. Predictions do not guarantee future inspection outcomes.**

---

## ✨ Key Features

### 1. Overview Dashboard
- Displays overall inspection and risk summaries.
- Provides a high-level view of restaurant risk.
- Helps users understand the distribution of predicted risks.

### 2. Geographic Risk Map
- Visualizes restaurant locations and predicted risks.
- Helps identify geographic concentrations of higher-risk restaurants.
- Uses PyDeck for geospatial visualization.

### 3. Inspector Priority Queue
- Ranks restaurants using predicted failure probability.
- Displays restaurant ID, risk tier, predicted failure probability, inspection date, violation count, latitude, longitude, and inspection result.
- Supports filtering restaurants using dashboard controls.
- Helps inspectors identify restaurants that may require earlier attention.
- Supports downloading the displayed queue as a CSV file.

### 4. Restaurant Report Card
- Allows users to search for a restaurant using its ID.
- Displays predicted failure risk, risk tier, and priority rank.
- Uses a gauge chart to visualize predicted failure probability.
- Displays the restaurant's geographic location.
- Helps users examine an individual restaurant without searching the entire queue.

### 5. Violation Analytics
- Explores historical violation patterns.
- Helps users understand how inspection violations relate to restaurant risk.
- Supports analysis of inspection-related trends when the required data is available.

### 6. Community Area Risk Analysis
- Aggregates restaurant risk by geographic area.
- Displays average predicted failure risk by neighborhood.
- Provides controls for minimum restaurant counts and the number of areas displayed.
- Shows inspection records, high-risk restaurants, restaurant counts, historical failure rates, and mean predicted failure probabilities.
- Supports exporting neighborhood summaries to CSV.

### 7. Model Evaluation
- Compares candidate machine learning models and baseline strategies.
- Reports ranking and probability-quality metrics.
- Supports evaluation of the selected model using held-out test data.

### 8. SHAP Explainability
- Uses SHAP feature importance to explain which features influence model predictions.
- Helps identify influential factors such as previous failures, violation counts, and time since the previous inspection.
- Supports model interpretation rather than treating predictions as unexplained outputs.

---

## 🧠 Machine Learning Approach

The project predicts whether a restaurant will fail its next eligible routine inspection.

### Target Variable

`Target_Next_Routine_Failure`

The target represents the failure outcome at the next eligible routine inspection, according to the project's target-generation logic.

### Feature Engineering

The model uses historical inspection information and engineered features. Examples include:

- `Days_Since_Previous_Inspection`
- `Previous_Failure_Count`
- `Hazard_Count`
- `Previous_Violation_Count`
- `Historical_Failure_Rate`
- `Inspection_Year`
- `Administrative_Count`
- `Violation_Count`
- Inspection-type and temporal features

The exact feature set depends on the trained model artifact.

### Model Training

The project benchmark compares the following approaches:

- Calibrated Random Forest
- Calibrated Logistic Regression
- Calibrated XGBoost
- Last Canvass Result baseline
- Current Inspection Result baseline
- Constant Training Prevalence baseline

The supplied benchmark identifies **calibrated XGBoost** as the model with the strongest PR-AUC and ROC-AUC among these candidates.

### Prediction Workflow

1. Historical inspection records are prepared.
2. Features are engineered from inspection history.
3. The target variable is generated.
4. Data is divided into training, validation, and test sets.
5. Candidate models are trained and evaluated.
6. The selected model produces failure-probability estimates.
7. A classification threshold can convert probabilities into predicted Pass/Fail classes.
8. Predicted risks are used to rank restaurants in the priority queue.
9. The dashboard displays restaurant-level and neighborhood-level results.
10. SHAP analysis helps explain feature importance.

**Note:** The classification threshold used for Pass/Fail predictions is distinct from the relative risk tiers used for inspection prioritization.

---

## 📊 Model Evaluation

The selected model is evaluated using classification metrics and ranking metrics to measure its ability to predict restaurant inspection failures.

### Evaluation Metrics

| Metric | Value | Interpretation |
|---|---:|---|
| Accuracy | 78.3% | Percentage of predictions classified correctly. |
| Precision | 68.7% | Of the restaurants predicted to fail, 68.7% actually failed. |
| Recall | 82.6% | Of the restaurants that actually failed, 82.6% were correctly identified. |
| F1 Score | 0.750 | Harmonic mean of precision and recall, balancing both metrics. |
| PR-AUC | 0.553 | Measures performance across the precision–recall curve, particularly useful for imbalanced classification. |
| ROC-AUC | 0.742 | Measures the model's ability to distinguish between failure and non-failure cases across classification thresholds. |

**Selected classification threshold:** `0.30`

### Interpretation

- **Accuracy (78.3%):** The model correctly classifies approximately 78 out of every 100 evaluated cases.
- **Precision (68.7%):** Approximately 69 out of every 100 restaurants predicted to fail are actual failures.
- **Recall (82.6%):** The model identifies approximately 83 out of every 100 actual failures.
- **F1 Score (0.750):** Provides a combined measure of precision and recall.
- **PR-AUC (0.553):** Summarizes precision and recall across different thresholds. Its interpretation depends on the prevalence of failures in the evaluation dataset.
- **ROC-AUC (0.742):** Indicates useful discrimination between restaurants that fail and those that do not, although performance is not perfect.

### Why Recall Matters

In restaurant inspection risk prediction, missing a restaurant that is likely to fail can be costly. A relatively high recall helps identify more potential failures for further inspection.

However, precision also matters because a low precision score means some restaurants prioritized by the model may not fail their next inspection.

The classification threshold controls the trade-off between precision and recall. Lowering the threshold generally identifies more potential failures but may also increase false positives.

### Important Distinction

The classification metrics above evaluate predicted classes at the selected threshold. PR-AUC and ROC-AUC evaluate model discrimination across a range of thresholds.

The classification threshold of `0.30` is separate from the dashboard's relative risk tiers, which rank restaurants by predicted failure probability.

These metrics should be reported as final test results only if they have been verified against the actual held-out test labels and predictions.

---

## 🔍 Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) helps explain how input features contribute to model predictions.

The feature-importance analysis can highlight factors such as:

- Time since the previous inspection.
- Previous failure count.
- Hazard count.
- Previous violation count.
- Historical failure rate.
- Inspection-related features.

### Why SHAP Is Useful

- Improves transparency.
- Helps users understand which features influence predictions.
- Supports model debugging and analysis.
- Provides additional context for interpreting restaurant risk.

SHAP importance does not establish that a feature causes an inspection failure. It describes how the feature contributes to the model's predictions.

---

## 🗺️ Neighborhood Risk Analysis

The neighborhood analysis groups restaurants by geographic location and calculates aggregate risk measures.

Depending on the available input data, the analysis includes:

- Number of inspection records.
- Number of high-risk restaurants.
- Number of restaurants represented.
- Historical failure rate.
- Mean predicted failure probability.

The dashboard allows users to adjust the minimum number of restaurants represented in an area and the number of areas displayed.

This helps users compare geographic risk patterns while reducing the chance of overinterpreting areas with very few observations.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Probability Calibration | Scikit-learn calibration utilities |
| Model Persistence | Joblib |
| Model Explainability | SHAP |
| Dashboard | Streamlit |
| Interactive Charts | Plotly |
| Geographic Visualization | PyDeck |
| Data Formats | CSV, JSON, Joblib |

The dashboard version described here uses PyDeck for geographic visualization. Folium is not listed as a required dependency for that implementation.

---

## 📁 Project Structure

The following is a representative structure. Confirm filenames against the actual repository before running individual scripts.

```text
food-safety-risk-predictor/
│
├── app.py
│
├── target.py
├── split_data.py
├── model_building.py
├── explain_model.py
├── audit_test_predictions.py
├── community_area_analysis.py
├── make_seasonal_csv.py
│
├── requirements.txt
├── README.md
│
└── outputs/
    ├── restaurant_risk_priority_queue_corrected.csv
    ├── restaurant_scoring_latest.csv
    ├── restaurant_failure_model.joblib
    ├── restaurant_failure_base_estimator.joblib
    ├── restaurant_failure_model_metadata.joblib
    ├── shap_feature_importance.csv
    └── model evaluation and neighborhood summary files
