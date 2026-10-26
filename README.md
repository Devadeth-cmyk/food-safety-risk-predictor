# 🍔 Chicago Restaurant Health Inspection Risk System

> **An AI-powered predictive decision-support system and interactive GIS risk map designed for city health inspectors and public safety managers.**

---

## 📌 Project Overview

Traditional restaurant health inspection workflows often rely on static schedules or simple reactive heuristics (e.g., inspecting facilities that failed previously). This project builds a **point-in-time, leakage-free machine learning pipeline** to predict whether a Chicago restaurant will fail its **next routine (Canvass) health inspection** within a **1 to 548-day window** based *strictly on past compliance history*.

By prioritizing high-risk facilities, city health officers can allocate inspection resources far more efficiently—increasing failure detection rates prior to foodborne illness outbreaks.

---

## ✨ Key Features & Interactive Dashboard Tabs

The project includes an **8-tab interactive Streamlit web dashboard** with full **Light/Dark Mode responsive UI/UX**:

1. **📊 Overview**: High-level KPI metrics, citywide risk distribution histograms, and priority tier breakdowns (Red: Top 10%, Yellow: Next 20%, Green: Remaining 70%).
2. **🗺️ Interactive Risk Map**:
   * **Standard 2D Scatter Mode**: Color-coded facility location markers.
   * **3D Hexagon Density Mode**: Extruded 3D towers showing geographic risk concentration across Chicago neighborhoods (rendered with CartoDB tiles).
3. **📋 Inspector Priority Queue**: Operational priority queue sorted by predicted failure probability, featuring progress bars and one-click CSV export.
4. **🏢 Facility Profile & Report Card**: Facility lookup tool featuring a **Speedometer Risk Gauge Chart**, plain-English compliance history summary, localized map, and single-facility report download.
5. **🧪 What-If ML Risk Simulator**: Interactive compliance simulator allowing managers to adjust sliders (violations, hazards, days since failure) and observe real-time model probability shifts.
6. **📅 Seasonal Trends**: Monthly historical inspection failure analysis identifying seasonal spikes in food safety violations.
7. **🏙️ Community Areas**: Chicago neighborhood-level risk aggregation with sample size filtering and area summary exports.
8. **🧠 Model & SHAP Explainability**: Side-by-side model evaluation metrics table, global SHAP feature importance bar charts, and constrained individual waterfall plots explaining specific high-risk predictions.

---

## 📈 Model Performance & Evaluation Results

All models were evaluated on an **untouched, held-out temporal test set** (inspections occurring post-August 4, 2022). Probability outputs were calibrated using **Sigmoid Calibration (Platt Scaling)** on a chronological validation holdout.

| Model / Baseline | PR-AUC ⬆️ | ROC-AUC ⬆️ | Brier Score ⬇️ | Precision @ Top 5% ⬆️ | Precision @ Top 10% ⬆️ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (sigmoid-calibrated)** 🏆 | **0.3796** | **0.6113** | **0.1988** | **46.48%** | **43.88%** |
| **Logistic Regression (sigmoid-calibrated)** | 0.3702 | 0.5934 | 0.2003 | 47.27% | 43.18% |
| **XGBoost (sigmoid-calibrated)** | 0.3522 | 0.5866 | 0.2022 | 41.64% | 38.69% |
| *Last Canvass Result Baseline* | 0.3060 | 0.5376 | 0.3983 | 31.07% | 31.91% |
| *Current Inspection Result Baseline* | 0.3008 | 0.5269 | 0.3737 | 31.95% | 32.57% |
| *Constant Training Prevalence Baseline* | 0.2881 | 0.5000 | 0.2073 | 31.78% | 31.91% |

### 💡 Key Takeaway:
* **Base Citywide Failure Rate**: **~28.8%** (random inspection).
* **Top 10% High-Priority Queue Precision**: **43.9%** (+52% efficiency boost over random selection).
* **Top 5% High-Priority Queue Precision**: **46.5%** (+61% efficiency boost over random selection).

---

## 🛡️ Leakage-Free Design Principles

To prevent data leakage (a common issue in food safety ML research):
* **No Future Features**: Features are computed strictly point-in-time *prior* to the next inspection.
* **Target Isolation**: Target is defined as failure at the *next Canvass inspection* within 1 to 548 days.
* **Strict Temporal Splitting**:
  * **Train Set**: Inspection targets prior to Jan 1, 2021.
  * **Validation Set**: Targets between Jan 1, 2021, and Aug 4, 2022.
  * **Test Set**: Targets post-Aug 4, 2022.

---

## 📁 Repository Structure

```text
food-safety-risk-predictor/
├── outputs/                                # Generated model joblib files, metrics, and queue CSVs
├── app.py                                  # Streamlit MVP Dashboard application
├── target.py                               # Prepares target labels and point-in-time features
├── split_data.py                           # Creates temporal train/validation/test datasets
├── model_building.py                       # Trains, calibrates (Sigmoid), evaluates models & generates queue
├── explain_model.py                        # Computes SHAP feature importance & waterfall visualizations
├── audit_test_predictions.py              # Audits deduplicated test set predictions
├── community_area_analysis.py              # Aggregates neighborhood-level community risk metrics
├── make_seasonal_csv.py                    # Generates monthly seasonal trend aggregations
├── requirements.txt                        # Python dependency manifest
└── README.md                               # Project documentation

```
---

## 💻 Installation & Setup Guide

Follow these step-by-step instructions to set up the environment, run the ML pipeline, and launch the dashboard locally.

---

### 1. Prerequisites

Make sure you have the following installed on your machine:
* **Python**: Version 3.9, 3.10, or 3.11
* **Git**: Installed and configured
* **Terminal**: Windows PowerShell (or Command Prompt / Bash)

---

### 2. Clone the Repository

Open PowerShell and clone the project:

```powershell
git clone https://github.com/YOUR-USERNAME/food-safety-risk-predictor.git
cd food-safety-risk-predictor
```

---

### 3. Create & Activate a Virtual Environment
Creating a virtual environment ensures all library dependencies remain isolated and compatible.

On Windows (PowerShell):

```PowerShell

python -m venv venv
.\venv\Scripts\Activate
(If you encounter an execution policy error on PowerShell, run Set-ExecutionPolicy Unrestricted -Scope Process first, then activate).
```
On macOS / Linux:

```Bash

python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Dependencies
Install all required Python packages:

```PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🚀 Pipeline Execution Order
To reproduce the model evaluation, SHAP plots, and dashboard data files from scratch, run the scripts in the following order:

## Step 1: Feature Engineering & Target Preparation
Builds point-in-time features and historical compliance indicators without data leakage.

```PowerShell

python target.py
```
---

## Step 2: Temporal Data Splitting
Creates chronological train, validation, and held-out test datasets based on inspection dates.

```PowerShell

python split_data.py
```
---

## Step 3: Model Training, Calibration & Queue Scoring
Trains Logistic Regression, Random Forest, and XGBoost models; applies Sigmoid Probability Calibration; evaluates against baselines; and exports the prioritized inspector queue.

```PowerShell

python model_building.py
```
---

## Step 4: Model Explainability (SHAP Plots)
Calculates global feature importance rankings and generates individual waterfall explanation plots.

```PowerShell

python explain_model.py
```
---

## Step 5: Neighborhood & Seasonal Aggregations
Generates community area risk summaries and monthly seasonal failure trends required by the dashboard.

```PowerShell

python community_area_analysis.py
python make_seasonal_csv.py
```
---

## Step 6: Launch the Streamlit Dashboard
Start the web dashboard application:

```PowerShell

python -m streamlit run app.py
```

---


# 🌐 Navigating the Application
Once launched, Streamlit will open automatically in your default browser at:
http://localhost:8501

Theme Toggling: Use the top-right Streamlit menu (⋮ -> Settings -> Theme) to switch seamlessly between Light and Dark modes.
Filter Panel: Use the left sidebar to filter facilities by priority tier (Red/Yellow/Green), risk probability thresholds, or Restaurant ID.
Exporting Data: Operational queues, facility reports, and community area summaries can be exported directly as CSV files using the embedded download buttons.

---


# 🔬 Tech Stack

- Language: Python 3.10+
- Data Processing: Pandas, NumPy
- Machine Learning: Scikit-Learn, XGBoost, Joblib
- Model Calibration: CalibratedClassifierCV (Sigmoid / Platt Scaling)
- Model Interpretability: SHAP (SHapley Additive exPlanations)
- Visualization & Mapping: Streamlit, Plotly Express/Graph Objects, PyDeck (3D Hexagon Layer) 



