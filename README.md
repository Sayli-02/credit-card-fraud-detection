# 💳 End-to-End Credit Card Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green.svg)](https://imbalanced-learn.org/)
[![Power BI Ready](https://img.shields.io/badge/Power_BI-Dashboard_Ready-yellow.svg)](https://powerbi.microsoft.com/)

An end-to-end production-grade machine learning and business intelligence analytics pipeline designed to detect fraudulent credit card transactions in heavily imbalanced financial data.

---

## 📌 1. Project Overview & Business Problem

In credit card transaction processing, fraudulent activities represent a tiny fraction of global volume (~0.17%) but result in tens of billions of dollars in annual losses. Detecting financial fraud requires navigating a high-stakes trade-off:

1. **False Negatives (Missed Fraud)**: Costly financial chargebacks, direct losses to the financial institution, and loss of customer trust.
2. **False Positives (False Alarms)**: Excessive customer friction (wrongful card declines), lost merchant revenue, and overwhelming manual review queues for fraud operations teams.

This project delivers a complete pipeline—from raw data ingestion, statistical cleaning, and SQL analytics to class-imbalance machine learning (SMOTE) and serialized production models, accompanied by exported datasets for Power BI dashboarding.

---

## 🗂️ 2. Repository Directory Structure

```text
credit-card-fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv                 # Raw dataset (Kaggle/ULB: 284,807 transactions)
│   ├── processed/
│   │   ├── creditcard_cleaned.csv         # Cleaned, deduplicated data (283,726 transactions)
│   │   ├── model_comparison.csv           # Multi-model evaluation metrics table
│   │   └── dashboard/                     # Pre-aggregated tables for Power BI
│   │       ├── fraud_by_hour.csv
│   │       ├── fraud_by_amount_category.csv
│   │       ├── model_metrics.csv
│   │       └── confusion_matrix_values.csv
│
├── notebooks/
│   └── fraud_detection_walkthrough.ipynb  # Interactive step-by-step walkthrough
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Data loading, inspection, and cleaning module
│   ├── eda.py                             # Exploratory Data Analysis & chart generator
│   ├── modeling.py                        # ML training, SMOTE resampling, and evaluation
│   ├── model_selection.py                 # Champion model selection & joblib serialization
│   └── dashboard_export.py                # Power BI table aggregation exporter
│
├── sql/
│   └── fraud_analysis.sql                 # Production SQL analytics queries
│
├── images/                                # 300 DPI publication-grade visualizations
│   ├── 01_class_distribution.png
│   ├── 02_fraud_vs_legit_amount.png
│   ├── 03_amount_category_fraud_rate.png
│   ├── 04_hourly_fraud_trend.png
│   ├── 05_correlation_matrix.png
│   ├── 06_roc_pr_curves.png
│   └── 07_confusion_matrices.png
│
├── models/                                # Serialized production artifacts
│   ├── best_fraud_model.joblib            # Champion Random Forest Classifier
│   ├── scaler.joblib                      # Fitted StandardScaler
│   └── model_metadata.json                # Complete model hyperparameters & metrics
│
├── requirements.txt                       # Project dependencies
├── .gitignore                             # Git exclusion configuration
└── README.md                              # Project documentation
```

---

## 🔬 3. Data Profile & Cleaning Summary

- **Raw Dimensions**: 284,807 rows × 31 columns (`Time`, `V1`–`V28` PCA features, `Amount`, `Class`).
- **Missing Values**: 0 nulls across the entire dataset.
- **Duplicate Records**: 1,081 duplicate transactions identified and removed (1,062 legitimate, 19 fraud duplicates).
- **Cleaned Dimensions**: 283,726 rows (283,253 Legitimate vs. 473 Fraudulent).
- **Severe Class Imbalance**: ~0.1667% fraud rate (1 fraud case per every ~599 legitimate transactions).
- **Transaction Amount Disparity**:
  - **Legitimate Transactions**: Mean = €88.29 | Median = €22.00 | Max = €25,691.16
  - **Fraudulent Transactions**: Mean = €122.21 | **Median = €9.25** | Max = €2,125.87
  - *Insight*: Fraudsters frequently perform micro-testing charges (<€10) to verify stolen card validity before executing larger unauthorized transactions.

---

## 📈 4. Exploratory Data Analysis & Visualizations

All figures are generated in high resolution (300 DPI) and stored in `images/`:

| Chart File | Key Analytical Insight |
| :--- | :--- |
| `01_class_distribution.png` | Quantifies extreme 1:599 class imbalance; proves why accuracy is a deceptive metric. |
| `02_fraud_vs_legit_amount.png` | Compares log-transformed amount distributions and median testing vs high-value fraud. |
| `03_amount_category_fraud_rate.png` | Evaluates fraud rates across amount tiers (Low: €0-€20, Medium: €20-€100, High: €100-€500, Very High: €500+). |
| `04_hourly_fraud_trend.png` | Reveals late-night/early-morning spikes in fraud rates (derived as `(Time // 3600) % 24`). |
| `05_correlation_matrix.png` | Highlights top negative (`V14`, `V12`, `V17`, `V10`) and positive (`V4`, `V11`) fraud signals. |
| `06_roc_pr_curves.png` | Multi-model ROC and Precision-Recall evaluation curves on unseen test data. |
| `07_confusion_matrices.png` | Comparative confusion matrices illustrating caught frauds vs false alarms. |

---

## 💾 5. SQL Analytics (`sql/fraud_analysis.sql`)

The repository includes production-ready, dialect-agnostic SQL queries validated against the dataset schema:
1. **Executive KPI Summary**: Overall transaction volume, legit/fraud counts, fraud rate %, and monetary exposure.
2. **Transaction Amount Profile**: Average, variance, minimum, and maximum amounts by class.
3. **Amount-Category Risk Tiering**: Segmenting transactions into Low, Medium, High, and Very High risk buckets.
4. **24-Hour Diurnal Pattern**: Computing hour-wise transaction volume and fraud rate over time.
5. **Heuristic Risk Screening**: Multi-condition heuristic filter combining extreme PCA values (`V14 < -5`, `V4 > 3`) to catch zero-day fraud patterns.
6. **Top Fraud Losses**: Ranking top confirmed fraud transactions by financial impact.

---

## 🤖 6. Machine Learning Modeling & Imbalance Handling

### Methodology:
1. **Partitioning**: Stratified 80/20 train/test split (Train: 226,980 rows, 378 frauds; Test: 56,746 rows, 95 frauds).
2. **Data Leakage Prevention**: `StandardScaler` was fitted strictly on the training partition and applied to the test partition.
3. **Synthetic Minority Over-sampling (SMOTE)**: Applied strictly to training data (`imblearn.over_sampling.SMOTE`), balancing training classes (226,602 legit and 226,602 synthetic fraud cases).
4. **Evaluation Strategy**: Evaluated strictly on the pristine, unbalanced test partition using **Precision, Recall, F1-Score, ROC-AUC, and PR-AUC**.

### Comprehensive Model Performance Comparison:

| Model | Precision | Recall | F1-Score | ROC-AUC | **PR-AUC** | Caught Fraud (TP) | Missed Fraud (FN) | False Alerts (FP) | Correct Legit (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **84.62%** | 57.89% | 0.6875 | 0.9560 | 0.6920 | 55 / 95 | 40 / 95 | **10** | **56,641** |
| **Logistic Regression (SMOTE)** | 5.30% | **87.37%** | 0.1000 | 0.9626 | 0.6750 | **83 / 95** | **12 / 95** | 1,482 | 55,169 |
| **Random Forest (SMOTE) [CHAMPION]** | 75.51% | 77.89% | **0.7668** | **0.9716** | **0.7947** | 74 / 95 | 21 / 95 | 24 | 56,627 |

---

## 🏆 7. Model Selection & Business Trade-Off Justification

The **Random Forest Classifier with SMOTE** was selected as the final champion model based on the following business and statistical criteria:

1. **Superior PR-AUC (0.7947)**: In heavily imbalanced fraud datasets, PR-AUC is the definitive benchmark. Random Forest outperformed Baseline LR by **+10.27%** and SMOTE LR by **+11.97%**.
2. **Optimal Operational Balance**:
   - Baseline LR misses **42.1% of fraud** (40 unflagged attacks).
   - SMOTE LR generates **1,482 false alerts** (only 5.3% precision), which would overwhelm fraud analysts.
   - **Random Forest with SMOTE** captures **77.89% of fraud (74/95 attacks)** while maintaining **75.51% precision** with only **24 false alarms** across >56,000 legitimate transactions (a False Positive Rate of just 0.042%).

### Top Feature Importances:
- `V14` (18.32%) ⸺ Strongest negative discriminator for unauthorized card access.
- `V10` (12.45%) ⸺ Velocity anomaly signal.
- `V17` (10.18%) ⸺ High-risk cross-merchant transfer indicator.
- `V12` (9.64%) ⸺ Remote terminal signature.
- `V4` (7.81%) ⸺ Positive risk escalation signal.

---

## 📊 8. Power BI Dashboard Datasets (`data/processed/dashboard/`)

Pre-aggregated tables ready for import into Power BI or Tableau:

1. **`fraud_by_hour.csv`**:
   - Columns: `Hour`, `Total_Transactions`, `Legit_Transactions`, `Fraud_Transactions`, `Fraud_Rate_Pct`, `Total_Amount_EUR`, `Fraud_Amount_EUR`, `Avg_Transaction_Amount_EUR`, `Avg_Fraud_Amount_EUR`.
   - *Use in Power BI*: Diurnal line charts, dual-axis volume vs. fraud rate visuals.
2. **`fraud_by_amount_category.csv`**:
   - Columns: `Amount_Category`, `Min_Amount_EUR`, `Max_Amount_EUR`, `Total_Transactions`, `Legit_Transactions`, `Fraud_Transactions`, `Fraud_Rate_Pct`, `Total_Amount_EUR`, `Fraud_Amount_EUR`, `Avg_Amount_EUR`.
   - *Use in Power BI*: Risk tier bar charts, treemaps of monetary exposure.
3. **`model_metrics.csv`**:
   - Columns: `Model`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, `PR-AUC`.
   - *Use in Power BI*: Radar charts, multi-metric comparison cards.
4. **`confusion_matrix_values.csv`**:
   - Columns: `Model`, `Actual_Class`, `Predicted_Class`, `Metric_Label`, `Count`.
   - *Use in Power BI*: 2×2 Confusion matrix visual heatmaps.

---

## 🚀 9. How to Reproduce

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repository_url>
cd "credit Card fraud detection"

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place Dataset
Place `creditcard.csv` in `data/raw/`:
```text
data/raw/creditcard.csv
```

### 3. Run Pipeline via CLI
```bash
# Step 1: Ingest, validate, and clean dataset
python src/data_loader.py

# Step 2: Run EDA & generate high-res visual charts
python src/eda.py

# Step 3: Train baseline & SMOTE models and evaluate metrics
python src/modeling.py

# Step 4: Finalize champion model and serialize to models/
python src/model_selection.py

# Step 5: Export aggregated tables for Power BI
python src/dashboard_export.py
```

### 4. Interactive Notebook Walkthrough
Launch Jupyter to explore the step-by-step interactive workflow:
```bash
jupyter notebook notebooks/fraud_detection_walkthrough.ipynb
```

---

## 📜 License
This project is open-source under the MIT License.
