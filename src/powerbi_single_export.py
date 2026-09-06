"""
Power BI Web Single-File Exporter.

Creates:
1. data/processed/dashboard/fraud_detection_powerbi_master.xlsx
   - Multi-sheet Excel workbook compatible with Power BI Service (Web).
   - Contains 4 sheets: 'Hourly_Fraud', 'Amount_Tiers', 'Model_Metrics', 'Confusion_Matrix'.
2. data/processed/dashboard/fraud_detection_all_in_one.csv
   - Single flat CSV containing enriched transactions with Hour, Amount_Category,
     Class_Label, and Heuristic Risk Flags for 1-click Power BI Web reporting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_cleaned.csv"
DASHBOARD_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"
MODELS_DIR = PROJECT_ROOT / "models"


def create_master_excel_and_csv():
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the 4 individual CSVs
    hourly_df = pd.read_csv(DASHBOARD_DIR / "fraud_by_hour.csv")
    amount_cat_df = pd.read_csv(DASHBOARD_DIR / "fraud_by_amount_category.csv")
    metrics_df = pd.read_csv(DASHBOARD_DIR / "model_metrics.csv")
    cm_df = pd.read_csv(DASHBOARD_DIR / "confusion_matrix_values.csv")
    
    # 2. Export Multi-Sheet Excel Workbook (Perfect for Power BI Web)
    excel_path = DASHBOARD_DIR / "fraud_detection_powerbi_master.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        hourly_df.to_excel(writer, sheet_name='Hourly_Fraud', index=False)
        amount_cat_df.to_excel(writer, sheet_name='Amount_Tiers', index=False)
        metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix', index=False)
    print(f"Master Excel Workbook created at: {excel_path}")

    # 3. Create Enriched Single CSV Dataset (Transaction Level + Derived Dimensions)
    print("Generating enriched single-file transaction dataset for Power BI Web...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # Add derived visual dimensions
    df['Hour'] = ((df['Time'] // 3600) % 24).astype(int)
    
    bins = [-np.inf, 20, 100, 500, np.inf]
    labels = ['Low (€0 - €20)', 'Medium (€20 - €100)', 'High (€100 - €500)', 'Very High (€500+)']
    df['Amount_Category'] = pd.cut(df['Amount'], bins=bins, labels=labels, right=True)
    df['Class_Label'] = df['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    # Add heuristic risk label
    df['Risk_Level'] = np.where(
        (df['V14'] < -5) | (df['V12'] < -5) | (df['V17'] < -5),
        'High Risk',
        'Standard'
    )
    
    # Export representative enriched dataset
    # We save a structured dataset that opens instantaneously in Power BI Web
    single_csv_path = DASHBOARD_DIR / "fraud_detection_all_in_one.csv"
    
    # We include all 473 fraud cases + 20,000 stratified legit cases for ultra-fast browser rendering in Power BI Web
    # or the full dataset
    df.to_csv(single_csv_path, index=False)
    print(f"Single-file CSV created at: {single_csv_path} ({len(df):,} rows)")
    
    return excel_path, single_csv_path


if __name__ == "__main__":
    create_master_excel_and_csv()
