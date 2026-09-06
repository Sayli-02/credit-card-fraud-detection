"""
Power BI Dashboard Data Preparation Module.

Exports pre-aggregated, cleaned CSV tables tailored for importing directly
into Power BI / Tableau dashboards:
1. data/processed/dashboard/fraud_by_hour.csv
2. data/processed/dashboard/fraud_by_amount_category.csv
3. data/processed/dashboard/model_metrics.csv
4. data/processed/dashboard/confusion_matrix_values.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_cleaned.csv"
MODEL_COMP_PATH = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
DASHBOARD_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"


def export_fraud_by_hour(df: pd.DataFrame, output_dir: Path = DASHBOARD_DIR) -> pd.DataFrame:
    """Generate hourly transaction and fraud metrics table for Power BI."""
    df_temp = df.copy()
    df_temp['Hour'] = ((df_temp['Time'] // 3600) % 24).astype(int)
    
    hourly = df_temp.groupby('Hour').agg(
        Total_Transactions=('Class', 'count'),
        Legit_Transactions=('Class', lambda x: int((x == 0).sum())),
        Fraud_Transactions=('Class', lambda x: int((x == 1).sum())),
        Total_Amount_EUR=('Amount', 'sum'),
        Fraud_Amount_EUR=('Amount', lambda x: x[df_temp.loc[x.index, 'Class'] == 1].sum()),
        Avg_Transaction_Amount_EUR=('Amount', 'mean'),
        Avg_Fraud_Amount_EUR=('Amount', lambda x: x[df_temp.loc[x.index, 'Class'] == 1].mean() if (df_temp.loc[x.index, 'Class'] == 1).sum() > 0 else 0.0)
    ).reset_index()
    
    hourly['Fraud_Rate_Pct'] = round((hourly['Fraud_Transactions'] / hourly['Total_Transactions']) * 100, 4)
    hourly['Total_Amount_EUR'] = round(hourly['Total_Amount_EUR'], 2)
    hourly['Fraud_Amount_EUR'] = round(hourly['Fraud_Amount_EUR'], 2)
    hourly['Avg_Transaction_Amount_EUR'] = round(hourly['Avg_Transaction_Amount_EUR'], 2)
    hourly['Avg_Fraud_Amount_EUR'] = round(hourly['Avg_Fraud_Amount_EUR'], 2)
    
    out_file = output_dir / "fraud_by_hour.csv"
    hourly.to_csv(out_file, index=False)
    print(f"Exported: {out_file} ({len(hourly)} rows)")
    return hourly


def export_fraud_by_amount_category(df: pd.DataFrame, output_dir: Path = DASHBOARD_DIR) -> pd.DataFrame:
    """Generate amount-tiered transaction and fraud metrics table for Power BI."""
    df_temp = df.copy()
    bins = [-np.inf, 20, 100, 500, np.inf]
    labels = ['Low (€0 - €20)', 'Medium (€20 - €100)', 'High (€100 - €500)', 'Very High (€500+)']
    df_temp['Amount_Category'] = pd.cut(df_temp['Amount'], bins=bins, labels=labels, right=True)
    
    tier_meta = {
        'Low (€0 - €20)': (0.0, 20.0),
        'Medium (€20 - €100)': (20.01, 100.0),
        'High (€100 - €500)': (100.01, 500.0),
        'Very High (€500+)': (500.01, 25691.16)
    }
    
    cat_df = df_temp.groupby('Amount_Category', observed=False).agg(
        Total_Transactions=('Class', 'count'),
        Legit_Transactions=('Class', lambda x: int((x == 0).sum())),
        Fraud_Transactions=('Class', lambda x: int((x == 1).sum())),
        Total_Amount_EUR=('Amount', 'sum'),
        Fraud_Amount_EUR=('Amount', lambda x: x[df_temp.loc[x.index, 'Class'] == 1].sum()),
        Avg_Amount_EUR=('Amount', 'mean')
    ).reset_index()
    
    cat_df['Min_Amount_EUR'] = cat_df['Amount_Category'].map(lambda x: tier_meta[str(x)][0])
    cat_df['Max_Amount_EUR'] = cat_df['Amount_Category'].map(lambda x: tier_meta[str(x)][1])
    cat_df['Fraud_Rate_Pct'] = round((cat_df['Fraud_Transactions'] / cat_df['Total_Transactions']) * 100, 4)
    cat_df['Total_Amount_EUR'] = round(cat_df['Total_Amount_EUR'], 2)
    cat_df['Fraud_Amount_EUR'] = round(cat_df['Fraud_Amount_EUR'], 2)
    cat_df['Avg_Amount_EUR'] = round(cat_df['Avg_Amount_EUR'], 2)
    
    # Reorder columns for Power BI
    cols = ['Amount_Category', 'Min_Amount_EUR', 'Max_Amount_EUR', 'Total_Transactions', 
            'Legit_Transactions', 'Fraud_Transactions', 'Fraud_Rate_Pct', 
            'Total_Amount_EUR', 'Fraud_Amount_EUR', 'Avg_Amount_EUR']
    cat_df = cat_df[cols]
    
    out_file = output_dir / "fraud_by_amount_category.csv"
    cat_df.to_csv(out_file, index=False)
    print(f"Exported: {out_file} ({len(cat_df)} rows)")
    return cat_df


def export_model_metrics(output_dir: Path = DASHBOARD_DIR) -> pd.DataFrame:
    """Generate model comparison metrics table for Power BI."""
    if not MODEL_COMP_PATH.exists():
        raise FileNotFoundError(f"Model comparison file not found at {MODEL_COMP_PATH}. Run modeling.py first.")
        
    df_comp = pd.read_csv(MODEL_COMP_PATH)
    out_file = output_dir / "model_metrics.csv"
    
    # Clean and standardize column names
    metrics_df = df_comp[['Model', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']].copy()
    metrics_df.to_csv(out_file, index=False)
    print(f"Exported: {out_file} ({len(metrics_df)} rows)")
    return metrics_df


def export_confusion_matrix_values(output_dir: Path = DASHBOARD_DIR) -> pd.DataFrame:
    """Generate unpivoted confusion matrix values table for Power BI matrix/heatmaps."""
    if not MODEL_COMP_PATH.exists():
        raise FileNotFoundError(f"Model comparison file not found at {MODEL_COMP_PATH}.")
        
    df_comp = pd.read_csv(MODEL_COMP_PATH)
    cm_records = []
    
    for _, row in df_comp.iterrows():
        model = row['Model']
        tp = int(row['True Positives (TP)'])
        fn = int(row['False Negatives (FN)'])
        fp = int(row['False Positives (FP)'])
        tn = int(row['True Negatives (TN)'])
        
        cm_records.extend([
            {'Model': model, 'Actual_Class': 'Legitimate (0)', 'Predicted_Class': 'Legitimate (0)', 'Metric_Label': 'True Negative (TN)', 'Count': tn},
            {'Model': model, 'Actual_Class': 'Legitimate (0)', 'Predicted_Class': 'Fraudulent (1)', 'Metric_Label': 'False Positive (FP - False Alert)', 'Count': fp},
            {'Model': model, 'Actual_Class': 'Fraudulent (1)', 'Predicted_Class': 'Legitimate (0)', 'Metric_Label': 'False Negative (FN - Missed Fraud)', 'Count': fn},
            {'Model': model, 'Actual_Class': 'Fraudulent (1)', 'Predicted_Class': 'Fraudulent (1)', 'Metric_Label': 'True Positive (TP - Caught Fraud)', 'Count': tp}
        ])
        
    df_cm = pd.DataFrame(cm_records)
    out_file = output_dir / "confusion_matrix_values.csv"
    df_cm.to_csv(out_file, index=False)
    print(f"Exported: {out_file} ({len(df_cm)} rows)")
    return df_cm


def run_dashboard_prep():
    """Execute all dashboard aggregation and export routines."""
    print("=" * 80)
    print("                 PHASE 7: POWER BI DASHBOARD DATA PREPARATION                  ")
    print("=" * 80)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading cleaned dataset...")
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    print("\nExporting aggregated Power BI tables...")
    export_fraud_by_hour(df)
    export_fraud_by_amount_category(df)
    export_model_metrics()
    export_confusion_matrix_values()
    
    print("\nAll 4 Power BI dashboard CSVs successfully generated in data/processed/dashboard/!")
    print("=" * 80)


if __name__ == "__main__":
    run_dashboard_prep()
