"""
Data Ingestion, Inspection, and Cleaning Module.

This module handles:
1. Loading the raw credit card fraud dataset.
2. Comprehensive data validation (shapes, dtypes, null values, duplicates).
3. Class distribution and imbalance analysis.
4. Descriptive statistics for Amount and Time.
5. Cleaning (duplicate removal, type validation) and persisting to data/processed/.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "creditcard_cleaned.csv"


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw credit card transactions dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            f"Please place 'creditcard.csv' in the 'data/raw/' directory."
        )
    print(f"Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset successfully loaded. Shape: {df.shape}")
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous data quality and summary inspections.
    
    Returns:
        Dict containing data shape, missing values, duplicates, class distribution,
        and statistical summaries.
    """
    total_rows, total_cols = df.shape
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    
    # Duplicate inspection
    duplicate_count = int(df.duplicated().sum())
    fraud_duplicates = int(df[df.duplicated()]['Class'].sum()) if duplicate_count > 0 and 'Class' in df.columns else 0
    
    # Class distribution
    class_counts = df['Class'].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    legit_count = int(class_counts.get(0, 0))
    fraud_percentage = (fraud_count / total_rows) * 100 if total_rows > 0 else 0.0
    imbalance_ratio = (legit_count / fraud_count) if fraud_count > 0 else float('inf')
    
    # Amount statistics by class
    amount_stats = {
        "overall": df['Amount'].describe().to_dict(),
        "legit": df[df['Class'] == 0]['Amount'].describe().to_dict() if legit_count > 0 else {},
        "fraud": df[df['Class'] == 1]['Amount'].describe().to_dict() if fraud_count > 0 else {},
    }
    
    # Time statistics
    time_stats = {
        "min_time_sec": float(df['Time'].min()),
        "max_time_sec": float(df['Time'].max()),
        "duration_hours": float(df['Time'].max() / 3600),
    }

    summary = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "total_nulls": total_nulls,
        "null_counts_by_col": null_counts[null_counts > 0].to_dict(),
        "duplicate_count": duplicate_count,
        "fraud_duplicates": fraud_duplicates,
        "class_distribution": {
            "legit_count": legit_count,
            "fraud_count": fraud_count,
            "fraud_percentage": round(fraud_percentage, 4),
            "imbalance_ratio": round(imbalance_ratio, 2)
        },
        "amount_statistics": amount_stats,
        "time_statistics": time_stats
    }
    return summary


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset:
    - Verifies numeric types
    - Handles duplicates (logging impact)
    - Returns cleaned dataframe and transformation metadata.
    """
    initial_shape = df.shape
    df_cleaned = df.copy()
    
    # Ensure Class is integer
    if 'Class' in df_cleaned.columns:
        df_cleaned['Class'] = df_cleaned['Class'].astype(int)
        
    removed_duplicates = 0
    if drop_duplicates:
        dup_mask = df_cleaned.duplicated()
        removed_duplicates = int(dup_mask.sum())
        if removed_duplicates > 0:
            dup_fraud_count = int(df_cleaned[dup_mask]['Class'].sum())
            print(f"Removing {removed_duplicates} duplicate records (including {dup_fraud_count} fraud duplicates)...")
            df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
            
    final_shape = df_cleaned.shape
    cleaning_meta = {
        "initial_rows": initial_shape[0],
        "final_rows": final_shape[0],
        "columns": final_shape[1],
        "removed_duplicates": removed_duplicates,
        "final_legit": int((df_cleaned['Class'] == 0).sum()),
        "final_fraud": int((df_cleaned['Class'] == 1).sum()),
        "final_fraud_rate_pct": round(((df_cleaned['Class'] == 1).sum() / len(df_cleaned)) * 100, 4)
    }
    return df_cleaned, cleaning_meta


def save_processed_data(df: pd.DataFrame, output_path: Path = CLEANED_DATA_PATH) -> None:
    """Save cleaned dataframe to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path} ({df.shape[0]:,} rows, {df.shape[1]} cols)")


def print_inspection_report(summary: Dict[str, Any], cleaning_meta: Dict[str, Any] = None) -> None:
    """Format and print an executive summary report for Phase 2 review."""
    print("=" * 80)
    print("                 PHASE 2: DATA UNDERSTANDING & CLEANING REPORT                 ")
    print("=" * 80)
    print(f"Raw Dataset Shape       : {summary['total_rows']:,} rows × {summary['total_columns']} columns")
    print(f"Missing Values (Nulls)  : {summary['total_nulls']} missing cells across all columns")
    print(f"Duplicate Transactions  : {summary['duplicate_count']:,} rows (Fraud duplicates: {summary['fraud_duplicates']})")
    
    cd = summary['class_distribution']
    print("\n--- CLASS DISTRIBUTION (RAW) ---")
    print(f"Legitimate (Class = 0)  : {cd['legit_count']:,} ({100 - cd['fraud_percentage']:.3f}%)")
    print(f"Fraudulent (Class = 1)  : {cd['fraud_count']:,} ({cd['fraud_percentage']:.4f}%)")
    print(f"Imbalance Ratio         : 1 Fraud per {cd['imbalance_ratio']:,.1f} Legit transactions")
    
    ts = summary['time_statistics']
    print("\n--- TEMPORAL PROFILE ---")
    print(f"Elapsed Time Span       : {ts['min_time_sec']:.0f}s to {ts['max_time_sec']:.0f}s ({ts['duration_hours']:.1f} hours / ~{ts['duration_hours']/24:.1f} days)")
    
    amt = summary['amount_statistics']
    print("\n--- TRANSACTION AMOUNT PROFILE (EUR) ---")
    print(f"Overall Mean (Median)   : €{amt['overall']['mean']:.2f} (€{amt['overall']['50%']:.2f}) | Min: €{amt['overall']['min']:.2f}, Max: €{amt['overall']['max']:.2f}")
    if amt['legit']:
        print(f"Legit Mean (Median)     : €{amt['legit']['mean']:.2f} (€{amt['legit']['50%']:.2f}) | Std: €{amt['legit']['std']:.2f}, Max: €{amt['legit']['max']:.2f}")
    if amt['fraud']:
        print(f"Fraud Mean (Median)     : €{amt['fraud']['mean']:.2f} (€{amt['fraud']['50%']:.2f}) | Std: €{amt['fraud']['std']:.2f}, Max: €{amt['fraud']['max']:.2f}")
        
    if cleaning_meta:
        print("\n--- CLEANING & PREPARATION SUMMARY ---")
        print(f"Deduplicated Rows       : {cleaning_meta['removed_duplicates']:,} removed")
        print(f"Cleaned Dataset Shape   : {cleaning_meta['final_rows']:,} rows × {cleaning_meta['columns']} columns")
        print(f"Post-Cleaning Legit     : {cleaning_meta['final_legit']:,} transactions")
        print(f"Post-Cleaning Fraud     : {cleaning_meta['final_fraud']:,} transactions ({cleaning_meta['final_fraud_rate_pct']}%)")
    print("=" * 80)


def run_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Execute full data loading and cleaning pipeline."""
    df_raw = load_raw_data(RAW_DATA_PATH)
    summary = inspect_dataset(df_raw)
    df_cleaned, cleaning_meta = clean_dataset(df_raw, drop_duplicates=True)
    save_processed_data(df_cleaned, CLEANED_DATA_PATH)
    print_inspection_report(summary, cleaning_meta)
    return df_cleaned, summary, cleaning_meta


if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        sys.exit(1)
