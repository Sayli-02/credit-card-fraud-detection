"""
Sparkov Credit Card Fraud Detection - Exploratory Data Analysis (EDA) Module.

Inspects fraudTrain.csv and fraudTest.csv:
1. Shapes, data types, null counts.
2. Class balance analysis for is_fraud.
3. Feature distributions: transaction category, state, hour of day, and amount tiers.
4. Generates and saves high-resolution publication-ready visualizations and summary JSON to artifacts/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
IMAGES_DIR = PROJECT_ROOT / "images"

TRAIN_PATH = DATA_DIR / "fraudTrain.csv"
TEST_PATH = DATA_DIR / "fraudTest.csv"

# Configure consistent plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300


def run_eda():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      TASK 1: SPARKOV DATASET EXPLORATION & CLASS BALANCE ANALYSIS        ")
    print("=" * 80)
    
    # 1. Load Datasets
    print(f"Loading Training Data from: {TRAIN_PATH} ...")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Loading Test Data from: {TEST_PATH} ...")
    test_df = pd.read_csv(TEST_PATH)
    
    # 2. Print Shapes, Dtypes, and Null Counts
    print("\n--- DATASET SHAPES ---")
    print(f"Train Dataset Shape : {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")
    print(f"Test Dataset Shape  : {test_df.shape[0]:,} rows, {test_df.shape[1]} columns")
    
    print("\n--- NULL VALUE COUNTS ---")
    train_nulls = train_df.isnull().sum()
    test_nulls = test_df.isnull().sum()
    print(f"Train Total Null Values: {train_nulls.sum()}")
    print(f"Test Total Null Values : {test_nulls.sum()}")
    
    print("\n--- DATA TYPES (TRAIN) ---")
    print(train_df.dtypes)
    
    # 3. Class Balance Analysis
    train_legit = int((train_df['is_fraud'] == 0).sum())
    train_fraud = int((train_df['is_fraud'] == 1).sum())
    train_fraud_rate = (train_fraud / len(train_df)) * 100
    
    test_legit = int((test_df['is_fraud'] == 0).sum())
    test_fraud = int((test_df['is_fraud'] == 1).sum())
    test_fraud_rate = (test_fraud / len(test_df)) * 100
    
    print("\n--- CLASS BALANCE (is_fraud) ---")
    print(f"Train - Legit : {train_legit:,} ({100 - train_fraud_rate:.3f}%) | Fraud : {train_fraud:,} ({train_fraud_rate:.3f}%)")
    print(f"Test  - Legit : {test_legit:,} ({100 - test_fraud_rate:.3f}%) | Fraud : {test_fraud:,} ({test_fraud_rate:.3f}%)")
    print(f"Imbalance Ratio (Train) : ~{train_legit // train_fraud}:1 Legit-to-Fraud")
    
    # 4. Feature Parsing for EDA
    train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
    train_df['hour'] = train_df['trans_date_trans_time'].dt.hour
    
    # 5. Category Analysis
    cat_summary = train_df.groupby('category').agg(
        total_txns=('is_fraud', 'count'),
        fraud_txns=('is_fraud', 'sum'),
        total_amount=('amt', 'sum'),
        mean_amount=('amt', 'mean')
    ).reset_index()
    cat_summary['fraud_rate_pct'] = (cat_summary['fraud_txns'] / cat_summary['total_txns']) * 100
    cat_summary = cat_summary.sort_values(by='fraud_rate_pct', ascending=False)
    
    print("\n--- TOP CATEGORIES BY FRAUD RATE (%) ---")
    print(cat_summary[['category', 'total_txns', 'fraud_txns', 'fraud_rate_pct']].to_string(index=False))
    
    # 6. State Analysis
    state_summary = train_df.groupby('state').agg(
        total_txns=('is_fraud', 'count'),
        fraud_txns=('is_fraud', 'sum')
    ).reset_index()
    state_summary['fraud_rate_pct'] = (state_summary['fraud_txns'] / state_summary['total_txns']) * 100
    state_top = state_summary.sort_values(by='fraud_txns', ascending=False).head(15)
    
    # 7. Hourly Fraud Analysis
    hourly_summary = train_df.groupby('hour').agg(
        total_txns=('is_fraud', 'count'),
        fraud_txns=('is_fraud', 'sum')
    ).reset_index()
    hourly_summary['fraud_rate_pct'] = (hourly_summary['fraud_txns'] / hourly_summary['total_txns']) * 100
    
    # 8. Save Visualizations
    # (a) Category Fraud Rate & Volume
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=cat_summary, 
        x='category', 
        y='fraud_rate_pct', 
        palette='Blues_r', 
        ax=ax1, 
        edgecolor='black',
        linewidth=0.5
    )
    ax1.set_title("Fraud Rate (%) by Transaction Category (Sparkov Dataset)", fontsize=14, weight='bold', pad=15)
    ax1.set_xlabel("Spending Category", fontsize=12, labelpad=10)
    ax1.set_ylabel("Fraud Rate (%)", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    for p in ax1.patches:
        val = p.get_height()
        if val > 0:
            ax1.annotate(f"{val:.2f}%", (p.get_x() + p.get_width() / 2., val),
                         ha='center', va='bottom', fontsize=8, xytext=(0, 3),
                         textcoords='offset points', weight='bold')
    plt.tight_layout()
    cat_plot_path = ARTIFACTS_DIR / "sparkov_eda_category.png"
    plt.savefig(cat_plot_path)
    plt.close()
    print(f"Saved: {cat_plot_path}")
    
    # (b) Hourly Fraud Rate Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.lineplot(
        data=hourly_summary, 
        x='hour', 
        y='fraud_rate_pct', 
        marker='o', 
        color='#EF4444', 
        linewidth=2.5, 
        ax=ax1
    )
    ax1.set_title("Fraud Probability (%) by Hour of Day", fontsize=13, weight='bold')
    ax1.set_xlabel("Hour of Day (0-23)", fontsize=11)
    ax1.set_ylabel("Fraud Rate (%)", fontsize=11)
    ax1.set_xticks(range(0, 24, 2))
    ax1.axvspan(22, 23.9, color='#EF4444', alpha=0.15, label='High Risk Window (22h-03h)')
    ax1.axvspan(0, 3.5, color='#EF4444', alpha=0.15)
    ax1.legend()
    
    sns.barplot(
        data=hourly_summary,
        x='hour',
        y='fraud_txns',
        palette='Reds',
        ax=ax2,
        edgecolor='black',
        linewidth=0.3
    )
    ax2.set_title("Absolute Number of Fraudulent Transactions by Hour", fontsize=13, weight='bold')
    ax2.set_xlabel("Hour of Day (0-23)", fontsize=11)
    ax2.set_ylabel("Total Fraud Cases", fontsize=11)
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    hourly_plot_path = ARTIFACTS_DIR / "sparkov_eda_hourly.png"
    plt.savefig(hourly_plot_path)
    plt.close()
    print(f"Saved: {hourly_plot_path}")
    
    # (c) State Level Fraud Distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=state_top, 
        x='state', 
        y='fraud_txns', 
        palette='viridis', 
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_title("Top 15 States by Absolute Fraud Volume", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("US State", fontsize=12)
    ax.set_ylabel("Fraudulent Transactions Count", fontsize=12)
    for p in ax.patches:
        val = int(p.get_height())
        if val > 0:
            ax.annotate(f"{val:,}", (p.get_x() + p.get_width() / 2., val),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                         textcoords='offset points', weight='bold')
    plt.tight_layout()
    state_plot_path = ARTIFACTS_DIR / "sparkov_eda_state.png"
    plt.savefig(state_plot_path)
    plt.close()
    print(f"Saved: {state_plot_path}")
    
    # 9. Save Summary JSON
    summary_data = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_columns": list(train_df.columns),
        "train_legit_count": train_legit,
        "train_fraud_count": train_fraud,
        "train_fraud_rate_pct": round(train_fraud_rate, 4),
        "test_legit_count": test_legit,
        "test_fraud_count": test_fraud,
        "test_fraud_rate_pct": round(test_fraud_rate, 4),
        "top_categories_by_fraud_rate": cat_summary.head(5)[['category', 'fraud_rate_pct', 'fraud_txns']].to_dict(orient='records'),
        "hourly_fraud_summary": hourly_summary[['hour', 'fraud_txns', 'fraud_rate_pct']].to_dict(orient='records')
    }
    
    summary_json_path = ARTIFACTS_DIR / "sparkov_eda_summary.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Saved: {summary_json_path}")
    print("=" * 80)
    print("TASK 1 COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
