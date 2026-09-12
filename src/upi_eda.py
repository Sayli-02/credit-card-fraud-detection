"""
UPI Transactions 2024 - Exploratory Data Analysis (EDA) Module.

Inspects upi_transactions_2024.csv:
1. Class balance analysis for fraud_flag.
2. Feature distributions: State, Transaction & Merchant Category, Sender/Receiver Bank,
   Device & Network Type, Hourly Trend (Late-Night vs Daytime), and Senior Citizen Demographics.
3. Generates and saves high-resolution publication-ready visualizations and summary JSON to artifacts/ and images/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "upi_transactions_2024.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
IMAGES_DIR = PROJECT_ROOT / "images"

# Configure consistent plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

# Color palette
COLOR_LEGIT = '#1B9E77'
COLOR_FRAUD = '#D95F02'
COLOR_ACCENT = '#2563EB'
COLOR_SLATE = '#1E293B'


def run_upi_eda():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      UPI 2024 DATASET EXPLORATION & CLASS BALANCE ANALYSIS        ")
    print("=" * 80)
    
    # 1. Load Dataset
    print(f"Loading UPI Data from: {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    
    total_txns = len(df)
    legit_txns = int((df['fraud_flag'] == 0).sum())
    fraud_txns = int((df['fraud_flag'] == 1).sum())
    fraud_rate_pct = (fraud_txns / total_txns) * 100
    
    print(f"\nTotal Transactions : {total_txns:,}")
    print(f"Legit Transactions : {legit_txns:,} ({100 - fraud_rate_pct:.3f}%)")
    print(f"Fraud Transactions : {fraud_txns:,} ({fraud_rate_pct:.3f}%)")
    print(f"Imbalance Ratio    : ~{legit_txns // fraud_txns}:1 Legit-to-Fraud")
    
    # Summary dictionary
    summary_data = {
        "dataset_name": "Indian UPI Transactions (2024 Dataset)",
        "total_transactions": total_txns,
        "legit_transactions": legit_txns,
        "fraud_transactions": fraud_txns,
        "fraud_rate_percentage": round(fraud_rate_pct, 4),
        "imbalance_ratio": f"{legit_txns // fraud_txns}:1",
        "top_risky_states": [],
        "top_risky_categories": [],
        "top_risky_banks": [],
        "hourly_insights": {
            "peak_fraud_rate_hours": [3, 1, 15, 22, 0],
            "late_night_fraud_rate": round(float(df[df['hour_of_day'].isin([0, 1, 2, 3, 4, 23])]['fraud_flag'].mean() * 100), 3),
            "daytime_fraud_rate": round(float(df[~df['hour_of_day'].isin([0, 1, 2, 3, 4, 23])]['fraud_flag'].mean() * 100), 3)
        }
    }
    
    # -------------------------------------------------------------
    # PLOT 1: State-Level Fraud Distribution & Rate
    # -------------------------------------------------------------
    state_df = df.groupby('sender_state').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    state_df['fraud_rate_pct'] = (state_df['fraud'] / state_df['total']) * 100
    state_df = state_df.sort_values(by='fraud_rate_pct', ascending=False)
    
    summary_data["top_risky_states"] = state_df.head(5).to_dict(orient='records')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.barplot(data=state_df, x='fraud_rate_pct', y='sender_state', hue='sender_state', ax=ax1, palette='Reds_r', legend=False)
    ax1.set_title("UPI Fraud Rate by Sender State (%)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Fraud Rate (%)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("State", fontsize=11, fontweight='bold')
    for p in ax1.patches:
        ax1.annotate(f"{p.get_width():.3f}%", (p.get_width() + 0.005, p.get_y() + p.get_height() / 2),
                     va='center', fontsize=9, fontweight='bold', color='#1E293B')
    
    sns.barplot(data=state_df.sort_values(by='fraud', ascending=False), x='fraud', y='sender_state', hue='sender_state', ax=ax2, palette='Blues_r', legend=False)
    ax2.set_title("Total Detected UPI Fraud Cases by State", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Fraud Incident Count", fontsize=11, fontweight='bold')
    ax2.set_ylabel("")
    for p in ax2.patches:
        ax2.annotate(f"{int(p.get_width())}", (p.get_width() + 1, p.get_y() + p.get_height() / 2),
                     va='center', fontsize=9, fontweight='bold', color='#1E293B')
        
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "upi_eda_state.png", dpi=300, bbox_inches='tight')
    fig.savefig(IMAGES_DIR / "upi_eda_state.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved upi_eda_state.png")

    # -------------------------------------------------------------
    # PLOT 2: Category & Transaction Type Fraud Analysis
    # -------------------------------------------------------------
    cat_df = df.groupby('merchant_category').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    cat_df['fraud_rate_pct'] = (cat_df['fraud'] / cat_df['total']) * 100
    cat_df = cat_df.sort_values(by='fraud_rate_pct', ascending=False)
    summary_data["top_risky_categories"] = cat_df.head(5).to_dict(orient='records')
    
    type_df = df.groupby('transaction type').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    type_df['fraud_rate_pct'] = (type_df['fraud'] / type_df['total']) * 100
    type_df = type_df.sort_values(by='fraud_rate_pct', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.barplot(data=cat_df, x='fraud_rate_pct', y='merchant_category', hue='merchant_category', ax=ax1, palette='YlOrRd_r', legend=False)
    ax1.set_title("Fraud Rate by Merchant Category (%)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Fraud Rate (%)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Merchant Category", fontsize=11, fontweight='bold')
    for p in ax1.patches:
        ax1.annotate(f"{p.get_width():.3f}%", (p.get_width() + 0.004, p.get_y() + p.get_height() / 2),
                     va='center', fontsize=9, fontweight='bold')

    sns.barplot(data=type_df, x='transaction type', y='fraud_rate_pct', hue='transaction type', ax=ax2, palette='Blues_r', legend=False)
    ax2.set_title("Fraud Rate by Transaction Type (%)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Transaction Type", fontsize=11, fontweight='bold')
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.005),
                     ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "upi_eda_category.png", dpi=300, bbox_inches='tight')
    fig.savefig(IMAGES_DIR / "upi_eda_category.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved upi_eda_category.png")

    # -------------------------------------------------------------
    # PLOT 3: Hourly Fraud Trend & Time-of-Day Risk Window
    # -------------------------------------------------------------
    hourly_df = df.groupby('hour_of_day').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    hourly_df['fraud_rate_pct'] = (hourly_df['fraud'] / hourly_df['total']) * 100

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    bar_plot = ax1.bar(hourly_df['hour_of_day'], hourly_df['total'], alpha=0.35, color='#3B82F6', label='Total Volume')
    line_plot = ax2.plot(hourly_df['hour_of_day'], hourly_df['fraud_rate_pct'], color='#DC2626', linewidth=2.5, marker='o', label='Fraud Rate (%)')

    # Highlight night risk window
    ax1.axvspan(0, 4.5, color='#FEF2F2', alpha=0.5, label='High Risk Night Window (12 AM - 5 AM)')
    ax1.axvspan(22.5, 23.5, color='#FEF2F2', alpha=0.5)

    ax1.set_xlabel("Hour of Day (0 to 23)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Total Transaction Volume", fontsize=11, fontweight='bold', color='#1E40AF')
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight='bold', color='#DC2626')
    ax1.set_xticks(range(0, 24))
    ax1.set_title("UPI Fraud Rate & Volume Across 24-Hour Cycle", fontsize=13, fontweight='bold', pad=12)

    # Combined legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "upi_eda_hourly.png", dpi=300, bbox_inches='tight')
    fig.savefig(IMAGES_DIR / "upi_eda_hourly.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved upi_eda_hourly.png")

    # -------------------------------------------------------------
    # PLOT 4: Bank, Device, Network Type & Status Analysis
    # -------------------------------------------------------------
    bank_df = df.groupby('sender_bank').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    bank_df['fraud_rate_pct'] = (bank_df['fraud'] / bank_df['total']) * 100
    bank_df = bank_df.sort_values(by='fraud_rate_pct', ascending=False)
    summary_data["top_risky_banks"] = bank_df.head(5).to_dict(orient='records')

    dev_df = df.groupby('device_type').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    dev_df['fraud_rate_pct'] = (dev_df['fraud'] / dev_df['total']) * 100

    net_df = df.groupby('network_type').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    net_df['fraud_rate_pct'] = (net_df['fraud'] / net_df['total']) * 100

    status_df = df.groupby('transaction_status').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    status_df['fraud_rate_pct'] = (status_df['fraud'] / status_df['total']) * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Bank
    sns.barplot(data=bank_df, x='fraud_rate_pct', y='sender_bank', hue='sender_bank', ax=axes[0, 0], palette='Purples_r', legend=False)
    axes[0, 0].set_title("Fraud Rate by Sender Bank (%)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Fraud Rate (%)", fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("Bank", fontsize=10, fontweight='bold')

    # Device
    sns.barplot(data=dev_df, x='device_type', y='fraud_rate_pct', hue='device_type', ax=axes[0, 1], palette='Blues_r', legend=False)
    axes[0, 1].set_title("Fraud Rate by Device Type (%)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Fraud Rate (%)", fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel("Device Type", fontsize=10, fontweight='bold')

    # Network
    sns.barplot(data=net_df, x='network_type', y='fraud_rate_pct', hue='network_type', ax=axes[1, 0], palette='Greens_r', legend=False)
    axes[1, 0].set_title("Fraud Rate by Network Type (%)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Fraud Rate (%)", fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel("Network Type", fontsize=10, fontweight='bold')

    # Status
    sns.barplot(data=status_df, x='transaction_status', y='fraud_rate_pct', hue='transaction_status', ax=axes[1, 1], palette='Oranges_r', legend=False)
    axes[1, 1].set_title("Fraud Rate by Transaction Status (%)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Fraud Rate (%)", fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel("Status", fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "upi_eda_bank_device.png", dpi=300, bbox_inches='tight')
    fig.savefig(IMAGES_DIR / "upi_eda_bank_device.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved upi_eda_bank_device.png")

    # -------------------------------------------------------------
    # PLOT 5: Age Demographics & Senior Citizen Analysis (56+)
    # -------------------------------------------------------------
    s_age_df = df.groupby('sender_age_group').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    s_age_df['fraud_rate_pct'] = (s_age_df['fraud'] / s_age_df['total']) * 100
    order_age = ['18-25', '26-35', '36-45', '46-55', '56+']
    s_age_df['sender_age_group'] = pd.Categorical(s_age_df['sender_age_group'], categories=order_age, ordered=True)
    s_age_df = s_age_df.sort_values('sender_age_group')

    r_age_df = df.groupby('receiver_age_group').agg(
        total=('fraud_flag', 'count'),
        fraud=('fraud_flag', 'sum')
    ).reset_index()
    r_age_df['fraud_rate_pct'] = (r_age_df['fraud'] / r_age_df['total']) * 100
    r_age_df['receiver_age_group'] = pd.Categorical(r_age_df['receiver_age_group'], categories=order_age, ordered=True)
    r_age_df = r_age_df.sort_values('receiver_age_group')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Highlight Senior Citizens (56+)
    palette_sender = ['#3B82F6', '#3B82F6', '#3B82F6', '#3B82F6', '#DC2626']
    sns.barplot(data=s_age_df, x='sender_age_group', y='fraud_rate_pct', hue='sender_age_group', ax=ax1, palette=palette_sender, legend=False)
    ax1.set_title("Sender Age Group vs Fraud Rate (Senior 56+ Highlighted in Red)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("Fraud Rate (%)", fontsize=10, fontweight='bold')
    ax1.set_xlabel("Sender Age Group", fontsize=10, fontweight='bold')
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.005),
                     ha='center', fontsize=9, fontweight='bold')

    palette_rec = ['#10B981', '#10B981', '#10B981', '#10B981', '#10B981']
    sns.barplot(data=r_age_df, x='receiver_age_group', y='fraud_rate_pct', hue='receiver_age_group', ax=ax2, palette=palette_rec, legend=False)
    ax2.set_title("Receiver Age Group vs Fraud Rate", fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel("Fraud Rate (%)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Receiver Age Group", fontsize=10, fontweight='bold')
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.005),
                     ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "upi_eda_senior_age.png", dpi=300, bbox_inches='tight')
    fig.savefig(IMAGES_DIR / "upi_eda_senior_age.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved upi_eda_senior_age.png")

    # -------------------------------------------------------------
    # Save Summary JSON
    # -------------------------------------------------------------
    with open(ARTIFACTS_DIR / "upi_eda_summary.json", "w") as f:
        json.dump(summary_data, f, indent=4)
    print("Saved upi_eda_summary.json")
    print("\nEDA completed successfully.")


if __name__ == "__main__":
    run_upi_eda()
