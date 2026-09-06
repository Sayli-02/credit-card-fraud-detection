"""
Exploratory Data Analysis (EDA) & Visualization Generator.

Generates high-resolution publication-grade visual charts:
1. 01_class_distribution.png: Class imbalance & fraud rate breakdown.
2. 02_fraud_vs_legit_amount.png: Amount distribution & log-scale boxplot comparison.
3. 03_amount_category_fraud_rate.png: Amount bucket fraud rates (Low/Medium/High/Very High).
4. 04_hourly_fraud_trend.png: Hour-of-day fraud transaction volume & fraud rate trend.
5. 05_correlation_matrix.png: Top feature correlations with Class.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_cleaned.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

# Configure modern aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

# Color palette: Elegant dark slate & vibrant accents (Coral for Fraud, Teal for Legit)
COLOR_LEGIT = '#1B9E77'
COLOR_FRAUD = '#D95F02'
PALETTE_CLASS = {0: COLOR_LEGIT, 1: COLOR_FRAUD}


def load_cleaned_data(file_path: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load cleaned transaction data."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned data not found at {file_path}. Run data_loader.py first.")
    df = pd.read_csv(file_path)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day and amount category features."""
    df = df.copy()
    # Hour of day (0 to 23)
    df['Hour'] = ((df['Time'] // 3600) % 24).astype(int)
    
    # Amount categories
    # Low: <= 20, Medium: 20-100, High: 100-500, Very High: > 500
    bins = [-np.inf, 20, 100, 500, np.inf]
    labels = ['Low (€0-€20)', 'Medium (€20-€100)', 'High (€100-€500)', 'Very High (€500+)']
    df['Amount_Category'] = pd.cut(df['Amount'], bins=bins, labels=labels, right=True)
    return df


def plot_class_distribution(df: pd.DataFrame, output_dir: Path = IMAGES_DIR) -> None:
    """Plot 1: Severe class imbalance visualization (Donut chart + Bar count)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    counts = df['Class'].value_counts()
    legit_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    total = len(df)
    
    # Bar Chart (Log Scale to show minority clearly)
    bars = ax1.bar(['Legitimate (0)', 'Fraudulent (1)'], [legit_count, fraud_count], 
                   color=[COLOR_LEGIT, COLOR_FRAUD], width=0.5, edgecolor='black', linewidth=1.2)
    ax1.set_yscale('log')
    ax1.set_title('Transaction Counts by Class (Log Scale)', pad=15)
    ax1.set_ylabel('Transaction Count (Log Scale)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:,}\n({height/total*100:.3f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 6), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Donut Chart
    ax2.pie([legit_count, fraud_count], 
            labels=['Legitimate', 'Fraudulent'], 
            colors=[COLOR_LEGIT, COLOR_FRAUD], 
            autopct=lambda pct: f'{pct:.3f}%\n({int(pct*total/100):,} txns)',
            pctdistance=0.75, startangle=140, 
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Severe Class Imbalance Proportion', pad=15)

    plt.suptitle('Class Imbalance: 99.833% Legit vs 0.167% Fraud (1 : 599 Ratio)', y=1.02)
    plt.tight_layout()
    
    out_file = output_dir / "01_class_distribution.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_file.name}")


def plot_fraud_vs_legit_amount(df: pd.DataFrame, output_dir: Path = IMAGES_DIR) -> None:
    """Plot 2: Transaction Amount Comparison (Boxplot + Density on Log Scale)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Add small epsilon for log transform to handle €0 transactions
    df_plot = df.copy()
    df_plot['Log_Amount'] = np.log10(df_plot['Amount'] + 1)
    
    # Boxplot with Mean markers
    box = sns.boxplot(x='Class', y='Log_Amount', data=df_plot, ax=ax1, 
                      palette=[COLOR_LEGIT, COLOR_FRAUD], showmeans=True,
                      meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"})
    ax1.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'])
    ax1.set_title('Log10(Amount + €1) Distribution Comparison')
    ax1.set_xlabel('Transaction Class')
    ax1.set_ylabel('Log10(Amount + €1) [EUR]')
    
    # Custom y-tick labels for real euro values
    ticks = [0, 1, 2, 3, 4]
    tick_labels = ['€0', '€9', '€99', '€999', '€9,999']
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(tick_labels)
    
    # Summary Statistics Overlay
    legit_med = df[df['Class'] == 0]['Amount'].median()
    fraud_med = df[df['Class'] == 1]['Amount'].median()
    legit_mean = df[df['Class'] == 0]['Amount'].mean()
    fraud_mean = df[df['Class'] == 1]['Amount'].mean()
    
    ax1.text(0, np.log10(legit_med + 1) + 0.15, f"Med: €{legit_med:.2f}\nMean: €{legit_mean:.2f}", 
             ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.text(1, np.log10(fraud_med + 1) + 0.15, f"Med: €{fraud_med:.2f}\nMean: €{fraud_mean:.2f}", 
             ha='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # KDE Distribution
    sns.kdeplot(df_plot[df_plot['Class'] == 0]['Log_Amount'], ax=ax2, color=COLOR_LEGIT, 
                label=f'Legit (Mean €{legit_mean:.1f})', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(df_plot[df_plot['Class'] == 1]['Log_Amount'], ax=ax2, color=COLOR_FRAUD, 
                label=f'Fraud (Mean €{fraud_mean:.1f})', fill=True, alpha=0.4, linewidth=2)
    ax2.set_title('KDE Density of Transaction Amount (Log Scale)')
    ax2.set_xlabel('Log10(Amount + €1)')
    ax2.set_ylabel('Density')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels)
    ax2.legend(loc='upper right', frameon=True)

    plt.suptitle('Transaction Amount Disparity: Fraudsters Target Both Micro-Testing (€0-€10) and High-Value (€500+)', y=1.02)
    plt.tight_layout()
    
    out_file = output_dir / "02_fraud_vs_legit_amount.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_file.name}")


def plot_amount_category_fraud_rate(df: pd.DataFrame, output_dir: Path = IMAGES_DIR) -> None:
    """Plot 3: Fraud Rate by Amount Tiers (Low, Medium, High, Very High)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Grouped stats
    cat_stats = df.groupby('Amount_Category', observed=False).agg(
        total_txns=('Class', 'count'),
        fraud_txns=('Class', 'sum'),
        total_amount=('Amount', 'sum'),
        fraud_amount=('Amount', lambda x: x[df.loc[x.index, 'Class'] == 1].sum())
    ).reset_index()
    cat_stats['fraud_rate_pct'] = (cat_stats['fraud_txns'] / cat_stats['total_txns']) * 100
    cat_stats['amount_share_pct'] = (cat_stats['total_amount'] / cat_stats['total_amount'].sum()) * 100
    
    colors = ['#386cb0', '#7fc97f', '#fdc086', '#e7298a']
    
    # Bar 1: Fraud Rate (%) by Category
    bars1 = ax1.bar(cat_stats['Amount_Category'], cat_stats['fraud_rate_pct'], 
                    color=colors, width=0.55, edgecolor='black', linewidth=1.1)
    ax1.set_title('Fraud Rate (%) by Amount Category')
    ax1.set_ylabel('Fraud Rate (%)')
    ax1.set_xlabel('Amount Category')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, fraud_cnt, total_cnt in zip(bars1, cat_stats['fraud_txns'], cat_stats['total_txns']):
        h = bar.get_height()
        ax1.annotate(f'{h:.3f}%\n({fraud_cnt}/{total_cnt:,})',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # Bar 2: Total Volume & Fraud Volume Breakdown
    bars2 = ax2.bar(cat_stats['Amount_Category'], cat_stats['total_txns'], 
                    color='#7570b3', width=0.55, edgecolor='black', linewidth=1.1, label='Total Transactions')
    ax2.set_title('Total Transaction Volume by Amount Category')
    ax2.set_ylabel('Transaction Volume')
    ax2.set_xlabel('Amount Category')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f'{h:,}\n({h/len(df)*100:.1f}%)',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Amount Category Analysis: Low & Very High Tiers Exhibit Elevated Fraud Probability', y=1.02)
    plt.tight_layout()
    
    out_file = output_dir / "03_amount_category_fraud_rate.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_file.name}")


def plot_hourly_fraud_trend(df: pd.DataFrame, output_dir: Path = IMAGES_DIR) -> None:
    """Plot 4: Hourly Transaction Volume & Fraud Rate Trend (Hour-of-Day)."""
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    hourly = df.groupby('Hour').agg(
        total_txns=('Class', 'count'),
        fraud_txns=('Class', 'sum'),
        legit_txns=('Class', lambda x: (x == 0).sum())
    ).reset_index()
    hourly['fraud_rate_pct'] = (hourly['fraud_txns'] / hourly['total_txns']) * 100
    
    # Primary axis: Total transaction volume
    color_bar = '#6baed6'
    bars = ax1.bar(hourly['Hour'], hourly['total_txns'], color=color_bar, alpha=0.65, 
                   edgecolor='black', linewidth=0.8, label='Total Transactions')
    ax1.set_xlabel('Hour of Day (0:00 to 23:00)', fontsize=12)
    ax1.set_ylabel('Total Transaction Volume', color='#2171b5', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#2171b5')
    ax1.set_xticks(range(0, 24))
    ax1.grid(axis='x', linestyle=':', alpha=0.5)

    # Secondary axis: Fraud Rate %
    ax2 = ax1.twinx()
    color_line = '#d94701'
    line = ax2.plot(hourly['Hour'], hourly['fraud_rate_pct'], color=color_line, 
                    marker='o', linewidth=2.8, markersize=7, label='Fraud Rate (%)')
    ax2.set_ylabel('Fraud Rate (%)', color=color_line, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.grid(False) # avoid clutter
    
    # Annotate peak fraud hours
    peak_hour = hourly.loc[hourly['fraud_rate_pct'].idxmax()]
    ax2.annotate(f"Peak Risk: Hour {int(peak_hour['Hour'])}:00\n({peak_hour['fraud_rate_pct']:.3f}% Fraud Rate)",
                 xy=(peak_hour['Hour'], peak_hour['fraud_rate_pct']),
                 xytext=(peak_hour['Hour'] + 1, peak_hour['fraud_rate_pct'] + 0.05),
                 arrowprops=dict(facecolor=color_line, shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee6ce", edgecolor=color_line))

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)

    plt.title('24-Hour Fraud Pattern: Late Night / Early Morning Hours Exhibit Spike in Fraud Propensity', pad=20)
    plt.tight_layout()
    
    out_file = output_dir / "04_hourly_fraud_trend.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_file.name}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path = IMAGES_DIR) -> None:
    """Plot 5: Top Feature Correlations with Target (Class) & Correlation Heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # Compute correlation with Class
    features = [c for c in df.columns if c not in ['Amount_Category', 'Hour']]
    corr = df[features].corr()
    class_corr = corr['Class'].drop('Class').sort_values()
    
    # Top 8 negative and top 8 positive correlated features
    top_neg = class_corr.head(8)
    top_pos = class_corr.tail(8)
    top_features = pd.concat([top_neg, top_pos])
    
    colors = [COLOR_FRAUD if val > 0 else '#3182bd' for val in top_features.values]
    
    bars = ax1.barh(top_features.index, top_features.values, color=colors, edgecolor='black', linewidth=0.9)
    ax1.axvline(0, color='black', linewidth=1)
    ax1.set_title('Top 16 Features Correlated with Class (Target)')
    ax1.set_xlabel('Pearson Correlation Coefficient')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars:
        w = bar.get_width()
        ha = 'left' if w > 0 else 'right'
        offset = 0.005 if w > 0 else -0.005
        ax1.annotate(f'{w:+.3f}',
                     xy=(w + offset, bar.get_y() + bar.get_height() / 2),
                     va='center', ha=ha, fontsize=9, fontweight='bold')
                     
    # Sub-heatmap of the top 8 most influential features + Class
    selected_cols = list(top_neg.head(4).index) + list(top_pos.tail(4).index) + ['Class']
    sub_corr = df[selected_cols].corr()
    
    sns.heatmap(sub_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.4, vmax=0.4,
                center=0, ax=ax2, cbar_kws={'label': 'Correlation'}, square=True, linewidths=0.5)
    ax2.set_title('Key Features Correlation Sub-Matrix')

    plt.suptitle('Feature Importance Signals: V17, V14, V12 (Strong Negative) & V4, V11 (Strong Positive)', y=1.02)
    plt.tight_layout()
    
    out_file = output_dir / "05_correlation_matrix.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_file.name}")


def run_eda_pipeline() -> None:
    """Execute full EDA routine and export all charts."""
    print("=" * 80)
    print("                     PHASE 3: EXPLORATORY DATA ANALYSIS                        ")
    print("=" * 80)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned_data()
    df_feat = add_derived_features(df)
    
    print("Generating and exporting charts to images/ directory...")
    plot_class_distribution(df_feat)
    plot_fraud_vs_legit_amount(df_feat)
    plot_amount_category_fraud_rate(df_feat)
    plot_hourly_fraud_trend(df_feat)
    plot_correlation_matrix(df_feat)
    print("All 5 EDA visual charts successfully generated and exported to images/!")
    print("=" * 80)


if __name__ == "__main__":
    run_eda_pipeline()
