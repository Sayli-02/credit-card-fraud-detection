"""
Machine Learning Modeling, Resampling (SMOTE), and Evaluation Pipeline.

Implements:
1. Stratified 80/20 train/test split (random_state=42).
2. Data-leakage-free feature scaling (StandardScaler fitted strictly on training data).
3. Baseline Model: Logistic Regression without resampling.
4. SMOTE Resampling: Over-sampling minority class on training set only.
5. Resampled Models:
   - Logistic Regression with SMOTE
   - Random Forest Classifier with SMOTE
6. Evaluation Metrics:
   - Precision, Recall, F1-Score
   - ROC-AUC
   - PR-AUC (Average Precision / Area under Precision-Recall Curve)
7. Artifact Generation:
   - Model Comparison Table (CSV)
   - ROC and Precision-Recall Curves (PNG)
   - Confusion Matrices Comparison (PNG)
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_cleaned.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = PROJECT_ROOT / "images"
MODELS_DIR = PROJECT_ROOT / "models"

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300


def load_and_split_data(file_path: Path = CLEANED_DATA_PATH, test_size: float = 0.20, random_state: int = 42):
    """Load cleaned data and perform stratified 80/20 train/test split."""
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {file_path}.")
    df = pd.read_csv(file_path)
    
    feature_cols = [c for c in df.columns if c != 'Class']
    X = df[feature_cols].copy()
    y = df['Class'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data Split: Train={X_train.shape[0]:,} rows ({y_train.sum()} fraud), Test={X_test.shape[0]:,} rows ({y_test.sum()} fraud)")
    return X_train, X_test, y_train, y_test, feature_cols


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler fitted ONLY on training data to prevent data leakage.
    Scales Amount and Time (or all features).
    """
    scaler = StandardScaler()
    # We fit scaler on the full feature matrix (V1-V28 are zero-mean PCA, Time and Amount are non-standardized)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Persist scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print("StandardScaler fitted on training data and saved to models/scaler.joblib")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train_scaled: np.ndarray, y_train: pd.Series, random_state: int = 42):
    """Apply SMOTE over-sampling strictly to training partition."""
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"SMOTE Applied on Training Set: Before={len(y_train):,} ({y_train.sum()} fraud) -> After={len(y_train_res):,} ({y_train_res.sum()} fraud)")
    return X_train_res, y_train_res


def train_models(X_train_scaled: np.ndarray, y_train: pd.Series, 
                 X_train_res: np.ndarray, y_train_res: np.ndarray) -> Dict[str, Any]:
    """Train Baseline Logistic Regression, SMOTE Logistic Regression, and SMOTE Random Forest."""
    models = {}
    
    # 1. Baseline Logistic Regression (no resampling)
    print("\n[1/3] Training Baseline Logistic Regression (No Resampling)...")
    lr_baseline = LogisticRegression(max_iter=1000, random_state=42, class_weight=None)
    lr_baseline.fit(X_train_scaled, y_train)
    models['Logistic Regression (Baseline)'] = lr_baseline
    
    # 2. SMOTE Logistic Regression
    print("[2/3] Training Logistic Regression with SMOTE...")
    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_res, y_train_res)
    models['Logistic Regression (SMOTE)'] = lr_smote
    
    # 3. SMOTE Random Forest Classifier
    print("[3/3] Training Random Forest with SMOTE (n_estimators=100, max_depth=15)...")
    rf_smote = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_smote.fit(X_train_res, y_train_res)
    models['Random Forest (SMOTE)'] = rf_smote
    
    return models


def evaluate_models(models: Dict[str, Any], X_test_scaled: np.ndarray, y_test: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate all models on the pristine test partition."""
    results = []
    eval_data = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.decision_function(X_test_scaled)
            
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'Model': name,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4),
            'PR-AUC': round(pr_auc, 4),
            'True Positives (TP)': int(tp),
            'False Negatives (FN)': int(fn),
            'False Positives (FP)': int(fp),
            'True Negatives (TN)': int(tn)
        })
        
        # Store detailed curve and CM arrays
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        pr_p, pr_r, _ = precision_recall_curve(y_test, y_proba)
        eval_data[name] = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'pr_precision': pr_p,
            'pr_recall': pr_r,
            'metrics': {'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC_AUC': roc_auc, 'PR_AUC': pr_auc}
        }
        
    df_results = pd.DataFrame(results)
    return df_results, eval_data


def plot_model_evaluation_curves(eval_data: Dict[str, Any], output_dir: Path = IMAGES_DIR) -> None:
    """Plot ROC and Precision-Recall curves side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'Logistic Regression (Baseline)': '#2b83ba', 
              'Logistic Regression (SMOTE)': '#fdae61', 
              'Random Forest (SMOTE)': '#d7191c'}
    
    # 1. ROC Curves
    for name, data in eval_data.items():
        roc_auc = data['metrics']['ROC_AUC']
        ax1.plot(data['fpr'], data['tpr'], color=colors.get(name, 'blue'), linewidth=2.5,
                 label=f"{name} (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance (AUC = 0.5000)')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curves', pad=12)
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Recall / Sensitivity)')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Precision-Recall Curves
    for name, data in eval_data.items():
        pr_auc = data['metrics']['PR_AUC']
        ax2.plot(data['pr_recall'], data['pr_precision'], color=colors.get(name, 'blue'), linewidth=2.5,
                 label=f"{name} (PR-AUC = {pr_auc:.4f})")
    ax2.set_title('Precision-Recall (PR) Curves [Crucial for Imbalance]', pad=12)
    ax2.set_xlabel('Recall (Fraud Detection Rate)')
    ax2.set_ylabel('Precision (True Positive Reliability)')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Model Performance Curves on Pristine Test Set (20% Split, 95 Fraud Cases)', y=1.02)
    plt.tight_layout()
    
    out_file = output_dir / "06_roc_pr_curves.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_file.name}")


def plot_confusion_matrices(eval_data: Dict[str, Any], output_dir: Path = IMAGES_DIR) -> None:
    """Plot Confusion Matrices for all models side-by-side."""
    n_models = len(eval_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, data) in zip(axes, eval_data.items()):
        cm = data['cm']
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Annotations text
        annot_text = np.array([
            [f"TN\n{tn:,}\n({tn/(tn+fp)*100:.1f}%)", f"FP\n{fp:,}\n(Alerts: {fp})"],
            [f"FN (Missed)\n{fn:,}\n({fn/(tp+fn)*100:.1f}%)", f"TP (Caught)\n{tp:,}\n({recall*100:.1f}%)"]
        ])
        
        sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Pred Legit', 'Pred Fraud'],
                    yticklabels=['Actual Legit', 'Actual Fraud'],
                    annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_title(f"{name}\nRecall: {recall*100:.1f}% | Precision: {prec*100:.1f}%", pad=10)
        
    plt.suptitle('Confusion Matrix Comparison: Trade-off Between Caught Fraud (TP) and False Alerts (FP)', y=1.05)
    plt.tight_layout()
    
    out_file = output_dir / "07_confusion_matrices.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Exported: {out_file.name}")


def run_modeling_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Execute full Phase 5 modeling and evaluation workflow."""
    print("=" * 80)
    print("                    PHASE 5: MACHINE LEARNING & SMOTE                          ")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test, feature_cols = load_and_split_data()
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)
    
    models = train_models(X_train_scaled, y_train, X_train_res, y_train_res)
    df_results, eval_data = evaluate_models(models, X_test_scaled, y_test)
    
    # Persist comparison results
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)
    print(f"\nModel Comparison Saved to: {PROCESSED_DIR / 'model_comparison.csv'}")
    
    # Plot evaluation artifacts
    plot_model_evaluation_curves(eval_data)
    plot_confusion_matrices(eval_data)
    
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(df_results.to_string(index=False))
    print("=" * 80)
    
    return df_results, models, eval_data


if __name__ == "__main__":
    run_modeling_pipeline()
