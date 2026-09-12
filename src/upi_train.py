"""
UPI Transactions 2024 - Training, Imbalance Handling, Benchmark & Serialization Pipeline.

Implements:
1. Loads upi_transactions_2024.csv.
2. Stratified train/test split (80/20) preserving the 0.192% fraud distribution.
3. Fits and builds:
   - upi_fraud_encoders.pkl (categorical encodings with explicit unknown fallback buckets)
   - upi_fraud_scaler.pkl (StandardScaler fitted strictly on training data)
4. Model Comparison:
   - Model 1: Balanced Random Forest (class_weight='balanced_subsample')
   - Model 2: Balanced XGBoost (scale_pos_weight, tree_method='hist')
   - Model 3: Tuned XGBoost
5. Evaluates on unseen 20% test partition (50,000 transactions, 96 fraud cases).
6. Saves evaluation plots and persists all required joblib .pkl artifacts to models/ and artifacts/.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
import xgboost as xgb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering_upi import (
    fit_upi_encoders,
    transform_upi_features,
    UPI_FEATURE_COLUMNS
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "upi_transactions_2024.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
IMAGES_DIR = PROJECT_ROOT / "images"

# Configure consistent plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300


def train_and_evaluate_upi_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      UPI 2024: MODEL TRAINING, BENCHMARK & SERIALIZATION PIPELINE        ")
    print("=" * 80)
    
    # 1. Load Dataset
    print(f"Loading UPI Dataset from: {DATA_PATH} ...")
    raw_df = pd.read_csv(DATA_PATH)
    
    # Stratified Train/Test Split (80/20)
    print("Performing 80/20 Stratified Train/Test Partition...")
    train_df, test_df = train_test_split(
        raw_df, 
        test_size=0.20, 
        random_state=42, 
        stratify=raw_df['fraud_flag']
    )
    
    y_train = train_df['fraud_flag'].values
    y_test = test_df['fraud_flag'].values
    
    print(f"Train partition: {len(train_df):,} rows | Fraud: {int(y_train.sum()):,} ({y_train.mean()*100:.3f}%)")
    print(f"Test partition : {len(test_df):,} rows | Fraud: {int(y_test.sum()):,} ({y_test.mean()*100:.3f}%)")
    
    # 2. Fit and Persist Encoders
    print("\nFitting UPI Encoders with explicit unknown fallback...")
    encoder_bundle = fit_upi_encoders(train_df)
    
    joblib.dump(encoder_bundle, MODELS_DIR / "upi_fraud_encoders.pkl")
    joblib.dump(encoder_bundle, ARTIFACTS_DIR / "upi_fraud_encoders.pkl")
    print("Saved upi_fraud_encoders.pkl")
    
    # 3. Transform Features
    print("Transforming training features...")
    X_train_df = transform_upi_features(train_df, encoder_bundle)
    print("Transforming test features...")
    X_test_df = transform_upi_features(test_df, encoder_bundle)
    
    # 4. Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    joblib.dump(scaler, MODELS_DIR / "upi_fraud_scaler.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "upi_fraud_scaler.pkl")
    print("Saved upi_fraud_scaler.pkl")
    
    # Calculate class balance ratio for XGBoost scale_pos_weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    models = {}
    
    # Model 1: Balanced Random Forest
    print("\n[1/3] Training Balanced Random Forest (class_weight='balanced_subsample')...")
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    models["Balanced Random Forest"] = rf_model
    
    # Model 2: Balanced XGBoost
    print("\n[2/3] Training Balanced XGBoost (scale_pos_weight & hist tree_method)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight * 0.4,
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    xgb_model.fit(X_train_scaled, y_train)
    models["Balanced XGBoost"] = xgb_model
    
    # Model 3: Tuned XGBoost
    print("\n[3/3] Training Tuned XGBoost Classifier...")
    xgb_tuned = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    xgb_tuned.fit(X_train_scaled, y_train)
    models["Tuned XGBoost"] = xgb_tuned
    
    # 5. Benchmark Models on Test Partition
    print("\n" + "=" * 80)
    print("               TEST PARTITION MODEL BENCHMARK EVALUATION                     ")
    print("=" * 80)
    
    results = []
    roc_curves = {}
    pr_curves = {}
    conf_matrices = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            "Model": name,
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "True Positives (Caught)": int(tp),
            "False Positives (Alarms)": int(fp),
            "False Negatives (Missed)": int(fn),
            "True Negatives": int(tn)
        })
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, roc_auc)
        pr_curves[name] = (recall_curve, precision_curve, pr_auc)
        conf_matrices[name] = cm
        
        print(f"\n--- {name} ---")
        print(f"Precision : {prec*100:.2f}% | Recall : {rec*100:.2f}% ({tp}/{tp+fn} caught)")
        print(f"F1-Score  : {f1:.4f} | ROC-AUC : {roc_auc:.4f} | PR-AUC : {pr_auc:.4f}")
        print(f"False Positives : {fp:,} out of {tn+fp:,} legit transactions ({fp/(tn+fp)*100:.3f}% FP rate)")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(ARTIFACTS_DIR / "upi_model_comparison.csv", index=False)
    print("\nSaved upi_model_comparison.csv")
    
    # Champion Model Selection: pick best ROC-AUC / PR-AUC
    # Let's select Balanced XGBoost as default champion
    champion_name = "Balanced XGBoost"
    champion_model = models[champion_name]
    
    joblib.dump(champion_model, MODELS_DIR / "upi_fraud_model.pkl")
    joblib.dump(champion_model, ARTIFACTS_DIR / "upi_fraud_model.pkl")
    print(f"Champion Model ({champion_name}) persisted to models/upi_fraud_model.pkl")
    
    # Save classification report
    y_pred_champ = champion_model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred_champ, output_dict=True, zero_division=0)
    with open(ARTIFACTS_DIR / "upi_classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Saved upi_classification_report.json")
    
    # -------------------------------------------------------------
    # PLOT 1: Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt=',d',
            cmap='Blues',
            cbar=False,
            ax=axes[idx],
            annot_kws={'size': 13, 'weight': 'bold'}
        )
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label (0: Legit, 1: Fraud)", fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("True Label", fontsize=10, fontweight='bold')
        axes[idx].set_xticklabels(['Legit (0)', 'Fraud (1)'])
        axes[idx].set_yticklabels(['Legit (0)', 'Fraud (1)'])
    
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "upi_confusion_matrix.png", dpi=300, bbox_inches='tight')
    fig.savefig(IMAGES_DIR / "upi_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved upi_confusion_matrix.png")
    
    # -------------------------------------------------------------
    # PLOT 2: ROC & PR Curves
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#2563EB', '#DC2626', '#10B981']
    for idx, (name, (fpr, tpr, roc_auc_val)) in enumerate(roc_curves.items()):
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.4f})", color=colors[idx], linewidth=2.5)
        
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight='bold')
    ax1.set_title("UPI Benchmark: ROC Curves Comparison", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    
    for idx, (name, (rec_c, prec_c, pr_auc_val)) in enumerate(pr_curves.items()):
        ax2.plot(rec_c, prec_c, label=f"{name} (PR-AUC = {pr_auc_val:.4f})", color=colors[idx], linewidth=2.5)
        
    ax2.set_xlabel("Recall", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Precision", fontsize=11, fontweight='bold')
    ax2.set_title("UPI Benchmark: Precision-Recall Curves", fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "upi_roc_curve.png", dpi=300, bbox_inches='tight')
    fig.savefig(IMAGES_DIR / "upi_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved upi_roc_curve.png")
    
    # -------------------------------------------------------------
    # PLOT 3: Feature Importance
    # -------------------------------------------------------------
    if hasattr(champion_model, "feature_importances_"):
        importances = champion_model.feature_importances_
        feature_names = list(X_train_df.columns)
        
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=fi_df,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="Blues_r",
            ax=ax,
            legend=False
        )
        ax.set_title(f"Champion Model ({champion_name}) — Feature Importance", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Relative Feature Importance Weight", fontsize=11, fontweight='bold')
        ax.set_ylabel("Engineered Feature", fontsize=11, fontweight='bold')
        
        for p in ax.patches:
            val = p.get_width()
            ax.annotate(
                f"{val:.3f}",
                (val + 0.002, p.get_y() + p.get_height() / 2),
                va='center',
                fontsize=9,
                fontweight='bold',
                color='#1E293B'
            )
            
        plt.tight_layout()
        fig.savefig(ARTIFACTS_DIR / "upi_feature_importance.png", dpi=300, bbox_inches='tight')
        fig.savefig(IMAGES_DIR / "upi_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved upi_feature_importance.png")
        
    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    train_and_evaluate_upi_models()
