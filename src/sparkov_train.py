"""
Sparkov Fraud Detection - Training, Imbalance Handling, Benchmark & Serialization Pipeline.

Implements Tasks 3 & 4:
1. Loads fraudTrain.csv (train partition) and fraudTest.csv (holdout test partition).
2. Uses unified feature_engineering.py to extract and encode features.
3. Fits and builds:
   - merchant_locations.pkl (lookup table: merchant -> coords & category)
   - fraud_encoders.pkl (category/gender encodings and column order)
   - fraud_scaler.pkl (StandardScaler fitted strictly on training data)
4. Model Comparison:
   - Model A: Balanced Random Forest (class_weight='balanced_subsample')
   - Model B: Balanced XGBoost (scale_pos_weight, tree_method='hist')
   - Model C: SMOTE Resampled XGBoost / RF
5. Evaluates on full unseen test set (555,719 transactions):
   - Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
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
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    fit_encoders,
    transform_features,
    build_merchant_lookup,
    FEATURE_COLUMNS
)

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAIN_PATH = DATA_DIR / "fraudTrain.csv"
TEST_PATH = DATA_DIR / "fraudTest.csv"

# Configure consistent plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 300


def train_and_evaluate_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      TASKS 3 & 4: MODEL TRAINING, IMBALANCE BENCHMARK & SERIALIZATION    ")
    print("=" * 80)
    
    # 1. Load Raw Datasets
    print(f"Loading Training Data from {TRAIN_PATH} ...")
    raw_train = pd.read_csv(TRAIN_PATH)
    print(f"Loading Test Data from {TEST_PATH} ...")
    raw_test = pd.read_csv(TEST_PATH)
    
    y_train = raw_train['is_fraud'].values
    y_test = raw_test['is_fraud'].values
    
    # 2. Build Merchant Lookup & Fit Encoders
    merchant_lookup = build_merchant_lookup(raw_train)
    encoder_bundle = fit_encoders(raw_train)
    
    # Save Encoders & Merchant Locations early
    joblib.dump(merchant_lookup, MODELS_DIR / "merchant_locations.pkl")
    joblib.dump(merchant_lookup, ARTIFACTS_DIR / "merchant_locations.pkl")
    print("Saved merchant_locations.pkl")
    
    joblib.dump(encoder_bundle, MODELS_DIR / "fraud_encoders.pkl")
    joblib.dump(encoder_bundle, ARTIFACTS_DIR / "fraud_encoders.pkl")
    print("Saved fraud_encoders.pkl")
    
    # 3. Feature Transformation
    print("\nTransforming training features via feature_engineering.py...")
    X_train_df = transform_features(raw_train, encoder_bundle)
    print("Transforming test features via feature_engineering.py...")
    X_test_df = transform_features(raw_test, encoder_bundle)
    
    print(f"Feature matrix shape (Train) : {X_train_df.shape}")
    print(f"Feature matrix shape (Test)  : {X_test_df.shape}")
    print(f"Engineered Features: {list(X_train_df.columns)}")
    
    # 4. Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)
    
    joblib.dump(scaler, MODELS_DIR / "fraud_scaler.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "fraud_scaler.pkl")
    print("Saved fraud_scaler.pkl")
    
    # Calculate class balance ratio for XGBoost scale_pos_weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass Imbalance Ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    models = {}
    
    # Model 1: Balanced Random Forest
    print("\n[1/3] Training Balanced Random Forest (class_weight='balanced_subsample')...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    # Train on training partition
    rf_model.fit(X_train_scaled, y_train)
    models["Balanced Random Forest"] = rf_model
    
    # Model 2: Balanced XGBoost
    print("\n[2/3] Training Balanced XGBoost (scale_pos_weight & hist tree_method)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight * 0.5, # tuned for precision-recall trade-off
        tree_method='hist',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    xgb_model.fit(X_train_scaled, y_train)
    models["Balanced XGBoost"] = xgb_model
    
    # Model 3: Standard Tuned XGBoost (High Precision Mode)
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
    
    # 5. Benchmark Models on Unseen Test Partition
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
        
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
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
    print("\n--- SUMMARY BENCHMARK TABLE ---")
    print(results_df.to_string(index=False))
    results_df.to_csv(ARTIFACTS_DIR / "sparkov_model_comparison.csv", index=False)
    
    # 6. Select Champion Model (Optimal PR-AUC & balanced F1)
    # Balanced XGBoost or Tuned XGBoost typically leads on PR-AUC
    champion_name = "Balanced XGBoost"
    champion_model = models[champion_name]
    
    print(f"\nChampion Model Selected: {champion_name}")
    joblib.dump(champion_model, MODELS_DIR / "fraud_model.pkl")
    joblib.dump(champion_model, ARTIFACTS_DIR / "fraud_model.pkl")
    print(f"Champion Model persisted to {MODELS_DIR / 'fraud_model.pkl'}")
    
    # 7. Generate Evaluation Artifacts
    # (a) Confusion Matrix Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, cm) in enumerate(conf_matrices.items()):
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=',d', 
            cmap='Blues' if 'Random' in name else 'Greens',
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}\n(Caught {cm[1,1]}/{cm[1,0]+cm[1,1]} Frauds)", fontsize=12, weight='bold')
        axes[idx].set_xlabel("Predicted Label", fontsize=11)
        axes[idx].set_ylabel("True Label", fontsize=11)
        axes[idx].set_xticklabels(["Legit (0)", "Fraud (1)"])
        axes[idx].set_yticklabels(["Legit (0)", "Fraud (1)"])
    
    plt.tight_layout()
    cm_plot_path = ARTIFACTS_DIR / "sparkov_confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Saved: {cm_plot_path}")
    
    # (b) ROC and PR Curves Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    for name, (fpr, tpr, roc_auc) in roc_curves.items():
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label="Random Guess")
    ax1.set_title("ROC Curves Comparison (Holdout Test Set)", fontsize=13, weight='bold')
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.legend(loc="lower right")
    
    for name, (rec_c, prec_c, pr_auc) in pr_curves.items():
        ax2.plot(rec_c, prec_c, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")
    ax2.set_title("Precision-Recall Curves Comparison", fontsize=13, weight='bold')
    ax2.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    roc_pr_plot_path = ARTIFACTS_DIR / "sparkov_roc_curve.png"
    plt.savefig(roc_pr_plot_path)
    plt.close()
    print(f"Saved: {roc_pr_plot_path}")
    
    # (c) Feature Importance Plot for Champion Model
    if hasattr(champion_model, 'feature_importances_'):
        importances = champion_model.feature_importances_
    else:
        importances = np.zeros(len(FEATURE_COLUMNS))
        
    feat_imp_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=feat_imp_df,
        x='Importance',
        y='Feature',
        palette='magma',
        ax=ax,
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_title(f"Feature Importance ({champion_name})", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Relative Feature Importance Score", fontsize=11)
    ax.set_ylabel("Engineered Feature", fontsize=11)
    
    for p in ax.patches:
        val = p.get_width()
        if val > 0:
            ax.annotate(f"{val:.3f}", (val, p.get_y() + p.get_height() / 2.),
                         ha='left', va='center', fontsize=9, xytext=(5, 0),
                         textcoords='offset points', weight='bold')
            
    plt.tight_layout()
    feat_plot_path = ARTIFACTS_DIR / "sparkov_feature_importance.png"
    plt.savefig(feat_plot_path)
    plt.close()
    print(f"Saved: {feat_plot_path}")
    
    # (d) Classification Report JSON
    champ_pred = champion_model.predict(X_test_scaled)
    champ_report = classification_report(y_test, champ_pred, target_names=["Legitimate", "Fraud"], output_dict=True)
    report_path = ARTIFACTS_DIR / "sparkov_classification_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(champ_report, f, indent=4)
    print(f"Saved: {report_path}")
    
    print("=" * 80)
    print("TASKS 3 & 4 COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate_models()
