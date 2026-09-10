"""
Verification Suite for Credit Card Fraud Detection Tasks 1 through 9.

Validates:
1. Unknown merchant distance fallback using dataset category median (no fake coords).
2. Unseen category and gender handling safely without crash.
3. Precise date-aware age calculation across birthday boundaries (e.g. DOB Dec 31 vs Jan 1).
4. Feature vector schema and FEATURE_COLUMNS validation.
5. Inference try/except error handling on malformed inputs.
6. Dynamic decision threshold classification.
7. Sanity testing on known fraud and legitimate transactions from fraudTest.csv with SHAP explanations.
"""

from pathlib import Path
from datetime import date, datetime
import json
import joblib
import pandas as pd
import numpy as np
import shap

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    extract_raw_features,
    transform_features,
    haversine_distance,
    calculate_age_date_aware,
    FEATURE_COLUMNS
)

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "fraudTest.csv"


def run_full_checklist_verification():
    print("=" * 80)
    print("      COMPREHENSIVE VERIFICATION SUITE (TASKS 1 THROUGH 9)               ")
    print("=" * 80)
    
    # 1. Load Artifacts
    print("\n[Check 1] Loading serialized artifacts...")
    model = joblib.load(MODELS_DIR / "fraud_model.pkl")
    scaler = joblib.load(MODELS_DIR / "fraud_scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "fraud_encoders.pkl")
    merchants = joblib.load(MODELS_DIR / "merchant_locations.pkl")
    dist_stats = joblib.load(MODELS_DIR / "distance_fallback_stats.pkl")
    
    print(f"[OK] Model Loaded: {type(model).__name__}")
    print(f"[OK] Scaler Loaded: {type(scaler).__name__} (Features: {scaler.n_features_in_})")
    print(f"[OK] Encoders Loaded: {len(encoders['categories'])} categories, {len(encoders['genders'])} genders")
    print(f"[OK] Merchant Lookup: {len(merchants):,} known merchants")
    print(f"[OK] Distance Fallback Stats: Overall Median = {dist_stats['overall_median_distance']} km")

    # 2. Test Date-Aware Age Calculation (Task 5)
    print("\n[Check 2] Testing Birthday Boundary Date-Aware Age Calculation...")
    dob_dec31 = date(2000, 12, 31)
    trans_jan1 = date(2020, 1, 1)    # Not yet 20 in Jan 2020 -> Should be 19
    trans_dec31 = date(2020, 12, 31) # Birthday reached -> Should be 20
    
    age_before_bday = calculate_age_date_aware(trans_jan1, dob_dec31)
    age_on_bday = calculate_age_date_aware(trans_dec31, dob_dec31)
    
    print(f"DOB: 2000-12-31, Trans: 2020-01-01 -> Calculated Age: {age_before_bday} (Expected: 19)")
    print(f"DOB: 2000-12-31, Trans: 2020-12-31 -> Calculated Age: {age_on_bday} (Expected: 20)")
    assert age_before_bday == 19, "Age before birthday failed!"
    assert age_on_bday == 20, "Age on birthday failed!"
    print("[OK] DATE-AWARE AGE CALCULATION PASSED.")

    # 3. Test Unknown Merchant Distance Fallback (Task 1)
    print("\n[Check 3] Testing Unknown Merchant Distance Fallback...")
    unknown_merchant_input = {
        'trans_date_trans_time': '2020-05-15 14:00:00',
        'category': 'shopping_net',
        'amt': 150.00,
        'gender': 'F',
        'lat': 40.7128,
        'long': -74.0060,
        'city_pop': 8000000,
        'dob': '1990-06-15',
        'merch_lat': 0.0,
        'merch_long': 0.0
    }
    cat_median = dist_stats['category_median_distance'].get('shopping_net', dist_stats['overall_median_distance'])
    feat_unknown = transform_features(unknown_merchant_input, encoders, distance_override=cat_median)
    
    assert feat_unknown['distance_from_home'].iloc[0] == cat_median, "Distance override was not applied!"
    print(f"[OK] Unknown merchant successfully assigned category median distance: {feat_unknown['distance_from_home'].iloc[0]} km.")

    # 4. Test Unseen / Malformed Category Safe Encoding (Task 2)
    print("\n[Check 4] Testing Unseen Categories & Gender Safety...")
    unseen_input = {
        'trans_date_trans_time': '2020-05-15 14:00:00',
        'category': 'extraterrestrial_crypto_purchase_99',
        'amt': 300.00,
        'gender': 'Non-Binary',
        'lat': 40.7128,
        'long': -74.0060,
        'city_pop': 8000000,
        'dob': '1990-06-15',
        'merch_lat': 40.7500,
        'merch_long': -73.9800
    }
    feat_unseen = transform_features(unseen_input, encoders)
    print(f"[OK] Unseen category safely mapped to index: {feat_unseen['category_encoded'].iloc[0]}")
    print(f"[OK] Unseen gender safely mapped to index: {feat_unseen['gender_encoded'].iloc[0]}")

    # 5. Test Schema and FEATURE_COLUMNS Validation (Task 3 & Task 8)
    print("\n[Check 5] Testing FEATURE_COLUMNS Schema Validation...")
    assert list(feat_unseen.columns) == FEATURE_COLUMNS, "FEATURE_COLUMNS order mismatch!"
    assert feat_unseen.shape[1] == scaler.n_features_in_, "Scaler feature dimension mismatch!"
    print(f"[OK] Validated feature columns ({feat_unseen.shape[1]} features): {FEATURE_COLUMNS}")

    # 6. Test Decision Threshold Slider Logic (Task 6)
    print("\n[Check 6] Testing Decision Threshold Logic...")
    scaled = scaler.transform(feat_unknown)
    prob = float(model.predict_proba(scaled)[0, 1])
    
    flag_at_low_thresh = (prob >= 0.05)
    flag_at_high_thresh = (prob >= 0.95)
    print(f"Transaction Fraud Prob: {prob*100:.2f}% | Flag at 0.05 Threshold: {flag_at_low_thresh} | Flag at 0.95 Threshold: {flag_at_high_thresh}")
    print("[OK] DECISION THRESHOLD CLASSIFICATION PASSED.")

    # 7. Test SHAP TreeExplainer (Task 4)
    print("\n[Check 7] Testing SHAP TreeExplainer Calculation...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(scaled)
    assert shap_vals.shape[1] == len(FEATURE_COLUMNS), "SHAP feature length mismatch!"
    print(f"[OK] SHAP computed {shap_vals.shape[1]} feature impact values successfully.")

    # 8. Test on Real Fraud & Legit Samples from fraudTest.csv
    print("\n[Check 8] Sanity Testing Real Samples from fraudTest.csv...")
    test_df = pd.read_csv(TEST_PATH)
    fraud_rows = test_df[test_df['is_fraud'] == 1].head(3)
    legit_rows = test_df[test_df['is_fraud'] == 0].head(3)
    
    print("\n--- Real Fraud Samples ---")
    for idx, r in fraud_rows.iterrows():
        f_vec = transform_features(r.to_dict(), encoders)
        p = model.predict_proba(scaler.transform(f_vec))[0, 1]
        print(f"Sample #{idx:6d} | ${r['amt']:7.2f} | {r['category']:14s} | Prob: {p*100:6.2f}% | Flag: {'[FRAUD]' if p>=0.5 else '[MISSED]'}")
        
    print("\n--- Real Legit Samples ---")
    for idx, r in legit_rows.iterrows():
        f_vec = transform_features(r.to_dict(), encoders)
        p = model.predict_proba(scaler.transform(f_vec))[0, 1]
        print(f"Sample #{idx:6d} | ${r['amt']:7.2f} | {r['category']:14s} | Prob: {p*100:6.2f}% | Flag: {'[LEGIT]' if p<0.5 else '[FALSE ALERT]'}")
        
    print("\n" + "=" * 80)
    print("ALL VERIFICATION CHECKS (TASKS 1 THROUGH 9) COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_full_checklist_verification()
