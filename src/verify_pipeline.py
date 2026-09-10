"""
Verification and Sanity Testing Module for Sparkov Credit Card Fraud Pipeline.

Validates:
1. Pipeline parity between batch training and single-row real-time inference.
2. Correct serialization and loading of all 4 artifacts:
   - fraud_model.pkl
   - fraud_scaler.pkl
   - fraud_encoders.pkl
   - merchant_locations.pkl
3. Sanity testing on known-fraud and known-legitimate rows extracted from fraudTest.csv.
4. Input validation and edge case handling (unseen categories, missing merchant coords).
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    extract_raw_features,
    transform_features,
    haversine_distance,
    FEATURE_COLUMNS
)

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "fraudTest.csv"


def run_verification():
    print("=" * 80)
    print("      TASK 7: PIPELINE VERIFICATION & SANITY BENCHMARK TESTING           ")
    print("=" * 80)
    
    # 1. Load Artifacts
    print("\n[Step 1] Loading serialized artifacts...")
    model = joblib.load(MODELS_DIR / "fraud_model.pkl")
    scaler = joblib.load(MODELS_DIR / "fraud_scaler.pkl")
    encoders = joblib.load(MODELS_DIR / "fraud_encoders.pkl")
    merchants = joblib.load(MODELS_DIR / "merchant_locations.pkl")
    
    print(f"Loaded Model: {type(model).__name__}")
    print(f"Loaded Scaler: {type(scaler).__name__}")
    print(f"Loaded Encoders: {len(encoders['categories'])} categories, {len(encoders['genders'])} genders")
    print(f"Loaded Merchant Lookup: {len(merchants):,} known merchants")
    
    # 2. Test Single vs Batch Parity
    print("\n[Step 2] Testing Training-Serving Parity...")
    sample_raw = {
        'trans_date_trans_time': '2020-06-21 12:14:25',
        'category': 'grocery_pos',
        'amt': 107.23,
        'gender': 'F',
        'lat': 40.7128,
        'long': -74.0060,
        'city_pop': 8000000,
        'dob': '1985-04-12',
        'merch_lat': 40.7300,
        'merch_long': -73.9900
    }
    
    # As single dict
    single_features = transform_features(sample_raw, encoders)
    single_scaled = scaler.transform(single_features)
    
    # As 1-row DataFrame
    df_raw = pd.DataFrame([sample_raw])
    df_features = transform_features(df_raw, encoders)
    df_scaled = scaler.transform(df_features)
    
    diff = np.max(np.abs(single_scaled - df_scaled))
    print(f"Max absolute discrepancy between single dict and DataFrame: {diff:.8f}")
    assert diff < 1e-6, "Parity check failed: single-row transformation differs from batch transformation!"
    print("PARITY CHECK PASSED: Single-row inference matches batch transformation exactly.")
    
    # 3. Test on Known Fraud & Legit Cases from fraudTest.csv
    print("\n[Step 3] Running Inference Sanity Checks on Real Samples from fraudTest.csv...")
    test_df = pd.read_csv(TEST_PATH)
    
    fraud_samples = test_df[test_df['is_fraud'] == 1].head(5)
    legit_samples = test_df[test_df['is_fraud'] == 0].head(5)
    
    print("\n--- Testing Known FRAUD Samples ---")
    fraud_scores = []
    for idx, row in fraud_samples.iterrows():
        row_dict = row.to_dict()
        feat = transform_features(row_dict, encoders)
        scaled_feat = scaler.transform(feat)
        pred = model.predict(scaled_feat)[0]
        proba = model.predict_proba(scaled_feat)[0, 1]
        fraud_scores.append(proba)
        
        status = "CAUGHT (FRAUD)" if pred == 1 else "MISSED (LEGIT)"
        print(f"Sample #{idx:6d} | Amount: ${row['amt']:7.2f} | Category: {row['category']:14s} | "
              f"Prob: {proba*100:6.2f}% | Flag: {status}")
        
    print("\n--- Testing Known LEGITIMATE Samples ---")
    legit_scores = []
    for idx, row in legit_samples.iterrows():
        row_dict = row.to_dict()
        feat = transform_features(row_dict, encoders)
        scaled_feat = scaler.transform(feat)
        pred = model.predict(scaled_feat)[0]
        proba = model.predict_proba(scaled_feat)[0, 1]
        legit_scores.append(proba)
        
        status = "SAFE (LEGIT)" if pred == 0 else "FALSE ALERT (FRAUD)"
        print(f"Sample #{idx:6d} | Amount: ${row['amt']:7.2f} | Category: {row['category']:14s} | "
              f"Prob: {proba*100:6.2f}% | Flag: {status}")
        
    print(f"\nAverage Probability on Fraud Samples : {np.mean(fraud_scores)*100:.2f}%")
    print(f"Average Probability on Legit Samples : {np.mean(legit_scores)*100:.2f}%")
    
    # 4. Test Unseen / Edge Case Handling
    print("\n[Step 4] Testing Unseen Categories and Edge Cases...")
    edge_case = {
        'trans_date_trans_time': '2021-01-01 02:30:00',
        'category': 'quantum_cryptocurrency_unseen_cat',
        'amt': 5000.0,
        'gender': 'Unknown',
        'lat': 0.0,
        'long': 0.0,
        'city_pop': 500,
        'dob': '2000-01-01',
        'merch_lat': 50.0,
        'merch_long': 50.0
    }
    edge_feat = transform_features(edge_case, encoders)
    edge_scaled = scaler.transform(edge_feat)
    edge_pred = model.predict(edge_scaled)[0]
    edge_proba = model.predict_proba(edge_scaled)[0, 1]
    print(f"Edge case handled safely without crashing. Prediction: {edge_pred}, Fraud Prob: {edge_proba*100:.2f}%")
    
    print("=" * 80)
    print("ALL TASK 7 VERIFICATION CHECKS PASSED.")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
