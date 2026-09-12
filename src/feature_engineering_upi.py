"""
Unified Feature Engineering Module for Indian UPI Fraud Detection.

Provides consistent, leak-free feature transformations shared identically
across batch model training and single-row real-time inference.

Features:
1. amount: Numerical transaction amount in INR.
2. hour_of_day: 0 to 23.
3. day_of_week_num: 0 (Monday) to 6 (Sunday).
4. is_weekend: Binary flag (0 or 1).
5. is_night: Binary flag (1 if hour in [0, 1, 2, 3, 4, 5, 23] else 0).
6. is_senior_citizen_sender: Binary flag (1 if sender_age_group == '56+' else 0).
7. is_senior_citizen_receiver: Binary flag (1 if receiver_age_group == '56+' else 0).
8. same_bank_flag: Binary flag (1 if sender_bank == receiver_bank else 0).
9. Encoded Categoricals (with explicit unknown handling):
   - txn_type_encoded
   - category_encoded
   - status_encoded
   - sender_state_encoded
   - sender_bank_encoded
   - receiver_bank_encoded
   - device_type_encoded
   - network_type_encoded
   - sender_age_encoded
   - receiver_age_encoded

Non-model ID/metadata dropped: transaction id, timestamp, day_of_week (raw string)
"""

from typing import Union, Dict, Any, List, Optional
import numpy as np
import pandas as pd

DAY_MAP = {
    'monday': 0, 'mon': 0,
    'tuesday': 1, 'tue': 1,
    'wednesday': 2, 'wed': 2,
    'thursday': 3, 'thu': 3,
    'friday': 4, 'fri': 4,
    'saturday': 5, 'sat': 5,
    'sunday': 6, 'sun': 6
}

UPI_FEATURE_COLUMNS = [
    'amount',
    'hour_of_day',
    'day_of_week_num',
    'is_weekend',
    'is_night',
    'is_senior_citizen_sender',
    'is_senior_citizen_receiver',
    'same_bank_flag',
    'txn_type_encoded',
    'category_encoded',
    'status_encoded',
    'sender_state_encoded',
    'sender_bank_encoded',
    'receiver_bank_encoded',
    'device_type_encoded',
    'network_type_encoded',
    'sender_age_encoded',
    'receiver_age_encoded'
]


def extract_raw_upi_features(data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    """
    Standardizes column names and extracts numerical and boolean signals
    from raw UPI transaction input (DataFrame or single dict).
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Standardize column naming if needed
    col_map = {
        'amount (INR)': 'amount',
        'transaction type': 'transaction_type',
        'merchant_category': 'merchant_category',
        'transaction_status': 'transaction_status',
        'sender_age_group': 'sender_age_group',
        'receiver_age_group': 'receiver_age_group',
        'sender_state': 'sender_state',
        'sender_bank': 'sender_bank',
        'receiver_bank': 'receiver_bank',
        'device_type': 'device_type',
        'network_type': 'network_type',
        'hour_of_day': 'hour_of_day',
        'day_of_week': 'day_of_week',
        'is_weekend': 'is_weekend'
    }
    df = df.rename(columns=col_map)

    # 1. Amount
    df['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0.0)

    # 2. Time & Date Fields
    if 'timestamp' in df.columns and ('hour_of_day' not in df.columns or df['hour_of_day'].isnull().any()):
        dt = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour_of_day'] = dt.dt.hour.fillna(12).astype(int)
        df['day_of_week_num'] = dt.dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)
    else:
        # If hour_of_day is directly present
        df['hour_of_day'] = pd.to_numeric(df.get('hour_of_day', 12), errors='coerce').fillna(12).astype(int)
        
        # Parse day of week if string
        if 'day_of_week' in df.columns:
            if pd.api.types.is_numeric_dtype(df['day_of_week']):
                df['day_of_week_num'] = df['day_of_week'].fillna(0).astype(int)
            else:
                df['day_of_week_num'] = df['day_of_week'].astype(str).str.strip().str.lower().map(DAY_MAP).fillna(0).astype(int)
        else:
            df['day_of_week_num'] = 0

        if 'is_weekend' not in df.columns:
            df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)
        else:
            df['is_weekend'] = pd.to_numeric(df['is_weekend'], errors='coerce').fillna(0).astype(int)

    # 3. Risk Flag Features
    # Late night / early morning (11 PM - 5 AM)
    df['is_night'] = df['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)

    # Senior Citizen Flags (56+)
    s_age = df.get('sender_age_group', '').astype(str).str.strip()
    r_age = df.get('receiver_age_group', '').astype(str).str.strip()
    df['is_senior_citizen_sender'] = (s_age == '56+').astype(int)
    df['is_senior_citizen_receiver'] = (r_age == '56+').astype(int)

    # Same Bank Flag
    s_bank = df.get('sender_bank', '').astype(str).str.strip().str.lower()
    r_bank = df.get('receiver_bank', '').astype(str).str.strip().str.lower()
    df['same_bank_flag'] = (s_bank == r_bank).astype(int)

    return df


def fit_upi_encoders(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits categorical mappings from training partition with explicit unknown fallback indices.
    """
    cat_columns = {
        'transaction_type': 'txn_type',
        'merchant_category': 'category',
        'transaction_status': 'status',
        'sender_state': 'sender_state',
        'sender_bank': 'sender_bank',
        'receiver_bank': 'receiver_bank',
        'device_type': 'device_type',
        'network_type': 'network_type',
        'sender_age_group': 'sender_age',
        'receiver_age_group': 'receiver_age'
    }

    # Normalize column names in train_df if needed
    clean_df = train_df.rename(columns={
        'transaction type': 'transaction_type',
        'merchant_category': 'merchant_category',
        'transaction_status': 'transaction_status',
        'sender_age_group': 'sender_age_group',
        'receiver_age_group': 'receiver_age_group'
    })

    encoder_bundle = {
        'mappings': {},
        'unknown_indices': {},
        'categories_list': {},
        'feature_columns': UPI_FEATURE_COLUMNS
    }

    for col, short_name in cat_columns.items():
        if col in clean_df.columns:
            unique_vals = sorted([str(x).strip() for x in clean_df[col].dropna().unique().tolist()])
        else:
            unique_vals = []
        
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        unknown_idx = len(unique_vals)
        
        encoder_bundle['mappings'][short_name] = mapping
        encoder_bundle['unknown_indices'][short_name] = unknown_idx
        encoder_bundle['categories_list'][short_name] = unique_vals

    return encoder_bundle


def transform_upi_features(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    encoder_bundle: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transforms raw UPI transactions into a guaranteed numerical feature DataFrame.
    Guaranteed column order and data types identical across train, test, and inference.
    Handles unseen categories cleanly using the unknown index fallback.
    """
    df = extract_raw_upi_features(data)

    mappings = encoder_bundle.get('mappings', {})
    unknowns = encoder_bundle.get('unknown_indices', {})

    field_map = {
        'transaction_type': ('txn_type', 'txn_type_encoded'),
        'merchant_category': ('category', 'category_encoded'),
        'transaction_status': ('status', 'status_encoded'),
        'sender_state': ('sender_state', 'sender_state_encoded'),
        'sender_bank': ('sender_bank', 'sender_bank_encoded'),
        'receiver_bank': ('receiver_bank', 'receiver_bank_encoded'),
        'device_type': ('device_type', 'device_type_encoded'),
        'network_type': ('network_type', 'network_type_encoded'),
        'sender_age_group': ('sender_age', 'sender_age_encoded'),
        'receiver_age_group': ('receiver_age', 'receiver_age_encoded')
    }

    for raw_col, (short_name, encoded_col) in field_map.items():
        mapping = mappings.get(short_name, {})
        unk_idx = unknowns.get(short_name, len(mapping))
        
        if raw_col in df.columns:
            s = df[raw_col].astype(str).str.strip()
            df[encoded_col] = s.map(mapping).fillna(unk_idx).astype(int)
        else:
            df[encoded_col] = unk_idx

    # Fixed column output
    X = df[UPI_FEATURE_COLUMNS].copy()
    return X
