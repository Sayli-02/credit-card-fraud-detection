"""
Unified Feature Engineering Module for Sparkov Credit Card Fraud Detection.

Provides consistent, leak-free feature transformation shared identically 
across batch model training and single-row real-time inference.

Engineered Features:
1. age = Precise date-aware calculation:
   trans_year - birth_year - ((trans_month, trans_day) < (birth_month, birth_day))
2. distance_from_home = Vectorized Haversine distance (km) between (lat, long) and (merch_lat, merch_long),
   or category-based statistical median distance when merchant is unknown/unregistered.
3. hour, day_of_week = Extracted from trans_date_trans_time
4. category_encoded, gender_encoded = Safely mapped categorical features with explicit unknown fallback
5. Numerical features: amt, city_pop, lat, long, merch_lat, merch_long

Non-model PII/metadata columns dropped: cc_num, first, last, street, trans_num, unix_time, job, city, state, zip
"""

from typing import Union, Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import date, datetime


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between pairs of coordinates using pure numpy math.
    Works seamlessly on scalar floats, numpy arrays, and pandas Series.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    
    return R * c


def calculate_age_date_aware(trans_date: Union[date, datetime], dob: Union[date, datetime]) -> int:
    """
    Precise date-aware age calculation considering whether birthday has occurred this calendar year.
    Clips age between 18 and 100.
    """
    trans_year = trans_date.year
    birth_year = dob.year
    had_birthday = (trans_date.month, trans_date.day) >= (dob.month, dob.day)
    raw_age = trans_year - birth_year if had_birthday else trans_year - birth_year - 1
    return int(max(18, min(100, raw_age)))


# Standard model feature column ordering
FEATURE_COLUMNS = [
    'amt',
    'distance_from_home',
    'age',
    'hour',
    'day_of_week',
    'category_encoded',
    'gender_encoded',
    'city_pop',
    'lat',
    'long',
    'merch_lat',
    'merch_long'
]


def extract_raw_features(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    distance_override: Optional[float] = None
) -> pd.DataFrame:
    """
    Preprocesses raw transaction fields (single dict or DataFrame) into numerical components:
    - Calculates date-aware age from dob and trans_date_trans_time
    - Calculates distance_from_home using haversine formula (or applies distance_override for unknown merchants)
    - Extracts hour and day_of_week
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # 1. Parse Transaction Date/Time
    if not pd.api.types.is_datetime64_any_dtype(df['trans_date_trans_time']):
        trans_dt = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    else:
        trans_dt = df['trans_date_trans_time']

    # Fallback to current datetime if invalid
    trans_year = trans_dt.dt.year.fillna(2020).astype(int)
    trans_month = trans_dt.dt.month.fillna(1).astype(int)
    trans_day = trans_dt.dt.day.fillna(1).astype(int)
    
    df['hour'] = trans_dt.dt.hour.fillna(12).astype(int)
    df['day_of_week'] = trans_dt.dt.dayofweek.fillna(0).astype(int)

    # 2. Parse Date of Birth & Date-Aware Age Calculation (Task 5)
    if not pd.api.types.is_datetime64_any_dtype(df['dob']):
        dob_dt = pd.to_datetime(df['dob'], errors='coerce')
    else:
        dob_dt = df['dob']
        
    birth_year = dob_dt.dt.year.fillna(1980).astype(int)
    birth_month = dob_dt.dt.month.fillna(1).astype(int)
    birth_day = dob_dt.dt.day.fillna(1).astype(int)

    # Accurate birthday boundary condition
    birthday_not_passed = (trans_month < birth_month) | ((trans_month == birth_month) & (trans_day < birth_day))
    age = trans_year - birth_year - birthday_not_passed.astype(int)
    df['age'] = np.clip(age, 18, 100)

    # 3. Vectorized Haversine Distance or Explicit Distance Override (Task 1)
    lat1 = pd.to_numeric(df['lat'], errors='coerce').fillna(0.0)
    lon1 = pd.to_numeric(df['long'], errors='coerce').fillna(0.0)
    lat2 = pd.to_numeric(df['merch_lat'], errors='coerce').fillna(0.0)
    lon2 = pd.to_numeric(df['merch_long'], errors='coerce').fillna(0.0)

    if distance_override is not None:
        df['distance_from_home'] = float(distance_override)
    elif 'distance_from_home' in df.columns and df['distance_from_home'].notna().all():
        # Keep precomputed distance if explicitly supplied
        pass
    else:
        df['distance_from_home'] = haversine_distance(lat1, lon1, lat2, lon2)
    
    # 4. Fill numeric columns
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce').fillna(0.0)
    df['city_pop'] = pd.to_numeric(df['city_pop'], errors='coerce').fillna(10000.0)
    df['lat'] = lat1
    df['long'] = lon1
    df['merch_lat'] = lat2
    df['merch_long'] = lon2

    return df


def fit_encoders(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits categorical mappings and stores category lists from the raw training set.
    Includes explicit unknown fallback buckets.
    """
    categories = sorted(train_df['category'].dropna().unique().tolist())
    category_map = {cat: idx for idx, cat in enumerate(categories)}
    
    genders = sorted(train_df['gender'].dropna().unique().tolist())
    gender_map = {g: idx for idx, g in enumerate(genders)}
    
    encoder_bundle = {
        'category_map': category_map,
        'gender_map': gender_map,
        'categories': categories,
        'genders': genders,
        'unknown_category_idx': len(categories),
        'unknown_gender_idx': len(genders),
        'feature_columns': FEATURE_COLUMNS
    }
    return encoder_bundle


def transform_features(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    encoder_bundle: Dict[str, Any],
    distance_override: Optional[float] = None
) -> pd.DataFrame:
    """
    Transforms raw transaction inputs into the model-ready feature vector DataFrame.
    Guaranteed column order and data types identical across train, test, and inference.
    Safely catches and handles any unknown categories or genders (Task 2).
    """
    df = extract_raw_features(data, distance_override=distance_override)
    
    category_map = encoder_bundle.get('category_map', {})
    gender_map = encoder_bundle.get('gender_map', {})
    unknown_cat = encoder_bundle.get('unknown_category_idx', len(category_map))
    unknown_gender = encoder_bundle.get('unknown_gender_idx', 0)
    
    # Safe encoding with explicit unknown mapping
    try:
        df['category_encoded'] = df['category'].astype(str).map(category_map).fillna(unknown_cat).astype(int)
    except Exception:
        df['category_encoded'] = unknown_cat

    try:
        df['gender_encoded'] = df['gender'].astype(str).map(gender_map).fillna(unknown_gender).astype(int)
    except Exception:
        df['gender_encoded'] = unknown_gender
    
    # Select exclusively the model feature columns in fixed, validated order
    X = df[FEATURE_COLUMNS].copy()
    return X


def compute_distance_stats(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes and stores dataset distance statistics at training time:
    - Overall median and mean distance
    - Category-level median and mean distance
    """
    print("Computing dataset distance statistics for unknown merchant fallback...")
    
    # Compute distances on training set
    lat1 = train_df['lat'].values
    lon1 = train_df['long'].values
    lat2 = train_df['merch_lat'].values
    lon2 = train_df['merch_long'].values
    
    dists = haversine_distance(lat1, lon1, lat2, lon2)
    train_with_dist = train_df.copy()
    train_with_dist['distance_from_home'] = dists
    
    overall_median = float(np.median(dists))
    overall_mean = float(np.mean(dists))
    
    cat_medians = train_with_dist.groupby('category')['distance_from_home'].median().to_dict()
    cat_means = train_with_dist.groupby('category')['distance_from_home'].mean().to_dict()
    
    stats = {
        'overall_median_distance': round(overall_median, 2),
        'overall_mean_distance': round(overall_mean, 2),
        'category_median_distance': {k: round(float(v), 2) for k, v in cat_medians.items()},
        'category_mean_distance': {k: round(float(v), 2) for k, v in cat_means.items()}
    }
    return stats


def build_merchant_lookup(train_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Builds merchant coordinate lookup dictionary mapping:
    merchant_name -> {'merch_lat': ..., 'merch_long': ..., 'category': ...}
    """
    print("Building merchant coordinate lookup table from training data...")
    grouped = train_df.groupby('merchant').agg(
        merch_lat=('merch_lat', 'mean'),
        merch_long=('merch_long', 'mean'),
        category=('category', lambda s: s.mode().iloc[0] if not s.empty else 'misc_pos')
    ).reset_index()
    
    lookup = {}
    for _, row in grouped.iterrows():
        lookup[str(row['merchant']).strip()] = {
            'merch_lat': float(round(row['merch_lat'], 5)),
            'merch_long': float(round(row['merch_long'], 5)),
            'category': str(row['category'])
        }
    return lookup
