"""
Unified Feature Engineering Module for Sparkov Credit Card Fraud Detection.

Provides consistent, leak-free feature transformation shared identically 
across batch model training and single-row real-time inference.

Engineered Features:
1. age = transaction year - birth year (extracted from dob and trans_date_trans_time)
2. distance_from_home = Vectorized Haversine distance (km) between (lat, long) and (merch_lat, merch_long)
3. hour, day_of_week = Extracted from trans_date_trans_time
4. category_encoded, gender_encoded = Mapped/encoded categorical features
5. Numerical features: amt, city_pop, lat, long, merch_lat, merch_long

Non-model PII/metadata columns dropped: cc_num, first, last, street, trans_num, unix_time, job, city, state, zip
"""

from typing import Union, Dict, Any, List, Tuple
import numpy as np
import pandas as pd


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


def extract_raw_features(data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    """
    Preprocesses raw transaction fields (single dict or DataFrame) into numerical components:
    - Calculates age from dob and trans_date_trans_time
    - Calculates distance_from_home using haversine formula
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

    # Fallback to current year if invalid
    trans_year = trans_dt.dt.year.fillna(2020).astype(int)
    df['hour'] = trans_dt.dt.hour.fillna(12).astype(int)
    df['day_of_week'] = trans_dt.dt.dayofweek.fillna(0).astype(int)

    # 2. Parse Date of Birth & Calculate Age
    if not pd.api.types.is_datetime64_any_dtype(df['dob']):
        dob_dt = pd.to_datetime(df['dob'], errors='coerce')
    else:
        dob_dt = df['dob']
        
    birth_year = dob_dt.dt.year.fillna(1980).astype(int)
    age = trans_year - birth_year
    # Clip age to realistic bounds
    df['age'] = np.clip(age, 18, 100)

    # 3. Vectorized Haversine Distance
    lat1 = pd.to_numeric(df['lat'], errors='coerce').fillna(0.0)
    lon1 = pd.to_numeric(df['long'], errors='coerce').fillna(0.0)
    lat2 = pd.to_numeric(df['merch_lat'], errors='coerce').fillna(0.0)
    lon2 = pd.to_numeric(df['merch_long'], errors='coerce').fillna(0.0)

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
    Fits categorical mappings and merchant locations from the raw training set.
    Returns an encoder artifact dictionary for serialization.
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
        'feature_columns': FEATURE_COLUMNS
    }
    return encoder_bundle


def transform_features(
    data: Union[pd.DataFrame, Dict[str, Any]], 
    encoder_bundle: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transforms raw transaction inputs into the model-ready feature vector DataFrame.
    Guaranteed column order and data types identical across train, test, and inference.
    """
    df = extract_raw_features(data)
    
    category_map = encoder_bundle.get('category_map', {})
    gender_map = encoder_bundle.get('gender_map', {})
    
    # Encode Category (map known categories; unseen assigned -1 or max index)
    default_cat_idx = len(category_map)
    df['category_encoded'] = df['category'].astype(str).map(category_map).fillna(default_cat_idx).astype(int)
    
    # Encode Gender (map known; unseen assigned default 0)
    df['gender_encoded'] = df['gender'].astype(str).map(gender_map).fillna(0).astype(int)
    
    # Select exclusively the model feature columns in fixed order
    X = df[FEATURE_COLUMNS].copy()
    return X


def build_merchant_lookup(train_df: pd.DataFrame) -> Dict[str, Tuple[float, float, str, str]]:
    """
    Builds merchant coordinate lookup dictionary mapping:
    merchant_name -> (merch_lat, merch_long, most_common_category, most_common_city/state)
    """
    print("Building merchant coordinate lookup table from training data...")
    lookup = {}
    grouped = train_df.groupby('merchant').agg(
        merch_lat=('merch_lat', 'mean'),
        merch_long=('merch_long', 'mean'),
        category=('category', lambda s: s.mode().iloc[0] if not s.empty else 'misc_pos')
    ).reset_index()
    
    for _, row in grouped.iterrows():
        lookup[str(row['merchant']).strip()] = {
            'merch_lat': float(round(row['merch_lat'], 5)),
            'merch_long': float(round(row['merch_long'], 5)),
            'category': str(row['category'])
        }
    return lookup
