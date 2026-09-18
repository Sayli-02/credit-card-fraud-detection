import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
try:
    import shap
except ImportError:
    shap = None
from datetime import datetime, date

# Import shared feature engineering logic and schema
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_engineering import (
    extract_raw_features,
    transform_features,
    haversine_distance,
    calculate_age_date_aware,
    FEATURE_COLUMNS
)

# Setup page config
st.set_page_config(
    page_title="CardShield & UPI — Fraud Detection Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a Friendly, Clean, and Modern Design
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    h1 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    h2, h3, h4 {
        color: #1E293B !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    div.stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #F1F5F9;
        margin-bottom: 1.5rem;
    }
    div.stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        font-size: 1.05rem;
        color: #64748B;
    }
    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #2563EB;
        color: #2563EB;
        background-color: #EFF6FF;
    }

    /* Friendly stat boxes */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
        height: 100%;
    }
    .stat-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stat-number {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .stat-desc {
        font-size: 0.85rem;
        color: #059669;
        font-weight: 600;
    }
    .stat-desc-alert {
        font-size: 0.85rem;
        color: #DC2626;
        font-weight: 600;
    }

    /* Styled buttons */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px -3px rgba(0, 0, 0, 0.08) !important;
        border-color: #94A3B8 !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 20px !important;
        font-size: 1.1rem !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        color: white !important;
    }

    /* Result Card Styles */
    .fraud-box {
        background-color: #FEF2F2;
        border: 2px solid #FCA5A5;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .legit-box {
        background-color: #F0FDF4;
        border: 2px solid #86EFAC;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .reason-item {
        background: white;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border: 1px solid #E2E8F0;
        font-size: 0.95rem;
    }

    .section-framing {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(BASE_DIR, 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

SPARKOV_MODEL_PATH = os.path.join(MODELS_DIR, 'fraud_model.pkl')
SPARKOV_SCALER_PATH = os.path.join(MODELS_DIR, 'fraud_scaler.pkl')
SPARKOV_ENCODERS_PATH = os.path.join(MODELS_DIR, 'fraud_encoders.pkl')
SPARKOV_MERCHANTS_PATH = os.path.join(MODELS_DIR, 'merchant_locations.pkl')
SPARKOV_DIST_STATS_PATH = os.path.join(MODELS_DIR, 'distance_fallback_stats.pkl')


@st.cache_resource
def load_app_assets():
    """Loads all models, merchant directories, and background intelligence safely."""
    if not os.path.exists(SPARKOV_MODEL_PATH):
        st.error("System setup is missing model files. Please run setup first.")
        st.stop()
        
    model = joblib.load(SPARKOV_MODEL_PATH)
    scaler = joblib.load(SPARKOV_SCALER_PATH)
    encoders = joblib.load(SPARKOV_ENCODERS_PATH)
    merchants = joblib.load(SPARKOV_MERCHANTS_PATH)
    
    if os.path.exists(SPARKOV_DIST_STATS_PATH):
        dist_stats = joblib.load(SPARKOV_DIST_STATS_PATH)
    else:
        dist_stats = {
            'overall_median_distance': 78.23,
            'category_median_distance': {}
        }
    
    explainer = None
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = None
        
    return model, scaler, encoders, merchants, dist_stats, explainer


model, scaler, encoders, merchant_lookup, dist_fallback_stats, shap_explainer = load_app_assets()

# User-Friendly Category Labels
CATEGORY_LABELS = {
    'shopping_net': '🌐 Online Shopping (Amazon, Flipkart, Myntra, etc.)',
    'misc_net': '🌐 Online Electronics & Digital Purchases',
    'grocery_pos': '🛒 Supermarket & Grocery Store (DMart, Reliance, Local Kirana)',
    'shopping_pos': '🛍️ Shopping Mall / Retail Store (In-Person)',
    'gas_transport': '⛽ Petrol Pump & Fuel Station (IndianOil, HPCL, BPCL)',
    'food_dining': '🍔 Restaurants, Swiggy, Zomato & Dining',
    'entertainment': '🎬 Movies (PVR / Inox), BookMyShow & Events',
    'travel': '✈️ Flights, Hotels & IRCTC / MakeMyTrip',
    'personal_care': '💇 Salon, Spa & Beauty Care',
    'health_fitness': '💊 Pharmacy (Apollo, Netmeds) & Clinics',
    'home': '🛋️ Furniture & Home Improvement',
    'kids_pets': '🐶 Pet Supplies & Baby Care',
    'misc_pos': '🏪 Other Local Stores & Shops',
    'grocery_net': '📦 Instant Grocery Delivery (Blinkit, Zepto, BigBasket)'
}

# ----------------- EXTENSIVE INDIA & GLOBAL CITIES DATABASE -----------------
ALL_LOCATIONS = {
    # --- Major Indian Metro Cities ---
    "🇮🇳 Mumbai, Maharashtra": {"lat": 19.0760, "long": 72.8777, "pop": 12500000},
    "🇮🇳 Delhi NCR (New Delhi)": {"lat": 28.6139, "long": 77.2090, "pop": 11000000},
    "🇮🇳 Bengaluru, Karnataka": {"lat": 12.9716, "long": 77.5946, "pop": 8400000},
    "🇮🇳 Hyderabad, Telangana": {"lat": 17.3850, "long": 78.4867, "pop": 6800000},
    "🇮🇳 Chennai, Tamil Nadu": {"lat": 13.0827, "long": 80.2707, "pop": 4680000},
    "🇮🇳 Kolkata, West Bengal": {"lat": 22.5726, "long": 88.3639, "pop": 4500000},
    "🇮🇳 Pune, Maharashtra": {"lat": 18.5204, "long": 73.8567, "pop": 3124000},
    "🇮🇳 Ahmedabad, Gujarat": {"lat": 23.0225, "long": 72.5714, "pop": 5570000},
    
    # --- Other Major Indian Cities (Tier 1 & 2) ---
    "🇮🇳 Jaipur, Rajasthan": {"lat": 26.9124, "long": 75.7873, "pop": 3046000},
    "🇮🇳 Surat, Gujarat": {"lat": 21.1702, "long": 72.8311, "pop": 4467000},
    "🇮🇳 Lucknow, Uttar Pradesh": {"lat": 26.8467, "long": 80.9462, "pop": 2817000},
    "🇮🇳 Chandigarh (Punjab / Haryana)": {"lat": 30.7333, "long": 76.7794, "pop": 1055000},
    "🇮🇳 Gurugram (Gurgaon), Haryana": {"lat": 28.4595, "long": 77.0266, "pop": 876000},
    "🇮🇳 Noida / Greater Noida, UP": {"lat": 28.5355, "long": 77.3910, "pop": 637000},
    "🇮🇳 Indore, Madhya Pradesh": {"lat": 22.7196, "long": 75.8577, "pop": 1994000},
    "🇮🇳 Bhopal, Madhya Pradesh": {"lat": 23.2599, "long": 77.4126, "pop": 1798000},
    "🇮🇳 Nagpur, Maharashtra": {"lat": 21.1458, "long": 79.0882, "pop": 2405000},
    "🇮🇳 Patna, Bihar": {"lat": 25.5941, "long": 85.1376, "pop": 1684000},
    "🇮🇳 Vadodara, Gujarat": {"lat": 22.3072, "long": 73.1812, "pop": 1822000},
    "🇮🇳 Visakhapatnam (Vizag), AP": {"lat": 17.6868, "long": 83.2185, "pop": 1728000},
    "🇮🇳 Kochi / Ernakulam, Kerala": {"lat": 9.9312, "long": 76.2673, "pop": 677000},
    "🇮🇳 Thiruvananthapuram, Kerala": {"lat": 8.5241, "long": 76.9366, "pop": 957000},
    "🇮🇳 Coimbatore, Tamil Nadu": {"lat": 11.0168, "long": 76.9558, "pop": 1601000},
    "🇮🇳 Goa (Panaji / Margao)": {"lat": 15.2993, "long": 74.1240, "pop": 400000},
    "🇮🇳 Bhubaneswar, Odisha": {"lat": 20.2961, "long": 85.8245, "pop": 843000},
    "🇮🇳 Guwahati, Assam": {"lat": 26.1445, "long": 91.7362, "pop": 962000},
    "🇮🇳 Dehradun, Uttarakhand": {"lat": 30.3165, "long": 78.0322, "pop": 578000},
    "🇮🇳 Varanasi, Uttar Pradesh": {"lat": 25.3176, "long": 82.9739, "pop": 1200000},
    "🇮🇳 Amritsar, Punjab": {"lat": 31.6340, "long": 74.8723, "pop": 1132000},
    "🇮🇳 Ranchi, Jharkhand": {"lat": 23.3441, "long": 85.3096, "pop": 1073000},
    "🇮🇳 Mysore, Karnataka": {"lat": 12.2958, "long": 76.6394, "pop": 920000},
    "🇮🇳 Nashik, Maharashtra": {"lat": 19.9975, "long": 73.7898, "pop": 1486000},
    "🇮🇳 Raipur, Chhattisgarh": {"lat": 21.2514, "long": 81.6296, "pop": 1010000},
    "🇮🇳 Rajkot, Gujarat": {"lat": 22.3039, "long": 70.8022, "pop": 1390000},
    "🇮🇳 Agra, Uttar Pradesh": {"lat": 27.1767, "long": 78.0081, "pop": 1585000},
    "🇮🇳 Madurai, Tamil Nadu": {"lat": 9.9252, "long": 78.1198, "pop": 1017000},
    "🇮🇳 Jodhpur, Rajasthan": {"lat": 26.2389, "long": 73.0243, "pop": 1033000},
    
    # --- Major Global Cities ---
    "🇺🇸 New York, USA": {"lat": 40.7128, "long": -74.0060, "pop": 8336817},
    "🇺🇸 Los Angeles, USA": {"lat": 34.0522, "long": -118.2437, "pop": 3979576},
    "🇺🇸 Chicago, USA": {"lat": 41.8781, "long": -87.6298, "pop": 2693976},
    "🇬🇧 London, United Kingdom": {"lat": 51.5074, "long": -0.1278, "pop": 8982000},
    "🇦🇪 Dubai, UAE": {"lat": 25.2048, "long": 55.2708, "pop": 3331000},
    "🇸🇬 Singapore": {"lat": 1.3521, "long": 103.8198, "pop": 5686000},
    "🇨🇦 Toronto, Canada": {"lat": 43.6532, "long": -79.3832, "pop": 2930000},
    "🇦🇺 Sydney, Australia": {"lat": -33.8688, "long": 151.2093, "pop": 5312000},
    
    # --- Custom Option ---
    "📍 Custom / Enter Coordinates Manually": {"lat": 19.0760, "long": 72.8777, "pop": 500000}
}

# Merchant Options with Common India & Everyday Stores
POPULAR_STORES = [
    "Amazon Online Store",
    "Flipkart Online",
    "Swiggy / Zomato Food Delivery",
    "Blinkit / Zepto / BigBasket",
    "DMart Supermarket",
    "Reliance Smart / Trends",
    "Petrol Pump (IndianOil / HPCL / BPCL)",
    "MakeMyTrip / IRCTC / IndiGo",
    "Nykaa / Myntra Online",
    "BookMyShow / PVR Inox",
    "Apollo Pharmacy / Netmeds",
    "Tata Neu / Croma Electronics",
    "Local Kirana / Neighborhood Store",
    "Other / Unregistered Merchant"
]

DATASET_MERCHANTS = sorted(list(merchant_lookup.keys()))
ALL_MERCHANT_OPTIONS = POPULAR_STORES + [f"Dataset Registry: {m}" for m in DATASET_MERCHANTS]

# Easy Preset Examples (India & Global friendly)
PRESET_ONLINE_FRAUD = {
    'name': 'Pooja Sharma',
    'gender': 'Female',
    'dob': date(1985, 6, 15),
    'merchant': 'Amazon Online Store',
    'category': 'shopping_net',
    'amt': 78500.00,
    'trans_date': date(2026, 8, 14),
    'trans_hour': 2,
    'city': '🇮🇳 Mumbai, Maharashtra',
    'lat': 19.0760,
    'long': 72.8777,
    'pop': 12500000
}

PRESET_LATE_FRAUD = {
    'name': 'Rahul Verma',
    'gender': 'Male',
    'dob': date(1992, 3, 20),
    'merchant': 'Flipkart Online',
    'category': 'misc_net',
    'amt': 54200.00,
    'trans_date': date(2026, 9, 3),
    'trans_hour': 23,
    'city': '🇮🇳 Delhi NCR (New Delhi)',
    'lat': 28.6139,
    'long': 77.2090,
    'pop': 11000000
}

PRESET_LEGIT_GROCERY = {
    'name': 'Ananya Iyer',
    'gender': 'Female',
    'dob': date(1990, 11, 12),
    'merchant': 'DMart Supermarket',
    'category': 'grocery_pos',
    'amt': 1850.00,
    'trans_date': date(2026, 10, 5),
    'trans_hour': 14,
    'city': '🇮🇳 Bengaluru, Karnataka',
    'lat': 12.9716,
    'long': 77.5946,
    'pop': 8400000
}

PRESET_LEGIT_GAS = {
    'name': 'Amit Patel',
    'gender': 'Male',
    'dob': date(1988, 4, 18),
    'merchant': 'Petrol Pump (IndianOil / HPCL / BPCL)',
    'category': 'gas_transport',
    'amt': 2200.00,
    'trans_date': date(2026, 10, 5),
    'trans_hour': 9,
    'city': '🇮🇳 Ahmedabad, Gujarat',
    'lat': 23.0225,
    'long': 72.5714,
    'pop': 5570000
}


def load_preset(preset: dict):
    st.session_state.usr_name = preset['name']
    st.session_state.usr_gender = preset['gender']
    st.session_state.usr_dob = preset['dob']
    st.session_state.usr_merchant = preset['merchant']
    st.session_state.usr_category = preset['category']
    st.session_state.usr_amt = preset['amt']
    st.session_state.usr_date = preset['trans_date']
    st.session_state.usr_hour = preset['trans_hour']
    st.session_state.usr_city = preset['city']
    st.session_state.usr_lat = preset['lat']
    st.session_state.usr_long = preset['long']
    st.session_state.usr_pop = preset['pop']
    st.rerun()


# =========================================================================
# HEADER
# =========================================================================
st.title("🛡️ CardShield & UPI — Fraud Detection Intelligence")
st.markdown("""
<div class="section-framing">
    Explore machine learning benchmarks across 3 major fraud datasets, test real-time transaction screening, and discover fraud prevention patterns across global credit cards and Indian UPI payments.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Analysis & Model Benchmarks", 
    "🔍 Check a Payment (Live Screener)", 
    "💡 Safety Rules & Fraud Prevention Tips"
])


# =========================================================================
# TAB 1: DATASET ANALYSIS & MODEL BENCHMARKS (ULB, SPARKOV, UPI 2024)
# =========================================================================
with tab1:
    st.markdown("#### Select Benchmark Dataset to Inspect:")
    st.markdown("""
    <div class="section-framing">
        Choose a benchmark dataset below to review exploratory data analysis (EDA), class imbalance distributions, and machine learning model evaluation metrics (Precision, Recall, PR-AUC, ROC-AUC).
    </div>
    """, unsafe_allow_html=True)

    selected_benchmark = st.radio(
        "Select Dataset & Model Benchmark:",
        options=[
            "🇮🇳 Indian UPI Transactions (2024 Dataset — 250k Txns)",
            "💳 Sparkov Credit Card Dataset (1.85M Txns)",
            "🇪🇺 ULB European Credit Card Benchmark (284k Txns)"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------
    # 1. INDIAN UPI TRANSACTIONS (2024 DATASET)
    # -------------------------------------------------------------
    if "Indian UPI" in selected_benchmark:
        st.markdown("### 🇮🇳 Indian UPI Transactions 2024 Dataset")
        st.markdown("""
        <div class="section-framing">
            Analysis of <strong>250,000 real-world Indian UPI transactions</strong> covering 10 states, 8 major banks, P2P/P2M transaction types, and demographic risk factors.
        </div>
        """, unsafe_allow_html=True)

        # KPI Cards with plain-language framing
        u1, u2, u3, u4 = st.columns(4)
        with u1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Total Dataset Records</div>
                <div class="stat-number">250,000</div>
                <div class="stat-desc">Complete 2024 Transaction Log</div>
            </div>
            """, unsafe_allow_html=True)
        with u2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Dataset Fraud Prevalence</div>
                <div class="stat-number" style="color: #DC2626;">480 (0.192%)</div>
                <div class="stat-desc-alert">Ground-Truth Fraud (520:1 Imbalance)</div>
            </div>
            """, unsafe_allow_html=True)
        with u3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Highest Risk States</div>
                <div class="stat-number">KA, RJ, GJ</div>
                <div class="stat-desc">Top State Rates (0.21% - 0.23%)</div>
            </div>
            """, unsafe_allow_html=True)
        with u4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Top Benchmark Model</div>
                <div class="stat-number" style="font-size: 1.35rem;">Balanced XGBoost</div>
                <div class="stat-desc">Scale_Pos_Weight: 519.8</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. EDA Section
        st.subheader("1. 📈 Exploratory Data Analysis & Risk Distributions")
        st.markdown("""
        <div class="section-framing">
            Visual inspection of fraud patterns across geographic states, merchant categories, 24-hour transaction timing, banking entities, and customer age brackets.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### (a) State-Level Fraud Rate and Volume")
        st.caption("Distribution of fraud rate (%) and absolute fraud incident count across Indian states.")
        upi_state_img = os.path.join(ARTIFACTS_DIR, "upi_eda_state.png")
        if os.path.exists(upi_state_img):
            st.image(upi_state_img, use_container_width=True)

        st.markdown("#### (b) Merchant Category & Transaction Type Risk")
        st.caption("Comparison of fraud prevalence across payment categories (Transport, Shopping, Food, Grocery) and transaction types (P2P, P2M, Recharge, Bill Payment).")
        upi_cat_img = os.path.join(ARTIFACTS_DIR, "upi_eda_category.png")
        if os.path.exists(upi_cat_img):
            st.image(upi_cat_img, use_container_width=True)

        st.markdown("#### (c) 24-Hour Hourly Trend & Late-Night Spike Window")
        st.caption("Transaction volume versus fraud rate across the 24-hour cycle, highlighting the heightened risk between 12 AM and 5 AM.")
        upi_hour_img = os.path.join(ARTIFACTS_DIR, "upi_eda_hourly.png")
        if os.path.exists(upi_hour_img):
            st.image(upi_hour_img, use_container_width=True)

        st.markdown("#### (d) Sender Bank, Device Type, and Network Distribution")
        st.caption("Fraud rate breakdown by sending bank (Kotak, ICICI, SBI, HDFC, Axis, etc.), device operating system, and connection network.")
        upi_bank_img = os.path.join(ARTIFACTS_DIR, "upi_eda_bank_device.png")
        if os.path.exists(upi_bank_img):
            st.image(upi_bank_img, use_container_width=True)

        st.markdown("#### (e) Demographics & Senior Citizen (56+) Risk")
        st.caption("Analysis of fraud rates across sender and receiver age brackets, highlighting vulnerabilities among senior citizens (56+).")
        upi_age_img = os.path.join(ARTIFACTS_DIR, "upi_eda_senior_age.png")
        if os.path.exists(upi_age_img):
            st.image(upi_age_img, use_container_width=True)

        st.markdown("---")

        # 2. Model Benchmarks
        st.subheader("2. 🏆 Supervised Model Benchmark Comparison")
        st.markdown("""
        <div class="section-framing">
            Performance comparison of candidate algorithms trained on class-balanced partitions and evaluated on an unseen 20% test partition (50,000 transactions, 96 fraud cases).
        </div>
        """, unsafe_allow_html=True)

        upi_comp_csv = os.path.join(ARTIFACTS_DIR, "upi_model_comparison.csv")
        if os.path.exists(upi_comp_csv):
            comp_df = pd.read_csv(upi_comp_csv)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("#### (a) Confusion Matrices Comparison")
        st.caption("Visual breakdown of True Positives (caught fraud), False Positives (false alarms), False Negatives (missed fraud), and True Negatives.")
        upi_cm_img = os.path.join(ARTIFACTS_DIR, "upi_confusion_matrix.png")
        if os.path.exists(upi_cm_img):
            st.image(upi_cm_img, use_container_width=True)

        st.markdown("#### (b) ROC & Precision-Recall Curves")
        st.caption("Trade-off between detection sensitivity (Recall) and False Positive Rate / Precision across decision thresholds.")
        upi_roc_img = os.path.join(ARTIFACTS_DIR, "upi_roc_curve.png")
        if os.path.exists(upi_roc_img):
            st.image(upi_roc_img, use_container_width=True)

        st.markdown("#### (c) Feature Importance Weights (Top Benchmark Model)")
        st.caption("Relative importance weight of each engineered feature in driving the model's fraud predictions.")
        upi_fi_img = os.path.join(ARTIFACTS_DIR, "upi_feature_importance.png")
        if os.path.exists(upi_fi_img):
            st.image(upi_fi_img, use_container_width=True)

    # -------------------------------------------------------------
    # 2. SPARKOV CREDIT CARD DATASET (1.85M TRANSACTIONS)
    # -------------------------------------------------------------
    elif "Sparkov" in selected_benchmark:
        st.markdown("### 💳 Sparkov Multi-Feature Credit Card Dataset")
        st.markdown("""
        <div class="section-framing">
            Benchmarked on <strong>1,848,503 synthetic card transactions</strong> with geographical coordinates, merchant profiles, timestamps, and customer demographics.
        </div>
        """, unsafe_allow_html=True)

        # KPI Cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Total Dataset Records</div>
                <div class="stat-number">1,848,503</div>
                <div class="stat-desc">1.29M Train | 555k Test Partition</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Dataset Fraud Prevalence</div>
                <div class="stat-number" style="color: #DC2626;">0.52%</div>
                <div class="stat-desc-alert">9,651 Total Labeled Fraud Samples</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Top Benchmark Recall</div>
                <div class="stat-number" style="color: #059669;">94.5% Caught</div>
                <div class="stat-desc">2,027 / 2,145 Test Frauds Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Top Benchmark PR-AUC</div>
                <div class="stat-number">0.8673</div>
                <div class="stat-desc">ROC-AUC: 0.9974 (Balanced XGBoost)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("1. 📈 Exploratory Data Analysis & Behavioral Trends")
        st.markdown("""
        <div class="section-framing">
            Analysis of transaction behavior, identifying high-risk merchant segments, late-night fraud spikes, and geographic variations.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### (a) High-Risk Merchant Categories")
        st.caption("Fraud rate breakdown by merchant category (Online Shopping, Digital Goods, Grocery, Dining, Travel).")
        sp_cat_img = os.path.join(ARTIFACTS_DIR, "sparkov_eda_category.png")
        if os.path.exists(sp_cat_img):
            st.image(sp_cat_img, use_container_width=True)

        st.markdown("#### (b) Hourly Fraud Spikes (Late-Night Window)")
        st.caption("Hourly volume versus fraud rate across the 24-hour cycle showing the late-night risk spike (10 PM – 3 AM).")
        sp_hour_img = os.path.join(ARTIFACTS_DIR, "sparkov_eda_hourly.png")
        if os.path.exists(sp_hour_img):
            st.image(sp_hour_img, use_container_width=True)

        st.markdown("#### (c) Geographic Distribution by State")
        st.caption("Total card transactions and detected fraud incidents by geographic state.")
        sp_state_img = os.path.join(ARTIFACTS_DIR, "sparkov_eda_state.png")
        if os.path.exists(sp_state_img):
            st.image(sp_state_img, use_container_width=True)

        st.markdown("---")

        st.subheader("2. 🏆 Supervised Model Benchmark Comparison")
        st.markdown("""
        <div class="section-framing">
            Performance comparison of Balanced Random Forest, Balanced XGBoost, and Tuned XGBoost evaluated on 555,719 holdout test transactions.
        </div>
        """, unsafe_allow_html=True)

        sp_comp_csv = os.path.join(ARTIFACTS_DIR, "sparkov_model_comparison.csv")
        if os.path.exists(sp_comp_csv):
            sp_df = pd.read_csv(sp_comp_csv)
            st.dataframe(sp_df, use_container_width=True, hide_index=True)

        st.markdown("#### (a) Confusion Matrix Benchmark")
        st.caption("Confusion matrices showing True Positives, False Positives, False Negatives, and True Negatives on the test partition.")
        sp_cm_img = os.path.join(ARTIFACTS_DIR, "sparkov_confusion_matrix.png")
        if os.path.exists(sp_cm_img):
            st.image(sp_cm_img, use_container_width=True)

        st.markdown("#### (b) ROC & Precision-Recall Curves")
        st.caption("Receiver Operating Characteristic and Precision-Recall Curves comparing model discrimination power.")
        sp_roc_img = os.path.join(ARTIFACTS_DIR, "sparkov_roc_curve.png")
        if os.path.exists(sp_roc_img):
            st.image(sp_roc_img, use_container_width=True)

        st.markdown("#### (c) Feature Importance Weights (Balanced XGBoost)")
        st.caption("Feature importance ranking demonstrating the dominance of transaction amount, distance from home, and time features.")
        sp_fi_img = os.path.join(ARTIFACTS_DIR, "sparkov_feature_importance.png")
        if os.path.exists(sp_fi_img):
            st.image(sp_fi_img, use_container_width=True)

    # -------------------------------------------------------------
    # 3. ULB EUROPEAN CREDIT CARD BENCHMARK (284k TRANSACTIONS)
    # -------------------------------------------------------------
    else:
        st.markdown("### 🇪🇺 ULB European Credit Card Benchmark")
        st.markdown("""
        <div class="section-framing">
            Classic academic benchmark dataset containing <strong>284,807 card transactions</strong> with 28 anonymized PCA components (V1-V28), transaction timestamp, and amount.
        </div>
        """, unsafe_allow_html=True)

        # KPI Cards
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Total Dataset Records</div>
                <div class="stat-number">284,807</div>
                <div class="stat-desc">2-Day European Card Log</div>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Dataset Fraud Prevalence</div>
                <div class="stat-number" style="color: #DC2626;">0.172%</div>
                <div class="stat-desc-alert">492 Labeled Fraud Attacks (578:1)</div>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Top Benchmark PR-AUC</div>
                <div class="stat-number" style="color: #059669;">0.7924</div>
                <div class="stat-desc">Random Forest + SMOTE</div>
            </div>
            """, unsafe_allow_html=True)
        with e4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-title">Top Benchmark ROC-AUC</div>
                <div class="stat-number">0.9673</div>
                <div class="stat-desc">Recall: 76.8% | Precision: 74.5%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("1. 📈 Exploratory Data Analysis & PCA Distributions")
        st.markdown("""
        <div class="section-framing">
            Distributional properties of PCA feature components, transaction amount tiers, and correlation matrix with the fraud target.
        </div>
        """, unsafe_allow_html=True)
        
        ulb_class_img = os.path.join(IMAGES_DIR, "01_class_distribution.png")
        if os.path.exists(ulb_class_img):
            st.image(ulb_class_img, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            ulb_amt_img = os.path.join(IMAGES_DIR, "02_fraud_vs_legit_amount.png")
            if os.path.exists(ulb_amt_img):
                st.image(ulb_amt_img, use_container_width=True)
        with c2:
            ulb_cat_amt = os.path.join(IMAGES_DIR, "03_amount_category_fraud_rate.png")
            if os.path.exists(ulb_cat_amt):
                st.image(ulb_cat_amt, use_container_width=True)

        ulb_hour_img = os.path.join(IMAGES_DIR, "04_hourly_fraud_trend.png")
        if os.path.exists(ulb_hour_img):
            st.image(ulb_hour_img, use_container_width=True)

        ulb_corr_img = os.path.join(IMAGES_DIR, "05_correlation_matrix.png")
        if os.path.exists(ulb_corr_img):
            st.image(ulb_corr_img, use_container_width=True)

        st.markdown("---")

        st.subheader("2. 🏆 Supervised Model Benchmark Comparison")
        st.markdown("""
        <div class="section-framing">
            Comparison of Baseline Logistic Regression, SMOTE Logistic Regression, and SMOTE Random Forest evaluated on 56,746 holdout test samples.
        </div>
        """, unsafe_allow_html=True)

        ulb_comp_csv = os.path.join(PROCESSED_DIR, "model_comparison.csv")
        if os.path.exists(ulb_comp_csv):
            ulb_df = pd.read_csv(ulb_comp_csv)
            st.dataframe(ulb_df, use_container_width=True, hide_index=True)

        st.markdown("#### (a) Confusion Matrices Comparison")
        st.caption("Performance on the test partition comparing baseline and SMOTE-resampled classifiers.")
        ulb_cm_img = os.path.join(IMAGES_DIR, "07_confusion_matrices.png")
        if os.path.exists(ulb_cm_img):
            st.image(ulb_cm_img, use_container_width=True)

        st.markdown("#### (b) ROC & Precision-Recall Curves")
        st.caption("Precision-Recall and ROC curves demonstrating the superior PR-AUC of the SMOTE Random Forest classifier.")
        ulb_roc_img = os.path.join(IMAGES_DIR, "06_roc_pr_curves.png")
        if os.path.exists(ulb_roc_img):
            st.image(ulb_roc_img, use_container_width=True)


# =========================================================================
# TAB 2: CHECK A TRANSACTION (LIVE PAYMENT SCREENER)
# =========================================================================
with tab2:
    st.markdown("### ⚡ Live Payment Screener & Quick Scenarios")
    st.markdown("""
    <div class="section-framing">
        Simulate a transaction through the calibrated machine learning pipeline to compute real-time fraud probability. Choose a pre-configured scenario below to auto-populate the form, or enter custom transaction parameters manually.
    </div>
    """, unsafe_allow_html=True)
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        if st.button("🚨 Midnight Order (₹78.5k)", use_container_width=True, help="High-value online purchase at 2 AM in Mumbai"):
            load_preset(PRESET_ONLINE_FRAUD)
    with col_e2:
        if st.button("🚨 Late Night Order (₹54.2k)", use_container_width=True, help="Late-night electronics purchase at 11 PM in Delhi"):
            load_preset(PRESET_LATE_FRAUD)
    with col_e3:
        if st.button("✅ DMart Grocery (₹1.8k)", use_container_width=True, help="Routine grocery store purchase at 2 PM in Bengaluru"):
            load_preset(PRESET_LEGIT_GROCERY)
    with col_e4:
        if st.button("✅ Petrol Pump (₹2.2k)", use_container_width=True, help="Morning fuel refill at 9 AM in Ahmedabad"):
            load_preset(PRESET_LEGIT_GAS)

    st.markdown("---")
    
    # State initialization
    if "usr_name" not in st.session_state:
        st.session_state.usr_name = "Pooja Sharma"
    if "usr_gender" not in st.session_state:
        st.session_state.usr_gender = "Female"
    if "usr_dob" not in st.session_state:
        st.session_state.usr_dob = date(1985, 6, 15)
    if "usr_merchant" not in st.session_state:
        st.session_state.usr_merchant = "Amazon Online Store"
    if "usr_category" not in st.session_state:
        st.session_state.usr_category = "shopping_net"
    if "usr_amt" not in st.session_state:
        st.session_state.usr_amt = 78500.00
    if "usr_date" not in st.session_state:
        st.session_state.usr_date = date.today()
    if "usr_hour" not in st.session_state:
        st.session_state.usr_hour = 2
    if "usr_city" not in st.session_state:
        st.session_state.usr_city = "🇮🇳 Mumbai, Maharashtra"
    if "usr_lat" not in st.session_state:
        st.session_state.usr_lat = 19.0760
    if "usr_long" not in st.session_state:
        st.session_state.usr_long = 72.8777
    if "usr_pop" not in st.session_state:
        st.session_state.usr_pop = 12500000

    with st.form("simple_screening_form"):
        st.subheader("1. Payment & Merchant Profile")
        st.markdown("""
        <div class="section-framing">
            Specify the transaction amount, customer name, and merchant category.
        </div>
        """, unsafe_allow_html=True)

        r1_c1, r1_c2 = st.columns(2)
        
        with r1_c1:
            cardholder_name = st.text_input("Cardholder Full Name", value=st.session_state.usr_name, help="Name of cardholder for notification dispatch")
            amount = st.number_input("Transaction Amount (₹ / $)", value=float(st.session_state.usr_amt), min_value=1.0, step=100.0, format="%.2f", help="Nominal purchase amount")
            
        with r1_c2:
            # Merchant - free text input with suggestions shown as helper
            popular_suggestions = ", ".join(POPULAR_STORES[:6])
            selected_merchant = st.text_input(
                "Store / Merchant Name",
                value=st.session_state.usr_merchant,
                help=f"Type any merchant name — known or unknown. Suggestions: {popular_suggestions}. Unregistered merchants automatically resolve to category statistical median distance.",
                placeholder="e.g. Amazon, Local Kirana, Any Store Name…"
            )
            if not selected_merchant.strip():
                selected_merchant = "Other / Unregistered Merchant"

            # Category with plain labels
            cat_keys = list(CATEGORY_LABELS.keys())
            cur_cat = st.session_state.usr_category if st.session_state.usr_category in cat_keys else cat_keys[0]
            cat_idx = cat_keys.index(cur_cat)
            
            selected_category_key = st.selectbox(
                "Transaction Category",
                options=cat_keys,
                format_func=lambda k: CATEGORY_LABELS[k],
                index=cat_idx,
                help="Merchant business classification"
            )

        st.subheader("2. Cardholder Location & Purchase Timing")
        st.markdown("""
        <div class="section-framing">
            Location and timestamp are used to calculate geographical distance from home and nocturnal risk windows.
        </div>
        """, unsafe_allow_html=True)

        r2_c1, r2_c2, r2_c3 = st.columns(3)

        with r2_c1:
            # Build a short suggestion hint (first 8 Indian cities)
            city_names = list(ALL_LOCATIONS.keys())
            city_suggestions_hint = ", ".join(
                [c.split("🇮🇳 ")[-1].split(",")[0] for c in city_names if "🇮🇳" in c][:8]
            )

            typed_city = st.text_input(
                "Cardholder Home Location",
                value=st.session_state.usr_city.replace("📍 Custom / Enter Coordinates Manually", "").strip(),
                help=f"Type any city or location. Suggestions: {city_suggestions_hint}, etc. If not found in our list, set coordinates below.",
                placeholder="e.g. Mumbai, Delhi, Pune, Singapore…"
            )

            # Try to find an exact or partial match in ALL_LOCATIONS
            matched_city_key = None
            if typed_city.strip():
                typed_lower = typed_city.strip().lower()
                # 1. Exact match
                for k in city_names:
                    if typed_lower == k.lower() or typed_lower in k.lower():
                        matched_city_key = k
                        break

            if matched_city_key:
                home_lat = ALL_LOCATIONS[matched_city_key]["lat"]
                home_long = ALL_LOCATIONS[matched_city_key]["long"]
                home_pop = ALL_LOCATIONS[matched_city_key]["pop"]
                st.caption(f"📍 Matched: {matched_city_key.split(' ', 1)[-1]} — Lat {home_lat:.4f}, Long {home_long:.4f}")
            else:
                # Unknown city — fall back to Mumbai and let user adjust
                st.caption("📍 Location not in our database. Using default coordinates below — adjust if needed.")
                home_lat = float(st.session_state.usr_lat)
                home_long = float(st.session_state.usr_long)
                home_pop = int(st.session_state.usr_pop)

            # Always show lat/long overrides for full flexibility
            with st.expander("🗺️ Override Coordinates (optional)", expanded=(matched_city_key is None)):
                c_lat_col, c_lon_col = st.columns(2)
                with c_lat_col:
                    home_lat = st.number_input("Latitude", value=home_lat, format="%.4f", key="override_lat")
                with c_lon_col:
                    home_long = st.number_input("Longitude", value=home_long, format="%.4f", key="override_long")
                home_pop = st.number_input("City Population (approx.)", value=home_pop, min_value=1000, step=100000, key="override_pop")

        with r2_c2:
            purchase_date = st.date_input("Date of Purchase", value=st.session_state.usr_date, help="Date when transaction was initiated")
            
        with r2_c3:
            # Friendly Time of Day Slider
            purchase_hour = st.slider("Time of Purchase (Hour: 00 to 23)", min_value=0, max_value=23, value=st.session_state.usr_hour, help="0 = Midnight, 12 = Noon, 23 = 11 PM")
            if 0 <= purchase_hour <= 4 or purchase_hour == 23:
                st.caption("🌙 Late Night Hours (Elevated Risk Window)")
            elif 5 <= purchase_hour <= 11:
                st.caption("☀️ Morning (Standard Business Window)")
            elif 12 <= purchase_hour <= 17:
                st.caption("🌤️ Afternoon (Standard Business Window)")
            else:
                st.caption("🌆 Evening (Active Spending Window)")

        # Simplified Cardholder Details
        with st.expander("👤 Optional: Cardholder Demographics (Date of Birth & Gender)", expanded=False):
            e_c1, e_c2 = st.columns(2)
            with e_c1:
                gender_choice = st.selectbox("Gender", ["Female", "Male"], index=0 if st.session_state.usr_gender == "Female" else 1)
            with e_c2:
                dob_val = st.date_input("Date of Birth", value=st.session_state.usr_dob, help="Used for exact date-aware age calculation")

        # Merchant Coordinate Resolution
        clean_merch_name = selected_merchant.replace("Dataset Registry: ", "").strip()
        is_known = (clean_merch_name in merchant_lookup)
        if is_known:
            m_lat = merchant_lookup[clean_merch_name]['merch_lat']
            m_long = merchant_lookup[clean_merch_name]['merch_long']
            distance_km = haversine_distance(home_lat, home_long, m_lat, m_long)
            dist_override = None
        else:
            m_lat = home_lat + 0.02
            m_long = home_long + 0.02
            if selected_category_key in ['grocery_pos', 'gas_transport', 'food_dining', 'personal_care']:
                local_dist = 6.5
            else:
                local_dist = float(dist_fallback_stats['category_median_distance'].get(selected_category_key, 78.0))
            distance_km = local_dist
            dist_override = float(local_dist)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🛡️ Screen Payment with AI Model", type="primary", use_container_width=True)

    # ------------------ PREDICTION & SIMPLE EXPLANATION ------------------
    if submit:
        if amount <= 0:
            st.error("Please enter a valid payment amount greater than 0.")
        else:
            try:
                # Convert amount to benchmark equivalent scale for model (approx 1 USD ~ 80 INR for standard scale)
                is_inr = ("🇮🇳" in typed_city) or (amount > 1000)
                norm_amt = (amount / 80.0) if is_inr else amount
                
                date_time_str = f"{purchase_date.strftime('%Y-%m-%d')} {purchase_hour:02d}:00:00"
                raw_dict = {
                    'trans_date_trans_time': date_time_str,
                    'category': selected_category_key,
                    'amt': float(norm_amt),
                    'gender': 'F' if gender_choice == "Female" else 'M',
                    'lat': float(home_lat),
                    'long': float(home_long),
                    'city_pop': int(home_pop),
                    'dob': dob_val.strftime('%Y-%m-%d'),
                    'merch_lat': float(m_lat),
                    'merch_long': float(m_long)
                }
                
                features_df = transform_features(raw_dict, encoders, distance_override=dist_override)
                scaled = scaler.transform(features_df)
                
                probs = model.predict_proba(scaled)[0]
                fraud_chance = float(probs[1]) * 100
                safe_chance = float(probs[0]) * 100
                is_fraud = (fraud_chance >= 50.0)
                
                cardholder_age = calculate_age_date_aware(purchase_date, dob_val)
                currency_symbol = "₹" if is_inr else "$"
                
            except Exception as e:
                st.error("We couldn't check this transaction. Please verify that all fields are filled properly.")
                st.stop()

            st.markdown("---")
            st.subheader("Results & Action Recommendation")
            st.markdown("""
            <div class="section-framing">
                Model assessment produced by our ensemble classifier combining transaction amount, merchant category, nocturnal timing, and geographic displacement features.
            </div>
            """, unsafe_allow_html=True)

            # Simple Verdict Banner
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-box">
                    <h2 style="color: #DC2626; margin: 0;">🚨 High Risk Fraud Flag Detected</h2>
                    <p style="font-size: 1.15rem; color: #7F1D1D; margin-top: 8px;">
                        The model identified anomalous spending signals with a <strong>{fraud_chance:.1f}% estimated fraud probability</strong>.
                    </p>
                    <p style="font-size: 1rem; color: #991B1B; margin-bottom: 0;">
                        <strong>👉 Recommended Action:</strong> <strong>Place payment on temporary hold.</strong> Send an instant verification SMS / WhatsApp prompt to <strong>{cardholder_name}</strong> to confirm payment of <strong>{currency_symbol}{amount:,.2f}</strong>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-box">
                    <h2 style="color: #16A34A; margin: 0;">✅ Legitimate Transaction Verified</h2>
                    <p style="font-size: 1.15rem; color: #14532D; margin-top: 8px;">
                        This payment matches typical, safe everyday spending habits (<strong>{safe_chance:.1f}% confidence score</strong>).
                    </p>
                    <p style="font-size: 1rem; color: #166534; margin-bottom: 0;">
                        <strong>👉 Recommended Action:</strong> <strong>Approve payment of {currency_symbol}{amount:,.2f}</strong>. No additional security escalation required.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Key Summary Cards (Plain English)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Estimated Fraud Risk</div>
                    <div class="stat-number" style="color: {'#DC2626' if is_fraud else '#16A34A'};">
                        {fraud_chance:.1f}%
                    </div>
                    <div class="{'stat-desc-alert' if is_fraud else 'stat-desc'}">
                        {'⚠️ High Fraud Risk' if is_fraud else '✔️ Safe & Verified'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with k2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Merchant Proximity</div>
                    <div class="stat-number" style="font-size: 1.25rem;">{typed_city.split(',')[0] if typed_city else 'Unknown'}</div>
                    <div class="stat-desc">{'Local in-city store' if distance_km < 30 else f'~{distance_km:.0f} km away'}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with k3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Transaction Hour</div>
                    <div class="stat-number">{purchase_hour:02d}:00</div>
                    <div class="{'stat-desc-alert' if (0 <= purchase_hour <= 4 or purchase_hour == 23) else 'stat-desc'}">
                        {'🌙 Late Night Hours' if (0 <= purchase_hour <= 4 or purchase_hour == 23) else '☀️ Daytime Hours'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with k4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Cardholder Age</div>
                    <div class="stat-number">{cardholder_age} yrs</div>
                    <div class="stat-desc">Date-Aware Calculated</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------ SIMPLE EXPLAINABILITY SECTION ------------------
            st.subheader("💡 Key Decision Drivers & Explainability")
            st.markdown("""
            <div class="section-framing">
                Plain-language breakdown of the main feature contributions driving the classifier's risk score:
            </div>
            """, unsafe_allow_html=True)

            col_why1, col_why2 = st.columns(2)

            with col_why1:
                st.markdown("#### 🔍 Feature Risk Breakdown")
                reasons = []

                if norm_amt > 400:
                    reasons.append(f"🔴 **High Amount:** {currency_symbol}{amount:,.2f} is significantly higher than normal everyday spending baseline.")
                elif norm_amt < 50:
                    reasons.append(f"🟢 **Routine Amount:** {currency_symbol}{amount:,.2f} falls within typical everyday transaction range.")

                if (0 <= purchase_hour <= 4) or purchase_hour == 23:
                    reasons.append(f"🔴 **Nocturnal Transaction Window ({purchase_hour:02d}:00):** Transactions initiated late at night carry higher risk of unauthorized account access.")
                else:
                    reasons.append(f"🟢 **Standard Hours ({purchase_hour:02d}:00):** Transaction occurred during active daytime business hours.")

                if distance_km > 60:
                    reasons.append(f"🔴 **Geographical Displacement:** Store is ~{distance_km:.0f} km away from cardholder's home address.")
                else:
                    reasons.append(f"🟢 **Local Merchant Proximity:** Transaction occurred within cardholder's home vicinity (~{distance_km:.0f} km).")

                if selected_category_key in ['shopping_net', 'misc_net']:
                    reasons.append("🔴 **Online / Digital Merchant:** Card-not-present online purchases exhibit higher baseline fraud prevalence.")
                elif selected_category_key in ['grocery_pos', 'gas_transport']:
                    reasons.append("🟢 **Physical POS Routine:** Supermarkets and petrol stations represent standard recurring in-person spend.")

                for r in reasons:
                    st.markdown(f"<div class='reason-item'>{r}</div>", unsafe_allow_html=True)

            with col_why2:
                st.markdown("#### 🛡️ Operational Security Protocol")
                if is_fraud:
                    st.markdown(f"""
                    <div class='reason-item'>
                        <strong>1. Hold Transaction:</strong> Temporarily pause payment settlement pending authorization.
                    </div>
                    <div class='reason-item'>
                        <strong>2. Two-Factor Verification:</strong> Send real-time OTP / SMS prompt: <em>"Did you authorize {currency_symbol}{amount:,.2f} at {selected_merchant.replace('Dataset Registry: ', '')}?"</em>
                    </div>
                    <div class='reason-item'>
                        <strong>3. Rapid Freeze:</strong> If rejected by cardholder, immediately lock card credentials to halt additional fraudulent charges.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='reason-item'>
                        <strong>1. Auto-Approve:</strong> Clean risk score; release payment of {currency_symbol}{amount:,.2f}.
                    </div>
                    <div class='reason-item'>
                        <strong>2. Frictionless Processing:</strong> No additional manual review or cardholder interruptions needed.
                    </div>
                    <div class='reason-item'>
                        <strong>3. Ledger Update:</strong> Confirm order and generate transaction receipt.
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================================
# TAB 3: SAFETY & FRAUD PREVENTION TIPS
# =========================================================================
with tab3:
    st.header("📊 Real-World Fraud Insights & Account Protection")
    st.markdown("""
    <div class="section-framing">
        Practical security rules and empirical insights derived from analyzing over <strong>2.35 million card and UPI payment records</strong> across India and globally.
    </div>
    """, unsafe_allow_html=True)

    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Baseline Fraud Rate</div>
            <div class="stat-number">0.2% – 0.5%</div>
            <div class="stat-desc">2 to 5 out of every 1,000 transactions are fraud.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_f2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Detection Sensitivity</div>
            <div class="stat-number">Up to 95%</div>
            <div class="stat-desc">Calibrated ML models detect 95 out of 100 attacks.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_f3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Peak Risk Hours</div>
            <div class="stat-number">11 PM – 4 AM</div>
            <div class="stat-desc-alert">Late-night transactions carry 3x-4x higher risk.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("💡 4 Essential Security Rules for UPI & Cardholders")
    st.markdown("""
    <div class="section-framing">
        Recommended defensive measures to protect payment credentials from unauthorized exploitation.
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div class="reason-item">
            <h4>📲 1. Enable Instant Transaction Alerts</h4>
            <p style="color: #64748B;">Turn on instant SMS and WhatsApp banking alerts for any transaction above ₹100 / $1 to immediately detect unauthorized account access.</p>
        </div>
        <div class="reason-item">
            <h4>🔒 2. Toggle International & Online Spending Off When Idle</h4>
            <p style="color: #64748B;">Use your banking application (SBI, HDFC, ICICI, Axis, Kotak) to disable international usage and set strict daily spending caps on debit/credit cards and UPI.</p>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="reason-item">
            <h4>🛒 3. Never Share OTPs, CVVs, or UPI PINs</h4>
            <p style="color: #64748B;">No legitimate bank official or payment support agent will ever request your UPI PIN or 3-digit CVV. UPI PINs are only required to <strong>SEND</strong> funds, never to <strong>RECEIVE</strong> money.</p>
        </div>
        <div class="reason-item">
            <h4>⚠️ 4. Avoid Unverified QR Codes & Screen-Sharing Apps</h4>
            <p style="color: #64748B;">Never scan QR codes sent by unknown buyers or install remote desktop utilities (AnyDesk, TeamViewer) at the request of callers claiming to represent bank support.</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================================
# FRIENDLY FLOATING ASSISTANT
# =========================================================================
st.markdown("""
    <style>
    div[data-testid="stElementContainer"]:has(div[data-testid="stPopover"]) {
        position: fixed !important;
        bottom: 25px !important;
        right: 25px !important;
        width: auto !important;
        z-index: 9999 !important;
    }
    div[data-testid="stPopover"] {
        position: static !important;
    }
    div[data-testid="stPopoverBody"] {
        width: 360px !important;
        max-width: 90vw !important;
        padding: 1.2rem !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.15) !important;
        right: 0 !important;
        left: auto !important;
    }
    div[data-testid="stPopover"] > button {
        border-radius: 50% !important;
        width: 60px !important;
        height: 60px !important;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0px 6px 16px rgba(37,99,235,0.3) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.2s !important;
    }
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.08) !important;
    }
    div[data-testid="stPopover"] > button p {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 28px !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.popover("💬"):
    st.markdown("##### 💬 Fraud Safety Helper")
    st.caption("Ask any question in simple everyday words!")
    
    if "friendly_chat" not in st.session_state:
        st.session_state.friendly_chat = []
        
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.friendly_chat:
            if msg["role"] == "user":
                st.markdown(f"**👤 You:** {msg['content']}")
            else:
                st.info(f"**🛡️ Helper:** {msg['content']}")
            
    with st.form("friendly_chat_form", clear_on_submit=True):
        prompt = st.text_input("Type your question...", label_visibility="collapsed", placeholder="e.g. Is UPI or credit card safer?")
        send = st.form_submit_button("Ask")
        
    if send and prompt:
        st.session_state.friendly_chat.append({"role": "user", "content": prompt})
        
        q = prompt.lower()
        if any(w in q for w in ["upi", "pin", "qr", "gpay", "phonepe", "paytm"]):
            resp = "For UPI safety: You only enter your UPI PIN to SEND money, never to RECEIVE money! Never scan unknown QR codes from strangers."
        elif any(w in q for w in ["india", "city", "mumbai", "delhi", "bengaluru", "pune"]):
            resp = "The app supports over 35+ Indian cities across Maharashtra, Karnataka, Delhi NCR, Gujarat, Tamil Nadu, Kerala, UP, and more!"
        elif any(w in q for w in ["why", "distance", "far", "away", "location"]):
            resp = "If a card is used far away from where the cardholder lives, it usually indicates the card was stolen or cloned."
        elif any(w in q for w in ["night", "time", "hour", "when"]):
            resp = "Fraud attacks spike late at night (11 PM to 4 AM) because victims are asleep and won't notice instant bank text alerts."
        elif any(w in q for w in ["safe", "how", "protect", "tips", "otp", "cvv"]):
            resp = "Never share your OTP or CVV with anyone, enable SMS transaction alerts, and disable international usage in your banking app when not traveling."
        else:
            resp = "I can help explain why payments get flagged, how locations and time affect fraud, or how to keep your card and UPI accounts safe!"
              
        st.session_state.friendly_chat.append({"role": "assistant", "content": resp})
        st.rerun()
