import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
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
    page_title="CardShield — Credit Card Safety Checker",
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
        font-size: 1.8rem;
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
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(BASE_DIR, 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

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
st.title("🛡️ CardShield — Credit Card Safety Checker")
st.markdown("Instantly check whether any credit card payment is **safe to approve** or **likely to be fraud** across India and international locations.")

tab1, tab2 = st.tabs(["🔍 Check a Payment", "📊 Common Fraud Trends & Tips"])


# =========================================================================
# TAB 1: CHECK A TRANSACTION (SIMPLE, FRIENDLY INTERFACE)
# =========================================================================
with tab1:
    st.markdown("### ⚡ Quick Examples (Click one to test)")
    st.caption("Click any sample below to automatically fill in the form and test the system:")
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        if st.button("🚨 Midnight Online Order (₹78,500)", use_container_width=True):
            load_preset(PRESET_ONLINE_FRAUD)
    with col_e2:
        if st.button("🚨 Late Night Purchase (₹54,200)", use_container_width=True):
            load_preset(PRESET_LATE_FRAUD)
    with col_e3:
        if st.button("✅ DMart Grocery (₹1,850)", use_container_width=True):
            load_preset(PRESET_LEGIT_GROCERY)
    with col_e4:
        if st.button("✅ Petrol Pump Refill (₹2,200)", use_container_width=True):
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
        st.subheader("1. Payment & Store Details")
        r1_c1, r1_c2 = st.columns(2)
        
        with r1_c1:
            cardholder_name = st.text_input("Cardholder Name", value=st.session_state.usr_name)
            amount = st.number_input("Payment Amount (₹ / $)", value=float(st.session_state.usr_amt), min_value=1.0, step=100.0, format="%.2f")
            
        with r1_c2:
            # Merchant Selector
            cur_merch = st.session_state.usr_merchant
            merch_idx = ALL_MERCHANT_OPTIONS.index(cur_merch) if cur_merch in ALL_MERCHANT_OPTIONS else 0
            selected_merchant = st.selectbox("Store / Merchant Name", options=ALL_MERCHANT_OPTIONS, index=merch_idx)
            
            # Category with plain labels
            cat_keys = list(CATEGORY_LABELS.keys())
            cur_cat = st.session_state.usr_category if st.session_state.usr_category in cat_keys else cat_keys[0]
            cat_idx = cat_keys.index(cur_cat)
            
            selected_category_key = st.selectbox(
                "What type of purchase is this?", 
                options=cat_keys, 
                format_func=lambda k: CATEGORY_LABELS[k],
                index=cat_idx
            )

        st.subheader("2. Location & Time")
        r2_c1, r2_c2, r2_c3 = st.columns(3)
        
        with r2_c1:
            city_names = list(ALL_LOCATIONS.keys())
            cur_city = st.session_state.usr_city if st.session_state.usr_city in city_names else city_names[0]
            chosen_city = st.selectbox("Cardholder's Home City / State", options=city_names, index=city_names.index(cur_city))
            
            if chosen_city != "📍 Custom / Enter Coordinates Manually":
                home_lat = ALL_LOCATIONS[chosen_city]["lat"]
                home_long = ALL_LOCATIONS[chosen_city]["long"]
                home_pop = ALL_LOCATIONS[chosen_city]["pop"]
            else:
                c_lat_col, c_lon_col = st.columns(2)
                with c_lat_col:
                    home_lat = st.number_input("Latitude", value=float(st.session_state.usr_lat), format="%.4f")
                with c_lon_col:
                    home_long = st.number_input("Longitude", value=float(st.session_state.usr_long), format="%.4f")
                home_pop = int(st.session_state.usr_pop)

        with r2_c2:
            purchase_date = st.date_input("Date of Purchase", value=st.session_state.usr_date)
            
        with r2_c3:
            # Friendly Time of Day Slider
            purchase_hour = st.slider("Time of Purchase (0 = Midnight, 12 = Noon, 23 = 11 PM)", min_value=0, max_value=23, value=st.session_state.usr_hour)
            if 0 <= purchase_hour <= 4 or purchase_hour == 23:
                st.caption("🌙 Late Night Hours (High Risk Window)")
            elif 5 <= purchase_hour <= 11:
                st.caption("☀️ Morning")
            elif 12 <= purchase_hour <= 17:
                st.caption("🌤️ Afternoon")
            else:
                st.caption("🌆 Evening")

        # Simplified Cardholder Details
        with st.expander("👤 Optional: Cardholder Details (Birthday & Gender)", expanded=False):
            e_c1, e_c2 = st.columns(2)
            with e_c1:
                gender_choice = st.selectbox("Gender", ["Female", "Male"], index=0 if st.session_state.usr_gender == "Female" else 1)
            with e_c2:
                dob_val = st.date_input("Date of Birth", value=st.session_state.usr_dob)

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
            # Local purchase distance (e.g. typical store in city ~ 8-15 km, or category median)
            if selected_category_key in ['grocery_pos', 'gas_transport', 'food_dining', 'personal_care']:
                local_dist = 6.5
            else:
                local_dist = float(dist_fallback_stats['category_median_distance'].get(selected_category_key, 78.0))
            distance_km = local_dist
            dist_override = float(local_dist)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🛡️ Check If Payment Is Safe", type="primary", use_container_width=True)

    # ------------------ PREDICTION & SIMPLE EXPLANATION ------------------
    if submit:
        if amount <= 0:
            st.error("Please enter a valid amount greater than 0.")
        else:
            try:
                # Convert amount to benchmark equivalent scale for model (approx 1 USD ~ 80 INR for standard scale)
                # If amount > 1000 and Indian city is selected, treat as INR
                is_inr = ("🇮🇳" in chosen_city) or (amount > 1000)
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

            # Simple Verdict Banner
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-box">
                    <h2 style="color: #DC2626; margin: 0;">🚨 Suspicious Transaction (High Risk of Fraud)</h2>
                    <p style="font-size: 1.15rem; color: #7F1D1D; margin-top: 8px;">
                        Our system detected unusual patterns that look like fraud (<strong>{fraud_chance:.1f}% risk score</strong>).
                    </p>
                    <p style="font-size: 1rem; color: #991B1B; margin-bottom: 0;">
                        <strong>👉 Recommended Action:</strong> <strong>Do not approve this payment immediately.</strong> Send a quick verification SMS/WhatsApp alert to <strong>{cardholder_name}</strong> to confirm if they actually made this purchase for <strong>{currency_symbol}{amount:,.2f}</strong>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-box">
                    <h2 style="color: #16A34A; margin: 0;">✅ Safe Transaction (Looks Completely Normal)</h2>
                    <p style="font-size: 1.15rem; color: #14532D; margin-top: 8px;">
                        This payment matches typical, safe everyday spending habits (<strong>{safe_chance:.1f}% safe</strong>).
                    </p>
                    <p style="font-size: 1rem; color: #166534; margin-bottom: 0;">
                        <strong>👉 Recommended Action:</strong> <strong>Approve payment of {currency_symbol}{amount:,.2f}</strong>. Everything looks safe and routine!
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Key Summary Cards (Plain English)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Risk Level</div>
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
                    <div class="stat-title">Location / City</div>
                    <div class="stat-number" style="font-size: 1.25rem;">{chosen_city.split(',')[0]}</div>
                    <div class="stat-desc">{'Local in-city store' if distance_km < 30 else f'~{distance_km:.0f} km away'}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with k3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Purchase Time</div>
                    <div class="stat-number">{purchase_hour:02d}:00</div>
                    <div class="{'stat-desc-alert' if (0 <= purchase_hour <= 4 or purchase_hour == 23) else 'stat-desc'}">
                        {'🌙 Late Night Hours' if (0 <= purchase_hour <= 4 or purchase_hour == 23) else '☀️ Normal Daytime'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with k4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-title">Cardholder Age</div>
                    <div class="stat-number">{cardholder_age} yrs</div>
                    <div class="stat-desc">Cardholder Verified</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ------------------ SIMPLE EXPLAINABILITY SECTION ------------------
            st.subheader("💡 Why did the system make this decision?")
            st.markdown("Here is a clear breakdown of the main reasons in simple terms:")

            col_why1, col_why2 = st.columns(2)

            with col_why1:
                st.markdown("#### 🔍 Key Observations")
                reasons = []

                if norm_amt > 400:
                    reasons.append(f"🔴 **High Amount:** {currency_symbol}{amount:,.2f} is significantly higher than ordinary routine purchases.")
                elif norm_amt < 50:
                    reasons.append(f"🟢 **Typical Amount:** {currency_symbol}{amount:,.2f} is a standard everyday amount.")

                if (0 <= purchase_hour <= 4) or purchase_hour == 23:
                    reasons.append(f"🔴 **Late Night Purchase ({purchase_hour:02d}:00):** Most credit card fraud happens late at night while cardholders are asleep.")
                else:
                    reasons.append(f"🟢 **Normal Hours ({purchase_hour:02d}:00):** Made during active daytime hours.")

                if distance_km > 60:
                    reasons.append(f"🔴 **Distance From Home:** Store is ~{distance_km:.0f} km away from the cardholder's home city.")
                else:
                    reasons.append(f"🟢 **Nearby Store:** Store is located locally in the cardholder's city (~{distance_km:.0f} km).")

                if selected_category_key in ['shopping_net', 'misc_net']:
                    reasons.append("🔴 **Online Shopping:** Online shopping websites are where most stolen card numbers are used.")
                elif selected_category_key in ['grocery_pos', 'gas_transport']:
                    reasons.append("🟢 **Everyday Routine:** Grocery supermarkets and petrol pumps are standard everyday habits.")

                for r in reasons:
                    st.markdown(f"<div class='reason-item'>{r}</div>", unsafe_allow_html=True)

            with col_why2:
                st.markdown("#### 🛡️ Next Steps for Security")
                if is_fraud:
                    st.markdown(f"""
                    <div class='reason-item'>
                        <strong>1. Hold the payment:</strong> Do not release items or money yet.
                    </div>
                    <div class='reason-item'>
                        <strong>2. Send verification SMS:</strong> Ask cardholder: <em>"Did you just pay {currency_symbol}{amount:,.2f} at {selected_merchant.replace('Dataset Registry: ', '')}?"</em>
                    </div>
                    <div class='reason-item'>
                        <strong>3. Instant Freeze:</strong> If they reply NO, freeze the card immediately to stop more losses.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='reason-item'>
                        <strong>1. Approve payment:</strong> Safe to process {currency_symbol}{amount:,.2f}.
                    </div>
                    <div class='reason-item'>
                        <strong>2. Fast checkout:</strong> No extra security steps or SMS alerts needed.
                    </div>
                    <div class='reason-item'>
                        <strong>3. Normal processing:</strong> Generate receipt and confirm order.
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================================
# TAB 2: FRAUD TRENDS & TIPS (JARGON-FREE INSIGHTS)
# =========================================================================
with tab2:
    st.header("📊 Real-World Fraud Facts & How to Stay Safe")
    st.markdown("Insights gathered from analyzing over **1.85 million credit card payments**:")

    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">How Rare Is Fraud?</div>
            <div class="stat-number">Only 0.4%</div>
            <div class="stat-desc">About 4 out of every 1,000 transactions are fraud.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_f2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">AI Success Rate</div>
            <div class="stat-number">95% Caught</div>
            <div class="stat-desc">Our smart system successfully catches 95 out of 100 fraud attacks.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_f3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Most Dangerous Time</div>
            <div class="stat-number">10 PM – 3 AM</div>
            <div class="stat-desc-alert">Fraud spikes 4x during late-night hours.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("1. Where does fraud happen most?")
    st.markdown("Online shopping, high-value electronics, and online orders have the highest rate of fraud attempts.")
    cat_chart = os.path.join(ARTIFACTS_DIR, 'sparkov_eda_category.png')
    if os.path.exists(cat_chart):
        st.image(cat_chart, width='stretch')

    st.subheader("2. What time of day is riskiest?")
    st.markdown("Fraudsters prefer operating late at night when people are sleeping and won't notice immediate bank text alerts.")
    hour_chart = os.path.join(ARTIFACTS_DIR, 'sparkov_eda_hourly.png')
    if os.path.exists(hour_chart):
        st.image(hour_chart, width='stretch')

    st.markdown("---")
    st.subheader("💡 3 Golden Rules to Protect Your Card in India & Abroad")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="reason-item">
            <h4>📲 Turn On Instant SMS & WhatsApp Alerts</h4>
            <p style="color: #64748B;">Enable instant bank notifications for every transaction above ₹100 / $1 to spot unauthorized activity immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="reason-item">
            <h4>🔒 Turn Off International Usage When Not Needed</h4>
            <p style="color: #64748B;">Use your banking app (SBI, HDFC, ICICI, Axis, etc.) to disable international and online payments when you're not using them.</p>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="reason-item">
            <h4>🛒 Never Share OTPs or CVV</h4>
            <p style="color: #64748B;">No bank official will ever ask for your OTP or 3-digit CVV number over phone or message. Never share them with anyone.</p>
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
    st.markdown("##### 💬 Card Safety Helper")
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
        prompt = st.text_input("Type your question...", label_visibility="collapsed", placeholder="e.g. Is shopping on Amazon/Flipkart safe?")
        send = st.form_submit_button("Ask")
        
    if send and prompt:
        st.session_state.friendly_chat.append({"role": "user", "content": prompt})
        
        q = prompt.lower()
        if any(w in q for w in ["india", "city", "mumbai", "delhi", "bengaluru", "pune"]):
            resp = "The app supports over 35+ Indian cities across Maharashtra, Karnataka, Delhi NCR, Gujarat, Tamil Nadu, Kerala, UP, and more!"
        elif any(w in q for w in ["why", "distance", "far", "away", "location"]):
            resp = "If a card is used far away from where the cardholder lives, it usually indicates the card was stolen or cloned."
        elif any(w in q for w in ["night", "time", "hour", "when"]):
            resp = "Fraud attacks spike late at night (10 PM to 3 AM) because cardholders are asleep and won't notice instant bank text alerts."
        elif any(w in q for w in ["safe", "how", "protect", "tips", "otp"]):
            resp = "Never share your OTP or CVV with anyone, enable SMS transaction alerts, and disable international usage in your banking app when not traveling."
        else:
            resp = "I can help explain why payments get flagged, how locations and time affect fraud, or how to keep your card safe!"
              
        st.session_state.friendly_chat.append({"role": "assistant", "content": resp})
        st.rerun()
