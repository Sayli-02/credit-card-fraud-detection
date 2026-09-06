import streamlit as st
import pandas as pd
import joblib
import json
import os
import re
import difflib

# Setup page config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Custom CSS for Premium Light UI
st.markdown("""
<style>
    /* Remove default Streamlit top padding to eliminate blank space */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Global Font and Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Elegant headers */
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    /* Soft shadow cards for tabs and containers */
    div.stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    div.stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        border-bottom: 3px solid transparent;
        font-weight: 600;
    }
    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #2563EB;
        color: #2563EB;
    }

    /* St.info, st.success, st.error, st.warning styling with glass/premium look */
    div.stAlert {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    }
    
    /* Styled buttons */
    div.stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        background-color: white !important;
        color: #334155 !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
        border-color: #CBD5E1 !important;
        color: #1E293B !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        color: white !important;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input {
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        padding: 10px 12px !important;
        transition: all 0.2s !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) inset !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Check working directory and file existence
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'best_fraud_model.joblib')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
meta_path = os.path.join(BASE_DIR, 'models', 'model_metadata.json')
images_dir = os.path.join(BASE_DIR, 'images')

# Load models and metadata
@st.cache_resource
def load_assets():
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_assets()

# Fraud and Legit Samples for Auto-Fill
SAMPLE_FRAUD = {'Time': 406.0, 'V1': -2.3122265423263, 'V2': 1.95199201064158, 'V3': -1.60985073229769, 'V4': 3.9979055875468, 'V5': -0.522187864667764, 'V6': -1.42654531920595, 'V7': -2.53738730624579, 'V8': 1.39165724829804, 'V9': -2.77008927719433, 'V10': -2.77227214465915, 'V11': 3.20203320709635, 'V12': -2.89990738849473, 'V13': -0.595221881324605, 'V14': -4.28925378244217, 'V15': 0.389724120274487, 'V16': -1.14074717980657, 'V17': -2.83005567450437, 'V18': -0.0168224681808257, 'V19': 0.416955705037907, 'V20': 0.126910559061474, 'V21': 0.517232370861764, 'V22': -0.0350493686052974, 'V23': -0.465211076182388, 'V24': 0.320198198514526, 'V25': 0.0445191674731724, 'V26': 0.177839798284401, 'V27': 0.261145002567677, 'V28': -0.143275874698919, 'Amount': 0.0}
SAMPLE_LEGIT = {'Time': 0.0, 'V1': -1.3598071336738, 'V2': -0.0727811733098497, 'V3': 2.53634673796914, 'V4': 1.37815522427443, 'V5': -0.338320769942518, 'V6': 0.462387777762292, 'V7': 0.239598554061257, 'V8': 0.0986979012610507, 'V9': 0.363786969611213, 'V10': 0.0907941719789316, 'V11': -0.551599533260813, 'V12': -0.617800855762348, 'V13': -0.991389847235408, 'V14': -0.311169353699879, 'V15': 1.46817697209427, 'V16': -0.470400525259478, 'V17': 0.207971241929242, 'V18': 0.0257905801985591, 'V19': 0.403992960255733, 'V20': 0.251412098239705, 'V21': -0.018306777944153, 'V22': 0.277837575558899, 'V23': -0.110473910188767, 'V24': 0.0669280749146731, 'V25': 0.128539358273528, 'V26': -0.189114843888824, 'V27': 0.133558376740387, 'V28': -0.0210530534538215, 'Amount': 149.62}

# --- NLP / Text Analysis Engine ---
KNOWN_MERCHANTS = [
    "amazon", "starbucks", "target", "walmart", "apple", "netflix", 
    "uber", "lyft", "doordash", "mcdonalds", "cvs", "walgreens", 
    "home depot", "lowes", "best buy", "costco", "local grocery supermarket"
]

def analyze_merchant_text(merchant_name):
    """Simple NLP heuristic engine to flag suspicious merchant names."""
    merchant_lower = merchant_name.lower().strip()
    
    # 1. Check for gibberish (e.g., no vowels)
    if not re.search(r'[aeiouy]', merchant_lower):
        return True, "Gibberish detected (no vowels)"
        
    # 2. Check for suspicious test words
    suspicious_words = ["test", "dummy", "asdf", "pani", "fake", "unknown"]
    for word in suspicious_words:
        if word in merchant_lower:
            return True, f"Suspicious keyword detected: '{word}'"
            
    # 3. Length check
    if len(merchant_lower) < 3:
         return True, "Merchant name is too short to be valid"
         
    # Optional: check if it's a known major merchant for a 'trusted' tag, 
    # but don't fail it just because it's unrecognized (like 'Nandini')
    matches = difflib.get_close_matches(merchant_lower, KNOWN_MERCHANTS, n=1, cutoff=0.4)
    if matches:
        return False, f"Recognized trusted merchant: {matches[0].title()}"
        
    return False, "Merchant name looks valid (Unrecognized but benign)"

# Main Application Title
st.title("💳 Credit Card Fraud Detection Platform")
st.markdown("End-to-end machine learning platform showcasing exploratory data analysis and live transaction risk prediction.")

# Create Tabs
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🔮 Live Fraud Prediction"])

# ----------------- TAB 1: EDA -----------------
with tab1:
    st.header("Business Insights & Risk Analysis")
    st.markdown("Below are the key findings from analyzing 283,726 European credit card transactions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temporal Distribution of Fraudulent Activity")
        st.markdown("Fraud attacks are not uniformly distributed throughout the day.")
        st.image(os.path.join(images_dir, '04_hourly_fraud_trend.png'), use_container_width=True)
            
        st.subheader("Risk Analysis by Transaction Amount")
        st.markdown("The majority of total fraud losses are concentrated in the 'Very High' (>€500) tier, which suffers the highest fraud rate.")
        st.image(os.path.join(images_dir, '03_amount_category_fraud_rate.png'), use_container_width=True)
            
    with col2:
        st.subheader("Class Imbalance in Transaction Data")
        st.markdown("Only 0.17% of all transactions are actually fraudulent.")
        st.image(os.path.join(images_dir, '01_class_distribution.png'), use_container_width=True)
            
        st.subheader("Model Performance Metrics")
        st.markdown(f"Our Champion Model (**{meta['model_name']}**) catches a massive amount of fraud with minimal false alarms.")
        st.image(os.path.join(images_dir, '07_confusion_matrices.png'), use_container_width=True)

# ----------------- TAB 2: LIVE PREDICTION -----------------
with tab2:
    st.header("Live Transaction Screener")
    st.markdown("Enter transaction details below. In a real banking environment, these details are securely processed and analyzed for fraud risk.")
    
    # State initialization for backend data
    if "backend_data" not in st.session_state:
        st.session_state.backend_data = {k: 0.0 for k in meta['feature_names']}
        
    # State initialization for UI fields
    if "ui_cardholder" not in st.session_state:
        st.session_state.ui_cardholder = ""
    if "ui_merchant" not in st.session_state:
        st.session_state.ui_merchant = ""
    if "ui_location" not in st.session_state:
        st.session_state.ui_location = ""
    if "ui_amount" not in st.session_state:
        st.session_state.ui_amount = 0.0

    # Action Buttons to simulate scenarios
    st.markdown("#### 🧪 Quick Testing Scenarios")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("🚨 Simulate FRAUD Attack", use_container_width=True):
            st.session_state.backend_data = SAMPLE_FRAUD.copy()
            st.session_state.ui_cardholder = "Jane Smith"
            st.session_state.ui_merchant = "Unknown Electronics Store (Online)"
            st.session_state.ui_location = "High-Risk Foreign IP Address"
            st.session_state.ui_amount = 999.00
            st.rerun()
    with colB:
        if st.button("✅ Simulate LEGIT Transaction", use_container_width=True):
            st.session_state.backend_data = SAMPLE_LEGIT.copy()
            st.session_state.ui_cardholder = "Jane Smith"
            st.session_state.ui_merchant = "Local Grocery Supermarket"
            st.session_state.ui_location = "Hometown, USA"
            st.session_state.ui_amount = 149.62
            st.rerun()
    with colC:
        if st.button("🔄 Clear Inputs", use_container_width=True):
            st.session_state.backend_data = {k: 0.0 for k in meta['feature_names']}
            st.session_state.ui_cardholder = ""
            st.session_state.ui_merchant = ""
            st.session_state.ui_location = ""
            st.session_state.ui_amount = 0.0
            st.rerun()

    st.markdown("---")
    st.markdown("#### 💳 Transaction Details")
    
    # Realistic UI Inputs
    c1, c2 = st.columns(2)
    with c1:
        cardholder = st.text_input("Cardholder Name", value=st.session_state.ui_cardholder)
        location = st.text_input("Location / IP Address", value=st.session_state.ui_location)
    with c2:
        merchant = st.text_input("Merchant Name", value=st.session_state.ui_merchant)
        amount = st.number_input("Transaction Amount ($)", value=st.session_state.ui_amount, min_value=0.0, format="%.2f")

    st.markdown("---")
    
    # Prediction
    if st.button("🔍 Predict Transaction Fraud Risk", type="primary", use_container_width=True):
        if not cardholder or not merchant or not location:
            st.warning("Please fill in the Cardholder, Merchant, and Location fields (or use one of the Quick Testing Scenarios above).")
        else:
            # --- 1. Run NLP Text Analysis ---
            nlp_flag, nlp_reason = analyze_merchant_text(merchant)
            
            # --- 2. Run Mathematical Model ---
            backend_dict = st.session_state.backend_data.copy()
            if 'Amount' in backend_dict:
                backend_dict['Amount'] = float(amount)
                
            input_df = pd.DataFrame([backend_dict])
            scaled_features = scaler.transform(input_df)
            
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]
            fraud_probability = probability[1]
            
            st.header("Prediction Results")
            
            # Combine the risk signals
            is_fraud = prediction == 1 or nlp_flag
            
            if is_fraud:
                st.error(f"🚨 **FRAUD DETECTED** 🚨")
                st.error("This transaction has been flagged as potentially fraudulent.")
                    
                st.warning(f"**Action Required:** We recommend declining the transaction for ${amount:.2f} at '{merchant}' and notifying {cardholder}.")
                
                st.markdown("### Risk Analysis")
                if nlp_flag:
                    st.error(f"📝 **Alert:** {nlp_reason}")
                
                st.write("We noticed some unusual patterns with this transaction:")
                if prediction == 1:
                    st.info(f"🚩 Unusual spending habits or location ({location})")
                if nlp_flag:
                    st.info(f"🚩 Suspicious merchant details: '{merchant}'")
            else:
                st.success(f"✅ **LEGITIMATE TRANSACTION**")
                st.success("The transaction is safe and you can proceed.")
                st.info(f"**Action:** Approve transaction for ${amount:.2f} at {merchant}.")
                
                st.markdown("### Risk Analysis")
                st.write("Everything looks good! This matches standard safe transaction patterns.")

# ----------------- GLOBAL FLOATING CHATBOT -----------------
st.markdown("""
    <style>
    /* 1. Target the outer Streamlit container to float it and remove the blank space */
    div[data-testid="stElementContainer"]:has(div[data-testid="stPopover"]) {
        position: fixed !important;
        bottom: 40px !important;
        right: 40px !important;
        width: auto !important;
        z-index: 9999 !important;
    }
    
    /* 2. Reset the inner popover to behave normally inside the fixed container */
    div[data-testid="stPopover"] {
        position: static !important;
    }
    
    /* 3. Ensure the popover body (chat window) opens nicely without stretching */
    div[data-testid="stPopoverBody"] {
        width: 350px !important;
        max-width: 90vw !important;
        padding: 1rem !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15) !important;
        right: 0 !important;
        left: auto !important;
        transform: none !important;
    }

    /* 4. Style the popover button (FAB) */
    div[data-testid="stPopover"] > button {
        border-radius: 50% !important;
        width: 70px !important;
        height: 70px !important;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.2s !important;
    }
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.3) !important;
    }
    div[data-testid="stPopover"] > button p {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 35px !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.popover("🤖"):
    st.markdown("##### 🤖 Graph Assistant")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**👤 You:** {msg['content']}")
            else:
                st.info(f"**🤖 AI:** {msg['content']}")
            
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_input("Type your question...", label_visibility="collapsed")
        submitted = st.form_submit_button("Send")
        
    if submitted and prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        q = prompt.lower()
        if any(w in q for w in ["time", "hour", "when"]):
             response = "Fraud spikes late at night (around 2:00 AM) when cardholders are asleep."
        elif any(w in q for w in ["amount", "much", "high", "risky", "tier"]):
             response = "Very high transactions (>€500) have the highest probability of being fraudulent."
        elif any(w in q for w in ["rare", "imbalance", "percent", "many", "how often"]):
             response = "Fraud is extremely rare—making up only 0.17% of all transactions."
        elif any(w in q for w in ["accurate", "model", "performance", "ai", "matrix", "catch"]):
             response = "Our AI catches most fraudulent transactions while keeping false alarms very low."
        else:
             response = "Try asking about the 'time' of fraud, 'amounts', data 'imbalance', or the model's 'accuracy'!"
             
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
