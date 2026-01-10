import streamlit as st
import requests
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Credit Risk Engine Pro",
    page_icon="🏦",
    layout="wide", # Uses the whole screen width
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR "BREATHTAKING" LOOK ---
# This applies custom styling to headers and metrics
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: 600; color: #333; }
    .metric-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    /* Hide the default submit button if we use real-time */
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_header1, col_header2 = st.columns([1, 4])
with col_header1:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=80)
with col_header2:
    st.title("Enterprise Credit Risk System")
    st.caption("Machine Learning-Powered Loan Assessment Engine v2.0")
st.divider()

# --- MAIN LAYOUT (2 Columns) ---
left_column, right_column = st.columns([2, 3], gap="large")

with left_column:
    st.markdown('<p class="big-font">Applicant Details</p>', unsafe_allow_html=True)
    
    # 1. Financial Inputs (Sliders with formatting)
    with st.container(border=True):
        st.subheader("💰 Financial Request")
        # Formatted as Currency
        credit_amount = st.slider("Loan Amount Request", 500, 20000, 5000, step=500, format="€%d")
        # Formatted as months
        duration = st.slider("Duration", 6, 72, 24, step=6, format="%d months")

    # 2. Personal Inputs
    with st.container(border=True):
        st.subheader("👤 Personal Profile")
        age = st.slider("Applicant Age", 18, 75, 30, format="%d years")
        
        # CLEAN DROPDOWNS (User sees clean text, we use index)
        job_display = ["Management / Self-Employed", "Skilled Employee", "Unemployed / Non-Res", "Unskilled Resident"]
        job_choice = st.selectbox("Current Job Level", job_display, index=1)

    # 3. Financial History (The most important part)
    with st.container(border=True):
        st.subheader("🏦 Financial Health")
        # CLEAN DROPDOWNS (Replacing ugly brackets)
        # NOTE: These MUST correspond alphabetically to the raw data categories!
        # 0: 0<=X<200, 1: <0, 2: >=200, 3: no checking
        checking_clean = [
            "Medium Balance (0-200 DM)",  # Index 0
            "Overdrawn (Negative)",       # Index 1
            "High Balance (>200 DM)",     # Index 2
            "No Checking Account"         # Index 3
        ]
        checking_choice = st.selectbox("Current Checking Status", checking_clean, index=0)
        
        savings_clean = ["Little / None (<100 DM)", "Moderate (100-1000 DM)", "Substantial (>1000 DM)"]
        savings_choice = st.selectbox("Savings Account History", savings_clean, index=0)

    # 4. Purpose
    with st.expander("Additional Details (Purpose)"):
        purpose_clean = ["Business Investment", "Domestic Appliances", "Education Costs", "Furniture/Equipment", "New Car Purchase", "Used Car Purchase"]
        purpose_choice = st.selectbox("Loan Purpose", purpose_clean, index=5)


# --- REAL-TIME ANALYSIS (RIGHT COLUMN) ---
with right_column:
    st.markdown('<p class="big-font">Risk Analysis Report</p>', unsafe_allow_html=True)
    
    # Prepare Data Payload
    # Map clean display inputs back to their alphabetical indices
    data = {
        "age": age,
        "credit_amount": credit_amount,
        "duration": duration,
        "job": job_display.index(job_choice),
        # IMPORTANT: Ensure these indices match the alphabetical order of your training data's ugly strings
        "checking_status": checking_clean.index(checking_choice), 
        "savings_status": savings_clean.index(savings_choice),
        "purpose": purpose_clean.index(purpose_choice)
    }

    # Call API instantly whenever inputs change
    try:
        # Add a tiny delay to simulate "processing" and prevent UI flickering
        with st.spinner("Running ML Inference Model..."):
            time.sleep(0.3) 
            response = requests.post("http://localhost:8000/predict", json=data)
            result = response.json()

        # DYNAMIC RESULT CARD
        container_color = "#d4edda" if result["decision"] == "APPROVE" else "#f8d7da"
        text_color = "#155724" if result["decision"] == "APPROVE" else "#721c24"
        icon = "✅" if result["decision"] == "APPROVE" else "🚨"
        title = "Recommended for Approval" if result["decision"] == "APPROVE" else "High Risk - Denied"

        st.markdown(f"""
            <div style="background-color: {container_color}; padding: 20px; border-radius: 10px; border-left: 10px solid {text_color};">
                <h2 style="color: {text_color}; margin:0;">{icon} {title}</h2>
            </div>
            <br>
        """, unsafe_allow_html=True)

        # Metrics Section (Custom CSS container)
        with st.container():
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Decision Status", result["decision"])
            with col_m2:
                prob_formatted = f"{result['risk_probability'] * 100:.1f}%"
                st.metric("ML Risk Score (Probability)", prob_formatted, delta="-Low Risk" if result["decision"]=="APPROVE" else "+High Risk", delta_color="inverse")
            
            st.write("Risk Probabilty Gauge:")
            risk_score = result['risk_probability']
            # Custom color gradient for progress bar depending on score
            if risk_score < 0.3: bar_color = "green"
            elif risk_score < 0.6: bar_color = "orange"
            else: bar_color = "red"
            st.progress(risk_score)
            st.caption(f"Applicant falls into the {result['risk_class']} category based on the German Credit dataset model.")
            st.markdown('</div>', unsafe_allow_html=True)

    except requests.exceptions.ConnectionError:
         st.error("⚠️ Could not connect to the Credit Engine API. Please ensure the backend (`main.py`) is running.")
    except Exception as e:
         st.error(f"An error occurred: {e}")

# --- SIDEBAR (For Context) ---
with st.sidebar:
    st.header("About the Engine")
    st.info("This tool uses a **Random Forest Classifier** trained on historical banking data to predict loan default probability in real-time.")
    st.markdown("---")
    st.write("**Backend:** FastAPI + Pydantic")
    st.write("**ML Model:** Scikit-Learn")
    st.write("**Frontend:** Streamlit")