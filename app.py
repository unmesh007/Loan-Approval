import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from fpdf import FPDF
import io

st.set_page_config(page_title="Advanced Loan & Risk Analyzer", layout="wide")

# ==========================================
# DYNAMIC THEME ENGINE
# ==========================================
def get_dynamic_theme(risk_level="Neutral", approved=None):
    """
    Returns custom CSS with a two-color diagonal gradient that reflects BOTH
    the risk level and the loan approval decision.

    Gradient mapping:
      Risk ↓ / Approval →    Approved          Not Approved
      Low Risk            →  Green  + Blue     Green  + Red
      Medium Risk         →  Yellow + Blue     Yellow + Red
      High Risk           →  Red    + Blue     Red    (solid red tones)
      Neutral (default)   →  Black  + Grey     (B&W)

    Uses deep/dark versions of each colour so glassmorphism + white text stay readable.
    """

    # ── Risk colour (left/top side of gradient) ──────────────────────────────
    if risk_level == "High Risk":
        risk_dark   = "#4a0a0a"   # deep crimson
        risk_mid    = "#2a0404"
        risk_accent = "#ff2d2d"   # bright red for the animated shimmer
    elif risk_level == "Medium Risk":
        risk_dark   = "#4a3a00"   # deep gold
        risk_mid    = "#2a2000"
        risk_accent = "#f5a623"   # amber for shimmer
    elif risk_level == "Low Risk":
        risk_dark   = "#003d12"   # deep forest green
        risk_mid    = "#001f09"
        risk_accent = "#1db954"   # vibrant green for shimmer
    else:
        # Neutral / initial default — pure black & white palette
        risk_dark   = "#1a1a1a"
        risk_mid    = "#0a0a0a"
        risk_accent = "#888888"

    # ── Approval colour (right/bottom side of gradient) ───────────────────────
    if approved is None:
        # No prediction yet → keep single-tone (same as risk, no second hue)
        appr_dark   = risk_mid
        appr_accent = risk_accent
    elif approved:
        appr_dark   = "#001a3d"   # deep navy blue
        appr_accent = "#0057ff"   # electric blue shimmer
    else:
        appr_dark   = "#3d0000"   # deep dark red
        appr_accent = "#cc0000"   # vivid red shimmer

    css = f"""
    <style>
    /* ── Animated diagonal gradient background ── */
    .stApp {{
        background: linear-gradient(
            135deg,
            {risk_dark}  0%,
            {risk_mid}  40%,
            {appr_dark} 70%,
            #000000    100%
        );
        background-size: 300% 300%;
        animation: dynamicBG 8s ease infinite;
        transition: background 2s ease-in-out;
    }}

    @keyframes dynamicBG {{
        0%   {{ background-position: 0%   50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0%   50%; }}
    }}

    /* Subtle top-edge glow using the accent colours */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, {risk_accent}, {appr_accent});
        z-index: 9999;
        animation: accentPulse 3s ease-in-out infinite alternate;
    }}

    @keyframes accentPulse {{
        from {{ opacity: 0.6; }}
        to   {{ opacity: 1.0; }}
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Glassmorphism for inputs and forms */
    div[data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 5px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 50px 0 rgba(0, 0, 0, 0.6);
    }}

    /* Input boxes */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div,
    div[data-baseweb="number-input"] > div {{
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }}

    /* Text inside inputs */
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] div,
    div[data-baseweb="number-input"] input,
    .stTextInput input, 
    .stNumberInput input {{
        color: white !important;
        background-color: transparent !important;
    }}

    /* Form Submit Button */
    button[kind="primaryFormSubmit"], button[data-testid="baseButton-primaryFormSubmit"] {{
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        color: white !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    button[kind="primaryFormSubmit"]:hover, button[data-testid="baseButton-primaryFormSubmit"]:hover {{
        background: rgba(255, 255, 255, 0.25) !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }}

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        font-size: 1.5rem;
        font-weight: bold;
    }}

    /* --- Card Navigation Style for Tabs --- */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(18, 15, 23, 0.6) !important;
        padding: 10px;
        border-radius: 16px;
        gap: 15px;
        border: 1px solid rgba(132, 0, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        border-radius: 10px !important;
        padding: 5px 20px !important;
        border: 1px solid transparent !important;
        transition: all 0.3s ease;
    }}

    /* Hover effect on unselected tabs */
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(132, 0, 255, 0.1) !important;
        border: 1px solid rgba(132, 0, 255, 0.3) !important;
    }}

    /* Active Selected Tab */
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, #8400ff, #b366ff) !important;
        box-shadow: 0 4px 15px rgba(132, 0, 255, 0.4) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-bottom-color: rgba(255,255,255,0.3) !important;
    }}

    /* --- Metric Cards Style --- */
    [data-testid="stMetric"] {{
        background-color: rgba(18, 15, 23, 0.7);
        border-radius: 16px;
        border: 1px solid rgba(132, 0, 255, 0.2);
        box-shadow: 0 0 15px rgba(132, 0, 255, 0.05),
                    inset 0 0 15px rgba(132, 0, 255, 0.02);
        padding: 1.5rem;
        transition: all 0.3s ease-in-out;
    }}

    [data-testid="stMetric"]:hover {{
        border-color: rgba(132, 0, 255, 0.6);
        box-shadow: 0 0 25px rgba(132, 0, 255, 0.2),
                    inset 0 0 25px rgba(132, 0, 255, 0.1);
        transform: translateY(-5px);
    }}

    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
        color: white !important;
    }}
    </style>
    """
    return css

# Initialize session state for the dynamic theme
# Stores both risk level and approval decision so the gradient is fully informed.
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "Neutral"
if 'current_approved' not in st.session_state:
    st.session_state.current_approved = None   # None = no prediction yet

# Placeholder at the top — CSS is injected here at the END of the script
# so it always captures the latest session_state after form submission.
css_placeholder = st.empty()

def animated_title(text):
    """
    Replicates the GSAP SplitText incoming animation using pure CSS and Python.
    Animates characters from y: 40px and opacity: 0 to y: 0px and opacity: 1.
    """
    html_content = """
    <style>
    .split-char {
        display: inline-block;
        opacity: 0;
        transform: translateY(40px);
        animation: textIn 1.25s cubic-bezier(0.165, 0.84, 0.44, 1) forwards;
    }
    @keyframes textIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .animated-title-container {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: "Source Sans Pro", sans-serif;
        margin-bottom: 1rem;
        color: inherit;
    }
    </style>
    <div class="animated-title-container">
    """
    
    # Base delay matches your GSAP 50ms (0.05s) stagger configuration
    delay = 0.05 
    for char in text:
        if char == " ":
            html_content += f'<span class="split-char" style="animation-delay: {delay}s;">&nbsp;</span>'
        else:
            html_content += f'<span class="split-char" style="animation-delay: {delay}s;">{char}</span>'
        delay += 0.05 
        
    html_content += "</div>"
    
    # Inject into Streamlit
    st.markdown(html_content, unsafe_allow_html=True)

def apply_bento_form_style():
    """
    Injects CSS to make the native Streamlit form look like a glowing Bento card.
    """
    st.markdown(
        """
        <style>
        /* Target the main Streamlit form container */
        [data-testid="stForm"] {
            background-color: #100F19;
            border-radius: 16px;
            border: 1px solid rgba(132, 0, 255, 0.2);
            box-shadow: 0 0 20px rgba(132, 0, 255, 0.1),
                        inset 0 0 20px rgba(132, 0, 255, 0.05);
            padding: 2rem;
            transition: all 0.3s ease-in-out;
        }
        
        /* Add a subtle hover glow */
        [data-testid="stForm"]:hover {
            border-color: rgba(132, 0, 255, 0.6);
            box-shadow: 0 0 30px rgba(132, 0, 255, 0.2),
                        inset 0 0 30px rgba(132, 0, 255, 0.1);
            transform: translateY(-2px);
        }

        /* Style the submit button to match the theme */
        [data-testid="stForm"] button {
            background: linear-gradient(90deg, #8400ff, #b366ff);
            color: white;
            border: none;
            border-radius: 8px;
            transition: transform 0.2s;
        }
        
        [data-testid="stForm"] button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(132, 0, 255, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )




# ==========================================
# 1. LOAD PRE-TRAINED CACHED MODEL
# ==========================================
@st.cache_resource
def load_assets():
    try:
        assets = joblib.load("loan_model_assets.joblib")
    except FileNotFoundError:
        st.error("Model assets not found. Please run 'train_model.py' first.")
        st.stop()
        
    try:
        hist_data = pd.read_csv("loan_approval_dataset.csv")
        hist_data.columns = hist_data.columns.str.strip()
        cat_cols = hist_data.select_dtypes(include=["object"]).columns
        for c in cat_cols:
            hist_data[c] = hist_data[c].str.strip()
    except FileNotFoundError:
        st.error("Historical data 'loan_approval_dataset.csv' not found.")
        st.stop()
        
    return assets["model"], assets["scaler"], assets["le_edu"], assets["training_columns"], assets["X_test_scaled"], assets["y_test"], hist_data

# Load Everything
rf_model, scaler, le_edu, training_columns, X_test_scaled, y_test, hist_data = load_assets()


from visualizations import create_risk_gauge, create_radar_chart, create_distribution_plot, create_feature_importance

# ==========================================
# 2. PDF REPORT GENERATOR
# ==========================================
def calculate_risk(cibil, lti, dti, loan_amount):
    points = 0
    factors = []
    
    if lti > 6: points += 3; factors.append("Extremely High Loan-to-Income Multiple")
    elif lti > 3: points += 1; factors.append("High Loan-to-Income Multiple")
        
    if dti > 0.43: points += 3; factors.append("High Debt-to-Income Ratio")
    elif dti > 0.36: points += 1; factors.append("Elevated Debt-to-Income Ratio")
        
    if cibil < 500: points += 3; factors.append("Poor CIBIL Score")
    elif cibil < 700: points += 1; factors.append("Fair CIBIL Score")
        
    if loan_amount > 10000000: points += 1; factors.append("Large Jumbo Loan Amount")
        
    if points >= 4: return "High Risk", factors, ["Require Guarantor", "Lower Loan Amount"]
    elif points >= 2: return "Medium Risk", factors, ["Higher Interest Rate", "Shorter Term"]
    return "Low Risk", ["None"], ["Standard Terms"]

def create_pdf(name, app_vals, pred_text, risk, conditions):
    # 1. Generate Images
    session_id = str(uuid.uuid4())
    gauge_file = f"temp_gauge_{session_id}.png"
    radar_file = f"temp_radar_{session_id}.png"
    dist_file = f"temp_dist_{session_id}.png"
    feat_file = f"temp_feat_{session_id}.png"
    
    try:
        create_risk_gauge(risk, gauge_file)
        create_radar_chart(app_vals, hist_data, radar_file)
        create_distribution_plot(app_vals['loan'], hist_data, dist_file)
        create_feature_importance(rf_model, training_columns, feat_file)
        
        pdf = FPDF()
        # PAGE 1: Text
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, text="Comprehensive Loan Analysis Report", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, text="--------------------------------------------------", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.cell(200, 10, text=f"Applicant Name/ID: {name}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(200, 10, text=f"Prediction: {pred_text} | Risk Level: {risk}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(200, 10, text="Conditions:", new_x="LMARGIN", new_y="NEXT")
        for c in conditions: pdf.cell(200, 8, text=f" - {c}", new_x="LMARGIN", new_y="NEXT")
        
        # PAGE 1: Add first two visuals (Gauge and Radar)
        pdf.image(gauge_file, x=10, y=80, w=80)
        pdf.image(radar_file, x=100, y=80, w=100)
        
        # PAGE 2: Add bottom visuals (Distribution and Features)
        pdf.add_page()
        pdf.cell(200, 10, text="Visual Context & AI Transparency", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.image(dist_file, x=20, y=30, w=160)
        pdf.image(feat_file, x=20, y=140, w=160)
    
        try:
            # Standard attempt
            pdf_output = pdf.output(dest='S').encode('latin-1')
        except UnicodeEncodeError:
            st.warning("Certain special characters were found and adjusted for PDF compatibility.")
            pdf_output = pdf.output(dest='S').encode('latin-1', errors='replace')
            
    finally:
        # Clean up temp images
        for file in [gauge_file, radar_file, dist_file, feat_file]:
            if os.path.exists(file): 
                os.remove(file)
        
    return pdf_output


# ==========================================
# 4. STREAMLIT UI
# ==========================================
animated_title("🏦 Advanced Loan & Risk Analyzer")
st.write("With full data visualization integration.")

tab_form, tab_dash, tab_batch = st.tabs(["📝 New Application", "📊 Model Dashboard", "📁 Batch Upload"])

with tab_form:
    st.markdown("### 📥 Pre-fill from Excel/CSV (Optional)")
    uploaded_single = st.file_uploader("Upload an Excel or CSV file to pre-fill the form below", type=["xlsx", "csv"], key="single_upload")
    if uploaded_single is not None:
        try:
            if uploaded_single.name.endswith(".csv"):
                df_single = pd.read_csv(uploaded_single)
            else:
                df_single = pd.read_excel(uploaded_single)
            
            if not df_single.empty:
                row = df_single.iloc[0]
                # Try to match typical column names in the dataset
                # The dataset uses: no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value
                st.session_state.prefill = {
                    'dependents': int(row.get('no_of_dependents', 0)) if 'no_of_dependents' in df_single.columns else 0,
                    'education': 'Graduate' if ('education' in df_single.columns and 'Not' not in str(row.get('education', ''))) else 'Not Graduate',
                    'self_employed': 'Yes' if ('self_employed' in df_single.columns and 'Yes' in str(row.get('self_employed', ''))) else 'No',
                    'income': int(row.get('income_annum', 0)) if 'income_annum' in df_single.columns else 0,
                    'loan': int(row.get('loan_amount', 0)) if 'loan_amount' in df_single.columns else 0,
                    'term': int(row.get('loan_term', 0)) if 'loan_term' in df_single.columns else 0,
                    'cibil': int(row.get('cibil_score', 300)) if 'cibil_score' in df_single.columns else 300,
                    'debt': int(row.get('existing_debt', 0)) if 'existing_debt' in df_single.columns else 0,
                    'res_assets': int(row.get('residential_assets_value', 0)) if 'residential_assets_value' in df_single.columns else 0,
                    'com_assets': int(row.get('commercial_assets_value', 0)) if 'commercial_assets_value' in df_single.columns else 0,
                    'lux_assets': int(row.get('luxury_assets_value', 0)) if 'luxury_assets_value' in df_single.columns else 0,
                    'bank_assets': int(row.get('bank_asset_value', 0)) if 'bank_asset_value' in df_single.columns else 0
                }
                st.success("Form pre-filled successfully from the uploaded file!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    prefill = st.session_state.get('prefill', {})

    apply_bento_form_style()
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            app_name = st.text_input("Applicant Name / ID", "")
            dependents = st.number_input("No. of Dependents", min_value=0, step=1, value=prefill.get('dependents', 0))
            
            # Helper to get the correct index for selectbox prefill
            edu_index = 0 if prefill.get('education', 'Graduate') == 'Graduate' else 1
            se_index = 0 if prefill.get('self_employed', 'Yes') == 'Yes' else 1
            
            education = st.selectbox("Education", ['Graduate', 'Not Graduate'], index=edu_index)
            self_employed = st.selectbox("Self Employed", ['Yes', 'No'], index=se_index)
            income_annum = st.number_input("Annual Income (INR)", min_value=0, step=100000, value=prefill.get('income', 0))
            loan_amount = st.number_input("Loan Amount (INR)", min_value=0, step=100000, value=prefill.get('loan', 0))
            
        with col2:
            loan_term = st.number_input("Loan Term (Months)", min_value=0, step=12, value=prefill.get('term', 0))
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=10, value=max(300, prefill.get('cibil', 300)))
            existing_debt = st.number_input("Existing Annual Debt (INR)", min_value=0, step=10000, value=prefill.get('debt', 0))
            res_assets = st.number_input("Residential Assets", min_value=0, step=100000, value=prefill.get('res_assets', 0))
            com_assets = st.number_input("Commercial Assets", min_value=0, step=100000, value=prefill.get('com_assets', 0))
            lux_assets = st.number_input("Luxury Assets", min_value=0, step=100000, value=prefill.get('lux_assets', 0))
            bank_assets = st.number_input("Bank Asset", min_value=0, step=100000, value=prefill.get('bank_assets', 0))
            
        submit_btn = st.form_submit_button("Run Analysis & Generate Report")
        
    if submit_btn:
        user_data = pd.DataFrame([{
            'no_of_dependents': dependents, 
            'education': education, 
            'self_employed': self_employed,
            'income_annum': income_annum, 
            'loan_amount': loan_amount, 
            'loan_term': loan_term, 
            'cibil_score': cibil_score,
            'residential_assets_value': res_assets, 
            'commercial_assets_value': com_assets, 
            'luxury_assets_value': lux_assets, 
            'bank_asset_value': bank_assets
        }])
        
        user_data["education"] = le_edu.transform(user_data["education"])
        
        # Apply the exact same engineered features to user input
        user_data["Total_Assets"] = user_data["residential_assets_value"] + user_data["commercial_assets_value"] + user_data["luxury_assets_value"] + user_data["bank_asset_value"]
        user_data["Loan_to_Income_Ratio"] = user_data["loan_amount"] / (user_data["income_annum"] + 1)
        user_data["cibil_score_sq"] = user_data["cibil_score"] ** 2
        
        user_data = pd.get_dummies(user_data, drop_first=True)
        for col in training_columns:
            if col not in user_data.columns: user_data[col] = 0
        user_data = user_data[training_columns]
        
        user_scaled = scaler.transform(user_data)
        
        pred = rf_model.predict(user_scaled)[0]
        pred_text = "Approved ✅" if pred == 0 else "Rejected ❌"
        pred_text_pdf = "Approved" if pred == 0 else "Rejected"
        
        calculated_dti = existing_debt / max(1, income_annum)
        risk_lvl, risk_factors, conditions = calculate_risk(cibil_score, user_data["Loan_to_Income_Ratio"][0], calculated_dti, loan_amount)
        
        # --- THEME UPDATE TRIGGER ---
        # Store BOTH risk level and approval decision so the CSS placeholder
        # at the bottom of the file can render the correct two-colour gradient.
        st.session_state.current_theme    = risk_lvl
        st.session_state.current_approved = (pred == 0)  # True = Approved, False = Rejected
        
        st.divider()
        st.subheader("Results")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Loan-to-Income Multiple", f"{user_data['Loan_to_Income_Ratio'][0]:.2f}")
        col_m2.metric("Debt-to-Income Ratio (DTI)", f"{calculated_dti:.2%}")
        
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.success(f"**Prediction:** {pred_text}")
            st.info(f"**Risk Level:** {risk_lvl}")
            st.write(f"**Risk Factors:** {', '.join(risk_factors)}")
        
        with col_res2:
            st.write("**Suggested Conditions:**")
            for c in conditions: st.write(f"- {c}")

        # Show visualizations on screen
        st.divider()
        st.write("### Data Visualizations")
        app_vals = {'income': income_annum, 'credit': cibil_score, 'loan': loan_amount, 'assets': bank_assets}
        
        session_id = str(uuid.uuid4())
        gauge_file = f"st_gauge_{session_id}.png"
        radar_file = f"st_radar_{session_id}.png"
        dist_file = f"st_dist_{session_id}.png"
        feat_file = f"st_feat_{session_id}.png"
        
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            create_risk_gauge(risk_lvl, gauge_file)
            st.image(gauge_file)
            create_distribution_plot(loan_amount, hist_data, dist_file)
            st.image(dist_file)
        with vcol2:
            create_radar_chart(app_vals, hist_data, radar_file)
            st.image(radar_file)
            create_feature_importance(rf_model, training_columns, feat_file)
            st.image(feat_file)
            
        for f in [gauge_file, dist_file, radar_file, feat_file]:
            if os.path.exists(f): os.remove(f)
        
        # PDF Generator
        pdf_file = create_pdf(app_name, app_vals, pred_text_pdf, risk_lvl, conditions)
        st.download_button("📄 Download Complete PDF Report (with Charts)", data=pdf_file, file_name="Loan_Visual_Report.pdf", type="primary")

with tab_dash:
    st.header("Overall AI Performance")
    test_preds = rf_model.predict(X_test_scaled)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, test_preds):.2%}")
    col2.metric("Precision", f"{precision_score(y_test, test_preds, average='weighted', zero_division=0):.2%}")
    col3.metric("Recall", f"{recall_score(y_test, test_preds, average='weighted', zero_division=0):.2%}")
    
    st.write("---")
    cm = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, use_container_width=False)

with tab_batch:
    st.header("📁 Batch Upload & Processing")
    st.write("Upload a CSV or Excel file containing multiple loan applications to process them all at once.")
    
    uploaded_batch = st.file_uploader("Upload Applications File", type=["xlsx", "csv"], key="batch_upload")
    
    if uploaded_batch is not None:
        try:
            if uploaded_batch.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_batch)
            else:
                batch_df = pd.read_excel(uploaded_batch)
                
            st.write("### Data Preview")
            st.dataframe(batch_df.head())
            
            if st.button("Run Batch Analysis"):
                with st.spinner("Processing applications..."):
                    # Create a copy for processing
                    proc_df = batch_df.copy()
                    
                    # Clean column names
                    proc_df.columns = proc_df.columns.str.strip()
                    
                    # Ensure required columns are present or set defaults
                    required_cols = [
                        'no_of_dependents', 'education', 'self_employed', 'income_annum', 
                        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 
                        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
                    ]
                    
                    missing_cols = [col for col in required_cols if col not in proc_df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns in uploaded file: {', '.join(missing_cols)}")
                    else:
                        # Clean categorical data
                        cat_cols = proc_df.select_dtypes(include=["object"]).columns
                        for c in cat_cols:
                            proc_df[c] = proc_df[c].str.strip()
                            
                        # Encode education
                        # Note: if there are unknown labels, transform will fail. We use a safe approach.
                        proc_df["education"] = proc_df["education"].apply(lambda x: x if x in le_edu.classes_ else le_edu.classes_[0])
                        proc_df["education"] = le_edu.transform(proc_df["education"])
                        
                        # Feature engineering
                        proc_df["Total_Assets"] = proc_df["residential_assets_value"] + proc_df["commercial_assets_value"] + proc_df["luxury_assets_value"] + proc_df["bank_asset_value"]
                        proc_df["Loan_to_Income_Ratio"] = proc_df["loan_amount"] / (proc_df["income_annum"] + 1)
                        proc_df["cibil_score_sq"] = proc_df["cibil_score"] ** 2
                        
                        proc_df = pd.get_dummies(proc_df, drop_first=True)
                        for col in training_columns:
                            if col not in proc_df.columns: proc_df[col] = 0
                        proc_df = proc_df[training_columns]
                        
                        # Scale
                        batch_scaled = scaler.transform(proc_df)
                        
                        # Predict
                        batch_preds = rf_model.predict(batch_scaled)
                        
                        # Add results to original dataframe
                        batch_df['Prediction'] = ["Approved ✅" if p == 0 else "Rejected ❌" for p in batch_preds]
                        
                        # Add risk levels
                        risk_levels = []
                        for idx, row in batch_df.iterrows():
                            cibil = row.get('cibil_score', 0)
                            lti = row.get('loan_amount', 0) / max(1, row.get('income_annum', 1))
                            dti = row.get('existing_debt', 0) / max(1, row.get('income_annum', 1))
                            loan_amt = row.get('loan_amount', 0)
                            risk, _, _ = calculate_risk(cibil, lti, dti, loan_amt)
                            risk_levels.append(risk)
                            
                        batch_df['Risk Level'] = risk_levels
                        
                        st.success("Batch analysis complete!")
                        st.write("### Analysis Results")
                        st.dataframe(batch_df)
                        
                        # Provide download link for Excel
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            batch_df.to_excel(writer, index=False, sheet_name='Analysis_Results')
                        
                        st.download_button(
                            label="📥 Download Results as Excel",
                            data=output.getvalue(),
                            file_name="Batch_Analysis_Results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")


# ==========================================
# INJECT CSS DYNAMICALLY
# ==========================================
# Written into the placeholder defined at the top of the file.
# Placing it here (bottom of script) ensures it always reads the LATEST
# session_state value — including any risk level set inside the form block.
css_placeholder.markdown(
    get_dynamic_theme(
        st.session_state.current_theme,
        st.session_state.current_approved,
    ),
    unsafe_allow_html=True,
)