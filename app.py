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

st.set_page_config(page_title="Advanced Loan & Risk Analyzer", layout="wide")

# Custom CSS for Dynamic Multi-color Gradient Background & Glassmorphism
st.markdown("""
<style>
/* Target the main app container */
.stApp {
    background: radial-gradient(circle at 50% 0%, #2c5364 0%, #203a43 50%, #111e3b 100%);
    background-attachment: fixed;
    background-size: 400% 400%;
    animation: gradientBG 5s ease infinite;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism for inputs and forms */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 5px solid rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 8px 50px 0 rgba(0, 0, 0, 0.6);
}

/* Input boxes */
div[data-baseweb="input"] > div, 
div[data-baseweb="select"] > div,
div[data-baseweb="number-input"] > div {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Text inside inputs */
div[data-baseweb="input"] input, 
div[data-baseweb="select"] div,
div[data-baseweb="number-input"] input,
.stTextInput input, 
.stNumberInput input {
    color: white !important;
    background-color: transparent !important;
}

/* Form Submit Button */
button[kind="primaryFormSubmit"], button[data-testid="baseButton-primaryFormSubmit"] {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 8px !important;
    color: white !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

button[kind="primaryFormSubmit"]:hover, button[data-testid="baseButton-primaryFormSubmit"]:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.5rem; /* Increase this number to make it bigger */
    font-weight: bold;  /* Optional: makes the text bold */
}
</style>
""", unsafe_allow_html=True)


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
        pdf.cell(200, 10, txt="Comprehensive Loan Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="--------------------------------------------------", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Applicant Name/ID: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {pred_text} | Risk Level: {risk}", ln=True)
        pdf.cell(200, 10, txt="Conditions:", ln=True)
        for c in conditions: pdf.cell(200, 8, txt=f" - {c}", ln=True)
        
        # PAGE 1: Add first two visuals (Gauge and Radar)
        pdf.image(gauge_file, x=10, y=80, w=80)
        pdf.image(radar_file, x=100, y=80, w=100)
        
        # PAGE 2: Add bottom visuals (Distribution and Features)
        pdf.add_page()
        pdf.cell(200, 10, txt="Visual Context & AI Transparency", ln=True, align='C')
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
st.title("🏦 Advanced Loan & Risk Analyzer")
st.write("With full data visualization integration.")

tab_form, tab_dash = st.tabs(["📝 New Application", "📊 Model Dashboard"])

with tab_form:
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            app_name = st.text_input("Applicant Name / ID", "")
            dependents = st.number_input("No. of Dependents", min_value=0, step=1)
            education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
            self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
            income_annum = st.number_input("Annual Income (INR)", min_value=0, step=100000)
            loan_amount = st.number_input("Loan Amount (INR)", min_value=0, step=100000)
            
        with col2:
            loan_term = st.number_input("Loan Term (Months)", min_value=0, step=12)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=10)
            existing_debt = st.number_input("Existing Annual Debt (INR)", min_value=0, step=10000)
            res_assets = st.number_input("Residential Assets", min_value=0, step=100000)
            com_assets = st.number_input("Commercial Assets", min_value=0, step=100000)
            lux_assets = st.number_input("Luxury Assets", min_value=0, step=100000)
            bank_assets = st.number_input("Bank Asset", min_value=0, step=100000)
            
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
