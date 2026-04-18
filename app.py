import streamlit as st
import pandas as pd
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from fpdf import FPDF

st.set_page_config(page_title="Advanced Loan & Risk Analyzer", layout="wide")

# ==========================================
# 1. LOAD DATA & TRAIN MODEL (Runs once)
# ==========================================
@st.cache_data
def prepare_model():
    try:
        df = pd.read_csv("loan_approval_data.csv")
    except FileNotFoundError:
        st.error("Error: 'loan_approval_data.csv' not found. Please place it in the same folder.")
        st.stop()
        
    if 'Applicant_ID' in df.columns:
        df = df.drop('Applicant_ID', axis=1)

    # Save original features for distribution visualizations before scaling/encoding
    historical_raw = df.copy()

    y_raw = df['Loan_Approved']
    X_raw = df.drop('Loan_Approved', axis=1)

    num_cols = X_raw.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_raw.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="mean")
    X_raw[num_cols] = num_imputer.fit_transform(X_raw[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_raw[cat_cols] = cat_imputer.fit_transform(X_raw[cat_cols])

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)

    le_edu = LabelEncoder()
    X_raw["Education_Level"] = le_edu.fit_transform(X_raw["Education_Level"])

    X_raw["DTI_Ratio_sq"] = X_raw["DTI_Ratio"] ** 2
    X_raw["Credit_Score_sq"] = X_raw["Credit_Score"] ** 2
    
    # Smarter Feature Engineering
    X_raw["Total_Income"] = X_raw["Applicant_Income"] + X_raw["Coapplicant_Income"]
    X_raw["Loan_to_Income_Ratio"] = X_raw["Loan_Amount"] / (X_raw["Total_Income"] + 1)
    
    X_raw = X_raw.drop(columns=["Credit_Score", "DTI_Ratio"])

    X = pd.get_dummies(X_raw, drop_first=True)
    training_columns = X.columns 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Handle Data Imbalance (class_weight='balanced')
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # 2. Hyperparameter Tuning (GridSearchCV)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    # Select the optimal model found
    model = grid_search.best_estimator_

    return model, scaler, num_imputer, cat_imputer, le_edu, training_columns, X_test_scaled, y_test, historical_raw

# Load Everything
rf_model, scaler, num_imputer, cat_imputer, le_edu, training_columns, X_test_scaled, y_test, hist_data = prepare_model()


# ==========================================
# 2. VISUALIZATION FUNCTIONS
# ==========================================
def create_risk_gauge(risk_level, filename):
    plt.figure(figsize=(4, 4))
    
    # Assign a score and color based on risk
    if risk_level == "Low Risk":
        score, color, remainder_color = 25, "#2ecc71", "#e0e0e0"
    elif risk_level == "Medium Risk":
        score, color, remainder_color = 60, "#f1c40f", "#e0e0e0"
    else:
        score, color, remainder_color = 90, "#e74c3c", "#e0e0e0"
        
    sizes = [score, 100 - score]
    colors = [color, remainder_color]
    
    plt.pie(sizes, colors=colors, startangle=90, counterclock=False, wedgeprops=dict(width=0.3, edgecolor='w'))
    plt.text(0, 0, risk_level, horizontalalignment='center', verticalalignment='center', fontsize=14, fontweight='bold')
    plt.title("Risk Meter")
    plt.savefig(filename, bbox_inches='tight', transparent=True)
    plt.close()

def create_radar_chart(applicant_vals, filename):
    # Get averages for Approved loans to compare against
    approved_hist = hist_data[hist_data['Loan_Approved'] == 'Yes']
    avg_income = approved_hist['Applicant_Income'].mean()
    avg_credit = approved_hist['Credit_Score'].mean()
    avg_loan = approved_hist['Loan_Amount'].mean()
    avg_savings = approved_hist['Savings'].mean()
    
    # Normalize values (Applicant vs Average Approved)
    categories = ['Income', 'Credit Score', 'Loan Amount', 'Savings']
    avg_vals = [1.0, 1.0, 1.0, 1.0] # Baseline is 1.0
    
    # Applicant normalized values
    app_norm = [
        applicant_vals['income'] / max(avg_income, 1),
        applicant_vals['credit'] / max(avg_credit, 1),
        applicant_vals['loan'] / max(avg_loan, 1),
        applicant_vals['savings'] / max(avg_savings, 1)
    ]
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    avg_vals += avg_vals[:1]
    app_norm += app_norm[:1]
    
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories)
    ax.plot(angles, avg_vals, linewidth=1, linestyle='solid', label="Avg Approved")
    ax.fill(angles, avg_vals, 'b', alpha=0.1)
    
    ax.plot(angles, app_norm, linewidth=2, linestyle='solid', label="This Applicant", color="orange")
    ax.fill(angles, app_norm, 'orange', alpha=0.25)
    
    plt.title("Applicant vs. Average Approved Profile", size=11, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def create_distribution_plot(loan_amount, filename):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(hist_data['Loan_Amount'].dropna(), fill=True, color="teal")
    plt.axvline(loan_amount, color='red', linestyle='dashed', linewidth=2, label=f"Applicant: ₹{loan_amount}")
    plt.title("Requested Loan Amount vs Historical Distribution")
    plt.xlabel("Loan Amount (INR)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def create_feature_importance(filename):
    plt.figure(figsize=(6, 4))
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': training_columns, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(5) # Top 5
    
    plt.barh(feat_df['Feature'], feat_df['Importance'], color='#3498db')
    plt.title("Top Factors Driving AI Decisions")
    plt.xlabel("Importance Score")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. PDF REPORT GENERATOR
# ==========================================
def calculate_risk(dti, credit_score, loan_amount):
    points = 0
    factors = []
    if dti > 0.4: points += 3; factors.append("High Debt-to-Income Ratio")
    elif dti > 0.25: points += 1; factors.append("Moderate Debt-to-Income Ratio")
        
    if credit_score < 600: points += 3; factors.append("Poor Credit Score")
    elif credit_score < 680: points += 1; factors.append("Fair Credit Score")
        
    if loan_amount > 30000: points += 1; factors.append("Large Loan Amount")
        
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
    
    create_risk_gauge(risk, gauge_file)
    create_radar_chart(app_vals, radar_file)
    create_distribution_plot(app_vals['loan'], dist_file)
    create_feature_importance(feat_file)
    
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

    pdf_output = pdf.output(dest='S').encode('latin-1')

    # Clean up temp images
    for file in [gauge_file, radar_file, dist_file, feat_file]:
        if os.path.exists(file): os.remove(file)
        
    return pdf_output


# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.title("🏦 Advanced Loan & Risk Analyzer")
st.write("Complete with full data visualization integration.")

tab_form, tab_dash = st.tabs(["📝 New Application", "📊 Model Dashboard"])

with tab_form:
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            app_name = st.text_input("Applicant Name / ID", "")
            app_income = st.number_input("Applicant Income (INR)")
            coapp_income = st.number_input("Coapplicant Income (INR)")
            age = st.number_input("Age")
            dependents = st.number_input("Dependents")
            credit_score = st.number_input("Credit Score")
            existing_loans = st.number_input("Existing Loans")
            education = st.selectbox("Education Level", ['Graduate', 'Not Graduate'])
            gender = st.selectbox("Gender", ['Male', 'Female'])
            
        with col2:
            employment = st.selectbox("Employment Status", ['Salaried', 'Self-employed', 'Unemployed'])
            marital_status = st.selectbox("Marital Status", ['Single', 'Married'])
            employer_cat = st.selectbox("Employer Category", ['Government', 'Private', 'MNC', 'Unemployed'])
            loan_amount = st.number_input("Loan Amount (INR)")
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84])
            loan_purpose = st.selectbox("Loan Purpose", ['Home', 'Car', 'Personal', 'Business', 'Education'])
            property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
            dti = st.number_input("Debt-To-Income Ratio (DTI)")
            savings = st.number_input("Savings (INR)")
            collateral = st.number_input("Collateral Value (INR)")
            
        submit_btn = st.form_submit_button("Run Analysis & Generate Report")
        
    if submit_btn:
        user_data = pd.DataFrame([{
            'Applicant_Income': app_income, 'Coapplicant_Income': coapp_income, 'Employment_Status': employment,
            'Age': age, 'Marital_Status': marital_status, 'Dependents': dependents, 'Credit_Score': credit_score,
            'Existing_Loans': existing_loans, 'DTI_Ratio': dti, 'Savings': savings, 'Collateral_Value': collateral,
            'Loan_Amount': loan_amount, 'Loan_Term': loan_term, 'Loan_Purpose': loan_purpose, 
            'Property_Area': property_area, 'Education_Level': education, 'Gender': gender, 'Employer_Category': employer_cat
        }])
        
        user_data["Education_Level"] = le_edu.transform(user_data["Education_Level"])
        user_data["DTI_Ratio_sq"] = user_data["DTI_Ratio"] ** 2
        user_data["Credit_Score_sq"] = user_data["Credit_Score"] ** 2
        
        # Apply the exact same engineered features to user input
        user_data["Total_Income"] = user_data["Applicant_Income"] + user_data["Coapplicant_Income"]
        user_data["Loan_to_Income_Ratio"] = user_data["Loan_Amount"] / (user_data["Total_Income"] + 1)
        
        user_data = user_data.drop(columns=["Credit_Score", "DTI_Ratio"])
        
        user_data = pd.get_dummies(user_data, drop_first=True)
        for col in training_columns:
            if col not in user_data.columns: user_data[col] = 0
        user_data = user_data[training_columns]
        
        user_scaled = scaler.transform(user_data)
        
        pred = rf_model.predict(user_scaled)[0]
        pred_text = "Approved ✅" if pred == 1 else "Rejected ❌"
        pred_text_pdf = "Approved" if pred == 1 else "Rejected"
        risk_lvl, risk_factors, conditions = calculate_risk(dti, credit_score, loan_amount)
        
        st.divider()
        st.subheader("Results")
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
        app_vals = {'income': app_income, 'credit': credit_score, 'loan': loan_amount, 'savings': savings}
        
        session_id = str(uuid.uuid4())
        gauge_file = f"st_gauge_{session_id}.png"
        radar_file = f"st_radar_{session_id}.png"
        dist_file = f"st_dist_{session_id}.png"
        feat_file = f"st_feat_{session_id}.png"
        
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            create_risk_gauge(risk_lvl, gauge_file)
            st.image(gauge_file)
            create_distribution_plot(loan_amount, dist_file)
            st.image(dist_file)
        with vcol2:
            create_radar_chart(app_vals, radar_file)
            st.image(radar_file)
            create_feature_importance(feat_file)
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
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
