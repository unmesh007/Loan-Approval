import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi

def create_risk_gauge(risk_level, filename):
    plt.figure(figsize=(4, 4))
    
    
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

def create_radar_chart(applicant_vals, hist_data, filename):
    
    approved_hist = hist_data[hist_data['loan_status'] == 'Approved']
    avg_income = approved_hist['income_annum'].mean()
    avg_credit = approved_hist['cibil_score'].mean()
    avg_loan = approved_hist['loan_amount'].mean()
    avg_assets = approved_hist['bank_asset_value'].mean()
    
    
    categories = ['Income', 'CIBIL Score', 'Loan Amount', 'Bank Assets']
    avg_vals = [1.0, 1.0, 1.0, 1.0] 
    
    
    app_norm = [
        applicant_vals['income'] / max(avg_income, 1),
        applicant_vals['credit'] / max(avg_credit, 1),
        applicant_vals['loan'] / max(avg_loan, 1),
        applicant_vals['assets'] / max(avg_assets, 1)
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

def create_distribution_plot(loan_amount, hist_data, filename):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(hist_data['loan_amount'].dropna(), fill=True, color="teal")
    plt.axvline(loan_amount, color='red', linestyle='dashed', linewidth=2, label=f"Applicant: ₹{loan_amount}")
    plt.title("Requested Loan Amount vs Historical Distribution")
    plt.xlabel("Loan Amount (INR)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def create_feature_importance(rf_model, training_columns, filename):
    plt.figure(figsize=(6, 4))
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': training_columns, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(5) 
    
    plt.barh(feat_df['Feature'], feat_df['Importance'], color='#3498db')
    plt.title("Top Factors Driving AI Decisions")
    plt.xlabel("Importance Score")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
