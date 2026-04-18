# Advanced Loan & Risk Analyzer

A professional-grade financial assessment tool that combines Machine Learning (Random Forest) with interactive data visualization and automated PDF reporting.

## Features
* **AI Predictions**: Real-time loan approval forecasting using an optimized Random Forest model.
* **Risk Meter**: Dynamic visual gauge indicating the applicant's risk category (Low, Medium, or High).
* **Comparison Profiling**: Radar charts comparing the current applicant to the average profile of historically approved loans.
* **Historical Distribution**: KDE plots visualizing the requested loan amount against historical data density.
* **Automated PDF Reports**: Professional multi-page reports containing prediction results, risk factors, suggested conditions, and all analytical charts.

## Tech Stack
* **Frontend**: Streamlit
* **ML & Analysis**: Scikit-Learn, Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Reporting**: FPDF

## Technical Optimizations & Stability
* **Scalability**: To ensure fast startup as the dataset grows, the model uses a "Train-Save-Load" workflow. Model assets (classifier, scaler, and encoders) are saved to disk using `joblib` rather than retraining on every application launch.
* **PDF Robustness**: The report generator includes `try-except` blocks specifically for `UnicodeEncodeError`. This ensures that names or IDs with special characters or emojis do not cause the application to crash during PDF output.
* **Resource Management**: Implements `try...finally` blocks to guarantee the deletion of temporary chart images, preventing server storage leaks.
* **Feature Engineering**: Calculates advanced metrics such as `Total_Income` and `Loan_to_Income_Ratio` to improve the AI's sensitivity to debt levels.

## Setup & Installation
1. **Dependencies**: Install the required libraries:
   `pip install streamlit pandas scikit-learn matplotlib seaborn fpdf joblib`
2. **Data**: Ensure `loan_approval_data.csv` is located in the root directory.
3. **Execution**: Run the Streamlit application:
   `streamlit run app.py`

## Data Schema
The analyzer evaluates a comprehensive set of features:
* **Financials**: Applicant/Coapplicant Income, Savings, Collateral Value, Loan Amount.
* **Credit History**: Credit Score, Existing Loans, Debt-to-Income (DTI) Ratio.
* **Demographics**: Age, Gender, Education Level, Marital Status, Dependents.
* **Context**: Employment Status, Employer Category, Loan Purpose, Property Area.

---
*Automated Credit Risk Management and Transparency.*
