import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def train_and_save():
    print("Loading data...")
    try:
        df = pd.read_csv("loan_approval_dataset.csv")
    except FileNotFoundError:
        print("Error: 'loan_approval_dataset.csv' not found.")
        return
        
    if 'Applicant_ID' in df.columns:
        df = df.drop('Applicant_ID', axis=1)

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
    
    print("Running GridSearchCV (this may take a few seconds)...")
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Pack what the app needs into a dictionary
    artifacts = {
        "model": model, 
        "scaler": scaler, 
        "le_edu": le_edu, 
        "training_columns": training_columns,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test
    }
    
    joblib.dump(artifacts, "loan_model_assets.joblib")
    print("Training complete. Assets saved to loan_model_assets.joblib")

if __name__ == "__main__":
    train_and_save()
