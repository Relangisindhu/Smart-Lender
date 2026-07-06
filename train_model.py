"""
train_model.py
---------------
Standalone training pipeline for the "Smart Lender" loan approval prediction system.

This script:
  1. Generates a realistic synthetic loan-applicant dataset.
  2. Cleans / imputes missing values.
  3. Encodes categorical variables.
  4. Handles outliers via capping (winsorization).
  5. Balances classes with SMOTE.
  6. Scales numeric features with StandardScaler.
  7. Trains and compares Decision Tree, Random Forest, KNN and XGBoost.
  8. Selects XGBoost as the production model and serializes everything
     needed for inference (model, scaler, encoders, feature order) into model.pkl.

Run with:
    python train_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_STATE = 42
N_SAMPLES = 4000


# --------------------------------------------------------------------------
# 1. SYNTHETIC DATASET GENERATION
# --------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a realistic synthetic loan-applicant dataset with the same
    feature schema commonly used in loan-approval prediction problems
    (Gender, Marital Status, Education, Employment, Income, Loan Amount,
    Loan Term, Credit History, Property Area) and a derived target label
    (Loan_Status) using a weighted scoring heuristic plus noise, so the
    relationships are realistic but not perfectly separable.
    """
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.78, 0.22])
    married = rng.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.55, 0.18, 0.16, 0.11])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.78, 0.22])
    self_employed = rng.choice(["Yes", "No"], size=n_samples, p=[0.14, 0.86])

    # Income is log-normal-ish to mimic real earnings distributions
    applicant_income = rng.lognormal(mean=8.4, sigma=0.55, size=n_samples).round(0)
    applicant_income = np.clip(applicant_income, 1500, 45000)

    coapplicant_income = np.where(
        married == "Yes",
        rng.lognormal(mean=7.6, sigma=0.7, size=n_samples).round(0),
        0
    )
    coapplicant_income = np.clip(coapplicant_income, 0, 25000)

    loan_amount = rng.normal(loc=145, scale=70, size=n_samples).round(0)  # in thousands
    loan_amount = np.clip(loan_amount, 9, 700)

    loan_term = rng.choice([360, 180, 120, 60, 300, 240, 84, 36],
                           size=n_samples, p=[0.65, 0.10, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02])

    credit_history = rng.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_samples, p=[0.38, 0.38, 0.24])

    # Inject some missing values to require imputation, mirroring real-world data
    def inject_nans(arr, frac):
        arr = arr.astype(object)
        idx = rng.choice(len(arr), size=int(len(arr) * frac), replace=False)
        arr[idx] = np.nan
        return arr

    gender = inject_nans(gender, 0.02)
    married = inject_nans(married, 0.01)
    dependents = inject_nans(dependents, 0.03)
    self_employed = inject_nans(self_employed, 0.05)
    loan_amount_missing = inject_nans(loan_amount.copy(), 0.03)
    loan_term_missing = inject_nans(loan_term.astype(object).copy(), 0.02)
    credit_history_missing = inject_nans(credit_history.astype(object).copy(), 0.08)

    df = pd.DataFrame({
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount_missing,
        "Loan_Amount_Term": loan_term_missing,
        "Credit_History": credit_history_missing,
        "Property_Area": property_area,
    })

    # ---- Build target label from a realistic latent "creditworthiness" score ----
    total_income = df["ApplicantIncome"].astype(float) + df["CoapplicantIncome"].astype(float)
    loan_to_income = (loan_amount * 1000) / (total_income * loan_term / 12 + 1)

    score = (
        2.6 * credit_history
        + 0.9 * (education == "Graduate").astype(int)
        + 0.35 * (property_area != "Rural").astype(int)
        + 0.5 * (married == "Yes").astype(int)
        - 1.8 * loan_to_income
        + 0.000015 * total_income
        - 0.4 * (dependents == "3+").astype(int)
    )
    noise = rng.normal(0, 0.9, size=n_samples)
    latent = score + noise
    prob_approved = 1 / (1 + np.exp(-(latent - np.median(latent))))
    loan_status = (prob_approved > 0.5).astype(int)  # 1 = Approved (Y), 0 = Rejected (N)

    df["Loan_Status"] = np.where(loan_status == 1, "Y", "N")

    return df


# --------------------------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# --------------------------------------------------------------------------
CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                     "Self_Employed", "Property_Area"]
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "Credit_History"]
FEATURE_ORDER = CATEGORICAL_COLS + NUMERIC_COLS


def cap_outliers(df: pd.DataFrame, cols, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Winsorize numeric columns to reduce the influence of extreme outliers."""
    df = df.copy()
    for col in cols:
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df[col] = df[col].clip(lower, upper)
    return df


def build_preprocessors(df: pd.DataFrame):
    """
    Fits imputers, label encoders and a scaler on the training dataframe.
    Returns the fitted objects plus the fully-transformed feature matrix and target.
    """
    df = df.copy()

    # ---- Impute missing categorical values with the mode ----
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[CATEGORICAL_COLS] = cat_imputer.fit_transform(df[CATEGORICAL_COLS])

    # ---- Impute missing numeric values with the median ----
    num_imputer = SimpleImputer(strategy="median")
    df[NUMERIC_COLS] = num_imputer.fit_transform(df[NUMERIC_COLS])

    # ---- Outlier handling (winsorization) on continuous numeric columns ----
    df = cap_outliers(df, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])

    # ---- Label-encode categorical variables ----
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # ---- Encode target ----
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Loan_Status"])  # N=0, Y=1 (alphabetical)

    X = df[FEATURE_ORDER].astype(float)

    # ---- Feature scaling ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    preprocessors = {
        "cat_imputer": cat_imputer,
        "num_imputer": num_imputer,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
    }

    return X_scaled, y, preprocessors


# --------------------------------------------------------------------------
# 3. MODEL TRAINING & COMPARISON
# --------------------------------------------------------------------------
def train_and_compare(X_train, y_train, X_test, y_test):
    """Trains Decision Tree, Random Forest, KNN and XGBoost, prints comparison."""
    results = {}

    dt = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    results["Decision Tree"] = accuracy_score(y_test, dt.predict(X_test))

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test))

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    results["KNN"] = accuracy_score(y_test, knn.predict(X_test))

    xgb = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, xgb.predict(X_train))
    test_acc = accuracy_score(y_test, xgb.predict(X_test))
    results["XGBoost"] = test_acc

    print("\n=== Model Comparison (Test Accuracy) ===")
    for name, acc in results.items():
        print(f"  {name:<15}: {acc * 100:.1f}%")

    print(f"\n=== XGBoost Train Accuracy: {train_acc * 100:.1f}% | Test Accuracy: {test_acc * 100:.1f}% ===")
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, xgb.predict(X_test), target_names=["Rejected", "Approved"]))

    return xgb, results


# --------------------------------------------------------------------------
# 4. MAIN TRAINING ROUTINE
# --------------------------------------------------------------------------
def main():
    print("Generating synthetic loan applicant dataset...")
    df = generate_synthetic_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df["Loan_Status"].value_counts(normalize=True))

    print("\nBuilding preprocessing pipeline (imputation, encoding, outlier capping, scaling)...")
    X, y, preprocessors = build_preprocessors(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Pre-SMOTE class distribution in training set: {np.bincount(y_train)}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Post-SMOTE class distribution in training set: {np.bincount(y_train_res)}")

    print("\nTraining and comparing models...")
    best_model, results = train_and_compare(X_train_res, y_train_res, X_test, y_test)

    artifact = {
        "model": best_model,
        "cat_imputer": preprocessors["cat_imputer"],
        "num_imputer": preprocessors["num_imputer"],
        "label_encoders": preprocessors["label_encoders"],
        "target_encoder": preprocessors["target_encoder"],
        "scaler": preprocessors["scaler"],
        "feature_order": preprocessors["feature_order"],
        "categorical_cols": preprocessors["categorical_cols"],
        "numeric_cols": preprocessors["numeric_cols"],
        "model_metrics": {
            "comparison": results,
            "xgboost_train_accuracy": accuracy_score(y_train_res, best_model.predict(X_train_res)),
            "xgboost_test_accuracy": results["XGBoost"],
        },
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print("\nSaved trained model & preprocessing objects to model.pkl")
    print("Done.")


if __name__ == "__main__":
    main()
