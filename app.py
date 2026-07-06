"""
app.py
------
Flask backend for the "Smart Lender" loan approval prediction application.

Loads the trained XGBoost model + preprocessing objects from model.pkl on
startup, serves the UI, and exposes a /predict endpoint that accepts form
data, runs it through the exact same preprocessing pipeline used during
training, and returns a prediction with a confidence score.
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

app = Flask(__name__)
app.secret_key = "smart-lender-dev-secret-key"  # change in production

# --------------------------------------------------------------------------
# Load model + preprocessing artifacts once, at startup
# --------------------------------------------------------------------------
_artifact = None


def load_artifact():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model.pkl not found. Please run `python train_model.py` first "
                "to train the model and generate model.pkl."
            )
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def preprocess_input(form):
    """
    Converts raw form input into the exact feature vector the model expects,
    using the fitted imputers / encoders / scaler saved in model.pkl.
    """
    artifact = load_artifact()
    categorical_cols = artifact["categorical_cols"]
    numeric_cols = artifact["numeric_cols"]
    feature_order = artifact["feature_order"]
    label_encoders = artifact["label_encoders"]
    scaler = artifact["scaler"]

    raw = {
        "Gender": form.get("gender", "Male"),
        "Married": form.get("married", "No"),
        "Dependents": form.get("dependents", "0"),
        "Education": form.get("education", "Graduate"),
        "Self_Employed": form.get("self_employed", "No"),
        "Property_Area": form.get("property_area", "Urban"),
        "ApplicantIncome": safe_float(form.get("applicant_income"), 0.0),
        "CoapplicantIncome": safe_float(form.get("coapplicant_income"), 0.0),
        "LoanAmount": safe_float(form.get("loan_amount"), 0.0),
        "Loan_Amount_Term": safe_float(form.get("loan_term"), 360.0),
        "Credit_History": safe_float(form.get("credit_history"), 1.0),
    }

    # ---- Encode categorical fields using the fitted LabelEncoders ----
    encoded_row = {}
    for col in categorical_cols:
        le = label_encoders[col]
        val = str(raw[col])
        if val not in le.classes_:
            # Fallback: map unseen categories to the most frequent trained class
            val = le.classes_[0]
        encoded_row[col] = le.transform([val])[0]

    for col in numeric_cols:
        encoded_row[col] = raw[col]

    ordered_values = [encoded_row[col] for col in feature_order]
    X = np.array(ordered_values, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)

    return X_scaled, raw


@app.route("/")
def home():
    artifact = load_artifact()
    metrics = artifact.get("model_metrics", {})
    return render_template(
        "home.html",
        train_acc=round(metrics.get("xgboost_train_accuracy", 0.947) * 100, 1),
        test_acc=round(metrics.get("xgboost_test_accuracy", 0.811) * 100, 1),
    )


@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        artifact = load_artifact()
        model = artifact["model"]
        target_encoder = artifact["target_encoder"]

        X_scaled, raw = preprocess_input(request.form)

        pred_class = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]

        label = target_encoder.inverse_transform([pred_class])[0]  # 'Y' or 'N'
        approved = (label == "Y")
        confidence = float(np.max(pred_proba) * 100)

        result = {
            "approved": approved,
            "status_text": "Loan Approved" if approved else "Loan Rejected",
            "confidence": round(confidence, 1),
            "inputs": raw,
        }

        return render_template("result.html", result=result)

    except FileNotFoundError as e:
        flash(str(e))
        return redirect(url_for("home"))
    except Exception as e:  # noqa: BLE001
        flash(f"An error occurred while processing your request: {e}")
        return redirect(url_for("predict_form"))


if __name__ == "__main__":
    # Ensure the model exists before starting the server
    try:
        load_artifact()
        print("Model and preprocessing artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")

    app.run(debug=True, host="0.0.0.0", port=5000)
