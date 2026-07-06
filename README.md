# Smart Lender

Smart Lender is an ML-powered web application that predicts the creditworthiness of loan applicants, helping financial institutions make faster, data-driven loan approval decisions.

It combines a scikit-learn/XGBoost preprocessing + modeling pipeline with a Flask web app and a clean, dashboard-style UI.

---

## Features

- Synthetic but realistic loan-applicant dataset generation (no external data needed)
- Full preprocessing pipeline: missing value imputation, categorical encoding, outlier capping, SMOTE class balancing, feature scaling
- Model comparison across Decision Tree, Random Forest, KNN, and XGBoost
- XGBoost selected as the production model
- Flask backend serving a home dashboard, prediction form, and result page
- Responsive, professional UI (deep blue/purple financial dashboard theme)
- Structured to be cloud-ready (e.g. for deployment on IBM Cloud or similar platforms)

---

## Project Structure

```
smart_lender/
├── requirements.txt        # Python dependencies
├── train_model.py          # Data generation, preprocessing, training, serialization
├── app.py                  # Flask backend (routes, inference)
├── model.pkl                # Generated after training (model + preprocessors)
├── templates/
│   ├── base.html            # Shared layout
│   ├── home.html            # Dashboard / landing page
│   ├── predict.html         # Loan applicant input form
│   └── result.html          # Prediction result page
└── static/
    └── css/
        └── style.css         # Application styling
```

---

## Requirements

- Python 3.9+
- pip

Dependencies (see `requirements.txt`):

- Flask
- numpy
- pandas
- scikit-learn
- xgboost
- imbalanced-learn (SMOTE)
- joblib
- Werkzeug

---

## Setup & Installation

1. **Unzip / clone the project** and move into the folder:
   ```bash
   cd smart_lender
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (generates `model.pkl` in the project root):
   ```bash
   python train_model.py
   ```
   This will:
   - Generate a synthetic dataset of loan applicants
   - Run the full preprocessing pipeline (imputation, encoding, outlier capping, SMOTE, scaling)
   - Train and compare Decision Tree, Random Forest, KNN, and XGBoost
   - Print accuracy/classification metrics for each model to the console
   - Save the trained XGBoost model + preprocessing objects to `model.pkl`

5. **Run the Flask app:**
   ```bash
   python app.py
   ```

6. **Open your browser** to:
   ```
   http://localhost:5000
   ```

---

## Usage

1. From the home page, click **"New Application"** or **"Start New Loan Assessment"**.
2. Fill in the applicant details: gender, marital status, dependents, education, employment status, applicant/co-applicant income, loan amount, loan term, credit history, and property area.
3. Submit the form.
4. View the result page showing:
   - **Loan Approved** or **Loan Rejected** (green/red styling)
   - Model confidence percentage
   - A breakdown of all submitted applicant details

---

## Notes on the Data & Model

- The dataset used for training is **synthetically generated** inside `train_model.py` (there is no external CSV dependency), with realistic income distributions and injected missing values to require real preprocessing.
- The target label (`Loan_Status`) is derived from a weighted "creditworthiness" score plus noise, so the model has genuine, learnable — but not perfectly separable — patterns to pick up on.
- Reported accuracy will vary slightly between runs/machines depending on library versions and random seeds, since the data is generated at training time rather than fixed.
- To retrain with different assumptions (e.g. more samples, different feature weights), edit `generate_synthetic_dataset()` in `train_model.py`.

---

## Retraining / Updating the Model

If you change anything in the preprocessing or feature schema in `train_model.py`, re-run:

```bash
python train_model.py
```

This regenerates `model.pkl`. Restart `app.py` afterward so the Flask app picks up the new model:

```bash
python app.py
```

---

## Deployment Notes (Cloud-Ready Structure)

The project is structured so it can be adapted for cloud deployment (e.g. IBM Cloud, Heroku, Render, AWS, etc.):

- `app.py` uses `app.run(host="0.0.0.0", port=5000)`, which is compatible with most container/cloud runtimes.
- For production, replace the Flask dev server with a WSGI server such as **gunicorn**:
  ```bash
  pip install gunicorn
  gunicorn -w 4 -b 0.0.0.0:5000 app:app
  ```
- Set `debug=False` in `app.py` before deploying to production.
- Store the Flask `secret_key` in an environment variable rather than hardcoding it.
- Ensure `model.pkl` is generated (via `train_model.py`) as part of your build/deploy process, or include it as a build artifact.

---

## Troubleshooting

- **`model.pkl not found` error on startup**: Run `python train_model.py` first to generate the model file before starting `app.py`.
- **`ModuleNotFoundError`**: Make sure you've activated your virtual environment and run `pip install -r requirements.txt`.
- **XGBoost install issues on some platforms**: Ensure you have an up-to-date `pip` (`pip install --upgrade pip`) before installing `requirements.txt`.

---

## License

This project is provided for educational and demonstration purposes.