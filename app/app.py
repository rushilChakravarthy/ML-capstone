"""
Streamlit app for the loan-default classifier.

SCAFFOLD: it needs a trained model that the team saves from classification.ipynb, e.g.

    import joblib
    from sklearn.pipeline import Pipeline
    best_pipeline = Pipeline([("pre", preprocessor), ("clf", best_model)])   # both already fitted
    joblib.dump(best_pipeline, "../models/best_classifier.joblib")

Run locally:   streamlit run app/app.py
Deploy:        Streamlit Community Cloud / Hugging Face Spaces / Render (commit models/best_classifier.joblib;
               see .gitignore, which un-ignores exactly that file).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_classifier.joblib"

# Columns the notebook's preprocessor expects (see Section B of notebooks/classification.ipynb).
NUMERIC = ["disbursed_amount", "asset_cost", "ltv", "age_years", "PERFORM_CNS.SCORE",
           "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE",
           "PRIMARY.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS",
           "avg_acct_age_months", "credit_history_months", "NO.OF_INQUIRIES"]
CATEGORICAL = ["Employment.Type", "PERFORM_CNS.SCORE.DESCRIPTION", "manufacturer_id", "State_ID",
               "Aadhar_flag", "PAN_flag", "VoterID_flag"]

st.set_page_config(page_title="Vehicle Loan Default Predictor", page_icon="🚗")
st.title("Vehicle Loan Default Predictor")
st.caption("23CSE301 ML Capstone — classification track")

if not MODEL_PATH.exists():
    st.warning(f"No trained model found at `{MODEL_PATH.relative_to(MODEL_PATH.parent.parent)}`. "
               "Save the fitted preprocessing + best classifier pipeline from the notebook first "
               "(instructions at the top of app/app.py).")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.subheader("Borrower & loan details")
col1, col2 = st.columns(2)
with col1:
    disbursed = st.number_input("Disbursed amount", 10_000, 1_000_000, 54_000, step=1_000)
    asset_cost = st.number_input("Asset cost", 30_000, 1_700_000, 76_000, step=1_000)
    ltv = st.slider("Loan-to-value ratio (%)", 10.0, 95.0, 75.0)
    age = st.slider("Age (years)", 18, 70, 33)
with col2:
    score = st.number_input("Credit bureau score (0 = no history)", 0, 900, 0)
    inquiries = st.number_input("Number of credit inquiries", 0, 36, 0)
    employment = st.selectbox("Employment type", ["Salaried", "Self employed"])

# Unspecified columns are left missing: the notebook's imputers fill them with train-set median / mode.
row = {c: np.nan for c in NUMERIC + CATEGORICAL}
row.update({"disbursed_amount": disbursed, "asset_cost": asset_cost, "ltv": ltv, "age_years": age,
            "PERFORM_CNS.SCORE": score, "NO.OF_INQUIRIES": inquiries, "Employment.Type": employment})
X_new = pd.DataFrame([row])[NUMERIC + CATEGORICAL]

if st.button("Predict"):
    proba = float(model.predict_proba(X_new)[0, 1])
    st.metric("Estimated default probability", f"{proba:.1%}")
    st.write("**Prediction:** " + ("⚠️ likely to default" if proba >= 0.5 else "✅ unlikely to default"))
    st.caption("Class imbalance (~22% defaults) means probabilities, not the 0.5 cut-off, are the more informative output.")
