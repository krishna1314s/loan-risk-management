"""Score a single applicant with the saved model:  python predict.py"""
import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.modeling import assign_risk_tier
from src.preprocessing import engineer_features

applicant = {
    "age": 34, "employment_status": "Employed", "years_employed": 6.0,
    "annual_income": 58000, "credit_score": 590, "credit_history_years": 9.0,
    "home_ownership": "Rent", "loan_purpose": "Personal", "loan_amount": 30000,
    "loan_term_months": 36, "interest_rate": 17.5, "existing_debt": 21000,
    "num_credit_lines": 5, "num_inquiries_6m": 4, "num_late_payments_2y": 3,
    "num_prior_defaults": 1, "previous_loans_repaid_pct": 0.7,
}

model = joblib.load(MODEL_DIR / "best_model.joblib")
X = engineer_features(pd.DataFrame([applicant]))
proba = model.predict_proba(X)[:, 1][0]
print(f"Default probability: {proba:.1%}")
print(f"Risk class         : {assign_risk_tier([proba])[0]}")
