# Predictive Loan Approval & Financial Risk Assessment

End-to-end ML project: preprocessing, feature engineering, default-risk modelling,
model comparison, explainability and visual reports.

## Run in VS Code
```bash
# 1. open this folder in VS Code (File > Open Folder)
# 2. open a terminal (Ctrl+`) and create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. install and run
pip install -r requirements.txt
python main.py                  # full pipeline (~1 min)
python predict.py               # score one sample applicant
```
Select the `.venv` interpreter in VS Code (Ctrl+Shift+P > "Python: Select Interpreter").

## Using your own data
Put a CSV at `data/loan_applications.csv` with these columns (target = `default`, 1 = defaulted):
age, employment_status, years_employed, annual_income, credit_score, credit_history_years,
home_ownership, loan_purpose, loan_amount, loan_term_months, interest_rate, existing_debt,
num_credit_lines, num_inquiries_6m, num_late_payments_2y, num_prior_defaults,
previous_loans_repaid_pct, default.
If the file is missing, a synthetic dataset is generated automatically (`--regenerate` to rebuild).

## Project layout
| Path | Purpose |
|---|---|
| `main.py` | Runs everything |
| `predict.py` | Score a new applicant with the saved model |
| `src/data_generator.py` | Synthetic loan data (with missing values) |
| `src/preprocessing.py` | Cleaning, feature engineering, sklearn pipeline |
| `src/modeling.py` | Logistic Regression / Random Forest / Gradient Boosting, metrics, risk tiers |
| `src/explain.py` | Feature importance, permutation importance, segment patterns, decision rules |
| `src/visualize.py` | All charts |

## Outputs
- `outputs/plots/` - 12 charts (EDA, patterns, model comparison, ROC/PR, confusion matrices, importance, risk tiers, calibration)
- `outputs/reports/` - model_comparison.csv, feature importance CSVs, risk_classification_test_set.csv, risk_tier_summary.csv, high_risk_decision_rules.txt, pattern_*.csv
- `models/best_model.joblib` - saved pipeline

## Risk classes
Predicted default probability: **Low** < 20% (approve), **Medium** 20-50% (manual review), **High** >= 50% (decline / collateral). Edit in `src/config.py`.

## Engineered features
monthly_payment, loan_to_income, debt_to_income, payment_to_income, total_debt_burden,
late_payments_per_year_of_history, has_prior_default, no_repayment_history,
employment_stability, credit_band.
