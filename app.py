from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
from urllib.parse import parse_qs

import joblib
import pandas as pd

from src.config import MODEL_DIR
from src.modeling import assign_risk_tier
from src.preprocessing import engineer_features


MODEL = joblib.load(MODEL_DIR / "best_model.joblib")

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Loan Risk Assessment</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f4f7fb; color: #172033; }
    main { max-width: 960px; margin: 32px auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px #dce3ed; }
    header { padding: 24px 32px; border-bottom: 1px solid #e7ecf3; }
    form { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; padding: 32px; }
    label { display: flex; flex-direction: column; font-weight: 600; font-size: 14px; gap: 7px; }
    input, select { padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 15px; }
    button { grid-column: 1 / -1; padding: 12px; border: 0; border-radius: 6px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    .result { margin: 0 32px 32px; padding: 18px; border-radius: 8px; background: #eef4ff; }
    .low { background: #ecfdf5; color: #166534; }
    .medium { background: #fffbeb; color: #92400e; }
    .high { background: #fef2f2; color: #991b1b; }
    @media (max-width: 700px) { form { grid-template-columns: 1fr; } main { margin: 0; border-radius: 0; } }
  </style>
</head>
<body>
  <main>
    <header><h1>Loan Default Risk Assessment</h1></header>
    <form method="post">
      <label>Age <input name="age" type="number" min="18" max="100" value="34" required></label>
      <label>Employment status
        <select name="employment_status">
          <option>Employed</option><option>Self-Employed</option><option>Unemployed</option><option>Retired</option>
        </select>
      </label>
      <label>Years employed <input name="years_employed" type="number" min="0" step="0.1" value="6"></label>
      <label>Annual income <input name="annual_income" type="number" min="0" step="100" value="58000" required></label>
      <label>Credit score <input name="credit_score" type="number" min="300" max="850" value="590" required></label>
      <label>Credit history (years) <input name="credit_history_years" type="number" min="0" step="0.1" value="9"></label>
      <label>Home ownership
        <select name="home_ownership"><option>Rent</option><option>Mortgage</option><option>Own</option><option>Other</option></select>
      </label>
      <label>Loan purpose
        <select name="loan_purpose"><option>Personal</option><option>Debt Consolidation</option><option>Auto</option><option>Home Improvement</option><option>Education</option><option>Business</option></select>
      </label>
      <label>Loan amount <input name="loan_amount" type="number" min="0" step="100" value="30000" required></label>
      <label>Loan term (months) <input name="loan_term_months" type="number" min="1" value="36" required></label>
      <label>Interest rate (%) <input name="interest_rate" type="number" min="0" step="0.1" value="17.5" required></label>
      <label>Existing debt <input name="existing_debt" type="number" min="0" step="100" value="21000"></label>
      <label>Credit lines <input name="num_credit_lines" type="number" min="0" value="5"></label>
      <label>Inquiries (6 months) <input name="num_inquiries_6m" type="number" min="0" value="4"></label>
      <label>Late payments (2 years) <input name="num_late_payments_2y" type="number" min="0" value="3"></label>
      <label>Prior defaults <input name="num_prior_defaults" type="number" min="0" value="1"></label>
      <label>Repaid loans (%) <input name="previous_loans_repaid_pct" type="number" min="0" max="100" value="70"></label>
      <button type="submit">Assess Risk</button>
    </form>
    __RESULT__
  </main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.replace("__RESULT__", "").encode("utf-8"))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        fields = parse_qs(self.rfile.read(length).decode("utf-8"))
        values = {key: parts[0] for key, parts in fields.items()}

        numeric_fields = [
            "age", "years_employed", "annual_income", "credit_score",
            "credit_history_years", "loan_amount", "loan_term_months",
            "interest_rate", "existing_debt", "num_credit_lines",
            "num_inquiries_6m", "num_late_payments_2y", "num_prior_defaults",
        ]
        applicant = {key: float(values[key]) if key in numeric_fields else values[key]
                     for key in numeric_fields + [
                         "employment_status", "home_ownership", "loan_purpose"
                     ]}
        applicant["previous_loans_repaid_pct"] = float(values["previous_loans_repaid_pct"]) / 100

        X = engineer_features(pd.DataFrame([applicant]))
        probability = float(MODEL.predict_proba(X)[:, 1][0])
        risk = str(assign_risk_tier([probability])[0])
        result = (
            f'<section class="result {risk.lower()}">'
            f"<h2>Result: {html.escape(risk)} risk</h2>"
            f"<p>Estimated default probability: <strong>{probability:.1%}</strong></p>"
            "</section>"
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.replace("__RESULT__", result).encode("utf-8"))

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Loan Risk Interface running at http://127.0.0.1:8000")
    server.serve_forever()
