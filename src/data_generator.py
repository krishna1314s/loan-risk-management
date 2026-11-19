"""Synthetic loan application dataset generator.

Use this when you do not have a real dataset. To use your own data, put a CSV at
data/loan_applications.csv with the same column names (see README).
"""
import numpy as np
import pandas as pd

from .config import DATA_PATH, RANDOM_STATE


def generate_loan_data(n: int = 10000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(40, 11, n), 21, 70).round().astype(int)
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Retired"],
        size=n, p=[0.62, 0.18, 0.08, 0.12],
    )
    years_employed = np.clip(rng.gamma(2.5, 3.0, n), 0, age - 18)
    years_employed = np.where(employment_status == "Unemployed", 0, years_employed).round(1)

    base_income = rng.lognormal(mean=10.9, sigma=0.5, size=n)
    income_mult = np.select(
        [employment_status == "Unemployed", employment_status == "Retired",
         employment_status == "Self-Employed"],
        [0.35, 0.7, 1.05], default=1.0,
    )
    annual_income = (base_income * income_mult).round(-2)

    credit_history_years = np.clip(age - 18 - rng.exponential(6, n), 0, None).round(1)
    credit_score = np.clip(
        rng.normal(660, 85, n) + 1.2 * np.minimum(credit_history_years, 20), 300, 850
    ).round().astype(int)

    loan_purpose = rng.choice(
        ["Debt Consolidation", "Home Improvement", "Car", "Business", "Education", "Medical", "Personal"],
        size=n, p=[0.28, 0.14, 0.16, 0.10, 0.10, 0.08, 0.14],
    )
    home_ownership = rng.choice(["Rent", "Mortgage", "Own"], size=n, p=[0.42, 0.40, 0.18])

    loan_amount = np.clip(annual_income * rng.uniform(0.08, 0.75, n), 1000, 150000).round(-2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.08, 0.17, 0.35, 0.15, 0.25])
    interest_rate = np.clip(
        26 - (credit_score - 300) / 550 * 19 + rng.normal(0, 1.5, n), 4.5, 30
    ).round(2)

    existing_debt = (annual_income * rng.beta(2, 4, n) * 1.3).round(-1)
    num_credit_lines = np.clip(rng.poisson(6, n), 0, 25)
    num_inquiries_6m = np.clip(rng.poisson(1.3, n) + (credit_score < 600) * rng.poisson(1.0, n), 0, 12)

    late_lambda = np.clip((720 - credit_score) / 90, 0.05, 4.5)
    num_late_payments_2y = rng.poisson(late_lambda)
    num_prior_defaults = rng.binomial(1, np.clip((650 - credit_score) / 600, 0.01, 0.45))
    total_prior_loans = rng.integers(0, 8, n)
    prev_repaid_pct = np.where(
        total_prior_loans == 0, np.nan,
        np.clip(1 - 0.08 * num_late_payments_2y - 0.3 * num_prior_defaults + rng.normal(0, 0.06, n), 0, 1),
    ).round(3)

    # --- latent default risk (the hidden "truth" the model must learn) ---
    monthly_pay = loan_amount * (interest_rate / 1200) / (1 - (1 + interest_rate / 1200) ** (-loan_term_months))
    pti = monthly_pay * 12 / np.maximum(annual_income, 1)
    dti = existing_debt / np.maximum(annual_income, 1)
    z = (
        -2.6
        - 0.012 * (credit_score - 650)
        + 3.2 * pti
        + 1.4 * dti
        + 0.35 * num_late_payments_2y
        + 1.1 * num_prior_defaults
        + 0.15 * num_inquiries_6m
        + (employment_status == "Unemployed") * 1.3
        + (employment_status == "Self-Employed") * 0.25
        - 0.04 * np.minimum(years_employed, 15)
        - 0.25 * (home_ownership == "Own")
        + 0.2 * (loan_purpose == "Business")
        + 0.15 * (loan_purpose == "Medical")
        - 0.5 * np.nan_to_num(prev_repaid_pct - 0.8, nan=0.0)
        + rng.normal(0, 0.5, n)
    )
    p_default = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, p_default)

    df = pd.DataFrame({
        "age": age,
        "employment_status": employment_status,
        "years_employed": years_employed,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "credit_history_years": credit_history_years,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "existing_debt": existing_debt,
        "num_credit_lines": num_credit_lines,
        "num_inquiries_6m": num_inquiries_6m,
        "num_late_payments_2y": num_late_payments_2y,
        "num_prior_defaults": num_prior_defaults,
        "previous_loans_repaid_pct": prev_repaid_pct,
        "default": default,
    })

    # inject realistic missing values / messiness
    for col, frac in [("annual_income", 0.02), ("credit_score", 0.015),
                      ("years_employed", 0.03), ("existing_debt", 0.02)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


def load_or_create_data(force_regenerate: bool = False) -> pd.DataFrame:
    if DATA_PATH.exists() and not force_regenerate:
        print(f"[data] Loading existing dataset: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("[data] Generating synthetic dataset ...")
    df = generate_loan_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"[data] Saved to {DATA_PATH}")
    return df
