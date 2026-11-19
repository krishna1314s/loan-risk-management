"""Cleaning, feature engineering and the sklearn preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TARGET

CATEGORICAL = ["employment_status", "home_ownership", "loan_purpose", "credit_band"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, fix impossible values."""
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    for col in ("annual_income", "loan_amount", "existing_debt"):
        if col in df:
            df.loc[df[col] < 0, col] = np.nan
    if "credit_score" in df:
        df.loc[~df["credit_score"].between(300, 850), "credit_score"] = np.nan
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise domain features (no statistics learned, so no leakage)."""
    df = df.copy()
    income = df["annual_income"].replace(0, np.nan)
    r = df["interest_rate"] / 1200
    n = df["loan_term_months"]
    df["monthly_payment"] = df["loan_amount"] * r / (1 - (1 + r) ** (-n))
    df["loan_to_income"] = df["loan_amount"] / income
    df["debt_to_income"] = df["existing_debt"] / income
    df["payment_to_income"] = df["monthly_payment"] * 12 / income
    df["total_debt_burden"] = (df["existing_debt"] + df["loan_amount"]) / income
    df["late_payments_per_year_of_history"] = df["num_late_payments_2y"] / np.maximum(df["credit_history_years"], 1)
    df["has_prior_default"] = (df["num_prior_defaults"] > 0).astype(int)
    df["no_repayment_history"] = df["previous_loans_repaid_pct"].isna().astype(int)
    df["employment_stability"] = df["years_employed"] / np.maximum(df["age"] - 18, 1)
    df["credit_band"] = pd.cut(
        df["credit_score"], bins=[0, 579, 669, 739, 799, 851],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    ).astype(object)
    df["credit_band"] = df["credit_band"].fillna("Unknown")
    # Winsorise ratio features so extreme values don't dominate linear models
    for col in ("loan_to_income", "debt_to_income", "payment_to_income", "total_debt_burden"):
        df[col] = df[col].clip(upper=df[col].quantile(0.99))
    return df.replace([np.inf, -np.inf], np.nan)


def build_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        verbose_feature_names_out=False,
    )


def prepare_xy(df: pd.DataFrame):
    df = engineer_features(clean_data(df))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y, df
