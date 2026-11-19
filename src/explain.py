"""Explainability: feature importance and high-risk pattern discovery."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

from .config import RANDOM_STATE


def feature_names(pipe):
    return list(pipe.named_steps["prep"].get_feature_names_out())


def model_importance(pipe) -> pd.Series:
    """Native importance: tree impurity importance, or |coefficient| for linear models."""
    clf, names = pipe.named_steps["clf"], feature_names(pipe)
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    else:
        vals = np.abs(clf.coef_[0])
    return pd.Series(vals, index=names).sort_values(ascending=False)


def logistic_coefficients(pipe) -> pd.Series:
    """Signed coefficients (positive => raises default risk). Features are standardised."""
    clf, names = pipe.named_steps["clf"], feature_names(pipe)
    return pd.Series(clf.coef_[0], index=names).sort_values()


def permutation_imp(pipe, X_test, y_test, n_repeats: int = 8) -> pd.DataFrame:
    """Model-agnostic importance on raw input columns (drop in ROC-AUC when shuffled)."""
    r = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"importance_mean": r.importances_mean, "importance_std": r.importances_std},
                         index=X_test.columns).sort_values("importance_mean", ascending=False))


def risk_patterns(df: pd.DataFrame) -> dict:
    """Default rate by meaningful segments - shows which profiles are high risk."""
    out = {}
    d = df.copy()
    segs = {
        "credit_band": "credit_band",
        "employment_status": "employment_status",
        "loan_purpose": "loan_purpose",
        "home_ownership": "home_ownership",
        "late_payments_bucket": pd.cut(d["num_late_payments_2y"], [-1, 0, 1, 3, 100],
                                       labels=["0", "1", "2-3", "4+"]),
        "dti_bucket": pd.qcut(d["debt_to_income"], 5, duplicates="drop"),
        "payment_to_income_bucket": pd.qcut(d["payment_to_income"], 5, duplicates="drop"),
        "prior_default": d["has_prior_default"].map({0: "No", 1: "Yes"}),
    }
    for name, key in segs.items():
        series = d[key] if isinstance(key, str) else key
        grp = (d.groupby(series, observed=True)["default"]
                 .agg(default_rate="mean", applicants="count").reset_index())
        grp.columns = [name, "default_rate", "applicants"]
        grp[name] = grp[name].astype(str)
        out[name] = grp
    return out


def decision_rules(X_train, y_train, max_depth: int = 3) -> str:
    """A shallow decision tree rendered as human-readable if/then rules (original units)."""
    from .preprocessing import build_preprocessor
    prep = build_preprocessor(X_train, scale=False)
    Xp = prep.fit_transform(X_train)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=100,
                                  class_weight="balanced", random_state=RANDOM_STATE)
    tree.fit(Xp, y_train)
    return export_text(tree, feature_names=list(prep.get_feature_names_out()))
