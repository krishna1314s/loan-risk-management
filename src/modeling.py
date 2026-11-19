"""Model training, comparison, evaluation, threshold selection, risk tiers."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             brier_score_loss)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .config import LOW_RISK_MAX, MEDIUM_RISK_MAX, RANDOM_STATE
from .preprocessing import build_preprocessor


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, learning_rate=0.05,
                                                        max_depth=3, subsample=0.8,
                                                        random_state=RANDOM_STATE),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Pick the probability cut-off that maximises F1."""
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
        "Threshold": threshold,
    }


def train_and_compare(X_train, y_train, X_test, y_test):
    """Fit every model; return fitted pipelines, a results table and test probabilities.

    The decision threshold for each model is tuned for F1 using out-of-fold predictions on the
    TRAINING data only, so the test set stays untouched until final evaluation.
    """
    fitted, rows, probas = {}, [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, clf in get_models().items():
        print(f"[model] Training {name} ...")
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = best_f1_threshold(y_train, oof)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        res = evaluate(y_test, proba, threshold=thr)
        res.update({"Model": name, "CV ROC-AUC": roc_auc_score(y_train, oof)})
        rows.append(res)
        fitted[name], probas[name] = pipe, proba
    results = pd.DataFrame(rows).set_index("Model")
    return fitted, results, probas


def assign_risk_tier(proba) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    return np.select([proba < LOW_RISK_MAX, proba < MEDIUM_RISK_MAX], ["Low", "Medium"], default="High")


def build_risk_report(X_test: pd.DataFrame, y_test, proba) -> pd.DataFrame:
    report = X_test.copy()
    report["actual_default"] = np.asarray(y_test)
    report["default_probability"] = np.round(proba, 4)
    report["risk_class"] = assign_risk_tier(proba)
    report["decision_suggestion"] = report["risk_class"].map(
        {"Low": "Approve", "Medium": "Manual review / adjust terms", "High": "Decline or require collateral"}
    )
    return report


def confusion(y_true, proba, threshold):
    return confusion_matrix(y_true, (proba >= threshold).astype(int))
