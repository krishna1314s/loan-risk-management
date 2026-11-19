"""Run the full pipeline:  python main.py

Steps: data -> cleaning -> feature engineering -> model comparison -> evaluation ->
explainability -> risk classification -> charts + reports.
"""
import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src import explain, visualize as viz
from src.config import MODEL_DIR, RANDOM_STATE, REPORT_DIR
from src.data_generator import load_or_create_data
from src.modeling import build_risk_report, train_and_compare
from src.preprocessing import prepare_xy


def main(regenerate: bool = False):
    # 1. Data ---------------------------------------------------------------
    raw = load_or_create_data(force_regenerate=regenerate)
    print(f"[data] Shape: {raw.shape} | missing cells: {int(raw.isna().sum().sum())}")

    # 2. Preprocessing + feature engineering --------------------------------
    X, y, full = prepare_xy(raw)
    print(f"[prep] Features after engineering: {X.shape[1]} | default rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3. EDA charts ------------------------------------------------------------
    viz.class_balance(y)
    viz.correlation_heatmap(full)
    viz.distributions(full)
    patterns = explain.risk_patterns(full)
    viz.pattern_charts(patterns)

    # 4. Train & compare models -----------------------------------------------
    fitted, results, probas = train_and_compare(X_train, y_train, X_test, y_test)
    results = results.round(4)
    results.to_csv(REPORT_DIR / "model_comparison.csv")
    print("\n=== MODEL COMPARISON (test set) ===")
    print(results[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
                   "CV ROC-AUC", "Threshold"]].to_string())

    viz.model_comparison(results)
    viz.roc_pr_curves(y_test, probas)
    viz.confusion_matrices(y_test, probas, results)
    viz.calibration_plot(y_test, probas)

    best_name = results["ROC-AUC"].idxmax()
    best = fitted[best_name]
    print(f"\n[best] {best_name} (highest ROC-AUC)")

    # 5. Explainability --------------------------------------------------------
    native = explain.model_importance(best)
    native.to_csv(REPORT_DIR / "feature_importance_best_model.csv", header=["importance"])
    viz.feature_importance(native, f"Feature importance - {best_name}", "08_feature_importance.png")

    perm = explain.permutation_imp(best, X_test, y_test)
    perm.to_csv(REPORT_DIR / "permutation_importance.csv")
    viz.permutation_chart(perm, "09_permutation_importance.png")

    coefs = explain.logistic_coefficients(fitted["Logistic Regression"])
    coefs.to_csv(REPORT_DIR / "logistic_coefficients.csv", header=["coefficient"])
    viz.coefficient_chart(coefs, "10_logistic_coefficients.png")

    rules = explain.decision_rules(X_train, y_train)
    (REPORT_DIR / "high_risk_decision_rules.txt").write_text(rules)

    for name, table in patterns.items():
        table.to_csv(REPORT_DIR / f"pattern_{name}.csv", index=False)

    # 6. Risk classification of test applicants -------------------------------
    report = build_risk_report(X_test, y_test, probas[best_name])
    report.to_csv(REPORT_DIR / "risk_classification_test_set.csv", index=False)
    viz.risk_distribution(report)

    tier = report.groupby("risk_class").agg(
        applicants=("actual_default", "size"),
        actual_default_rate=("actual_default", "mean"),
        avg_predicted_prob=("default_probability", "mean"),
    ).reindex(["Low", "Medium", "High"]).round(4)
    tier.to_csv(REPORT_DIR / "risk_tier_summary.csv")
    print("\n=== RISK TIER SUMMARY ===")
    print(tier.to_string())

    print("\n=== TOP 10 DRIVERS OF DEFAULT RISK (permutation importance) ===")
    print(perm.head(10).round(4).to_string())

    print("\n=== HIGH-RISK DECISION RULES (shallow tree) ===")
    print(rules)

    # 7. Save model ----------------------------------------------------------------
    joblib.dump(best, MODEL_DIR / "best_model.joblib")
    print(f"[done] Model saved -> {MODEL_DIR / 'best_model.joblib'}")
    print(f"[done] Reports -> {REPORT_DIR}")
    print("[done] Charts  -> outputs/plots")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic dataset")
    main(ap.parse_args().regenerate)
