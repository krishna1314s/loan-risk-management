"""All charts for the visual report."""
import matplotlib
matplotlib.use("Agg")  # works without a display; files are saved to outputs/plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

from .config import PLOT_DIR

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"Low": "#2e9e5b", "Medium": "#f0a30a", "High": "#d64545"}


def _save(fig, name):
    path = PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path.name}")


def class_balance(y):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    ax.bar(["Repaid (0)", "Default (1)"], counts.values, color=["#4c78a8", "#d64545"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Target class balance"); ax.set_ylabel("Applicants")
    _save(fig, "01_class_balance.png")


def correlation_heatmap(df):
    num = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation matrix (numeric & engineered features)")
    _save(fig, "02_correlation_heatmap.png")


def distributions(df):
    cols = ["credit_score", "debt_to_income", "payment_to_income", "annual_income"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, c in zip(axes.ravel(), cols):
        sns.kdeplot(data=df, x=c, hue="default", common_norm=False, fill=True, ax=ax,
                    palette={0: "#4c78a8", 1: "#d64545"})
        ax.set_title(f"{c} by outcome")
    _save(fig, "03_feature_distributions.png")


def pattern_charts(patterns: dict):
    names = ["credit_band", "employment_status", "late_payments_bucket", "dti_bucket",
             "payment_to_income_bucket", "loan_purpose"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    overall = None
    for ax, n in zip(axes.ravel(), names):
        g = patterns[n]
        if n == "credit_band":
            g = g.set_index(n).reindex(["Poor", "Fair", "Good", "Very Good", "Excellent", "Unknown"]).dropna().reset_index()
        sns.barplot(data=g, x=n, y="default_rate", ax=ax, color="#d64545")
        ax.set_title(f"Default rate by {n}")
        ax.set_ylabel("Default rate"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    _save(fig, "04_high_risk_patterns.png")


def model_comparison(results: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    data = results[metrics].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=data, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Model performance comparison (test set)")
    ax.legend(loc="lower right")
    _save(fig, "05_model_comparison.png")


def roc_pr_curves(y_test, probas: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=name)
        pr, rc, _ = precision_recall_curve(y_test, p)
        axes[1].plot(rc, pr, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
    axes[0].legend(); axes[1].legend()
    _save(fig, "06_roc_pr_curves.png")


def confusion_matrices(y_test, probas: dict, results: pd.DataFrame):
    fig, axes = plt.subplots(1, len(probas), figsize=(5 * len(probas), 4.5))
    for ax, (name, p) in zip(np.atleast_1d(axes), probas.items()):
        thr = results.loc[name, "Threshold"]
        ConfusionMatrixDisplay.from_predictions(y_test, (p >= thr).astype(int), ax=ax,
                                                display_labels=["Repaid", "Default"],
                                                cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\n(threshold={thr:.2f})")
    _save(fig, "07_confusion_matrices.png")


def feature_importance(series: pd.Series, title: str, filename: str, top: int = 15, color="#4c78a8"):
    s = series.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(s.index, s.values, color=color)
    ax.set_title(title)
    _save(fig, filename)


def permutation_chart(df: pd.DataFrame, filename: str):
    d = df.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d.index, d["importance_mean"], xerr=d["importance_std"], color="#7b5ea7")
    ax.set_title("Permutation importance (drop in ROC-AUC when shuffled)")
    _save(fig, filename)


def coefficient_chart(coefs: pd.Series, filename: str, top: int = 12):
    s = pd.concat([coefs.head(top), coefs.tail(top)]).drop_duplicates()
    colors = ["#2e9e5b" if v < 0 else "#d64545" for v in s.values]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0, color="k", lw=.8)
    ax.set_title("Logistic regression coefficients\n(red = raises default risk, green = lowers it)")
    _save(fig, filename)


def risk_distribution(report: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    order = ["Low", "Medium", "High"]
    counts = report["risk_class"].value_counts().reindex(order)
    axes[0].bar(order, counts.values, color=[PALETTE[o] for o in order])
    axes[0].set_title("Applicants per risk class")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    rate = report.groupby("risk_class")["actual_default"].mean().reindex(order)
    axes[1].bar(order, rate.values, color=[PALETTE[o] for o in order])
    axes[1].set_title("Actual default rate per risk class")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(rate.values):
        axes[1].text(i, v, f"{v:.1%}", ha="center", va="bottom")
    sns.histplot(data=report, x="default_probability", hue="actual_default", bins=40,
                 palette={0: "#4c78a8", 1: "#d64545"}, ax=axes[2])
    axes[2].set_title("Predicted default probability")
    _save(fig, "11_risk_classification.png")


def calibration_plot(y_test, probas: dict):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, p in probas.items():
        frac, mean = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4, label="Perfect")
    ax.set(title="Calibration curve", xlabel="Mean predicted probability", ylabel="Observed default rate")
    ax.legend()
    _save(fig, "12_calibration.png")
