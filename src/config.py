"""Central configuration for the loan risk project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "loan_applications.csv"
TARGET = "default"
RANDOM_STATE = 42

# Risk tiers based on predicted probability of default
LOW_RISK_MAX = 0.20
MEDIUM_RISK_MAX = 0.50

for d in (DATA_DIR, PLOT_DIR, REPORT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
