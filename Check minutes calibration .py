"""
FPL Points Predictor — Minutes Probability Calibration Check
==================================================================
Diagnoses the bottom-bucket overprediction found in the pipeline's
bucketed bias check: a large cluster of low-probability players (P any
minutes, P 60+) predicted around 0.35-0.38 points averaged a true
outcome of just 0.031 — a 12x overprediction. Since bonus is already
well-calibrated for this group, this specifically tests whether
pred_p_any_minutes / pred_p_60plus themselves are overestimating
low-end playing-time chances, the same way we tested clean sheet
probability calibration earlier.

Run:
    python check_minutes_calibration.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75


def drop_zero_variance(df: pd.DataFrame, features: list) -> list:
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_and_check(train: pd.DataFrame, test: pd.DataFrame, threshold: int, label: str):
    features = ["roll5_starts", "consecutive_starts", "days_since_last_game", "roll5_minutes"]
    features = [f for f in features if f in train.columns]
    features = drop_zero_variance(train, features)

    X_train = sm.add_constant(train[features].fillna(0))
    y_train = (train["minutes"] >= threshold).astype(int)
    model = sm.Logit(y_train, X_train).fit(disp=0)

    X_test = sm.add_constant(test[features].fillna(0), has_constant="add")
    y_test = (test["minutes"] >= threshold).astype(int)
    pred_proba = model.predict(X_test)

    print(f"\n{'=' * 70}")
    print(f"P(minutes >= {threshold}) — {label}")
    print(f"{'=' * 70}")
    print("Calibration curve (predicted vs actual rate per bin):")
    prob_true, prob_pred = calibration_curve(y_test, pred_proba, n_bins=10, strategy="quantile")
    for pt, pp in zip(prob_true, prob_pred):
        direction = "OVERCONFIDENT" if pp > pt + 0.05 else ("UNDERCONFIDENT" if pp < pt - 0.05 else "reasonable")
        print(f"  predicted ~{pp:.3f} -> actual rate {pt:.3f}  [{direction}]")

    # specifically zoom in on the LOW end, since that's where the pipeline
    # bug was found — the overall calibration curve could look fine on
    # average while still being badly wrong at the bottom
    low_end = test[pred_proba < 0.3].copy()
    low_end["pred"] = pred_proba[pred_proba < 0.3]
    if len(low_end) > 0:
        print(f"\nLOW-END ZOOM (predicted probability < 0.3, {len(low_end)} rows):")
        print(f"  Mean predicted probability: {low_end['pred'].mean():.3f}")
        print(f"  Actual rate of minutes >= {threshold}: {(low_end['minutes'] >= threshold).mean():.3f}")


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[df["roll5_minutes"].notna()]

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw]
    print(f"Train: {len(train)} | Test: {len(test)}")

    fit_and_check(train, test, threshold=1, label="any involvement")
    fit_and_check(train, test, threshold=60, label="60+ minutes")


if __name__ == "__main__":
    main()