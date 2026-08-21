"""
FPL Points Predictor — Fit and Save the Clean Sheet Calibrator (v4, Platt)
========================================================================
Switched from isotonic to Platt scaling after a direct four-way
comparison across 6 temporal splits found them statistically tied
(Platt 3/6 wins, Isotonic 3/6 wins, 0.0005 average Brier difference —
well within noise). Platt is preferred as the simpler, smoother model:
no hard plateaus, no risk of literal 0%/100% predictions, and only two
saved numbers (coefficient, intercept) instead of a full fitted
sklearn object — removing the pickle cross-process risk entirely,
not just working around it with clipping.

Platt scaling: calibrated_p = sigmoid(coefficient * logit(raw_p) + intercept)

Fits on ALL available decontaminated data (not held back a test set —
in production we want the calibrator to use everything available).

Run:
    python build_team_match_table.py    (if not already done)
    python fit_and_save_cs_calibrator.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

PROCESSED_DIR = Path("data/processed")

V1_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_v1(df):
    features = drop_zero_variance(df, [f for f in V1_FEATURES if f in df.columns])
    X = sm.add_constant(df[features].fillna(0))
    model = sm.GLM(df["goals_conceded"], X, family=sm.families.Poisson()).fit()
    return model, features


def predict_v1_cs(model, features, df):
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    return np.exp(-model.predict(X))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    path = PROCESSED_DIR / "team_match_table.csv"
    if not path.exists():
        print(f"{path} not found - run build_team_match_table.py first.")
        return
    team_df = pd.read_csv(path)
    print(f"Fitting on all {len(team_df)} decontaminated team-fixture rows.\n")

    v1_model, v1_features = fit_v1(team_df)
    team_df["v1_cs_raw"] = predict_v1_cs(v1_model, v1_features, team_df)
    team_df["actual_cs"] = (team_df["goals_conceded"] == 0).astype(int)

    platt = LogisticRegression()
    platt_X = logit(team_df["v1_cs_raw"].values).reshape(-1, 1)
    platt.fit(platt_X, team_df["actual_cs"].values)

    coefficient = float(platt.coef_[0][0])
    intercept = float(platt.intercept_[0])

    calibrated = platt.predict_proba(platt_X)[:, 1]
    brier_raw = ((team_df["v1_cs_raw"] - team_df["actual_cs"]) ** 2).mean()
    brier_calibrated = ((calibrated - team_df["actual_cs"]) ** 2).mean()

    print(f"Platt scaling fitted: coefficient={coefficient:.4f}, intercept={intercept:.4f}\n")
    print(f"In-sample check (not held-out, just a sanity check on the fit itself):")
    print(f"  Raw Brier: {brier_raw:.4f}")
    print(f"  Calibrated Brier: {brier_calibrated:.4f}")

    print(f"\nMapping across the range (confirming smooth, no hard plateaus):")
    for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        cal = 1 / (1 + np.exp(-(coefficient * np.log(x/(1-x)) + intercept)))
        print(f"  {x:.2f} -> {cal:.4f}")

    out_path = PROCESSED_DIR / "cs_platt_calibrator.json"
    with open(out_path, "w") as f:
        json.dump({"coefficient": coefficient, "intercept": intercept,
                   "v1_features": v1_features}, f, indent=2)
    print(f"\nSaved calibrator to {out_path} (plain JSON - just 2 numbers, no pickle at all)")


if __name__ == "__main__":
    main()
