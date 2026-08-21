"""
FPL Points Predictor — V1 Calibration Testing
========================================================================
V2 lost on Brier score, but the calibration table revealed something
more useful: V1's predictions in the 40-60% range are systematically
under-confident on real, substantial sample sizes (N=70 and N=73,
not small-sample noise) — predicted ~46-54%, actual clean sheet rate
~61-75%. This tests whether recalibrating V1's OWN predictions (not
replacing its architecture) closes that gap.

Two calibration methods, both fit on TRAINING data only, evaluated on
the SAME untouched held-out test set used throughout today:

  1. Platt scaling — fits a logistic regression of actual outcome on
     the logit of V1's raw predicted probability. Assumes a smooth
     sigmoid relationship between raw and true probability.

  2. Isotonic regression — a flexible, monotonic (non-parametric)
     mapping from raw to calibrated probability. No shape assumption
     beyond "higher raw prediction should mean higher true
     probability", so it can capture the specific under-confidence
     pattern in the 40-60% range directly, whatever its shape.

Compares: raw V1, Platt-calibrated V1, isotonic-calibrated V1 — on
Brier score AND a full calibration table for each, on held-out data.

Run:
    python build_team_match_table.py    (if not already done)
    python calibrate_v1_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75
CALIBRATION_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

V1_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_v1(train):
    features = drop_zero_variance(train, [f for f in V1_FEATURES if f in train.columns])
    X = sm.add_constant(train[features].fillna(0))
    model = sm.GLM(train["goals_conceded"], X, family=sm.families.Poisson()).fit()
    return model, features


def predict_v1_cs(model, features, df):
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    goals_conceded_pred = model.predict(X)
    return np.exp(-goals_conceded_pred)


def brier(pred, actual):
    return ((pred - actual) ** 2).mean()


def calibration_table(pred, actual, label):
    print(f"\n{label} calibration:")
    print(f"{'Predicted range':<18}{'N':>6}{'Avg predicted':>16}{'Actual rate':>14}")
    df = pd.DataFrame({"pred": pred, "actual": actual})
    for i in range(len(CALIBRATION_BINS) - 1):
        lo, hi = CALIBRATION_BINS[i], CALIBRATION_BINS[i + 1]
        bucket = df[(df["pred"] >= lo) & (df["pred"] < hi)]
        if len(bucket) == 0:
            continue
        print(f"{lo:.1f}-{hi:.1f}{'':<12}{len(bucket):>6}{bucket['pred'].mean():>16.3f}{bucket['actual'].mean():>14.3f}")


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main():
    path = PROCESSED_DIR / "team_match_table.csv"
    if not path.exists():
        print(f"{path} not found - run build_team_match_table.py first.")
        return
    team_df = pd.read_csv(path)

    cutoff_gw = team_df["gameweek"].quantile(TRAIN_FRACTION)
    train = team_df[team_df["gameweek"] <= cutoff_gw].copy()
    test = team_df[team_df["gameweek"] > cutoff_gw].copy()
    print(f"Train: {len(train)} team-matches | Test: {len(test)} team-matches\n")

    v1_model, v1_features = fit_v1(train)

    train["v1_cs_raw"] = predict_v1_cs(v1_model, v1_features, train)
    test["v1_cs_raw"] = predict_v1_cs(v1_model, v1_features, test)
    train["actual_cs"] = (train["goals_conceded"] == 0).astype(int)
    test["actual_cs"] = (test["goals_conceded"] == 0).astype(int)

    print("=" * 70)
    print("FITTING CALIBRATION MODELS ON TRAINING DATA ONLY")
    print("=" * 70)

    platt_X_train = logit(train["v1_cs_raw"].values).reshape(-1, 1)
    platt_model = LogisticRegression()
    platt_model.fit(platt_X_train, train["actual_cs"].values)

    platt_X_test = logit(test["v1_cs_raw"].values).reshape(-1, 1)
    test["v1_cs_platt"] = platt_model.predict_proba(platt_X_test)[:, 1]

    print(f"Platt scaling fitted: coefficient={platt_model.coef_[0][0]:.4f}, "
          f"intercept={platt_model.intercept_[0]:.4f}")

    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(train["v1_cs_raw"].values, train["actual_cs"].values)
    test["v1_cs_isotonic"] = iso_model.predict(test["v1_cs_raw"].values)

    print("Isotonic regression fitted.\n")

    print("=" * 70)
    print("RESULTS ON HELD-OUT TEST DATA")
    print("=" * 70)
    brier_raw = brier(test["v1_cs_raw"], test["actual_cs"])
    brier_platt = brier(test["v1_cs_platt"], test["actual_cs"])
    brier_isotonic = brier(test["v1_cs_isotonic"], test["actual_cs"])

    print(f"\nBrier scores (lower is better):")
    print(f"  Raw V1:               {brier_raw:.4f}")
    print(f"  V1 + Platt scaling:   {brier_platt:.4f}")
    print(f"  V1 + Isotonic:        {brier_isotonic:.4f}")

    results = {"Raw V1": brier_raw, "Platt": brier_platt, "Isotonic": brier_isotonic}
    best = min(results, key=results.get)
    print(f"\nBest: {best}")

    calibration_table(test["v1_cs_raw"], test["actual_cs"], "Raw V1")
    calibration_table(test["v1_cs_platt"], test["actual_cs"], "V1 + Platt scaling")
    calibration_table(test["v1_cs_isotonic"], test["actual_cs"], "V1 + Isotonic")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if best != "Raw V1":
        improvement = (brier_raw - results[best]) / brier_raw * 100
        print(f"{best} calibration improves the Brier score by {improvement:.1f}% over raw V1.")
        print("Check the calibration tables above: does the specific 40-60% under-confidence")
        print("gap actually close, not just the single overall number?")
    else:
        print("Neither calibration method improved on raw V1's Brier score.")
        print("The compression/under-confidence pattern may be a real, harder structural")
        print("issue (e.g. genuinely insufficient opponent-attack sensitivity) rather than")
        print("something a simple post-hoc recalibration can fix.")

    out_path = PROCESSED_DIR / "v1_calibration_comparison.csv"
    test[["team", "opponent_team", "gameweek", "v1_cs_raw", "v1_cs_platt",
          "v1_cs_isotonic", "actual_cs"]].to_csv(out_path, index=False)
    print(f"\nFull comparison saved to {out_path}")


if __name__ == "__main__":
    main()
