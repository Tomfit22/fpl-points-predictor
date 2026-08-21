"""
FPL Points Predictor — Multi-Split Calibration Validation Gate
========================================================================
The gate before wiring isotonic calibration into production: does it
still beat raw V1 across MULTIPLE independent temporal train/test
splits, not just the one 75% cutoff used earlier? A single split could
show an improvement that's really specific to that particular
train/test boundary rather than a genuine, robust effect.

Tests several independent cutoffs (55%, 60%, 65%, 70%, 75%, 80% of the
season by gameweek), refitting BOTH the V1 model and the isotonic
calibrator fresh on each split's own training portion, evaluated only
on that split's own held-out portion. Reports whether calibration wins
consistently, not just once.

Run:
    python build_team_match_table.py    (if not already done)
    python validate_calibration_multi_split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

PROCESSED_DIR = Path("data/processed")
CUTOFF_FRACTIONS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

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
    return np.exp(-model.predict(X))


def brier(pred, actual):
    return ((pred - actual) ** 2).mean()


def main():
    path = PROCESSED_DIR / "team_match_table.csv"
    if not path.exists():
        print(f"{path} not found - run build_team_match_table.py first.")
        return
    team_df = pd.read_csv(path)
    print(f"Testing isotonic calibration across {len(CUTOFF_FRACTIONS)} independent "
          f"temporal splits, on {len(team_df)} decontaminated team-fixture rows.\n")

    results = []
    for frac in CUTOFF_FRACTIONS:
        cutoff_gw = team_df["gameweek"].quantile(frac)
        train = team_df[team_df["gameweek"] <= cutoff_gw].copy()
        test = team_df[team_df["gameweek"] > cutoff_gw].copy()

        if len(train) < 50 or len(test) < 20:
            print(f"Split {frac:.0%} (gw<={cutoff_gw:.0f}): skipped - too few rows "
                  f"(train={len(train)}, test={len(test)})")
            continue

        v1_model, v1_features = fit_v1(train)
        train["v1_cs_raw"] = predict_v1_cs(v1_model, v1_features, train)
        test["v1_cs_raw"] = predict_v1_cs(v1_model, v1_features, test)
        train["actual_cs"] = (train["goals_conceded"] == 0).astype(int)
        test["actual_cs"] = (test["goals_conceded"] == 0).astype(int)

        iso_model = IsotonicRegression(out_of_bounds="clip")
        iso_model.fit(train["v1_cs_raw"].values, train["actual_cs"].values)
        test["v1_cs_calibrated"] = iso_model.predict(test["v1_cs_raw"].values)

        brier_raw = brier(test["v1_cs_raw"], test["actual_cs"])
        brier_cal = brier(test["v1_cs_calibrated"], test["actual_cs"])
        pct_change = (brier_raw - brier_cal) / brier_raw * 100
        winner = "Calibrated" if brier_cal < brier_raw else "Raw"

        print(f"Split {frac:.0%} (train={len(train)}, test={len(test)}): "
              f"Raw={brier_raw:.4f}, Calibrated={brier_cal:.4f}, "
              f"change={pct_change:+.1f}% -> {winner} wins")

        results.append({"cutoff": frac, "raw_brier": brier_raw, "calibrated_brier": brier_cal,
                         "pct_change": pct_change, "winner": winner})

    if not results:
        print("\nNo valid splits could be tested.")
        return

    results_df = pd.DataFrame(results)
    n_calibrated_wins = (results_df["winner"] == "Calibrated").sum()
    n_total = len(results_df)
    avg_improvement = results_df["pct_change"].mean()

    print("\n" + "=" * 70)
    print("GATE DECISION")
    print("=" * 70)
    print(f"Calibration won in {n_calibrated_wins}/{n_total} splits.")
    print(f"Average change across all splits: {avg_improvement:+.1f}%")

    if n_calibrated_wins == n_total:
        print("\nCalibration won EVERY split tested. This is a genuinely robust effect -")
        print("safe to proceed with wiring it into the live pipeline.")
    elif n_calibrated_wins >= n_total * 0.7:
        print(f"\nCalibration won most ({n_calibrated_wins}/{n_total}) splits, with a positive")
        print(f"average improvement. Reasonably good evidence, though not unanimous -")
        print(f"worth proceeding but keep an eye on live performance after wiring it in.")
    else:
        print(f"\nCalibration did NOT win consistently across splits ({n_calibrated_wins}/{n_total}).")
        print(f"The earlier 8.9% result may have been specific to that one split rather than")
        print(f"a robust effect. Do NOT wire this into production yet.")


if __name__ == "__main__":
    main()
