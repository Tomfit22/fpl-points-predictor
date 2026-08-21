"""
FPL Points Predictor — Final Calibration Method Comparison
========================================================================
The last validation before finishing CS V1.1: compares four approaches
across the same 6 independent temporal splits used before, settling
whether isotonic's small edge over Platt (0.2184 vs 0.2199, a 0.0015
Brier difference) is worth its more aggressive, occasionally-extreme
mapping (raw 80% -> literal 100%), or whether a smoother, more
conservative method should be preferred instead.

  1. Raw V1 — no calibration.
  2. Platt scaling — smooth sigmoid, no hard plateaus.
  3. Isotonic (unclipped) — flexible, can map to literal 0.0/1.0.
  4. Isotonic (clipped to [0.05, 0.95]) — same flexible mapping in the
     middle, but never claims literal certainty at the extremes.

All four fit fresh on each split's own training portion, evaluated
only on that split's own held-out portion — no leakage across splits.

Run:
    python build_team_match_table.py    (if not already done)
    python compare_calibration_methods.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

PROCESSED_DIR = Path("data/processed")
CUTOFF_FRACTIONS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CLIP_MIN = 0.05
CLIP_MAX = 0.95

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


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier(pred, actual):
    return ((pred - actual) ** 2).mean()


def main():
    path = PROCESSED_DIR / "team_match_table.csv"
    if not path.exists():
        print(f"{path} not found - run build_team_match_table.py first.")
        return
    team_df = pd.read_csv(path)
    print(f"Comparing 4 calibration approaches across {len(CUTOFF_FRACTIONS)} independent "
          f"temporal splits, on {len(team_df)} decontaminated team-fixture rows.\n")

    results = []
    for frac in CUTOFF_FRACTIONS:
        cutoff_gw = team_df["gameweek"].quantile(frac)
        train = team_df[team_df["gameweek"] <= cutoff_gw].copy()
        test = team_df[team_df["gameweek"] > cutoff_gw].copy()

        if len(train) < 50 or len(test) < 20:
            print(f"Split {frac:.0%}: skipped - too few rows")
            continue

        v1_model, v1_features = fit_v1(train)
        train["v1_cs_raw"] = predict_v1_cs(v1_model, v1_features, train)
        test["v1_cs_raw"] = predict_v1_cs(v1_model, v1_features, test)
        train["actual_cs"] = (train["goals_conceded"] == 0).astype(int)
        test["actual_cs"] = (test["goals_conceded"] == 0).astype(int)

        platt = LogisticRegression()
        platt.fit(logit(train["v1_cs_raw"].values).reshape(-1, 1), train["actual_cs"].values)
        test["cs_platt"] = platt.predict_proba(logit(test["v1_cs_raw"].values).reshape(-1, 1))[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train["v1_cs_raw"].values, train["actual_cs"].values)
        test["cs_isotonic"] = iso.predict(test["v1_cs_raw"].values)
        test["cs_isotonic_clipped"] = np.clip(test["cs_isotonic"], CLIP_MIN, CLIP_MAX)

        b_raw = brier(test["v1_cs_raw"], test["actual_cs"])
        b_platt = brier(test["cs_platt"], test["actual_cs"])
        b_iso = brier(test["cs_isotonic"], test["actual_cs"])
        b_iso_clip = brier(test["cs_isotonic_clipped"], test["actual_cs"])

        print(f"Split {frac:.0%} (train={len(train)}, test={len(test)}):")
        print(f"  Raw={b_raw:.4f}  Platt={b_platt:.4f}  Isotonic={b_iso:.4f}  "
              f"Isotonic-clipped={b_iso_clip:.4f}")

        results.append({"cutoff": frac, "raw": b_raw, "platt": b_platt,
                         "isotonic": b_iso, "isotonic_clipped": b_iso_clip})

    if not results:
        print("\nNo valid splits could be tested.")
        return

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL SPLITS")
    print("=" * 70)
    for method in ["raw", "platt", "isotonic", "isotonic_clipped"]:
        wins = sum(results_df[method] == results_df[["raw", "platt", "isotonic", "isotonic_clipped"]].min(axis=1))
        print(f"  {method:<20} avg Brier: {results_df[method].mean():.4f}  wins: {wins}/{len(results_df)}")

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: PLATT vs ISOTONIC-CLIPPED")
    print("=" * 70)
    platt_wins = (results_df["platt"] < results_df["isotonic_clipped"]).sum()
    iso_clip_wins = (results_df["isotonic_clipped"] < results_df["platt"]).sum()
    print(f"Platt wins: {platt_wins}/{len(results_df)}")
    print(f"Isotonic-clipped wins: {iso_clip_wins}/{len(results_df)}")
    avg_diff = (results_df["platt"] - results_df["isotonic_clipped"]).mean()
    print(f"Average Brier difference (Platt - Isotonic-clipped): {avg_diff:+.4f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    if abs(avg_diff) < 0.005:
        print("The two methods are essentially equivalent on this data.")
        print("Given Platt's smoother, more interpretable, less extreme behavior,")
        print("Platt scaling is the recommended choice for production.")
    elif avg_diff > 0:
        print(f"Isotonic-clipped meaningfully outperforms Platt by {avg_diff:.4f} Brier on average.")
        print("Worth keeping isotonic (clipped) despite its less smooth shape.")
    else:
        print(f"Platt meaningfully outperforms Isotonic-clipped by {-avg_diff:.4f} Brier on average.")
        print("Platt scaling is the clear choice for production.")


if __name__ == "__main__":
    main()
