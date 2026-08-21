"""
FPL Points Predictor — CS Model V2: Two-Lambda Match Model
========================================================================
Phases 1-6 of the CS Model V2 project, run as one comprehensive script.
Phase 7 (replace V1) is NOT automated — that decision needs a human
looking at the results below, on purpose.

Builds a genuinely symmetric two-lambda match model, applying the real
lessons validated earlier today rather than repeating the earlier
rushed prototype:
  - Season-based features, not rolling — rolling features were tested
    and lost on real held-out data earlier today.
  - Correctly-oriented opponent DEFENSIVE weakness features (the
    GOALS_FEATURES bug found earlier — opp_season_shots_for vs
    opp_season_shots_against — is deliberately avoided here).
  - Correlation pruning at the same threshold proven to work for the
    existing CS model (0.75).
  - Fit on the DECONTAMINATED team_match_table.csv, not the raw
    player-level data (avoiding the stale-team-label bug found and
    fixed earlier today).

PHASE 1: Load the fixture-level dataset (already symmetric via the
         own_/opp_ prefixes in team_match_table.csv).
PHASE 2: Fit the two-lambda expected-goals model.
PHASE 3: Derive clean sheet probability from the OPPONENT's lambda.
PHASE 4: Compare V1 (existing CS model) vs V2 (this model) on
         genuine held-out Brier score, MAE, and home/away splits.
PHASE 5: Calibration check — for both models, bucket predictions by
         probability and compare against actual clean-sheet rate.
PHASE 6: Coherence/disagreement-bucket testing, same methodology as
         earlier today, applied to this properly-built V2.

Run:
    python build_team_match_table.py    (if not already done)
    python build_cs_model_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75
PRUNE_THRESHOLD = 0.75
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 100)]
CALIBRATION_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

V2_FEATURES = [
    "own_season_xG_for", "own_season_goals_scored", "own_season_shots_for",
    "opp_season_xG_against", "opp_season_goals_conceded", "opp_season_shots_against",
    "was_home_int",
]

V1_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def prune_correlated(df, features, threshold):
    pruned = []
    for f in features:
        too_similar = any(abs(df[f].corr(df[g])) > threshold for g in pruned)
        if not too_similar:
            pruned.append(f)
    return pruned


def fit_v1(train):
    features = drop_zero_variance(train, [f for f in V1_FEATURES if f in train.columns])
    X = sm.add_constant(train[features].fillna(0))
    model = sm.GLM(train["goals_conceded"], X, family=sm.families.Poisson()).fit()
    return model, features


def fit_v2(train):
    features = drop_zero_variance(train, [f for f in V2_FEATURES if f in train.columns])
    features = prune_correlated(train, features, PRUNE_THRESHOLD)
    X = sm.add_constant(train[features].fillna(0))
    model = sm.GLM(train["team_goals_scored"], X, family=sm.families.Poisson()).fit()
    return model, features


def predict(model, features, df):
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    return model.predict(X)


def brier(pred, actual):
    return ((pred - actual) ** 2).mean()


def mae(pred, actual):
    return (pred - actual).abs().mean()


def calibration_table(pred_col, actual_col, df, label):
    print(f"\n{label} calibration:")
    print(f"{'Predicted range':<18}{'N':>6}{'Avg predicted':>16}{'Actual rate':>14}")
    for i in range(len(CALIBRATION_BINS) - 1):
        lo, hi = CALIBRATION_BINS[i], CALIBRATION_BINS[i + 1]
        bucket = df[(df[pred_col] >= lo) & (df[pred_col] < hi)]
        if len(bucket) == 0:
            continue
        avg_pred = bucket[pred_col].mean()
        actual_rate = bucket[actual_col].mean()
        print(f"{lo:.1f}-{hi:.1f}{'':<12}{len(bucket):>6}{avg_pred:>16.3f}{actual_rate:>14.3f}")


def main():
    print("=" * 70)
    print("PHASE 1: LOAD THE DECONTAMINATED FIXTURE-LEVEL DATASET")
    print("=" * 70)
    path = PROCESSED_DIR / "team_match_table.csv"
    if not path.exists():
        print(f"{path} not found - run build_team_match_table.py first.")
        return
    team_df = pd.read_csv(path)
    print(f"Loaded {len(team_df)} decontaminated team-fixture rows.\n")

    opp_conceded = team_df[["team", "fixture_id", "goals_conceded"]].rename(
        columns={"team": "opponent_team", "goals_conceded": "team_goals_scored"}
    )
    team_df = team_df.merge(opp_conceded, on=["opponent_team", "fixture_id"], how="left")

    cutoff_gw = team_df["gameweek"].quantile(TRAIN_FRACTION)
    train = team_df[team_df["gameweek"] <= cutoff_gw]
    test = team_df[team_df["gameweek"] > cutoff_gw].copy()
    print(f"Train: {len(train)} team-matches | Test: {len(test)} team-matches\n")

    print("=" * 70)
    print("PHASE 2: FIT THE TWO-LAMBDA EXPECTED-GOALS MODEL (V2)")
    print("=" * 70)
    v1_model, v1_features = fit_v1(train)
    v2_model, v2_features = fit_v2(train)

    print(f"V2 features after pruning: {v2_features}\n")
    for f in v2_features:
        coef = v2_model.params.get(f)
        pval = v2_model.pvalues.get(f)
        sig = "significant" if pval < 0.05 else "not significant"
        print(f"  {f}: coef={coef:+.4f}, p={pval:.4f} ({sig})")

    print("\n" + "=" * 70)
    print("PHASE 3: DERIVE CLEAN SHEET PROBABILITY FROM OPPONENT'S LAMBDA")
    print("=" * 70)
    test["v2_own_lambda"] = predict(v2_model, v2_features, test)
    opp_lookup = test[["team", "fixture_id", "v2_own_lambda"]].rename(
        columns={"team": "opponent_team", "v2_own_lambda": "opponent_own_lambda"}
    )
    test = test.merge(opp_lookup, on=["opponent_team", "fixture_id"], how="left")
    test["v2_cs"] = np.exp(-test["opponent_own_lambda"])

    X_v1 = sm.add_constant(test[v1_features].fillna(0), has_constant="add")
    X_v1 = X_v1.reindex(columns=v1_model.params.index, fill_value=0)
    test["v1_goals_conceded"] = v1_model.predict(X_v1)
    test["v1_cs"] = np.exp(-test["v1_goals_conceded"])

    test["actual_cs"] = (test["goals_conceded"] == 0).astype(int)
    print("Done - v1_cs and v2_cs computed for every held-out fixture.\n")

    print("=" * 70)
    print("PHASE 4: COMPARE V1 vs V2 - OVERALL AND BY HOME/AWAY")
    print("=" * 70)
    brier_v1 = brier(test["v1_cs"], test["actual_cs"])
    brier_v2 = brier(test["v2_cs"], test["actual_cs"])
    print(f"\nOverall Brier score:")
    print(f"  V1 (existing): {brier_v1:.4f}")
    print(f"  V2 (two-lambda): {brier_v2:.4f}")
    if brier_v2 < brier_v1:
        print(f"  -> V2 WINS by {(brier_v1-brier_v2)/brier_v1*100:.1f}%")
    else:
        print(f"  -> V1 WINS by {(brier_v2-brier_v1)/brier_v1*100:.1f}%")

    for wh, label in [(1, "HOME"), (0, "AWAY")]:
        subset = test[test["was_home_int"] == wh]
        if len(subset) == 0:
            continue
        b1 = brier(subset["v1_cs"], subset["actual_cs"])
        b2 = brier(subset["v2_cs"], subset["actual_cs"])
        winner = "V1" if b1 < b2 else "V2"
        print(f"\n{label} fixtures (n={len(subset)}): V1={b1:.4f}, V2={b2:.4f} -> {winner} wins")

    print("\n" + "=" * 70)
    print("PHASE 5: CALIBRATION CHECK")
    print("=" * 70)
    calibration_table("v1_cs", "actual_cs", test, "V1 (existing)")
    calibration_table("v2_cs", "actual_cs", test, "V2 (two-lambda)")

    print("\n" + "=" * 70)
    print("PHASE 6: DISAGREEMENT-BUCKET / COHERENCE TESTING")
    print("=" * 70)
    test["diff_pp"] = (test["v1_cs"] - test["v2_cs"]).abs() * 100
    print(f"\n{'Bucket':<12}{'N':>6}{'V1 Brier':>12}{'V2 Brier':>12}{'Winner':>10}")
    for lo, hi in BUCKETS:
        bucket = test[(test["diff_pp"] >= lo) & (test["diff_pp"] < hi)]
        if len(bucket) == 0:
            print(f"{lo}-{hi}pp{'':<6}{'0':>6}  (no fixtures)")
            continue
        b1 = brier(bucket["v1_cs"], bucket["actual_cs"])
        b2 = brier(bucket["v2_cs"], bucket["actual_cs"])
        winner = "V1" if b1 < b2 else "V2"
        print(f"{lo}-{hi}pp{'':<6}{len(bucket):>6}{b1:>12.4f}{b2:>12.4f}{winner:>10}")

    print("\n" + "=" * 70)
    print("SUMMARY - PHASE 7 DECISION IS YOURS, NOT AUTOMATED")
    print("=" * 70)
    print(f"Overall: V1={brier_v1:.4f} vs V2={brier_v2:.4f}")
    if brier_v2 < brier_v1:
        print("V2 shows a genuine overall improvement on this held-out test set.")
        print("Before replacing V1, check: does V2 also win in the calibration table above")
        print("(not just the single overall number), and does the improvement hold up")
        print("as more real gameweeks of results become available?")
    else:
        print("V1 remains the stronger model overall on this test set.")
        print("Keep V1 as primary. V2 may still be useful as a coherence check")
        print("(see the disagreement-bucket results above) even without replacing V1.")

    out_path = PROCESSED_DIR / "cs_model_v2_comparison.csv"
    test[["team", "opponent_team", "gameweek", "v1_cs", "v2_cs", "actual_cs", "diff_pp"]].to_csv(out_path, index=False)
    print(f"\nFull fixture-level comparison saved to {out_path}")


if __name__ == "__main__":
    main()
