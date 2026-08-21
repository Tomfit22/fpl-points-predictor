"""
FPL Points Predictor — Disagreement-Bucket Validation
========================================================================
Directly answers the original question: when the existing clean sheet
model and the joint expected-goals model disagree, which one is
actually closer to what really happens?

Buckets held-out fixtures by how much the two models disagree, then
computes each model's OWN Brier score within each bucket. If the
existing model still wins even in the high-disagreement bucket, the
"this feels backwards" intuition (e.g. the original Haaland case) is
likely just a mismatch between football intuition and the real
probabilistic relationship - not a genuine model problem. If the joint
model starts winning specifically where they disagree most, that's a
real, usable coherence signal.

Run:
    python build_team_match_table.py       (if not already done)
    python validate_disagreement_buckets.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

TEAM_GOALS_FEATURES = [
    "own_roll5_xG_for", "own_season_xG_for",
    "own_roll5_goals_scored", "own_season_goals_scored",
    "own_season_shots_for",
    "opp_roll5_xG_against", "opp_season_xG_against",
    "opp_roll5_goals_conceded", "opp_season_goals_conceded",
    "opp_season_shots_against",
    "was_home_int",
]

EXISTING_CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
                         "own_season_xG_against", "opp_season_goals_scored",
                         "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
                         "own_season_possession", "opp_season_possession",
                         "own_season_ppda", "opp_season_ppda"]

PRUNE_THRESHOLD = 0.75
BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 100)]


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


def fit_existing_cs(train):
    features = drop_zero_variance(train, [f for f in EXISTING_CS_FEATURES if f in train.columns])
    X = sm.add_constant(train[features].fillna(0))
    model = sm.GLM(train["goals_conceded"], X, family=sm.families.Poisson()).fit()
    return model, features


def fit_joint_goals(train):
    features = drop_zero_variance(train, [f for f in TEAM_GOALS_FEATURES if f in train.columns])
    features = prune_correlated(train, features, PRUNE_THRESHOLD)
    X = sm.add_constant(train[features].fillna(0))
    model = sm.GLM(train["team_goals_scored"], X, family=sm.families.Poisson()).fit()
    return model, features


def predict(model, features, df):
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    return model.predict(X)


def main():
    path = PROCESSED_DIR / "team_match_table.csv"
    if not path.exists():
        print(f"{path} not found - run build_team_match_table.py first.")
        return
    team_df = pd.read_csv(path)

    opp_conceded = team_df[["team", "fixture_id", "goals_conceded"]].rename(
        columns={"team": "opponent_team", "goals_conceded": "team_goals_scored"}
    )
    team_df = team_df.merge(opp_conceded, on=["opponent_team", "fixture_id"], how="left")

    cutoff_gw = team_df["gameweek"].quantile(TRAIN_FRACTION)
    train = team_df[team_df["gameweek"] <= cutoff_gw]
    test = team_df[team_df["gameweek"] > cutoff_gw].copy()

    existing_model, existing_features = fit_existing_cs(train)
    joint_model, joint_features = fit_joint_goals(train)

    test["own_lambda"] = predict(joint_model, joint_features, test)
    opp_lookup = test[["team", "fixture_id", "own_lambda"]].rename(
        columns={"team": "opponent_team", "own_lambda": "opponent_own_lambda"}
    )
    test = test.merge(opp_lookup, on=["opponent_team", "fixture_id"], how="left")
    test["joint_implied_cs"] = np.exp(-test["opponent_own_lambda"])

    X_existing = sm.add_constant(test[existing_features].fillna(0), has_constant="add")
    X_existing = X_existing.reindex(columns=existing_model.params.index, fill_value=0)
    test["existing_cs"] = np.exp(-existing_model.predict(X_existing))

    test["actual_cs"] = (test["goals_conceded"] == 0).astype(int)
    test["diff_pp"] = (test["existing_cs"] - test["joint_implied_cs"]).abs() * 100

    print(f"Test set: {len(test)} team-fixtures\n")
    print("=" * 70)
    print("DISAGREEMENT-BUCKET VALIDATION")
    print("=" * 70)
    print(f"{'Bucket':<12}{'N':>6}{'Existing Brier':>16}{'Joint Brier':>14}{'Winner':>12}")

    for lo, hi in BUCKETS:
        bucket = test[(test["diff_pp"] >= lo) & (test["diff_pp"] < hi)]
        if len(bucket) == 0:
            print(f"{lo}-{hi}pp{'':<6}{'0':>6}  (no fixtures in this range)")
            continue
        brier_existing = ((bucket["existing_cs"] - bucket["actual_cs"]) ** 2).mean()
        brier_joint = ((bucket["joint_implied_cs"] - bucket["actual_cs"]) ** 2).mean()
        winner = "Existing" if brier_existing < brier_joint else "Joint"
        print(f"{lo}-{hi}pp{'':<6}{len(bucket):>6}{brier_existing:>16.4f}{brier_joint:>14.4f}{winner:>12}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    high_disagreement = test[test["diff_pp"] >= 20]
    if len(high_disagreement) >= 5:
        brier_existing_high = ((high_disagreement["existing_cs"] - high_disagreement["actual_cs"]) ** 2).mean()
        brier_joint_high = ((high_disagreement["joint_implied_cs"] - high_disagreement["actual_cs"]) ** 2).mean()
        print(f"In the {len(high_disagreement)} fixtures with 20+ percentage point disagreement:")
        print(f"  Existing model Brier: {brier_existing_high:.4f}")
        print(f"  Joint model Brier:    {brier_joint_high:.4f}")
        if brier_existing_high < brier_joint_high:
            print(f"\n  The existing model STILL wins even where the two models disagree most.")
            print(f"  This suggests the 'feels backwards' intuition (e.g. the original Haaland case)")
            print(f"  is likely a mismatch between football intuition and the real probabilistic")
            print(f"  relationship, not a genuine problem with the existing model.")
        else:
            print(f"\n  The joint model wins specifically in high-disagreement fixtures.")
            print(f"  This IS a genuine, usable coherence signal - worth flagging these cases")
            print(f"  for extra scrutiny even while keeping the existing model as primary.")
    else:
        print(f"Only {len(high_disagreement)} fixtures with 20+ pp disagreement in this test set - "
              f"too few to draw a reliable conclusion. Re-run once more gameweeks of real "
              f"results are available.")


if __name__ == "__main__":
    main()
