"""
FPL Points Predictor — Final Comparison on Decontaminated Data
========================================================================
Reruns ONLY the comparison that matters: existing clean sheet model
vs. two-lambda joint expected-goals model, this time on
team_match_table.csv — the validated, decontaminated team-level data
(built by build_team_match_table.py), instead of the naive
df.groupby(["team","fixture_id"]).first() every earlier script used
today, which was vulnerable to the confirmed stale-team-label bug
(~13-15 transferred players, 321 affected rows, creating phantom
team-fixture entries).

Run:
    python build_team_match_table.py   (first, if not already done)
    python rerun_clean_sheet_comparison.py
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
    print(f"Train: {len(train)} team-matches | Test: {len(test)} team-matches "
          f"(on DECONTAMINATED data - {len(team_df)} total, no phantom rows)\n")

    existing_model, existing_features = fit_existing_cs(train)
    joint_model, joint_features = fit_joint_goals(train)

    print(f"Existing CS model features: {existing_features}")
    print(f"Joint goals model features: {joint_features}\n")

    for f in joint_features:
        coef = joint_model.params.get(f)
        pval = joint_model.pvalues.get(f)
        sig = "significant" if pval < 0.05 else "not significant"
        print(f"  {f}: coef={coef:+.4f}, p={pval:.4f} ({sig})")
    print()

    test["own_lambda"] = predict(joint_model, joint_features, test)
    opp_lookup = test[["team", "fixture_id", "own_lambda"]].rename(
        columns={"team": "opponent_team", "own_lambda": "opponent_own_lambda"}
    )
    test = test.merge(opp_lookup, on=["opponent_team", "fixture_id"], how="left")
    test["joint_implied_cs"] = np.exp(-test["opponent_own_lambda"])

    X_existing = sm.add_constant(test[existing_features].fillna(0), has_constant="add")
    X_existing = X_existing.reindex(columns=existing_model.params.index, fill_value=0)
    test["pred_goals_conceded_existing"] = existing_model.predict(X_existing)
    test["existing_cs"] = np.exp(-test["pred_goals_conceded_existing"])

    actual_cs = (test["goals_conceded"] == 0).astype(int)
    brier_existing = ((test["existing_cs"] - actual_cs) ** 2).mean()
    brier_joint = ((test["joint_implied_cs"] - actual_cs) ** 2).mean()

    print("=" * 70)
    print("FINAL RESULT - ON DECONTAMINATED DATA")
    print("=" * 70)
    print(f"Existing CS model Brier: {brier_existing:.4f}")
    print(f"Joint-implied Brier:     {brier_joint:.4f}")
    if brier_joint < brier_existing:
        pct = (brier_existing - brier_joint) / brier_existing * 100
        print(f"\nJoint model is now BETTER by {pct:.1f}%")
    else:
        pct = (brier_joint - brier_existing) / brier_existing * 100
        print(f"\nExisting model is still BETTER by {pct:.1f}%")

    print(f"\n(For comparison: on the earlier, contaminated data, the result was "
          f"Existing=0.2319, Joint=0.2770 - Joint worse by 19.5%)")


if __name__ == "__main__":
    main()
