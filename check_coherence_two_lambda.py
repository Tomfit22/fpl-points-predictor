"""
FPL Points Predictor — Two-Lambda Coherence Check
========================================================================
Does NOT replace the existing clean sheet model — that model is
temporally sound (see audit_temporal_leakage.py) and already
outperforms the joint expected-goals approach on held-out Brier score
(0.2319 vs 0.2770). This tool instead builds a genuine two-lambda joint
match model purely as a CROSS-CHECK, flagging fixtures where the two
approaches meaningfully disagree — the actual diagnostic originally
wanted, without discarding the model that's currently winning.

The critical relationship, stated explicitly to avoid the earlier
mistake (using a team's own lambda for their own clean sheet):

    For Man City vs Team X:
      lambda_city  = Man City's expected goals (their attack vs Team X's defense)
      lambda_teamx = Team X's expected goals (their attack vs Man City's defense)

      Team X's clean sheet probability = exp(-lambda_city)   <- City must score 0
      Man City's clean sheet probability = exp(-lambda_teamx) <- Team X must score 0

    NEVER exp(-lambda_city) for City's OWN clean sheet — that would be
    "probability City scores zero themselves", a different, irrelevant
    quantity, which is exactly the mistake the earlier coherence-check
    print statement made (the underlying Brier score calculation was
    correct; only that one diagnostic print was wrong).

Run:
    python check_coherence_two_lambda.py
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
MAJOR_DISAGREEMENT_PP = 20
MINOR_DISAGREEMENT_PP = 8


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


def build_team_match_data(df: pd.DataFrame) -> pd.DataFrame:
    team_goals_actual = df.groupby(["team", "fixture_id"])["goals"].sum().reset_index()
    team_goals_actual = team_goals_actual.rename(columns={"goals": "team_goals_scored"})
    team_df = df.groupby(["team", "fixture_id"], as_index=False).first()
    team_df = team_df.merge(team_goals_actual, on=["team", "fixture_id"])
    return team_df


def fit_goals_model(train_team: pd.DataFrame):
    features = drop_zero_variance(train_team, [f for f in TEAM_GOALS_FEATURES if f in train_team.columns])
    features = prune_correlated(train_team, features, PRUNE_THRESHOLD)
    X = sm.add_constant(train_team[features].fillna(0))
    model = sm.GLM(train_team["team_goals_scored"], X, family=sm.families.Poisson()).fit()
    return model, features


def fit_existing_cs_model(train_team: pd.DataFrame):
    features = drop_zero_variance(train_team, [f for f in EXISTING_CS_FEATURES if f in train_team.columns])
    X = sm.add_constant(train_team[features].fillna(0))
    model = sm.GLM(train_team["goals_conceded"], X, family=sm.families.Poisson()).fit()
    return model, features


def predict_with_model(model, features, df):
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    return model.predict(X)


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    required = ["fixture_id", "goals", "goals_conceded", "opponent_team", "gameweek", "team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return

    team_df = build_team_match_data(df)
    cutoff_gw = team_df["gameweek"].quantile(TRAIN_FRACTION)
    train_team = team_df[team_df["gameweek"] <= cutoff_gw]
    test_team = team_df[team_df["gameweek"] > cutoff_gw].copy()

    print(f"Train: {len(train_team)} team-matches | Test: {len(test_team)} team-matches\n")

    goals_model, goals_features = fit_goals_model(train_team)
    existing_model, existing_features = fit_existing_cs_model(train_team)

    print(f"Joint goals model features: {goals_features}")
    print(f"Existing CS model features: {existing_features}\n")

    test_team["own_lambda"] = predict_with_model(goals_model, goals_features, test_team)

    opp_lookup = test_team[["team", "fixture_id", "own_lambda"]].rename(
        columns={"team": "opponent_team", "own_lambda": "opponent_own_lambda"}
    )
    test_team = test_team.merge(opp_lookup, on=["opponent_team", "fixture_id"], how="left")

    test_team["joint_implied_cs"] = np.exp(-test_team["opponent_own_lambda"])

    X_existing = sm.add_constant(test_team[existing_features].fillna(0), has_constant="add")
    X_existing = X_existing.reindex(columns=existing_model.params.index, fill_value=0)
    test_team["pred_goals_conceded_existing"] = existing_model.predict(X_existing)
    test_team["existing_cs"] = np.exp(-test_team["pred_goals_conceded_existing"])

    actual_cs = (test_team["goals_conceded"] == 0).astype(int)
    brier_existing = ((test_team["existing_cs"] - actual_cs) ** 2).mean()
    brier_joint = ((test_team["joint_implied_cs"] - actual_cs) ** 2).mean()
    print(f"(Sanity check, same as before) Existing model Brier: {brier_existing:.4f}, "
          f"Joint-implied Brier: {brier_joint:.4f}\n")

    test_team["diff_pp"] = (test_team["existing_cs"] - test_team["joint_implied_cs"]) * 100

    def flag(diff_pp):
        if abs(diff_pp) >= MAJOR_DISAGREEMENT_PP:
            return "MAJOR"
        elif abs(diff_pp) >= MINOR_DISAGREEMENT_PP:
            return "DISAGREE"
        return "AGREE"

    test_team["flag"] = test_team["diff_pp"].apply(flag)

    print("=" * 70)
    print("COHERENCE COMPARISON - held-out fixtures")
    print("=" * 70)
    display_cols = ["team", "opponent_team", "gameweek", "existing_cs", "joint_implied_cs", "diff_pp", "flag"]
    sorted_by_disagreement = test_team.reindex(test_team["diff_pp"].abs().sort_values(ascending=False).index)
    print(sorted_by_disagreement[display_cols].head(15).to_string(index=False,
          formatters={"existing_cs": "{:.1%}".format, "joint_implied_cs": "{:.1%}".format,
                      "diff_pp": "{:+.1f}".format}))

    n_major = (test_team["flag"] == "MAJOR").sum()
    n_disagree = (test_team["flag"] == "DISAGREE").sum()
    n_agree = (test_team["flag"] == "AGREE").sum()
    print(f"\nSummary: {n_major} MAJOR disagreements, {n_disagree} moderate disagreements, "
          f"{n_agree} broadly agree (out of {len(test_team)} team-fixtures)\n")

    print("=" * 70)
    print("DETAILED EXPLANATION FOR THE TOP DISAGREEMENT")
    print("=" * 70)
    if len(sorted_by_disagreement) > 0:
        top = sorted_by_disagreement.iloc[0]
        print(f"Fixture: {top['team']} vs {top['opponent_team']} (GW{top['gameweek']:.0f})")
        print(f"\nExisting CS model:")
        print(f"  {top['team']} clean sheet probability = {top['existing_cs']:.1%}")
        print(f"\nJoint model:")
        print(f"  {top['opponent_team']} expected goals (lambda) = {top['opponent_own_lambda']:.2f}")
        print(f"  Implied {top['team']} clean sheet = exp(-{top['opponent_own_lambda']:.2f}) = {top['joint_implied_cs']:.1%}")
        print(f"\nDifference: {top['diff_pp']:+.1f} percentage points")
        direction = "more optimistic" if top['diff_pp'] > 0 else "more pessimistic"
        print(f"\nInterpretation: the existing CS model is substantially {direction} about "
              f"{top['team']} keeping a clean sheet than the joint goals model implies.")


if __name__ == "__main__":
    main()
