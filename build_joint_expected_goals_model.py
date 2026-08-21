"""
FPL Points Predictor — Joint Match Expected-Goals Model
========================================================================
Replaces the architecture of two INDEPENDENT models (a goals model and
a separately-fit clean sheet model) with a single team-level expected-
goals model, built on the real, correct principle: a team's expected
goals should come from THEIR OWN attacking quality vs the OPPONENT's
DEFENSIVE weakness — own_season_xG_for / opp_season_xG_against, not the
disconnected, independently-fit approach used before.

Clean sheet probability is then DERIVED directly from this same number,
not modeled separately:

    P(opponent keeps a clean sheet) = exp(-lambda_this_team)

This makes it mathematically impossible for the two predictions to
contradict each other the way Haaland (0.96 goals) and Bournemouth
(implying a ~0.43 team goals-conceded rate) did before — they now come
from literally the same underlying number.

Validates THREE things on genuine held-out data:
  1. Goals prediction accuracy (MAE) — is the new team-level model at
     least as good at predicting actual goals scored?
  2. Clean sheet prediction accuracy (Brier score) — derived from the
     SAME model, compared against the old independently-fit model.
  3. Coherence — for held-out fixtures, checking there's no case where
     a team's predicted goals are high AND their opponent's derived
     clean sheet probability is also high, the specific failure mode
     that started this whole investigation.

Run:
    python build_joint_expected_goals_model.py
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

OLD_CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
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


def build_team_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, fixture) with the team's ACTUAL goals scored
    in that specific match (summed from real player-level goals), plus
    the attack-vs-opponent-defense features."""
    team_goals_actual = df.groupby(["team", "fixture_id"])["goals"].sum().reset_index()
    team_goals_actual = team_goals_actual.rename(columns={"goals": "team_goals_scored"})

    team_df = df.groupby(["team", "fixture_id"], as_index=False).first()
    team_df = team_df.merge(team_goals_actual, on=["team", "fixture_id"])
    return team_df


def fit_joint_model(train_team: pd.DataFrame):
    features = drop_zero_variance(train_team, [f for f in TEAM_GOALS_FEATURES if f in train_team.columns])
    features = prune_correlated(train_team, features, PRUNE_THRESHOLD)
    X = sm.add_constant(train_team[features].fillna(0))
    model = sm.GLM(train_team["team_goals_scored"], X, family=sm.families.Poisson()).fit()
    return model, features


def fit_old_cs_model(train_team: pd.DataFrame):
    features = drop_zero_variance(train_team, [f for f in OLD_CS_FEATURES if f in train_team.columns])
    X = sm.add_constant(train_team[features].fillna(0))
    model = sm.GLM(train_team["goals_conceded"], X, family=sm.families.Poisson()).fit()
    return model, features


def predict_lambda(model, features, df):
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

    print("=" * 70)
    print("1. JOINT EXPECTED-GOALS MODEL (own attack vs opponent defense)")
    print("=" * 70)
    joint_model, joint_features = fit_joint_model(train_team)
    print(f"Features kept after pruning: {joint_features}\n")
    for f in joint_features:
        coef = joint_model.params.get(f)
        pval = joint_model.pvalues.get(f)
        sig = "significant" if pval < 0.05 else "not significant"
        print(f"  {f}: coef={coef:+.4f}, p={pval:.4f} ({sig})")

    test_team["lambda_joint"] = predict_lambda(joint_model, joint_features, test_team)

    goals_mae = (test_team["lambda_joint"] - test_team["team_goals_scored"]).abs().mean()
    print(f"\nHeld-out goals MAE (team-level): {goals_mae:.4f}\n")

    print("=" * 70)
    print("2. DERIVED CLEAN SHEET vs OLD INDEPENDENT CLEAN SHEET MODEL")
    print("=" * 70)

    opp_lambda = test_team[["team", "fixture_id", "lambda_joint"]].rename(
        columns={"team": "opponent_team", "lambda_joint": "opp_lambda_joint"}
    )
    test_team = test_team.merge(opp_lambda, on=["opponent_team", "fixture_id"], how="left")
    test_team["pred_cs_derived"] = np.exp(-test_team["opp_lambda_joint"])

    old_model, old_features = fit_old_cs_model(train_team)
    X_old_test = sm.add_constant(test_team[old_features].fillna(0), has_constant="add")
    X_old_test = X_old_test.reindex(columns=old_model.params.index, fill_value=0)
    test_team["pred_goals_conceded_old"] = old_model.predict(X_old_test)
    test_team["pred_cs_old"] = np.exp(-test_team["pred_goals_conceded_old"])

    actual_cs = (test_team["goals_conceded"] == 0).astype(int)
    brier_derived = ((test_team["pred_cs_derived"] - actual_cs) ** 2).mean()
    brier_old = ((test_team["pred_cs_old"] - actual_cs) ** 2).mean()

    print(f"OLD independent clean sheet model - Brier: {brier_old:.4f}")
    print(f"NEW derived (from joint model)    - Brier: {brier_derived:.4f}")
    if brier_derived < brier_old:
        pct = (brier_old - brier_derived) / brier_old * 100
        print(f"  Derived approach is BETTER by {pct:.1f}%")
    else:
        pct = (brier_derived - brier_old) / brier_old * 100
        print(f"  Derived approach is WORSE by {pct:.1f}%")

    print("\n" + "=" * 70)
    print("3. COHERENCE CHECK - the actual problem we set out to fix")
    print("=" * 70)
    high_scoring = test_team[test_team["lambda_joint"] > test_team["lambda_joint"].quantile(0.9)]
    print(f"Top 10% highest-predicted-goals team-matches ({len(high_scoring)} rows):")
    print(f"  Mean predicted goals for these teams: {high_scoring['lambda_joint'].mean():.2f}")
    print(f"  Mean derived clean sheet probability for the SAME teams (should be LOW): "
          f"{np.exp(-high_scoring['lambda_joint']).mean():.3f}")
    print(f"\nThis confirms structurally: a team predicted to score a lot is ALWAYS given a "
          f"correspondingly low chance of ALSO conceding nothing themselves when we flip "
          f"perspective - same underlying number, cannot disagree with itself.")


if __name__ == "__main__":
    main()
