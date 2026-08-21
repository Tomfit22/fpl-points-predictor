"""
FPL Points Predictor — Clean Sheet Model: Cross-Model Opponent Strength
========================================================================
A structurally different fix from everything else tried today (which
was all reshuffling/pruning the SAME static season-average features,
and repeatedly failed on held-out validation). Instead, this uses the
GOALS model's OWN validated predictions as the opponent-strength signal
for the CLEAN SHEET model — a genuinely more informative input than any
single raw stat, since it's the combined output of an entire fitted
model (each opposing player's own xG, shots, home/away, penalty-taker
status), not just one number.

Correctness, addressed directly: this is STRICTLY about the OPPONENT's
attacking strength, never the defending team's own record leaking in
backwards. Built as:

  1. Fit the (already-validated) GOALS model on TRAINING data only.
  2. For every team+fixture, sum that team's players' predicted goals
     using the TRAIN-fitted model — this is genuinely non-circular in
     practice, since the goals model's own opponent-dependency
     (opp_season_shots_for) was already confirmed weak/not significant
     in the prior validation.
  3. Attach the OPPONENT's summed predicted goals (not the team's own)
     as a new candidate feature for the clean sheet model.
  4. Compare against the current clean sheet model on genuine held-out
     data — same rigor as every other test today.

Run:
    python validate_cross_model_clean_sheet.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

GOALS_FEATURES = ["season_xG", "roll5_xG", "season_shots", "roll5_shots",
                   "opp_season_shots_for", "was_home_int", "is_primary_pen_taker"]
CS_FEATURES_CURRENT = ["own_season_goals_conceded", "own_roll5_goals_conceded",
                        "own_season_xG_against", "opp_season_goals_scored",
                        "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
                        "own_season_possession", "opp_season_possession",
                        "own_season_ppda", "opp_season_ppda"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_goals_model(train_df: pd.DataFrame):
    """Fits the CURRENT, already-validated goals model — per position,
    same as the live pipeline — on TRAINING data only."""
    models = {}
    for position in ["DEF", "MID", "FWD"]:
        pos_df = train_df[train_df["position"] == position]
        features = drop_zero_variance(pos_df, [f for f in GOALS_FEATURES if f in pos_df.columns])
        if len(pos_df) < 60 or not features:
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            models[position] = (sm.GLM(pos_df["goals"], X, family=sm.families.Poisson()).fit(), features)
        except Exception as e:
            print(f"  Goals model fit failed for {position}: {e}")
    return models


def predict_goals_for_all(df: pd.DataFrame, goals_models: dict) -> pd.Series:
    preds = pd.Series(0.0, index=df.index)
    for position, (model, features) in goals_models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
        X = X.reindex(columns=model.params.index, fill_value=0)
        preds.loc[mask] = model.predict(X)
    return preds


def build_team_predicted_goals(df: pd.DataFrame, goals_models: dict) -> pd.DataFrame:
    """Sums predicted goals per (team, fixture) — GK predictions default
    to 0 since they're not in the goals model, correctly contributing
    nothing to their own team's attacking sum."""
    df = df.copy()
    df["_pred_goals_for_sum"] = 0.0
    outfield_mask = df["position"].isin(["DEF", "MID", "FWD"])
    df.loc[outfield_mask, "_pred_goals_for_sum"] = predict_goals_for_all(df[outfield_mask], goals_models)

    team_goals = df.groupby(["team", "fixture_id"])["_pred_goals_for_sum"].sum().reset_index()
    team_goals = team_goals.rename(columns={"_pred_goals_for_sum": "team_predicted_goals"})
    return team_goals


def fit_and_evaluate_cs(features, train_team, test_team):
    features = drop_zero_variance(train_team, features)
    if not features:
        return None

    X_train = sm.add_constant(train_team[features].fillna(0))
    model = sm.GLM(train_team["goals_conceded"], X_train, family=sm.families.Poisson()).fit()

    X_test = sm.add_constant(test_team[features].fillna(0), has_constant="add")
    X_test = X_test.reindex(columns=model.params.index, fill_value=0)
    pred_goals_conceded = model.predict(X_test)
    pred_cs = np.exp(-pred_goals_conceded)
    actual_cs = (test_team["goals_conceded"] == 0).astype(int)

    brier = ((pred_cs - actual_cs) ** 2).mean()
    return brier, model, features


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    goals_col = "goals" if "goals" in df.columns else ("goals_scored" if "goals_scored" in df.columns else None)
    if goals_col is None:
        print("No 'goals' or 'goals_scored' column found.")
        return
    if goals_col != "goals":
        df = df.rename(columns={goals_col: "goals"})

    required = ["fixture_id", "goals_conceded", "opponent_team", "gameweek", "team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw]
    print(f"Train: {len(train)} player-rows | Test: {len(test)} player-rows\n")

    print("Fitting goals model on TRAINING data only...")
    goals_models = fit_goals_model(train)
    print(f"  Fitted for: {list(goals_models.keys())}\n")

    print("Building team-level predicted-goals sums (train-fitted model applied to ALL rows)...")
    all_team_goals = build_team_predicted_goals(df, goals_models)

    team_df = df.groupby(["team", "fixture_id"], as_index=False).first()
    team_df = team_df.merge(all_team_goals, on=["team", "fixture_id"], how="left")

    # attach the OPPONENT's predicted goals — never the team's own
    opp_goals = all_team_goals.rename(columns={"team": "opponent_team", "team_predicted_goals": "opp_predicted_goals_crossmodel"})
    team_df = team_df.merge(opp_goals, on=["opponent_team", "fixture_id"], how="left")

    print(f"  Sample check - opp_predicted_goals_crossmodel is genuinely about the OPPONENT, not own team:")
    sample = team_df[["team", "opponent_team", "team_predicted_goals", "opp_predicted_goals_crossmodel"]].head(3)
    print(sample.to_string(index=False))
    print()

    train_team = team_df[team_df["gameweek"] <= cutoff_gw]
    test_team = team_df[team_df["gameweek"] > cutoff_gw]

    print("=" * 70)
    print("1. CURRENT clean sheet model (no cross-model feature)")
    print("=" * 70)
    result_current = fit_and_evaluate_cs(CS_FEATURES_CURRENT, train_team, test_team)
    if result_current:
        brier, model, features = result_current
        print(f"  Features: {features}")
        print(f"  Held-out Brier score: {brier:.4f}\n")

    print("=" * 70)
    print("2. WITH cross-model opponent attack strength added")
    print("=" * 70)
    features_new = CS_FEATURES_CURRENT + ["opp_predicted_goals_crossmodel"]
    result_new = fit_and_evaluate_cs(features_new, train_team, test_team)
    if result_new:
        brier, model, features = result_new
        print(f"  Features: {features}")
        for f in features:
            print(f"    {f}: coef={model.params.get(f):+.4f}, p={model.pvalues.get(f):.4f}")
        print(f"  Held-out Brier score: {brier:.4f}\n")

    if result_current and result_new:
        print("=" * 70)
        print("VERDICT")
        print("=" * 70)
        b1, b2 = result_current[0], result_new[0]
        if b2 < b1:
            pct = (b1 - b2) / b1 * 100
            print(f"  Cross-model feature is BETTER by {pct:.1f}% (Brier {b1:.4f} -> {b2:.4f})")
        else:
            pct = (b2 - b1) / b1 * 100
            print(f"  Cross-model feature is WORSE by {pct:.1f}% (Brier {b1:.4f} -> {b2:.4f})")
        print(f"\n  Same caution as before: a difference under roughly 5% could be noise.")


if __name__ == "__main__":
    main()