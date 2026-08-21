"""
FPL Points Predictor — Clean Sheet Model: Definitive Held-Out Validation
========================================================================
Stops the trial-and-error on the clean sheet model's opponent features
and settles it properly, the same way every other model in this
project was validated — genuine time-based held-out comparison, not
eyeballing coefficients or one narrow metric on the full dataset.

Compares FOUR versions tried today, all fit and evaluated identically:

  1. ORIGINAL      — the pre-session feature set, no correlation pruning
  2. PRUNED_SEASON  — correlation pruning added (threshold 0.75),
                       original feature order
  3. PRUNED_XG      — reordered to prioritize opp_season_xG_for,
                       threshold 0.65
  4. PRUNED_ROLLING — rolling (roll5) opponent features added as
                       candidates, threshold 0.65

Primary metric: Brier score on P(clean sheet) — the metric this
project already established for this specific model. Secondary:
elite-vs-weak-opponent differentiation, reported for context but NOT
the deciding factor, since chasing that one number in isolation is
what led to three rounds of inconsistent results.

Run:
    python validate_clean_sheet_versions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75
ELITE_TEAMS = ["Man City", "Arsenal", "Liverpool"]

VERSIONS = {
    "1_ORIGINAL": {
        "features": ["own_season_goals_conceded", "own_roll5_goals_conceded",
                     "own_season_xG_against", "opp_season_goals_scored",
                     "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
                     "own_season_possession", "opp_season_possession",
                     "own_season_ppda", "opp_season_ppda"],
        "prune_threshold": None,
    },
    "2_PRUNED_SEASON": {
        "features": ["own_season_goals_conceded", "own_roll5_goals_conceded",
                     "own_season_xG_against", "opp_season_goals_scored",
                     "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
                     "own_season_possession", "opp_season_possession",
                     "own_season_ppda", "opp_season_ppda"],
        "prune_threshold": 0.75,
    },
    "3_PRUNED_XG": {
        "features": ["own_roll5_goals_conceded", "own_season_goals_conceded",
                     "own_season_xG_against", "opp_season_xG_for",
                     "opp_season_goals_scored", "opp_season_shots_for", "was_home_int",
                     "own_season_possession", "opp_season_possession",
                     "own_season_ppda", "opp_season_ppda"],
        "prune_threshold": 0.65,
    },
    "4_PRUNED_ROLLING": {
        "features": ["own_roll5_goals_conceded", "own_season_goals_conceded",
                     "own_roll5_xG_against", "own_season_xG_against",
                     "opp_roll5_xG_for", "opp_season_xG_for",
                     "opp_roll5_goals_scored", "opp_season_goals_scored",
                     "opp_season_shots_for", "was_home_int",
                     "own_season_possession", "opp_season_possession",
                     "own_season_ppda", "opp_season_ppda"],
        "prune_threshold": 0.65,
    },
}


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def prune_correlated(df, features, threshold):
    if threshold is None:
        return features
    pruned = []
    for f in features:
        too_similar = any(abs(df[f].corr(df[g])) > threshold for g in pruned)
        if not too_similar:
            pruned.append(f)
    return pruned


def fit_and_evaluate(version_name, config, train, test):
    candidate_features = [f for f in config["features"] if f in train.columns]
    features = drop_zero_variance(train, candidate_features)
    features = prune_correlated(train, features, config["prune_threshold"])

    if len(features) == 0:
        print(f"  {version_name}: no usable features, skipping.")
        return None

    X_train = sm.add_constant(train[features].fillna(0))
    model = sm.GLM(train["goals_conceded"], X_train, family=sm.families.Poisson()).fit()

    X_test = sm.add_constant(test[features].fillna(0), has_constant="add")
    X_test = X_test.reindex(columns=model.params.index, fill_value=0)
    pred_goals_conceded = model.predict(X_test)
    pred_cs = np.exp(-pred_goals_conceded)
    actual_cs = (test["goals_conceded"] == 0).astype(int)

    brier = ((pred_cs - actual_cs) ** 2).mean()

    result = {
        "version": version_name,
        "features_kept": features,
        "brier_score": brier,
    }

    if "opponent_team" in test.columns:
        test_with_pred = test.copy()
        test_with_pred["pred_goals_conceded"] = pred_goals_conceded
        vs_elite = test_with_pred[test_with_pred["opponent_team"].isin(ELITE_TEAMS)]
        vs_other = test_with_pred[~test_with_pred["opponent_team"].isin(ELITE_TEAMS)]
        if len(vs_elite) > 0 and len(vs_other) > 0:
            elite_mean = vs_elite["pred_goals_conceded"].mean()
            other_mean = vs_other["pred_goals_conceded"].mean()
            result["elite_vs_weak_pct"] = (elite_mean - other_mean) / other_mean * 100

    return result


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    if "fixture_id" not in df.columns or "goals_conceded" not in df.columns:
        print("Missing fixture_id or goals_conceded.")
        return

    team_df = df.groupby(["team", "fixture_id"], as_index=False).first()
    if "gameweek" not in team_df.columns:
        print("No gameweek column - cannot do a time-based split.")
        return

    cutoff_gw = team_df["gameweek"].quantile(TRAIN_FRACTION)
    train = team_df[team_df["gameweek"] <= cutoff_gw]
    test = team_df[team_df["gameweek"] > cutoff_gw]

    print(f"Train: {len(train)} team-matches (gw <= {cutoff_gw:.0f}) | "
          f"Test: {len(test)} team-matches (gw > {cutoff_gw:.0f})\n")

    results = []
    for version_name, config in VERSIONS.items():
        print(f"--- {version_name} ---")
        result = fit_and_evaluate(version_name, config, train, test)
        if result:
            print(f"  Features kept: {result['features_kept']}")
            print(f"  Brier score (held-out): {result['brier_score']:.4f}")
            if "elite_vs_weak_pct" in result:
                print(f"  Elite-vs-weak differentiation: {result['elite_vs_weak_pct']:+.1f}%")
            results.append(result)
        print()

    if results:
        print("=" * 70)
        print("FINAL VERDICT - ranked by held-out Brier score (lower is better)")
        print("=" * 70)
        results_sorted = sorted(results, key=lambda r: r["brier_score"])
        for i, r in enumerate(results_sorted):
            marker = "  <-- BEST" if i == 0 else ""
            elite_str = f", elite-vs-weak: {r['elite_vs_weak_pct']:+.1f}%" if "elite_vs_weak_pct" in r else ""
            print(f"  {r['version']}: Brier={r['brier_score']:.4f}{elite_str}{marker}")
        print(f"\nThis Brier score comparison is the deciding factor - not the "
              f"elite-vs-weak percentage alone, which proved an unreliable, noisy "
              f"guide across the last three rounds of changes.")


if __name__ == "__main__":
    main()
