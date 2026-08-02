"""
FPL Points Predictor — BONUS POINTS, Stage 2: match-roster ranking
========================================================================
Stage 1 (analyze_bps_by_position.py) predicts a player's expected BPS.
This stage converts BPS into actual bonus points (0-3), which requires
ranking a player's BPS against everyone else in that SAME real match —
bonus is inherently relative, not an absolute threshold.

FPL's real tie-break rule (implemented here via dense ranking):
  - The distinct HIGHEST BPS value in the match -> everyone tied there
    gets 3 bonus points
  - The next distinct BPS value -> everyone tied there gets 2
  - The next distinct BPS value after that -> everyone tied there gets 1
  - Ties can mean MORE than 3 players get bonus in a match (if e.g. two
    players tie for 2nd, both get 2) — this is real FPL behavior, not
    a bug in the ranking logic.

TWO SEPARATE VALIDATIONS, deliberately kept apart:
  1. Rank using REAL historical BPS -> compare to REAL historical bonus.
     This checks ONLY whether the ranking mechanism itself is correctly
     implemented — no prediction involved. Should match ~100% if right.
  2. Rank using MODEL-PREDICTED BPS on genuinely held-out future
     gameweeks -> compare to real bonus. This is the honest measure of
     how good the full prediction pipeline actually is, since it
     includes real-world prediction error, not just the ranking logic.

Run:
    python build_bonus_predictions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["GK", "DEF", "MID", "FWD"]
TRAIN_FRACTION = 0.75  # earlier gameweeks train, later ones validate — time-based, not random

FULL_CANDIDATE_POOL = [
    "season_bps", "roll5_bps",
    "season_goals", "roll5_goals",
    "season_assists", "roll5_assists",
    "season_xG", "roll5_xG",
    "season_xA", "roll5_xA",
    "season_def_contrib", "roll5_def_contrib",
    "season_TklW", "roll5_TklW",
    "season_Int", "roll5_Int",
    "season_clean_sheets", "roll5_clean_sheets",
    "season_CrdY", "roll5_CrdY",
    "roll5_minutes", "roll5_starts",
    "opp_season_shots_for", "opp_roll5_shots_for",
    "own_season_shots_against", "own_roll5_shots_against",
    "was_home_int",
    "days_since_last_game",
]

WINDOW_PREFIXES = ["season_", "roll10_", "roll5_", "roll3_"]


def get_feature_family(feature_name: str) -> str:
    name = feature_name
    prefix = ""
    for p in ("opp_", "own_"):
        if name.startswith(p):
            prefix = p
            name = name[len(p):]
            break
    for w in WINDOW_PREFIXES:
        if name.startswith(w):
            name = name[len(w):]
            break
    return prefix + name


def assign_bonus_from_bps(df: pd.DataFrame, bps_col: str) -> pd.Series:
    """FPL's real dense-rank tie-break rule, applied within each fixture."""
    dense_rank = df.groupby("fixture_id")[bps_col].rank(method="dense", ascending=False)
    return dense_rank.map({1: 3, 2: 2, 3: 1}).fillna(0).astype(int)


def select_features(pos_df: pd.DataFrame) -> list:
    n_feats = max(4, min(len(FULL_CANDIDATE_POOL), len(pos_df) // 15))
    available = [f for f in FULL_CANDIDATE_POOL if f in pos_df.columns]

    variances = pos_df[available].var(numeric_only=True)
    zero_var_cols = variances[variances.fillna(0) == 0].index.tolist()
    available = [f for f in available if f not in zero_var_cols]

    corrs = pos_df[available + ["bps"]].corr(numeric_only=True)["bps"].drop("bps").abs()
    corr_df = corrs.reset_index()
    corr_df.columns = ["feature", "abs_corr"]
    corr_df["family"] = corr_df["feature"].apply(get_feature_family)
    best_per_family = corr_df.sort_values("abs_corr", ascending=False).drop_duplicates(subset="family", keep="first")
    return best_per_family.sort_values("abs_corr", ascending=False).head(n_feats)["feature"].tolist()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    return df[(df["minutes"] > 0) & (df["roll5_bps"].notna())].copy()


def validate_ranking_mechanism(df: pd.DataFrame):
    """Step 1: using REAL bps, does our tie-break logic reproduce real bonus?"""
    predicted_bonus = assign_bonus_from_bps(df, "bps")
    exact_match = (predicted_bonus == df["bonus"]).mean()

    print("=" * 70)
    print("STEP 1: Ranking mechanism sanity check (using REAL historical BPS)")
    print("=" * 70)
    print(f"Exact match with real bonus: {exact_match:.1%}")
    if exact_match < 0.98:
        mismatches = df[predicted_bonus != df["bonus"]]
        print(f"\n{len(mismatches)} mismatches — sample:")
        print(mismatches[["player_name", "team", "gameweek", "bps", "bonus"]]
              .assign(our_bonus=predicted_bonus[mismatches.index]).head(10).to_string(index=False))
    else:
        print("Ranking mechanism confirmed correct — safe to trust it for Step 2.")


def predict_bps_out_of_sample(df: pd.DataFrame) -> pd.Series:
    """Step 2 prep: fit each position's model on EARLY gameweeks only, predict
    on LATER gameweeks — genuinely held-out, not in-sample fitted values.
    Uses Random Forest rather than OLS — testing whether a nonlinear model
    captures more signal than the linear Stage-1 version did."""
    from sklearn.ensemble import RandomForestRegressor

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    predictions = pd.Series(index=df.index, dtype=float)

    for position in POSITIONS:
        pos_df = df[df["position"] == position]
        train = pos_df[pos_df["gameweek"] <= cutoff_gw]
        test = pos_df[pos_df["gameweek"] > cutoff_gw]
        if len(train) < 60 or len(test) == 0:
            print(f"  {position}: not enough data for a train/test split, skipping — "
                  f"predicted BPS left as NaN for this position's held-out rows.")
            continue

        features = select_features(train)
        X_train = train[features].fillna(0)
        y_train = train["bps"]

        rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        X_test = test[features].fillna(0)
        preds = rf.predict(X_test)
        predictions.loc[test.index] = preds
        print(f"  {position}: trained on {len(train)} rows (gw <= {cutoff_gw:.0f}), "
              f"predicting {len(test)} held-out rows (gw > {cutoff_gw:.0f})")

    return predictions


def validate_real_predictions(df: pd.DataFrame):
    """Step 2: using genuinely held-out PREDICTED bps, how well does bonus prediction do?"""
    print("\n" + "=" * 70)
    print("STEP 2: Real prediction accuracy (predicted BPS, held-out gameweeks)")
    print("=" * 70)

    df = df.copy()
    df["predicted_bps"] = predict_bps_out_of_sample(df)
    held_out = df[df["predicted_bps"].notna()].copy()

    predicted_bonus = assign_bonus_from_bps(held_out, "predicted_bps")
    exact_match = (predicted_bonus == held_out["bonus"]).mean()
    baseline_accuracy = (held_out["bonus"] == 0).mean()  # naive "always predict 0" baseline

    print(f"\nHeld-out rows evaluated: {len(held_out)}")
    print(f"Naive baseline (always predict 0 bonus): {baseline_accuracy:.1%}")
    print(f"Exact bonus match (0/1/2/3 all correct): {exact_match:.1%}")
    if exact_match <= baseline_accuracy:
        print("*** WARNING: model does NOT beat the naive baseline of always predicting 0. ***")
    else:
        print(f"Model beats the naive baseline by {(exact_match - baseline_accuracy) * 100:.1f} points.")

    # a softer, arguably more useful metric: did we correctly identify anyone
    # who earned bonus at all (bonus > 0), regardless of getting the exact
    # 1 vs 2 vs 3 right?
    actual_got_bonus = held_out["bonus"] > 0
    predicted_got_bonus = predicted_bonus > 0
    if actual_got_bonus.sum() > 0:
        recall = (actual_got_bonus & predicted_got_bonus).sum() / actual_got_bonus.sum()
        precision = (actual_got_bonus & predicted_got_bonus).sum() / max(predicted_got_bonus.sum(), 1)
        print(f"\nOf players who ACTUALLY earned bonus, we correctly flagged: {recall:.1%} (recall)")
        print(f"Of players we PREDICTED would earn bonus, they actually did: {precision:.1%} (precision)")

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(pd.crosstab(held_out["bonus"], predicted_bonus, rownames=["actual"], colnames=["predicted"]))


def main():
    df = load_data()
    print(f"Total rows: {len(df)}\n")

    validate_ranking_mechanism(df)
    validate_real_predictions(df)


if __name__ == "__main__":
    main()