"""
FPL Points Predictor — Cards Validation: 4-Way Honest Comparison
========================================================================
Compares FOUR approaches to yellow card prediction on the SAME
genuinely held-out (time-based) data, closing a real gap from earlier
in this project — the original cards model was wired in after a
mechanical sanity check, but never given the same honest held-out MAE
comparison bonus v2 got.

  1. Naive baseline    — a player's own roll5_CrdY directly (the
                          ORIGINAL approach, before any model was fit
                          at all)
  2. Linear (player)   — the CURRENT live model: Poisson per position,
                          player's own features only
  3. Linear (+opponent) — same Poisson approach, adding team-level
                          opponent/own-team card-tendency features
  4. Random Forest      — nonlinear, same feature set as #3, testing
                          whether card-taking has real thresholds or
                          interactions a linear model can't capture

Run:
    python validate_cards_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["GK", "DEF", "MID", "FWD"]
TRAIN_FRACTION = 0.75

PLAYER_ONLY_FEATURES = ["season_CrdY", "roll5_CrdY", "was_home_int"]
WITH_OPPONENT_FEATURES = ["season_CrdY", "roll5_CrdY", "was_home_int",
                           "own_season_match_yellow_cards", "opp_season_match_yellow_cards"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_linear_by_position(train_df: pd.DataFrame, features: list):
    models = {}
    for position in POSITIONS:
        pos_df = train_df[train_df["position"] == position]
        feats = drop_zero_variance(pos_df, [f for f in features if f in pos_df.columns])
        if len(pos_df) < 60 or not feats:
            continue
        X = sm.add_constant(pos_df[feats].fillna(0))
        weights = pos_df["sample_weight"] if "sample_weight" in pos_df.columns else None
        fit_kwargs = {"freq_weights": weights} if weights is not None else {}
        try:
            models[position] = (sm.GLM(pos_df["yellow_cards"], X, family=sm.families.Poisson(), **fit_kwargs).fit(), feats)
        except Exception as e:
            print(f"    {position}: linear fit failed ({e})")
    return models


def fit_rf_by_position(train_df: pd.DataFrame, features: list):
    models = {}
    for position in POSITIONS:
        pos_df = train_df[train_df["position"] == position]
        feats = drop_zero_variance(pos_df, [f for f in features if f in pos_df.columns])
        if len(pos_df) < 60 or not feats:
            continue
        X = pos_df[feats].fillna(0)
        y = pos_df["yellow_cards"]
        weights = pos_df["sample_weight"] if "sample_weight" in pos_df.columns else None
        rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
        rf.fit(X, y, sample_weight=weights)
        models[position] = (rf, feats)
    return models


def predict_linear(df: pd.DataFrame, models: dict) -> pd.Series:
    preds = pd.Series(np.nan, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
        X = X.reindex(columns=model.params.index, fill_value=0)
        preds.loc[mask] = model.predict(X)
    return preds


def predict_rf(df: pd.DataFrame, models: dict) -> pd.Series:
    preds = pd.Series(np.nan, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = df.loc[mask, features].fillna(0)
        preds.loc[mask] = model.predict(X)
    return preds


def main():
    path = PROCESSED_DIR / "cards_training_combined_v2.csv"
    if not path.exists():
        print(f"{path} not found — run build_cards_training_data_v2.py first.")
        return

    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}\n")

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw].copy()
    print(f"Train: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})\n")

    print("=" * 70)
    print("1. Naive baseline (player's own roll5_CrdY directly)")
    print("=" * 70)
    test["pred_naive"] = test["roll5_CrdY"].fillna(0)

    print("\n" + "=" * 70)
    print("2. Linear, player features only (CURRENT live model)")
    print("=" * 70)
    models_linear_player = fit_linear_by_position(train, PLAYER_ONLY_FEATURES)
    test["pred_linear_player"] = predict_linear(test, models_linear_player)

    print("\n" + "=" * 70)
    print("3. Linear, WITH opponent/own-team features")
    print("=" * 70)
    models_linear_opp = fit_linear_by_position(train, WITH_OPPONENT_FEATURES)
    test["pred_linear_opp"] = predict_linear(test, models_linear_opp)

    print("\n" + "=" * 70)
    print("4. Random Forest, WITH opponent/own-team features")
    print("=" * 70)
    models_rf = fit_rf_by_position(train, WITH_OPPONENT_FEATURES)
    test["pred_rf"] = predict_rf(test, models_rf)

    evaluable = test.dropna(subset=["pred_naive", "pred_linear_player", "pred_linear_opp", "pred_rf", "yellow_cards"])
    print(f"\nEvaluable held-out rows (all 4 approaches have predictions): {len(evaluable)}\n")

    results = {}
    for name, col in [("1. Naive baseline", "pred_naive"),
                       ("2. Linear (player only)", "pred_linear_player"),
                       ("3. Linear (+opponent)", "pred_linear_opp"),
                       ("4. Random Forest (+opponent)", "pred_rf")]:
        mae = (evaluable[col] - evaluable["yellow_cards"]).abs().mean()
        results[name] = mae

    print("=" * 70)
    print("HONEST COMPARISON — held-out MAE (lower is better)")
    print("=" * 70)
    best_name = min(results, key=results.get)
    for name, mae in results.items():
        marker = "  <-- BEST" if name == best_name else ""
        print(f"  {name}: {mae:.4f}{marker}")

    baseline_mae = results["1. Naive baseline"]
    print(f"\nRelative to naive baseline ({baseline_mae:.4f}):")
    for name, mae in results.items():
        if name == "1. Naive baseline":
            continue
        if mae < baseline_mae:
            pct = (baseline_mae - mae) / baseline_mae * 100
            print(f"  {name}: {pct:.1f}% better")
        else:
            pct = (mae - baseline_mae) / baseline_mae * 100
            print(f"  {name}: {pct:.1f}% WORSE")

    print(f"\nNote on interpreting small differences: a difference under "
          f"roughly 5% should be treated with real suspicion — could easily "
          f"be noise from this specific train/test split rather than a "
          f"genuine improvement (confirmed with a pure-random control test "
          f"earlier this project). Only trust a clear, substantial margin.")


if __name__ == "__main__":
    main()
