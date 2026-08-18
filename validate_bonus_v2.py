"""
FPL Points Predictor — Bonus v2: Per-Player Expected Bonus Regression
========================================================================
Fits a real per-player expected-bonus model (Poisson per position, same
approach as goals/assists/cards) on the cross-season combined data, and
HONESTLY validates it against the CURRENT flat position-average
baseline on genuinely held-out data before recommending anything be
wired into live predictions.

This is a time-based train/validate split (earlier gameweeks train,
later ones validate), same philosophy as the original bonus
investigation — never validate on data the model could have seen.

Run:
    python validate_bonus_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["GK", "DEF", "MID", "FWD"]
TRAIN_FRACTION = 0.75

CANDIDATE_FEATURES = ["roll5_bps", "season_bps", "roll5_fpl_xG", "season_fpl_xG",
                       "roll5_fpl_xA", "season_fpl_xA", "roll5_CrdY", "roll5_minutes",
                       "was_home_int"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_bonus_poisson_by_position(train_df: pd.DataFrame):
    """Same pattern as fit_poisson_by_position used for goals/assists/
    cards elsewhere in this project — fits a Poisson GLM per position,
    with sample weights so 25/26 dominates the fit."""
    models = {}
    for position in POSITIONS:
        pos_df = train_df[train_df["position"] == position]
        features = drop_zero_variance(pos_df, [f for f in CANDIDATE_FEATURES if f in pos_df.columns])
        if len(pos_df) < 60 or not features:
            print(f"  {position}: not enough data/features, skipping")
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            weights = pos_df["sample_weight"] if "sample_weight" in pos_df.columns else None
            fit_kwargs = {"freq_weights": weights} if weights is not None else {}
            model = sm.GLM(pos_df["bonus"], X, family=sm.families.Poisson(), **fit_kwargs).fit()
            models[position] = (model, features)
            print(f"  {position}: fitted on {len(pos_df)} rows, features: {features}")
        except Exception as e:
            print(f"  {position}: fit failed ({e}), skipping")
    return models


def predict_from_models(df: pd.DataFrame, models: dict) -> pd.Series:
    preds = pd.Series(np.nan, index=df.index)
    for position, (model, features) in models.items():
        mask = df["position"] == position
        if mask.sum() == 0:
            continue
        X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
        X = X.reindex(columns=model.params.index, fill_value=0)
        preds.loc[mask] = model.predict(X)
    return preds


def main():
    path = PROCESSED_DIR / "bonus_training_combined.csv"
    if not path.exists():
        print(f"{path} not found — run build_bonus_training_data.py first.")
        return

    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}\n")

    # time-based split — genuinely held-out, same as the original
    # investigation's own methodology
    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw]
    print(f"Train: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})\n")

    print("=" * 70)
    print("Fitting per-player bonus model (Poisson by position)")
    print("=" * 70)
    models = fit_bonus_poisson_by_position(train)

    test = test.copy()
    test["pred_bonus_v2"] = predict_from_models(test, models)

    # baseline: CURRENT behavior — flat average per position, computed
    # from the SAME training data, so this is a fair, apples-to-apples
    # comparison rather than a different data slice
    position_avg = train.groupby("position")["bonus"].mean().to_dict()
    overall_avg = train["bonus"].mean()
    test["pred_bonus_baseline"] = test["position"].map(position_avg).fillna(overall_avg)

    evaluable = test.dropna(subset=["pred_bonus_v2", "bonus"])
    print(f"\nEvaluable held-out rows: {len(evaluable)}\n")

    mae_v2 = (evaluable["pred_bonus_v2"] - evaluable["bonus"]).abs().mean()
    mae_baseline = (evaluable["pred_bonus_baseline"] - evaluable["bonus"]).abs().mean()

    print("=" * 70)
    print("HONEST COMPARISON — held-out MAE (lower is better)")
    print("=" * 70)
    print(f"Current flat position-average baseline: {mae_baseline:.4f}")
    print(f"New per-player Poisson model (v2):       {mae_v2:.4f}")

    if mae_v2 < mae_baseline:
        improvement = (mae_baseline - mae_v2) / mae_baseline * 100
        print(f"\nv2 is BETTER by {improvement:.1f}% — genuinely worth wiring into live predictions.")
    else:
        worse = (mae_v2 - mae_baseline) / mae_baseline * 100
        print(f"\nv2 is WORSE by {worse:.1f}% — the flat baseline actually holds up better here. "
              f"NOT recommended for wiring in as-is.")

    # also check: does v2 at least correctly RANK players better within
    # a position, even if the raw MAE is similar? A model that's
    # slightly worse on MAE but meaningfully better at ordering players
    # (Haaland-types above rotation players) could still be useful for
    # the dashboard's relative rankings, even if not a clear MAE win.
    print("\n" + "=" * 70)
    print("Supplementary check: correlation with actual bonus (rank quality)")
    print("=" * 70)
    for position in POSITIONS:
        pos_test = evaluable[evaluable["position"] == position]
        if len(pos_test) < 10:
            continue
        corr_v2 = pos_test["pred_bonus_v2"].corr(pos_test["bonus"])
        corr_baseline = pos_test["pred_bonus_baseline"].corr(pos_test["bonus"])
        print(f"  {position}: v2 correlation = {corr_v2:.3f} | baseline correlation = {corr_baseline:.3f} "
              f"(baseline is a constant per position, so this is usually ~0 or undefined by construction)")


if __name__ == "__main__":
    main()
