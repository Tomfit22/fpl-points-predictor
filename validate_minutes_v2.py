"""
FPL Points Predictor — Minutes Validation: Honest Comparison
========================================================================
Compares the CURRENT live minutes model (fit on 25/26 data only) against
a cross-season weighted version (24/25 + 25/26, weighted 3:1 favoring
25/26) on genuinely held-out data — same rigor as cards/bonus.

Minutes is the highest-value candidate for cross-season enhancement in
this whole batch: every feature the model uses (roll5_minutes,
roll5_starts, consecutive_starts, days_since_last_game) is fully
computable from the 24/25 archive, no scoped/degraded feature set
needed like goals/assists or clean sheets require.

Predicts P(60+ minutes) — same threshold as the live "sixty_plus"
component, since that's the one that gates clean sheet/bonus points
downstream and matters most.

Run:
    python validate_minutes_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

MINUTES_FEATURES = ["roll5_minutes", "roll5_starts", "consecutive_starts", "days_since_last_game"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def fit_minutes_binomial(train_df: pd.DataFrame, threshold: int, weight_col: str = None):
    """Same correlation-pruning logic as the live fit_minutes_model,
    but using Binomial GLM instead of raw Logit so it can accept
    sample weights — confirmed mathematically identical to Logit when
    unweighted (verified earlier this project)."""
    candidates = drop_zero_variance(train_df, [f for f in MINUTES_FEATURES if f in train_df.columns])
    features = []
    for f in candidates:
        too_similar = any(abs(train_df[f].corr(train_df[g])) > 0.85 for g in features)
        if not too_similar:
            features.append(f)

    X = sm.add_constant(train_df[features].fillna(0))
    y = (train_df["minutes"] >= threshold).astype(int)
    fit_kwargs = {}
    if weight_col and weight_col in train_df.columns:
        fit_kwargs["freq_weights"] = train_df[weight_col]
    model = sm.GLM(y, X, family=sm.families.Binomial(), **fit_kwargs).fit()
    return model, features


def predict_binomial(df: pd.DataFrame, model, features: list) -> pd.Series:
    X = sm.add_constant(df[features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    return pd.Series(model.predict(X), index=df.index)


def main():
    path = PROCESSED_DIR / "minutes_training_combined.csv"
    if not path.exists():
        print(f"{path} not found — run build_minutes_training_data.py first.")
        return

    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}\n")

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train_current_only = df[(df["gameweek"] <= cutoff_gw) & (df["season"] == "2025-26")]
    train_combined = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw].copy()
    test = test[test["season"] == "2025-26"]  # only evaluate on real current-season held-out rows
    print(f"Train (current only): {len(train_current_only)} rows | "
          f"Train (combined, weighted): {len(train_combined)} rows | "
          f"Test: {len(test)} rows (gw > {cutoff_gw:.0f}, 25/26 only)\n")

    print("=" * 70)
    print("1. CURRENT live model (25/26 data only, unweighted)")
    print("=" * 70)
    model_current, feats_current = fit_minutes_binomial(train_current_only, threshold=60)
    print(f"  Features used: {feats_current}")
    test["pred_current"] = predict_binomial(test, model_current, feats_current)

    print("\n" + "=" * 70)
    print("2. Cross-season model (24/25 + 25/26, weighted 3:1)")
    print("=" * 70)
    model_combined, feats_combined = fit_minutes_binomial(train_combined, threshold=60, weight_col="sample_weight")
    print(f"  Features used: {feats_combined}")
    test["pred_combined"] = predict_binomial(test, model_combined, feats_combined)

    evaluable = test.dropna(subset=["pred_current", "pred_combined"])
    actual = (evaluable["minutes"] >= 60).astype(int)
    print(f"\nEvaluable held-out rows: {len(evaluable)}\n")

    mae_current = (evaluable["pred_current"] - actual).abs().mean()
    mae_combined = (evaluable["pred_combined"] - actual).abs().mean()

    print("=" * 70)
    print("HONEST COMPARISON — held-out MAE on P(60+ minutes) (lower is better)")
    print("=" * 70)
    print(f"  1. Current live model (25/26 only): {mae_current:.4f}")
    print(f"  2. Cross-season weighted model:      {mae_combined:.4f}")

    if mae_combined < mae_current:
        pct = (mae_current - mae_combined) / mae_current * 100
        print(f"\nCross-season model is BETTER by {pct:.1f}%.")
    else:
        pct = (mae_combined - mae_current) / mae_current * 100
        print(f"\nCross-season model is WORSE by {pct:.1f}%.")

    print(f"\nSame caution as before: a difference under roughly 5% could be "
          f"noise rather than a real improvement — only trust a clear margin.")


if __name__ == "__main__":
    main()