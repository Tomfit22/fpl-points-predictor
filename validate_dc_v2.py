"""
FPL Points Predictor — DC Nonlinear Validation: Honest Comparison
========================================================================
DC already has opponent features validated in earlier work (real Brier
improvement: DEF 0.164->0.146, MID 0.101->0.088). The open question
left untested is whether a NONLINEAR model (Random Forest) beats the
current logistic regression on the SAME features — testing whether
defensive-contribution-threshold hitting has real nonlinear structure
(e.g. genuine thresholds/interactions) a linear model can't capture,
same question already asked and answered for cards (no) and bonus
(no) — worth checking DC specifically rather than assuming the same
answer applies here.

Uses Brier score (mean squared error between predicted probability and
actual binary outcome) — the same metric this project already
established for DC, not a new one introduced just for this comparison.

Run:
    python validate_dc_v2.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DIR = Path("data/processed")
POSITIONS = ["DEF", "MID", "FWD"]
TRAIN_FRACTION = 0.75

DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

DC_FEATURES = ["season_def_contrib", "roll5_def_contrib", "season_TklW", "season_Int",
               "opp_season_shots_for", "was_home_int",
               "opp_season_possession", "opp_roll5_possession",
               "own_season_ppda", "own_roll5_ppda"]


def drop_zero_variance(df, features):
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


def prune_correlated(df, features, threshold=0.98):
    pruned = []
    for f in features:
        too_similar = any(abs(df[f].corr(df[g])) > threshold for g in pruned)
        if not too_similar:
            pruned.append(f)
    return pruned


def fit_logistic_dc(train_df: pd.DataFrame, position: str):
    pos_df = train_df[train_df["position"] == position].copy()
    pos_df["hit"] = (pos_df["defensive_contribution"] >= DC_THRESHOLD[position]).astype(int)
    if pos_df["hit"].sum() < 20:
        return None, None
    features = drop_zero_variance(pos_df, [f for f in DC_FEATURES if f in pos_df.columns])
    features = prune_correlated(pos_df, features)
    X = sm.add_constant(pos_df[features].fillna(0))
    try:
        model = sm.Logit(pos_df["hit"], X).fit(disp=0)
        if not model.mle_retvals.get("converged", True):
            raise np.linalg.LinAlgError("did not converge")
    except np.linalg.LinAlgError:
        model = sm.Logit(pos_df["hit"], X).fit_regularized(disp=0, alpha=0.1)
    return model, features


def fit_rf_dc(train_df: pd.DataFrame, position: str, features: list):
    pos_df = train_df[train_df["position"] == position].copy()
    pos_df["hit"] = (pos_df["defensive_contribution"] >= DC_THRESHOLD[position]).astype(int)
    if pos_df["hit"].sum() < 20 or not features:
        return None
    X = pos_df[features].fillna(0)
    y = pos_df["hit"]
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf


def predict_logistic(df, model, features, position):
    mask = df["position"] == position
    preds = pd.Series(np.nan, index=df.index)
    if model is None or mask.sum() == 0:
        return preds
    X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
    X = X.reindex(columns=model.params.index, fill_value=0)
    preds.loc[mask] = model.predict(X)
    return preds


def predict_rf(df, model, features, position):
    mask = df["position"] == position
    preds = pd.Series(np.nan, index=df.index)
    if model is None or mask.sum() == 0:
        return preds
    X = df.loc[mask, features].fillna(0)
    preds.loc[mask] = model.predict_proba(X)[:, 1]
    return preds


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found — run the main pipeline first.")
        return

    df = pd.read_csv(path)
    df = df[(df["minutes"] > 0)].copy()
    if "defensive_contribution" not in df.columns:
        print("'defensive_contribution' column not found — check the dataset.")
        return

    print(f"Total rows: {len(df)}\n")

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw].copy()
    print(f"Train: {len(train)} rows (gw <= {cutoff_gw:.0f}) | Test: {len(test)} rows (gw > {cutoff_gw:.0f})\n")

    overall_results = {}
    for position in POSITIONS:
        print("=" * 70)
        print(f"POSITION: {position}")
        print("=" * 70)

        logit_model, features = fit_logistic_dc(train, position)
        if logit_model is None:
            print(f"  Not enough positive cases for {position} — skipping.")
            continue
        print(f"  Features used: {features}")

        rf_model = fit_rf_dc(train, position, features)

        test_pos = test[test["position"] == position].copy()
        test_pos["hit"] = (test_pos["defensive_contribution"] >= DC_THRESHOLD[position]).astype(int)

        pred_logit = predict_logistic(test_pos, logit_model, features, position)
        pred_rf = predict_rf(test_pos, rf_model, features, position)

        evaluable = test_pos[pred_logit.notna() & pred_rf.notna()].copy()
        evaluable["pred_logit"] = pred_logit
        evaluable["pred_rf"] = pred_rf

        if len(evaluable) == 0:
            print("  No evaluable rows — skipping.")
            continue

        brier_logit = ((evaluable["pred_logit"] - evaluable["hit"]) ** 2).mean()
        brier_rf = ((evaluable["pred_rf"] - evaluable["hit"]) ** 2).mean()

        print(f"\n  Evaluable held-out rows: {len(evaluable)}")
        print(f"  Current logistic Brier score: {brier_logit:.4f} (lower is better)")
        print(f"  Random Forest Brier score:    {brier_rf:.4f}")
        if brier_rf < brier_logit:
            pct = (brier_logit - brier_rf) / brier_logit * 100
            print(f"  Random Forest is BETTER by {pct:.1f}%")
        else:
            pct = (brier_rf - brier_logit) / brier_logit * 100
            print(f"  Random Forest is WORSE by {pct:.1f}%")
        overall_results[position] = (brier_logit, brier_rf)
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for position, (bl, br) in overall_results.items():
        verdict = "RF better" if br < bl else "RF worse"
        print(f"  {position}: logistic={bl:.4f}, RF={br:.4f} -> {verdict}")
    print(f"\nSame caution as before: a difference under roughly 5% could be noise "
          f"rather than a real improvement — only trust a clear margin.")


if __name__ == "__main__":
    main()