"""
FPL Points Predictor — XGBoost Validation: Cards, Bonus, DC, Clean Sheets
========================================================================
Extends the Random Forest comparisons already run this project with
XGBoost — a genuinely different, generally stronger nonlinear method
(sequential error-correcting trees with built-in regularization,
versus Random Forest's independent-tree averaging). Random Forest
losing everywhere doesn't necessarily mean there's no nonlinear signal
to find — it could mean RF specifically wasn't strong enough. This
checks that honestly rather than assuming the same conclusion applies.

Same held-out (time-based) methodology, same features, same metrics
(MAE for cards/bonus, Brier score for DC/clean sheets) as the earlier
Random Forest validations, so results are directly comparable.

Run:
    python validate_xgboost_all.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75


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


# =========================================================================
# CARDS
# =========================================================================
def validate_cards():
    print("#" * 70)
    print("# CARDS")
    print("#" * 70)
    path = PROCESSED_DIR / "cards_training_combined.csv"
    if not path.exists():
        print(f"{path} not found — skipping cards.")
        return
    df = pd.read_csv(path)
    features = ["season_CrdY", "roll5_CrdY", "was_home_int"]
    positions = ["GK", "DEF", "MID", "FWD"]

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw].copy()

    for position in positions:
        pos_train = train[train["position"] == position]
        pos_test = test[test["position"] == position]
        feats = drop_zero_variance(pos_train, [f for f in features if f in pos_train.columns])
        if len(pos_train) < 60 or not feats:
            continue
        weights = pos_train["sample_weight"] if "sample_weight" in pos_train.columns else None

        xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
        xgb.fit(pos_train[feats].fillna(0), pos_train["yellow_cards"], sample_weight=weights)
        pred_xgb = xgb.predict(pos_test[feats].fillna(0))

        naive_pred = pos_test["roll5_CrdY"].fillna(0)

        mae_xgb = (pred_xgb - pos_test["yellow_cards"]).abs().mean()
        mae_naive = (naive_pred - pos_test["yellow_cards"]).abs().mean()
        print(f"  {position}: naive={mae_naive:.4f}, XGBoost={mae_xgb:.4f} "
              f"-> {'XGBoost better' if mae_xgb < mae_naive else 'XGBoost worse'} "
              f"({abs(mae_xgb-mae_naive)/mae_naive*100:.1f}%)")


# =========================================================================
# BONUS
# =========================================================================
def validate_bonus():
    print("\n" + "#" * 70)
    print("# BONUS")
    print("#" * 70)
    path = PROCESSED_DIR / "bonus_training_combined.csv"
    if not path.exists():
        print(f"{path} not found — skipping bonus.")
        return
    df = pd.read_csv(path)
    features = ["roll5_bps", "season_bps", "roll5_CrdY", "roll5_minutes", "was_home_int"]
    positions = ["GK", "DEF", "MID", "FWD"]

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw].copy()

    for position in positions:
        pos_train = train[train["position"] == position]
        pos_test = test[test["position"] == position]
        feats = drop_zero_variance(pos_train, [f for f in features if f in pos_train.columns])
        if len(pos_train) < 60 or not feats:
            continue
        weights = pos_train["sample_weight"] if "sample_weight" in pos_train.columns else None

        X_train = sm.add_constant(pos_train[feats].fillna(0))
        fit_kwargs = {"freq_weights": weights} if weights is not None else {}
        linear_model = sm.GLM(pos_train["bonus"], X_train, family=sm.families.Poisson(), **fit_kwargs).fit()
        X_test = sm.add_constant(pos_test[feats].fillna(0), has_constant="add")
        X_test = X_test.reindex(columns=linear_model.params.index, fill_value=0)
        pred_linear = linear_model.predict(X_test)

        xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
        xgb.fit(pos_train[feats].fillna(0), pos_train["bonus"], sample_weight=weights)
        pred_xgb = xgb.predict(pos_test[feats].fillna(0))

        mae_linear = (pred_linear - pos_test["bonus"]).abs().mean()
        mae_xgb = (pred_xgb - pos_test["bonus"]).abs().mean()
        print(f"  {position}: current linear={mae_linear:.4f}, XGBoost={mae_xgb:.4f} "
              f"-> {'XGBoost better' if mae_xgb < mae_linear else 'XGBoost worse'} "
              f"({abs(mae_xgb-mae_linear)/mae_linear*100:.1f}%)")


# =========================================================================
# DC
# =========================================================================
def validate_dc():
    print("\n" + "#" * 70)
    print("# DC")
    print("#" * 70)
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found — skipping DC.")
        return
    df = pd.read_csv(path)
    if "defensive_contribution" not in df.columns:
        print("No defensive_contribution column — skipping DC.")
        return
    df = df[df["minutes"] > 0].copy()

    DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
    features = ["season_def_contrib", "roll5_def_contrib", "season_TklW", "season_Int",
                "opp_season_shots_for", "was_home_int",
                "opp_season_possession", "opp_roll5_possession",
                "own_season_ppda", "own_roll5_ppda"]

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw].copy()

    for position in ["DEF", "MID", "FWD"]:
        pos_train = train[train["position"] == position].copy()
        pos_train["hit"] = (pos_train["defensive_contribution"] >= DC_THRESHOLD[position]).astype(int)
        pos_test = test[test["position"] == position].copy()
        pos_test["hit"] = (pos_test["defensive_contribution"] >= DC_THRESHOLD[position]).astype(int)

        if pos_train["hit"].sum() < 20:
            print(f"  {position}: not enough positive cases, skipping.")
            continue
        feats = drop_zero_variance(pos_train, [f for f in features if f in pos_train.columns])
        feats = prune_correlated(pos_train, feats)

        X_train = sm.add_constant(pos_train[feats].fillna(0))
        try:
            logit_model = sm.Logit(pos_train["hit"], X_train).fit(disp=0)
            if not logit_model.mle_retvals.get("converged", True):
                raise np.linalg.LinAlgError()
        except np.linalg.LinAlgError:
            logit_model = sm.Logit(pos_train["hit"], X_train).fit_regularized(disp=0, alpha=0.1)
        X_test = sm.add_constant(pos_test[feats].fillna(0), has_constant="add")
        X_test = X_test.reindex(columns=logit_model.params.index, fill_value=0)
        pred_logit = logit_model.predict(X_test)

        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42,
                             eval_metric="logloss")
        xgb.fit(pos_train[feats].fillna(0), pos_train["hit"])
        pred_xgb = xgb.predict_proba(pos_test[feats].fillna(0))[:, 1]

        brier_logit = ((pred_logit - pos_test["hit"]) ** 2).mean()
        brier_xgb = ((pred_xgb - pos_test["hit"]) ** 2).mean()
        print(f"  {position}: current logistic={brier_logit:.4f}, XGBoost={brier_xgb:.4f} "
              f"-> {'XGBoost better' if brier_xgb < brier_logit else 'XGBoost worse'} "
              f"({abs(brier_xgb-brier_logit)/brier_logit*100:.1f}%)")


# =========================================================================
# CLEAN SHEETS
# =========================================================================
def validate_clean_sheets():
    print("\n" + "#" * 70)
    print("# CLEAN SHEETS")
    print("#" * 70)
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found — skipping clean sheets.")
        return
    df = pd.read_csv(path)
    if "fixture_id" not in df.columns or "goals_conceded" not in df.columns:
        print("Missing fixture_id or goals_conceded — skipping clean sheets.")
        return

    team_df = df.groupby(["team", "fixture_id", "gameweek"], as_index=False).first()
    features = ["own_season_goals_conceded", "own_roll5_goals_conceded",
                "own_season_xG_against", "opp_season_goals_scored",
                "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
                "own_season_possession", "opp_season_possession",
                "own_season_ppda", "opp_season_ppda"]

    cutoff_gw = team_df["gameweek"].quantile(TRAIN_FRACTION)
    train = team_df[team_df["gameweek"] <= cutoff_gw]
    test = team_df[team_df["gameweek"] > cutoff_gw].copy()

    feats = drop_zero_variance(train, [f for f in features if f in train.columns])

    X_train = sm.add_constant(train[feats].fillna(0))
    poisson_model = sm.GLM(train["goals_conceded"], X_train, family=sm.families.Poisson()).fit()
    X_test = sm.add_constant(test[feats].fillna(0), has_constant="add")
    X_test = X_test.reindex(columns=poisson_model.params.index, fill_value=0)
    pred_poisson = poisson_model.predict(X_test)
    pred_cs_poisson = np.exp(-pred_poisson)

    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    xgb.fit(train[feats].fillna(0), train["goals_conceded"])
    pred_xgb = xgb.predict(test[feats].fillna(0))
    pred_cs_xgb = np.exp(-np.clip(pred_xgb, 0, None))

    actual_cs = (test["goals_conceded"] == 0).astype(int)
    brier_poisson = ((pred_cs_poisson - actual_cs) ** 2).mean()
    brier_xgb = ((pred_cs_xgb - actual_cs) ** 2).mean()
    print(f"  Current Poisson Brier={brier_poisson:.4f}, XGBoost Brier={brier_xgb:.4f} "
          f"-> {'XGBoost better' if brier_xgb < brier_poisson else 'XGBoost worse'} "
          f"({abs(brier_xgb-brier_poisson)/brier_poisson*100:.1f}%)")


def main():
    validate_cards()
    validate_bonus()
    validate_dc()
    validate_clean_sheets()
    print("\n" + "=" * 70)
    print("Same caution as before: a difference under roughly 5% could be noise "
          "rather than a real improvement — only trust a clear margin.")


if __name__ == "__main__":
    main()