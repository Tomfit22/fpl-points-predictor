"""
FPL Points Predictor — CLEAN SHEETS: calibration & baseline investigation
==============================================================================
Follow-up to build_clean_sheets_model.py. Tests three specific hypotheses
about why the model didn't beat baseline on Brier score, rather than
assuming any one of them is the answer:

  1. Was "naive baseline" too weak? A flat "always predict majority class"
     is a strawman. A fairer, stronger baseline: each team's OWN historical
     clean sheet rate (still zero modeling, just not a constant).

  2. Is this a CALIBRATION problem rather than a signal problem? Tests
     CalibratedClassifierCV with BOTH sigmoid (Platt) and isotonic — with
     only ~750 training rows, isotonic (nonparametric) risks overfitting,
     so both are tested rather than assuming isotonic is automatically
     better, and compared against the raw uncalibrated model.

  3. Is modeling goals CONCEDED via Poisson (then deriving P(0) from the
     fitted rate) better-calibrated than direct logistic classification?
     Poisson's P(0) comes from an actual distributional assumption about
     how scores are generated, which could plausibly calibrate better
     than an ad-hoc classifier decision boundary.

Run:
    python validate_clean_sheets_calibration.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DIR = Path("data/processed")
TRAIN_FRACTION = 0.75

CANDIDATE_POOL = [
    "own_season_goals_conceded", "own_roll5_goals_conceded",
    "own_season_xG_against", "own_roll5_xG_against",
    "own_season_shots_against", "own_roll5_shots_against",
    "opp_season_goals_scored", "opp_roll5_goals_scored",
    "opp_season_xG_for", "opp_roll5_xG_for",
    "opp_season_shots_for", "opp_roll5_shots_for",
    "was_home_int",
]


def build_team_match_table() -> pd.DataFrame:
    """Same fixed construction as build_clean_sheets_model.py — goals_conceded
    via max() per team-match, not an arbitrary drop_duplicates row."""
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    own_opp_cols = [c for c in CANDIDATE_POOL if c in df.columns and c != "was_home_int"]

    goals_conceded = df.groupby(["team", "fixture_id"])["goals_conceded"].max().reset_index()
    other_cols = df.groupby(["team", "fixture_id"])[
        ["gameweek", "opponent_team", "match_date", "was_home_int"] + own_opp_cols
    ].first().reset_index()

    team_df = goals_conceded.merge(other_cols, on=["team", "fixture_id"])
    team_df["kept_clean_sheet"] = (team_df["goals_conceded"] == 0).astype(int)
    return team_df.sort_values("gameweek")


def add_team_own_rate_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Each team's own expanding (season-to-date) clean sheet rate, using
    only STRICTLY PRIOR matches — a genuine, still-zero-modeling baseline,
    much stronger than a flat constant."""
    df = df.sort_values(["team", "gameweek"]).reset_index(drop=True)
    shifted = df.groupby("team")["kept_clean_sheet"].shift(1)
    df["team_own_cs_rate"] = (
        shifted.groupby(df["team"]).expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    return df


def brier(y_true, y_prob) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def investigation_1_stronger_baseline(train, test):
    print("=" * 70)
    print("INVESTIGATION 1: Is 'always predict majority class' too weak a baseline?")
    print("=" * 70)

    flat_baseline_prob = train["kept_clean_sheet"].mean()
    flat_brier = brier(test["kept_clean_sheet"], flat_baseline_prob)

    team_rate_test = test.dropna(subset=["team_own_cs_rate"])
    team_rate_brier = brier(team_rate_test["kept_clean_sheet"], team_rate_test["team_own_cs_rate"])

    print(f"Flat baseline (league-wide training rate, {flat_baseline_prob:.1%} for every row): "
          f"Brier = {flat_brier:.4f}")
    print(f"Team's-own-rate baseline (each team's own history, {len(team_rate_test)} rows with history): "
          f"Brier = {team_rate_brier:.4f}")
    if team_rate_brier < flat_brier:
        print("-> Yes, the team-rate baseline is genuinely stronger. THIS is the real bar to beat.")
    else:
        print("-> Surprisingly, team-rate baseline isn't better — flat baseline may be fine as-is.")
    return team_rate_brier


def investigation_2_calibration(train, test, features):
    print("\n" + "=" * 70)
    print("INVESTIGATION 2: Is this a calibration problem?")
    print("=" * 70)

    X_train, y_train = train[features].fillna(0), train["kept_clean_sheet"]
    X_test, y_test = test[features].fillna(0), test["kept_clean_sheet"]

    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    raw_proba = rf.predict_proba(X_test)[:, 1]
    raw_brier = brier(y_test, raw_proba)
    print(f"Uncalibrated Random Forest: Brier = {raw_brier:.4f}")

    # train vs test Brier — checks over/underfitting per the pasted advice's
    # point 5, before concluding anything about calibration specifically
    train_proba = rf.predict_proba(X_train)[:, 1]
    train_brier = brier(y_train, train_proba)
    print(f"  (train Brier = {train_brier:.4f} vs test Brier = {raw_brier:.4f} — "
          f"{'similar, so likely not overfitting' if abs(train_brier - raw_brier) < 0.02 else 'meaningfully different, check for overfitting'})")

    best_method, best_brier = "uncalibrated", raw_brier
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1),
                                             method=method, cv=5)
        calibrated.fit(X_train, y_train)
        cal_proba = calibrated.predict_proba(X_test)[:, 1]
        cal_brier = brier(y_test, cal_proba)
        print(f"Calibrated ({method}): Brier = {cal_brier:.4f}")
        if cal_brier < best_brier:
            best_method, best_brier = method, cal_brier

    print(f"\nBest approach: {best_method} (Brier = {best_brier:.4f})")

    # calibration curve — are predicted probabilities systematically off?
    print("\nCalibration curve (uncalibrated model) — predicted vs actual rate per bin:")
    prob_true, prob_pred = calibration_curve(y_test, raw_proba, n_bins=5, strategy="quantile")
    for pt, pp in zip(prob_true, prob_pred):
        direction = "OVERCONFIDENT" if pp > pt + 0.05 else ("UNDERCONFIDENT" if pp < pt - 0.05 else "reasonable")
        print(f"  predicted ~{pp:.2f} -> actual rate {pt:.2f}  [{direction}]")

    return best_brier


def investigation_3_poisson_goals_conceded(train, test, features):
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: Poisson on goals conceded, derive P(0) from the fitted rate")
    print("=" * 70)

    X_train = sm.add_constant(train[features].fillna(0))
    y_train = train["goals_conceded"]
    model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    X_test = sm.add_constant(test[features].fillna(0), has_constant="add")
    predicted_rate = model.predict(X_test)  # expected goals conceded
    predicted_p0 = np.exp(-predicted_rate)  # Poisson P(X=0) = e^-lambda

    poisson_brier = brier(test["kept_clean_sheet"], predicted_p0)
    print(f"Poisson-derived P(clean sheet): Brier = {poisson_brier:.4f}")
    return poisson_brier


def main():
    df = build_team_match_table()
    df = add_team_own_rate_baseline(df)
    df = df[df["own_roll5_goals_conceded"].notna()]

    cutoff_gw = df["gameweek"].quantile(TRAIN_FRACTION)
    train = df[df["gameweek"] <= cutoff_gw]
    test = df[df["gameweek"] > cutoff_gw]
    print(f"Train: {len(train)} | Test: {len(test)}\n")

    features = [f for f in CANDIDATE_POOL if f in df.columns]

    team_rate_brier = investigation_1_stronger_baseline(train, test)
    calibrated_brier = investigation_2_calibration(train, test, features)
    poisson_brier = investigation_3_poisson_goals_conceded(train, test, features)

    print("\n" + "=" * 70)
    print("SUMMARY — Brier scores, lower is better (all evaluated on the same held-out test set)")
    print("=" * 70)
    results = pd.Series({
        "Team's own historical rate (no modeling)": team_rate_brier,
        "Best calibrated classifier": calibrated_brier,
        "Poisson goals-conceded -> derived P(0)": poisson_brier,
    }).sort_values()
    print(results.to_string())
    print(f"\nBest overall approach: {results.index[0]}")


if __name__ == "__main__":
    main()