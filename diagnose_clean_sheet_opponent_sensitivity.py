"""
FPL Points Predictor — Diagnostic: Clean Sheet Model Opponent Sensitivity
========================================================================
Investigates why pred_goals_conceded barely differs between genuinely
different-strength opponents (e.g. facing Ipswich vs facing Man City) —
checking the ACTUAL FITTED COEFFICIENTS of the clean sheet model, not
just the output data, since CS_FEATURES includes several features that
are naturally highly correlated with each other (opp_season_goals_scored,
opp_season_xG_for, opp_season_shots_for all move together for a team
that generally attacks well) — the same class of multicollinearity
issue already found and fixed for the minutes model earlier in this
project (roll5_starts/roll5_minutes). Without correlation pruning, a
GLM can spread its coefficient weight thin across correlated features,
weakening each one's apparent individual effect even if the combined
signal is real.

Also directly checks whether players facing genuinely elite attacking
teams (Man City, Arsenal, Liverpool) show a real, expected difference
in predicted goals conceded compared to players facing weaker sides.

Run:
    python diagnose_clean_sheet_opponent_sensitivity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")

CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]

ELITE_TEAMS = ["Man City", "Arsenal", "Liverpool"]


def check_feature_correlations():
    print("=" * 70)
    print("PART 1: CORRELATION BETWEEN CS_FEATURES")
    print("=" * 70)
    print("(Checking for the same class of multicollinearity issue already")
    print("found and fixed for the minutes model earlier in this project)\n")

    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return None
    df = pd.read_csv(path)

    available = [f for f in CS_FEATURES if f in df.columns]
    if len(available) < 2:
        print("Not enough CS_FEATURES present to check correlations.")
        return None

    corr = df[available].corr()
    print("Correlations above 0.7 (potential multicollinearity):\n")
    found_any = False
    for i, f1 in enumerate(available):
        for f2 in available[i+1:]:
            c = corr.loc[f1, f2]
            if abs(c) > 0.7:
                found_any = True
                print(f"  {f1} <-> {f2}: {c:.3f}")
    if not found_any:
        print("  None found above 0.7 - multicollinearity doesn't look like the cause.")
    return df


def check_fitted_coefficients(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PART 2: ACTUAL FITTED CLEAN SHEET MODEL COEFFICIENTS")
    print("=" * 70)

    if "fixture_id" not in df.columns or "goals_conceded" not in df.columns:
        print("Missing fixture_id or goals_conceded.")
        return

    team_df = df.groupby(["team", "fixture_id"]).first().reset_index()
    available = [f for f in CS_FEATURES if f in team_df.columns]
    variances = team_df[available].var(numeric_only=True)
    features = [f for f in available if variances.get(f, 1) > 0]

    X = sm.add_constant(team_df[features].fillna(0))
    model = sm.GLM(team_df["goals_conceded"], X, family=sm.families.Poisson()).fit()

    print("\nCoefficient, p-value, and whether the SIGN makes real football sense:")
    print("(positive = more goals conceded; a HIGHER opp_season_goals_scored SHOULD")
    print("have a POSITIVE coefficient - more prolific opponent, more goals against us)\n")

    for feat in features:
        coef = model.params.get(feat, float("nan"))
        pval = model.pvalues.get(feat, float("nan"))
        expected_sign = "+" if feat.startswith("opp_") and ("scored" in feat or "xG_for" in feat or "shots_for" in feat) else "?"
        actual_sign = "+" if coef > 0 else "-"
        flag = ""
        if expected_sign == "+" and actual_sign == "-":
            flag = "  <-- BACKWARDS from what football logic expects"
        significance = "significant" if pval < 0.05 else "NOT statistically significant"
        print(f"  {feat:<28} coef={coef:+.4f}  p={pval:.4f}  ({significance}){flag}")


def check_elite_vs_weak_opponent(live_path: Path):
    print("\n" + "=" * 70)
    print("PART 3: REAL DIFFERENTIATION - ELITE vs WEAK OPPONENTS")
    print("=" * 70)

    if not live_path.exists():
        print(f"{live_path} not found.")
        return
    live_df = pd.read_csv(live_path)
    if "pred_goals_conceded" not in live_df.columns or "opponent_team" not in live_df.columns:
        print("Missing pred_goals_conceded or opponent_team in live_predictions.csv.")
        return

    def_gk = live_df[live_df["position"].isin(["GK", "DEF"])].drop_duplicates(["player_id", "gameweek"])

    vs_elite = def_gk[def_gk["opponent_team"].isin(ELITE_TEAMS)]
    vs_other = def_gk[~def_gk["opponent_team"].isin(ELITE_TEAMS)]

    print(f"\nFacing an elite attacking side ({', '.join(ELITE_TEAMS)}):")
    print(f"  Rows: {len(vs_elite)}, mean pred_goals_conceded: {vs_elite['pred_goals_conceded'].mean():.3f}")

    print(f"\nFacing anyone else:")
    print(f"  Rows: {len(vs_other)}, mean pred_goals_conceded: {vs_other['pred_goals_conceded'].mean():.3f}")

    if len(vs_elite) > 0 and len(vs_other) > 0:
        diff = vs_elite["pred_goals_conceded"].mean() - vs_other["pred_goals_conceded"].mean()
        pct = diff / vs_other["pred_goals_conceded"].mean() * 100
        print(f"\nDifference: {diff:+.3f} ({pct:+.1f}%)")
        if abs(pct) < 15:
            print("This is a SMALL difference for facing elite attacking sides vs everyone else -")
            print("confirms the real problem you spotted, not just a display artifact.")
        else:
            print("This is a meaningful difference - the model IS differentiating by opponent strength,")
            print("the flat-looking table may reflect a specific subset of similar-strength opponents.")


if __name__ == "__main__":
    df = check_feature_correlations()
    if df is not None:
        check_fitted_coefficients(df)
    check_elite_vs_weak_opponent(PROCESSED_DIR / "live_predictions.csv")
