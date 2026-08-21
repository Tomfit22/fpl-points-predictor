"""
FPL Points Predictor — Cross-Season Clean Sheets Training Data
========================================================================
Builds a combined 24/25 + 25/26 dataset for the clean sheet model —
with an upfront honest limitation: the CURRENT model leans heavily on
possession and PPDA (pressing intensity), Understat/FBref-only stats
that genuinely don't exist in the FPL-only 24/25 archive. This uses a
SCOPED feature set — team-level goals conceded/scored, computable from
both seasons — same honest tradeoff as goals/assists' fpl_xG situation.

The clean sheet model is TEAM-level, not player-level like the others
in this batch — so this aggregates the 24/25 archive up to team-match
totals first (own goals conceded directly; own goals scored by summing
all of that team's players' goals in the match), then attaches the
OPPONENT's own attacking rate via the same team-lookup pattern already
validated for cards' opponent feature — same shift-first, no-leakage
rolling-window discipline throughout.

Run:
    python build_clean_sheets_training_data.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

WEIGHT_CURRENT_SEASON = 3.0
WEIGHT_HISTORICAL_SEASON = 1.0

CS_FEATURE_COLS = ["own_season_goals_conceded", "own_roll5_goals_conceded",
                    "opp_season_goals_scored", "was_home_int",
                    "team", "opponent_team", "gameweek", "season", "goals_conceded"]


def build_team_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, gameweek): goals conceded (a team-level stat,
    same for every player on that team that match — take the first)
    and goals scored (summed across all of that team's players)."""
    conceded = (
        df.groupby(["team", "gameweek", "opponent_team", "was_home_int", "season"])["goals_conceded"]
        .first().reset_index()
    )
    scored = (
        df.groupby(["team", "gameweek"])["goals_scored"].sum().reset_index()
        .rename(columns={"goals_scored": "team_total_goals_scored"})
    )
    team_matches = conceded.merge(scored, on=["team", "gameweek"], how="left")
    return team_matches


def add_rolling_team_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Own team's rolling/season goals-conceded rate, plus the
    opponent's own goals-scored rate (their attacking threat) looked
    up via the same cross-team pattern already validated for cards'
    opponent feature — shift-first throughout, no leakage."""
    team_matches = team_matches.sort_values(["team", "gameweek"]).copy()

    team_matches["own_season_goals_conceded"] = (
        team_matches.groupby("team")["goals_conceded"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    team_matches["own_roll5_goals_conceded"] = (
        team_matches.groupby("team")["goals_conceded"]
        .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    )
    team_matches["own_season_goals_scored"] = (
        team_matches.groupby("team")["team_total_goals_scored"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    # attach the OPPONENT's own attacking rate by looking up their
    # season goals-scored average under their own team name
    opp_attack = team_matches[["team", "gameweek", "own_season_goals_scored"]].rename(
        columns={"team": "opponent_team", "own_season_goals_scored": "opp_season_goals_scored"}
    )
    team_matches = team_matches.merge(opp_attack, on=["opponent_team", "gameweek"], how="left")

    return team_matches


def main():
    hist_path = PROCESSED_DIR / "historical_2024_25_reconciled.csv"
    if not hist_path.exists():
        print(f"{hist_path} not found — run build_historical_season_data.py first.")
        return
    hist = pd.read_csv(hist_path)
    hist = hist[hist["player_id"].notna()].copy()

    required = ["opponent_team", "goals_conceded", "goals_scored"]
    missing_req = [c for c in required if c not in hist.columns]
    if missing_req:
        print(f"historical_2024_25_reconciled.csv is missing {missing_req} — "
              f"check build_historical_season_data.py's KEEP_COLUMNS.")
        return

    hist_team = build_team_match_data(hist)
    hist_team = add_rolling_team_features(hist_team)
    hist_team["sample_weight"] = WEIGHT_HISTORICAL_SEASON

    current_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not current_path.exists():
        print(f"{current_path} not found — run the main pipeline first.")
        return
    current = pd.read_csv(current_path).copy()
    current["season"] = "2025-26"
    current["sample_weight"] = WEIGHT_CURRENT_SEASON

    missing = [c for c in CS_FEATURE_COLS if c not in current.columns]
    if missing:
        print(f"(Note: {missing} not found in current data — check column names.)")

    combined_cols = [c for c in CS_FEATURE_COLS if c in hist_team.columns and c in current.columns]
    if len(combined_cols) < 3:
        print(f"Too few shared columns ({combined_cols}) — clean sheets cross-season "
              f"enhancement isn't viable with what's available. Recommend leaving "
              f"the current model as-is.")
        return

    combined = pd.concat([
        hist_team[combined_cols + ["sample_weight"]],
        current[combined_cols + ["sample_weight"]],
    ], ignore_index=True)

    before = len(combined)
    combined = combined.dropna(subset=["own_season_goals_conceded"])
    print(f"Dropped {before - len(combined)} rows with no goals-conceded history yet.")

    output_path = PROCESSED_DIR / "clean_sheets_training_combined.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path} ({len(combined)} rows)")
    print(f"  2024-25 rows: {(combined['season'] == '2024-25').sum()} (weight {WEIGHT_HISTORICAL_SEASON})")
    print(f"  2025-26 rows: {(combined['season'] == '2025-26').sum()} (weight {WEIGHT_CURRENT_SEASON})")
    if "opp_season_goals_scored" in combined.columns:
        print(f"  Rows with opponent attack feature present: {combined['opp_season_goals_scored'].notna().sum()}")
    print(f"\nColumns available for modeling: {combined_cols}")


if __name__ == "__main__":
    main()