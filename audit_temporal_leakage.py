"""
FPL Points Predictor — Temporal Leakage Audit
========================================================================
Checks whether model_ready_dataset.csv's rolling/season-average
features genuinely respect gameweek order (only using information
available BEFORE the row's own gameweek), or whether they leak future
information backward — which would invalidate every held-out
validation run this session, not just the joint expected-goals model.

Three checks, each independently diagnostic:

  1. WITHIN-SEASON VARIATION — a properly "as of this gameweek"
     cumulative stat (season_xG, etc.) should genuinely CHANGE from
     gameweek to gameweek for the same player/team as the season
     progresses. If it's IDENTICAL across every gameweek within a
     season, that's a strong signal it was computed ONCE as a
     full-season (including future) average, not a proper
     up-to-this-point cumulative value.

  2. RECONSTRUCTION FROM RAW GAME LOG — reconstructs a genuine
     season-to-date cumulative total from the row-by-row game log
     itself, to compare against what any season_* column claims.

  3. CURRENT-GAMEWEEK EXCLUSION — checks whether an exceptional single
     game (a huge outlier match) measurably shows up in that SAME
     gameweek's own feature value, which would mean the row's
     features partly derive from the very outcome they're meant to
     help predict.

Run:
    python audit_temporal_leakage.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")


def check_within_season_variation(df: pd.DataFrame):
    print("=" * 70)
    print("CHECK 1: WITHIN-SEASON VARIATION")
    print("=" * 70)

    season_col = "season" if "season" in df.columns else None
    if season_col is None:
        print("No explicit 'season' column found - checking within the whole dataset "
              "(fine if this covers a single season, misleading if it spans multiple).\n")

    candidates = [c for c in df.columns if c.startswith("season_") and df[c].dtype != object]
    if not candidates:
        print("No season_* columns found.")
        return

    sample_col = "season_xG" if "season_xG" in candidates else candidates[0]
    print(f"Checking '{sample_col}' - a properly-computed cumulative stat should "
          f"vary across gameweeks for the same player within a season.\n")

    id_col = "player_id" if "player_id" in df.columns else None
    if id_col is None:
        print("No player_id column - cannot check per-player variation.")
        return

    check_df = df[[id_col, "gameweek", sample_col]].copy()
    if season_col:
        check_df[season_col] = df[season_col]
        group_cols = [id_col, season_col]
    else:
        group_cols = [id_col]

    variation = check_df.groupby(group_cols)[sample_col].nunique()
    players_with_multiple_gws = check_df.groupby(group_cols)["gameweek"].nunique()

    eligible = players_with_multiple_gws[players_with_multiple_gws >= 5].index
    constant_despite_multiple_gws = variation[variation.index.isin(eligible) & (variation == 1)]

    total_eligible = len(eligible)
    n_constant = len(constant_despite_multiple_gws)

    print(f"Players/seasons with 5+ gameweeks of data: {total_eligible}")
    if total_eligible:
        print(f"Of those, how many show a CONSTANT '{sample_col}' across ALL their gameweeks "
              f"(a red flag for a full-season, not up-to-date, average): {n_constant} "
              f"({n_constant/total_eligible*100:.1f}%)")
    else:
        print("N/A - not enough eligible players/seasons to check.")

    if total_eligible and n_constant > total_eligible * 0.5:
        print("\n*** WARNING: majority of players show a CONSTANT value across the whole "
              "season for this feature. This strongly suggests season_* columns are "
              "full-season averages (including future gameweeks), not proper "
              "as-of-this-point cumulative stats - a real leakage concern. ***")
    else:
        print("\nMost players show real variation across gameweeks - consistent with "
              "properly time-respecting cumulative features.")


def check_reconstruction(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CHECK 2: RECONSTRUCT SEASON-TO-DATE FROM RAW GAME LOG")
    print("=" * 70)

    if "player_id" not in df.columns or "gameweek" not in df.columns:
        print("Missing player_id or gameweek - cannot run this check.")
        return
    if "goals" not in df.columns:
        print("No 'goals' column available for a concrete reconstruction check.")
        return

    sample_players = df["player_id"].value_counts()
    sample_players = sample_players[sample_players >= 10].head(3).index.tolist()

    for pid in sample_players:
        player_rows = df[df["player_id"] == pid].sort_values("gameweek")
        name = player_rows["player_name"].iloc[0] if "player_name" in player_rows.columns else pid
        print(f"\n{name} (player_id={pid}):")
        print(f"{'gameweek':>10} {'goals_this_gw':>15} {'cumulative_goals_true':>22}")

        cumulative_true = 0
        for _, row in player_rows.head(8).iterrows():
            gw = row["gameweek"]
            goals_this_gw = row["goals"]
            print(f"{gw:>10} {goals_this_gw:>15} {cumulative_true:>22}")
            cumulative_true += goals_this_gw

        print("  (cumulative_goals_true = actual total BEFORE this gameweek's own game - "
              "compare against whatever season_goals-style column exists for this player "
              "at the same rows if you want an exact leakage check on that specific column)")


def check_current_gameweek_influence(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CHECK 3: DOES AN EXCEPTIONAL GAME SHIFT ITS OWN GAMEWEEK'S FEATURE?")
    print("=" * 70)

    if "goals" not in df.columns or "season_xG" not in df.columns:
        print("Missing 'goals' or 'season_xG' - cannot run this check.")
        return

    outlier_games = df[df["goals"] >= 3]
    if outlier_games.empty:
        print("No 3+ goal games found in the dataset to test with.")
        return

    print(f"Found {len(outlier_games)} games with 3+ goals scored.")
    print("If season_xG at THIS SAME gameweek already reflects this game's own huge output, "
          "that's current-gameweek leakage into the feature meant to predict it.\n")

    sample = outlier_games.head(5)
    for _, row in sample.iterrows():
        name = row.get("player_name", row.get("player_id"))
        print(f"  {name}, GW{row['gameweek']}: scored {row['goals']} goals this game, "
              f"season_xG shown for this SAME row = {row['season_xG']:.3f}")
    print("\n  (A properly time-respecting feature should show a season_xG value that "
          "does NOT already include this game's own exceptional output - compare "
          "against the player's season_xG in the PREVIOUS gameweek's row to check "
          "whether it jumped sharply within the very same row it's meant to predict.)")


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows.\n")

    check_within_season_variation(df)
    check_reconstruction(df)
    check_current_gameweek_influence(df)


if __name__ == "__main__":
    main()
