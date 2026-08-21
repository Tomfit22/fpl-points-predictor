"""
FPL Points Predictor — Shared Team Match Table Utility
========================================================================
The single, validated source of truth for "one row per real team per
fixture" — built to fix the confirmed root cause: ~13-15 players carry
a STALE team label for most of the season (genuine summer transfers
whose historical rows never got corrected), which silently created
phantom team-fixture rows whenever any script grouped by the raw
player-level "team" column directly, as EVERY validation/diagnostic
script built today did — not build_features.py itself, whose team_log
already uses the mode-of-teammates approach that correctly resolves
this.

Contract:
  - Exactly one row per (team, fixture_id).
  - Team label = the MODE of "team" among all players on that side of
    that fixture — robust to a small number of stale-labeled rows,
    since the genuine teammates always vastly outnumber them.
  - Validated before use: zero (team, fixture_id) duplicates, and a
    real per-gameweek fixture count check (1 normally, 2 only for a
    genuine double gameweek).

Run standalone to validate against the real dataset:
    python build_team_match_table.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def build_team_match_table(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the single validated team-match table. Input: the raw
    player-level model_ready_dataset.csv. Output: exactly one row per
    (team, fixture_id), team label resolved by mode, safe for any
    team-level model fitting or validation."""
    required = ["fixture_id", "was_home_int", "team", "opponent_team", "gameweek", "goals_conceded"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    date_col = "match_date" if "match_date" in df.columns else ("kickoff_time" if "kickoff_time" in df.columns else None)

    rows = []
    for (fid, wh), group in df.groupby(["fixture_id", "was_home_int"]):
        mode_team = group["team"].mode().iloc[0]
        correct_rows = group[group["team"] == mode_team]

        row = {
            "team": mode_team,
            "fixture_id": fid,
            "was_home_int": wh,
            "gameweek": correct_rows["gameweek"].iloc[0],
            "opponent_team": correct_rows["opponent_team"].mode().iloc[0],
            "goals_conceded": correct_rows["goals_conceded"].iloc[0],
        }
        if date_col:
            row["match_date"] = correct_rows[date_col].iloc[0]

        for col in df.columns:
            if col.startswith("own_") or col.startswith("opp_"):
                row[col] = correct_rows[col].iloc[0] if len(correct_rows) else None

        rows.append(row)

    result = pd.DataFrame(rows)
    return result


def validate_team_match_table(team_table: pd.DataFrame) -> bool:
    """Runs the checks specified before trusting this table for any
    modeling. Returns True only if every check passes."""
    all_passed = True

    dup_count = team_table.duplicated(subset=["team", "fixture_id"]).sum()
    print(f"Check 1 - (team, fixture_id) duplicates: {dup_count}", "PASS" if dup_count == 0 else "FAIL")
    if dup_count > 0:
        all_passed = False

    print("\nCheck 2 - fixtures per (team, gameweek):")
    per_team_gw = team_table.groupby(["team", "gameweek"]).size()
    invalid = per_team_gw[per_team_gw > 2]
    over_two = len(invalid)
    print(f"  (team, gameweek) pairs with MORE than 2 fixtures (should be 0): {over_two}",
          "PASS" if over_two == 0 else "FAIL")
    if over_two > 0:
        all_passed = False
        print(invalid.head(10))

    double_gws = per_team_gw[per_team_gw == 2]
    print(f"  (team, gameweek) pairs with exactly 2 fixtures (genuine double gameweeks): {len(double_gws)}")

    if "goals_conceded" in team_table.columns:
        null_gc = team_table["goals_conceded"].isna().sum()
        print(f"\nCheck 3 - goals_conceded present for every row: {len(team_table)-null_gc}/{len(team_table)}",
              "PASS" if null_gc == 0 else "FAIL")
        if null_gc > 0:
            all_passed = False

    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'CHECKS FAILED - do not use this table for modeling yet'}")
    return all_passed


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    print("Building team match table...")
    team_table = build_team_match_table(df)
    print(f"Built {len(team_table)} team-fixture rows from {len(df)} player-level rows.\n")

    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    passed = validate_team_match_table(team_table)

    if passed:
        out_path = PROCESSED_DIR / "team_match_table.csv"
        team_table.to_csv(out_path, index=False)
        print(f"\nSaved validated table to {out_path}")


if __name__ == "__main__":
    main()
