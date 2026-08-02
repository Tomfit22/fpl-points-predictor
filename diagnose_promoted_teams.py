"""
FPL Points Predictor — Promoted Team Diagnostic
=====================================================
Arsenal vs Coventry City is a real fixture in fixtures_upcoming.csv,
but neither the "missing player" nor "zero PL history team" warning
fired in the live predictions run. This checks exactly why, at each
step of the chain, rather than guessing.

Run:
    python diagnose_promoted_teams.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROMOTED_TEAMS_TO_CHECK = ["Coventry City", "Hull City", "Ipswich Town"]


def main():
    print("=== Step 1: do these teams appear in fixtures_upcoming.csv? ===")
    fixtures = pd.read_csv(RAW_DIR / "fixtures_upcoming.csv")
    for team in PROMOTED_TEAMS_TO_CHECK:
        n = len(fixtures[(fixtures["home_team"] == team) | (fixtures["away_team"] == team)])
        print(f"  {team}: {n} fixtures")

    print("\n=== Step 2: do these teams appear in model_ready_dataset.csv (last season's data)? ===")
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    for team in PROMOTED_TEAMS_TO_CHECK:
        n = len(df[df["team"] == team])
        print(f"  {team}: {n} historical rows (should be 0 if genuinely promoted)")

    print(f"\n  Total unique teams in model_ready_dataset.csv: {df['team'].nunique()}")
    print(f"  Teams: {sorted(df['team'].unique())}")

    print("\n=== Step 3: do these teams' players appear in current_roster_snapshot.csv? ===")
    roster_path = PROCESSED_DIR / "current_roster_snapshot.csv"
    if roster_path.exists():
        roster = pd.read_csv(roster_path)
        for team in PROMOTED_TEAMS_TO_CHECK:
            team_players = roster[roster["team"] == team]
            print(f"  {team}: {len(team_players)} players in roster snapshot")
            if len(team_players) > 0:
                sample_ids = team_players["player_id"].head(3).tolist()
                print(f"    sample player_ids: {sample_ids}")
                # check if these IDs coincidentally already exist in df —
                # this would explain why add_missing_roster_players found nothing
                overlap = df[df["player_id"].isin(team_players["player_id"])]
                if len(overlap) > 0:
                    overlap_teams = overlap["team"].unique()
                    print(f"    *** {len(overlap)} of these player_ids ALREADY exist in "
                          f"model_ready_dataset.csv, under team(s): {list(overlap_teams)} — "
                          f"this is why they weren't flagged as 'missing': they're not missing, "
                          f"they're TAGGED WITH THE WRONG (OLD) TEAM. ***")
    else:
        print("  current_roster_snapshot.csv not found.")

    print("\n=== Step 4: did any live_predictions.csv rows actually involve these teams? ===")
    live_path = PROCESSED_DIR / "live_predictions.csv"
    if live_path.exists():
        live = pd.read_csv(live_path)
        for team in PROMOTED_TEAMS_TO_CHECK:
            n_as_team = len(live[live["team"] == team])
            print(f"  {team} as player's own team: {n_as_team} prediction rows")
    else:
        print("  live_predictions.csv not found.")


    print("\n=== Step 5: THE CRITICAL TEST — is player_id even stable across the season "
          "boundary for players who definitely did NOT change teams? ===")
    known_stars = ["Haaland", "Salah", "Palmer", "Saka"]
    for name in known_stars:
        old_rows = df[df["player_name"].str.contains(name, case=False, na=False)]
        new_rows = roster[roster["player_name"].str.contains(name, case=False, na=False)] if roster_path.exists() else pd.DataFrame()
        if old_rows.empty or new_rows.empty:
            print(f"  {name}: could not find in one or both datasets — skipping")
            continue
        old_id = old_rows["player_id"].iloc[0]
        new_id = new_rows["player_id"].iloc[0]
        old_team = old_rows["team"].iloc[0]
        new_team = new_rows["team"].iloc[0]
        match = "MATCH" if old_id == new_id else "*** MISMATCH ***"
        print(f"  {name}: old_id={old_id} ({old_team}) vs new_id={new_id} ({new_team}) — {match}")


if __name__ == "__main__":
    main()