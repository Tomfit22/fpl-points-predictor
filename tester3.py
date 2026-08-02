"""
FPL Points Predictor — Henderson Full-Season Check
========================================================
Claim: Henderson played every minute for Crystal Palace last season,
yet the model predicts only 91% chance of 60+ minutes — seems low for
someone with a flawless record. This checks the FULL season's raw
minutes (not just last 5 games) to confirm the claim, and inspects
what's specifically pulling his predicted probability down from what
a perfect record would suggest.

Run:
    python check_henderson.py
"""

from pathlib import Path

import pandas as pd

import build_live_predictions as blp

PROCESSED_DIR = Path("data/processed")


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[df["roll5_minutes"].notna()]
    roster = pd.read_csv(PROCESSED_DIR / "current_roster_snapshot.csv")
    reconciled = blp.reconcile_player_ids(df, roster)

    matches = reconciled[reconciled["player_name"].str.contains("Henderson", case=False, na=False)]
    if matches.empty:
        print("Henderson not found.")
        return

    for pid in matches["player_id"].unique():
        player_rows = reconciled[reconciled["player_id"] == pid].sort_values("gameweek")
        name = player_rows["player_name"].iloc[0]
        team = player_rows["team"].iloc[0]
        print(f"\n=== {name} ({team}, player_id={pid}) — FULL season raw minutes ===")
        print(f"Games played: {len(player_rows)}")
        print(f"Games with 60+ minutes: {(player_rows['minutes'] >= 60).sum()} / {len(player_rows)}")
        print(f"Games with 0 minutes: {(player_rows['minutes'] == 0).sum()}")
        print(f"Mean minutes per game: {player_rows['minutes'].mean():.1f}")

        last_row = player_rows.iloc[-1]
        print(f"\nLast row (used for the live prediction), gameweek {last_row['gameweek']}:")
        for col in ["roll5_minutes", "roll5_starts", "consecutive_starts", "days_since_last_game",
                    "season_minutes", "was_home_int"]:
            if col in last_row.index:
                print(f"  {col}: {last_row[col]}")


if __name__ == "__main__":
    main()