"""
FPL Points Predictor — Cold-Start False-Positive Diagnostic
=================================================================
41.5% of players (349/840) got flagged as "insufficient PL history" in
a real run, including established stars (Rashford, Luis Diaz,
Gundogan, Son) who definitely played real minutes all season. This
checks the most likely cause: a player_id mismatch between historical
rows and the current snapshot, possibly from the 2026/27 season
transition changing how IDs are assigned.

Run:
    python diagnose_cold_start_bug.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")

# a few of the specific players wrongly flagged in the real run — check these by name
SUSPECT_NAMES = ["Rashford", "Díaz", "Gündoğan", "Son", "Kulusevski", "Onana", "Nkunku"]


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")

    print(f"Total rows: {len(df)}")
    print(f"player_id dtype: {df['player_id'].dtype}")
    print(f"Any null player_id: {df['player_id'].isna().sum()}")
    print(f"Unique player_id count: {df['player_id'].nunique()}")
    print(f"Unique player_name count: {df['player_name'].nunique()}")

    if df['player_id'].nunique() != df['player_name'].nunique():
        print("\n*** MISMATCH: number of unique player_ids differs from number of unique "
              "player_names — this alone doesn't prove a bug (some names are genuinely "
              "shared by different real players), worth checking specific cases below. ***\n")

    print("\n=== Checking specific wrongly-flagged players by EXACT name match ===")
    for name in SUSPECT_NAMES:
        # exact match, not substring — a loose .str.contains("Son") would
        # incorrectly also match Anderson/Wilson/Robertson/Henderson/etc,
        # manufacturing a false "multiple IDs" signal that isn't real
        matches = df[df["player_name"] == name]
        if matches.empty:
            print(f"\n{name}: no EXACT match found — trying partial match for reference only:")
            partial = df[df["player_name"].str.contains(name, case=False, na=False)]
            print(f"  {partial['player_name'].unique().tolist()[:10]}")
            continue

        unique_ids = matches["player_id"].unique()
        print(f"\n{name}: {len(matches)} total rows, player_id(s) used: {list(unique_ids)}")

        for pid in unique_ids:
            pid_rows = matches[matches["player_id"] == pid]
            games_with_minutes = (pid_rows["minutes"] > 0).sum()
            gw_range = f"{pid_rows['gameweek'].min()}-{pid_rows['gameweek'].max()}"
            print(f"  player_id={pid}: {len(pid_rows)} rows, "
                  f"{games_with_minutes} with minutes>0, gameweeks {gw_range}")

        if len(unique_ids) > 1:
            print(f"  *** {name} has MULTIPLE player_ids across their rows — this is the bug. "
                  f"Their real minutes are being split across different IDs, so no single ID "
                  f"has enough recorded games to clear the cold-start threshold, even though "
                  f"the PLAYER genuinely has plenty of history. ***")


if __name__ == "__main__":
    main()