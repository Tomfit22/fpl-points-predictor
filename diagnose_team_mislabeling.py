"""
FPL Points Predictor — Diagnose Team/Fixture Mislabeling
========================================================================
Identifies EXACTLY what the 321 individually-mislabeled rows represent,
before building any fix. Distinguishes between two fundamentally
different bugs:

  A) SAME real fixture split across multiple fixture_ids (a fixture-ID
     assignment bug) — e.g. Forest vs Chelsea appearing as fixture 101,
     102, AND 103, all genuinely the same match.

  B) Individual player rows carrying a STALE team label from before a
     transfer, sitting within an otherwise-correctly-labeled fixture
     (a player-level source-data bug) — e.g. Zinchenko's row showing
     "Nott'm Forest" inside an Arsenal vs Leeds fixture where every
     other Arsenal player is correctly labeled.

These need completely different fixes. Does NOT touch model_ready_dataset.csv
or any model — purely diagnostic.

Run:
    python diagnose_team_mislabeling.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def main():
    path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not path.exists():
        print(f"{path} not found.")
        return
    df = pd.read_csv(path)

    required = ["team", "fixture_id", "was_home_int", "opponent_team", "gameweek", "player_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return

    date_col = "match_date" if "match_date" in df.columns else ("kickoff_time" if "kickoff_time" in df.columns else None)

    print("=" * 70)
    print("PART 1: IDENTIFY EVERY MISLABELED ROW PRECISELY")
    print("=" * 70)

    mislabeled_rows = []
    for (fid, wh), group in df.groupby(["fixture_id", "was_home_int"]):
        if len(group) < 3:
            continue
        mode_team = group["team"].mode().iloc[0]
        disagreeing = group[group["team"] != mode_team]
        for _, row in disagreeing.iterrows():
            mislabeled_rows.append({
                "player_name": row["player_name"],
                "shown_team": row["team"],
                "expected_team": mode_team,
                "opponent_team": row["opponent_team"],
                "fixture_id": fid,
                "gameweek": row["gameweek"],
                "was_home_int": wh,
                "match_date": row[date_col] if date_col else None,
            })

    mislabeled_df = pd.DataFrame(mislabeled_rows)
    print(f"\nTotal mislabeled rows found: {len(mislabeled_df)}\n")

    print("=" * 70)
    print("PART 2: TEST HYPOTHESIS A - SAME REAL FIXTURE, MULTIPLE fixture_ids?")
    print("=" * 70)
    print("(If TRUE: the 'expected_team' vs 'shown_team' pairs, when matched by")
    print(" date+opponent, should reveal the SAME real match under different IDs)\n")

    if date_col:
        duplicate_fixture_evidence = 0
        for _, mrow in mislabeled_df.head(30).iterrows():
            same_date_fixtures = df[
                (df[date_col] == mrow["match_date"]) &
                (df["team"] == mrow["shown_team"]) &
                (df["opponent_team"] != mrow["opponent_team"])
            ]
            if not same_date_fixtures.empty:
                duplicate_fixture_evidence += 1

        print(f"Of the first 30 mislabeled rows checked, {duplicate_fixture_evidence} showed evidence "
              f"of the SAME team playing multiple DIFFERENT-opponent fixtures on the exact same date "
              f"(which would support hypothesis A - true fixture duplication).")
    else:
        print("No date column available - cannot test hypothesis A directly.")

    print("\n" + "=" * 70)
    print("PART 3: TEST HYPOTHESIS B - INDIVIDUAL PLAYER STALE-TEAM-LABEL BUG?")
    print("=" * 70)
    print("(If TRUE: mislabeled rows should cluster around a SMALL, REPEATED set")
    print(" of specific PLAYERS - consistent with mid-season transfers - rather")
    print(" than being spread evenly/randomly across many different players)\n")

    player_counts = mislabeled_df["player_name"].value_counts()
    print(f"Number of distinct players among the {len(mislabeled_df)} mislabeled rows: {len(player_counts)}")
    print(f"\nTop 15 players by mislabeled row count (repeated appearance = supports hypothesis B):")
    print(player_counts.head(15).to_string())

    concentration_ratio = player_counts.head(15).sum() / len(mislabeled_df) if len(mislabeled_df) else 0
    print(f"\nTop 15 players account for {concentration_ratio*100:.1f}% of all mislabeled rows.")
    if concentration_ratio > 0.5:
        print("HIGH concentration in a small number of players - strongly supports hypothesis B "
              "(individual player stale-team-label bug, e.g. mid-season transfers).")
    else:
        print("Mislabeled rows are spread across many different players - does not strongly "
              "support hypothesis B alone; worth checking hypothesis A more closely.")

    print("\n" + "=" * 70)
    print("PART 4: FOR THE TOP REPEATED PLAYERS, SHOW THEIR EXPECTED vs SHOWN TEAM")
    print("=" * 70)
    for player in player_counts.head(5).index:
        prows = mislabeled_df[mislabeled_df["player_name"] == player]
        print(f"\n{player}:")
        print(prows[["gameweek", "shown_team", "expected_team", "opponent_team"]].to_string(index=False))

    out_path = PROCESSED_DIR / "diagnosed_mislabeled_rows.csv"
    mislabeled_df.to_csv(out_path, index=False)
    print(f"\n\nFull list of {len(mislabeled_df)} mislabeled rows saved to {out_path} for further inspection.")


if __name__ == "__main__":
    main()
