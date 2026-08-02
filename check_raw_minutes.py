"""
FPL Points Predictor — Raw Minutes Check (settles real-data vs bug)
=========================================================================
Salah/Saka/Robertson/Wan-Bissaka show low ROLLING minutes stats at
gameweek 38 (0.4-0.6 roll5_starts). Row selection is confirmed correct.
This shows the RAW, per-game minutes for their last 5 real gameweeks —
if the raw numbers genuinely show reduced minutes, that's real data
(possibly genuine end-of-season rotation), not a bug. If the raw
numbers look normal but the ROLLING average doesn't reflect them,
that's a real computation bug in build_features.py.

Run:
    python check_raw_minutes.py
"""

from pathlib import Path

import pandas as pd

import build_live_predictions as blp

PROCESSED_DIR = Path("data/processed")
CHECK_NAMES = ["Salah", "Saka", "Robertson", "Wan-Bissaka"]


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[df["roll5_minutes"].notna()]
    roster = pd.read_csv(PROCESSED_DIR / "current_roster_snapshot.csv")
    reconciled = blp.reconcile_player_ids(df, roster)

    for name in CHECK_NAMES:
        matches = reconciled[reconciled["player_name"].str.contains(name, case=False, na=False)]
        if matches.empty:
            continue
        for pid in matches["player_id"].unique():
            player_rows = reconciled[reconciled["player_id"] == pid].sort_values("gameweek")
            last5 = player_rows.tail(5)
            print(f"\n=== {player_rows['player_name'].iloc[0]} (player_id={pid}) — last 5 real gameweeks ===")
            cols = [c for c in ["gameweek", "minutes", "starts", "roll5_minutes", "roll5_starts"] if c in last5.columns]
            print(last5[cols].to_string(index=False))


if __name__ == "__main__":
    main()