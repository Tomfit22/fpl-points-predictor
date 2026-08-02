"""
FPL Points Predictor — Position History Snapshot
=======================================================
FPL occasionally reclassifies players between positions (11 confirmed
changes for 2026/27: Marmoush MID->FWD, Dorgu DEF->MID, Lewis-Skelly
DEF->MID, Kroupi FWD->MID, and others). Our extraction pulls each
player's CURRENT position and stamps it onto ALL their historical rows
— meaning a reclassified player's PAST season, played under their OLD
position, gets silently relabeled with their NEW position the next
time we re-extract. This contaminates position-specific model training
with historical rows that don't reflect the role the player was
actually playing at the time.

This script freezes a snapshot of (player_id, gameweek, position) from
the CURRENT merged dataset — capturing positions as they stand right
now, before any reclassification has overwritten them — into a
persisted history file. build_merged_dataset.py then uses this
snapshot to correct historical rows' positions on every future run,
while genuinely NEW gameweeks still correctly reflect a player's
current classification.

RUN THIS NOW, BEFORE re-running extract_fpl_clean_dataset.py once the
new season's data goes live — this is the window where your existing
data still reflects the correct pre-reclassification positions.

Run:
    python build_position_history.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
HISTORY_PATH = PROCESSED_DIR / "position_history.csv"


def main():
    merged_path = PROCESSED_DIR / "merged_player_gameweek.csv"
    if not merged_path.exists():
        print(f"{merged_path} not found — run build_merged_dataset.py first.")
        return

    df = pd.read_csv(merged_path)
    snapshot = df[["player_id", "gameweek", "position"]].drop_duplicates()

    if HISTORY_PATH.exists():
        existing = pd.read_csv(HISTORY_PATH)
        # keep existing entries for (player_id, gameweek) pairs already
        # recorded — never let a later snapshot silently overwrite an
        # earlier, presumably-correct-at-the-time entry
        combined = pd.concat([existing, snapshot], ignore_index=True)
        combined = combined.drop_duplicates(subset=["player_id", "gameweek"], keep="first")
        n_new = len(combined) - len(existing)
        print(f"Existing history had {len(existing)} entries. "
              f"Adding {n_new} new (player_id, gameweek) entries not previously recorded.")
        combined.to_csv(HISTORY_PATH, index=False)
    else:
        snapshot.to_csv(HISTORY_PATH, index=False)
        print(f"Created new position history with {len(snapshot)} entries.")

    print(f"Saved -> {HISTORY_PATH}")


if __name__ == "__main__":
    main()