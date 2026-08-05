"""
FPL Points Predictor — Missing Players Diagnostic
=======================================================
Established 25/26 players are reportedly missing from the 26/27
dashboard entirely — not flagged as "new signings," just absent. This
finds exactly which current-roster players don't make it into the
final predictions, and checks each of the likely causes:

  1. Are they missing from current_roster_snapshot.csv itself (stale
     roster snapshot)?
  2. Do they exist in historical data under ANY name, but reconciliation
     failed to link them to their current roster entry?
  3. Are they being dropped by the roll5_minutes filter at the top of
     the pipeline?
  4. Are they genuinely new (correctly excluded from history, but
     SHOULD have been added by add_missing_roster_players)?

Run:
    python diagnose_missing_players.py
"""

from pathlib import Path

import pandas as pd

import build_live_predictions as blp

PROCESSED_DIR = Path("data/processed")


def main():
    roster = pd.read_csv(PROCESSED_DIR / "current_roster_snapshot.csv")
    print(f"Current roster snapshot: {len(roster)} players")

    df_raw = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df_filtered = df_raw[df_raw["roll5_minutes"].notna()]
    print(f"model_ready_dataset.csv: {len(df_raw)} total rows, "
          f"{len(df_filtered)} after roll5_minutes filter")
    print(f"  Unique players in raw data: {df_raw['player_id'].nunique()}")
    print(f"  Unique players after filter: {df_filtered['player_id'].nunique()}")

    dropped_entirely = set(df_raw["player_id"]) - set(df_filtered["player_id"])
    if dropped_entirely:
        print(f"\n*** {len(dropped_entirely)} players have EVERY row dropped by the "
              f"roll5_minutes filter (meaning they have NO valid rolling-feature row at "
              f"all) — these would be invisible to the rest of the pipeline: ***")
        sample = df_raw[df_raw["player_id"].isin(dropped_entirely)]["player_name"].unique()
        print(f"  {list(sample)[:20]}")

    reconciled = blp.reconcile_player_ids(df_filtered, roster)
    final_snapshot = blp.get_latest_player_snapshot(reconciled)
    final_with_missing = blp.add_missing_roster_players(final_snapshot, reconciled)

    final_ids = set(final_with_missing["player_id"])
    roster_ids = set(roster["player_id"])
    truly_missing = roster_ids - final_ids

    print(f"\n=== FINAL CHECK ===")
    print(f"Roster: {len(roster_ids)} players")
    print(f"Final predictions cover: {len(final_ids)} players")
    print(f"Genuinely missing from predictions: {len(truly_missing)}")

    if truly_missing:
        missing_rows = roster[roster["player_id"].isin(truly_missing)]
        print(f"\nMissing players (showing up to 30):")
        for _, row in missing_rows.head(30).iterrows():
            # check if this exact name exists SOMEWHERE in historical data,
            # under a DIFFERENT id — tells us if reconciliation should have
            # caught this but didn't
            norm = blp.normalize_name(row["player_name"])
            hist_matches = df_filtered[df_filtered["player_name"].apply(blp.normalize_name) == norm]
            if len(hist_matches) > 0:
                hist_ids = hist_matches["player_id"].unique().tolist()
                print(f"  {row['player_name']} ({row['team']}) — HAS history under "
                      f"id(s) {hist_ids} but wasn't reconciled to their current id "
                      f"({row['player_id']})")
            else:
                print(f"  {row['player_name']} ({row['team']}) — NO historical match "
                      f"found by exact name at all")


if __name__ == "__main__":
    main()