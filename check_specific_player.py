"""
FPL Points Predictor — Single Missing Player Trace
========================================================
Checks exactly why a specific named player isn't showing up in final
predictions — whether they're missing from the roster snapshot itself,
missing from historical data, or getting dropped/mismatched somewhere
in between.

Run:
    python check_specific_player.py "Munoz"
"""

import sys
from pathlib import Path

import pandas as pd

import build_live_predictions as blp

PROCESSED_DIR = Path("data/processed")


def main():
    if len(sys.argv) < 2:
        print('Usage: python check_specific_player.py "Player Name"')
        return
    name = sys.argv[1]
    norm_query = blp.normalize_name(name)

    roster = pd.read_csv(PROCESSED_DIR / "current_roster_snapshot.csv")
    roster_matches = roster[roster["player_name"].apply(blp.normalize_name).str.contains(norm_query, na=False)]
    print(f"=== In current_roster_snapshot.csv ({len(roster)} total players) ===")
    if roster_matches.empty:
        print(f"  NOT FOUND — '{name}' doesn't appear in the roster snapshot at all. "
              f"Either the name is spelled differently, or the snapshot is stale "
              f"(re-run build_current_roster_snapshot.py).")
        return
    print(roster_matches[["player_id", "player_name", "team", "position", "price"]].to_string(index=False))

    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[df["roll5_minutes"].notna()]
    hist_matches = df[df["player_name"].apply(blp.normalize_name).str.contains(norm_query, na=False)]
    print(f"\n=== In model_ready_dataset.csv (historical data) ===")
    if hist_matches.empty:
        print(f"  NOT FOUND — genuinely no historical PL data under this name. Should be "
              f"caught by add_missing_roster_players() as a new signing.")
    else:
        print(f"  Found under player_id(s): {hist_matches['player_id'].unique().tolist()}, "
              f"team(s): {hist_matches['team'].unique().tolist()}")

    roster_for_reconciliation = roster
    reconciled = blp.reconcile_player_ids(df, roster_for_reconciliation)
    snapshot = blp.get_latest_player_snapshot(reconciled)
    snapshot_matches = snapshot[snapshot["player_name"].apply(blp.normalize_name).str.contains(norm_query, na=False)]
    print(f"\n=== In player_snapshot after reconciliation ===")
    if snapshot_matches.empty:
        print(f"  NOT in the reconciled snapshot — check the reconciliation output above for "
              f"an 'ambiguous' warning naming this player, which would explain why.")
    else:
        print(snapshot_matches[["player_id", "player_name", "team"]].to_string(index=False))

    final_ids_from_roster = set(roster["player_id"])
    filtered = snapshot[snapshot["player_id"].isin(final_ids_from_roster)]
    filtered_matches = filtered[filtered["player_name"].apply(blp.normalize_name).str.contains(norm_query, na=False)]
    print(f"\n=== After filtering to current roster IDs only ===")
    if filtered_matches.empty:
        print(f"  MISSING at this stage — their reconciled player_id doesn't match any "
              f"current roster ID, meaning reconciliation genuinely failed to link them.")
    else:
        print(f"  Present: {filtered_matches[['player_id', 'player_name', 'team']].to_string(index=False)}")

    with_missing_added = blp.add_missing_roster_players(filtered, df)
    final_matches = with_missing_added[with_missing_added["player_name"].apply(blp.normalize_name).str.contains(norm_query, na=False)]
    print(f"\n=== FINAL: after add_missing_roster_players ===")
    if final_matches.empty:
        print(f"  Still MISSING — genuinely not appearing anywhere in the final pipeline. "
              f"This needs investigation of why the name search itself isn't matching "
              f"(check spelling/formatting differences between sources).")
    else:
        print(f"  Present in final output: {final_matches[['player_id', 'player_name', 'team']].to_string(index=False)}")


if __name__ == "__main__":
    main()