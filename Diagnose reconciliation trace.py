"""
FPL Points Predictor — Reconciliation Trace Diagnostic (v2)
==================================================================
The name-matching itself looks clean for Saka/Gabriel/Rice/Ødegaard/
Martinelli (single unambiguous IDs on both sides) — so the bug must be
in how the reconciliation result gets used afterward, not in the
matching logic itself. This actually RUNS reconcile_player_ids() on
your real data and traces exactly what happens to these players.

Run:
    python diagnose_reconciliation_trace.py
"""

from pathlib import Path

import pandas as pd

import build_live_predictions as blp

PROCESSED_DIR = Path("data/processed")
CHECK = [
    ("Saka", 16, 12),
    ("Gabriel", 5, 4),
    ("Rice", 21, 13),
    ("Ødegaard", 17, 15),
    ("Martinelli", 19, 18),
]


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[df["roll5_minutes"].notna()]
    roster = pd.read_csv(PROCESSED_DIR / "current_roster_snapshot.csv")

    print(f"Rows before reconciliation: {len(df)}")
    reconciled = blp.reconcile_player_ids(df, roster)
    print(f"Rows after reconciliation: {len(reconciled)}")

    print("\n=== Checking each player's actual post-reconciliation state ===")
    for name, old_id, new_id in CHECK:
        rows_at_new_id = reconciled[reconciled["player_id"] == new_id]
        rows_at_old_id = reconciled[reconciled["player_id"] == old_id]
        print(f"\n{name}: expected old_id={old_id} -> new_id={new_id}")
        print(f"  Rows now at new_id ({new_id}): {len(rows_at_new_id)}")
        if len(rows_at_new_id) > 0:
            print(f"    names there: {rows_at_new_id['player_name'].unique().tolist()}")
        print(f"  Rows still at old_id ({old_id}): {len(rows_at_old_id)}")
        if len(rows_at_old_id) > 0:
            print(f"    names there: {rows_at_old_id['player_name'].unique().tolist()}")

    print("\n=== Now checking get_latest_player_snapshot on the reconciled data ===")
    snapshot = blp.get_latest_player_snapshot(reconciled)
    print(f"Snapshot has {len(snapshot)} players")
    for name, old_id, new_id in CHECK:
        in_snapshot = new_id in set(snapshot["player_id"])
        print(f"  {name} (id {new_id}) in snapshot: {in_snapshot}")


if __name__ == "__main__":
    main()