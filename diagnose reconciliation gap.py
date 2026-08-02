"""
FPL Points Predictor — Reconciliation/Missing-Player Inconsistency Diagnostic
===================================================================================
Real run showed 457 players reconciled by name, but ALSO showed obviously
established players (Saka, Gabriel, Rice) flagged as having "no history
at all" — these two findings shouldn't both be true for the same
players. This checks exactly where the inconsistency is, for a specific
set of known-established players.

Run:
    python diagnose_reconciliation_gap.py
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
CHECK_NAMES = ["Saka", "Gabriel", "Rice", "Ødegaard", "Martinelli"]


def normalize_name(name: str) -> str:
    import unicodedata
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    roster = pd.read_csv(PROCESSED_DIR / "current_roster_snapshot.csv")

    for name in CHECK_NAMES:
        print(f"\n=== {name} ===")

        old_rows = df[df["player_name"].str.contains(name, case=False, na=False)]
        new_rows = roster[roster["player_name"].str.contains(name, case=False, na=False)]

        if old_rows.empty:
            print(f"  NOT FOUND in model_ready_dataset.csv at all under this name.")
        else:
            old_ids = old_rows["player_id"].unique()
            print(f"  Found in OLD data: player_id(s) {list(old_ids)}, "
                  f"{len(old_rows)} rows, roll5_minutes non-null: {old_rows['roll5_minutes'].notna().sum()}")

        if new_rows.empty:
            print(f"  NOT FOUND in current_roster_snapshot.csv at all under this name.")
        else:
            new_ids = new_rows["player_id"].unique()
            print(f"  Found in ROSTER: player_id(s) {list(new_ids)}, team(s): {new_rows['team'].unique().tolist()}")

        if not old_rows.empty and not new_rows.empty:
            old_norm = set(old_rows["player_name"].apply(normalize_name))
            new_norm = set(new_rows["player_name"].apply(normalize_name))
            print(f"  Normalized OLD name(s): {old_norm}")
            print(f"  Normalized NEW name(s): {new_norm}")
            print(f"  Exact normalized match? {bool(old_norm & new_norm)}")


if __name__ == "__main__":
    main()