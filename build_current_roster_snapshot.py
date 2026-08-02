"""
FPL Points Predictor — Current Roster Snapshot
====================================================
Reads FPL's current player list (position, price, team) directly from
bootstrap-static's 'elements' — independent of whether any gameweeks
have actually been played yet. This is what actually answers "what are
the new positions and prices" during preseason, when
extract_fpl_clean_dataset.py legitimately returns nothing (it depends
on per-gameweek history, which doesn't exist until the season starts).

Compares against the LAST snapshot (if one exists) to report exactly
what changed: new players, position changes, price changes — the same
kind of comparison build_position_history.py does, but usable RIGHT
NOW rather than waiting for real match data to flow through the full
pipeline.

Run:
    python build_current_roster_snapshot.py
"""

import json
from pathlib import Path

import pandas as pd
import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
REQUEST_TIMEOUT = 15
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_PATH = PROCESSED_DIR / "current_roster_snapshot.csv"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fetch_roster() -> pd.DataFrame:
    r = requests.get(BOOTSTRAP_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    team_lookup = {t["id"]: t["name"] for t in data["teams"]}

    rows = []
    for el in data["elements"]:
        rows.append({
            "player_id": el["id"],
            "player_name": el.get("web_name"),
            "full_name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip(),
            "team": team_lookup.get(el.get("team")),
            "position": POSITION_MAP.get(el.get("element_type")),
            "price": el.get("now_cost"),  # tenths of a million, e.g. 55 = £5.5m
        })
    return pd.DataFrame(rows)


def compare_and_report(current: pd.DataFrame, previous: pd.DataFrame):
    prev_by_id = previous.set_index("player_id")
    curr_by_id = current.set_index("player_id")

    new_players = curr_by_id.index.difference(prev_by_id.index)
    if len(new_players) > 0:
        print(f"\n{len(new_players)} players NEW to the roster (not in the last snapshot):")
        for pid in new_players:
            row = curr_by_id.loc[pid]
            print(f"  {row['player_name']} ({row['team']}, {row['position']}) — £{row['price']/10:.1f}m")

    common_ids = curr_by_id.index.intersection(prev_by_id.index)
    position_changes = curr_by_id.loc[common_ids][
        curr_by_id.loc[common_ids, "position"] != prev_by_id.loc[common_ids, "position"]
    ]
    if len(position_changes) > 0:
        print(f"\n{len(position_changes)} players with a CHANGED position since the last snapshot:")
        for pid in position_changes.index:
            old_pos = prev_by_id.loc[pid, "position"]
            new_pos = curr_by_id.loc[pid, "position"]
            name = curr_by_id.loc[pid, "player_name"]
            print(f"  {name}: {old_pos} -> {new_pos}")

    price_changes = curr_by_id.loc[common_ids][
        curr_by_id.loc[common_ids, "price"] != prev_by_id.loc[common_ids, "price"]
    ]
    if len(price_changes) > 0:
        print(f"\n{len(price_changes)} players with a CHANGED price since the last snapshot "
              f"(showing largest 10 moves):")
        price_deltas = (curr_by_id.loc[price_changes.index, "price"] - prev_by_id.loc[price_changes.index, "price"])
        for pid in price_deltas.abs().sort_values(ascending=False).head(10).index:
            name = curr_by_id.loc[pid, "player_name"]
            old_p, new_p = prev_by_id.loc[pid, "price"], curr_by_id.loc[pid, "price"]
            print(f"  {name}: £{old_p/10:.1f}m -> £{new_p/10:.1f}m")


def main():
    print("Fetching current FPL roster (positions, prices) — works regardless of "
          "whether any gameweeks have been played yet...")
    current = fetch_roster()
    print(f"  {len(current)} players fetched")

    if SNAPSHOT_PATH.exists():
        previous = pd.read_csv(SNAPSHOT_PATH)
        compare_and_report(current, previous)
    else:
        print("\nNo previous snapshot found — this run establishes the baseline. "
              "Nothing to compare against yet.")

    current.to_csv(SNAPSHOT_PATH, index=False)
    print(f"\nSaved -> {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()