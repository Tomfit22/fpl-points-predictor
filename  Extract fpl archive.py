"""
FPL Points Predictor — Historical Season Extraction (vaastav GitHub archive)
=================================================================================
Pulls full gameweek-level FPL data for past seasons from the vaastav/
Fantasy-Premier-League GitHub repository — a well-known, openly-licensed
public archive of FPL's own data (not a third-party redistributing
someone else's licensed commercial data, unlike the FBref/Opta
situation — this is FPL's own numbers, archived by an open-source
maintainer the same way we archive our own).

Verified LIVE from this sandbox (raw.githubusercontent.com is directly
reachable): 2024-25 season data confirmed present and complete.

HONEST LIMITATION, confirmed directly from the real data: defensive
contribution didn't exist as an FPL stat in past seasons — no
'defensive_contribution', 'clearances_blocks_interceptions',
'recoveries', or 'tackles' columns exist for any season before 2025-26.
This script will clearly report which of our current columns are
missing for whatever season you pull.

Run:
    python extract_fpl_archive.py --season 2024-25
"""

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_TIMEOUT = 30

# columns our CURRENT (2025-26+) pipeline has that we know won't exist in
# older archive seasons — checked against explicitly so this is never a
# silent gap
DC_ERA_COLUMNS = ["defensive_contribution", "clearances_blocks_interceptions", "recoveries", "tackles"]


def fetch_csv(season: str, filename: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{season}/{filename}"
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=str, default="2024-25", help="e.g. 2024-25")
    args = parser.parse_args()
    season = args.season

    print(f"Fetching {season} gameweek data from vaastav/Fantasy-Premier-League...")
    gws = fetch_csv(season, "gws/merged_gw.csv")
    print(f"  {len(gws)} rows fetched")

    print(f"Fetching {season} teams data for opponent name mapping...")
    teams = fetch_csv(season, "teams.csv")
    team_lookup = dict(zip(teams["id"], teams["name"]))
    gws["opponent_team_name"] = gws["opponent_team"].map(team_lookup)

    # rename to align with this project's existing column conventions
    # where there's a clean, unambiguous match
    rename_map = {
        "goals_scored": "goals",
        "GW": "gameweek",
        "total_points": "actual_points",
        "opponent_team": "opponent_team_id",
        "opponent_team_name": "opponent_team",
        "element": "fpl_element_id",  # NOTE: season-specific ID, does NOT
                                        # necessarily match current-season
                                        # player_id — do not assume it does
    }
    gws = gws.rename(columns=rename_map)

    output_path = OUTPUT_DIR / f"fpl_archive_{season.replace('-', '_')}.csv"
    gws.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path} ({len(gws)} rows, {len(gws.columns)} columns)")

    missing_dc_cols = [c for c in DC_ERA_COLUMNS if c not in gws.columns]
    print(f"\n{'*' * 70}")
    if missing_dc_cols:
        print(f"CONFIRMED: this season predates defensive contributions. Missing: {missing_dc_cols}")
        print("This is expected, not an error — DC didn't exist as an FPL stat before 2025-26.")
        print("Do not use this season's data for defensive contribution / DC-related bonus modeling.")
    else:
        print("Defensive contribution columns ARE present in this season's data (unexpected — verify).")
    print(f"{'*' * 70}")

    print(f"\nColumns: {list(gws.columns)}")
    print(f"\nSample row:\n{gws.head(1).to_string()}")


if __name__ == "__main__":
    main()