"""
FPL Points Predictor — FBref Advanced Stats (Blocks, Clearances, Aerial Duels, Passing)
=============================================================================================
Extracts stat tables soccerdata doesn't officially expose (defense,
passing, passing_types, possession, misc) — the ones with Blocks,
Clearances, Aerial duels won, progressive passes, passes into the box,
etc. that we identified as missing much earlier in this project.

HOW THIS WORKS: soccerdata's own read_player_match_stats() only allows
stat_type='summary' or 'keepers' via a hardcoded whitelist check — but
the actual table-lookup logic underneath that check is completely
generic. It just builds an HTML element ID like 'stats_{team_id}_defense'
and searches the page for it. This script calls that same underlying
logic directly, just without the artificial whitelist restriction.

IMPORTANT — this reuses FBref's own caching mechanism. If you still have
the raw match HTML pages cached locally from the original full scrape
(~/soccerdata/data/FBref/match_*.html by default), this requires ZERO
new network requests — just re-parsing files you already downloaded,
so it should run in well under a minute. If that cache has been
cleared, it will fall back to re-scraping (same ~35-40 min process as
the original FBref run).

I could not test this against the live fbref.com site from this
sandbox (no network access to it here) — the table ID pattern is
based on FBref's documented/observed structure, so verify the printed
columns carefully on your first real run, same as with every other
FPL-site extraction in this project.

Run:
    python extract_fbref_advanced_stats.py
"""

from pathlib import Path

import pandas as pd
import soccerdata as sd
from lxml import html
from soccerdata.fbref import _parse_table

LEAGUE = "ENG-Premier League"
SEASON = "2025-2026"
STAT_TYPES = ["defense", "passing", "passing_types", "possession", "misc"]
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Same fix as the original FBref extraction — these tables come back
    with multi-level columns (e.g. ('Tackles', 'Tkl')) that need flattening
    to round-trip cleanly through a CSV. Also strips pandas' 'Unnamed: N'
    placeholder text that appears when a table's group header is blank
    (common for Player/Nation and simple ungrouped stats like Int/Clr) —
    without this, columns come out as 'Unnamed: 0_Player' instead of
    just 'Player'."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(level) for level in col if level and "Unnamed" not in str(level)])
            if isinstance(col, tuple) else col
            for col in df.columns
        ]
    return df


def extract_advanced_stats(fbref: sd.FBref, stat_types: list) -> dict:
    """Returns {stat_type: DataFrame}, one entry per requested table type."""
    schedule = fbref.read_schedule().reset_index()
    schedule = schedule[~schedule.game_id.isna() & ~schedule.match_report.isnull()]

    results = {stat_type: [] for stat_type in stat_types}
    urlmask = "https://fbref.com/en/matches/{}"
    filemask = "match_{}.html"
    n_from_cache = 0
    n_failed = 0

    for i, game in schedule.iterrows():
        game_id = game["game_id"]
        url = urlmask.format(game_id)
        filepath = fbref.data_dir / filemask.format(game_id)
        was_cached = filepath.exists()

        print(f"[{i + 1}/{len(schedule)}] {game_id}" + (" (cached)" if was_cached else " (fetching...)"))
        try:
            reader = fbref.get(url, filepath)
        except Exception as e:
            print(f"  FAILED to get page for {game_id}: {e}")
            n_failed += 1
            continue
        if was_cached:
            n_from_cache += 1

        tree = html.parse(reader)
        try:
            home_team, away_team = fbref._parse_teams(tree)
        except Exception as e:
            print(f"  FAILED to parse teams for {game_id}: {e}")
            n_failed += 1
            continue

        for stat_type in stat_types:
            id_format = "stats_{}_" + stat_type
            for team, side in [(home_team, "home"), (away_team, "away")]:
                html_table = tree.find("//table[@id='" + id_format.format(team["id"]) + "']")
                if html_table is None:
                    continue
                try:
                    df_table = _parse_table(html_table)
                except Exception:
                    continue
                df_table = flatten_columns(df_table)
                df_table["team"] = team["name"]
                df_table["side"] = side
                df_table["game"] = game["game"]
                df_table["game_id"] = game_id
                results[stat_type].append(df_table)

    print(f"\nDone. {n_from_cache}/{len(schedule)} games loaded from cache (no network needed), "
          f"{n_failed} failed.")
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in results.items()}


def main():
    fbref = sd.FBref(leagues=LEAGUE, seasons=SEASON)

    print("Extracting advanced stat tables not exposed by soccerdata's official API...")
    tables = extract_advanced_stats(fbref, STAT_TYPES)

    for stat_type, df in tables.items():
        if df.empty:
            print(f"\n{stat_type}: no data extracted — the table ID pattern may not match "
                  f"FBref's current page structure, worth checking manually if this happens.")
            continue
        output_path = OUTPUT_DIR / f"fbref_{stat_type}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n{stat_type}: saved {len(df)} rows -> {output_path}")
        print(f"  columns: {list(df.columns)}")


if __name__ == "__main__":
    main()