"""
FPL Points Predictor — Historical Season Reconciliation (Phase 0)
========================================================================
Reconciles the 24/25 archive season against the CURRENT roster, tagging
every row with its real season and current player_id — the foundation
every other multi-season improvement (cards, bonus, general model
refitting) depends on.

Matches on FULL NAME, not the short display name we normally prioritize
elsewhere — this archive source (vaastav's GitHub mirror of FPL's own
API) only publishes full names (e.g. "Carlos Miguel dos Santos
Pereira"), not FPL's short "web_name" field. Same safety principles as
the season-transition reconciliation built earlier this project: id+name
verification, chain-collision-safe two-phase remapping, and genuine
ambiguity left unresolved rather than guessed at.

Confirmed via direct inspection of the real archive: this season has NO
raw defensive-action counting stats (no tackles/CBI/recoveries columns)
— DC didn't exist as a scoring category yet, so FPL never tracked these
fields for 24/25. This dataset is genuinely useful for cards and bonus,
not defensive contributions.

Run:
    python build_historical_season_data.py
"""

import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ARCHIVE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"

# columns from the archive worth keeping — cards and bonus are the real
# targets; DC-relevant raw stats genuinely don't exist for this season
KEEP_COLUMNS = {
    "name": "full_name",
    "position": "position",
    "team": "team",
    "assists": "assists",
    "bonus": "bonus",
    "bps": "bps",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "goals_scored": "goals_scored",
    "minutes": "minutes",
    "red_cards": "red_cards",
    "yellow_cards": "yellow_cards",
    "saves": "saves",
    "starts": "starts",
    "total_points": "total_points",
    "value": "value",
    "was_home": "was_home_int",
    "GW": "gameweek",
    "opponent_team": "opponent_team",
    "expected_goals": "fpl_xG",
    "expected_assists": "fpl_xA",
    "expected_goals_conceded": "fpl_xGC",
    "ict_index": "ict_index",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "fixture": "fixture_id",
    "opponent_team": "opponent_team",
}


def normalize_name(name: str) -> str:
    """Same normalization used throughout the project — lowercase,
    strip accents, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def load_archive_season(season: str) -> pd.DataFrame:
    """Downloads (or reuses a cached copy of) one season's archive."""
    cache_path = RAW_DIR / f"fpl_archive_{season.replace('-', '_')}.csv"
    if cache_path.exists():
        print(f"Using cached archive: {cache_path}")
        return pd.read_csv(cache_path)

    url = ARCHIVE_URL.format(season=season)
    print(f"Downloading {season} archive from {url} ...")
    df = pd.read_csv(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Cached -> {cache_path} ({len(df)} rows)")
    return df


def reconcile_by_full_name(archive_df: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """
    Matches archive rows to the CURRENT roster's player_id via full
    name — this source has no short display name to match on, unlike
    the season-transition reconciliation elsewhere in this project.

    Position is used as a tiebreaker for genuine full-name collisions
    (rare, but the same real risk found earlier this project — e.g.
    two different real people sharing an identical full name). Left
    unresolved rather than guessed at if still ambiguous after that.
    """
    archive_df = archive_df.copy()
    archive_df["_norm_full"] = archive_df["full_name"].apply(normalize_name)

    roster = roster.copy()
    roster["_norm_full"] = roster["full_name"].apply(normalize_name)

    name_groups = roster.groupby("_norm_full")

    id_map = {}
    ambiguous = []
    matched, unmatched = 0, 0

    for norm_name in archive_df["_norm_full"].unique():
        if norm_name not in name_groups.groups:
            unmatched += 1
            continue

        candidates = name_groups.get_group(norm_name)
        if len(candidates) == 1:
            id_map[norm_name] = candidates.iloc[0]["player_id"]
            matched += 1
            continue

        # full-name collision — try position as a tiebreaker
        archive_positions = archive_df[archive_df["_norm_full"] == norm_name]["position"].unique()
        pos_matches = candidates[candidates["position"].isin(archive_positions)]
        if len(pos_matches) == 1:
            id_map[norm_name] = pos_matches.iloc[0]["player_id"]
            matched += 1
        else:
            ambiguous.append((norm_name, len(candidates)))
            unmatched += 1

    print(f"\nMatched {matched} unique players by full name, {unmatched} unmatched "
          f"(genuinely not on the current roster — left the league since, expected "
          f"for a two-year-old season).")
    if ambiguous:
        print(f"\n*** {len(ambiguous)} full name(s) genuinely ambiguous even with position "
              f"tie-breaking — left unmatched rather than guessed at: ***")
        for name, n in ambiguous:
            print(f"  {name}: {n} possible candidates")

    archive_df["player_id"] = archive_df["_norm_full"].map(id_map)
    archive_df = archive_df.drop(columns=["_norm_full"])
    return archive_df


def main():
    season = "2024-25"
    raw = load_archive_season(season)

    # Handles two different naming conventions: raw downloads from
    # GitHub use FPL's own field names directly (goals_scored,
    # total_points, GW); a LOCAL cached copy of this file was already
    # processed earlier in this project and renamed some fields to
    # match this project's own conventions (goals, actual_points,
    # gameweek). Checking both prevents this script from thinking real
    # data is "missing" just because of a naming mismatch.
    ALTERNATE_SOURCE_NAMES = {
        "goals": "goals_scored",
        "actual_points": "total_points",
    }
    for local_name, raw_name in ALTERNATE_SOURCE_NAMES.items():
        if raw_name not in raw.columns and local_name in raw.columns:
            KEEP_COLUMNS[local_name] = KEEP_COLUMNS.pop(raw_name, KEEP_COLUMNS.get(raw_name))
    if "GW" not in raw.columns and "gameweek" in raw.columns and "GW" in KEEP_COLUMNS:
        KEEP_COLUMNS["gameweek"] = KEEP_COLUMNS.pop("GW")

    available_cols = [c for c in KEEP_COLUMNS if c in raw.columns]
    missing_cols = [c for c in KEEP_COLUMNS if c not in raw.columns]
    if missing_cols:
        print(f"\n(Note: {len(missing_cols)} expected columns not found in this archive: "
              f"{missing_cols} — proceeding with what's available.)")

    df = raw[available_cols].rename(columns=KEEP_COLUMNS)
    df["season"] = season

    roster_path = PROCESSED_DIR / "current_roster_snapshot.csv"
    if not roster_path.exists():
        print(f"\n{roster_path} not found — run build_current_roster_snapshot.py first "
              f"so this season's players can be matched to their current identity.")
        return

    roster = pd.read_csv(roster_path)
    if "full_name" not in roster.columns:
        print(f"\ncurrent_roster_snapshot.csv has no 'full_name' column — re-run "
              f"build_current_roster_snapshot.py with the latest version of the script.")
        return

    df = reconcile_by_full_name(df, roster)

    output_path = PROCESSED_DIR / f"historical_{season.replace('-', '_')}_reconciled.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path} ({len(df)} rows, {df['player_id'].notna().sum()} "
          f"with a matched current player_id)")

    print("\n=== Columns kept ===")
    print(list(df.columns))


if __name__ == "__main__":
    main()