"""
FPL Points Predictor — Merge
==============================
Combines the three raw sources into a single player-gameweek training
table, using the entity-matching table to link players across sources
and match date to align each FPL gameweek row with the correct
Understat/FBref match record.

Only CONFIDENT matches (needs_review == False) are used to pull in
Understat/FBref stats — flagged/fringe players still keep their FPL
row, just with NaN for the columns we couldn't confidently source,
rather than risking joining the wrong person's stats onto them.

Run:
    python build_merged_dataset.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD + PREP EACH SOURCE
# =========================
def load_fpl() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "fpl_clean_dataset.csv")
    df["match_date"] = pd.to_datetime(df["kickoff_time"], errors="coerce", utc=True).dt.date
    return df


def load_matching() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "entity_matching_table.csv")
    # only trust confident matches for pulling in external stats — a flagged
    # match is often "closest of a bad bunch," not the real player
    confident = df[~df["needs_review"]].copy()
    return confident[["player_id", "understat_id", "understat_name", "fbref_name"]]


def load_understat() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "understat_player_match_stats.csv")
    df["match_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # prefix stat columns so they don't collide with FPL/FBref columns of the
    # same name (both sources have 'minutes', 'team', 'position', etc.)
    keep_cols = ["understat_player_id", "match_date", "minutes", "goals", "assists",
                 "shots", "key_passes", "xG", "xA", "npxG", "npg", "xGChain", "xGBuildup"]
    df = df[keep_cols].rename(columns={c: f"us_{c}" for c in keep_cols if c not in ["understat_player_id", "match_date"]})
    return df


def load_fbref() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "fbref_player_match_summary.csv")
    # 'game' looks like "2025-08-15 Liverpool-Bournemouth" — pull the date off the front
    df["match_date"] = pd.to_datetime(
        df["game"].str.extract(r"^(\d{4}-\d{2}-\d{2})")[0], errors="coerce"
    ).dt.date

    stat_cols = [c for c in df.columns if c.startswith("Performance_")]
    keep_cols = ["player", "team", "match_date", "min"] + stat_cols
    df = df[keep_cols].rename(columns={c: f"fb_{c}" for c in keep_cols if c not in ["player", "match_date"]})
    # fb_team is FBref's own team-at-time-of-match — more historically accurate
    # than FPL's 'team' column (which reflects each player's CURRENT club
    # applied to every row), so worth keeping under a distinct name rather
    # than letting it collide with FPL's 'team' column during the merge.
    return df


# =========================
# MERGE
# =========================
def build_merged_dataset() -> pd.DataFrame:
    log.info("Loading sources...")
    fpl = load_fpl()
    matching = load_matching()
    understat = load_understat()
    fbref = load_fbref()

    log.info("FPL rows: %d | confident matches: %d | Understat rows: %d | FBref rows: %d",
              len(fpl), len(matching), len(understat), len(fbref))

    # attach each FPL row's Understat/FBref identity (NaN if not a confident match)
    df = fpl.merge(matching, on="player_id", how="left")

    # join in Understat stats on (player identity, match date)
    df = df.merge(
        understat,
        left_on=["understat_id", "match_date"],
        right_on=["understat_player_id", "match_date"],
        how="left",
    )

    # join in FBref stats on (player name, match date). Deliberately NOT
    # joining on team too: FPL's 'team' column reflects each player's CURRENT
    # club applied to every historical row, so for anyone who transferred
    # mid-season, pre-transfer rows carry the wrong team and would silently
    # fail a team-based join. Name + exact date is unique enough on its own.
    df = df.merge(
        fbref,
        left_on=["fbref_name", "match_date"],
        right_on=["player", "match_date"],
        how="left",
    )

    # drop the now-redundant join-key columns from the joined sources
    df = df.drop(columns=["understat_player_id", "player"], errors="ignore")

    return df


def apply_position_history(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects historical rows' positions using the frozen snapshot from
    build_position_history.py — prevents a reclassified player's PAST
    season (played under their OLD position) from being silently
    relabeled with their NEW position, which would contaminate
    position-specific model training. Genuinely new gameweeks not in
    the snapshot still correctly get the current/new classification."""
    history_path = PROCESSED_DIR / "position_history.csv"
    if not history_path.exists():
        log.info("No position_history.csv found — skipping position correction "
                 "(run build_position_history.py to enable this safeguard).")
        return df

    history = pd.read_csv(history_path)
    df = df.merge(history, on=["player_id", "gameweek"], how="left", suffixes=("", "_frozen"))

    changed_mask = df["position_frozen"].notna() & (df["position"] != df["position_frozen"])
    if changed_mask.any():
        changed_players = df.loc[changed_mask, ["player_name", "team", "position", "position_frozen"]].drop_duplicates()
        log.warning("Position correction applied to %d rows across %d players — "
                    "their CURRENT classification differs from what was recorded "
                    "for these historical gameweeks:", changed_mask.sum(), len(changed_players))
        for _, row in changed_players.iterrows():
            log.warning("  %s (%s): %s -> %s (using %s for these historical rows)",
                        row["player_name"], row["team"], row["position_frozen"], row["position"], row["position_frozen"])

    df["position"] = df["position_frozen"].fillna(df["position"])
    df = df.drop(columns=["position_frozen"])
    return df


def main():
    fpl_path = DATA_DIR / "fpl_clean_dataset.csv"

    def preseason_skip_message():
        print(f"\n{fpl_path} has no usable data — expected during preseason before "
              f"any real gameweek data exists yet (the extraction step correctly "
              f"refuses to overwrite good data with an empty preseason result, but "
              f"there's genuinely nothing new to merge in until real games have been "
              f"played). Skipping — the existing merged_player_gameweek.csv and "
              f"downstream model_ready_dataset.csv are untouched and still valid "
              f"for predictions.")

    if not fpl_path.exists():
        preseason_skip_message()
        return

    # Catching the actual pandas error directly here, rather than trying
    # to predict every possible way a file can be "empty" from its file
    # properties beforehand (confirmed on real data: a file can have
    # nonzero size — e.g. a stray blank line — while still being
    # functionally unparseable, which a simple size==0 check misses).
    try:
        df = build_merged_dataset()
    except pd.errors.EmptyDataError:
        preseason_skip_message()
        return

    df = apply_position_history(df)

    if len(df) == 0:
        preseason_skip_message()
        return

    output_path = PROCESSED_DIR / "merged_player_gameweek.csv"
    df.to_csv(output_path, index=False)

    n_total = len(df)
    n_with_understat = df["us_xG"].notna().sum()
    n_with_fbref = df["fb_Performance_Gls"].notna().sum()

    print(f"\nSaved merged dataset -> {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nRows with Understat stats matched: {n_with_understat} / {n_total} ({n_with_understat/n_total:.1%})")
    print(f"Rows with FBref stats matched: {n_with_fbref} / {n_total} ({n_with_fbref/n_total:.1%})")

    # A 0-minute row (unused sub, not in matchday squad) genuinely can't have
    # Understat/FBref stats, no matter how good the matching is — so the raw
    # percentages above understate true join health. This narrows to rows
    # where the player actually played AND has a confident source match,
    # which is the real ceiling worth judging the join against.
    played = df[(df["minutes"] > 0) & (df["understat_id"].notna())]
    if len(played) > 0:
        played_us = played["us_xG"].notna().sum()
        print(f"\nOf rows where the player actually played AND has a confident Understat match: "
              f"{played_us} / {len(played)} ({played_us/len(played):.1%}) joined successfully")

    played_fb = df[(df["minutes"] > 0) & (df["fbref_name"].notna())]
    if len(played_fb) > 0:
        matched_fb = played_fb["fb_Performance_Gls"].notna().sum()
        print(f"Of rows where the player actually played AND has a confident FBref match: "
              f"{matched_fb} / {len(played_fb)} ({matched_fb/len(played_fb):.1%}) joined successfully")

    print("\n=== Sample row ===")
    print(df.head(3).to_string())

    print("\n=== Columns ===")
    print(list(df.columns))


if __name__ == "__main__":
    main()
