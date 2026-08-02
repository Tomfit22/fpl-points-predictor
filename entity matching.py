"""
FPL Points Predictor — Entity Matching
=========================================
Links players and teams across the three data sources so they can be
merged into a single player-gameweek table:

    FPL (player_id)  <-->  Understat (understat_player_id)  <-->  FBref (player name)

Team names differ across sources (e.g. FPL "Man Utd" vs Understat
"Manchester United" vs FBref "Manchester Utd"), and player names differ
too (short names, accents, nicknames). This script normalizes both and
uses fuzzy string matching (rapidfuzz) to link them, flagging anything
below a confidence threshold for manual review rather than guessing.

Install:
    pip install rapidfuzz

Run:
    python build_entity_matching.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

# WRatio handles partial/subset matches well — e.g. "David Raya Martín" (FPL's
# full_name) vs "David Raya" (Understat/FBref's shorter name) — better than
# token_sort_ratio, which penalizes the extra "Martín" token too heavily.
NAME_SCORER = fuzz.WRatio

# =========================
# CONFIG
# =========================
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NAME_MATCH_THRESHOLD = 85   # below this, a match is flagged for manual review
TEAM_MATCH_THRESHOLD = 80

# Known team-name aliases across sources. Add to this as you discover
# more mismatches when you inspect the "needs_review" output.
# Known team-name aliases across sources. Keys must already be in normalized
# form (lowercase, no punctuation) since normalize_team() strips punctuation
# BEFORE checking this dict — a key like "nott'm forest" would never match
# because by lookup time the apostrophe is already gone.
TEAM_ALIASES = {
    "man utd": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "nottm forest": "nottingham forest",       # was "nott'm forest" — apostrophe never survives normalization
    "nottham forest": "nottingham forest",     # was "nott'ham forest" — same issue
    "wolves": "wolverhampton wanderers",
    "brighton hove albion": "brighton",        # FBref: "Brighton & Hove Albion"
    "west ham": "west ham united",
    "newcastle": "newcastle united",
    "leeds": "leeds united",
}


# =========================
# NORMALIZATION HELPERS
# =========================
# Common nickname -> full first name mappings. Sources inconsistently use
# "Matty Cash" vs "Matthew Cash", "Alex" vs "Alexander", etc. — normalize the
# first token so these compare equal instead of relying on fuzzy score alone.
NICKNAME_ALIASES = {
    "matty": "matthew", "alex": "alexander", "sam": "samuel", "will": "william",
    "ben": "benjamin", "joe": "joseph", "mo": "mohamed", "danny": "daniel",
    "jimmy": "james", "robbie": "robert", "tommy": "thomas", "charlie": "charles",
    "nick": "nicholas", "mike": "michael", "chris": "christopher", "tony": "anthony",
    "andy": "andrew", "jack": "john", "harry": "harold", "eddie": "edward",
}


# Manual overrides for known cases where a player's name differs so much
# between sources (stage names, middle names used instead of first names)
# that generic fuzzy matching can't reliably clear the confidence threshold.
# Keyed by the FPL full_name's normalized form -> the exact search term to
# use against that specific source instead.
UNDERSTAT_NAME_OVERRIDES = {
    "carlos henrique casimiro": "casemiro",
    "norberto bercique gomes betuncal": "beto",
    "lesley chimuanya ugochukwu": "chimuanya ugochukwu",
    "lesley ugochukwu": "chimuanya ugochukwu",
    "rayan cherki": "mathis cherki",  # Understat has his first name wrong in their data
    "fer lopez gonzalez": "fernando lopez",  # Understat lists him under a longer first name
    "benjamin gannondoak": "benjamin doak",  # Understat drops the "Gannon" from his surname (both sides go through ben->benjamin normalization)
}
FBREF_NAME_OVERRIDES = {
    "norberto bercique gomes betuncal": "beto",
    "carlos henrique casimiro": "casemiro",
    "jamie bynoegittens": "jamie gittens",  # FBref drops the "Bynoe" from his surname
}


def strip_accents(text: str) -> str:
    """'João Pedro' -> 'joao pedro' — needed since sources spell accented names differently."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = strip_accents(str(name)).lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)  # drop punctuation (periods, apostrophes)
    name = re.sub(r"\s+", " ", name).strip()

    tokens = name.split(" ")
    if tokens and tokens[0] in NICKNAME_ALIASES:
        tokens[0] = NICKNAME_ALIASES[tokens[0]]
    return " ".join(tokens)


def normalize_team(team: str) -> str:
    norm = normalize_name(team)
    return TEAM_ALIASES.get(norm, norm)


# =========================
# LOAD SOURCES
# =========================
def load_fpl_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "fpl_clean_dataset.csv")
    players = df[["player_id", "player_name", "full_name", "team", "position"]].drop_duplicates()
    players["norm_name"] = players["full_name"].apply(normalize_name)
    players["norm_team"] = players["team"].apply(normalize_team)
    return players.reset_index(drop=True)


def load_understat_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "understat_player_match_stats.csv")
    players = df[["understat_player_id", "player_name", "team"]].drop_duplicates()

    # Understat joins multiple clubs into one string for players who transferred
    # mid-season, e.g. "Arsenal,Crystal Palace". Rather than guess which position
    # in the list is their current club, explode into one row per club so they
    # match correctly regardless of which club FPL currently shows them at.
    players["team"] = players["team"].astype(str).str.split(",")
    players = players.explode("team")
    players["team"] = players["team"].str.strip()

    players["norm_name"] = players["player_name"].apply(normalize_name)
    players["norm_team"] = players["team"].apply(normalize_team)
    return players.reset_index(drop=True)


def load_fbref_players() -> pd.DataFrame:
    # Using the match-level summary file since extract_fbref_v2.py only produces
    # that one — it has 'player' and 'team' on every row, so distinct pairs work
    # just as well as a dedicated season-stats file would.
    df = pd.read_csv(DATA_DIR / "fbref_player_match_summary.csv")
    players = df[["player", "team"]].drop_duplicates()
    players["norm_name"] = players["player"].apply(normalize_name)
    players["norm_team"] = players["team"].apply(normalize_team)
    return players.reset_index(drop=True)


# =========================
# MATCHING
# =========================
def match_players(
    fpl_players: pd.DataFrame,
    other_players: pd.DataFrame,
    other_id_col: str,
    other_name_col: str,
    match_label: str,
    override_map: dict | None = None,
) -> pd.DataFrame:
    """
    For each FPL player, find the best name match among other_players
    that share the same normalized team. Falls back to fuzzy team
    matching if no exact normalized-team match exists.
    """
    override_map = override_map or {}
    results = []

    for _, fpl_row in fpl_players.iterrows():
        candidates = other_players[other_players["norm_team"] == fpl_row["norm_team"]]

        # fallback: fuzzy-match the team itself if no exact team match found
        if candidates.empty:
            team_choices = other_players["norm_team"].unique().tolist()
            best_team = process.extractOne(fpl_row["norm_team"], team_choices, scorer=fuzz.token_sort_ratio)
            if best_team and best_team[1] >= TEAM_MATCH_THRESHOLD:
                candidates = other_players[other_players["norm_team"] == best_team[0]]

        if candidates.empty:
            results.append({
                "player_id": fpl_row["player_id"],
                "fpl_full_name": fpl_row["full_name"],
                "fpl_team": fpl_row["team"],
                f"{match_label}_id": None,
                f"{match_label}_name": None,
                f"{match_label}_score": 0,
                "needs_review": True,
                "review_reason": "no team match found",
            })
            continue

        # manual override: search for the known correct name instead of the
        # FPL name, and treat it as a fully-confident (100) match if found
        search_name = override_map.get(fpl_row["norm_name"], fpl_row["norm_name"])
        is_override = fpl_row["norm_name"] in override_map

        name_choices = candidates["norm_name"].tolist()
        best_match = process.extractOne(search_name, name_choices, scorer=NAME_SCORER)

        if best_match is None:
            results.append({
                "player_id": fpl_row["player_id"],
                "fpl_full_name": fpl_row["full_name"],
                "fpl_team": fpl_row["team"],
                f"{match_label}_id": None,
                f"{match_label}_name": None,
                f"{match_label}_score": 0,
                "needs_review": True,
                "review_reason": "no name candidates in matched team",
            })
            continue

        matched_row = candidates[candidates["norm_name"] == best_match[0]].iloc[0]
        score = 100 if is_override else best_match[1]

        results.append({
            "player_id": fpl_row["player_id"],
            "fpl_full_name": fpl_row["full_name"],
            "fpl_team": fpl_row["team"],
            f"{match_label}_id": matched_row.get(other_id_col, matched_row.get(other_name_col)),
            f"{match_label}_name": matched_row[other_name_col],
            f"{match_label}_score": score,
            "needs_review": score < NAME_MATCH_THRESHOLD,
            "review_reason": "" if score >= NAME_MATCH_THRESHOLD else "low name-match confidence",
        })

    return pd.DataFrame(results)


# =========================
# MAIN
# =========================
def main():
    print("Loading sources...")
    fpl_players = load_fpl_players()
    understat_players = load_understat_players()
    fbref_players = load_fbref_players()

    print(f"FPL players: {len(fpl_players)}")
    print(f"Understat players: {len(understat_players)}")
    print(f"FBref players: {len(fbref_players)}")

    print("\nMatching FPL <-> Understat...")
    understat_matches = match_players(
        fpl_players, understat_players,
        other_id_col="understat_player_id", other_name_col="player_name",
        match_label="understat", override_map=UNDERSTAT_NAME_OVERRIDES,
    )

    print("Matching FPL <-> FBref...")
    fbref_matches = match_players(
        fpl_players, fbref_players,
        other_id_col="player", other_name_col="player",
        match_label="fbref", override_map=FBREF_NAME_OVERRIDES,
    )

    # combine into one lookup table keyed on FPL player_id
    merged = understat_matches.merge(
        fbref_matches.drop(columns=["fpl_full_name", "fpl_team"]),
        on="player_id",
        suffixes=("_understat", "_fbref"),
    )

    # a row needs review if EITHER match was flagged
    merged["needs_review"] = merged["needs_review_understat"] | merged["needs_review_fbref"]

    output_path = OUTPUT_DIR / "entity_matching_table.csv"
    merged.to_csv(output_path, index=False)

    n_review = merged["needs_review"].sum()
    print(f"\nSaved entity matching table -> {output_path}")
    print(f"Total FPL players processed: {len(merged)}")
    print(f"Flagged for manual review: {n_review} ({n_review / len(merged):.1%})")

    if n_review > 0:
        print("\nSample of rows needing review:")
        review_cols = ["fpl_full_name", "fpl_team", "understat_name", "understat_score",
                        "fbref_name", "fbref_score", "review_reason_understat", "review_reason_fbref"]
        available_cols = [c for c in review_cols if c in merged.columns]
        print(merged[merged["needs_review"]][available_cols].head(20).to_string())

    print("\nNext: open the CSV, manually fix any 'needs_review' rows "
          "(usually a nickname, accent, or team-alias gap), then re-run "
          "with corrections added to TEAM_ALIASES or a manual override dict.")


if __name__ == "__main__":
    main()