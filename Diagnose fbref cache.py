"""
FPL Points Predictor — FBref Team Stats (Possession, SoT%, Saves%, Cards)
=============================================================================
Extracts the "Team Stats" box from each match report — possession,
shots on target efficiency, save efficiency, and cards — a genuinely
new, real signal we haven't had before. Parser validated directly
against real FBref HTML before this script was written (not a guess).

Unlike the player-level defense/passing/misc tables (which turned out
to require actual tab-clicks we don't currently simulate), this table
is part of the main match report page — already present in the
existing ~/soccerdata/data/FBref/match_*.html cache, so this reads
directly from disk with ZERO new network requests, same as the
original successful cache-reuse approach.

Run:
    python extract_fbref_team_stats.py
"""

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

CACHE_DIR = Path.home() / "soccerdata" / "data" / "FBref"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_team_stats(html_content: str) -> dict | None:
    soup = BeautifulSoup(html_content, "html.parser")
    container = soup.find("div", id="team_stats")
    if container is None:
        return None

    table = container.find("table")
    if table is None or table.find("tbody") is None:
        return None
    rows = table.find("tbody").find_all("tr", recursive=False)
    if not rows:
        return None

    header_row = rows[0]
    spans = header_row.find_all("span", class_="teamandlogo")
    if len(spans) < 2:
        return None
    home_team = spans[0].get_text(strip=True)
    away_team = spans[1].get_text(strip=True)

    stats = {"home_team": home_team, "away_team": away_team}

    i = 1
    while i < len(rows) - 1:
        label_row = rows[i]
        th = label_row.find("th", colspan="2")
        if th is None:
            i += 1
            continue
        label = th.get_text(strip=True)
        value_row = rows[i + 1]
        tds = value_row.find_all("td", recursive=False)
        if len(tds) < 2:
            i += 2
            continue

        if label == "Possession":
            home_strong = tds[0].find("strong")
            away_strong = tds[1].find("strong")
            if home_strong is not None:
                stats["possession_home"] = int(home_strong.get_text(strip=True).replace("%", ""))
            if away_strong is not None:
                stats["possession_away"] = int(away_strong.get_text(strip=True).replace("%", ""))

        elif label in ("Shots on Target", "Saves"):
            prefix = "sot" if label == "Shots on Target" else "saves"
            for side, td in [("home", tds[0]), ("away", tds[1])]:
                text = td.get_text(" ", strip=True)
                made_of = re.search(r"(\d+)\s+of\s+(\d+)", text)
                pct = re.search(r"(\d+)%", text)
                stats[f"{prefix}_made_{side}"] = int(made_of.group(1)) if made_of else None
                stats[f"{prefix}_attempted_{side}"] = int(made_of.group(2)) if made_of else None
                stats[f"{prefix}_pct_{side}"] = int(pct.group(1)) if pct else None

        elif label == "Cards":
            for side, td in [("home", tds[0]), ("away", tds[1])]:
                stats[f"yellow_cards_{side}"] = len(td.find_all("span", class_="yellow_card"))
                stats[f"red_cards_{side}"] = len(td.find_all("span", class_="red_card"))

        i += 2

    return stats


def main():
    if not CACHE_DIR.exists():
        print(f"Cache directory not found at {CACHE_DIR} — "
              f"edit CACHE_DIR at the top of this script if yours is elsewhere.")
        return

    match_files = sorted(CACHE_DIR.glob("match_*.html"))
    print(f"Found {len(match_files)} cached match files at {CACHE_DIR}")
    if not match_files:
        return

    rows = []
    n_failed = 0
    for i, filepath in enumerate(match_files):
        game_id = filepath.stem.replace("match_", "")
        try:
            html_content = filepath.read_text(encoding="utf-8", errors="replace")
            stats = parse_team_stats(html_content)
            if stats is None:
                n_failed += 1
                continue
            stats["game_id"] = game_id
            rows.append(stats)
        except Exception as e:
            print(f"  FAILED on {filepath.name}: {e}")
            n_failed += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(match_files)}] processed")

    print(f"\nDone. {len(rows)} matches parsed successfully, {n_failed} failed/skipped.")

    if not rows:
        print("No data extracted — check whether the 'team_stats' div structure "
              "matches what this parser expects (FBref may have changed its layout).")
        return

    df = pd.DataFrame(rows)

    # reshape to one row per team per match (matches the structure of every
    # other data source in this project, easier to merge downstream)
    home_cols = [c for c in df.columns if c.endswith("_home")] + ["home_team", "away_team", "game_id"]
    away_cols = [c for c in df.columns if c.endswith("_away")] + ["home_team", "away_team", "game_id"]

    home_df = df[home_cols].rename(columns={c: c.replace("_home", "") for c in home_cols if c.endswith("_home")})
    home_df["team"] = home_df["home_team"]
    home_df["opponent"] = home_df["away_team"]
    home_df["was_home"] = True

    away_df = df[away_cols].rename(columns={c: c.replace("_away", "") for c in away_cols if c.endswith("_away")})
    away_df["team"] = away_df["away_team"]
    away_df["opponent"] = away_df["home_team"]
    away_df["was_home"] = False

    combined = pd.concat([home_df, away_df], ignore_index=True)
    combined = combined.drop(columns=["home_team", "away_team"])

    output_path = OUTPUT_DIR / "fbref_team_stats.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path} ({len(combined)} team-match rows)")
    print(f"Columns: {list(combined.columns)}")
    print(f"\nSample row:\n{combined.head(1).to_string()}")


if __name__ == "__main__":
    main()