"""
FPL Points Predictor — Understat Match Stats (PPDA, Deep Completions, xPTS)
================================================================================
Extracts team-level match stats from Understat — PPDA (passes allowed
per defensive action, a pressing-intensity metric nobody else in this
project gives us), deep completions, and xPTS (expected points from
match xG), alongside xG/goals already captured elsewhere.

Uses understatapi's dedicated Match client (.get_match_info()) — the
same reliable, plain-requests-based package already validated for
player data earlier in this project, avoiding the TLS-fingerprinting
issues soccerdata's Understat reader had.

Reuses the match IDs already captured in
data/raw/understat_player_match_stats.csv (the 'understat_match_id'
column) — no need to rediscover them.

I could not verify the exact return structure of get_match_info() live
from this sandbox (no network access to understat.com here) — check
the printed structure from your first real match carefully before
trusting the saved CSV's columns.

Run:
    python extract_understat_match_stats.py
"""

import json
import time
from pathlib import Path

import pandas as pd
from understatapi import UnderstatClient

RAW_DIR = Path("data/raw")
SLEEP_TIME = 0.3


def get_match_ids() -> list:
    df = pd.read_csv(RAW_DIR / "understat_player_match_stats.csv")
    return sorted(df["understat_match_id"].dropna().unique().tolist())


def main():
    match_ids = get_match_ids()
    print(f"Found {len(match_ids)} unique Understat match IDs to fetch")

    client = UnderstatClient()
    rows = []
    n_failed = 0

    for i, match_id in enumerate(match_ids):
        try:
            info = client.match(match=str(int(match_id))).get_match_info()
            if i == 0:
                print("\n=== Raw structure of the FIRST match's response (verify this matches "
                      "what you saw in the screenshot: PPDA, deep completions, xPTS, etc.) ===")
                print(json.dumps(info, indent=2)[:2000])
                print("...\n")
            rows.append(info)
        except Exception as e:
            print(f"  [{i+1}/{len(match_ids)}] FAILED for match {match_id}: {e}")
            n_failed += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(match_ids)}] processed")
        time.sleep(SLEEP_TIME)

    print(f"\nDone. {len(rows)} matches fetched, {n_failed} failed.")

    if not rows:
        print("No data collected — nothing to save.")
        return

    # get_match_info's return shape is unverified from this sandbox — try a
    # direct DataFrame conversion, and if the structure is nested (e.g. a
    # dict with separate home/away sub-dicts rather than a flat one), this
    # will need adjusting once we see the real printed structure above.
    try:
        df = pd.json_normalize(rows)
    except Exception as e:
        print(f"Could not flatten the response into a table directly ({e}) — "
              f"saving raw JSON instead so nothing is lost.")
        output_path = RAW_DIR / "understat_match_stats_raw.json"
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved raw responses -> {output_path}")
        return

    output_path = RAW_DIR / "understat_match_stats.csv"

    # theory: Understat's per-player match history may include European
    # competition fixtures (Champions League etc.) alongside domestic
    # league matches, inflating the count well past 380 — check directly
    if "league" in df.columns:
        league_counts = df["league"].value_counts()
        print(f"\nLeague breakdown of all fetched matches:\n{league_counts.to_string()}")
        epl_only = df[df["league"] == "EPL"]
        print(f"\nRows where league == 'EPL': {len(epl_only)} "
              f"(this should land close to 380 if the theory is right)")
        if len(epl_only) < len(df):
            print(f"Filtering to EPL-only fixtures — {len(df) - len(epl_only)} non-EPL rows excluded.")
            df = epl_only

    # xPTS isn't a direct field, but IS the standard formula derived from
    # the win/draw/loss probabilities Understat does give us directly
    if "h_w" in df.columns and "h_d" in df.columns:
        df["h_xpts"] = 3 * df["h_w"].astype(float) + df["h_d"].astype(float)
    if "a_w" in df.columns and "a_d" in df.columns:
        df["a_xpts"] = 3 * df["a_w"].astype(float) + df["a_d"].astype(float)
    elif "h_w" in df.columns:
        # Understat sometimes only gives one side's w/d/l since a_w = h_l,
        # a_l = h_w, a_d = h_d for a two-outcome-swap — derive the away
        # side from the home side's numbers if a_w/a_l aren't present
        df["a_xpts"] = 3 * df["h_l"].astype(float) + df["h_d"].astype(float)

    # diagnostic: 690 unique match IDs for a 380-fixture season is more
    # than expected — check for real duplicates before trusting this file
    if "id" in df.columns:
        n_unique_ids = df["id"].nunique()
        n_unique_fixtures = df[["team_h", "team_a", "date"]].drop_duplicates().shape[0] if "team_h" in df.columns else None
        print(f"\nDiagnostic: {len(df)} rows, {n_unique_ids} unique 'id' values, "
              f"{n_unique_fixtures} unique (team_h, team_a, date) combinations.")
        if n_unique_fixtures is not None and n_unique_ids > n_unique_fixtures:
            print(f"*** {n_unique_ids - n_unique_fixtures} likely duplicate rows — "
                  f"same real fixture appearing under more than one match ID. "
                  f"Kept as-is for now so you can inspect data/raw/understat_match_stats.csv "
                  f"directly; consider de-duplicating on (team_h, team_a, date) before merging "
                  f"into the main pipeline. ***")

    df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path} ({len(df)} rows)")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()