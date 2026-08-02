"""
FPL Points Predictor — FBref Extraction (v2, resilient)
==========================================================
FBref runs Cloudflare bot protection. Over a long scraping session
(380 matches) it can eventually flag the session and kill the
underlying browser entirely — which happened on the first run at
game 81. This version fixes that with:

  1. Per-game try/except — one failed game doesn't kill the whole run
  2. Incremental checkpointing — progress is saved to CSV every N games,
     not just once at the very end
  3. Resume support — re-running this script skips games already saved
  4. Driver recovery — if the browser session dies, a fresh one is
     started rather than continuing to hammer a dead session
  5. A deliberate delay between games — gentler pacing to reduce the
     chance of tripping bot detection again

Run:
    python extract_fbref_v2.py

If it crashes anyway, just run it again — it'll pick up where it left off.
"""

import logging
import time
import traceback
from pathlib import Path

import pandas as pd
import soccerdata as sd

# =========================
# CONFIG
# =========================
LEAGUE = "ENG-Premier League"
SEASON = "2025-2026"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "fbref_player_match_summary.csv"

SLEEP_BETWEEN_GAMES = 3     # seconds — gentler pacing than back-to-back requests
SAVE_EVERY_N_GAMES = 10     # checkpoint frequency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(level) for level in col if level]) if isinstance(col, tuple) else col
            for col in df.columns
        ]
    return df


def load_existing_progress():
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        done_ids = set(df["game_id"].unique())
        log.info("Resuming: %d games already saved.", len(done_ids))
        return df, done_ids
    return pd.DataFrame(), set()


def save_progress(df: pd.DataFrame):
    df.to_csv(OUTPUT_PATH, index=False)


def new_fbref_reader():
    return sd.FBref(leagues=LEAGUE, seasons=SEASON)


def main():
    fbref = new_fbref_reader()

    log.info("Fetching schedule...")
    schedule = fbref.read_schedule().reset_index()
    all_game_ids = schedule["game_id"].tolist()

    existing_df, done_ids = load_existing_progress()
    remaining = [g for g in all_game_ids if g not in done_ids]
    log.info("Total games: %d | already done: %d | remaining: %d",
              len(all_game_ids), len(done_ids), len(remaining))

    saved_df = existing_df
    batch = []
    n_failed = 0

    for i, game_id in enumerate(remaining):
        try:
            df = fbref.read_player_match_stats(stat_type="summary", match_id=[game_id])
            df = df.reset_index()
            df = flatten_columns(df)
            batch.append(df)
            log.info("[%d/%d] OK: %s", i + 1, len(remaining), game_id)
        except Exception as e:
            n_failed += 1
            log.error("[%d/%d] FAILED: %s -> %s", i + 1, len(remaining), game_id, e)
            traceback.print_exc()
            # the browser session may be dead — start a fresh one before continuing
            try:
                fbref = new_fbref_reader()
                log.info("Recreated FBref browser session after failure.")
            except Exception as recreate_error:
                log.error("Could not recreate FBref session: %s", recreate_error)

        if (i + 1) % SAVE_EVERY_N_GAMES == 0 and batch:
            saved_df = pd.concat([saved_df] + batch, ignore_index=True) if not saved_df.empty else pd.concat(batch, ignore_index=True)
            save_progress(saved_df)
            batch = []
            log.info("  -> checkpoint saved (%d rows so far)", len(saved_df))

        time.sleep(SLEEP_BETWEEN_GAMES)

    if batch:
        saved_df = pd.concat([saved_df] + batch, ignore_index=True) if not saved_df.empty else pd.concat(batch, ignore_index=True)
        save_progress(saved_df)

    log.info("Done. %d games failed and were skipped (safe to re-run this script to retry them).", n_failed)
    log.info("Final saved shape: %s", saved_df.shape)
    print("\n=== Columns ===")
    print(list(saved_df.columns))


if __name__ == "__main__":
    main()