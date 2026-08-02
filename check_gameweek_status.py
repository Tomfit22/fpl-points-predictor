"""
FPL Points Predictor — Gameweek Completion Checker
========================================================
Instead of guessing a fixed day/time to run the pipeline, this polls
FPL's own bootstrap-static endpoint and checks whether a NEW gameweek
has genuinely finished since the last time the full pipeline ran.

Uses TWO signals from FPL's own data, preferring the stricter one:
  - 'finished': the last fixture in the gameweek has ended
  - 'data_checked': FPL has finalized ALL stats for that gameweek,
    including bonus points — which are NOT always final immediately
    after the last match ends (BPS-based bonus can take a few hours
    to be confirmed). Running the pipeline before this is finalized
    would train on data that's about to change.

I could not verify these exact field names live from this sandbox (no
network access to fantasy.premierleague.com here) — same situation as
every other new FPL API field in this project. The script prints the
raw event structure on first run so you can confirm the field names
are what's expected before trusting it unattended.

Designed to run FREQUENTLY and cheaply via cron (e.g. every 2 hours) —
it's a single small API call, not the full pipeline. Only when it
detects a genuinely newly-completed gameweek does it invoke
orchestrate_pipeline.sh.

Run manually:
    python check_gameweek_status.py

Schedule via cron (checks every 2 hours, cheap, only runs the full
pipeline when something's actually new):
    crontab -e
    0 */2 * * * cd /path/to/FPL_Project && ./venv/bin/python check_gameweek_status.py >> logs/gw_check.log 2>&1
"""

import json
import subprocess
import sys
from pathlib import Path

import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
PROJECT_DIR = Path(__file__).parent
STATE_PATH = PROJECT_DIR / "data" / "processed" / "last_processed_gameweek.json"
REQUEST_TIMEOUT = 15


def get_current_events() -> list:
    r = requests.get(BOOTSTRAP_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["events"]


def get_last_processed_gameweek() -> int:
    if not STATE_PATH.exists():
        return 0
    return json.loads(STATE_PATH.read_text()).get("gameweek", 0)


def save_last_processed_gameweek(gw: int):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"gameweek": gw}))


def find_newly_completed_gameweek(events: list, last_processed: int) -> int | None:
    """Returns the gameweek number if a NEW one has finished (both
    'finished' AND 'data_checked' true, so bonus points are final) that
    we haven't already processed — otherwise None."""
    for event in events:
        gw = event.get("id")
        finished = event.get("finished", False)
        data_checked = event.get("data_checked", False)
        if gw is not None and gw > last_processed and finished and data_checked:
            return gw
    return None


def main():
    print("Checking FPL gameweek status...")
    try:
        events = get_current_events()
    except Exception as e:
        print(f"Could not reach FPL API: {e} — will try again next scheduled check.")
        sys.exit(0)  # don't treat a transient network issue as a failure worth escalating

    last_processed = get_last_processed_gameweek()
    print(f"Last gameweek the pipeline processed: {last_processed}")

    # first run only — print the raw structure so field names can be
    # verified against what this script assumes
    if last_processed == 0:
        current = next((e for e in events if e.get("is_current") or e.get("finished")), events[0])
        print("\n=== First run — raw structure of one gameweek event, verify field names ===")
        print(json.dumps(current, indent=2)[:1000])
        print("...\n")

    newly_completed = find_newly_completed_gameweek(events, last_processed)

    if newly_completed is None:
        print("No newly-completed gameweek since last run — nothing to do.")
        sys.exit(0)

    print(f"Gameweek {newly_completed} has finished AND its data is finalized "
          f"(data_checked=true) — triggering the full pipeline.")

    result = subprocess.run(["./orchestrate_pipeline.sh"], cwd=PROJECT_DIR)

    if result.returncode == 0:
        save_last_processed_gameweek(newly_completed)
        print(f"Pipeline run triggered for gameweek {newly_completed}. State updated.")
    else:
        print(f"orchestrate_pipeline.sh exited with code {result.returncode} — "
              f"NOT updating last-processed gameweek, so this will be retried next check.")


if __name__ == "__main__":
    main()