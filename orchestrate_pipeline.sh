#!/usr/bin/env bash
#
# FPL Points Predictor — Pipeline Orchestration
# ===================================================
# Runs the full pipeline in order and logs everything, so this can be
# scheduled (cron/launchd) to run automatically after each gameweek.
#
# Order matters here — each step depends on the previous one's output:
#   1. extract_fpl_clean_dataset.py     (FPL API — ground truth)
#   2. extract_understat_v2.py          (Understat player data)
#   3. extract_fbref_v2.py              (FBref player data — slow if new
#                                         games aren't cached yet)
#   4. extract_fbref_team_stats.py      (possession/SoT%/cards, from cache)
#   5. extract_understat_match_stats.py (PPDA/deep/xPTS)
#   6. build_merged_dataset.py          (join all sources)
#   7. merge_advanced_team_stats.py     (join possession/PPDA data)
#   8. build_features.py                (rolling own_/opp_ features)
#   9. build_prediction_pipeline.py     (VALIDATE — this is what the
#                                         watchdog checks)
#  10. build_current_roster_snapshot.py (refresh CURRENT squads — who's
#                                         actually still on each team —
#                                         directly from FPL's live API)
#  11. extract_fixtures.py              (refresh upcoming fixture list)
#  12. build_live_predictions.py        (real predictions)
#  13. build_dashboard.py               (regenerate dashboard.html)
#
# Every run's output is logged to logs/run_<timestamp>.log — the
# watchdog script (check_pipeline_health.py) reads the latest log to
# decide whether anything needs the AI agent's attention.
#
# Usage:
#   ./orchestrate_pipeline.sh
#
# To schedule weekly (every Monday 6am, after weekend gameweeks
# complete) via cron:
#   crontab -e
#   0 6 * * 1 cd /path/to/FPL_Project && ./orchestrate_pipeline.sh >> logs/cron.log 2>&1

set -uo pipefail  # NOT -e: we want to keep going and log failures, not
                   # abort the whole run on the first script that errors

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

PYTHON="${PYTHON:-python3}"  # override with PYTHON=./venv/bin/python if needed
STEPS_FAILED=0

run_step() {
    local script_name="$1"
    echo "" | tee -a "$LOG_FILE"
    echo "=== [$(date +%H:%M:%S)] Running: $script_name ===" | tee -a "$LOG_FILE"
    if "$PYTHON" "$script_name" >> "$LOG_FILE" 2>&1; then
        echo "=== OK: $script_name ===" | tee -a "$LOG_FILE"
    else
        echo "=== FAILED: $script_name (exit code $?) ===" | tee -a "$LOG_FILE"
        STEPS_FAILED=$((STEPS_FAILED + 1))
    fi
}

echo "FPL Pipeline Run — $TIMESTAMP" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"

run_step "extract_fpl_clean_dataset.py"
run_step "extract_understat_v2.py"

# FBref steps: PAUSED by default as of the Opta/FBref data licensing
# dispute (Opta terminated FBref's data feed in Jan 2026, citing an
# agreement violation some sources speculate was related to automated/
# AI-driven scraping — uncomfortably close to what this pipeline does).
# Cached data from before the dispute is still fine to use as-is; this
# just stops further AUTOMATED, UNATTENDED re-scraping until the
# situation is clearer. Set RUN_FBREF_STEPS=true to re-enable, but only
# after manually confirming what FBref currently serves and that
# scraping it is still appropriate.
RUN_FBREF_STEPS="${RUN_FBREF_STEPS:-false}"
if [ "$RUN_FBREF_STEPS" = "true" ]; then
    run_step "extract_fbref_v2.py"
    run_step "extract_fbref_team_stats.py"
else
    echo "*** FBref scraping steps SKIPPED (paused pending the Opta/FBref licensing dispute — set RUN_FBREF_STEPS=true to re-enable) ***" | tee -a "$LOG_FILE"
fi
run_step "extract_understat_match_stats.py"
run_step "build_merged_dataset.py"
run_step "merge_advanced_team_stats.py"
run_step "build_features.py"
run_step "build_prediction_pipeline.py"
run_step "build_current_roster_snapshot.py"
run_step "extract_fixtures.py"
run_step "build_live_predictions.py"
run_step "build_dashboard.py"

echo "" | tee -a "$LOG_FILE"
echo "================================" | tee -a "$LOG_FILE"
echo "Run complete. $STEPS_FAILED step(s) failed. Full log: $LOG_FILE" | tee -a "$LOG_FILE"

# hand off to the watchdog — it decides whether anything needs the AI
# agent's attention, and only invokes it if something actually looks wrong
"$PYTHON" check_pipeline_health.py "$LOG_FILE"
WATCHDOG_EXIT=$?

if [ "$WATCHDOG_EXIT" -ne 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Watchdog flagged an issue — see logs/watchdog_${TIMESTAMP}.log and " | tee -a "$LOG_FILE"
    echo "the 'ai-review-*' git branch it may have created for a proposed fix." | tee -a "$LOG_FILE"
fi

exit 0  # the orchestration script itself always exits 0 — failures are
        # logged and surfaced via the watchdog, not via a crashed cron job