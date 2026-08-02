"""
FPL Points Predictor — Pipeline Health Watchdog
=====================================================
Reads the latest pipeline run's log, checks for two kinds of problems:
  1. A script crashed outright (a data source broke, a new bug, etc.)
  2. The validated MAE/bias drifted meaningfully from the last known-
     good baseline — a real symptom of something changing (new season
     structure, a rule change, data quality regression) even if
     nothing literally crashed.

Only if something is genuinely flagged does this invoke Claude Code
headless mode (`claude -p`) to diagnose and propose a fix — and even
then, on an ISOLATED git branch, never touching main directly. A
human reviews and merges. This is a deliberate safety boundary: an
unsupervised agent silently rewriting its own prediction logic is a
real risk for something that influences real decisions, even when
well-intentioned.

I could not test the actual `claude -p` invocation from this sandbox
(no Claude Code CLI available here) — verify the exact command works
in your own terminal before relying on this unattended. The
baseline-comparison and log-parsing logic IS tested and working.

Usage:
    python check_pipeline_health.py <path_to_run_log>

Exit code 0 = healthy (or first run, baseline established).
Exit code 1 = anomaly flagged (see logs/watchdog_<timestamp>.log).
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
LOG_DIR = PROJECT_DIR / "logs"
BASELINE_PATH = PROJECT_DIR / "data" / "processed" / "pipeline_health_baseline.json"

# how much the MAE or |bias| can drift from baseline before flagging —
# tune these once you've seen a few real runs' natural week-to-week noise
MAE_DRIFT_THRESHOLD = 0.15    # 15% relative increase
BIAS_DRIFT_THRESHOLD = 0.10   # absolute increase in |signed bias|

CLAUDE_MAX_TURNS = 15
CLAUDE_MAX_BUDGET_USD = 2.00


def parse_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    failed_steps = re.findall(r"=== FAILED: (.+?) \(exit code", text)

    mae_match = re.search(r"Mean Absolute Error:\s*([\d.]+)\s*points", text)
    bias_match = re.search(r"Signed bias.*?:\s*([+-]?[\d.]+)\s*points", text)
    corr_match = re.search(r"Correlation.*?:\s*([\d.]+)", text)

    return {
        "failed_steps": failed_steps,
        "mae": float(mae_match.group(1)) if mae_match else None,
        "bias": float(bias_match.group(1)) if bias_match else None,
        "correlation": float(corr_match.group(1)) if corr_match else None,
        "timestamp": datetime.now().isoformat(),
    }


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text())


def save_baseline(metrics: dict):
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(metrics, indent=2))


def detect_anomalies(current: dict, baseline: dict | None) -> list:
    issues = []

    if current["failed_steps"]:
        for step in current["failed_steps"]:
            issues.append(f"Script failed outright: {step}")

    if baseline is None:
        return issues  # nothing to compare against yet — first run

    if current["mae"] is not None and baseline.get("mae") is not None:
        relative_change = (current["mae"] - baseline["mae"]) / baseline["mae"]
        if relative_change > MAE_DRIFT_THRESHOLD:
            issues.append(
                f"MAE increased {relative_change:.1%}: {baseline['mae']:.3f} -> {current['mae']:.3f}"
            )

    if current["bias"] is not None and baseline.get("bias") is not None:
        bias_change = abs(current["bias"]) - abs(baseline["bias"])
        if bias_change > BIAS_DRIFT_THRESHOLD:
            issues.append(
                f"Signed bias magnitude grew: {baseline['bias']:+.3f} -> {current['bias']:+.3f}"
            )

    return issues


def invoke_claude_agent(issues: list, log_path: Path) -> bool:
    branch_name = f"ai-review-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Creating isolated branch '{branch_name}' for the agent to work on...")
    subprocess.run(["git", "checkout", "-b", branch_name], cwd=PROJECT_DIR, check=False)

    issues_text = "\n".join(f"- {i}" for i in issues)
    prompt = f"""The FPL points predictor pipeline flagged these issues on its latest run:

{issues_text}

Full run log: {log_path}

Investigate the cause (check the log, check recent data files in
data/raw and data/processed, check for API/schema changes in the
extraction scripts). If you find a clear, specific fix, make it. If
the cause is ambiguous or the fix is risky, do NOT guess — instead
write a clear summary of what you found and what you'd recommend to
data/processed/ai_review_summary.txt and stop there.

Do not touch model scoring logic (GOAL_POINTS, CLEAN_SHEET_POINTS, DC
thresholds) without being certain a real FPL rule change is the cause
— verify against the official FPL rules page first if you suspect this.

You are on an isolated branch. Do not merge or push. A human will
review your changes before anything reaches the main branch."""

    print(f"Invoking Claude Code headless mode (max {CLAUDE_MAX_TURNS} turns, "
          f"${CLAUDE_MAX_BUDGET_USD} budget cap)...")

    result = subprocess.run(
        [
            "claude", "-p", prompt,
            "--allowedTools", "Read,Edit,Bash(git:*),Bash(python3:*),Bash(cat:*),Bash(grep:*)",
            "--max-turns", str(CLAUDE_MAX_TURNS),
            "--max-budget-usd", str(CLAUDE_MAX_BUDGET_USD),
            "--output-format", "json",
        ],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )

    watchdog_log = LOG_DIR / f"watchdog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    watchdog_log.write_text(
        f"Issues detected:\n{issues_text}\n\n"
        f"Branch: {branch_name}\n\n"
        f"Claude Code exit code: {result.returncode}\n\n"
        f"--- stdout ---\n{result.stdout}\n\n"
        f"--- stderr ---\n{result.stderr}\n"
    )

    print(f"Agent run complete (exit code {result.returncode}). "
          f"Review branch '{branch_name}' and {watchdog_log} before merging anything.")
    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_pipeline_health.py <path_to_run_log>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    current = parse_log(log_path)
    baseline = load_baseline()

    if baseline is None:
        print("No baseline yet — this run establishes one. Nothing to compare against.")
        save_baseline(current)
        sys.exit(0)

    issues = detect_anomalies(current, baseline)

    if not issues:
        print(f"Pipeline healthy — MAE {current['mae']}, bias {current['bias']:+.3f} "
              f"(baseline: MAE {baseline['mae']}, bias {baseline['bias']:+.3f}). "
              f"Updating baseline.")
        save_baseline(current)
        sys.exit(0)

    print("*** Anomalies detected: ***")
    for issue in issues:
        print(f"  - {issue}")

    invoke_claude_agent(issues, log_path)
    sys.exit(1)


if __name__ == "__main__":
    main()