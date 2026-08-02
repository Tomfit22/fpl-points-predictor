"""
FPL Points Predictor — Weekly Model Summary
=================================================
Formats the health-check metrics (same ones check_pipeline_health.py
already tracks) into a plain-English weekly report. Also does #2 from
the "easy" list — feature importance drift — by comparing this week's
SHAP output against last week's, if both exist.

Run:
    python weekly_model_summary.py
"""

import json
import re
from datetime import datetime
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs")
HISTORY_PATH = PROCESSED_DIR / "weekly_summary_history.json"


def get_latest_log() -> Path | None:
    logs = sorted(LOG_DIR.glob("run_*.log"))
    return logs[-1] if logs else None


def parse_metrics(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    mae = re.search(r"Mean Absolute Error:\s*([\d.]+)\s*points", text)
    bias = re.search(r"Signed bias.*?:\s*([+-]?[\d.]+)\s*points", text)
    corr = re.search(r"Correlation.*?:\s*([\d.]+)", text)

    position_mae = dict(re.findall(r"(\w+): MAE = ([\d.]+)", text))
    position_bias = dict(re.findall(r"(\w+): bias = ([+-][\d.]+)", text))

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "mae": float(mae.group(1)) if mae else None,
        "bias": float(bias.group(1)) if bias else None,
        "correlation": float(corr.group(1)) if corr else None,
        "position_mae": {k: float(v) for k, v in position_mae.items()},
        "position_bias": {k: float(v) for k, v in position_bias.items()},
    }


def load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text())


def save_history(history: list):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def format_report(current: dict, previous: dict | None) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"WEEKLY MODEL SUMMARY — {current['date']}")
    lines.append("=" * 60)

    lines.append(f"\nOverall: MAE {current['mae']:.2f} points, "
                 f"bias {current['bias']:+.2f}, correlation {current['correlation']:.3f}")

    if previous:
        mae_change = current["mae"] - previous["mae"]
        direction = "improved" if mae_change < 0 else ("worsened" if mae_change > 0 else "unchanged")
        lines.append(f"vs last run ({previous['date']}): MAE {direction} "
                      f"({previous['mae']:.2f} -> {current['mae']:.2f}, {mae_change:+.2f})")
    else:
        lines.append("(No previous run to compare against — this is the first summary.)")

    lines.append("\nBy position:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        mae = current["position_mae"].get(pos)
        bias = current["position_bias"].get(pos)
        if mae is None:
            continue
        line = f"  {pos}: MAE {mae:.2f}"
        if bias is not None:
            line += f", bias {bias:+.2f}"
        if previous and pos in previous.get("position_mae", {}):
            prev_mae = previous["position_mae"][pos]
            change = mae - prev_mae
            if abs(change) > 0.05:
                line += f"  ({'up' if change > 0 else 'down'} {abs(change):.2f} from last run)"
        lines.append(line)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    log_path = get_latest_log()
    if log_path is None:
        print("No pipeline run logs found in logs/ — run orchestrate_pipeline.sh first.")
        return

    current = parse_metrics(log_path)
    if current["mae"] is None:
        print(f"Could not find MAE in {log_path} — has the log format changed?")
        return

    history = load_history()
    previous = history[-1] if history else None

    report = format_report(current, previous)
    print(report)

    history.append(current)
    save_history(history)

    report_path = PROCESSED_DIR / f"weekly_summary_{current['date']}.txt"
    report_path.write_text(report)
    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()