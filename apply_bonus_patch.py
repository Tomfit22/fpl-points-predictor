"""
Applies the position-aware bonus averaging patch to build_live_predictions.py.
Run once from inside FPL_Project:

    python apply_bonus_patch.py

Replaces the single flat bonus average (which understates forwards'
real bonus potential and overstates goalkeepers'/defenders' — confirmed
on real data: FWD 0.52 vs GK 0.21 vs DEF 0.22 vs MID 0.32, against one
flat 0.29 average for everyone) with a per-position average, falling
back to the overall average for any position with too little data to
trust its own number.

This is a deliberately modest, honest fix — NOT a full per-player
bonus model. That was investigated already (see "Bonus by pos.py" /
"Bonus point pred.py") and confirmed genuinely hard: BPS regression
R²=0.082, and exact bonus-value prediction doesn't beat a naive
"always 0" baseline on held-out data. Real signal exists (recall/
precision both ~2x random chance) but not enough to trust an exact
per-player prediction — position-level averaging is the safe,
defensible improvement available right now.

Makes a backup first and is safe to re-run.
"""

import shutil
from pathlib import Path

TARGET = Path("build_live_predictions.py")


def main():
    if not TARGET.exists():
        print(f"{TARGET} not found — run this from inside FPL_Project.")
        return

    content = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(".py.bonus_bak")
    shutil.copy(TARGET, backup)
    print(f"Backed up original to {backup}")

    changes_made = []

    # --- Change 1: compute per-position bonus averages in main() ---
    old_1 = '    avg_bonus_when_played = df[df["minutes"] >= 60]["bonus"].mean() if "bonus" in df.columns else 0.3'
    new_1 = '''    # Per-position average bonus when played 60+ minutes — confirmed on
    # real data this varies meaningfully by position (FWD 0.52 vs GK
    # 0.21 vs DEF 0.22 vs MID 0.32), so one flat average understates
    # forwards' real bonus potential and overstates goalkeepers'/
    # defenders'. Falls back to the overall average for any position
    # with too little data to trust its own number, and to the same
    # 0.3 default as before if there's no bonus column at all.
    #
    # Deliberately NOT a full per-player bonus model — that was
    # investigated already and confirmed genuinely hard (BPS regression
    # R²=0.082; exact bonus prediction doesn't beat a naive "always 0"
    # baseline on held-out data — see "Bonus by pos.py" / "Bonus point
    # pred.py"). Position averaging is the honest, defensible
    # improvement available without overclaiming precision we don't have.
    MIN_ROWS_FOR_POSITION_BONUS_AVERAGE = 100
    if "bonus" in df.columns:
        played_60_df = df[df["minutes"] >= 60]
        overall_avg_bonus = played_60_df["bonus"].mean()
        position_counts = played_60_df.groupby("position")["bonus"].count()
        avg_bonus_by_position = played_60_df.groupby("position")["bonus"].mean().to_dict()
        for position, count in position_counts.items():
            if count < MIN_ROWS_FOR_POSITION_BONUS_AVERAGE:
                avg_bonus_by_position[position] = overall_avg_bonus
        print(f"  Bonus averages by position: "
              f"{ {k: round(v, 3) for k, v in avg_bonus_by_position.items()} } "
              f"(overall flat average was {overall_avg_bonus:.3f})")
    else:
        avg_bonus_by_position = {}
        overall_avg_bonus = 0.3'''

    if old_1 in content:
        content = content.replace(old_1, new_1)
        changes_made.append("1. Added per-position bonus average computation")
    elif "avg_bonus_by_position" in content:
        changes_made.append("1. Already applied — skipped")
    else:
        print("*** Change 1 FAILED: exact original avg_bonus_when_played line not found. ***")

    # --- Change 2: predict_points() signature + bonus_pts line ---
    old_2a = "def predict_points(df: pd.DataFrame, components: dict, avg_bonus_when_played: float) -> pd.DataFrame:"
    new_2a = "def predict_points(df: pd.DataFrame, components: dict, avg_bonus_by_position: dict, overall_avg_bonus: float) -> pd.DataFrame:"

    old_2b = '    bonus_pts = avg_bonus_when_played * df["pred_p_60plus"]'
    new_2b = '    bonus_pts = df["position"].map(avg_bonus_by_position).fillna(overall_avg_bonus) * df["pred_p_60plus"]'

    if old_2a in content and old_2b in content:
        content = content.replace(old_2a, new_2a)
        content = content.replace(old_2b, new_2b)
        changes_made.append("2. Updated predict_points() to use per-position bonus")
    elif "def predict_points(df: pd.DataFrame, components: dict, avg_bonus_by_position: dict" in content:
        changes_made.append("2. Already applied — skipped")
    else:
        print("*** Change 2 FAILED: predict_points() text not found as expected. ***")

    # --- Change 3: run_monte_carlo_simulation() signature + bonus_pts line ---
    old_3a = '''def run_monte_carlo_simulation(df: pd.DataFrame, avg_bonus_when_played: float,
                                n_sims: int = 5000, random_seed: int = 42) -> pd.DataFrame:'''
    new_3a = '''def run_monte_carlo_simulation(df: pd.DataFrame, avg_bonus_by_position: dict, overall_avg_bonus: float,
                                n_sims: int = 5000, random_seed: int = 42) -> pd.DataFrame:'''

    old_3b = '    bonus_pts = avg_bonus_when_played * played_60.astype(float)'
    new_3b = '''    bonus_rate_per_player = df["position"].map(avg_bonus_by_position).fillna(overall_avg_bonus).values[:, None]
    bonus_pts = bonus_rate_per_player * played_60.astype(float)'''

    old_3_docstring = '''    Honest limitation: bonus has no real per-player distribution to draw
    from (already proven bonus can't be reliably differentiated between
    players — see build_bonus_predictions.py). It's added as a FIXED
    amount conditional on playing 60+, matching the same expected value
    as the deterministic formula, but contributing zero simulated
    variance — better to be upfront about this than fake uncertainty we
    don't actually have.
    """'''
    new_3_docstring = '''    Honest limitation: bonus has no real PER-PLAYER distribution to draw
    from (already proven exact bonus value can't be reliably predicted
    per player — see "Bonus by pos.py" / "Bonus point pred.py": BPS
    regression R²=0.082, doesn't beat a naive "always 0" baseline on
    held-out data). It IS differentiated by POSITION now (forwards
    genuinely earn more bonus than goalkeepers on average), but still
    added as a FIXED amount conditional on playing 60+ WITHIN a
    position — matching that position's expected value, but
    contributing zero simulated variance within it. Better to be
    upfront about this than fake player-level uncertainty we don't
    actually have.
    """'''

    if old_3a in content and old_3b in content:
        content = content.replace(old_3a, new_3a)
        content = content.replace(old_3b, new_3b)
        if old_3_docstring in content:
            content = content.replace(old_3_docstring, new_3_docstring)
        changes_made.append("3. Updated run_monte_carlo_simulation() to use per-position bonus")
    elif "def run_monte_carlo_simulation(df: pd.DataFrame, avg_bonus_by_position: dict" in content:
        changes_made.append("3. Already applied — skipped")
    else:
        print("*** Change 3 FAILED: run_monte_carlo_simulation() text not found as expected. ***")

    # --- Change 4: update the two call sites ---
    old_4a = "    result = predict_points(fixture_rows, components, avg_bonus_when_played)"
    new_4a = "    result = predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus)"

    old_4b = "    result = run_monte_carlo_simulation(result, avg_bonus_when_played)"
    new_4b = "    result = run_monte_carlo_simulation(result, avg_bonus_by_position, overall_avg_bonus)"

    if old_4a in content and old_4b in content:
        content = content.replace(old_4a, new_4a)
        content = content.replace(old_4b, new_4b)
        changes_made.append("4. Updated both call sites")
    elif "predict_points(fixture_rows, components, avg_bonus_by_position, overall_avg_bonus)" in content:
        changes_made.append("4. Already applied — skipped")
    else:
        print("*** Change 4 FAILED: call site text not found as expected. ***")

    TARGET.write_text(content, encoding="utf-8")

    print("\n=== Summary ===")
    for c in changes_made:
        print(f"  {c}")
    print(f"\nDone. If anything looks wrong, restore with:\n  cp {backup} {TARGET}")


if __name__ == "__main__":
    main()
