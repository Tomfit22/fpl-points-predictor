"""
FPL Points Predictor — Final Check: FPL Points Impact of Calibration Choice
========================================================================
The real question for an FPL predictor isn't "does the clean sheet
probability change" — it's "does WHO gets recommended change". CS
probability is only one input into predicted_points; a shift in CS%
might barely move a player's overall predicted points if the clean
sheet points contribution is a small share of their total.

Recomputes predicted_points under all three approaches (raw, isotonic,
Platt) using the SAME already-saved components (pts_clean_sheet,
CLEAN_SHEET_POINTS by position, pred_p_60plus) already in
live_predictions.csv from the current Platt-based run — no need to
regenerate the whole pipeline three times.

Compares:
  - Top 10 GK by expected points, under each method
  - Top 20 DEF by expected points, under each method
  - Top 50 overall by expected points, under each method
  - Average |predicted_points difference| between methods

Run:
    python compare_fpl_points_impact.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")

CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}


def compute_cs_pts(cs_prob, position, p_60plus):
    points_per_position = position.map(CLEAN_SHEET_POINTS).fillna(0)
    return cs_prob * points_per_position * p_60plus


def top_n_overlap(df, col, n, label_a, label_b, col_a, col_b):
    top_a = set(df.sort_values(col_a, ascending=False).head(n)["player_name"])
    top_b = set(df.sort_values(col_b, ascending=False).head(n)["player_name"])
    overlap = top_a & top_b
    print(f"  {label_a} vs {label_b}: {len(overlap)}/{n} overlap")
    only_a = top_a - top_b
    only_b = top_b - top_a
    if only_a:
        print(f"    Only in {label_a}: {sorted(only_a)}")
    if only_b:
        print(f"    Only in {label_b}: {sorted(only_b)}")


def main():
    live_path = PROCESSED_DIR / "live_predictions.csv"
    isotonic_path = PROCESSED_DIR / "cs_isotonic_calibrator.pkl"

    if not live_path.exists():
        print(f"{live_path} not found.")
        return
    df = pd.read_csv(live_path)

    required = ["pred_p_clean_sheet_raw", "pts_clean_sheet", "predicted_points",
                "position", "pred_p_60plus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return

    df["cs_pts_platt"] = df["pts_clean_sheet"]
    df["points_platt"] = df["predicted_points"]

    df["cs_pts_raw"] = compute_cs_pts(df["pred_p_clean_sheet_raw"], df["position"], df["pred_p_60plus"])
    df["points_raw"] = df["predicted_points"] - df["cs_pts_platt"] + df["cs_pts_raw"]

    have_isotonic = isotonic_path.exists()
    if have_isotonic:
        with open(isotonic_path, "rb") as f:
            saved = pickle.load(f)
        iso_model = saved["isotonic_model"] if isinstance(saved, dict) and "isotonic_model" in saved else saved
        clip_min = saved.get("clip_min", 0.0) if isinstance(saved, dict) else 0.0
        clip_max = saved.get("clip_max", 1.0) if isinstance(saved, dict) else 1.0
        cs_isotonic = iso_model.predict(df["pred_p_clean_sheet_raw"].values)
        cs_isotonic = np.clip(cs_isotonic, clip_min, clip_max)
        df["cs_pts_isotonic"] = compute_cs_pts(pd.Series(cs_isotonic), df["position"], df["pred_p_60plus"])
        df["points_isotonic"] = df["predicted_points"] - df["cs_pts_platt"] + df["cs_pts_isotonic"]
    else:
        print("(cs_isotonic_calibrator.pkl no longer found - skipping isotonic comparison, "
              "showing raw vs Platt only)\n")

    print("=" * 70)
    print("AVERAGE ABSOLUTE PREDICTED_POINTS DIFFERENCE")
    print("=" * 70)
    diff_raw_platt = (df["points_raw"] - df["points_platt"]).abs().mean()
    print(f"Raw vs Platt: {diff_raw_platt:.4f} points on average")
    if have_isotonic:
        diff_iso_platt = (df["points_isotonic"] - df["points_platt"]).abs().mean()
        print(f"Isotonic vs Platt: {diff_iso_platt:.4f} points on average")

    def_gk = df[df["position"].isin(["GK", "DEF"])].drop_duplicates("player_id")

    print("\n" + "=" * 70)
    print("TOP 10 GK BY EXPECTED POINTS")
    print("=" * 70)
    gk = def_gk[def_gk["position"] == "GK"]
    top_n_overlap(gk, "player_name", 10, "Raw", "Platt", "points_raw", "points_platt")
    if have_isotonic:
        top_n_overlap(gk, "player_name", 10, "Isotonic", "Platt", "points_isotonic", "points_platt")

    print("\n" + "=" * 70)
    print("TOP 20 DEF BY EXPECTED POINTS")
    print("=" * 70)
    defn = def_gk[def_gk["position"] == "DEF"]
    top_n_overlap(defn, "player_name", 20, "Raw", "Platt", "points_raw", "points_platt")
    if have_isotonic:
        top_n_overlap(defn, "player_name", 20, "Isotonic", "Platt", "points_isotonic", "points_platt")

    print("\n" + "=" * 70)
    print("TOP 50 OVERALL BY EXPECTED POINTS")
    print("=" * 70)
    all_players = df.drop_duplicates("player_id")
    top_n_overlap(all_players, "player_name", 50, "Raw", "Platt", "points_raw", "points_platt")
    if have_isotonic:
        top_n_overlap(all_players, "player_name", 50, "Isotonic", "Platt", "points_isotonic", "points_platt")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("If overlaps above are high (45+/50, 18+/20, 9+/10) and average point")
    print("differences are small (well under 0.5 points), the calibration method")
    print("choice has minimal practical impact on actual FPL recommendations -")
    print("calibration work can be considered finished for now.")


if __name__ == "__main__":
    main()
