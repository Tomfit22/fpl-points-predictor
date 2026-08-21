"""
FPL Points Predictor — Sanity Check: Platt vs Isotonic Rankings
========================================================================
The final check before trusting the switch to Platt: do the two
calibration methods actually produce meaningfully different player
rankings, or is this a purely academic distinction? Uses the EXISTING
live_predictions.csv (already has both pred_p_clean_sheet_raw and the
isotonic-calibrated pred_p_clean_sheet saved from the earlier run) —
computes what Platt would give for the same raw values directly, no
need to re-run the whole pipeline first.

Compares:
  1. Spearman rank correlation between isotonic and Platt clean sheet
     probabilities, for GK/DEF specifically (the positions this
     actually affects).
  2. Top 25 defenders/GKs by each method, side by side.

Run:
    python fit_and_save_cs_calibrator.py    (produces cs_platt_calibrator.json)
    python compare_platt_vs_isotonic_rankings.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROCESSED_DIR = Path("data/processed")


def main():
    live_path = PROCESSED_DIR / "live_predictions.csv"
    platt_path = PROCESSED_DIR / "cs_platt_calibrator.json"

    if not live_path.exists():
        print(f"{live_path} not found.")
        return
    if not platt_path.exists():
        print(f"{platt_path} not found - run fit_and_save_cs_calibrator.py first.")
        return

    df = pd.read_csv(live_path)
    if "pred_p_clean_sheet_raw" not in df.columns:
        print("pred_p_clean_sheet_raw not found in live_predictions.csv - "
              "this file needs to be from a run AFTER the calibration wiring was added.")
        return

    with open(platt_path) as f:
        platt = json.load(f)
    coef, intercept = platt["coefficient"], platt["intercept"]

    raw = df["pred_p_clean_sheet_raw"].clip(1e-6, 1 - 1e-6)
    logit_raw = np.log(raw / (1 - raw))
    df["pred_p_clean_sheet_platt"] = 1 / (1 + np.exp(-(coef * logit_raw + intercept)))

    def_gk = df[df["position"].isin(["GK", "DEF"])].drop_duplicates("player_id").copy()

    print("=" * 70)
    print("SPEARMAN RANK CORRELATION - Isotonic vs Platt (GK/DEF)")
    print("=" * 70)
    rho, pval = spearmanr(def_gk["pred_p_clean_sheet"], def_gk["pred_p_clean_sheet_platt"])
    print(f"Spearman rho: {rho:.4f} (p={pval:.2e})")
    if rho > 0.95:
        print("Extremely high agreement - the two methods produce nearly identical rankings.")
    elif rho > 0.85:
        print("High agreement - rankings are very similar with minor reordering.")
    else:
        print("Meaningful disagreement - worth inspecting individual cases before switching.")

    print("\n" + "=" * 70)
    print("TOP 25 GK/DEF BY ISOTONIC vs TOP 25 BY PLATT")
    print("=" * 70)
    top_iso = def_gk.sort_values("pred_p_clean_sheet", ascending=False).head(25)["player_name"].tolist()
    top_platt = def_gk.sort_values("pred_p_clean_sheet_platt", ascending=False).head(25)["player_name"].tolist()

    overlap = set(top_iso) & set(top_platt)
    print(f"Overlap: {len(overlap)}/25 players appear in BOTH top-25 lists")

    only_iso = set(top_iso) - set(top_platt)
    only_platt = set(top_platt) - set(top_iso)
    if only_iso:
        print(f"\nIn isotonic top-25 but NOT Platt top-25: {sorted(only_iso)}")
    if only_platt:
        print(f"In Platt top-25 but NOT isotonic top-25: {sorted(only_platt)}")
    if not only_iso and not only_platt:
        print("\nIdentical top-25 sets (order may still differ slightly).")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if len(overlap) >= 22 and rho > 0.9:
        print("The switch to Platt does not meaningfully change who gets recommended.")
        print("Safe to proceed with Platt as the production calibrator.")
    else:
        print("There's real, meaningful disagreement between the two methods for specific")
        print("players - worth a closer look at the 'only in one list' players above before")
        print("finalizing the switch.")


if __name__ == "__main__":
    main()
