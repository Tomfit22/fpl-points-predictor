"""
FPL Points Predictor — Ownership Percentage Estimator
===========================================================
FPL's own API's per-gameweek 'selected' field is a raw COUNT of
managers who own a player, not a percentage — and we don't have the
total manager count needed to convert it, since that's not published
per-gameweek historically (only as a live "right now" snapshot via a
different field).

This estimates the total manager count using FPL's own fixed squad
composition rule: every valid 15-man squad has EXACTLY 2 GK, 5 DEF,
5 MID, 3 FWD. That means summing 'selected' across all players of one
position, then dividing by that position's squad-slot count, gives an
independent estimate of the total manager count — and we can compute
this FOUR separate ways (one per position) and cross-check them
against each other as a sanity check, rather than trusting one number
blindly.

Usable on ANY gameweek's data, historical or current — solves the
"can't get historical ownership %" gap that a live API snapshot alone
can't, since FPL only ever tells you TODAY's percentage, not what it
was on a past gameweek.

Import and use:
    from estimate_ownership import add_ownership_pct
    df = add_ownership_pct(df)  # adds an 'ownership_pct' column

Run standalone to see the cross-check diagnostic on your real data:
    python estimate_ownership.py
"""

from pathlib import Path

import pandas as pd

SQUAD_SLOTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
PROCESSED_DIR = Path("data/processed")


def estimate_total_managers(df: pd.DataFrame, gameweek: int = None) -> dict:
    """Returns the per-position estimate of total managers, plus the
    consensus (median) estimate. Cross-checking all four independently
    is the whole point — if they wildly disagree, something's off
    (e.g. mixed gameweeks, missing players) and shouldn't be trusted
    blindly."""
    if gameweek is not None:
        df = df[df["gameweek"] == gameweek]

    estimates = {}
    for position, slots in SQUAD_SLOTS.items():
        pos_df = df[df["position"] == position]
        total_selected = pos_df["selected"].sum()
        estimates[position] = total_selected / slots

    values = list(estimates.values())
    estimates["consensus_median"] = pd.Series(values).median()
    estimates["consensus_mean"] = pd.Series(values).mean()
    # spread as a % of the median — a quick, honest measure of how much
    # the four independent estimates actually agree with each other
    estimates["spread_pct"] = (max(values) - min(values)) / estimates["consensus_median"] * 100
    return estimates


def add_ownership_pct(df: pd.DataFrame, gameweek: int = None) -> pd.DataFrame:
    """Adds an 'ownership_pct' column using the consensus total-manager
    estimate. If gameweek is None, assumes df is already a single
    gameweek's snapshot (e.g. the latest-per-player snapshot used for
    live predictions)."""
    df = df.copy()
    estimates = estimate_total_managers(df, gameweek=gameweek)
    total_managers = estimates["consensus_median"]
    df["ownership_pct"] = (df["selected"] / total_managers) * 100
    return df


def main():
    model_ready_path = PROCESSED_DIR / "model_ready_dataset.csv"
    if not model_ready_path.exists():
        print(f"{model_ready_path} not found — nothing to check.")
        return

    df = pd.read_csv(model_ready_path)
    latest_gw = df["gameweek"].max()

    print(f"Cross-checking total manager estimate using gameweek {latest_gw}...")
    estimates = estimate_total_managers(df, gameweek=latest_gw)

    print("\nPer-position independent estimates of total managers:")
    for pos in SQUAD_SLOTS:
        print(f"  {pos}: {estimates[pos]:,.0f}")

    print(f"\nConsensus (median): {estimates['consensus_median']:,.0f}")
    print(f"Consensus (mean):   {estimates['consensus_mean']:,.0f}")
    print(f"Spread across the 4 estimates: {estimates['spread_pct']:.1f}%")

    if estimates["spread_pct"] > 15:
        print("\n*** WARNING: the four position-based estimates disagree by more than 15% — "
              "treat this gameweek's ownership % as unreliable. Possible causes: mixed "
              "gameweeks in the data, players with missing 'selected' values, or a very "
              "early gameweek where squad compositions haven't stabilized yet. ***")
    else:
        print("\nEstimates agree reasonably well — consensus figure looks trustworthy.")


if __name__ == "__main__":
    main()