"""
FPL Points Predictor — Feature Engineering
=============================================
Turns the raw merged player-gameweek table into a model-ready table by
adding:

  1. Rolling player form (last 3/5/10 games) — minutes, points, xG, xA,
     bonus, defensive contributions. Using ROLLING AVERAGES rather than
     single-match snapshots, since form predicts next week's points far
     better than one match's raw numbers.

  2. Opponent context — each upcoming opponent's recent defensive/attacking
     strength (goals conceded, xG conceded, goals scored), so the model can
     actually learn "this player usually does well against leaky defenses"
     rather than just "this player is generally good."

  3. Days of rest since the player's last match.

CRITICAL: every rolling/form feature is shifted by 1 game before the
window is computed, so a gameweek's features only ever use STRICTLY
PRIOR matches. Without this, the model would be trained on data that
includes the outcome it's trying to predict (leakage) — e.g. gameweek
10's "rolling average points" would otherwise include gameweek 10's own
points, which the model won't actually know at prediction time.

Run:
    python build_features.py
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

ROLLING_WINDOWS = [3, 5, 10]

PLAYER_ROLLING_COLS = {
    "minutes": "minutes",
    "actual_points": "points",
    "goals": "goals",
    "assists": "assists",
    "bonus": "bonus",
    "bps": "bps",
    "defensive_contribution": "def_contrib",
    "us_xG": "xG",
    "us_xA": "xA",
    "us_shots": "shots",
    "us_key_passes": "key_passes",
    "fb_Performance_SoT": "SoT",  # shots on target — filters out blocked/off-target attempts
    "us_xGChain": "xGChain",  # this player's involvement in ANY possession leading to a shot
    "us_xGBuildup": "xGBuildup",  # same, but excluding their own shots/key passes — pure buildup play
    "fb_Performance_Fld": "Fld",  # fouls drawn — proxy for being fouled in dangerous areas, incl. penalty-won likelihood
    "fb_Performance_Crs": "Crs",  # crosses — relevant assist predictor, esp. for wide players/fullbacks
    "fb_Performance_CrdY": "CrdY",  # yellow cards — player's own disciplinary tendency
    "fb_Performance_CrdR": "CrdR",  # red cards — rare, mostly for a naive rate baseline rather than real regression
    "fb_Performance_Fls": "Fls",  # fouls COMMITTED — different from Fld (fouls drawn); relevant to card risk
    "saves": "saves",  # goalkeeper saves
    "penalties_saved": "pens_saved",  # goalkeeper penalty saves — very rare, mostly for a naive baseline
    "starts": "starts",  # rolling mean of this = start RATE over that window (e.g. roll5_starts=0.8 -> started 4 of last 5)
    "fb_Performance_TklW": "TklW",  # tackles won — component of defensive_contribution's CBIT tally
    "fb_Performance_Int": "Int",  # interceptions — same
    # NEW: FPL's own official granular defensive breakdown — separate from
    # the FBref-sourced TklW/Int above (which depend on the now-uncertain
    # Opta/FBref data feed). These are zero legal risk, same source as
    # everything else in this project.
    "clearances_blocks_interceptions": "CBI",
    "recoveries": "Recoveries",
    "tackles": "FPL_Tackles",  # distinct name from FBref's TklW to avoid collision
    "clean_sheets": "clean_sheets",  # directly feeds BPS for DEF/GK
}


# =========================
# PLAYER ROLLING FORM
# =========================
def add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "gameweek"]).reset_index(drop=True)

    for source_col, short_name in PLAYER_ROLLING_COLS.items():
        if source_col not in df.columns:
            continue
        # shift(1) first: a gameweek's rolling window uses only PRIOR games
        shifted = df.groupby("player_id")[source_col].shift(1)
        for window in ROLLING_WINDOWS:
            df[f"roll{window}_{short_name}"] = (
                shifted.groupby(df["player_id"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        # season-to-date (expanding) average — uses ALL prior games this
        # season, not just the last N. Much less noisy than a short window,
        # especially early in the season when a 5-game window barely exists.
        df[f"season_{short_name}"] = (
            shifted.groupby(df["player_id"]).expanding(min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

    # --- derived efficiency ratios (built from already-shifted rolling
    # numbers above, so they inherit the same no-leakage guarantee) ---
    # xG per shot: average chance QUALITY, independent of shot volume —
    # separates "gets into great positions" from "shoots from anywhere"
    for window in ROLLING_WINDOWS + ["season"]:
        w = window if window == "season" else f"roll{window}"
        xg_col, shots_col, sot_col, goals_col = f"{w}_xG", f"{w}_shots", f"{w}_SoT", f"{w}_goals"
        if xg_col in df.columns and shots_col in df.columns:
            df[f"{w}_xG_per_shot"] = (df[xg_col] / df[shots_col]).replace([float("inf")], pd.NA)
        if sot_col in df.columns and shots_col in df.columns:
            df[f"{w}_SoT_rate"] = (df[sot_col] / df[shots_col]).replace([float("inf")], pd.NA)
        if goals_col in df.columns and shots_col in df.columns:
            df[f"{w}_conversion_rate"] = (df[goals_col] / df[shots_col]).replace([float("inf")], pd.NA)

    # --- per-90 normalization ---
    # a rolling AVERAGE of goals-per-game conflates scoring rate with playing
    # time — a player averaging 0.3 across full 90-min starts is a very
    # different signal than the same 0.3 built from 30-minute cameos.
    # Per-90 rate = (rolling total goals / rolling total minutes) * 90,
    # computed from shifted RAW sums (not the already-averaged columns
    # above) so it's a true rate, not an average-of-averages.
    for stat_col, short_name in [("goals", "goals"), ("assists", "assists"), ("us_xG", "xG"), ("us_xA", "xA")]:
        if stat_col not in df.columns:
            continue
        shifted_stat = df.groupby("player_id")[stat_col].shift(1)
        shifted_minutes = df.groupby("player_id")["minutes"].shift(1)
        for window in ROLLING_WINDOWS:
            sum_stat = shifted_stat.groupby(df["player_id"]).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
            sum_minutes = shifted_minutes.groupby(df["player_id"]).rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
            df[f"roll{window}_{short_name}_per90"] = (sum_stat / sum_minutes.replace(0, pd.NA)) * 90

    # days of rest since this player's previous match
    df["prev_match_date"] = df.groupby("player_id")["match_date"].shift(1)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["prev_match_date"] = pd.to_datetime(df["prev_match_date"])
    df["days_since_last_game"] = (df["match_date"] - df["prev_match_date"]).dt.days

    # home/away as a clean 0/1 feature
    df["was_home_int"] = df["was_home"].astype(int)

    # consecutive starts streak — a stronger "nailed on" signal than a rolling
    # average alone, e.g. distinguishes a player who started their last 5 in a
    # row from one who started 4 of the last 5 but was just rested/rotated.
    # Computed on the SHIFTED series (prior games only) so it never includes
    # the very game whose features we're building.
    if "starts" in df.columns:
        shifted_starts = df.groupby("player_id")["starts"].shift(1)

        def _consecutive_run(s: pd.Series) -> pd.Series:
            s = s.fillna(0)
            streak, current = [], 0
            for v in s:
                current = current + 1 if v == 1 else 0
                streak.append(current)
            return pd.Series(streak, index=s.index)

        df["consecutive_starts"] = shifted_starts.groupby(df["player_id"]).transform(_consecutive_run)
    else:
        log.warning("'starts' column not found — skipping consecutive_starts feature. "
                     "Check whether the FPL API's history actually includes a 'starts' field.")

    return df.copy()  # defragment: many individual column inserts above slow down later ops otherwise


def add_penalty_taker_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers penalty-taker status from historical penalty attempts
    (FBref's Performance_PKatt) — no source directly states who a team's
    designated taker is, so this is a PROXY, not a confirmed fact:

      - "primary" = the player with the most cumulative penalty attempts
        on their team so far this season (strictly prior games only)
      - "backup" = the player with the second-most, PROVIDED they have at
        least one attempt themselves

    Known limitations, worth keeping in mind when using these features:
      - A true backup who simply hasn't had an attempt yet won't be
        captured — this only sees taker history that's actually happened.
      - It reacts SLOWLY to real-world pecking-order changes (e.g. a new
        signing named first-choice taker won't show up as "primary" here
        until they've actually taken and built up a few penalties).
      - Early season, most players will have 0 attempts and neither flag
        will be set for anyone on that team yet — expected, not a bug.
    """
    if "fb_Performance_PKatt" not in df.columns:
        log.warning("fb_Performance_PKatt column not found — skipping penalty-taker features")
        return df

    df = df.sort_values(["player_id", "gameweek"]).reset_index(drop=True)

    # season-to-date cumulative attempts, using only strictly prior games
    shifted_pkatt = df.groupby("player_id")["fb_Performance_PKatt"].shift(1).fillna(0)
    df["season_PK_attempts"] = (
        shifted_pkatt.groupby(df["player_id"]).expanding(min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )

    # rank each team's players by season-to-date attempts, per gameweek
    df["_pen_rank"] = (
        df.groupby(["team", "gameweek"])["season_PK_attempts"]
        .rank(method="first", ascending=False)
    )
    df["is_primary_pen_taker"] = ((df["_pen_rank"] == 1) & (df["season_PK_attempts"] > 0)).astype(int)
    df["is_backup_pen_taker"] = ((df["_pen_rank"] == 2) & (df["season_PK_attempts"] > 0)).astype(int)
    df = df.drop(columns=["_pen_rank"])

    return df.copy()


# =========================
# TEAM-LEVEL MATCH LOG (derived from player rows)
# =========================
def build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per team per fixture: goals scored/conceded and total xG for
    that match, aggregated from the player-level rows.

    Deliberately grouped by (fixture_id, was_home) rather than (team,
    gameweek): FPL's 'team' column reflects each player's CURRENT club
    applied to every historical row, so for anyone who transferred
    mid-season, their pre-transfer rows carry the WRONG team. Grouping by
    team directly caused rows to fan out when merged back onto the player
    table (29,747 -> 63,860 rows) because a transferred player's
    mislabeled rows created phantom extra "matches" for their current
    club. fixture_id uniquely identifies a real match regardless of any
    individual row's team label, so it avoids the problem at the source.
    """
    side_goals = (
        df.groupby(["fixture_id", "was_home", "gameweek", "match_date", "opponent_team"])["goals"]
        .sum()
        .reset_index()
        .rename(columns={"goals": "team_goals_scored"})
    )
    # label each side with the most common team name among its rows — robust
    # to the rare transferred-player row, since most teammates on a side
    # still carry the correct label
    team_label = (
        df.groupby(["fixture_id", "was_home"])["team"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={"team": "team"})
    )
    side_goals = side_goals.merge(team_label, on=["fixture_id", "was_home"], how="left")

    team_goals_conceded = (
        df.groupby(["fixture_id", "was_home"])["goals_conceded"]
        .max()
        .reset_index()
        .rename(columns={"goals_conceded": "team_goals_conceded"})
    )

    team_xg_for = (
        df.groupby(["fixture_id", "was_home"])["us_xG"]
        .sum()
        .reset_index()
        .rename(columns={"us_xG": "team_xG_for"})
    )

    # NEW: team-level shots — free from data we already have (Understat's
    # per-player us_shots), same fixture_id-mirroring trick as xG_against
    team_shots_for = (
        df.groupby(["fixture_id", "was_home"])["us_shots"]
        .sum()
        .reset_index()
        .rename(columns={"us_shots": "team_shots_for"})
    )

    team_log = side_goals.merge(team_goals_conceded, on=["fixture_id", "was_home"], how="left")
    team_log = team_log.merge(team_xg_for, on=["fixture_id", "was_home"], how="left")
    team_log = team_log.merge(team_shots_for, on=["fixture_id", "was_home"], how="left")

    # xG AGAINST for a team = the opponent's (the other side of the SAME
    # fixture) xG FOR — join on fixture_id + the flipped was_home, which is
    # exact and doesn't depend on team-name matching at all
    mirror = team_log[["fixture_id", "was_home", "team_xG_for"]].copy()
    mirror["was_home"] = ~mirror["was_home"]
    mirror = mirror.rename(columns={"team_xG_for": "team_xG_against"})
    team_log = team_log.merge(mirror, on=["fixture_id", "was_home"], how="left")

    # NEW: same mirroring trick for shots against
    shots_mirror = team_log[["fixture_id", "was_home", "team_shots_for"]].copy()
    shots_mirror["was_home"] = ~shots_mirror["was_home"]
    shots_mirror = shots_mirror.rename(columns={"team_shots_for": "team_shots_against"})
    team_log = team_log.merge(shots_mirror, on=["fixture_id", "was_home"], how="left")

    # NEW: merge in the advanced team stats collected from FBref's team_stats
    # box (possession, SoT%, save%, real recorded cards) and Understat's
    # match info (PPDA, deep completions, xPTS) — joined on (team,
    # match_date), the same key pattern used throughout this project
    advanced_path = PROCESSED_DIR / "advanced_team_stats.csv"
    if advanced_path.exists():
        advanced_stats = pd.read_csv(advanced_path)
        advanced_stats["match_date"] = pd.to_datetime(advanced_stats["match_date"]).dt.date
        team_log["match_date"] = pd.to_datetime(team_log["match_date"]).dt.date
        team_log = team_log.merge(advanced_stats, on=["team", "match_date"], how="left")
    else:
        log.warning("data/processed/advanced_team_stats.csv not found — skipping the new "
                     "possession/PPDA/cards features. Run merge_advanced_team_stats.py first "
                     "if you want these included.")

    return team_log


def add_team_rolling_features(team_log: pd.DataFrame) -> pd.DataFrame:
    team_log = team_log.sort_values(["team", "gameweek"]).reset_index(drop=True)

    team_stat_cols = {
        "team_goals_scored": "goals_scored",
        "team_goals_conceded": "goals_conceded",
        "team_xG_for": "xG_for",
        "team_xG_against": "xG_against",
        "team_shots_for": "shots_for",
        "team_shots_against": "shots_against",
        # NEW: possession/SoT%/save%/cards from FBref, PPDA/deep/xPTS from
        # Understat — added here so the existing generic rolling+own_/opp_
        # mirroring logic picks them up automatically, no other changes needed
        "fb_team_possession": "possession",
        "fb_team_sot_pct": "sot_pct",
        "fb_team_saves_pct": "saves_pct",
        "fb_team_yellow_cards": "match_yellow_cards",  # distinct name from the existing roll5_CrdY (rolling-average estimate) — this is the REAL recorded count per match
        "fb_team_red_cards": "match_red_cards",
        "us_team_ppda": "ppda",
        "us_team_deep": "deep_completions",
        "us_team_xpts": "xpts",
    }

    for source_col, short_name in team_stat_cols.items():
        if source_col not in team_log.columns:
            log.warning("'%s' not found in team_log — skipping this feature "
                         "(likely means advanced_team_stats.csv wasn't merged in).", source_col)
            continue
        shifted = team_log.groupby("team")[source_col].shift(1)
        for window in ROLLING_WINDOWS:
            team_log[f"team_roll{window}_{short_name}"] = (
                shifted.groupby(team_log["team"])
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        # season-to-date (expanding) average — a much more stable read on
        # true team strength than a 5-game window, especially early season
        # when a short window is itself mostly noise.
        team_log[f"team_season_{short_name}"] = (
            shifted.groupby(team_log["team"]).expanding(min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

    return team_log


# =========================
# MAIN
# =========================
def main():
    log.info("Loading merged dataset...")
    df = pd.read_csv(PROCESSED_DIR / "merged_player_gameweek.csv")
    n_input_rows = len(df)
    log.info("Loaded %d rows", n_input_rows)

    log.info("Adding player rolling form features...")
    df = add_player_rolling_features(df)

    log.info("Inferring penalty-taker status...")
    df = add_penalty_taker_features(df)

    log.info("Building team match log...")
    team_log = build_team_match_log(df)

    # Safety net: even after the fixture_id-based fix, a rare mislabeled row
    # (or a genuine double-gameweek) could still produce more than one
    # team_log row for the same (team, gameweek) pair, which would silently
    # fan out the final merge below. Guarantee at most one row per
    # (team, gameweek) no matter what, rather than trusting that edge case
    # can't happen at full scale.
    dupe_mask = team_log.duplicated(subset=["team", "gameweek"], keep=False)
    if dupe_mask.any():
        log.warning("Found %d team_log rows sharing a (team, gameweek) pair — "
                     "keeping one per pair to guarantee no row duplication downstream. "
                     "This is usually a rare mislabeled fixture or a genuine "
                     "double-gameweek; investigate data/processed/team_log_duplicates.csv "
                     "if this number looks large.", dupe_mask.sum())
        team_log[dupe_mask].to_csv(PROCESSED_DIR / "team_log_duplicates.csv", index=False)
    team_log = team_log.drop_duplicates(subset=["team", "gameweek"], keep="first")

    log.info("Adding team rolling (opponent strength) features...")
    team_log = add_team_rolling_features(team_log)

    # attach the UPCOMING OPPONENT's rolling form to each player row —
    # this is what lets the model learn "how leaky is this opponent"
    opponent_cols = [c for c in team_log.columns if c.startswith("team_roll") or c.startswith("team_season")]
    opponent_form = team_log[["team", "gameweek"] + opponent_cols].rename(
        columns={c: f"opp_{c.replace('team_roll', 'roll').replace('team_season', 'season')}" for c in opponent_cols}
    )
    opponent_form = opponent_form.rename(columns={"team": "opponent_team"})

    df = df.merge(opponent_form, on=["opponent_team", "gameweek"], how="left")

    # attach the player's OWN team's rolling/season strength too — a player
    # on a high-scoring team gets more chances overall regardless of their
    # personal quality, and this was previously missing: only the opponent's
    # side of this exact same data was ever surfaced onto player rows.
    #
    # Deliberately joined on (fixture_id, was_home) rather than the player's
    # own 'team' column — that column reflects each player's CURRENT club
    # applied to every historical row (the same stale-team issue fixed
    # earlier for the FBref join), so joining on it here would silently
    # reintroduce it for anyone who transferred mid-season. fixture_id +
    # was_home identifies the correct historical side regardless.
    own_team_form = team_log[["fixture_id", "was_home"] + opponent_cols].rename(
        columns={c: f"own_{c.replace('team_roll', 'roll').replace('team_season', 'season')}" for c in opponent_cols}
    )
    df = df.merge(own_team_form, on=["fixture_id", "was_home"], how="left")

    # HARD GUARD: feature engineering must never change row count — it only
    # adds columns to existing rows. If this fires, something upstream is
    # fanning out via a duplicate-key merge (this exact bug happened once
    # already with the team-based match log — don't let it happen silently
    # again).
    if len(df) != n_input_rows:
        raise AssertionError(
            f"Row count changed during feature engineering: {n_input_rows} -> {len(df)}. "
            f"This means a merge somewhere is fanning out on a duplicate key. "
            f"Do not trust this output — investigate before proceeding."
        )

    output_path = PROCESSED_DIR / "model_ready_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path}")
    print(f"Shape: {df.shape}")

    print("\n=== New feature columns added ===")
    new_cols = [c for c in df.columns if c.startswith("roll") or c.startswith("opp_") or c.startswith("own_") or c.startswith("season_") or c == "days_since_last_game"]
    print(new_cols)

    # sanity check: for a player's first-ever gameweek, rolling features
    # should be NaN (no prior games exist yet) — confirms no leakage
    first_games = df.sort_values(["player_id", "gameweek"]).groupby("player_id").head(1)
    n_first_game_with_data = first_games["roll3_points"].notna().sum()
    print(f"\nSanity check — player's FIRST gameweek should have NaN rolling features "
          f"(no prior games exist): {n_first_game_with_data} / {len(first_games)} unexpectedly have data "
          f"(should be 0)")


if __name__ == "__main__":
    main()