"""
FPL Points Predictor — LIVE PREDICTIONS (fixture-driven)
================================================================
The missing piece between "validate against history" and "predict next
week": takes each player's CURRENT rolling form (their latest available
snapshot) plus an UPCOMING fixture (opponent, home/away) and runs it
through the same trained component models from build_prediction_pipeline.py
to produce a real forward-looking points prediction.

Models are refit on ALL available historical data here (not a
train/test split) — for real prediction, we want the best possible
model, not a held-out evaluation. build_prediction_pipeline.py remains
the place to check honest accuracy; this script is for generating
actual predictions once accuracy is already trusted.

STAND-IN MODE: data/raw/fixtures_upcoming.csv is currently empty (the
2025/26 season is finished; 2026/27 fixtures aren't loaded into FPL's
API yet, as of writing). Until they are, this script can run in
STAND_IN_MODE, using the last gameweek of the COMPLETED season as a
stand-in "upcoming" fixture list purely to test the connector's
mechanics end-to-end. Switch STAND_IN_MODE to False once real fixtures
are available — no other changes needed.

BLANK/DOUBLE GAMEWEEKS: handled naturally — a team with no fixture that
gameweek simply produces no rows (0 predicted points); a team with two
fixtures produces two rows per player, summed into one gameweek total.

Run:
    python build_live_predictions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

STAND_IN_MODE = False  # flipped off — expects data/raw/fixtures_upcoming.csv to have real rows now

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

GOALS_FEATURES = ["season_xG", "roll5_xG", "season_shots", "roll5_shots",
                   "opp_season_shots_for", "was_home_int", "is_primary_pen_taker"]
ASSISTS_FEATURES = ["season_xA", "roll5_xA", "season_key_passes", "roll5_key_passes",
                     "season_Crs", "opp_season_goals_conceded", "was_home_int"]
DC_FEATURES = ["season_def_contrib", "roll5_def_contrib", "season_TklW", "season_Int",
               "opp_season_shots_for", "was_home_int",
               # validated via analyze_defcontrib_by_position.py's out-of-sample
               # test: DEF Brier 0.164->0.146, MID Brier 0.101->0.088 — a real,
               # if modest, improvement, not assumed
               "opp_season_possession", "opp_roll5_possession",
               "own_season_ppda", "own_roll5_ppda"]
CS_FEATURES = ["own_season_goals_conceded", "own_roll5_goals_conceded",
               "own_season_xG_against", "opp_season_goals_scored",
               "opp_season_xG_for", "opp_season_shots_for", "was_home_int",
               # validated via build_clean_sheets_model.py's out-of-sample
               # test: Brier 0.1947->0.1917 — real but small
               "own_season_possession", "opp_season_possession",
               "own_season_ppda", "opp_season_ppda"]
SAVES_FEATURES = ["season_saves", "own_season_shots_against", "opp_season_shots_for", "was_home_int"]
MINUTES_FEATURES = ["roll5_minutes", "roll5_starts", "consecutive_starts", "days_since_last_game"]


def drop_zero_variance(df: pd.DataFrame, features: list) -> list:
    variances = df[features].var(numeric_only=True)
    return [f for f in features if variances.get(f, 1) > 0]


# =========================
# FIT ALL COMPONENTS ON FULL HISTORY
# =========================
def fit_minutes_model(df: pd.DataFrame, threshold: int):
    candidates = drop_zero_variance(df, [f for f in MINUTES_FEATURES if f in df.columns])

    # proactive correlation-pruning — same fix already validated for the
    # DC model earlier (CBI/Tackles/def_contrib). Confirmed on real data:
    # roll5_starts and roll5_minutes are highly correlated (if you
    # started, you almost certainly played close to 90 minutes), and
    # fitting both together produced backwards, uninterpretable
    # coefficients (roll5_starts came out NEGATIVE — "starting more of
    # your last 5 games makes you LESS likely to play"). Keeping
    # roll5_minutes (the more information-rich, continuous signal) and
    # dropping whichever correlates too strongly with it.
    features = []
    for f in candidates:
        too_similar = any(abs(df[f].corr(df[g])) > 0.85 for g in features)
        if not too_similar:
            features.append(f)

    X = sm.add_constant(df[features].fillna(0))
    y = (df["minutes"] >= threshold).astype(int)
    return sm.Logit(y, X).fit(disp=0), features


def fit_poisson_by_position(df: pd.DataFrame, target: str, candidate_features: list, positions: list):
    models = {}
    for position in positions:
        pos_df = df[df["position"] == position]
        features = drop_zero_variance(pos_df, [f for f in candidate_features if f in pos_df.columns])
        if len(pos_df) < 60 or not features:
            continue
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            models[position] = (sm.GLM(pos_df[target], X, family=sm.families.Poisson()).fit(), features)
        except Exception as e:
            print(f"  (skipping {target} model for {position}: {e})")
    return models


def fit_dc_models(df: pd.DataFrame):
    models = {}
    for position in ["DEF", "MID", "FWD"]:
        pos_df = df[df["position"] == position].copy()
        pos_df["hit"] = (pos_df["defensive_contribution"] >= DC_THRESHOLD[position]).astype(int)
        if pos_df["hit"].sum() < 20:
            continue
        features = drop_zero_variance(pos_df, [f for f in DC_FEATURES if f in pos_df.columns])
        pruned = []
        for f in features:
            too_similar = any(abs(pos_df[f].corr(pos_df[g])) > 0.98 for g in pruned)
            if not too_similar:
                pruned.append(f)
        features = pruned
        X = sm.add_constant(pos_df[features].fillna(0))
        try:
            model = sm.Logit(pos_df["hit"], X).fit(disp=0)
            if not model.mle_retvals.get("converged", True):
                raise np.linalg.LinAlgError("did not converge")
        except np.linalg.LinAlgError as e:
            print(f"  (standard MLE failed for {position} ({e}) — falling back to regularized fit)")
            model = sm.Logit(pos_df["hit"], X).fit_regularized(disp=0, alpha=0.1)
        models[position] = (model, features)
    return models


def fit_clean_sheet_model(df: pd.DataFrame):
    team_df = df.groupby(["team", "fixture_id"]).first().reset_index()
    features = drop_zero_variance(team_df, [f for f in CS_FEATURES if f in team_df.columns])
    X = sm.add_constant(team_df[features].fillna(0))
    return sm.GLM(team_df["goals_conceded"], X, family=sm.families.Poisson()).fit(), features


def fit_saves_model(df: pd.DataFrame):
    gk_df = df[df["position"] == "GK"]
    features = drop_zero_variance(gk_df, [f for f in SAVES_FEATURES if f in gk_df.columns])
    X = sm.add_constant(gk_df[features].fillna(0))
    return sm.GLM(gk_df["saves"], X, family=sm.families.Poisson()).fit(), features


# =========================
# BUILD "CURRENT STATE" SNAPSHOTS
# =========================
def get_latest_player_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's most recent row — their rolling features as of NOW,
    the starting point for predicting their NEXT match."""
    return df.sort_values("gameweek").groupby("player_id").tail(1).reset_index(drop=True)


COLD_START_THRESHOLD_GAMES = 3  # fewer than this many minutes>0 appearances = insufficient PL history
COLD_START_STAT_COLS = [
    "season_xG", "roll5_xG", "season_xA", "roll5_xA",
    "season_shots", "roll5_shots", "season_key_passes", "roll5_key_passes",
    "season_def_contrib", "roll5_def_contrib", "season_TklW", "season_Int",
    "season_saves", "season_clean_sheets", "roll5_clean_sheets",
]


def apply_cold_start_priors(player_snapshot: pd.DataFrame, full_history: pd.DataFrame) -> pd.DataFrame:
    """
    New signings from other leagues (or otherwise debuting players) have
    NO Premier League history, so their rolling features are NaN and
    downstream fillna(0) treats that as literally zero talent — a real,
    systematically wrong default that specifically underrates exactly
    the high-profile transfers people most want good predictions for.

    IMPORTANT DISTINCTION (found via real-data testing — a naive version
    of this that only checked games_played flagged 41% of the player
    pool, including long-tenured fringe/reserve players who simply don't
    get selected). Two different populations both show "<3 games":
      1. Genuine new signings — zero PL history doesn't reflect zero
         ability, a price-based prior is the right call.
      2. Long-tenured fringe/reserve players who just don't play — their
         low minutes IS real, informative signal, not missing data.
         Applying a position-average prior to them actively makes
         things WORSE, inflating someone who genuinely isn't a regular.

    Gated on PRICE as well as games played: a genuine notable signing is
    priced meaningfully above their position average (real transfer fees
    get reflected in FPL pricing); a genuine fringe player is typically
    priced at or near the position minimum. Only players clearing BOTH
    the low-games AND above-average-price bar get the prior — everyone
    else with genuinely low minutes keeps their real (low) prediction.

    NOT a rigorous cross-league statistical translation — there isn't a
    clean scientific conversion factor between leagues; real analysts
    disagree on this too. This is a pragmatic proxy using data we
    already have, not a precise translation.

    Only touches the INFERENCE-time snapshot used for live predictions
    — never the historical training data itself, which stays honest.
    """
    player_snapshot = player_snapshot.copy()

    games_played = full_history[full_history["minutes"] > 0].groupby("player_id").size()
    player_snapshot["_games_played"] = player_snapshot["player_id"].map(games_played).fillna(0)

    played_history = full_history[full_history["minutes"] > 0]
    stat_cols = [c for c in COLD_START_STAT_COLS if c in full_history.columns]
    position_avg_stats = played_history.groupby("position")[stat_cols].mean()
    position_avg_price = played_history.groupby("position")["value"].mean() if "value" in full_history.columns else None

    # price ratio computed for everyone up front, since it's now also the
    # gating condition, not just the scaling factor
    player_snapshot["_price_ratio"] = 1.0
    if position_avg_price is not None:
        for position in position_avg_price.index:
            mask = (player_snapshot["position"] == position) & player_snapshot["value"].notna() & (position_avg_price[position] > 0)
            player_snapshot.loc[mask, "_price_ratio"] = player_snapshot.loc[mask, "value"] / position_avg_price[position]

    # PRICE_GATE: require price to be genuinely above position average,
    # not just "not below" — a real signing typically costs more than a
    # typical squad player at that position, a fringe player typically doesn't
    PRICE_GATE = 1.15
    cold_start_mask = (player_snapshot["_games_played"] < COLD_START_THRESHOLD_GAMES) & (player_snapshot["_price_ratio"] >= PRICE_GATE)

    n_low_games_not_flagged = ((player_snapshot["_games_played"] < COLD_START_THRESHOLD_GAMES) & ~cold_start_mask).sum()
    if n_low_games_not_flagged > 0:
        print(f"\n{n_low_games_not_flagged} players have <{COLD_START_THRESHOLD_GAMES} games but are priced at "
              f"or below their position average — treated as genuine fringe/reserve players (real signal, "
              f"not missing data), NOT given a cold-start prior.")

    if not cold_start_mask.any():
        return player_snapshot.drop(columns=["_games_played", "_price_ratio"])

    affected = []
    for idx in player_snapshot[cold_start_mask].index:
        row = player_snapshot.loc[idx]
        position = row["position"]
        if position not in position_avg_stats.index:
            continue

        price_ratio = max(0.3, min(row["_price_ratio"], 3.0))  # clamp against extreme values from data issues
        for col in stat_cols:
            player_snapshot.at[idx, col] = position_avg_stats.at[position, col] * price_ratio

        affected.append((row["player_name"], row["team"], position, price_ratio))

    if affected:
        print(f"\nApplied cold-start price-based priors to {len(affected)} players with insufficient PL history "
              f"AND price notably above their position average (likely genuine new signings, not fringe players):")
        for name, team, pos, ratio in affected:
            print(f"  {name} ({team}, {pos}): price ratio {ratio:.2f}x position average")

    return player_snapshot.drop(columns=["_games_played", "_price_ratio"])


def normalize_name(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace — for robust name
    matching across data sources that spell the same player differently."""
    import unicodedata
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def reconcile_player_ids(df: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """
    FPL's player_id is NOT stable across the season boundary — confirmed
    empirically (Haaland, Palmer, Saka all showed ID mismatches despite
    not changing teams, likely because FPL regenerates the elements list
    each season and 3 relegated/3 promoted teams shift everyone's
    position in it). Left unfixed, every player-matching mechanism in
    this pipeline (cold-start logic, position-history safeguard, missing-
    player detection) would silently treat continuing players as unknown.

    This runs ONCE, early, remapping df's player_id column to match the
    CURRENT roster's IDs wherever a confident name match exists — so
    everything downstream just works without needing to know this
    problem exists at all.

    Matches on the SHORT player_name first (e.g. "Haaland"), not
    full_name — confirmed on real data that full_name can differ in
    formatting between two API pulls of the same real person (e.g. a
    middle name included in one pull, dropped in the other), silently
    breaking full-name matching, while FPL's short display name stays
    consistent. full_name + position are used only as tie-breakers when
    a short name matches MULTIPLE historical players (e.g. the two real
    "Onana"s). If still ambiguous after that, left UNCHANGED rather than
    guessed at — better to under-fix than silently merge two different
    people's history.
    """
    df = df.copy()
    roster = roster.copy()
    df["_norm_name"] = df["player_name"].apply(normalize_name)
    roster["_norm_name"] = roster["player_name"].apply(normalize_name)
    df["_norm_full"] = df["full_name"].apply(normalize_name) if "full_name" in df.columns else df["_norm_name"]
    roster["_norm_full"] = roster["full_name"].apply(normalize_name) if "full_name" in roster.columns else roster["_norm_name"]

    # a lookup of {id: normalized_name} — used to check whether a matching
    # ID ALSO matches by name, not just by raw number. Proven necessary:
    # a player's NEW id can coincidentally already belong to a completely
    # DIFFERENT real person's OLD historical row (confirmed on real data
    # — promoted-team IDs frequently collide numerically with unrelated
    # departed players). Checking ID alone silently treated these as
    # "already fine" and skipped reconciliation entirely.
    id_to_name = df.drop_duplicates("player_id").set_index("player_id")["_norm_name"].to_dict()

    name_groups = df.groupby("_norm_name")

    remap = {}
    new_id_owner_name = {}
    ambiguous = []
    reconciled_count = 0

    for _, row in roster.iterrows():
        # only skip if the SAME id ALSO has the SAME name in the old data —
        # true agreement, not just a coincidental number match
        if row["player_id"] in id_to_name and id_to_name[row["player_id"]] == row["_norm_name"]:
            continue

        if row["_norm_name"] not in name_groups.groups:
            continue  # no name match at all — genuinely new player, not this function's job

        candidates_df = name_groups.get_group(row["_norm_name"])
        candidate_ids = candidates_df["player_id"].unique().tolist()
        candidate_ids = [c for c in candidate_ids if c != row["player_id"]]
        if not candidate_ids:
            continue

        if len(candidate_ids) == 1:
            remap[candidate_ids[0]] = row["player_id"]
            new_id_owner_name[row["player_id"]] = row["_norm_name"]
            reconciled_count += 1
            continue

        # short name matched multiple historical players — try full_name as a tie-breaker
        full_match = candidates_df[candidates_df["_norm_full"] == row["_norm_full"]]["player_id"].unique().tolist()
        full_match = [c for c in full_match if c != row["player_id"]]
        if len(full_match) == 1:
            remap[full_match[0]] = row["player_id"]
            new_id_owner_name[row["player_id"]] = row["_norm_name"]
            reconciled_count += 1
            continue

        # try position as a second tie-breaker if full_name didn't resolve it
        if "position" in candidates_df.columns and "position" in row.index:
            pos_match = candidates_df[candidates_df["position"] == row["position"]]["player_id"].unique().tolist()
            pos_match = [c for c in pos_match if c != row["player_id"]]
            if len(pos_match) == 1:
                remap[pos_match[0]] = row["player_id"]
                new_id_owner_name[row["player_id"]] = row["_norm_name"]
                reconciled_count += 1
                continue

        ambiguous.append((row["player_name"], row["team"], candidate_ids))

    if reconciled_count > 0:
        print(f"\n{reconciled_count} players reconciled by name match — their historical data "
              f"(under a since-changed old ID) now correctly links to their current roster entry.")

    if ambiguous:
        print(f"\n*** {len(ambiguous)} name(s) matched MULTIPLE historical players even after "
              f"full-name/position tie-breaking — left UNCHANGED rather than guessed at (real "
              f"risk of merging two different people's history, same issue found earlier with "
              f"two different real 'Onana's). Review manually if these players' predictions "
              f"look off: ***")
        for name, team, candidates in ambiguous:
            print(f"  {name} ({team}): {len(candidates)} possible historical matches")

    # TWO-PHASE move, immune to chain collisions. A naive single-pass
    # remap breaks when a player's OLD id is ALSO some OTHER player's
    # NEW target — confirmed on real data (Gabriel's old id=5 was
    # simultaneously another player's reconciliation target, so a
    # single-pass collision check wrongly displaced Gabriel's own rows
    # as "unrelated blocking data" before his own remap could apply,
    # silently erasing 457 reconciliations' worth of edge cases).
    #
    # Phase 1: move every reconciled player's rows to a GUARANTEED-safe
    # negative temporary id first — completely out of the way of any
    # collision check, regardless of how old/new ids overlap.
    temp_id_for_old = {old_id: -(i + 1) for i, old_id in enumerate(remap.keys())}
    df["player_id"] = df["player_id"].map(lambda x: temp_id_for_old.get(x, x))

    # Phase 2: NOW check the real target ids for genuinely unrelated
    # leftover data (players who were never being reconciled at all) —
    # safe to displace, since anyone who WAS being reconciled has
    # already moved to a negative temp id and won't be caught here.
    displaced_count = 0
    next_placeholder_id = int(df["player_id"].max()) + 1000 if (df["player_id"] > 0).any() else 100000
    for new_id in remap.values():
        collision_mask = (df["player_id"] == new_id) & (df["_norm_name"] != new_id_owner_name.get(new_id, ""))
        if collision_mask.any():
            df.loc[collision_mask, "player_id"] = next_placeholder_id
            displaced_count += 1
            next_placeholder_id += 1

    if displaced_count > 0:
        print(f"  ({displaced_count} unrelated player(s)' old data relocated to placeholder IDs "
              f"to avoid colliding with the reconciled players above — likely players no longer "
              f"in the league.)")

    # Phase 3: move everyone from their safe temporary id to their real
    # final target — no collisions possible at this point, since phase 2
    # already cleared out anything that was in the way.
    old_id_for_temp = {v: k for k, v in temp_id_for_old.items()}
    df["player_id"] = df["player_id"].map(
        lambda x: remap[old_id_for_temp[x]] if x in old_id_for_temp else x
    )

    return df.drop(columns=["_norm_name", "_norm_full"])


def add_missing_roster_players(player_snapshot: pd.DataFrame, full_history: pd.DataFrame) -> pd.DataFrame:
    """Players who exist on the CURRENT FPL roster (e.g. a newly promoted
    team's squad) but have NEVER appeared in our historical dataset at
    all — different from the cold-start case, which handles players
    with SOME but insufficient history. These players don't even have a
    row to correct; they need one created from scratch.

    Uses current_roster_snapshot.csv (build_current_roster_snapshot.py)
    specifically because it reads live from FPL's current roster,
    independent of gameweek history — the same reason it was built to
    answer "what are the new positions/prices" during preseason.

    Adds bare rows with NaN stat columns; apply_cold_start_priors() then
    fills them in via the same price-based logic it already uses for
    low-history players — no duplicated logic, this just gives it
    something to act on.
    """
    roster_path = PROCESSED_DIR / "current_roster_snapshot.csv"
    if not roster_path.exists():
        print("  (current_roster_snapshot.csv not found — skipping missing-player check. "
              "Run build_current_roster_snapshot.py to enable this safeguard.)")
        return player_snapshot

    roster = pd.read_csv(roster_path)
    known_ids = set(player_snapshot["player_id"])
    missing = roster[~roster["player_id"].isin(known_ids)]

    if len(missing) == 0:
        return player_snapshot

    print(f"\n{len(missing)} players on the current roster have NO history in our dataset at all "
          f"(likely a newly promoted team's squad) — creating placeholder rows for the cold-start "
          f"price-based prior to fill in:")
    for _, row in missing.head(15).iterrows():
        print(f"  {row['player_name']} ({row['team']}, {row['position']})")
    if len(missing) > 15:
        print(f"  ... and {len(missing) - 15} more")

    new_rows = missing[["player_id", "player_name", "team", "position", "price"]].copy()
    new_rows = new_rows.rename(columns={"price": "value"})
    new_rows["minutes"] = 0
    new_rows["starts"] = 0
    new_rows["selected"] = 0  # unknown — will show as 0% ownership until real data exists
    # every rolling stat column starts as NaN, same as any player with
    # insufficient history — apply_cold_start_priors() fills these in
    for col in COLD_START_STAT_COLS + ["roll5_minutes", "roll5_starts", "pred_p_dc_hit"]:
        if col in player_snapshot.columns and col not in new_rows.columns:
            new_rows[col] = float("nan")

    combined = pd.concat([player_snapshot, new_rows], ignore_index=True)
    return combined


def get_latest_team_snapshot(df: pd.DataFrame) -> dict:
    """Each team's most recent own_ rolling stats — used both as this
    team's OWN features for their next match, and as the OPPONENT's
    features when they're the away/home side against someone else."""
    own_cols = [c for c in df.columns if c.startswith("own_")]
    latest = df.sort_values("gameweek").groupby("team", as_index=False)[["team"] + own_cols].tail(1)
    return {row["team"]: row[own_cols].to_dict() for _, row in latest.iterrows()}


# =========================
# BUILD PREDICTION ROWS FOR UPCOMING FIXTURES
# =========================
def get_league_average_team_stats(team_stats: dict) -> dict:
    """Neutral fallback for a team with zero PL history (newly promoted) —
    the average of every KNOWN team's own_ stats. Not a rigorous
    prediction of how a promoted team will actually perform, just an
    honest 'we genuinely don't know, use the league-wide baseline'
    default rather than silently dropping their fixtures entirely."""
    if not team_stats:
        return {}
    all_teams_df = pd.DataFrame(team_stats).T
    return all_teams_df.mean(numeric_only=True).to_dict()


def build_fixture_rows(player_snapshot: pd.DataFrame, team_stats: dict, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    league_avg_stats = get_league_average_team_stats(team_stats)
    teams_using_fallback = set()

    for _, fixture in fixtures.iterrows():
        for team, opponent, was_home in [
            (fixture["home_team"], fixture["away_team"], True),
            (fixture["away_team"], fixture["home_team"], False),
        ]:
            team_players = player_snapshot[player_snapshot["team"] == team].copy()
            if team_players.empty:
                continue

            # a team with zero PL history (newly promoted) gets the league
            # average instead of silently dropping this fixture — which
            # would otherwise ALSO wrongly drop the OTHER team's players
            opp_stats = team_stats.get(opponent)
            if opp_stats is None:
                opp_stats = league_avg_stats
                teams_using_fallback.add(opponent)
            if team not in team_stats:
                teams_using_fallback.add(team)

            team_players["opponent_team"] = opponent
            team_players["was_home_int"] = int(was_home)
            team_players["gameweek"] = fixture["gameweek"]
            team_players["fixture_id"] = fixture["fixture_id"]

            # opponent's current own_ stats become THIS row's opp_ fields
            for col, val in opp_stats.items():
                opp_col = col.replace("own_", "opp_", 1)
                team_players[opp_col] = val
            # if THIS team itself has no history, its own_ stats also need
            # the fallback (team_players won't have real own_ cols set
            # otherwise, since those come from the per-player snapshot,
            # which for a promoted team's players will also be missing)
            if team in teams_using_fallback:
                for col, val in league_avg_stats.items():
                    team_players[col] = val

            rows.append(team_players)

    if teams_using_fallback:
        print(f"\n*** {len(teams_using_fallback)} team(s) have zero PL history (newly promoted) — "
              f"using league-average stats as a fallback: {sorted(teams_using_fallback)}. "
              f"Treat predictions involving these teams as lower-confidence. ***")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# =========================
# PREDICT
# =========================
def predict_points(df: pd.DataFrame, components: dict, avg_bonus_when_played: float) -> pd.DataFrame:
    def predict_logit(model, features):
        X = sm.add_constant(df[features].fillna(0), has_constant="add")
        return pd.Series(model.predict(X), index=df.index)

    def predict_poisson_by_pos(models):
        preds = pd.Series(0.0, index=df.index)
        for position, (model, features) in models.items():
            mask = df["position"] == position
            if mask.sum() == 0:
                continue
            X = sm.add_constant(df.loc[mask, features].fillna(0), has_constant="add")
            preds.loc[mask] = model.predict(X)
        return preds

    df["pred_p_any_minutes"] = predict_logit(*components["any_minutes"])
    df["pred_p_60plus"] = predict_logit(*components["sixty_plus"])
    df["pred_goals"] = predict_poisson_by_pos(components["goals"])
    df["pred_assists"] = predict_poisson_by_pos(components["assists"])
    df["pred_p_dc_hit"] = predict_poisson_by_pos(components["dc"])  # same shape, logistic though

    cs_model, cs_features = components["clean_sheet"]
    X_cs = sm.add_constant(df[cs_features].fillna(0), has_constant="add")
    df["pred_p_clean_sheet"] = np.exp(-cs_model.predict(X_cs))

    saves_model, saves_features = components["saves"]
    save_mask = df["position"] == "GK"
    df["pred_saves"] = 0.0
    if save_mask.sum() > 0:
        X_saves = sm.add_constant(df.loc[save_mask, saves_features].fillna(0), has_constant="add")
        df.loc[save_mask, "pred_saves"] = saves_model.predict(X_saves)

    df["pred_cards"] = df["roll5_CrdY"].fillna(0) if "roll5_CrdY" in df.columns else 0
    df["pred_red_cards"] = df["roll5_CrdR"].fillna(0) if "roll5_CrdR" in df.columns else 0
    # penalty saves: too rare to model properly (agreed early in this project) —
    # just use each keeper's own historical rate directly, not a fitted model
    df["pred_pens_saved"] = df["season_pens_saved"].fillna(0) if "season_pens_saved" in df.columns else 0.0

    appearance_pts = df["pred_p_any_minutes"] * 1 + df["pred_p_60plus"] * 1
    goal_pts = df.apply(lambda r: r["pred_goals"] * GOAL_POINTS.get(r["position"], 0), axis=1) * df["pred_p_any_minutes"]
    assist_pts = df["pred_assists"] * 3 * df["pred_p_any_minutes"]
    dc_pts = df["pred_p_dc_hit"] * 2 * df["pred_p_any_minutes"]
    cs_pts = df.apply(
        lambda r: r["pred_p_clean_sheet"] * CLEAN_SHEET_POINTS.get(r["position"], 0) * r["pred_p_60plus"],
        axis=1
    )
    save_pts = (df["pred_saves"] / 3) * df["pred_p_any_minutes"]
    pen_save_pts = df["pred_pens_saved"] * 5 * df["pred_p_any_minutes"]
    card_pts = -(df["pred_cards"] * 1 + df["pred_red_cards"] * 3) * df["pred_p_any_minutes"]
    bonus_pts = avg_bonus_when_played * df["pred_p_60plus"]

    df["predicted_points"] = appearance_pts + goal_pts + assist_pts + dc_pts + cs_pts + save_pts + pen_save_pts + card_pts + bonus_pts

    # save the actual POINT contribution of each component, not just the
    # raw probability/rate — needed for anything that explains WHY a
    # prediction landed where it did, in terms a person actually cares about
    df["pts_appearance"] = appearance_pts
    df["pts_goals"] = goal_pts
    df["pts_assists"] = assist_pts
    df["pts_dc"] = dc_pts
    df["pts_clean_sheet"] = cs_pts
    df["pts_saves"] = save_pts
    df["pts_pen_saves"] = pen_save_pts
    df["pts_cards"] = card_pts
    df["pts_bonus"] = bonus_pts
    return df


def run_monte_carlo_simulation(df: pd.DataFrame, avg_bonus_when_played: float,
                                n_sims: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Runs the SAME points formula as predict_points() above, but draws a
    random outcome from each component's real distribution instead of
    using its expected value — repeated n_sims times per player. This
    gives a genuine empirical distribution of possible outcomes, not
    just one number, directly answering "how much variance/risk does
    this prediction actually carry."

    Honest limitation: bonus has no real per-player distribution to draw
    from (already proven bonus can't be reliably differentiated between
    players — see build_bonus_predictions.py). It's added as a FIXED
    amount conditional on playing 60+, matching the same expected value
    as the deterministic formula, but contributing zero simulated
    variance — better to be upfront about this than fake uncertainty we
    don't actually have.
    """
    rng = np.random.default_rng(random_seed)
    n_players = len(df)

    p_any = df["pred_p_any_minutes"].fillna(0).values
    p_60 = df["pred_p_60plus"].fillna(0).values
    p_60 = np.minimum(p_60, p_any)  # defensive clamp: can't play 60+ without playing at all

    # conditional P(60+ minutes | played at all) — needed so the two
    # draws below are logically consistent (never "played 60+" without
    # "played at all" also being true)
    p_60_given_any = np.divide(p_60, p_any, out=np.zeros_like(p_60), where=p_any > 0)
    p_60_given_any = np.clip(p_60_given_any, 0, 1)

    played_any = rng.random((n_players, n_sims)) < p_any[:, None]
    played_60 = played_any & (rng.random((n_players, n_sims)) < p_60_given_any[:, None])

    goals = rng.poisson(df["pred_goals"].fillna(0).values[:, None], size=(n_players, n_sims))
    assists = rng.poisson(df["pred_assists"].fillna(0).values[:, None], size=(n_players, n_sims))
    dc_hit = rng.random((n_players, n_sims)) < df["pred_p_dc_hit"].fillna(0).values[:, None]
    clean_sheet = rng.random((n_players, n_sims)) < df["pred_p_clean_sheet"].fillna(0).values[:, None]
    saves = rng.poisson(df["pred_saves"].fillna(0).values[:, None], size=(n_players, n_sims))
    pen_saves = rng.poisson(df["pred_pens_saved"].fillna(0).values[:, None], size=(n_players, n_sims))
    yellow = rng.random((n_players, n_sims)) < np.clip(df["pred_cards"].fillna(0).values[:, None], 0, 1)
    red = rng.random((n_players, n_sims)) < np.clip(df["pred_red_cards"].fillna(0).values[:, None], 0, 1)

    goal_points_arr = df["position"].map(GOAL_POINTS).fillna(0).values[:, None]
    cs_points_arr = df["position"].map(CLEAN_SHEET_POINTS).fillna(0).values[:, None]

    appearance_pts = played_any.astype(float) + played_60.astype(float)
    goal_pts = goals * goal_points_arr * played_any
    assist_pts = assists * 3 * played_any
    dc_pts = dc_hit * 2 * played_any
    cs_pts = clean_sheet * cs_points_arr * played_60
    save_pts = (saves / 3) * played_any
    pen_save_pts = pen_saves * 5 * played_any
    card_pts = -(yellow.astype(float) * 1 + red.astype(float) * 3) * played_any
    bonus_pts = avg_bonus_when_played * played_60.astype(float)

    total = (appearance_pts + goal_pts + assist_pts + dc_pts + cs_pts
             + save_pts + pen_save_pts + card_pts + bonus_pts)

    result = df.copy()
    result["sim_floor"] = np.percentile(total, 10, axis=1)
    result["sim_p25"] = np.percentile(total, 25, axis=1)
    result["sim_median"] = np.percentile(total, 50, axis=1)
    result["sim_p75"] = np.percentile(total, 75, axis=1)
    result["sim_ceiling"] = np.percentile(total, 90, axis=1)
    result["sim_std"] = total.std(axis=1)
    return result


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_ready_dataset.csv")
    df = df[df["roll5_minutes"].notna()]

    roster_path = PROCESSED_DIR / "current_roster_snapshot.csv"
    if roster_path.exists():
        print("Reconciling player IDs against the current roster (FPL's IDs are not "
              "stable across season boundaries — see build_live_predictions.py docs)...")
        roster_for_reconciliation = pd.read_csv(roster_path)
        df = reconcile_player_ids(df, roster_for_reconciliation)
    else:
        print("  (current_roster_snapshot.csv not found — skipping ID reconciliation. "
              "Run build_current_roster_snapshot.py first for this season-transition safeguard.)")

    print("Fitting all component models on full historical data...")
    components = {
        "any_minutes": fit_minutes_model(df, threshold=1),
        "sixty_plus": fit_minutes_model(df, threshold=60),
        "goals": fit_poisson_by_position(df, "goals", GOALS_FEATURES, ["DEF", "MID", "FWD"]),
        "assists": fit_poisson_by_position(df, "assists", ASSISTS_FEATURES, ["DEF", "MID", "FWD"]),
        "dc": fit_dc_models(df),
        "clean_sheet": fit_clean_sheet_model(df),
        "saves": fit_saves_model(df),
    }
    avg_bonus_when_played = df[df["minutes"] >= 60]["bonus"].mean() if "bonus" in df.columns else 0.3

    print("Building current player and team snapshots...")
    player_snapshot = get_latest_player_snapshot(df)

    # Filter to genuinely CURRENT roster members only. Reconciliation
    # redirects players who are STILL in the league to their correct
    # new ID, but it doesn't remove anyone — a player who's genuinely
    # left the league entirely (retired, transferred out, relegated
    # with their old team) never matches anything in the current
    # roster and just sits in the historical data under their old,
    # untouched ID forever. Left unfiltered, get_latest_player_snapshot
    # includes them anyway with stale, outdated team info — confirmed
    # on real data: 890 "current" players when the actual roster only
    # has 558. Anyone not genuinely on today's roster gets dropped here.
    if "roster_for_reconciliation" in locals():
        before_count = len(player_snapshot)
        current_roster_ids = set(roster_for_reconciliation["player_id"])
        player_snapshot = player_snapshot[player_snapshot["player_id"].isin(current_roster_ids)]
        dropped = before_count - len(player_snapshot)
        if dropped > 0:
            print(f"  Dropped {dropped} players no longer on any current team's roster "
                  f"(left the league, retired, or transferred out — their historical data "
                  f"was never a reconciliation target since nobody currently shares their name).")

        # Reconciliation fixes player_id but NOT team/position — a
        # player's historical rows still say whatever club they were at
        # LAST season, even after being correctly matched to their
        # current roster entry. Confirmed on real data: reconciled
        # players who transferred to a DIFFERENT club still in the
        # league (e.g. Liverpool players who moved elsewhere) kept
        # showing their old team. Overwrite with the roster's
        # authoritative current values for anyone genuinely reconciled.
        roster_lookup = roster_for_reconciliation.set_index("player_id")
        team_updates = 0
        for idx in player_snapshot.index:
            pid = player_snapshot.at[idx, "player_id"]
            if pid in roster_lookup.index:
                current_team = roster_lookup.at[pid, "team"]
                if player_snapshot.at[idx, "team"] != current_team:
                    player_snapshot.at[idx, "team"] = current_team
                    team_updates += 1
                if "position" in roster_lookup.columns:
                    player_snapshot.at[idx, "position"] = roster_lookup.at[pid, "position"]
        if team_updates > 0:
            print(f"  Corrected {team_updates} players' team field to their current club "
                  f"(reconciliation matches the right PERSON, but doesn't update team/position "
                  f"on its own — this fixes cases like a player showing their old club after "
                  f"transferring elsewhere within the league).")

    player_snapshot = add_missing_roster_players(player_snapshot, df)
    player_snapshot = apply_cold_start_priors(player_snapshot, df)
    team_stats = get_latest_team_snapshot(df)
    print(f"  {len(player_snapshot)} players, {len(team_stats)} teams")

    if STAND_IN_MODE:
        print("\n*** STAND_IN_MODE is ON — using the LAST completed gameweek's real fixtures "
              "as a stand-in for 'upcoming', purely to test this connector's mechanics. "
              "Set STAND_IN_MODE = False once data/raw/fixtures_upcoming.csv has real rows. ***\n")
        all_fixtures = pd.read_csv(RAW_DIR / "fixtures_all.csv")
        last_gw = all_fixtures["gameweek"].max()
        fixtures = all_fixtures[all_fixtures["gameweek"] == last_gw]
    else:
        fixtures = pd.read_csv(RAW_DIR / "fixtures_upcoming.csv")
        if fixtures.empty:
            print("data/raw/fixtures_upcoming.csv is empty — nothing to predict. "
                  "Re-run extract_fixtures.py once next season's fixtures are live.")
            return

    print(f"Generating prediction rows for {len(fixtures)} fixtures...")
    fixture_rows = build_fixture_rows(player_snapshot, team_stats, fixtures)
    if fixture_rows.empty:
        print("No prediction rows generated — check team name matching between "
              "the fixtures file and model_ready_dataset.csv.")
        return

    result = predict_points(fixture_rows, components, avg_bonus_when_played)

    print("Running Monte Carlo simulation (5,000 draws per player) for prediction ranges...")
    result = run_monte_carlo_simulation(result, avg_bonus_when_played)

    # sum across fixtures per player, per gameweek — handles double
    # gameweeks automatically (a team with 2 fixtures produces 2 rows,
    # which sum here into one gameweek total). Keeping the component
    # columns too, not just the final total, so downstream consumers
    # (like a dashboard) can show real detail, not just one number.
    # NOTE: for a genuine double gameweek, predicted_points/goals/assists/etc.
    # correctly SUM (points/counts across two matches are additive). The
    # probability columns (pred_p_clean_sheet, pred_p_any_minutes,
    # pred_p_60plus) technically aren't valid probabilities anymore once
    # summed this way (e.g. 0.8 + 0.8 = 1.6) — a minor, known imprecision,
    # acceptable since double gameweeks are rare and these are secondary
    # display columns, not the points calculation itself.
    component_cols = ["predicted_points", "pred_goals", "pred_assists", "pred_p_dc_hit",
                       "pred_p_clean_sheet", "pred_saves", "pred_pens_saved", "pred_cards", "pred_red_cards",
                       "pred_p_any_minutes", "pred_p_60plus",
                       "pts_appearance", "pts_goals", "pts_assists", "pts_dc", "pts_clean_sheet",
                       "pts_saves", "pts_pen_saves", "pts_cards", "pts_bonus",
                       "sim_floor", "sim_p25", "sim_median", "sim_p75", "sim_ceiling"]
    component_cols = [c for c in component_cols if c in result.columns]
    summed = result.groupby(["player_id", "player_name", "team", "position", "gameweek"], as_index=False)[
        component_cols
    ].sum()

    # Opponent + home/away — taken from the FIRST fixture per
    # (player, gameweek) group. For a genuine double gameweek this only
    # shows one of the two opponents, same documented simplification
    # already accepted above for the probability columns — rare enough
    # not to be worth the complexity of showing both.
    opponent_cols = [c for c in ["opponent_team", "was_home_int", "fixture_id"] if c in result.columns]
    if opponent_cols:
        opponent_info = result.groupby(["player_id", "gameweek"], as_index=False)[opponent_cols].first()
        summed = summed.merge(opponent_info, on=["player_id", "gameweek"], how="left")

        # FPL's official 3-letter team code, not a hand-typed guess —
        # falls back to the full name if a fixtures file with short
        # codes hasn't been generated yet (older extract_fixtures.py run)
        fixtures_path = Path("data/raw/fixtures_all.csv")
        if fixtures_path.exists():
            fixtures_ref = pd.read_csv(fixtures_path)
            short_name_map = {}
            if "home_team" in fixtures_ref.columns and "home_team_short" in fixtures_ref.columns:
                short_name_map.update(dict(zip(fixtures_ref["home_team"], fixtures_ref["home_team_short"])))
            if "away_team" in fixtures_ref.columns and "away_team_short" in fixtures_ref.columns:
                short_name_map.update(dict(zip(fixtures_ref["away_team"], fixtures_ref["away_team_short"])))
            summed["opponent_short"] = summed["opponent_team"].map(short_name_map).fillna(summed["opponent_team"])
        else:
            summed["opponent_short"] = summed["opponent_team"]

        # FPL's own 1-5 difficulty rating, from THIS player's team's
        # perspective (home_team_difficulty if they were home,
        # away_team_difficulty if away) — powers the fixture-ticker
        # coloring on the dashboard, same convention as the real FPL site
        if fixtures_path.exists():
            fixtures_ref["fixture_id"] = fixtures_ref["fixture_id"].astype(str)
            summed["fixture_id_str"] = summed["fixture_id"].astype(str) if "fixture_id" in summed.columns else None
            diff_lookup = fixtures_ref.set_index("fixture_id")[["home_team_difficulty", "away_team_difficulty"]]

            def get_own_difficulty(row):
                if "fixture_id_str" not in row or row["fixture_id_str"] not in diff_lookup.index:
                    return None
                d = diff_lookup.loc[row["fixture_id_str"]]
                return d["home_team_difficulty"] if row["was_home_int"] == 1 else d["away_team_difficulty"]

            if "fixture_id" in summed.columns:
                summed["difficulty"] = summed.apply(get_own_difficulty, axis=1)
                summed = summed.drop(columns=["fixture_id_str"])

    # price and ownership are STATIC per-player attributes, not per-fixture
    # contributions — merged in separately rather than summed (summing a
    # price across a double gameweek would be meaningless).
    static_cols = [c for c in ["value", "selected"] if c in player_snapshot.columns]
    if static_cols:
        summed = summed.merge(player_snapshot[["player_id"] + static_cols], on="player_id", how="left")

    # genuine ownership % via the squad-composition estimator — 'selected'
    # alone is just a raw manager count; this converts it into a real
    # percentage using FPL's fixed squad rules (2 GK/5 DEF/5 MID/3 FWD)
    if "selected" in summed.columns:
        try:
            from estimate_ownership import add_ownership_pct
            summed = add_ownership_pct(summed, gameweek=None)  # summed is already one snapshot
        except ImportError:
            print("  (estimate_ownership.py not found — skipping ownership % calculation, "
                  "raw 'selected' counts still available)")

    output_path = PROCESSED_DIR / "live_predictions.csv"
    summed.sort_values("predicted_points", ascending=False).to_csv(output_path, index=False)
    print(f"\nSaved live predictions -> {output_path} ({len(summed)} player-gameweek predictions)")

    print("\nTop 15 predicted scorers:")
    print(summed.sort_values("predicted_points", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()