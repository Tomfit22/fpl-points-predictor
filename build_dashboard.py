"""
FPL Points Predictor — Dashboard Generator
================================================
Reads data/processed/live_predictions.csv and generates a single,
self-contained HTML dashboard — no server, no build step, just open
the file in any browser. Re-run this any time predictions are
regenerated to refresh the dashboard.

Rebuilt to avoid a giant Python f-string with hundreds of escaped
{{ }} braces for the embedded CSS/JS — that pattern is fragile to
hand-copy and can confuse some editors' syntax highlighting even when
technically valid. This version keeps the HTML/CSS/JS as a plain
string with simple __PLACEHOLDER__ substitution instead.

Run:
    python build_dashboard.py
"""

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = Path("dashboard.html")

POSITION_COLORS = {
    "GK": "#F0B429",
    "DEF": "#3D8BFF",
    "MID": "#2FBF71",
    "FWD": "#FF5C5C",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FPL Predictor — Gameweek __GAMEWEEK__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0D1321;
    --surface: #161F32;
    --surface-2: #1D2941;
    --chalk: #EDF1F7;
    --fog: #7C8AA3;
    --gold: #F0B429;
    --green: #2FBF71;
    --border: #26314A;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--chalk);
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 32px 20px 80px; }

  .topbar { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 28px; flex-wrap: wrap; gap: 8px; }
  .brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 20px; letter-spacing: 0.02em; }
  .brand span { color: var(--gold); }
  .subtitle { color: var(--fog); font-size: 14px; }

  .hero-card {
    display: flex;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 16px;
  }
  .hero-bar { width: 6px; align-self: stretch; }
  .hero-content { padding: 22px 24px; flex: 1; min-width: 0; }
  .hero-label { font-size: 11px; letter-spacing: 0.12em; color: var(--fog); font-weight: 600; }
  .hero-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(24px, 4vw, 36px);
    margin: 4px 0 10px;
    font-weight: 700;
  }
  .hero-meta { display: flex; align-items: center; gap: 10px; }
  .hero-team { color: var(--fog); font-size: 14px; }
  .hero-score {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: clamp(36px, 6vw, 56px);
    color: var(--gold);
    padding: 0 32px;
  }

  .chip {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 3px 9px;
    border-radius: 100px;
    border: 1px solid;
  }
  .chip.small { font-size: 10px; padding: 2px 7px; }

  .strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 32px; }
  .strip-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
  }
  .strip-name { font-weight: 600; font-size: 14px; margin-top: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .strip-team { color: var(--fog); font-size: 12px; margin-bottom: 8px; }
  .strip-score { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 20px; }

  .controls { display: flex; gap: 10px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }
  .tabs { display: flex; gap: 6px; }
  .tab {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--fog);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
  }
  .tab.active { color: var(--bg); background: var(--gold); border-color: var(--gold); }
  .tab:focus-visible, .search:focus-visible, th button:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }

  .search {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--chalk);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    padding: 8px 12px;
    border-radius: 8px;
    flex: 1;
    min-width: 160px;
  }
  .search::placeholder { color: var(--fog); }

  .team-select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--chalk);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
  }

  .table-scroll { overflow-x: auto; border-radius: 12px; }
  table { width: 100%; min-width: 480px; border-collapse: collapse; background: var(--surface); }
  thead th {
    text-align: left;
    font-size: 11px;
    letter-spacing: 0.06em;
    color: var(--fog);
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
  }
  th button {
    background: none; border: none; color: inherit; font: inherit;
    cursor: pointer; padding: 0; text-transform: uppercase; letter-spacing: 0.06em;
  }
  tbody td { padding: 11px 14px; border-bottom: 1px solid var(--border); font-size: 14px; }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: var(--surface-2); }
  .num { font-family: 'IBM Plex Mono', monospace; text-align: right; }
  .pts { font-weight: 600; color: var(--gold); }
  .player-cell { font-weight: 500; }
  .team-cell { color: var(--fog); font-size: 13px; }

  .empty { text-align: center; color: var(--fog); padding: 40px; font-size: 14px; }

  @media (max-width: 640px) {
    .strip { grid-template-columns: repeat(2, 1fr); }
    .hero-card { flex-direction: column; align-items: stretch; }
    .hero-bar { width: 100%; height: 6px; }
    .hero-score { padding: 0 24px 20px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <div class="brand">FPL <span>PREDICTOR</span></div>
    <div class="subtitle">Gameweek __GAMEWEEK__ &middot; __PLAYER_COUNT__ players</div>
  </div>

  __HERO_CARD__
  <div class="strip">__STRIP_CARDS__</div>

  <div class="controls">
    <div class="tabs" id="posTabs">
      <button class="tab active" data-pos="ALL">All</button>
      <button class="tab" data-pos="GK">GK</button>
      <button class="tab" data-pos="DEF">DEF</button>
      <button class="tab" data-pos="MID">MID</button>
      <button class="tab" data-pos="FWD">FWD</button>
    </div>
    <select class="team-select" id="teamFilter">
      <option value="ALL">All teams</option>
    </select>
    <input class="search" id="search" type="text" placeholder="Search player or team&hellip;">
  </div>

  <div class="table-scroll">
  <table>
    <thead>
      <tr id="tableHeadRow"></tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
  </div>
  <div class="empty" id="emptyState" style="display:none">No players match this filter.</div>

</div>

<script>
  const DATA = __DATA_JSON__;
  const POSITION_COLORS = __COLORS_JSON__;

  // Column sets per position — a goalkeeper's relevant stats (clean sheets,
  // saves, penalty saves) are meaningless for a forward, and defensive
  // contribution/clean sheet don't apply the same way across positions, so
  // the table itself adapts rather than showing one fixed set of columns
  // with lots of blank/irrelevant cells.
  const COLUMN_SETS = {
    ALL: [
      { key: 'player_name', label: 'Player', type: 'text' },
      { key: 'team', label: 'Team', type: 'team' },
      { key: 'opponent_display', label: 'Opponent', type: 'opponent' },
      { key: 'position', label: 'Pos', type: 'chip' },
      { key: 'value', label: 'Price', type: 'price' },
      { key: 'ownership_pct', label: 'Owned %', type: 'ownership_pct' },
      { key: 'predicted_points', label: 'Points', type: 'pts' },
      { key: 'sim_range', label: 'Range', type: 'range' },
      { key: 'next_fixtures', label: 'Next 5', type: 'fixtures' }
    ],
    GK: [
      { key: 'player_name', label: 'Player', type: 'text' },
      { key: 'team', label: 'Team', type: 'team' },
      { key: 'opponent_display', label: 'Opponent', type: 'opponent' },
      { key: 'pred_p_clean_sheet', label: 'CS%', type: 'pct' },
      { key: 'pred_saves', label: 'Saves', type: 'num2' },
      { key: 'pred_pens_saved', label: 'Pen Saves', type: 'num2' },
      { key: 'pred_cards', label: 'Yellow%', type: 'pct' },
      { key: 'pred_red_cards', label: 'Red%', type: 'pct' },
      { key: 'value', label: 'Price', type: 'price' },
      { key: 'ownership_pct', label: 'Owned %', type: 'ownership_pct' },
      { key: 'predicted_points', label: 'Points', type: 'pts' },
      { key: 'sim_range', label: 'Range', type: 'range' },
      { key: 'next_fixtures', label: 'Next 5', type: 'fixtures' }
    ],
    DEF: [
      { key: 'player_name', label: 'Player', type: 'text' },
      { key: 'team', label: 'Team', type: 'team' },
      { key: 'opponent_display', label: 'Opponent', type: 'opponent' },
      { key: 'pred_goals', label: 'Goals', type: 'num2' },
      { key: 'pred_assists', label: 'Assists', type: 'num2' },
      { key: 'pred_p_dc_hit', label: 'DC%', type: 'pct' },
      { key: 'pred_p_clean_sheet', label: 'CS%', type: 'pct' },
      { key: 'pred_cards', label: 'Yellow%', type: 'pct' },
      { key: 'pred_red_cards', label: 'Red%', type: 'pct' },
      { key: 'value', label: 'Price', type: 'price' },
      { key: 'ownership_pct', label: 'Owned %', type: 'ownership_pct' },
      { key: 'predicted_points', label: 'Points', type: 'pts' },
      { key: 'sim_range', label: 'Range', type: 'range' },
      { key: 'next_fixtures', label: 'Next 5', type: 'fixtures' }
    ],
    MID: [
      { key: 'player_name', label: 'Player', type: 'text' },
      { key: 'team', label: 'Team', type: 'team' },
      { key: 'opponent_display', label: 'Opponent', type: 'opponent' },
      { key: 'pred_goals', label: 'Goals', type: 'num2' },
      { key: 'pred_assists', label: 'Assists', type: 'num2' },
      { key: 'pred_p_dc_hit', label: 'DC%', type: 'pct' },
      { key: 'pred_p_clean_sheet', label: 'CS%', type: 'pct' },
      { key: 'pred_cards', label: 'Yellow%', type: 'pct' },
      { key: 'pred_red_cards', label: 'Red%', type: 'pct' },
      { key: 'value', label: 'Price', type: 'price' },
      { key: 'ownership_pct', label: 'Owned %', type: 'ownership_pct' },
      { key: 'predicted_points', label: 'Points', type: 'pts' },
      { key: 'sim_range', label: 'Range', type: 'range' },
      { key: 'next_fixtures', label: 'Next 5', type: 'fixtures' }
    ],
    // no CS% or DC% for forwards — clean sheets are worth 0 points for FWD,
    // and defensive contribution is negligible enough at this position
    // (agreed earlier — too few forwards ever hit the threshold to be a
    // meaningful column) that showing it just adds noise
    FWD: [
      { key: 'player_name', label: 'Player', type: 'text' },
      { key: 'team', label: 'Team', type: 'team' },
      { key: 'opponent_display', label: 'Opponent', type: 'opponent' },
      { key: 'pred_goals', label: 'Goals', type: 'num2' },
      { key: 'pred_assists', label: 'Assists', type: 'num2' },
      { key: 'pred_cards', label: 'Yellow%', type: 'pct' },
      { key: 'pred_red_cards', label: 'Red%', type: 'pct' },
      { key: 'value', label: 'Price', type: 'price' },
      { key: 'ownership_pct', label: 'Owned %', type: 'ownership_pct' },
      { key: 'predicted_points', label: 'Points', type: 'pts' },
      { key: 'sim_range', label: 'Range', type: 'range' },
      { key: 'next_fixtures', label: 'Next 5', type: 'fixtures' }
    ]
  };

  let state = { pos: 'ALL', team: 'ALL', query: '', sortKey: 'predicted_points', sortDir: -1 };

  function formatCell(row, col) {
    const val = row[col.key];
    if (col.type === 'text') return row[col.key];
    if (col.type === 'team') return row[col.key];
    if (col.type === 'chip') {
      var color = POSITION_COLORS[row.position] || '#7C8AA3';
      return '<span class="chip small" style="background:' + color + '22;color:' + color + ';border-color:' + color + '55">' + row.position + '</span>';
    }
    // Range combines TWO separate fields (sim_floor, sim_ceiling) from the
    // Monte Carlo simulation — a wide range flags a boom-or-bust player
    // (differential potential), a narrow range flags a reliable floor pick
    if (col.type === 'range') {
      if (row.sim_floor == null || row.sim_ceiling == null) return '\u2014';
      return row.sim_floor.toFixed(1) + '\u2013' + row.sim_ceiling.toFixed(1);
    }
    // Opponent combines opponent_short (FPL's official 3-letter code)
    // with was_home_int into e.g. "CHE (H)" or "MUN (A)"
    if (col.type === 'opponent') {
      const code = row.opponent_short != null ? row.opponent_short : row.opponent_team;
      if (code == null) return '\u2014';
      const venue = row.was_home_int === 1 ? 'H' : 'A';
      return code + ' (' + venue + ')';
    }
    // Next 5 fixtures ticker — small colored chips per fixture, colored
    // by FPL's own 1-5 difficulty rating (green=easy, grey=medium,
    // red=hard), same convention as the real FPL site's fixture ticker
    if (col.type === 'fixtures') {
      const fixtures = row.next_fixtures;
      if (!fixtures || fixtures.length === 0) return '\u2014';
      const difficultyColor = (d) => {
        if (d == null) return '#7C8AA3';
        if (d <= 2) return '#2ECC71';
        if (d === 3) return '#95A5A6';
        return '#E74C3C';
      };
      return fixtures.map(function(fx) {
        var code = fx.opponent_short != null ? fx.opponent_short : '?';
        var venue = fx.was_home_int === 1 ? 'H' : 'A';
        var color = difficultyColor(fx.difficulty);
        return '<span class="chip small" style="background:' + color + '22;color:' + color +
               ';border-color:' + color + '55;margin-right:3px;">' + code + ' ' + venue + '</span>';
      }).join('');
    }
    if (col.type === 'pct') return (val != null) ? (val * 100).toFixed(0) + '%' : '\\u2014';
    if (col.type === 'num2') return (val != null) ? val.toFixed(2) : '\\u2014';
    // price is stored in tenths of a million (FPL convention, e.g. 55 -> £5.5m)
    if (col.type === 'price') return (val != null) ? '\\u00A3' + (val / 10).toFixed(1) + 'm' : '\\u2014';
    // ownership % is now a genuine estimate (see estimate_ownership.py),
    // using FPL's fixed squad composition rules (2 GK/5 DEF/5 MID/3 FWD)
    // to back out the total manager count from the raw 'selected' counts
    // we already have — not a true published FPL figure, but a real,
    // cross-validated estimate rather than just a raw count
    if (col.type === 'ownership_pct') return (val != null) ? val.toFixed(1) + '%' : '\\u2014';
    if (col.type === 'pts') return val.toFixed(2);
    return val;
  }

  function renderHeader() {
    var cols = COLUMN_SETS[state.pos];
    var headRow = document.getElementById('tableHeadRow');
    headRow.innerHTML = cols.map(function(c) {
      return '<th><button data-sort="' + c.key + '">' + c.label + '</button></th>';
    }).join('');

    headRow.querySelectorAll('th button').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var key = btn.dataset.sort;
        if (state.sortKey === key) {
          state.sortDir *= -1;
        } else {
          state.sortKey = key;
          state.sortDir = (key === 'player_name' || key === 'team' || key === 'position') ? 1 : -1;
        }
        render();
      });
    });
  }

  function render() {
    var cols = COLUMN_SETS[state.pos];
    let rows = DATA.filter(function(r) {
      return (state.pos === 'ALL' || r.position === state.pos) && (state.team === 'ALL' || r.team === state.team);
    });
    if (state.query) {
      var q = state.query.toLowerCase();
      rows = rows.filter(function(r) {
        return r.player_name.toLowerCase().indexOf(q) !== -1 || r.team.toLowerCase().indexOf(q) !== -1;
      });
    }
    rows.sort(function(a, b) {
      var av = a[state.sortKey], bv = b[state.sortKey];
      if (typeof av === 'string') return state.sortDir * av.localeCompare(bv);
      return state.sortDir * ((av || 0) - (bv || 0));
    });

    var body = document.getElementById('tableBody');
    var empty = document.getElementById('emptyState');
    if (rows.length === 0) {
      body.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';

    body.innerHTML = rows.map(function(r) {
      var cells = cols.map(function(c) {
        var cls = 'td-' + c.type;
        if (c.type === 'text') cls += ' player-cell';
        if (c.type === 'team') cls += ' team-cell';
        if (c.type === 'pct' || c.type === 'num2' || c.type === 'pts') cls += ' num';
        if (c.type === 'pts') cls += ' pts';
        return '<td class="' + cls + '">' + formatCell(r, c) + '</td>';
      }).join('');
      return '<tr>' + cells + '</tr>';
    }).join('');
  }

  document.getElementById('posTabs').addEventListener('click', function(e) {
    var btn = e.target.closest('.tab');
    if (!btn) return;
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    btn.classList.add('active');
    state.pos = btn.dataset.pos;
    state.sortKey = 'predicted_points';
    state.sortDir = -1;
    renderHeader();
    render();
  });

  document.getElementById('search').addEventListener('input', function(e) {
    state.query = e.target.value;
    render();
  });

  (function populateTeamFilter() {
    var teams = Array.from(new Set(DATA.map(function(r) { return r.team; }))).sort();
    var select = document.getElementById('teamFilter');
    teams.forEach(function(t) {
      var opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      select.appendChild(opt);
    });
    select.addEventListener('change', function(e) {
      state.team = e.target.value;
      render();
    });
  })();

  renderHeader();
  render();
</script>
</body>
</html>
"""

HERO_CARD_TEMPLATE = """
        <div class="hero-card">
          <div class="hero-bar" style="background:__COLOR__"></div>
          <div class="hero-content">
            <span class="hero-label">TOP PREDICTED &mdash; GAMEWEEK __GAMEWEEK__</span>
            <h1 class="hero-name">__NAME__</h1>
            <div class="hero-meta">
              <span class="chip" style="background:__COLOR__22;color:__COLOR__;border-color:__COLOR__55">__POSITION__</span>
              <span class="hero-team">__TEAM__</span>
            </div>
          </div>
          <div class="hero-score">__SCORE__</div>
        </div>"""

STRIP_CARD_TEMPLATE = """
        <div class="strip-card">
          <span class="chip small" style="background:__COLOR__22;color:__COLOR__;border-color:__COLOR__55">__POSITION__</span>
          <div class="strip-name">__NAME__</div>
          <div class="strip-team">__TEAM__</div>
          <div class="strip-score">__SCORE__</div>
        </div>"""


def build_next_fixtures_lookup(full_df: pd.DataFrame, n: int = 5) -> dict:
    """For each player, their next N fixtures (starting with the
    immediate upcoming one) as a list of (opponent_short, was_home_int,
    difficulty) tuples — built from the FULL multi-gameweek data before
    it gets filtered down to a single gameweek for the main table."""
    lookup = {}
    if "gameweek" not in full_df.columns:
        return lookup

    cols = [c for c in ["opponent_short", "was_home_int", "difficulty"] if c in full_df.columns]
    if not cols:
        return lookup

    for player_id, group in full_df.sort_values("gameweek").groupby("player_id"):
        rows = group[cols].head(n).to_dict("records")
        lookup[player_id] = rows
    return lookup


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "live_predictions.csv")

    next_fixtures_lookup = build_next_fixtures_lookup(df)

    # extract_fixtures.py now returns the FULL season's fixture list
    # (confirmed: 380 fixtures, all 38 gameweeks), not just the next
    # gameweek — meaning live_predictions.csv has one row per player
    # PER GAMEWEEK for the whole rest of the season. Without filtering,
    # the same players appear repeatedly (once per gameweek) in the
    # "top predicted" list, which is what broke the dashboard. Filter
    # down to the single EARLIEST upcoming gameweek, since that's the
    # one that's actually relevant right now.
    if "gameweek" in df.columns and df["gameweek"].nunique() > 1:
        target_gw = df["gameweek"].min()
        print(f"live_predictions.csv covers {df['gameweek'].nunique()} gameweeks "
              f"({df['gameweek'].min()}-{df['gameweek'].max()}) — filtering to gameweek "
              f"{target_gw} only, since that's the next upcoming one.")
        df = df[df["gameweek"] == target_gw]

    df = df.copy()
    df["next_fixtures"] = df["player_id"].map(next_fixtures_lookup)

    return df.sort_values("predicted_points", ascending=False).reset_index(drop=True)


def build_hero_card(player: dict, gameweek) -> str:
    color = POSITION_COLORS.get(player["position"], "#7C8AA3")
    return (HERO_CARD_TEMPLATE
            .replace("__COLOR__", color)
            .replace("__GAMEWEEK__", str(gameweek))
            .replace("__NAME__", str(player["player_name"]))
            .replace("__POSITION__", str(player["position"]))
            .replace("__TEAM__", str(player["team"]))
            .replace("__SCORE__", "{:.2f}".format(player["predicted_points"])))


def build_strip_card(player: dict) -> str:
    color = POSITION_COLORS.get(player["position"], "#7C8AA3")
    return (STRIP_CARD_TEMPLATE
            .replace("__COLOR__", color)
            .replace("__NAME__", str(player["player_name"]))
            .replace("__POSITION__", str(player["position"]))
            .replace("__TEAM__", str(player["team"]))
            .replace("__SCORE__", "{:.2f}".format(player["predicted_points"])))


def build_html(df: pd.DataFrame) -> str:
    records = df.round(3).to_dict(orient="records")
    gameweek = int(df["gameweek"].iloc[0]) if len(df) else "?"
    top5 = df.head(5).to_dict(orient="records")

    hero_html = build_hero_card(top5[0], gameweek) if top5 else ""
    strip_html = "".join(build_strip_card(p) for p in top5[1:5])

    html = HTML_TEMPLATE
    html = html.replace("__GAMEWEEK__", str(gameweek))
    html = html.replace("__PLAYER_COUNT__", str(len(df)))
    html = html.replace("__HERO_CARD__", hero_html)
    html = html.replace("__STRIP_CARDS__", strip_html)
    html = html.replace("__DATA_JSON__", json.dumps(records))
    html = html.replace("__COLORS_JSON__", json.dumps(POSITION_COLORS))
    return html


def main():
    df = load_predictions()
    print("Loaded {} predictions for gameweek {}".format(len(df), df["gameweek"].iloc[0] if len(df) else "?"))

    html = build_html(df)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print("Saved dashboard -> {}".format(OUTPUT_PATH.resolve()))
    print("Open this file directly in any browser — no server needed.")


if __name__ == "__main__":
    main()