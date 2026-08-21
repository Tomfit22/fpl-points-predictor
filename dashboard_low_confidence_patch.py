import shutil
from pathlib import Path

TARGET = Path("build_dashboard.py")
content = TARGET.read_text(encoding="utf-8")
backup = TARGET.with_suffix(".py.lowconf_dash_bak")
shutil.copy(TARGET, backup)
print(f"Backed up to {backup}")

changes = []

old_text = '''  function formatCell(row, col) {
    const val = row[col.key];
    if (col.type === 'text') return row[col.key];'''
new_text = '''  function formatCell(row, col) {
    const val = row[col.key];
    if (col.type === 'text') {
      // low-confidence marker: cold-start player (new signing, no real
      // history) or a fixture involving a promoted team's league-average
      // fallback stats — a real, known reason to trust this specific
      // prediction less, made visible instead of only a console message
      if (col.key === 'player_name' && row.is_low_confidence) {
        return row[col.key] + ' <span class="low-confidence-marker" title="Lower confidence: limited real history (new signing or involves a newly-promoted team)">\\u26A0</span>';
      }
      return row[col.key];
    }'''

already_added = 'low-confidence-marker' in content
if old_text in content and not already_added:
    content = content.replace(old_text, new_text)
    changes.append("1. Added low-confidence marker to player name cells")
elif already_added:
    changes.append("1. Already applied — skipped")
else:
    print("*** Change 1 FAILED: formatCell text-type block not found as expected ***")

old_css = "  .player-cell { font-weight: 500; }"
new_css = '''  .player-cell { font-weight: 500; }
  .low-confidence-marker { color: #E8A33D; font-size: 12px; cursor: help; opacity: 0.85; }'''

already_styled = '.low-confidence-marker {' in content
if old_css in content and not already_styled:
    content = content.replace(old_css, new_css)
    changes.append("2. Added low-confidence marker CSS styling")
elif already_styled:
    changes.append("2. Already applied — skipped")
else:
    print("*** Change 2 FAILED: player-cell CSS rule not found as expected ***")

TARGET.write_text(content, encoding="utf-8")
print("\n=== Summary ===")
for c in changes:
    print(f"  {c}")
