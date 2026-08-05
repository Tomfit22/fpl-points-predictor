#!/usr/bin/env bash
#
# Consolidates your 5 real weekly DATA422/DATA201 assignment repos
# (confirmed: week_3, week_4, week_5, week_6, week_8 — no week_1/2/7
# exist) into one repo, one folder per week, under your personal
# Weekly_Uni_assignments repo.
#
# Uses `gh repo clone` rather than a plain git URL — these are PRIVATE
# repos under the DATA422-DATA201-25S2 organization, and gh handles
# authentication via your already-logged-in CLI session automatically.
#
# Nothing is deleted from GitHub Classroom during this step — the
# original 5 repos stay completely untouched until you manually
# archive/delete them afterward, once you've confirmed everything
# came across correctly.

set -euo pipefail

ORG="DATA422-DATA201-25S2"
GITHUB_USER="Tomfit22"
CONSOLIDATED_REPO_DIR="$HOME/Weekly_Uni_assignments"

# confirmed real repo names — kept as their actual week numbers
# (3, 4, 5, 6, 8) rather than relabeled 1-5, so the context stays clear
REPOS=(
    "week_3_65774262"
    "week_4_65774262"
    "week_5_65774262"
    "week_6_65774262"
    "week_8_65774262"
)

if [ ! -d "$CONSOLIDATED_REPO_DIR" ]; then
    echo "Cloning the consolidated repo..."
    gh repo clone "$GITHUB_USER/Weekly_Uni_assignments" "$CONSOLIDATED_REPO_DIR"
fi
cd "$CONSOLIDATED_REPO_DIR"

for repo in "${REPOS[@]}"; do
    # extract the week number (e.g. "week_3_65774262" -> "3")
    week_num=$(echo "$repo" | sed -E 's/week_([0-9]+)_.*/\1/')
    dest="week-$week_num"

    echo ""
    echo "=== $dest: $repo ==="
    tmp_dir="/tmp/consolidate_$repo"
    rm -rf "$tmp_dir"
    gh repo clone "$ORG/$repo" "$tmp_dir"
    rm -rf "$tmp_dir/.git"

    mkdir -p "$dest"
    cp -r "$tmp_dir/." "$dest/"
    rm -rf "$tmp_dir"

    echo "  Copied $repo -> $dest/"
done

echo ""
echo "All 5 repos copied. Review the result, then commit and push:"
echo "  cd $CONSOLIDATED_REPO_DIR"
echo "  git add ."
echo "  git commit -m 'Consolidate weekly DATA422/DATA201 assignments into one repo'"
echo "  git push"
echo ""
echo "Only AFTER confirming everything looks right on GitHub, you can archive/delete"
echo "the old repos from the DATA422-DATA201-25S2 organization (these are Classroom"
echo "repos, not your personal ones — deleting may require instructor/org permissions,"
echo "so this might not even be something you're able to do yourself, and that's fine"
echo "to leave as-is if so):"
for repo in "${REPOS[@]}"; do
    echo "  gh repo delete $ORG/$repo --yes"
done