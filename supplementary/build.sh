#!/usr/bin/env bash
# Build the anonymized supplementary.zip for NeurIPS 2026 submission.
#
# Usage:  bash supplementary/build.sh
# Output: ./supplementary.zip (in repo root)
#
# Layout produced inside the ZIP:
#   code/
#     README.md            -- install + reproduction instructions
#     LICENSE              -- MIT (anonymous)
#     requirements.txt     -- pinned dependency versions
#     pyproject.toml       -- pip-installable package definition
#     src/scptr/           -- the library (preprocessing, kinetics, deep, plotting)
#     tests/               -- pytest suite
#     scripts/             -- numbered analysis scripts grouped by paper claim
#       aim1_benchmarking/    half-life, miRNA enrichment, robustness
#       aim2_hidden_states/   PT-state discovery + invisibility scores
#       aim3_pt_velocity/     PT velocity, temporal precedence
#       aim4_cancer/          RBP network on neuroblastoma
#       deep/                 DeepPTR ablations (numbered 01-35)
#       _common.py            shared utilities
#     notebooks/           -- per-figure reproduction notebooks
#     data/README.md       -- dataset access (Pooch + manual download)
#
# Author-identifying strings are scrubbed; the script aborts if any are found.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUP="$ROOT/supplementary"

# In-repo build dir so paths resolve cleanly on Windows native Python.
BUILD="$ROOT/.supplementary-build"
STAGE="$BUILD/code"
rm -rf "$BUILD"
mkdir -p "$STAGE/scripts" "$STAGE/notebooks" "$STAGE/data"

echo "Staging into $STAGE"

# 1. Library and tests, copied verbatim from the live tree.
cp -r "$ROOT/src"            "$STAGE/"
cp -r "$ROOT/tests"          "$STAGE/"
cp    "$ROOT/pyproject.toml" "$STAGE/"

# 2. Analysis scripts: keep the topic-organised subdirs, drop scratch run_*.py
#    files at analyses/ top level (those are author-development churn).
cp -r "$ROOT/analyses/aim1_benchmarking"  "$STAGE/scripts/"
cp -r "$ROOT/analyses/aim2_hidden_states" "$STAGE/scripts/"
cp -r "$ROOT/analyses/aim3_pt_velocity"   "$STAGE/scripts/"
cp -r "$ROOT/analyses/aim4_cancer"        "$STAGE/scripts/"
cp -r "$ROOT/analyses/deep"               "$STAGE/scripts/"
cp    "$ROOT/analyses/_common.py"         "$STAGE/scripts/"
cp    "$ROOT/analyses/download_eclip.py"  "$STAGE/scripts/"

# 3. Anonymised overrides and the reproduction notebooks.
cp    "$SUP/code/README.md"     "$STAGE/"
cp    "$SUP/code/LICENSE"       "$STAGE/"
cp    "$SUP/code/requirements.txt" "$STAGE/"
cp -r "$SUP/code/repro/."       "$STAGE/notebooks/"
cp    "$SUP/code/data/README.md" "$STAGE/data/"

# 4. Strip caches and editor cruft.
find "$STAGE" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type f \( -name '*.pyc' -o -name '.DS_Store' -o -name 'Thumbs.db' \) -delete 2>/dev/null || true

# 5. Anonymisation sanity check.
if grep -rln -E "Bryan Cheng|Austin Jin|bryanc5864|ahanchijin" "$STAGE" >/dev/null; then
  echo "ERROR: identifying strings found in staged supplementary:" >&2
  grep -rln -E "Bryan Cheng|Austin Jin|bryanc5864|ahanchijin" "$STAGE" >&2
  exit 1
fi

# 6. Reject manuscript-like artefacts that should not ship with code.
if find "$STAGE" -type f \( -name '*.docx' -o -name 'manuscript*' -o -name 'paper*.pdf' -o -name 'draft*' \) | grep -q .; then
  echo "ERROR: manuscript-like files found in staged supplementary:" >&2
  find "$STAGE" -type f \( -name '*.docx' -o -name 'manuscript*' -o -name 'paper*.pdf' -o -name 'draft*' \) >&2
  exit 1
fi

# 7. Zip it up.
rm -f "$ROOT/supplementary.zip"
( cd "$BUILD" && python -c "import shutil; shutil.make_archive('supplementary', 'zip', root_dir='.', base_dir='code')" )
mv "$BUILD/supplementary.zip" "$ROOT/supplementary.zip"
rm -rf "$BUILD"

echo "Built: $ROOT/supplementary.zip ($(du -h "$ROOT/supplementary.zip" | cut -f1))"
echo "Files: $(unzip -l "$ROOT/supplementary.zip" | tail -1 | awk '{print $2}')"
