#!/usr/bin/env bash
# Build the anonymized supplementary.zip for NeurIPS 2026 submission.
#
# Usage: bash supplementary/build.sh
# Output: ./supplementary.zip (in repo root)
#
# Assembles a temporary directory containing:
#   * src/, tests/, pyproject.toml   — copied verbatim from the live repo
#   * supplementary/code/{README, LICENSE, repro/}  — anonymized overrides
#   * analyses/                       — copied verbatim from the live repo
# then zips it. ~/.cache/__pycache__ etc. are stripped.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Use an in-repo build dir so paths resolve cleanly on Windows native Python.
BUILD="$ROOT/.supplementary-build"
STAGE="$BUILD/code"
rm -rf "$BUILD"
mkdir -p "$STAGE"

echo "Staging into $STAGE"
cp -r "$ROOT/src"          "$STAGE/"
cp -r "$ROOT/tests"        "$STAGE/"
cp -r "$ROOT/analyses"     "$STAGE/"
cp    "$ROOT/pyproject.toml" "$STAGE/"

# Anonymized overrides win over anything copied from live tree.
cp    "$ROOT/supplementary/code/README.md" "$STAGE/"
cp    "$ROOT/supplementary/code/LICENSE"   "$STAGE/"
cp -r "$ROOT/supplementary/code/repro"     "$STAGE/"

# Strip caches and editor cruft.
find "$STAGE" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true

# Quick anonymization sanity check.
if grep -rln -E "Bryan Cheng|Austin Jin|bryanc5864|ahanchijin" "$STAGE" >/dev/null; then
  echo "ERROR: identifying strings found in staged supplementary:" >&2
  grep -rln -E "Bryan Cheng|Austin Jin|bryanc5864|ahanchijin" "$STAGE" >&2
  exit 1
fi

rm -f "$ROOT/supplementary.zip"
( cd "$BUILD" && python -c "import shutil; shutil.make_archive('supplementary', 'zip', root_dir='.', base_dir='code')" )
mv "$BUILD/supplementary.zip" "$ROOT/supplementary.zip"
rm -rf "$BUILD"
echo "Built: $ROOT/supplementary.zip ($(du -h "$ROOT/supplementary.zip" | cut -f1))"
