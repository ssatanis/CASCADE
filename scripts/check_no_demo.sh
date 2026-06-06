#!/usr/bin/env bash
# Realness guard: fail if any product/runtime/validation path contains a
# data-fabrication marker. Tests and fixtures are exempt (seeded fixtures live
# there by design). Patterns are high-signal (actual fabrication), not bare
# substrings, so legitimate UI ("placeholder=" attrs), biology ("synthetic
# lethality", "synthesized gene"), control-sample regexes ("mock" transfection),
# and "Coming soon" labels do NOT trip it.
set -uo pipefail
cd "$(dirname "$0")/.."

# Product/runtime/validation roots (NOT tests/fixtures).
ROOTS=(cascade/src cascade/data cli/src frontend/src frontend/lib scripts src backend modal_app tools analysis-engine sdk worker)

# Directories/files never scanned.
EXCLUDES=(
  --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=.next
  --exclude-dir=dist --exclude-dir=build --exclude-dir=__pycache__ --exclude-dir=.git
  --exclude-dir=tests --exclude-dir=__tests__ --exclude-dir=fixtures --exclude-dir=coverage
  --exclude-dir=*.egg-info
  --exclude=*.json --exclude=check_no_demo.sh --exclude=realness_audit.py
  --exclude=check_realness_manifest.py
)

# High-signal data-fabrication patterns.
PATTERNS=(
  "resultsSource['\"]?[[:space:]]*[:=][[:space:]]*['\"]demo"   # demo results source value
  "generate(Mock|Demo|Synthetic|Fake|Dummy|Sample)[A-Za-z]*\(" # fabricated-data factories
  "mock(Data|Results|Response|Scores|Genes|Essentials)\b"      # mock data containers
  "(DEMO|MOCK|FAKE|DUMMY)_[A-Z]"                               # fabricated constants
  "(//|#)[[:space:]]*[Pp]laceholder"                           # placeholder-value comments
  "[Hh]ardcoded|hard-coded"                                    # admitted hardcoding
  "\blorem\b|\bfaker\b"                                        # lorem/faker
  "from[[:space:]].*\.synth[[:space:]]import|import[[:space:]].*\bcascade\.synth\b"  # quarantined synth
  "generate_cohort\(|SynthConfig\(|synthetic_screens"          # synth fixture API in product
)

found=0
for pat in "${PATTERNS[@]}"; do
  for root in "${ROOTS[@]}"; do
    [ -e "$root" ] || continue
    hits=$(grep -rnIE "${EXCLUDES[@]}" "$pat" "$root" 2>/dev/null || true)
    if [ -n "$hits" ]; then
      echo "BANNED PATTERN: /$pat/"
      echo "$hits"
      echo
      found=1
    fi
  done
done

if [ "$found" -ne 0 ]; then
  echo "❌ check_no_demo: fabricated-data markers found in product/validation paths."
  exit 1
fi
echo "✅ check_no_demo: no fabricated-data markers in product/validation paths."
