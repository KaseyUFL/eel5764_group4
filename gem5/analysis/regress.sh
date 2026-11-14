#!/usr/bin/env bash
# Regression check of the default model: rerun the representative points
# and diff their statistics against analysis/golden_stats.csv, exactly.
#
#   bash analysis/regress.sh            # what `make regress` runs
#
# Run this after every model change.  Details: analysis/regress.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
exec "$PYTHON" "$SYSTOLIC_ROOT/analysis/regress.py" "$@"
