#!/usr/bin/env bash
# Local verification pipeline: lint, format, types, dependency security, tests.
# Run before considering any change done: ./scripts/check.sh
# Auto-fix what's fixable (ruff lint --fix, ruff format): ./scripts/check.sh --fix
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FIX=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX=true
fi

run() {
    echo "==> $*"
    "$@"
    echo
}

if $FIX; then
    run uv run ruff check --fix .
    run uv run ruff format .
else
    run uv run ruff check .
    run uv run ruff format --check .
fi

run uv run mypy

run uv run pip-audit

run uv run pytest

echo "All checks passed."
