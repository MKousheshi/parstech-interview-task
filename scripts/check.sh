#!/usr/bin/env bash
# Local verification pipeline: lint, format, types, tests (+ optional dependency security scan).
# Run before considering any change done: ./scripts/check.sh
# Auto-fix what's fixable (ruff lint --fix, ruff format): ./scripts/check.sh --fix
# Also run the pip-audit vulnerability scan — only needed when dependencies were added or changed:
#   ./scripts/check.sh --audit   (combinable with --fix)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FIX=false
AUDIT=false
for arg in "$@"; do
    case "$arg" in
        --fix) FIX=true ;;
        --audit) AUDIT=true ;;
        *) echo "unknown option: $arg (expected --fix and/or --audit)" >&2; exit 2 ;;
    esac
done

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

if $AUDIT; then
    run uv run pip-audit
fi

run uv run pytest

echo "All checks passed."
