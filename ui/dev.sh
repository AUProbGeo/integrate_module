#!/usr/bin/env bash
# Dev mode for the INTEGRATE web UI:
#   ./ui/dev.sh          — backend (reload) on :8000 + Vite dev server on :5173
#   ./ui/dev.sh build    — build the frontend for production
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$1" = "build" ]; then
  cd "$REPO_ROOT/ui/frontend" && npm install && npm run build
  exit 0
fi

if [ -n "$1" ] && [ -d "$1" ]; then
  export INTEGRATE_WORKSPACE="$(cd "$1" && pwd)"
  shift
fi

# Default workspace: the caller's working directory (before cd to repo root).
export INTEGRATE_WORKSPACE="${INTEGRATE_WORKSPACE:-$PWD}"

cd "$REPO_ROOT"

"$REPO_ROOT/.venv/bin/python" -m ui.backend.main --reload &
BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
cd ui/frontend && npm install && npm run dev
