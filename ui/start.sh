#!/usr/bin/env bash
# Start the INTEGRATE web UI (production mode: API + built frontend).
#
#   ./ui/start.sh [--port 8000] [DATA_DIR]
#
# DATA_DIR: directory containing the project .h5 files (default: cwd).
# First run: build the frontend once with  ./ui/dev.sh build  (or:
#   cd ui/frontend && npm install && npm run build)
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "$1" ] && [ -d "$1" ]; then
  export INTEGRATE_WORKSPACE="$(cd "$1" && pwd)"
  shift
fi
# Default workspace: the caller's working directory (before cd to repo root).
export INTEGRATE_WORKSPACE="${INTEGRATE_WORKSPACE:-$PWD}"
cd "$REPO_ROOT"
if [ ! -d ui/frontend/dist ]; then
  echo "Frontend not built. Run: cd ui/frontend && npm install && npm run build" >&2
  exit 1
fi
exec "$REPO_ROOT/.venv/bin/python" -m ui.backend.main "$@"
