#!/usr/bin/env bash
# Regenerate integrate_getting_started.ipynb from integrate_getting_started.py.
# --update keeps existing cell outputs and ids in the notebook.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f integrate_getting_started.ipynb ]; then
    jupytext --update --to ipynb integrate_getting_started.py
else
    jupytext --to ipynb integrate_getting_started.py
fi
