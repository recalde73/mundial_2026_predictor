#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python scripts/update_world_cup_live.py "$@"
