#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
[ -f data/train.csv ] || ./download_data.sh
.venv/bin/python -m pytest -q
