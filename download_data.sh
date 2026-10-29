#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

.venv/bin/kaggle competitions download -c global-wheat-detection -p data
unzip -q -o data/global-wheat-detection.zip -d data
rm data/global-wheat-detection.zip
