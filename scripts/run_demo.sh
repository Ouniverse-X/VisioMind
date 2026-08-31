#!/usr/bin/env bash
set -euo pipefail

COMPETITION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SCRIPT="${CONDA_SCRIPT:-$HOME/miniconda3/etc/profile.d/conda.sh}"
VISIOMIND_ENV="${VISIOMIND_ENV:-visiomind}"
INSTRUCTION="${1:-现在请把钳子收纳至料箱的第3格，完成后报告状态}"
CONFIG_FILE="${CONFIG:-$COMPETITION_ROOT/visiomind/action/configs/compact_industrial_pliers_to_toolbox_cell3_i00.json}"

source "$CONDA_SCRIPT"
conda activate "$VISIOMIND_ENV"
exec python "$COMPETITION_ROOT/run_visiomind.py" \
  "$INSTRUCTION" \
  --config "$CONFIG_FILE" \
  --grounding "$COMPETITION_ROOT/configs/scene_grounding_industrial.json"
