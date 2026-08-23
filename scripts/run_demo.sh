#!/usr/bin/env bash
set -euo pipefail

COMPETITION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SCRIPT="${CONDA_SCRIPT:-/home/huangyixuan/miniconda3/etc/profile.d/conda.sh}"
VOLTRON_ENV="${VOLTRON_ENV:-/mnt/data/huangyixuan/conda_envs/voltron}"
INSTRUCTION="${1:-把半个苹果放进包装箱}"

source "$CONDA_SCRIPT"
conda activate "$VOLTRON_ENV"
exec python "$COMPETITION_ROOT/run_instruction_demo.py" "$INSTRUCTION"
