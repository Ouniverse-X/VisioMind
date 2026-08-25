#!/usr/bin/env bash
set -eo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/huangyixuan/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/huangyixuan/conda_envs/voltron
set -u
cd "${repo_root}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR="${repo_root}/logs/matplotlib"

instruction="${1:-请把混杂工位上的钳子放进工具箱第三格}"
config_file="${CONFIG:-voltron/configs/plier_to_toolbox_cell3_industrial_i00.json}"

exec python run_instruction_demo.py "${instruction}" \
  --config "${config_file}" \
  --grounding configs/scene_grounding_industrial.json


