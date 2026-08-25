#!/usr/bin/env bash
set -euo pipefail

COMPETITION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. 自动寻找 Conda 初始化脚本路径
if [ -z "${CONDA_SCRIPT:-}" ]; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SCRIPT="$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SCRIPT="/usr/local/miniconda3/etc/profile.d/conda.sh"
  else
    # 默认回退路径
    CONDA_SCRIPT="/home/huangyixuan/miniconda3/etc/profile.d/conda.sh"
  fi
fi

# 2. 自动寻找 Voltron Conda 虚拟环境
if [ -z "${VOLTRON_ENV:-}" ]; then
  if [ -d "/mnt/data/huangyixuan/conda_envs/voltron" ]; then
    VOLTRON_ENV="/mnt/data/huangyixuan/conda_envs/voltron"
  else
    # 尝试直接激活命名环境
    VOLTRON_ENV="voltron"
  fi
fi

# 3. 设置默认工业指令（钳子放入工具箱第三格）
INSTRUCTION="${1:-现在请把钳子收纳至料箱的第3格，完成后报告状态}"
CONFIG_FILE="${CONFIG:-$COMPETITION_ROOT/voltron/configs/compact_industrial_pliers_to_toolbox_cell3_i00.json}"

echo "=========================================================="
echo "VisioMind 工业闭环仿真启动器（紧凑工位优化版）"
echo "Conda 脚本: $CONDA_SCRIPT"
echo "激活环境: $VOLTRON_ENV"
echo "配置模板: $CONFIG_FILE"
echo "输入指令: $INSTRUCTION"
echo "=========================================================="

source "$CONDA_SCRIPT"
conda activate "$VOLTRON_ENV"

# 运行工业场景的端到端闭环主进程
exec python "$COMPETITION_ROOT/run_instruction_demo.py" \
  "$INSTRUCTION" \
  --config "$CONFIG_FILE" \
  --grounding "$COMPETITION_ROOT/configs/scene_grounding_industrial.json"

