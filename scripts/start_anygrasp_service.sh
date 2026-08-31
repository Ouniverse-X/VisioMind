#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANYGRASP_PYTHON="${ANYGRASP_PYTHON:-/mnt/data/huangyixuan/conda_envs/anygrasp/bin/python}"
ANYGRASP_SDK_ROOT="${ANYGRASP_SDK_ROOT:-/mnt/data/huangyixuan/isaac/anygrasp_sdk}"
ANYGRASP_HOST="${ANYGRASP_HOST:-127.0.0.1}"
ANYGRASP_PORT="${ANYGRASP_PORT:-18090}"
ANYGRASP_MAX_WIDTH="${ANYGRASP_MAX_WIDTH:-0.1}"
ANYGRASP_GRIPPER_HEIGHT="${ANYGRASP_GRIPPER_HEIGHT:-0.03}"
ANYGRASP_TOP_DOWN="${ANYGRASP_TOP_DOWN:-1}"

ARGS=(
  "$REPO_ROOT/visiomind/action/integrations/manipulation/anygrasp/server.py"
  --sdk-root "$ANYGRASP_SDK_ROOT"
  --host "$ANYGRASP_HOST"
  --port "$ANYGRASP_PORT"
  --max-gripper-width "$ANYGRASP_MAX_WIDTH"
  --gripper-height "$ANYGRASP_GRIPPER_HEIGHT"
)

if [[ "$ANYGRASP_TOP_DOWN" == "1" ]]; then
  ARGS+=(--top-down-grasp)
fi

exec "$ANYGRASP_PYTHON" "${ARGS[@]}"
