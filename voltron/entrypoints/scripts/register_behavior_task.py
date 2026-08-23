"""Register additional GR00T BEHAVIOR tasks for Voltron closed-loop runs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_INDICES_MARKER = "TASK_NAMES_TO_INDICES = {"
_INSTRUCTIONS_MARKER = "TASK_NAMES_TO_INSTRUCTIONS = {k: k.replace"
_REGISTER_FUNC_MARKER = "def register_behavior_envs():"


@dataclass(frozen=True)
class BehaviorTaskRegistration:
    task_name: str
    task_index: int | None
    scene_model: str
    robot_start_position: list[float]
    robot_start_orientation: list[float]
    behavior_env_path: Path
    available_tasks_path: Path
    dry_run: bool = False
    backup: bool = False


def infer_behavior_dir() -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    return workspace_root / "isaac_gr00t_learn" / "gr00t" / "eval" / "sim" / "BEHAVIOR"


def normalize_task_name(value: str) -> str:
    task_name = value.strip()
    if not task_name:
        raise ValueError("task name must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9_]+", task_name):
        raise ValueError(f"task name must contain only letters, digits, and underscores: {task_name!r}")
    return task_name


def parse_vector(values: list[str], *, length: int, field_name: str) -> list[float]:
    if len(values) != length:
        raise ValueError(f"{field_name} requires exactly {length} values")
    return [float(value) for value in values]


def next_task_index(behavior_env_text: str) -> int:
    indices = [int(match) for match in re.findall(r'"[A-Za-z0-9_]+"\s*:\s*(\d+)\s*,', behavior_env_text)]
    if not indices:
        raise ValueError("could not find existing TASK_NAMES_TO_INDICES entries")
    return max(indices) + 1


def update_behavior_env_text(text: str, *, task_name: str, task_index: int | None) -> str:
    task_name = normalize_task_name(task_name)
    resolved_index = next_task_index(text) if task_index is None else int(task_index)
    updated = _ensure_task_index(text, task_name=task_name, task_index=resolved_index)
    updated = _ensure_gym_registration(updated, task_name=task_name)
    return updated


def update_available_tasks_yaml_text(
    text: str,
    *,
    task_name: str,
    scene_model: str,
    robot_start_position: list[float],
    robot_start_orientation: list[float],
) -> str:
    task_name = normalize_task_name(task_name)
    if re.search(rf"(?m)^{re.escape(task_name)}:\s*$", text):
        return text

    block = _available_task_block(
        task_name=task_name,
        scene_model=scene_model,
        robot_start_position=robot_start_position,
        robot_start_orientation=robot_start_orientation,
    )
    prefix = text.rstrip()
    return f"{prefix}\n{block}" if prefix else block


def register_behavior_task(registration: BehaviorTaskRegistration) -> dict[str, Any]:
    behavior_env_path = registration.behavior_env_path.expanduser().resolve()
    available_tasks_path = registration.available_tasks_path.expanduser().resolve()

    behavior_env_text = behavior_env_path.read_text(encoding="utf-8")
    available_tasks_text = (
        available_tasks_path.read_text(encoding="utf-8") if available_tasks_path.exists() else ""
    )

    updated_behavior_env = update_behavior_env_text(
        behavior_env_text,
        task_name=registration.task_name,
        task_index=registration.task_index,
    )
    updated_available_tasks = update_available_tasks_yaml_text(
        available_tasks_text,
        task_name=registration.task_name,
        scene_model=registration.scene_model,
        robot_start_position=registration.robot_start_position,
        robot_start_orientation=registration.robot_start_orientation,
    )

    behavior_env_changed = updated_behavior_env != behavior_env_text
    available_tasks_changed = updated_available_tasks != available_tasks_text

    if not registration.dry_run:
        if registration.backup:
            if behavior_env_changed:
                _write_backup(behavior_env_path, behavior_env_text)
            if available_tasks_changed:
                _write_backup(available_tasks_path, available_tasks_text)
        if behavior_env_changed:
            behavior_env_path.write_text(updated_behavior_env, encoding="utf-8")
        if available_tasks_changed:
            available_tasks_path.parent.mkdir(parents=True, exist_ok=True)
            available_tasks_path.write_text(updated_available_tasks, encoding="utf-8")

    return {
        "task_name": normalize_task_name(registration.task_name),
        "task_index": registration.task_index
        if registration.task_index is not None
        else next_task_index(behavior_env_text),
        "behavior_env_path": str(behavior_env_path),
        "available_tasks_path": str(available_tasks_path),
        "behavior_env_updated": behavior_env_changed,
        "available_tasks_updated": available_tasks_changed,
        "dry_run": registration.dry_run,
    }


def _ensure_task_index(text: str, *, task_name: str, task_index: int) -> str:
    existing = re.search(rf'"{re.escape(task_name)}"\s*:\s*(\d+)\s*,', text)
    if existing:
        existing_index = int(existing.group(1))
        if existing_index != task_index:
            raise ValueError(
                f"{task_name!r} already has index {existing_index}, not requested index {task_index}"
            )
        return text

    marker_index = text.find(_INSTRUCTIONS_MARKER)
    if marker_index < 0:
        raise ValueError("could not find TASK_NAMES_TO_INSTRUCTIONS marker")
    closing_index = text.rfind("}", 0, marker_index)
    if closing_index < 0 or _INDICES_MARKER not in text[:closing_index]:
        raise ValueError("could not find TASK_NAMES_TO_INDICES closing brace")

    insertion = f'    "{task_name}": {task_index},\n'
    return text[:closing_index] + insertion + text[closing_index:]


def _ensure_gym_registration(text: str, *, task_name: str) -> str:
    env_id = f"sim_behavior_r1_pro/{task_name}"
    if env_id in text:
        return text
    function_index = text.find(_REGISTER_FUNC_MARKER)
    if function_index < 0:
        raise ValueError("could not find register_behavior_envs function")
    block = _gym_registration_block(task_name)
    return f"{text.rstrip()}\n\n{block}"


def _gym_registration_block(task_name: str) -> str:
    return f'''    register(
        id="sim_behavior_r1_pro/{task_name}",
        entry_point="gr00t.eval.sim.BEHAVIOR.behavior_env:BEHAVIORGr00tEnv",
        kwargs={{
            "task_name": "{task_name}",
        }},
    )
'''


def _available_task_block(
    *,
    task_name: str,
    scene_model: str,
    robot_start_position: list[float],
    robot_start_orientation: list[float],
) -> str:
    orientation_lines = "\n".join(f"    - {value}" for value in robot_start_orientation)
    position_lines = "\n".join(f"    - {value}" for value in robot_start_position)
    return f'''{task_name}:
  0:
    robot_start_orientation:
{orientation_lines}
    robot_start_position:
{position_lines}
    scene_model: {scene_model}
'''


def _write_backup(path: Path, content: str) -> None:
    backup_path = path.with_suffix(f"{path.suffix}.bak")
    backup_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    behavior_dir = infer_behavior_dir()
    parser = argparse.ArgumentParser(
        description="Register a GR00T BEHAVIOR task so gymnasium can make sim_behavior_r1_pro/<task>."
    )
    parser.add_argument("--task-name", required=True, help="Task name, e.g. packing_bags_or_suitcase.")
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="TASK_NAMES_TO_INDICES value. Defaults to max existing index + 1.",
    )
    parser.add_argument("--scene-model", required=True, help="Scene model for available_tasks.yaml.")
    parser.add_argument(
        "--robot-start-position",
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Robot start position stored in available_tasks.yaml.",
    )
    parser.add_argument(
        "--robot-start-orientation",
        nargs=4,
        metavar=("X", "Y", "Z", "W"),
        required=True,
        help="Robot start orientation quaternion stored in available_tasks.yaml.",
    )
    parser.add_argument(
        "--behavior-env-py",
        type=Path,
        default=behavior_dir / "behavior_env.py",
        help="Path to GR00T behavior_env.py.",
    )
    parser.add_argument(
        "--available-tasks-yaml",
        type=Path,
        default=behavior_dir / "available_tasks.yaml",
        help="Path to GR00T available_tasks.yaml.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing files.")
    parser.add_argument("--backup", action="store_true", help="Write .bak files before changing existing files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registration = BehaviorTaskRegistration(
        task_name=normalize_task_name(args.task_name),
        task_index=args.task_index,
        scene_model=args.scene_model,
        robot_start_position=parse_vector(
            args.robot_start_position,
            length=3,
            field_name="--robot-start-position",
        ),
        robot_start_orientation=parse_vector(
            args.robot_start_orientation,
            length=4,
            field_name="--robot-start-orientation",
        ),
        behavior_env_path=args.behavior_env_py,
        available_tasks_path=args.available_tasks_yaml,
        dry_run=args.dry_run,
        backup=args.backup,
    )
    result = register_behavior_task(registration)
    for key, value in result.items():
        print(f"{key}={value}")
    if not args.dry_run:
        print("note=Scene templates, TRO states, test_instances.csv, and episodes.jsonl must still exist for real simulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
