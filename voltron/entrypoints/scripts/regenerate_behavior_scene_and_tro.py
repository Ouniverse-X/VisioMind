#!/usr/bin/env python3
"""Regenerate a BEHAVIOR scene JSON and matching TRO from a Voltron config."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from omnigibson.utils.config_utils import TorchEncoder
from omnigibson.utils.asset_utils import get_dataset_path

from voltron.integrations.simulator.behavior.runtime_bridge import BehaviorRuntimeEnvironment
from voltron.config_loader import load_config_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a Voltron closed-loop JSON config.")
    parser.add_argument(
        "--output-scene-file",
        default=None,
        help="Path to write the regenerated full scene JSON. Defaults beside the source scene file.",
    )
    parser.add_argument(
        "--output-tro-file",
        default=None,
        help="Path to write the regenerated TRO JSON. Defaults beside the source scene file.",
    )
    parser.add_argument(
        "--seed-tro-mode",
        choices=("empty", "config"),
        default="empty",
        help="How to seed task-relevant object states before regeneration.",
    )
    return parser


def _unwrap_env(env: Any) -> Any:
    visited: set[int] = set()
    current = env
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "task") and hasattr(current, "scene"):
            return current
        next_env = getattr(current, "wrapped_env", None) or getattr(current, "env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return current


def _load_scene_metadata(scene_file: Path) -> dict[str, Any]:
    with scene_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _default_output_paths(scene_file: Path) -> tuple[Path, Path]:
    scene_stem = scene_file.stem
    scene_output = scene_file.with_name(f"{scene_stem}_consistent_runtime.json")
    tro_output = scene_file.with_name(f"{scene_stem}_consistent_runtime-tro_state.json")
    return scene_output, tro_output


def _build_env_kwargs(config: dict[str, Any], seed_tro_path: Path | None) -> dict[str, Any]:
    env_kwargs: dict[str, Any] = {
        "env_idx": int(config.get("env_idx", 0)),
        "total_n_envs": int(config.get("total_n_envs", 1)),
    }
    behavior_scene_file = config.get("behavior_scene_file")
    if behavior_scene_file:
        env_kwargs["scene_file"] = str(Path(behavior_scene_file).expanduser().resolve())
    if seed_tro_path is not None:
        env_kwargs["tro_state_file"] = str(seed_tro_path)
    elif config.get("behavior_tro_state_file"):
        env_kwargs["tro_state_file"] = str(Path(config["behavior_tro_state_file"]).expanduser().resolve())
    if config.get("behavior_task_instance_id") is not None:
        env_kwargs["task_instance_id"] = int(config["behavior_task_instance_id"])
    return env_kwargs


def _compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_expected_hash_source(entry: dict[str, Any]) -> Path | None:
    class_name = str(entry.get("class_name") or "").strip()
    args = entry.get("args")
    if not isinstance(args, dict):
        return None

    usd_path: Path | None = None
    if class_name == "DatasetObject" and args.get("category") and args.get("model"):
        usd_path = (
            Path(get_dataset_path("behavior-1k-assets"))
            / "objects"
            / str(args["category"])
            / str(args["model"])
            / "usd"
            / f"{args['model']}.usd"
        )
    elif args.get("usd_path"):
        usd_path = Path(str(args["usd_path"]))

    if usd_path is None:
        return None

    encrypted_path = usd_path.with_name(usd_path.name.replace(".usd", ".encrypted.usd"))
    if encrypted_path.exists():
        return encrypted_path
    if usd_path.exists():
        return usd_path
    return None


def _refresh_expected_file_hashes(scene_file: Path) -> int:
    payload = json.loads(scene_file.read_text(encoding="utf-8"))
    objects_info = payload.get("objects_info")
    if not isinstance(objects_info, dict):
        return 0
    init_info = objects_info.get("init_info")
    if not isinstance(init_info, dict):
        return 0

    updated = 0
    for entry in init_info.values():
        if not isinstance(entry, dict):
            continue
        args = entry.get("args")
        if not isinstance(args, dict):
            continue
        hash_source = _resolve_expected_hash_source(entry)
        if hash_source is None:
            continue
        args["expected_file_hash"] = _compute_md5(hash_source)
        updated += 1

    scene_file.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return updated


def _resolve_robot_poses(scene_metadata: dict[str, Any], base_env: Any) -> dict[str, Any]:
    task_metadata = scene_metadata.get("task")
    if isinstance(task_metadata, dict):
        robot_poses = task_metadata.get("robot_poses")
        if isinstance(robot_poses, dict) and robot_poses:
            return robot_poses

    robot = base_env.robots[0]
    robot_key = (
        getattr(robot, "model_name", None)
        or getattr(robot, "model", None)
        or getattr(robot, "name", None)
        or "robot"
    )
    position, orientation = robot.get_position_orientation()
    if hasattr(position, "tolist"):
        position = position.tolist()
    if hasattr(orientation, "tolist"):
        orientation = orientation.tolist()
    return {
        str(robot_key): [
            {
                "position": position,
                "orientation": orientation,
            }
        ]
    }


def _write_outputs(
    *,
    runtime_env: BehaviorRuntimeEnvironment,
    source_scene_file: Path,
    output_scene_file: Path,
    output_tro_file: Path,
) -> None:
    print("[regen] creating environment", flush=True)
    raw_env = runtime_env._ensure_env()
    print("[regen] resetting environment", flush=True)
    raw_env.reset()
    print("[regen] reset complete", flush=True)
    base_env = _unwrap_env(raw_env)

    output_scene_file.parent.mkdir(parents=True, exist_ok=True)
    output_tro_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[regen] saving scene to {output_scene_file}", flush=True)
    base_env.task.update_bddl_scope_metadata(base_env)
    base_env.scene.save(json_path=str(output_scene_file))
    refreshed_hashes = _refresh_expected_file_hashes(output_scene_file)
    print("[regen] scene save complete", flush=True)
    print(f"[regen] refreshed expected_file_hash for {refreshed_hashes} objects", flush=True)

    scene_metadata = _load_scene_metadata(source_scene_file)
    print(f"[regen] writing tro to {output_tro_file}", flush=True)
    tro_state = {
        bddl_name: bddl_inst.dump_state(serialized=False)
        for bddl_name, bddl_inst in base_env.task.object_scope.items()
        if bddl_inst.exists
    }
    tro_state["robot_poses"] = _resolve_robot_poses(scene_metadata=scene_metadata, base_env=base_env)

    with output_tro_file.open("w", encoding="utf-8") as handle:
        json.dump(tro_state, handle, cls=TorchEncoder, indent=4)
        handle.write("\n")
    print("[regen] tro write complete", flush=True)


def main() -> None:
    args = build_parser().parse_args()

    config = load_config_file(args.config)
    if not config.get("behavior_scene_file"):
        raise ValueError("Config must set environment.behavior_scene_file")

    source_scene_file = Path(config["behavior_scene_file"]).expanduser().resolve()
    default_scene_output, default_tro_output = _default_output_paths(source_scene_file)
    output_scene_file = (
        Path(args.output_scene_file).expanduser().resolve()
        if args.output_scene_file
        else default_scene_output
    )
    output_tro_file = (
        Path(args.output_tro_file).expanduser().resolve()
        if args.output_tro_file
        else default_tro_output
    )

    seed_tro_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="behavior_regen_") as temp_dir:
        if args.seed_tro_mode == "empty":
            seed_tro_path = Path(temp_dir) / "empty_tro_state.json"
            seed_tro_path.write_text("{}\n", encoding="utf-8")
            print(f"[regen] using empty seed tro {seed_tro_path}", flush=True)

        runtime_env = BehaviorRuntimeEnvironment(
            env_id=str(config["env_id"]),
            env_kwargs=_build_env_kwargs(config=config, seed_tro_path=seed_tro_path),
            auto_register=not bool(config.get("no_auto_register", False)),
        )

        try:
            _write_outputs(
                runtime_env=runtime_env,
                source_scene_file=source_scene_file,
                output_scene_file=output_scene_file,
                output_tro_file=output_tro_file,
            )
        finally:
            runtime_env.close()

    print(json.dumps({"scene_file": str(output_scene_file), "tro_state_file": str(output_tro_file)}, indent=2))


if __name__ == "__main__":
    main()
