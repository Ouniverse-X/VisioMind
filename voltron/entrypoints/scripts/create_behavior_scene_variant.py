#!/usr/bin/env python3
"""Create a BEHAVIOR scene JSON variant with overridden object joint positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source BEHAVIOR scene JSON.")
    parser.add_argument("--target", required=True, help="Output scene JSON path.")
    parser.add_argument(
        "--set-joint",
        nargs=2,
        metavar=("OBJECT_NAME", "JOINT_POS"),
        action="append",
        default=[],
        help="Override an object's first joint position in state.registry.object_registry.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    source_path = Path(args.source).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    object_registry = payload["state"]["registry"]["object_registry"]
    for object_name, raw_joint_pos in args.set_joint:
        if object_name not in object_registry:
            raise KeyError(f"Object '{object_name}' not found in state.registry.object_registry")
        joint_state = object_registry[object_name].get("joint_pos")
        if not isinstance(joint_state, list) or len(joint_state) == 0:
            raise ValueError(f"Object '{object_name}' does not expose a writable joint_pos list")
        joint_state[0] = float(raw_joint_pos)

    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")

    print(target_path)


if __name__ == "__main__":
    main()
