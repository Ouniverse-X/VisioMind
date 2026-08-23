"""Parse a natural-language instruction and execute its grounded Isaac plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from visiomind.decision import IndustrialInstructionModel


ROOT = Path(__file__).resolve().parent


def _ground_action_sequence(
    sequence: list[dict[str, object]], grounding: dict[str, dict[str, str]]
) -> list[dict[str, object]]:
    grounded = []
    for step in sequence:
        copy = json.loads(json.dumps(step))
        target = copy.get("target", {})
        if target.get("object") is not None:
            canonical = target["object"]
            try:
                target["object"] = grounding["objects"][canonical]
            except KeyError as exc:
                raise ValueError(f"scene has no grounded object for {canonical!r}") from exc
        if target.get("container") is not None:
            canonical = target["container"]
            try:
                target["container"] = grounding["containers"][canonical]
            except KeyError as exc:
                raise ValueError(f"scene has no grounded container for {canonical!r}") from exc
        grounded.append(copy)
    return grounded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "voltron" / "configs" / "half_apple_to_packing_box_place_inside_i10.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "industrial_instruction.joblib",
    )
    parser.add_argument(
        "--grounding",
        type=Path,
        default=ROOT / "configs" / "scene_grounding_preparing_lunch_box.json",
    )
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = IndustrialInstructionModel(args.model).parse(args.instruction)
    if plan.confidence < args.min_confidence:
        raise RuntimeError(
            f"instruction confidence {plan.confidence:.3f} is below "
            f"threshold {args.min_confidence:.3f}"
        )
    grounding = json.loads(args.grounding.read_text(encoding="utf-8"))
    action_sequence = _ground_action_sequence(plan.action_sequence, grounding)
    output = plan.to_dict()
    output["grounded_action_sequence"] = action_sequence
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment["VOLTRON_HOME"] = str(ROOT / "voltron")
    command = [
        sys.executable,
        str(ROOT / "run_action_only_overlay.py"),
        "--config",
        str(args.config),
        "--action-sequence",
        json.dumps(action_sequence, ensure_ascii=False, separators=(",", ":")),
        "--action-instruction",
        args.instruction,
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
