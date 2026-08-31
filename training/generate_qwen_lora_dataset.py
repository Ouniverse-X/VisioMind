from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visiomind.plan.instruction_model import (
    CONTAINER_ALIASES,
    OBJECT_ALIASES,
    SPATIAL_ALIASES,
    _extract_cell_index,
    _find_alias,
)
from visiomind.plan.qwen_plan_schema import (
    SYSTEM_PROMPT,
    build_plan,
    compact_json,
)


DEFAULT_SOURCE = ROOT / "data" / "instructions"
DEFAULT_OUTPUT = ROOT / "data" / "qwen25_industrial"


UNSEEN_TOOLS = {
    "caliper": ("游标卡尺", "caliper"),
    "bearing": ("轴承", "bearing"),
    "gear": ("齿轮", "gear"),
    "valve": ("阀门", "valve"),
    "motor_shaft": ("电机轴", "motor shaft"),
}


UNSEEN_PARAPHRASE_TEMPLATES = {
    "pick_up": (
        "锁定{position}{object}并将它安全提离台面",
        "secure and lift the {position}{object} clear of the bench",
    ),
    "transfer_inside": (
        "将{position}{object}送往{destination}，目标仓位编号为{cell}",
        "route the {position}{object} to storage position {cell} within the {destination}",
    ),
    "transfer_on_top": (
        "完成{position}{object}到{destination}上表面的转运",
        "transfer the {position}{object} onto the upper surface of the {destination}",
    ),
    "inspect": (
        "对{position}{object}执行身份确认与位姿测量",
        "perform identity confirmation and pose measurement on the {position}{object}",
    ),
    "move_near": (
        "在{position}{object}旁的安全停靠点就位",
        "take position at a safe standoff beside the {position}{object}",
    ),
    "recover_placement": (
        "复核{position}{object}的错误入格状态并恢复到{destination}第{cell}格",
        "audit the misplaced {position}{object} and restore it to position {cell} in the {destination}",
    ),
    "stop": (
        "撤销后续动作并让所有执行机构保持静止",
        "revoke pending actions and hold every actuator stationary",
    ),
}


SEEN_OBJECTS = {
    key: (aliases[0], next((a for a in aliases if a.isascii()), aliases[0]))
    for key, aliases in OBJECT_ALIASES.items()
    if key != "half_apple"
}
SEEN_CONTAINERS = {
    "parts_bin": ("料箱", "parts bin"),
    "toolbox": ("工具箱", "toolbox"),
    "packing_box": ("包装箱", "packing box"),
    "tray": ("托盘", "tray"),
}
POSITIONS = {
    None: ("", ""),
    "left": ("左侧的", "left "),
    "right": ("右侧的", "right "),
    "front": ("前面的", "front "),
    "nearest": ("最近的", "nearest "),
}


ENGLISH_CELL_PATTERN = re.compile(
    r"\b(?:cell|slot|position|number)\s*(?:no\.?\s*)?(\d+)\b",
    flags=re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _expected_from_source(record: dict[str, Any]) -> dict[str, Any]:
    text = str(record["text"])
    cell_index = _extract_cell_index(text)
    if cell_index is None:
        match = ENGLISH_CELL_PATTERN.search(text)
        cell_index = int(match.group(1)) if match is not None else None
    return build_plan(
        intent=str(record["intent"]),
        object_name=_find_alias(text, OBJECT_ALIASES),
        container=_find_alias(text, CONTAINER_ALIASES),
        cell_index=cell_index,
        spatial_relation=_find_alias(text, SPATIAL_ALIASES),
    )


def _record(
    *,
    record_id: str,
    split: str,
    category: str,
    instruction: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "split": split,
        "category": category,
        "instruction": instruction,
        "expected": expected,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": compact_json(expected)},
        ],
    }


def _convert_source(
    source: list[dict[str, Any]], split: str, category: str
) -> list[dict[str, Any]]:
    return [
        _record(
            record_id=f"{split}-{index:05d}",
            split=split,
            category=category,
            instruction=str(item["text"]),
            expected=_expected_from_source(item),
        )
        for index, item in enumerate(source)
    ]


def _format_template(
    template: str,
    *,
    object_pair: tuple[str, str],
    container_pair: tuple[str, str],
    position_pair: tuple[str, str],
    cell: int,
) -> str:
    chinese = any("\u4e00" <= char <= "\u9fff" for char in template)
    language_index = 0 if chinese else 1
    return (
        template.format(
            object=object_pair[language_index],
            destination=container_pair[language_index],
            position=position_pair[language_index],
            cell=cell,
        )
        .replace("  ", " ")
        .strip()
    )


def _generate_unseen_tools(seed: int, count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    templates = (
        ("transfer_inside", "请把{object}存入{destination}第{cell}格"),
        ("transfer_inside", "place the {object} into cell {cell} of the {destination}"),
        ("pick_up", "请抓取{object}并稳定抬升"),
        ("pick_up", "pick up and hold the {object}"),
        ("inspect", "检查{object}并报告位姿"),
        ("inspect", "inspect the {object} and report its pose"),
        ("move_near", "移动到{object}附近"),
        ("move_near", "move to a safe standoff near the {object}"),
    )
    output = []
    tool_items = list(UNSEEN_TOOLS.items())
    container_items = list(SEEN_CONTAINERS.items())
    for index in range(count):
        object_name, object_pair = tool_items[index % len(tool_items)]
        intent, template = templates[index % len(templates)]
        container_name, container_pair = rng.choice(container_items)
        cell = rng.randint(1, 6)
        instruction = _format_template(
            template,
            object_pair=object_pair,
            container_pair=container_pair,
            position_pair=POSITIONS[None],
            cell=cell,
        )
        requires_container = intent == "transfer_inside"
        expected = build_plan(
            intent=intent,
            object_name=object_name,
            container=container_name if requires_container else None,
            cell_index=cell if requires_container else None,
        )
        output.append(
            _record(
                record_id=f"unseen-tool-{index:05d}",
                split="test_unseen_tools",
                category="unseen_tool",
                instruction=instruction,
                expected=expected,
            )
        )
    return output


def _generate_unseen_paraphrases(seed: int, count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    intents = tuple(UNSEEN_PARAPHRASE_TEMPLATES)
    object_items = list(SEEN_OBJECTS.items())
    container_items = list(SEEN_CONTAINERS.items())
    position_items = list(POSITIONS.items())
    output = []
    for index in range(count):
        intent = intents[index % len(intents)]
        templates = UNSEEN_PARAPHRASE_TEMPLATES[intent]
        template = templates[(index // len(intents)) % len(templates)]
        object_name, object_pair = rng.choice(object_items)
        container_name, container_pair = rng.choice(container_items)
        spatial_relation, position_pair = rng.choice(position_items)
        cell = rng.randint(1, 6)
        instruction = _format_template(
            template,
            object_pair=object_pair,
            container_pair=container_pair,
            position_pair=position_pair,
            cell=cell,
        )
        if intent == "stop":
            object_name = container_name = spatial_relation = None
            cell_value = None
        elif intent in {"transfer_inside", "recover_placement"}:
            cell_value = cell
        elif intent == "transfer_on_top":
            cell_value = None
        else:
            container_name = None
            cell_value = None
        expected = build_plan(
            intent=intent,
            object_name=object_name,
            container=container_name,
            cell_index=cell_value,
            spatial_relation=spatial_relation,
        )
        output.append(
            _record(
                record_id=f"unseen-paraphrase-{index:05d}",
                split="test_unseen_paraphrases",
                category="unseen_paraphrase",
                instruction=instruction,
                expected=expected,
            )
        )
    return output


def _write(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    return {"path": path.name, "records": len(records), "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-per-template", type=int, default=14)
    parser.add_argument("--val-per-template", type=int, default=4)
    parser.add_argument("--unseen-tool-count", type=int, default=120)
    parser.add_argument("--unseen-paraphrase-count", type=int, default=126)
    args = parser.parse_args()

    try:
        base_source = str(args.source_dir.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        base_source = str(args.source_dir.resolve())

    source_train = _read_jsonl(args.source_dir / "train.jsonl")
    source_test = _read_jsonl(args.source_dir / "test.jsonl")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in source_train:
        grouped[(str(item["intent"]), int(item["template_id"]))].append(item)
    rng = random.Random(20262501)
    train_source: list[dict[str, Any]] = []
    val_source: list[dict[str, Any]] = []
    for key in sorted(grouped):
        items = sorted(grouped[key], key=lambda value: str(value["text"]))
        rng.shuffle(items)
        val_source.extend(items[: args.val_per_template])
        train_source.extend(
            items[args.val_per_template : args.val_per_template + args.train_per_template]
        )
    rng.shuffle(train_source)
    rng.shuffle(val_source)
    splits = {
        "train": _convert_source(train_source, "train", "standard_train"),
        "validation": _convert_source(val_source, "validation", "standard_validation"),
        "test": _convert_source(source_test, "test", "heldout_template"),
        "test_unseen_tools": _generate_unseen_tools(20262502, args.unseen_tool_count),
        "test_unseen_paraphrases": _generate_unseen_paraphrases(
            20262503, args.unseen_paraphrase_count
        ),
    }
    instruction_sets = {
        name: {record["instruction"] for record in records} for name, records in splits.items()
    }
    leakage = {}
    for left_name, left in instruction_sets.items():
        for right_name, right in instruction_sets.items():
            if left_name < right_name:
                overlap = sorted(left & right)
                leakage[f"{left_name}__{right_name}"] = len(overlap)
                if overlap:
                    raise RuntimeError(
                        f"instruction leakage between {left_name} and {right_name}: {overlap[:3]}"
                    )
    files = [_write(args.output_dir / f"{name}.jsonl", records) for name, records in splits.items()]
    manifest = {
        "dataset_version": "qwen25-industrial-plan-v2-label-audit",
        "seed": 20262501,
        "base_source": base_source,
        "split_policy": {
            "train_validation": "same template families, disjoint rendered instructions",
            "test": "held-out sentence templates from the original benchmark",
            "test_unseen_tools": sorted(UNSEEN_TOOLS),
            "test_unseen_paraphrases": "new templates absent from train/validation/test",
            "cell_label_parser": (
                "Chinese 第N格 plus English cell/slot/position/number N; "
                "explicit references must never map to null"
            ),
        },
        "leakage_exact_instruction_overlap": leakage,
        "files": files,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
