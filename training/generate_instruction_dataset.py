"""Generate a deterministic bilingual industrial-command dataset.

Templates, object aliases, and destinations are deliberately separated from
the runtime parser so evaluation measures paraphrase generalization rather
than replaying hard-coded command strings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


OBJECTS_ZH = ("螺丝刀", "内六角扳手", "扳手", "滚柱", "螺栓", "螺母", "手电筒", "钳子", "电钻")
OBJECTS_EN = ("screwdriver", "allen wrench", "wrench", "roller", "bolt", "nut", "flashlight", "pliers", "power drill")
DESTINATIONS_ZH = ("料箱", "工具箱", "包装箱", "托盘")
DESTINATIONS_EN = ("parts bin", "toolbox", "packing box", "tray")
POSITIONS_ZH = ("左侧的", "右侧的", "前面的", "最近的", "")
POSITIONS_EN = ("left ", "right ", "front ", "nearest ", "")

TRAIN_TEMPLATES = {
    "pick_up": (
        "拿起{position}{object}",
        "请取出{position}{object}",
        "帮我取来{position}{object}",
        "把{position}{object}拿给我",
        "拿一下{position}{object}并保持稳定",
        "pick up the {position}{object}",
        "retrieve the {position}{object}",
        "fetch the {position}{object}",
        "bring the {position}{object} to the operator",
        "grasp and lift the {position}{object}",
    ),
    "transfer_inside": (
        "把{position}{object}放进{destination}第{cell}格",
        "请将{position}{object}装入{destination}的第{cell}个格子",
        "将{position}{object}归位到{destination}第{cell}槽",
        "把{position}{object}收进{destination}的{cell}号位",
        "收纳{position}{object}到{destination}第{cell}位",
        "put the {position}{object} into cell {cell} of the {destination}",
        "place the {position}{object} inside bin {cell} in the {destination}",
        "store the {position}{object} in slot {cell} of the {destination}",
        "stow the {position}{object} in {destination} slot {cell}",
        "load the {position}{object} into the {destination} cell {cell}",
    ),
    "transfer_on_top": (
        "把{position}{object}放到{destination}上",
        "请将{position}{object}摆在{destination}表面",
        "将{position}{object}安放在{destination}顶部",
        "把{position}{object}搁在{destination}表面",
        "put the {position}{object} on the {destination}",
        "place the {position}{object} on top of the {destination}",
        "set the {position}{object} down on the {destination}",
        "leave the {position}{object} atop the {destination}",
    ),
    "inspect": (
        "识别场景中的{object}",
        "检查一下{position}{object}",
        "查看{position}{object}的位置",
        "找出场景内的{object}",
        "确认{position}{object}的空间坐标",
        "inspect the {position}{object}",
        "locate every {object} in the scene",
        "find and inspect the {position}{object}",
        "survey the {position}{object} and report its pose",
        "detect the {object} and report its pose",
    ),
    "move_near": (
        "移动到{position}{object}附近",
        "靠近{position}{object}",
        "去{position}{object}旁边",
        "驶向{position}{object}附近",
        "前往{position}{object}所在区域",
        "move near the {position}{object}",
        "navigate to the {position}{object}",
        "approach the {position}{object}",
        "proceed toward the {position}{object}",
        "go next to the {position}{object}",
    ),
    "recover_placement": (
        "检测到{position}{object}摆放失败，请重新放进{destination}第{cell}格",
        "{position}{object}放错格了，把它重新归位到{destination}第{cell}格",
        "检查失败的{position}{object}并重放到{destination}第{cell}槽",
        "{position}{object}掉在格外，请恢复摆放到{destination}第{cell}位",
        "重新抓取摆放错误的{position}{object}并放入{destination}第三格",
        "发现{position}{object}没有正确放好，请纠正到{destination}第{cell}格",
        "若{position}{object}未放好，请修正并归位至{destination}第{cell}格",
        "placement failed for the {position}{object}; recover it into cell {cell} of the {destination}",
        "the {position}{object} is outside its slot; re-place it in {destination} cell {cell}",
        "inspect and recover the misplaced {position}{object} into slot {cell} of the {destination}",
        "retry the failed placement of the {position}{object} into {destination} bin {cell}",
        "regrasp the incorrectly placed {position}{object} and stow it in cell {cell} of the {destination}",
    ),
    "stop": (
        "停止当前任务",
        "立即停下",
        "终止操作",
        "取消任务并停车",
        "stop the current task",
        "halt now",
        "cancel and stop",
        "abort the operation",
        "停止并保持静止",
    ),
}

TEST_TEMPLATES = {
    "pick_up": ("麻烦拿一下{position}{object}", "bring me the {position}{object}"),
    "transfer_inside": (
        "请把{position}{object}收纳至{destination}的第{cell}个位置",
        "stow the {position}{object} in number {cell} of the {destination}",
    ),
    "transfer_on_top": (
        "请把{position}{object}搁到{destination}顶面",
        "leave the {position}{object} atop the {destination}",
    ),
    "inspect": ("确认{position}{object}在哪", "survey the {position}{object}"),
    "move_near": ("前往{position}{object}所在处", "proceed toward the {position}{object}"),
    "recover_placement": (
        "发现{position}{object}没有放好，请纠正到{destination}第{cell}格",
        "correct the failed placement of the {position}{object} into number {cell} of the {destination}",
    ),
    "stop": ("停止并保持不动", "emergency halt"),
}


def _render(template: str, rng: random.Random) -> str:
    chinese = any("\u4e00" <= char <= "\u9fff" for char in template)
    rendered = template.format(
        object=rng.choice(OBJECTS_ZH if chinese else OBJECTS_EN),
        destination=rng.choice(DESTINATIONS_ZH if chinese else DESTINATIONS_EN),
        position=rng.choice(POSITIONS_ZH if chinese else POSITIONS_EN),
        cell=rng.randint(1, 6),
    ).replace("  ", " ").strip()
    if chinese:
        prefix = rng.choice(("", "请", "现在", "麻烦", "操作员要求："))
        suffix = rng.choice(("", "。", "，谢谢", "，完成后报告状态"))
    else:
        prefix = rng.choice(("", "please ", "now ", "operator request: "))
        suffix = rng.choice(("", ".", " safely", " and report status"))
    return f"{prefix}{rendered}{suffix}".strip()


def generate(templates: dict[str, tuple[str, ...]], count_per_template: int, seed: int):
    rng = random.Random(seed)
    records = []
    for intent, intent_templates in templates.items():
        for template_id, template in enumerate(intent_templates):
            seen = set()
            attempts = 0
            while len(seen) < count_per_template and attempts < count_per_template * 20:
                attempts += 1
                seen.add(_render(template, rng))
            for text in sorted(seen):
                records.append(
                    {"text": text, "intent": intent, "template_id": template_id}
                )
    rng.shuffle(records)
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-template", type=int, default=45)
    parser.add_argument("--test-per-template", type=int, default=30)
    args = parser.parse_args()
    train = generate(TRAIN_TEMPLATES, args.train_per_template, seed=202607)
    test = generate(TEST_TEMPLATES, args.test_per_template, seed=202608)
    _write_jsonl(args.output_dir / "train.jsonl", train)
    _write_jsonl(args.output_dir / "test.jsonl", test)
    print(json.dumps({"train": len(train), "test": len(test)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
