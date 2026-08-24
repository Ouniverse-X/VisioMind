from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "industrial_instruction.joblib"
sys.path.append(str(ROOT))
try:
    from visiomind.decision import IndustrialInstructionModel
finally:
    sys.path.remove(str(ROOT))


def test_bilingual_transfer_instruction_decomposes_to_pick_and_place() -> None:
    plan = IndustrialInstructionModel(MODEL).parse(
        "帮我把左侧的滚柱放到料箱的第三个格子中"
    )
    assert plan.intent == "transfer_inside"
    assert plan.slots == {
        "object": "roller",
        "container": "parts_bin",
        "cell_index": 3,
        "spatial_relation": "left",
    }
    assert [step["action"] for step in plan.action_sequence] == [
        "pick_up",
        "place_inside",
    ]
    assert plan.task_sequence[-1]["step"] == "recover_if_needed"


def test_english_pick_instruction_extracts_tool() -> None:
    plan = IndustrialInstructionModel(MODEL).parse(
        "Please bring me the nearest allen wrench."
    )
    assert plan.intent == "pick_up"
    assert plan.slots["object"] == "allen_wrench"
    assert plan.slots["spatial_relation"] == "nearest"


def test_missing_destination_is_rejected() -> None:
    model = IndustrialInstructionModel(MODEL)
    try:
        model.parse("把螺丝刀收进去")
    except ValueError as exc:
        assert "destination" in str(exc)
    else:
        raise AssertionError("missing placement destination must be rejected")


def test_recovery_instruction_preserves_third_cell_evidence() -> None:
    plan = IndustrialInstructionModel(MODEL).parse(
        "检测到钳子摆放失败，请重新放进工具箱第三格"
    )
    assert plan.intent == "recover_placement"
    assert plan.slots["object"] == "pliers"
    assert plan.slots["container"] == "toolbox"
    assert plan.slots["cell_index"] == 3
    assert plan.task_sequence[0]["step"] == "detect_failed_placement"
    assert plan.action_sequence[0]["target"]["recovery"] is True
    assert plan.action_sequence[1]["target"]["cell_index"] == 3


def test_power_drill_alias_is_supported() -> None:
    plan = IndustrialInstructionModel(MODEL).parse(
        "Please inspect the power drill on the industrial workbench."
    )
    assert plan.intent == "inspect"
    assert plan.slots["object"] == "drill"
