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
