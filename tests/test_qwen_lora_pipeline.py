import hashlib
import json
from pathlib import Path

from visiomind.decision.qwen_plan_schema import (
    SCHEMA_VERSION,
    build_plan,
    compact_json,
    extract_json_object,
    task_steps,
    validate_plan,
)
from training.generate_qwen_lora_dataset import _expected_from_source


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "qwen25_industrial"


def _records(name):
    with (DATA_DIR / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_transfer_plan_has_fixed_auditable_sequence():
    plan = build_plan(
        intent="transfer_inside",
        object_name="pliers",
        container="toolbox",
        cell_index=3,
    )
    valid, errors = validate_plan(plan)
    assert valid, errors
    assert plan["schema_version"] == SCHEMA_VERSION
    assert task_steps(plan) == [
        "select_target",
        "plan_grasp",
        "pick_up",
        "localize_destination",
        "navigate_with_object",
        "place_inside",
        "verify_placement",
        "recover_if_needed",
    ]


def test_json_extractor_accepts_plain_and_fenced_json():
    plan = build_plan(intent="stop")
    raw = compact_json(plan)
    assert extract_json_object(raw) == (plan, True)
    assert extract_json_object(f"```json\n{raw}\n```") == (plan, True)
    assert extract_json_object("not json") == (None, False)


def test_all_committed_qwen_targets_conform_to_schema():
    for split in (
        "train",
        "validation",
        "test",
        "test_unseen_tools",
        "test_unseen_paraphrases",
    ):
        records = _records(split)
        assert records
        for record in records:
            valid, errors = validate_plan(record["expected"])
            assert valid, (record["id"], errors)


def test_qwen_splits_have_no_exact_instruction_leakage():
    splits = {
        name: {record["instruction"] for record in _records(name)}
        for name in (
            "train",
            "validation",
            "test",
            "test_unseen_tools",
            "test_unseen_paraphrases",
        )
    }
    names = sorted(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert not splits[left] & splits[right], (left, right)


def test_unseen_tools_are_absent_from_supervised_training_text():
    training_text = "\n".join(record["instruction"] for record in _records("train"))
    for surface_form in (
        "游标卡尺",
        "caliper",
        "轴承",
        "bearing",
        "齿轮",
        "gear",
        "阀门",
        "valve",
        "电机轴",
        "motor shaft",
    ):
        assert surface_form not in training_text


def test_english_number_cell_label_is_not_silently_dropped():
    expected = _expected_from_source(
        {
            "text": "stow the pliers in number 5 of the toolbox",
            "intent": "transfer_inside",
        }
    )
    assert expected["slots"]["cell_index"] == 5


def test_committed_adapter_matches_manifest_and_github_size_limit():
    adapter_dir = ROOT / "models" / "qwen25_3b_industrial_lora"
    adapter = adapter_dir / "adapter_model.safetensors"
    manifest = json.loads(
        (adapter_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert adapter.stat().st_size == manifest["adapter_bytes"]
    assert digest == manifest["adapter_sha256"]
    assert adapter.stat().st_size < 100_000_000
    adapter_config = json.loads(
        (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert adapter_config["base_model_name_or_path"] == (
        "Qwen/Qwen2.5-3B-Instruct"
    )
