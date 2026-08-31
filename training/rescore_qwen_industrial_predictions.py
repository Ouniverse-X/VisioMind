"""Rescore saved generations after an independently audited label correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_qwen_industrial_planner import MetricAccumulator, extract_json_object


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "qwen25_industrial"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--input-predictions", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    report = json.loads(args.input_report.read_text(encoding="utf-8"))
    predictions = _read_jsonl(args.input_predictions)
    split_names = list(report["splits"])
    expected_by_id: dict[str, dict[str, Any]] = {}
    for split in split_names:
        for record in _read_jsonl(args.data_dir / f"{split}.jsonl"):
            if record["id"] in expected_by_id:
                raise ValueError(f"duplicate record id: {record['id']}")
            expected_by_id[record["id"]] = record

    combined = MetricAccumulator()
    per_split = {split: MetricAccumulator() for split in split_names}
    corrected_predictions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for prediction in predictions:
        record_id = prediction["id"]
        if record_id not in expected_by_id:
            raise ValueError(f"prediction id absent from current dataset: {record_id}")
        gold = expected_by_id[record_id]
        if prediction["instruction"] != gold["instruction"]:
            raise ValueError(f"instruction changed for {record_id}")
        split = prediction["split"]
        reparsed, parsed_json = extract_json_object(prediction["raw_output"])
        if reparsed != prediction.get("predicted"):
            raise ValueError(f"stored prediction disagrees with raw output: {record_id}")
        sample_metrics = per_split[split].update(
            expected=gold["expected"],
            predicted=prediction.get("predicted"),
            parsed_json=parsed_json,
        )
        combined.update(
            expected=gold["expected"],
            predicted=prediction.get("predicted"),
            parsed_json=parsed_json,
        )
        prediction["expected"] = gold["expected"]
        prediction["metrics"] = sample_metrics
        corrected_predictions.append(prediction)
        seen_ids.add(record_id)
    missing = set(expected_by_id) - seen_ids
    if missing:
        raise ValueError(f"missing {len(missing)} predictions, e.g. {sorted(missing)[:3]}")

    for split in split_names:
        elapsed = report["splits"][split].get("elapsed_seconds")
        report["splits"][split] = {
            **per_split[split].metrics(),
            "elapsed_seconds": elapsed,
        }
    report["combined"] = combined.metrics()
    report["evaluation_version"] = "qwen25-industrial-plan-eval-v2-label-audit"
    manifest_path = args.data_dir / "dataset_manifest.json"
    report["scoring"] = {
        "source": "saved deterministic model generations",
        "gold_labels": str(manifest_path.relative_to(ROOT)),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "note": (
            "Metrics recomputed without regeneration after fixing the gold-label "
            "parser for explicit English 'number N' cell references."
        ),
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_predictions.open("w", encoding="utf-8") as handle:
        for prediction in corrected_predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
