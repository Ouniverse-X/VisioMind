"""Evaluate prompt-only and LoRA Qwen planners on all industrial splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visiomind.decision.qwen_plan_schema import (  # noqa: E402
    SYSTEM_PROMPT,
    extract_json_object,
    slot_pairs,
    task_steps,
    validate_plan,
)


DEFAULT_BASE = ROOT / "models" / "base" / "Qwen2.5-3B-Instruct"
DEFAULT_DATA = ROOT / "data" / "qwen25_industrial"
DEFAULT_SPLITS = (
    "test",
    "test_unseen_tools",
    "test_unseen_paraphrases",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


class MetricAccumulator:
    def __init__(self):
        self.total = 0
        self.json_parse = 0
        self.json_object = 0
        self.schema_valid = 0
        self.intent_correct = 0
        self.task_exact = 0
        self.action_exact = 0
        self.plan_exact = 0
        self.slot_tp = 0
        self.slot_fp = 0
        self.slot_fn = 0
        self.schema_errors: Counter[str] = Counter()

    def update(
        self,
        *,
        expected: dict[str, Any],
        predicted: dict[str, Any] | None,
        parsed_json: bool,
    ) -> dict[str, Any]:
        self.total += 1
        self.json_parse += int(parsed_json)
        self.json_object += int(predicted is not None)
        schema_valid, errors = validate_plan(predicted)
        self.schema_valid += int(schema_valid)
        self.schema_errors.update(errors)
        intent_correct = bool(
            predicted is not None and predicted.get("intent") == expected["intent"]
        )
        self.intent_correct += int(intent_correct)
        expected_slots = slot_pairs(expected)
        predicted_slots = slot_pairs(predicted)
        tp = len(expected_slots & predicted_slots)
        fp = len(predicted_slots - expected_slots)
        fn = len(expected_slots - predicted_slots)
        self.slot_tp += tp
        self.slot_fp += fp
        self.slot_fn += fn
        task_exact = task_steps(predicted) == task_steps(expected)
        action_exact = bool(
            predicted is not None
            and predicted.get("action_sequence") == expected["action_sequence"]
        )
        plan_exact = predicted == expected
        self.task_exact += int(task_exact)
        self.action_exact += int(action_exact)
        self.plan_exact += int(plan_exact)
        return {
            "schema_valid": schema_valid,
            "schema_errors": errors,
            "intent_correct": intent_correct,
            "slot_tp": tp,
            "slot_fp": fp,
            "slot_fn": fn,
            "task_sequence_exact": task_exact,
            "action_sequence_exact": action_exact,
            "plan_exact": plan_exact,
        }

    def metrics(self) -> dict[str, Any]:
        precision = _safe_div(self.slot_tp, self.slot_tp + self.slot_fp)
        recall = _safe_div(self.slot_tp, self.slot_tp + self.slot_fn)
        slot_f1 = _safe_div(2 * precision * recall, precision + recall)
        return {
            "records": self.total,
            "json_parse_rate": _safe_div(self.json_parse, self.total),
            "json_object_rate": _safe_div(self.json_object, self.total),
            "json_schema_valid_rate": _safe_div(self.schema_valid, self.total),
            "intent_accuracy": _safe_div(self.intent_correct, self.total),
            "slot_micro_precision": precision,
            "slot_micro_recall": recall,
            "slot_micro_f1": slot_f1,
            "task_sequence_exact_match": _safe_div(self.task_exact, self.total),
            "action_sequence_exact_match": _safe_div(self.action_exact, self.total),
            "full_plan_exact_match": _safe_div(self.plan_exact, self.total),
            "schema_error_counts": dict(self.schema_errors.most_common()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.base_model = args.base_model.resolve()
    if args.adapter is not None:
        args.adapter = args.adapter.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    args.predictions = args.predictions.resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2.5-3B evaluation")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    split_metrics: dict[str, Any] = {}
    combined = MetricAccumulator()
    predictions_output: list[dict[str, Any]] = []
    started = time.time()
    generated_tokens = 0
    for split in args.splits:
        records = _read_jsonl(args.data_dir / f"{split}.jsonl")
        if args.limit is not None:
            records = records[: args.limit]
        accumulator = MetricAccumulator()
        split_started = time.time()
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": record["instruction"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in batch
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to("cuda")
            input_length = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output[:, input_length:]
            generated_tokens += int(generated.numel())
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for record, text in zip(batch, texts, strict=True):
                predicted, parsed_json = extract_json_object(text)
                sample_metrics = accumulator.update(
                    expected=record["expected"],
                    predicted=predicted,
                    parsed_json=parsed_json,
                )
                combined.update(
                    expected=record["expected"],
                    predicted=predicted,
                    parsed_json=parsed_json,
                )
                predictions_output.append(
                    {
                        "id": record["id"],
                        "split": split,
                        "instruction": record["instruction"],
                        "expected": record["expected"],
                        "raw_output": text,
                        "predicted": predicted,
                        "metrics": sample_metrics,
                    }
                )
        split_metrics[split] = {
            **accumulator.metrics(),
            "elapsed_seconds": time.time() - split_started,
        }
        # Keep completed split evidence recoverable even if a later split or
        # final report serialization fails.
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8") as handle:
            for record in predictions_output:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"completed split={split} records={len(records)} "
            f"schema_valid_rate={split_metrics[split]['json_schema_valid_rate']:.4f}",
            flush=True,
        )
    elapsed = time.time() - started
    report = {
        "evaluation_version": "qwen25-industrial-plan-eval-v1",
        "mode": "lora" if args.adapter is not None else "prompt_only",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter": (
            str(args.adapter.relative_to(ROOT.resolve()))
            if args.adapter is not None
            else None
        ),
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "splits": split_metrics,
        "combined": combined.metrics(),
        "runtime": {
            "elapsed_seconds": elapsed,
            "generated_token_tensor_elements": generated_tokens,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.predictions.open("w", encoding="utf-8") as handle:
        for record in predictions_output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
