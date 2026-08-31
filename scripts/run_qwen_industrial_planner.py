#!/usr/bin/env python3
"""Run the Qwen2.5 industrial planner and print its fixed JSON plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visiomind.decision.qwen_plan_schema import (  # noqa: E402
    SYSTEM_PROMPT,
    extract_json_object,
    validate_plan,
)


DEFAULT_BASE = ROOT / "models" / "base" / "Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = ROOT / "models" / "qwen25_3b_industrial_lora"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural-language industrial instruction to auditable JSON plan"
    )
    parser.add_argument("instruction", help="Chinese or English operator instruction")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Disable the project LoRA and run the reproducible baseline",
    )
    parser.add_argument("--max-new-tokens", type=int, default=700)
    args = parser.parse_args()

    if not args.base_model.is_dir():
        parser.error(
            "base model is missing; download Qwen/Qwen2.5-3B-Instruct to "
            f"{args.base_model}"
        )
    if not args.prompt_only and not (args.adapter / "adapter_config.json").is_file():
        parser.error(f"LoRA adapter is missing or incomplete: {args.adapter}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    if not args.prompt_only:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.instruction},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
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
    output = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    plan, parsed_json = extract_json_object(output)
    valid, errors = validate_plan(plan)
    if not parsed_json or not valid:
        print(output)
        print(
            "planner output failed JSON schema validation: " + "; ".join(errors),
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
