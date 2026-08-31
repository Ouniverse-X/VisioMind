from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "models" / "base" / "Qwen2.5-3B-Instruct"
DEFAULT_DATA = ROOT / "data" / "qwen25_industrial"
DEFAULT_OUTPUT = ROOT / "models" / "qwen25_3b_industrial_lora"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ChatSftDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int):
        self.examples = []
        truncated = 0
        for record in records:
            messages = record["messages"]
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            complete = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]
            tokenized = tokenizer(
                complete,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            labels = list(tokenized["input_ids"])
            prefix_length = min(len(prompt_ids), len(labels))
            labels[:prefix_length] = [-100] * prefix_length
            if not any(value != -100 for value in labels):
                raise ValueError(f"assistant target truncated for record {record['id']}")
            if len(tokenized["input_ids"]) >= max_length:
                truncated += 1
            self.examples.append(
                {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": labels,
                }
            )
        self.truncated_examples = truncated

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20262504)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-validation-records", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen2.5-3B LoRA training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_records = _load_jsonl(args.data_dir / "train.jsonl")
    validation_records = _load_jsonl(args.data_dir / "validation.jsonl")
    if args.max_train_records is not None:
        train_records = train_records[: args.max_train_records]
    if args.max_validation_records is not None:
        validation_records = validation_records[: args.max_validation_records]
    train_dataset = ChatSftDataset(train_records, tokenizer, args.max_length)
    validation_dataset = ChatSftDataset(validation_records, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    model.peft_config["default"].base_model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_result = trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    license_source = args.base_model / "LICENSE"
    if license_source.is_file():
        shutil.copyfile(license_source, args.output_dir / "QWEN_RESEARCH_LICENSE.txt")
    (args.output_dir / "NOTICE").write_text(
        "Improved using Qwen.\n\n"
        "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT,\n"
        "Copyright (c) Alibaba Cloud. All Rights Reserved.\n\n"
        "VisioMind modification notice: this directory contains a "
        "project-generated\nLoRA adapter trained for industrial task planning. "
        "The Qwen base-model\nweights are not redistributed.\n",
        encoding="utf-8",
    )
    adapter_path = args.output_dir / "adapter_model.safetensors"
    validation_started = time.time()
    validation_metrics = trainer.evaluate()
    validation_elapsed = time.time() - validation_started
    elapsed = time.time() - started
    manifest = {
        "model": "Qwen2.5-3B-Instruct industrial task planner LoRA",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_local_path": str(args.base_model.relative_to(ROOT)),
        "adapter_path": adapter_path.name,
        "adapter_sha256": _sha256(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "tokenizer": "inherited unchanged from Qwen/Qwen2.5-3B-Instruct",
        "training": {
            "method": "BF16 LoRA",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "seed": args.seed,
            "train_records": len(train_records),
            "validation_records": len(validation_records),
            "train_truncated_examples": train_dataset.truncated_examples,
            "validation_truncated_examples": validation_dataset.truncated_examples,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
            "elapsed_seconds": elapsed,
            "final_validation_elapsed_seconds": validation_elapsed,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "train_metrics": train_result.metrics,
            "validation_metrics": validation_metrics,
        },
        "license": (
            "project-generated adapter derived from Qwen2.5-3B-Instruct; "
            "Qwen Research License (non-commercial research/evaluation only)"
        ),
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
