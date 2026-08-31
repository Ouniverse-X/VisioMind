---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- industrial-robotics
- task-planning
- structured-output
- zh
- en
license: other
license_name: qwen-research
---

# VisioMind Qwen2.5-3B Industrial Planner LoRA

Improved using Qwen.

This project-generated PEFT Adapter converts Chinese or English industrial
operator instructions into the fixed `visiomind-industrial-plan-v1` JSON
contract. It predicts intent, object/container/cell/spatial slots, an auditable
perception-to-recovery task sequence, and an executable action sequence.

## Artifact

- Base: `Qwen/Qwen2.5-3B-Instruct`
- LoRA: rank 8, alpha 16, dropout 0.05
- Adapted modules: q/k/v/o and gate/up/down projections
- Trainable parameters: 14,966,784 (0.4827%)
- Weight size: 59,933,632 bytes
- SHA-256: `ec723139fd6443c625f7886eb9efb63960eaf2051c20f87010ace06606906726`

The base weights are not included. Place the upstream model at
`models/base/Qwen2.5-3B-Instruct` or pass another local path to the scripts. The
LoRA adds no tokens and inherits the upstream tokenizer, which is not duplicated
in this directory.

## Evaluation

All results use deterministic greedy decoding on 642 programmatic held-out
instructions. Compared with the same base model and prompt without LoRA:

| Metric | Prompt-only | LoRA |
|---|---:|---:|
| Fixed JSON Schema valid | 0.00% | 100.00% |
| Intent accuracy | 80.06% | 98.60% |
| Slot Micro-F1 | 39.44% | 95.47% |
| Task-sequence exact match | 0.62% | 98.60% |
| Full-plan exact match | 0.00% | 84.27% |

Unseen-tool slot F1 is 83.29%; unseen-paraphrase intent accuracy / slot F1 are
92.86% / 95.03%. Full reports and every raw prediction are committed under
`reports/`. See `docs/qwen25_industrial_lora_model_card.md` for data splits,
training configuration, metric definitions, failure analysis, and provenance.

## Run

```bash
python scripts/run_qwen_industrial_planner.py \
  "现在请把钳子收纳至料箱的第3格，完成后报告状态"
```

The model output is semantic planning evidence, not permission to move a
robot. Runtime grounding, reachability, collision checks, emergency stop, and
physical placement verification remain mandatory.

## License

This derivative Adapter is governed by the Qwen Research License Agreement and
is limited to non-commercial research/evaluation. See
`QWEN_RESEARCH_LICENSE.txt` and `NOTICE` in this directory. Commercial use
requires a separate upstream license.
