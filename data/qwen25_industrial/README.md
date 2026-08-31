# Qwen2.5 工业任务规划数据卡

这是 VisioMind 固定 JSON 任务规划器的程序化中英工业指令数据。每条 JSONL 包含
`id`、`split`、`category`、`instruction`、`expected` 和可直接用于 Qwen Chat SFT 的
`messages`。

## 拆分

- `train.jsonl`：958 条；
- `validation.jsonl`：276 条；
- `test.jsonl`：396 条未见原始模板；
- `test_unseen_tools.jsonl`：120 条，工具为 caliper、bearing、gear、valve、
  motor_shaft，工具中英文表面形式均不出现在训练指令；
- `test_unseen_paraphrases.jsonl`：126 条全新中英句式。

所有 split 之间精确指令重合数均为 0。文件哈希、种子和拆分策略见
`dataset_manifest.json`。

## 标签规则

输出遵循 `visiomind-industrial-plan-v1`。对象、容器、格位和空间关系只从显式文本
提取；缺失槽位为 `null`。格位解析覆盖中文 `第N格` 与英文 `cell/slot/position/number
N`。数据版本 `qwen25-industrial-plan-v2-label-audit` 修复了早期英文 `number N`
被错误标为空值的问题。

## 生成与限制

```bash
python training/generate_qwen_lora_dataset.py
pytest -q tests/test_qwen_lora_pipeline.py
```

该数据由模板程序生成，不含个人数据，也不是实采工厂对话。它适合验证格式遵循、
意图/槽位解析和受控泛化，不应用于声称真实工厂口语性能。
