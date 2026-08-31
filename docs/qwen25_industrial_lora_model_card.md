# VisioMind Qwen2.5-3B 工业任务规划 LoRA 模型卡

## 模型用途

该模型将一条中文或英文工业操作指令转换为唯一、可审计的
`visiomind-industrial-plan-v1` JSON 对象。输出含七类意图、固定语义槽位、
感知—决策—执行—验证—恢复任务序列，以及可交给 ACTION Agent 的动作序列。
它负责语言理解与任务分解；物理实例 ID、可达性和安全许可仍由场景本体、感知与
执行安全门控确定。

- 基础模型：`Qwen/Qwen2.5-3B-Instruct`（3.09B 参数）
- 方法：BF16 LoRA，Attention 与 MLP 全投影适配
- Adapter：`models/qwen25_3b_industrial_lora/adapter_model.safetensors`
- Adapter 大小：59,933,632 bytes
- Adapter SHA-256：`ec723139fd6443c625f7886eb9efb63960eaf2051c20f87010ace06606906726`
- 可训练参数：14,966,784 / 3,100,905,472（0.4827%）

Improved using Qwen.

## 固定输出契约

顶层键固定为 `schema_version`、`intent`、`slots`、`task_sequence`、
`action_sequence`；槽位固定为 `object`、`container`、`cell_index`、
`spatial_relation`。缺失信息必须为 `null`，不得由语言模型虚构。Schema、标准任务
序列和验证器实现在 `visiomind/decision/qwen_plan_schema.py`。

示例指令：

```text
现在请把钳子收纳至料箱的第3格，完成后报告状态
```

对应意图为 `transfer_inside`，执行序列为 `pick_up(pliers)` 后
`place_inside(pliers, parts_bin, cell_index=3)`；审计任务序列覆盖目标选择、抓取规划、
抓取物理验证、目标格定位、携物导航、放置、物理验证和失败恢复。

## 数据

数据位于 `data/qwen25_industrial/`，均为程序化工业模板，不含个人数据：

| Split | 数量 | 作用 |
|---|---:|---|
| train | 958 | 中英工业指令监督微调 |
| validation | 276 | 训练期独立渲染验证 |
| test | 396 | 原基准未见句式模板 |
| test_unseen_tools | 120 | 未在训练指令出现的 caliper、bearing、gear、valve、motor_shaft |
| test_unseen_paraphrases | 126 | 新增未见中英句式 |

所有 split 的完整指令交集为 0。`dataset_manifest.json` 记录文件 SHA-256、随机种子和
拆分策略。标签审计 v2 修复了旧解析器未识别英文 `number N` 而把明确格位标成
`null` 的问题；修复规则独立于模型输出，明确格位不得映射为空。

## 训练配置

| 项目 | 配置 |
|---|---|
| LoRA | rank 8，alpha 16，dropout 0.05，bias none |
| 目标层 | q/k/v/o、gate/up/down projection |
| 精度 | BF16，TF32 matmul |
| Epoch / 学习率 | 2 / 2e-4 cosine，warmup 5% |
| 序列长度 | 1024；训练与验证截断样本均为 0 |
| Batch | 1，gradient accumulation 8（有效 batch 8） |
| 随机种子 | 20262504 |
| 硬件 | NVIDIA RTX 3090 24 GB |
| 训练 / 总耗时 | 574.07 s / 596.34 s |
| 峰值显存 | 7,603,795,968 bytes |
| 最终验证损失 | 0.00024390 |

冻结基础模型，仅训练 LoRA 参数。训练代码会先保存 Adapter，再做最终验证，避免
训练已经完成却因末尾评估故障丢失权重。

## 完整评估结果

Prompt-only 与 LoRA 使用相同基础模型、System Prompt、测试样本、greedy decoding、
`max_new_tokens=700` 和 batch size 6。以下不是抽样结果；共评估 642 条并提交逐样本
原始输出。

| 642 条合并指标 | Prompt-only | LoRA | 绝对提升 |
|---|---:|---:|---:|
| JSON 可解析率 | 100.00% | 100.00% | +0.00 pp |
| 固定 Schema 有效率 | 0.00% | 100.00% | +100.00 pp |
| 意图准确率 | 80.06% | 98.60% | +18.54 pp |
| 槽位 Micro-F1 | 39.44% | 95.47% | +56.03 pp |
| 任务序列精确匹配 | 0.62% | 98.60% | +97.98 pp |
| 完整计划精确匹配 | 0.00% | 84.27% | +84.27 pp |

LoRA 泛化拆分：

| Split | Schema 有效 | 意图准确率 | 槽位 Micro-F1 | 任务序列 EM | 完整计划 EM |
|---|---:|---:|---:|---:|---:|
| test (396) | 100.00% | 100.00% | 98.00% | 100.00% | 90.91% |
| unseen tools (120) | 100.00% | 100.00% | 83.29% | 100.00% | 74.17% |
| unseen paraphrases (126) | 100.00% | 92.86% | 95.03% | 92.86% | 73.02% |

正式报告为 `reports/qwen25_{prompt_only,lora}_metrics.json`，逐样本指令、金标准、
原始输出、解析结果和指标为对应的 `*_predictions.jsonl`。评估后发现并修复金标准
解析器的 `number N` 漏标；报告中的 `scoring` 字段记录了数据 manifest 哈希和
基于已保存确定性输出重评分的 provenance。

## 复现

基础权重体积约 5.8 GB，不在 Git 中分发。按上游许可取得
`Qwen/Qwen2.5-3B-Instruct` 并置于 `models/base/Qwen2.5-3B-Instruct` 后运行：

```bash
python training/generate_qwen_lora_dataset.py
python training/train_qwen_industrial_lora.py

python training/evaluate_qwen_industrial_planner.py \
  --output reports/qwen25_prompt_only_metrics.json \
  --predictions reports/qwen25_prompt_only_predictions.jsonl

python training/evaluate_qwen_industrial_planner.py \
  --adapter models/qwen25_3b_industrial_lora \
  --output reports/qwen25_lora_metrics.json \
  --predictions reports/qwen25_lora_predictions.jsonl

python scripts/run_qwen_industrial_planner.py \
  "现在请把钳子收纳至料箱的第3格，完成后报告状态"
```

已验证软件版本：Python 3.10、PyTorch 2.7.0+cu128、Transformers 4.51.3、
PEFT 0.17.1。

## 局限与安全边界

- 数据是程序化中英模板，不能外推为真实工厂方言、ASR 噪声、复杂指代或开放域
  工具上的同等性能；真实操作员语料仍需补测。
- 未见工具主要错误为 canonical name 同义词漂移（如 `motor_axle`/`shaft`）及中文
  `游标卡尺` 被归一成 `ruler`；运行时必须用场景本体拒绝未知或非唯一对象。
- 未见句式主要错误来自安全停靠语句被误判为抓取，以及少量格位漏抽取。
- Schema 有效不等于物理可执行。不得绕过可达性、碰撞、唯一 grounding、急停和
  最终释放/AABB 包含验证；高风险或歧义指令必须拒绝或请求确认。

## 许可

Qwen2.5-3B-Instruct 及其衍生 Adapter 受 **Qwen Research License Agreement**
约束，仅限非商业研究/评估；商业使用需向上游另行申请许可。Adapter 目录包含官方
协议副本 `QWEN_RESEARCH_LICENSE.txt`、归属与修改声明 `NOTICE`。基础权重不随本仓库
分发。
