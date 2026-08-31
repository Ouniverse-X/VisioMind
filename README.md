# 灵眸智控

面向挑战杯赛题 **XH-202607「工业环境下物体感知识别与指令交互型智能体研发」** 的交互式机器人智能体。系统接收中英文自然语言指令，完成工业工具感知、指令理解、任务分解、抓取放置、结果验证与失败恢复，形成“感知—决策—执行—验证”闭环。

[工业钳子识别、抓取并放入料箱第三格演示](demo/visiomind_industrial_demo.mp4)

## 核心能力

- **感知**：轻量多任务检测器识别螺栓、扳手、滚柱、螺丝刀、钳子、螺母、料箱等对象，并结合 RGB-D、实例掩码和相机参数恢复三维位置。
- **决策**：字符级 TF-IDF 分类器提供轻量离线解析；Qwen2.5-3B LoRA 提供固定 JSON Schema 的中英文任务规划。
- **执行**：AnyGrasp 生成 6D 抓取候选，CuRobo 完成机械臂/全身规划，Nav2 完成携物导航，多格料箱模型计算指定格位的安全放置区域。
- **验证与恢复**：连续检查目标身份、抬升、附着、释放和 AABB 包含状态；摆放失败时重新定位、抓取和放置。
- **仿真与真机接口**：支持 Isaac Sim、OmniGibson/BEHAVIOR-1K、R1 Pro、ROS 2 底盘串口和多相机接入。

运行链路如下：

```text
自然语言指令 → 意图/槽位解析 → 场景目标绑定 → ACTION 序列
             → RGB-D/AnyGrasp → CuRobo/Nav2 执行 → 物理状态验证 → 完成或恢复
```

## 项目结构

```text
灵眸智控/
├── visiomind/
│   ├── decision/             # 指令解析、任务分解与 Qwen 计划约束
│   ├── perception/           # 工业目标检测、分割、三维定位与评估
│   └── simulation/           # 工业工位和多格料箱注入
├── voltron/                  # 感知、规划、控制、仿真和闭环运行时
├── training/                 # 数据生成、模型训练和评估
├── scripts/                  # 仿真启动、Qwen 推理和真机接口
├── configs/                  # 工业场景对象绑定
├── models/                   # 项目模型与 LoRA Adapter
├── demo/                     # 工业闭环验证视频
├── run_instruction_demo.py   # 自然语言到仿真执行的统一入口
└── pyproject.toml
```

## 快速开始

推荐环境为 Ubuntu 22.04、Python 3.10 和 NVIDIA GPU。

```bash
git clone https://github.com/Ouniverse-X/VisioMind.git lingmou-zhikong
cd lingmou-zhikong
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

无需 GPU 或仿真器即可验证轻量指令模型：

```bash
python run_instruction_demo.py \
  "现在请把钳子收纳至料箱的第3格，完成后报告状态" \
  --dry-run
```

输出包含意图、目标对象、容器、格位、任务序列和已绑定到仿真实例的 ACTION 序列。

## Qwen 工业规划器

安装大模型推理依赖，并将 `Qwen/Qwen2.5-3B-Instruct` 基础权重放在 `models/base/Qwen2.5-3B-Instruct/`。仓库只分发项目训练的 LoRA Adapter。

```bash
pip install -e '.[llm]'
python scripts/run_qwen_industrial_planner.py \
  "把左侧的扳手放入料箱第2格"
```

可通过 `--base-model` 和 `--adapter` 指定其他本地路径。推理输出必须通过 `visiomind/decision/qwen_plan_schema.py` 定义的 JSON 结构校验。

## 完整仿真闭环

完整运行还需要独立安装 Isaac Sim 4.5、OmniGibson/BEHAVIOR-1K、CuRobo、Nav2 和 AnyGrasp SDK。AnyGrasp 的授权文件及官方权重不随仓库分发。

首次运行前需要修改 `voltron/configs/compact_industrial_pliers_to_toolbox_cell3_i00.json` 中的 BEHAVIOR 场景文件、状态文件和导航图路径。随后配置环境并启动服务：

```bash
export CONDA_SCRIPT=/path/to/miniconda3/etc/profile.d/conda.sh
export VOLTRON_ENV=voltron
export ANYGRASP_PYTHON=/path/to/anygrasp/bin/python
export ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk

./scripts/start_anygrasp_service.sh
./scripts/run_demo.sh \
  "现在请把钳子收纳至料箱的第3格，完成后报告状态"
```

入口会先解析指令，再调用仓库内的 Voltron 运行时执行 `pick_up → place_inside`。最终成功状态由目标释放状态和指定格位安全 AABB 包含关系共同判定。

## 真机接口

底盘网桥将 ROS 2 `cmd_vel` 和升降指令转换为 15 字节串口帧，并发布里程计、关节状态和 TF：

```bash
python scripts/chassis_serial_bridge.py \
  --port /dev/ttyUSB_chassis \
  --baudrate 115200
```

多相机启动文件接入三路 USB 相机和一路深度相机：

```bash
ros2 launch scripts/sensor_integration.launch.py
```

实际部署时应按机器人标定结果修改相机设备号、静态 TF、轮径、轴距和串口协议参数。

## 训练与评估

### 轻量指令模型

```bash
python training/generate_instruction_dataset.py --output-dir data/instructions
python training/train_instruction_model.py \
  --train data/instructions/train.jsonl \
  --test data/instructions/test.jsonl \
  --output models/industrial_instruction.joblib \
  --metrics reports/instruction_model_metrics.json
```

### 工业视觉模型

```bash
pip install -e '.[vision]'
python training/train_industrial_vision_model.py \
  --data-dir data/industrial_vision \
  --epochs 12
```

数据目录不存在时，训练入口会先生成固定随机种子的 RGB-D 工业场景数据。

### Qwen2.5-3B LoRA

```bash
python training/generate_instruction_dataset.py --output-dir data/instructions
python training/generate_qwen_lora_dataset.py
python training/train_qwen_industrial_lora.py
python training/evaluate_qwen_industrial_planner.py \
  --adapter models/qwen25_3b_industrial_lora \
  --output reports/qwen25_lora_metrics.json \
  --predictions reports/qwen25_lora_predictions.jsonl
```

## 已有结果

| 模块 | 数据划分 | 指标 |
| --- | --- | --- |
| 工业视觉模型 | 30 个合成 RGB-D 测试场景 | mAP@0.5 `0.7449`，mAP@0.5:0.95 `0.3854`，3D 定位误差中位数 `2.31 cm` |
| 轻量指令模型 | 396 条留出句式测试 | Accuracy `1.0000`，Macro-F1 `1.0000` |
| Qwen 工业 LoRA | 642 条常规与泛化测试 | JSON Schema 有效率 `1.0000`，意图准确率 `0.9860`，槽位 Micro-F1 `0.9547` |

训练与评估使用固定随机种子的程序化工业数据，数据生成与复现实验命令见上文。

## 模型与许可

- `models/industrial_instruction.joblib`：项目生成的中英文轻量指令模型。
- `models/industrial_part_detector.pt`：项目训练的工业目标检测、分割和三维定位模型。
- `models/qwen25_3b_industrial_lora/adapter_model.safetensors`：基于 Qwen2.5-3B-Instruct 的工业规划 LoRA。
- `models/manifest.json`：模型大小、SHA-256、格式与来源。

Qwen Adapter 受目录内 `QWEN_RESEARCH_LICENSE.txt` 约束，仅用于非商业研究与评估。其他第三方组件及权重说明见 `THIRD_PARTY_NOTICES.md`。
