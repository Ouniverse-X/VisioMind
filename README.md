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

## 代码架构

项目仅使用 `visiomind` 一个 Python 包，其中四个子模块共同组成“灵眸智控”闭环：

- `visiomind.plan`：解析自然语言指令，生成结构化任务与 ACTION 序列。
- `visiomind.perception`：检测工业零件，结合 RGB-D 信息完成分割和三维定位。
- `visiomind.simulation`：注册工业工位、多格料箱和仿真状态验证逻辑。
- `visiomind.action`：调度感知、规划、抓取、导航、放置和失败恢复，对接 AnyGrasp、CuRobo、Nav2 及仿真环境。

`plan` 将指令转换为可执行步骤，`action` 在运行过程中调用 `perception` 和 `simulation`。四者属于同一项目，不是彼此独立的两套代码。

## 入口与调用关系

完整仿真从 `scripts/run_demo.sh` 启动，调用链如下：

```text
scripts/run_demo.sh
└── run_visiomind.py
    ├── visiomind.plan                 指令解析与任务分解
    ├── configs/scene_grounding_industrial.json
    │                                      对象、容器与仿真实例绑定
    ├── --dry-run                       输出计划后结束
    └── visiomind.action                闭环执行入口
        ├── visiomind.perception        目标检测与三维定位
        ├── visiomind.simulation        工位建模与状态验证
        └── AnyGrasp / CuRobo / Nav2   抓取、规划、导航与放置
```

根目录只保留 `run_visiomind.py` 这一个 Python 入口。它先调用 `visiomind.plan`，再以 Python 模块方式进入 `visiomind.action`，因此无需额外的转发脚本。安装项目后还会提供两个模块级命令：`visiomind-plan` 用于单独验证指令解析，`visiomind-action` 用于带明确配置参数的闭环执行。

`scripts/` 中各入口的职责如下：

- `run_demo.sh`：设置运行环境并调用根目录的 `run_visiomind.py`。
- `start_anygrasp_service.sh`：独立启动 AnyGrasp 推理服务，完整仿真前先运行该脚本。
- `run_qwen_industrial_planner.py`：单独运行 Qwen LoRA 规划器并输出 JSON 任务计划，不经过仿真执行链。
- `chassis_serial_bridge.py`：连接真实机器人底盘与升降机构，将 ROS 2 指令转换为串口控制帧。
- `sensor_integration.launch.py`：启动真实机器人使用的 USB 相机、深度相机和静态 TF。

## 项目结构

```text
VisioMind/
├── visiomind/
│   ├── plan/                 # 指令解析、任务分解与 Qwen 计划约束
│   ├── perception/           # 工业目标检测、分割与三维定位
│   ├── simulation/           # 工业工位和多格料箱建模
│   └── action/               # 智能体调度、规划、控制与闭环验证
├── training/                 # 数据生成、模型训练和评估
├── scripts/
│   ├── run_demo.sh           # 完整仿真启动脚本
│   ├── start_anygrasp_service.sh
│   ├── run_qwen_industrial_planner.py
│   ├── chassis_serial_bridge.py
│   └── sensor_integration.launch.py
├── configs/                  # 工业场景对象绑定
├── models/                   # 项目模型与 LoRA Adapter
├── demo/                     # 工业闭环验证视频
├── run_visiomind.py           # 自然语言到仿真执行的统一入口
└── pyproject.toml
```

## 快速开始

推荐环境为 Ubuntu 22.04、Python 3.10 和 NVIDIA GPU。

```bash
git clone https://github.com/Ouniverse-X/VisioMind.git
cd VisioMind
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

无需 GPU 或仿真器即可验证轻量指令模型：

```bash
python run_visiomind.py \
  "现在请把钳子收纳至料箱的第3格，完成后报告状态" \
  --dry-run
```

输出包含意图、目标对象、容器、格位、任务序列和已绑定到仿真实例的 ACTION 序列。

## Qwen 工业规划器

本项目的推理大模型使用经过 LoRA 微调的 Qwen2.5-3B-Instruct 模型，运行前需将其基础权重放至 `models/base/Qwen2.5-3B-Instruct/` 文件夹。项目的 LoRA Adapter 位于 `models/qwen25_3b_industrial_lora/` 目录下。

```bash
pip install -e '.[llm]'
python scripts/run_qwen_industrial_planner.py \
  "把左侧的扳手放入料箱第2格"
```

可通过 `--base-model` 和 `--adapter` 指定其他本地路径。推理输出必须通过 `visiomind/plan/qwen_plan_schema.py` 定义的 JSON 结构校验。

## 完整仿真闭环

完整运行还需要独立安装 Isaac Sim 4.5、OmniGibson/BEHAVIOR-1K、CuRobo、Nav2 和 AnyGrasp SDK。AnyGrasp 的授权文件及官方权重不随仓库分发。

首次运行前需要修改 `visiomind/action/configs/compact_industrial_pliers_to_toolbox_cell3_i00.json` 中的 BEHAVIOR 场景文件、状态文件和导航图路径。随后配置环境并启动服务：

```bash
export CONDA_SCRIPT=/path/to/miniconda3/etc/profile.d/conda.sh
export VISIOMIND_ENV=visiomind
export ANYGRASP_PYTHON=/path/to/anygrasp/bin/python
export ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk

./scripts/start_anygrasp_service.sh
./scripts/run_demo.sh \
  "现在请把钳子收纳至料箱的第3格，完成后报告状态"
```

入口会先解析指令，再调用 `visiomind.action` 执行 `pick_up → place_inside`。最终成功状态由目标释放状态和指定格位安全 AABB 包含关系共同判定。

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
