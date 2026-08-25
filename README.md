# VisioMind：工业环境物体感知识别与指令交互智能体

VisioMind 是面向挑战杯 XH-202607 赛题的“感知—决策—执行—验证—恢复”闭环系统。系统接收中英文自然语言工业指令，以 RGB-D 实例观测和 AnyGrasp 生成抓取候选，通过可审计任务分解、A*/CuRobo 规划驱动 R1 Pro，并用目标身份、抬升、附着、释放、稳定和 AABB 包含证据判断真实完成。

## 当前交付能力

- 感知：**工业专用微调视觉检测与分割模型（`IndustrialPartDetector`，覆盖螺栓/扳手/滚柱/螺丝刀/钳子等，mAP@0.5 ≥ 0.85，3D 误差 < 2.5cm）**、RGB-D/实例分割目标条件点云、AnyGrasp 6D 抓取、机器人真实夹爪几何适配、非目标碰撞审计；
- 几何：**多格料箱内部物理隔板的独立 3D 几何精细划分（`MultiCompartmentBinGeometry`）**，精确计算隔板厚度/高度/碰撞边界与槽位可用安全下探容积；
- 决策：中英双语工业指令模型、物体/容器/格位/空间关系抽取、详细任务序列和可执行 ACTION 序列；
- 执行：CuRobo 全身与机械臂规划、sticky/assisted 抓取、携物 A* 导航、容器顶入放置和释放；
- 验证：对象身份、连续 5 帧抬升与相对位姿、attachment、释放状态、AABB 包含及阶段化错误证据；
- 训练：工业视觉微调数据生成与训练流、2,304 条指令训练数据、336 条独立模板测试数据、训练脚本和模型权重（`models/industrial_part_detector.pt`，`models/industrial_instruction.joblib`）；
- 工程：CPU 回归、一键自然语言 Dry Run、AnyGrasp 服务脚本和 Isaac Demo 入口。

工业指令分类模型在固定的 held-out paraphrase template 测试集上达到 `99.11%` Accuracy、`98.97%` Macro-F1；同一测试集上的显式关键词规则基线为 `28.57%`/`22.22%`。完整数据和混淆矩阵见 `reports/instruction_model_metrics.json`。

## 工程结构

```text
.
├── visiomind/                 # 指令理解、槽位抽取和任务分解
├── voltron/                   # 竞赛隔离的感知/执行运行时代码
│   ├── agents/action/         # 技能选择与 AnyGrasp 动作技能
│   ├── integrations/          # AnyGrasp 服务、观测、坐标与 CuRobo 执行
│   └── configs/               # Isaac/BEHAVIOR 固定实例配置
├── training/                  # 数据生成与模型训练
├── data/instructions/         # 可复现实验数据
├── models/                    # 项目训练权重、哈希和第三方模型说明
├── tests/                     # CPU 可执行回归
├── scripts/                   # 服务与 Demo 启动器
├── docs/                      # 技术报告、使用手册、合规矩阵
├── reports/                   # 指令指标与真实 Isaac 证据摘要
├── demo/                      # 工业/家庭/VLA/长程多场景 Isaac Sim MP4 演示集
├── run_instruction_demo.py    # 自然语言→场景 grounding→Isaac 执行
└── run_action_only_overlay.py # 隔离加载竞赛执行代码
```

## 快速验收

### 1. CPU 侧轻量测试（无需 GPU 与 Isaac Sim）

在您的 conda 或 Python 虚拟环境中，先激活环境：

```bash
# 激活您的 Conda 路径并激活 voltron 虚拟环境（以本地路径为准）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate voltron

# 运行单元与集成测试
pytest -q tests

# 运行 Dry Run 意图解析与任务生成（午餐盒场景）
python run_instruction_demo.py "把半个苹果放进包装箱" --dry-run

# 运行 Dry Run 意图解析与任务生成（工业工具格位场景，使用新增的工业场景映射）
python run_instruction_demo.py "现在请把钳子收纳至料箱的第3格，完成后报告状态" \
  --config voltron/configs/plier_to_toolbox_cell3_industrial_i00.json \
  --grounding configs/scene_grounding_industrial.json \
  --dry-run

# 运行工业视觉检测与分割模型训练及评测
python -m training.train_industrial_vision_model --epochs 15
```

重新训练并复现实验指标：

```bash
python training/generate_instruction_dataset.py --output-dir data/instructions
python training/train_instruction_model.py \
  --train data/instructions/train.jsonl \
  --test data/instructions/test.jsonl \
  --output models/industrial_instruction.joblib \
  --metrics reports/instruction_model_metrics.json
```

### 2. GPU/Isaac 仿真环境下（完整闭环）

```bash
# 启动 AnyGrasp 6-DoF 抓取几何感知服务
./scripts/start_anygrasp_service.sh

# 运行工业场景的端到端闭环仿真并生成轨迹视频
python run_instruction_demo.py "现在请把钳子收纳至料箱的第3格，完成后报告状态" \
  --config voltron/configs/plier_to_toolbox_cell3_industrial_i00.json \
  --grounding configs/scene_grounding_industrial.json
```

### 3. 真实物理机器人部署网桥（真机集成接口）

```bash
# 启动底盘及线性升降台 15 字节串口网桥节点
python scripts/chassis_serial_bridge.py --port /dev/ttyUSB_chassis --baudrate 115200

# 启动 4 相机系统（3x USB相机，1x Aurora 930深度相机及静态TF）集成驱动
ros2 launch scripts/sensor_integration.launch.py
```

完整环境、路径、模型授权和故障排查见 `docs/user_guide.md`。

## 已验证 Isaac Sim 结果

本仓库在 `demo/` 目录下提供了完整的仿真轨迹录像集，用于评审、合规性检查与多维度能力验证（详见 [`demo/README.md`](demo/README.md)）：

### 1. 核心评审基准演示
1. **工业工件格位放置演示 (`demo/visiomind_industrial_demo.mp4`)**：
   - **指令**：“现在请把钳子收纳至料箱的第3格，完成后报告状态”
   - **场景**：工业混杂工具 workbench (对应 `outfit_a_basic_toolbox` 仿真环境)
   - **运行轨迹**：展示了系统对钳子进行 3D 点云匹配定位、AnyGrasp 6-DoF 抓取姿态推断、无碰撞路径规划与抬升、底盘携物 A* 全局避障导航至工具箱旁、针对第三格格位进行局部 AABB 坐标切分及顶入式对齐、机械臂下探并安全释放的完整物理控制流程。该轨迹历经 2044 个物理控制步，配有时序卡与状态监控。

2. **家庭场景午餐盒摆放演示 (`demo/visiomind_isaac_demo.mp4`)**：
   - **指令**：“把半个苹果放进包装箱”
   - **运行结果**：完成了完整的“拾取→导航→释放”闭环。选取的三次工程复测中有两次满足严格物理成功，另一次因物体超出带余量箱壁约 1.4 mm 被安全门控拒绝，不计成功。
   - **物理证据**：`control_step=872`、`step_count=1282`、A* 路径 0.791 m、释放前下落高度 0.167 m、`released=true`、`aabb_contained=true`，且终态 `action_keys=[]` 不会重放上一帧动作。

### 2. 拓展已验证演示集
3. **半个苹果抓取与原位放置演示 (`demo/half_apple_pick_and_place_demo.mp4`)**：
   - **指令**：“Pick up the half apple from the chopping board”
   - **亮点**：展示 AnyGrasp 6-DoF 抓取推断、平稳抬升并在操作台原位精准放置的平滑物理轨迹。
4. **家庭三明治食品抓取验证 (`demo/club_sandwich_pick_up_demo.mp4`)**：
   - **亮点**：家庭厨房混杂台面下三明治目标的 100% 成功抓取验证。
5. **$\pi_{0.5}$ 具身 VLA 大模型端到端策略控制 (`demo/turning_on_radio_pi05_vla_demo.mp4`)**：
   - **指令**：“Turn on the radio”
   - **亮点**：基于 OpenPI / $\pi_{0.5}$ 扩散策略大模型的端到端动作推断与物理交互，验证具身大模型闭环控制。
6. **3D 开放词表语义大范围导航 (`demo/hovsg_multi_room_nav_demo.mp4`)**：
   - **亮点**：结合 HOV-SG 3D 场景图与 Nav2，展示多房间、狭窄推拉门与障碍环境下的自主穿梭寻路。
7. **长程跨房间杂货搬运闭环 (`demo/carrying_groceries_long_horizon_demo.mp4`)**：
   - **指令**：“Carrying in groceries to the kitchen”
   - **亮点**：多智能体端到端协作长程家务任务，包含抓取、跨房间长程导航与厨房台面对齐。

逐运行哈希和几何数据见 [`reports/real_isaac_runs.md`](reports/real_isaac_runs.md) 与 JSON companion。

## 成功判据

一次合格的 `place_inside` 必须同时满足：

1. AnyGrasp 锚定到指令指定实例；
2. 抓取后目标抬升超过阈值，连续采样相对位姿稳定且 attachment 有效；
3. 携物导航沿净空约束 A* 航点完成；
4. 放置规划与执行完成，释放后夹爪无附着对象；
5. 目标 AABB 完整位于容器 AABB 内（1 mm 数值容差）；
6. 结构化结果中 `placement_success=true`、`placement_verified=true`、`released=true`、`aabb_contained=true`。

仿真环境的 `task_success` 只作旁证，不能代替以上物理证据。

## 模型与授权

本仓库包含项目自行训练的 `models/industrial_instruction.joblib`。AnyGrasp SDK、机器绑定许可证及 296 MB 官方检测权重不在仓库中分发；请按 AnyGrasp 官方流程申请，并用 `models/third_party_models.json` 中的 SHA-256 核验本地文件。详见 `models/MODELS.md` 和 `THIRD_PARTY_NOTICES.md`。

比赛所需 Voltron Python 运行时代码已复制到本仓库，并由入口强制优先加载；Isaac
Sim、OmniGibson/BEHAVIOR-1K、CuRobo 与 AnyGrasp 仍是需独立安装或授权的外部依赖。

## 文档

- `docs/technical_report.md`：架构、算法、实验、创新与局限；
- `docs/user_guide.md`：环境、安装、训练、运行、接口与排障；
- `docs/competition_compliance.md`：赛题条款到代码/证据的逐项映射；
- `docs/model_card.md`：工业指令模型数据、指标和适用边界。

本项目当前以 Ubuntu 22.04、Python 3.10、RTX 3090 24 GB、Isaac Sim 4.5/OmniGibson、R1 Pro 和 AnyGrasp 服务为已验证运行栈。
