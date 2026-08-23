# 使用说明与复现手册

## 1. 已验证环境

- Ubuntu 22.04，NVIDIA Driver 580.65.06；
- RTX 3090 24 GB，主存 32 GB；
- Python 3.10，Voltron 环境 `/mnt/data/huangyixuan/conda_envs/voltron`；
- Isaac Sim 4.5、OmniGibson/BEHAVIOR-1K、CuRobo；
- AnyGrasp 独立 Python 环境和机器授权 SDK。

在其他机器上建议先完成 Isaac/OmniGibson 官方 smoke test，再安装本项目。不要把 HuggingFace、Isaac 或 AnyGrasp 大文件缓存写入根分区。

## 2. CPU 安装与测试

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q tests/test_instruction_model.py
```

竞赛服务器使用既有 Voltron 环境时：

```bash
source /home/huangyixuan/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/huangyixuan/conda_envs/voltron
pytest -q tests
python -m compileall -q visiomind voltron
```

## 3. 模型准备

工业指令权重随仓库提供。执行 `sha256sum models/industrial_instruction.joblib`，结果应与 `models/manifest.json` 一致。

AnyGrasp 必须从官方申请 SDK、机器许可证和检测权重。设置：

```bash
export ANYGRASP_PYTHON=/path/to/anygrasp/bin/python
export ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk
```

权重默认放置到 `$ANYGRASP_SDK_ROOT/grasp_detection/log/checkpoint_detection.tar`。期望大小和哈希见 `models/third_party_models.json`。禁止复制其他机器的许可证。

## 4. 自然语言与训练

只生成计划：

```bash
python run_instruction_demo.py "把半个苹果放进包装箱" --dry-run
python -m visiomind.decision.cli "帮我把左侧的滚柱放到料箱第三格"
```

模型低于默认 0.55 置信度、缺少对象或放置目的地时会拒绝执行。新场景需要编辑 `configs/scene_grounding_*.json`，把本体类别映射为场景唯一实例；禁止用模糊子串自动选择同类物体。

重新生成训练数据和训练命令见 README。固定随机种子和 held-out template 分割已写入脚本；提交新指标时应同时提交数据、模型哈希和混淆矩阵。

## 5. Isaac Demo

终端一启动 AnyGrasp：

```bash
./scripts/start_anygrasp_service.sh
curl -sS http://127.0.0.1:18090/health
```

健康输出必须包含 `detector_loaded=true`。终端二运行自然语言闭环：

```bash
./scripts/run_demo.sh "把半个苹果放进包装箱"
```

也可绕过语言层直接验证固定配置：

```bash
python run_action_only_overlay.py \
  --config voltron/configs/half_apple_to_packing_box_place_inside_i10.json
```

入口会强制设置 `VOLTRON_HOME` 并优先加载本仓库 `voltron/`，不会被 conda activate
hook 中另一个研究目录静默覆盖。运行时日志默认进入本地 vendored runtime 的 runs 目录，
该目录已加入 `.gitignore`。

## 6. 输出与验收

每次运行生成 `process_data.jsonl`、轨迹视频和最终结构化结果。最终验收至少检查：

```text
grasp_success=true
physical_grasp_verified=true
placement_success=true
placement_verified=true
released=true
aabb_contained=true
```

如果 `task_success=true` 但任一物理字段不成立，整次运行仍为失败。Demo 视频应保留自然语言输入、模型解析结果、感知候选、抓取、导航、放置、释放和最终证据字幕。

从一个或多个真实运行生成可提交的精简证据：

```bash
python scripts/collect_real_run_evidence.py /path/to/run_a /path/to/run_b
```

脚本只接受 `action_terminal_success` 中完整的释放/包含门控作为严格成功，输出
`reports/real_isaac_runs.json`、Markdown 表及原始日志/视频 SHA-256。

## 7. 常见故障

- AnyGrasp 连接失败：检查 18090 端口、SDK 路径、机器许可证和权重哈希；
- CuRobo OOM：停止其他 GPU 进程；RTX 3090 上为 Isaac+CuRobo 预留至少约 16 GB；
- 导航在台面拐角停滞：确认 7×7 腐蚀、0.30 m 净空和逐网格航点未被改回稀疏路径；
- 放置规划无解：检查 `preplan_base_pose_world/eef_pose_world`、姿态保持模式和 planning attempts；
- 环境成功但动作失败：以 `placement_verified` 为准，这是已知的 BEHAVIOR 子任务谓词语义差异；
- 退出时状态码 139：当前 Isaac/viewport 在 stage teardown 时可能段错误；若结构化结果和视频已落盘，应单独记录为清理缺陷，不得覆盖任务本身结果。

## 8. 真实硬件接口建议

感知接口应输出时间戳、相机系 6D 位姿、类别、置信度和点云掩膜；机器人接口通过 ROS 2 action 或厂商 API 接收带速度/加速度约束的轨迹；PLC/安全控制器独立管理急停、围栏和互锁。部署前必须完成手眼标定、TCP 标定、碰撞体校核、空载慢速试运行和有载验收。仿真中的 sticky attachment 不可直接等同于真实夹持力闭环。
