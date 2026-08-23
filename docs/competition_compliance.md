# XH-202607 赛题符合性矩阵

| 赛题要求/评分点 | 实现位置 | 验收证据 | 状态 |
|---|---|---|---|
| 工业场景与数据集 | `data/instructions`、BEHAVIOR 配置 | 固定 split、场景实例配置 | 已实现指令数据；工业视觉标注集待扩充 |
| 多物体识别与 3D 定位 | `voltron/integrations/manipulation/anygrasp` | 目标掩膜点云、候选 6D 位姿、锚定统计 | 已实现 |
| 自然语言理解 | `visiomind/decision` | 意图概率、对象/容器/格位/空间槽位 | 已实现 |
| 任务序列分解 | `InstructionPlan.task_sequence` | 感知—规划—抓取—验证—放置—恢复序列 | 已实现 |
| 智能体集成 | ACTION skill、overlay runner | 单条指令驱动两步运行 | 已实现 |
| 抓取和指定格位放置 | AnyGrasp skill、place executor | 身份/抬升/attachment/释放/AABB | 单容器严格闭环已验证；多格几何待完成 |
| 失败感知与重新摆放 | typed failure evidence、recover step | failure phase 与 retry 决策 | 框架已实现，需完成多轮统计 |
| 工业场景模型训练/微调 | `training/`、`models/` | 权重、哈希、99.11%/98.97% | 已实现语言意图模型 |
| 感知精度与效率 | AnyGrasp 审计 | 候选数和运行耗时 | 真实工业标注集 mAP 尚待补测 |
| 执行成功率 | structured physical evidence | `reports/real_isaac_runs.*`：2 次严格成功 + 1 次安全拒绝 | 已有工程回归证据；正式统计集待扩充 |
| 仿真验证视频 | 轨迹录制器、`demo/` | 52.7 秒 H.264 字幕版 Demo | 已完成 |
| 技术报告 | `docs/technical_report.md` | 本文件映射 | 已实现初版 |
| 使用说明 | `docs/user_guide.md` | CPU/GPU/训练/运行命令 | 已实现 |
| 真实机械臂案例 | 真实接口建议 | ROS 2/相机接口说明 | 未完成，不宣称 |

该矩阵刻意区分“已实现”“验证中”和“未完成”，避免用仿真环境谓词或少量工程复测替代评分要求中的正式统计证据。
