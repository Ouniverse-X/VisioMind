# 模型文件说明

## 1. 工业指令解析模型 (`industrial_instruction.joblib`)
- **用途**：本项目可直接分发的中英双语工业指令意图与槽位解析模型。
- **架构**：字符 1–5 gram TF-IDF 编码器 + 平衡 Logistic Regression 分类头 + 结构化 JSON Schema 槽位提取器。
- **产物哈希**：`a03c3a50b3bc994813f1758567320255926ff36bcb6261f30b09d068d3d6ff49`。

## 2. 工业零件微调视觉检测与分割模型 (`industrial_part_detector.pt`)
- **用途**：专用于工业零件（螺栓、扳手、滚柱/圆柱零件、螺丝刀、钳子、螺母、螺钉、料箱、工具箱等）的 2D 目标检测、实例掩码分割及 3D 点云与位姿解算。
- **架构**：轻量级多任务 FPN/ConvNet（Stem + Layer1 + Layer2 Backbone + Detection Head + High-Res Mask Decoder），支持多尺度特征融合与 3D 点云空间逆投影。
- **输入输出**：输入 RGB-D 图像帧与相机内外参；输出 2D Bounding Box、分类置信度、二值化实例分割掩码、3D 空间几何中心 (Centroid) 及 3D AABB/OBB 边界框。
- **指标表现**：在固定测试集上 `mAP@0.5 >= 0.85`，3D 空间定位中位误差小于 2.5 cm。
- **产物哈希**：`dba82a9086708558c3e21c1fe9b22dbaec97b7dc18db8e36e7ce0421f68b6857`。

## 3. 第三方模型与环境依赖
AnyGrasp 检测权重没有复制进 Git：SDK 使用机器绑定授权，且单文件约 296 MB。参赛机需要自行申请 AnyGrasp SDK 和许可证，将检测权重置于 `$ANYGRASP_SDK_ROOT/grasp_detection/log/checkpoint_detection.tar`。期望哈希记录在 `third_party_models.json`。不得把本机 `license/`、二进制 SDK 或许可证压缩包提交到公开仓库。

CuRobo、Isaac Sim、OmniGibson 和 BEHAVIOR-1K 的模型/资产随各自环境安装，不属于本项目训练产物。
