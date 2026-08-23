# 模型文件说明

`industrial_instruction.joblib` 是本项目可直接分发的中英双语工业指令意图模型，包含字符 1–5 gram TF-IDF 编码器、平衡 Logistic Regression 分类头、标签和训练版本信息。`manifest.json` 记录文件大小与 SHA-256；可用 `sha256sum models/industrial_instruction.joblib` 复核。

AnyGrasp 检测权重没有复制进 Git：SDK 使用机器绑定授权，且单文件约 296 MB。参赛机需要自行申请 AnyGrasp SDK 和许可证，将检测权重置于 `$ANYGRASP_SDK_ROOT/grasp_detection/log/checkpoint_detection.tar`。期望哈希记录在 `third_party_models.json`。不得把本机 `license/`、二进制 SDK 或许可证压缩包提交到公开仓库。

CuRobo、Isaac Sim、OmniGibson 和 BEHAVIOR-1K 的模型/资产随各自环境安装，不属于本项目训练产物。
