# 使用说明与复现手册（赛题第四部分：使用说明文档）

本分册为挑战杯“XH-202607 工业环境下物体感知识别与指令交互型智能体研发”赛题需提交的**第四部分使用说明文档**。本手册详细规定了本智能体系统（VisioMind / Voltron 运行时）的软硬件环境配置、依赖安装、运行步骤、实际机器人硬件配置、通信接口定义以及系统集成方式。

---

## 1. 硬件配置与系统集成方式

VisioMind 系统在实际部署中采用“主控算力中心 + 分布式传感器 + 总线通信执行机构”的具身智能机器人硬件集成架构。

```text
               +----------------------------------------+
               |         IPC 主控计算中心 (Ubuntu)       |
               |                                        |
               |   +--------------+  +---------------+  |
               |   |  VisioMind   |  |    Voltron    |  |
               |   |  意图与语义  |  |  多智能体内核 |  |
               |   +--------------+  +---------------+  |
               +---/---------------\---------\--------\--+
                  /                 \         \        \
  USB (V4L2)     /        ROS 2      \  ROS 2  \        \ 串口 (TTL/RS232)
                /                     \         \        \
               v                       v         v        v
         +------------+         +------------+ +--------+ +-------------+
         |  3x USB    |         | 1x Aurora  | |OpenArm | |  全向底盘    |
         |  头部/双臂  |         |  930 深度  | | 双臂   | |  & 升降台   |
         |  RGB 相机  |         |  胸部相机  | | (ROS2) | |  (串行协议)  |
         +------------+         +------------+ +--------+ +-------------+
```

### 1.1 传感器系统配置与数据流
系统配置有 4 个相机，用于实现全局环境感知、目标三维定位与局部操作闭环引导：
1. **3x 头部与双臂 RGB 相机**：
   - **硬件型号**：Microdia USB 2.0 Camera
   - **设备识别码 (USB ID)**：`0c45:636b`
   - **部署位置**：头部部署 1 个（用于全局视野与目标识别），左右机械臂的末端/小臂上各部署 1 个（用于近场抓取引导与对齐微调）。
   - **数据与集成方式**：通过 USB 2.0 接口直接连入主控 IPC，采用 V4L2 协议，在 Python 中通过 OpenCV 库访问 `/dev/video*` 设备节点直接传回 raw 画面帧。
2. **1x 胸部深度相机**：
   - **硬件型号**：Linux Foundation Aurora 930
   - **设备识别码 (USB ID)**：`3251:1930`
   - **部署位置**：机器人胸部中央（用于场景深度信息获取及建立三维避障/抓取点云）。
   - **数据与集成方式**：通过 USB 接口与 IPC 通信，但在软件集成上，其数据通过运行 ROS 2 驱动节点（如 realsense2_camera 兼容包或专用 aurora_depth_camera_node）进行读取，直接订阅 ROS 2 发布的图像与三维点云 Topic：
     - 深度图像 Topic：`/camera/depth/image_rect_raw`
     - RGB 彩色图像 Topic：`/camera/color/image_raw`
     - 稠密三维点云 Topic：`/camera/depth/color/points`

### 1.2 机械臂系统配置与 ROS 2 通信
本系统集成森之高科 OpenArm 旗舰版（ASR-01）双臂系统，具备 14 自由度（单臂 7 自由度，采用类人 S-R-S 构型）。
- **执行器与关节**：关节采用达妙准直驱电机 (QDD)，具备低阻抗和高反驱特性。
- **物理接口与通信**：机械臂控制器通过 CAN/CAN-FD 总线进行物理级通信，波特率为 1Mbps。
- **系统集成与 ROS 2 接口**：
  IPC 主控运行 ROS 2 Humble，并通过 `ros2_control` 框架进行系统集成（参考官方文档：`https://docs.openarm.dev/1.0/software/ros2/control/`）。
  - **关节状态反馈**：订阅 `/joint_states` topic 实时获取机械臂 14 个关节的位置、速度与力矩反馈。
  - **轨迹规划与执行接口**：通过 ROS 2 Action 接口与 `FollowJointTrajectory` 服务端进行异步通信：
    - 左臂轨迹控制：`/left_arm_controller/follow_joint_trajectory` (类型: `control_msgs/action/FollowJointTrajectory`)
    - 右臂轨迹控制：`/right_arm_controller/follow_joint_trajectory`
  - **夹爪控制器接口**：采用 ROS 2 action 或 topic 控制单自由度夹爪开合：
    - `/left_gripper_controller/gripper_cmd` (类型: `control_msgs/action/GripperCommand`)
    - `/right_gripper_controller/gripper_cmd`

### 1.3 移动底盘与升降机构串口协议
移动底盘采用全向移动底盘（支持三轮/四轮全向移动，底盘名义负载 ~100kg），并配有升降范围约为 500mm 的线性升降台。
- **物理接口**：IPC 与底盘主控之间通过串口（Serial 通信）连接，串口设备节点通常挂载为 `/dev/ttyUSB_chassis`（或 `/dev/ttyUSB0`），硬件波特率固定为 **115200**。
- **通信特征**：串口发送间隔为 **20ms** 准定时发送。
- **底盘串口控制协议定义（根据《02 串口控制底盘协议(1).doc》）**：
  底盘控制数据帧采用固定 **15 字节**（15 Bytes）的帧结构，具体字节定义如下表所示：

| 字节序号 (Byte) | 字段名称 | 数据类型 / 取值范围 | 说明 |
| :--- | :--- | :--- | :--- |
| **Byte 0 - 1** | **包头** | 2 字节 (0xA5 0x5A) | 帧起始同步标志，固定为 `0xA5 0x5A` |
| **Byte 2** | **工作模式** | 1 字节 | `0x00`：速度模式（当前实现及使用模式）<br>`0x01`：位置模式 |
| **Byte 3** | **升降台/车轮方向** | 1 字节 (按位定义) | 字节各 Bit 代表电机方向，`0`为正向/升/前，`1`为反向/降/后。<br>**Bit 0**：A 电机方向（0向前，1向后）<br>**Bit 1**：B 电机方向（0向前，1向后）<br>**Bit 2**：C 电机方向（0向前，1向后）<br>**Bit 3**：D 电机方向（0向前，1向后）<br>**Bit 4**：升降台电机方向（0升，1降） |
| **Byte 4 - 5** | **A电机速度** | 2 字节 (高字节在前) | 轮 A 转速，无符号整型。数值 1 = 0.1 转/秒 |
| **Byte 6 - 7** | **B电机速度** | 2 字节 (高字节在前) | 轮 B 转速，无符号整型。数值 1 = 0.1 转/秒 |
| **Byte 8 - 9** | **C电机速度** | 2 字节 (高字节在前) | 轮 C 转速，无符号整型。数值 1 = 0.1 转/秒 |
| **Byte 10 - 11** | **D电机速度** | 2 字节 (高字节在前) | 轮 D 转速，无符号整型。数值 1 = 0.1 转/秒 |
| **Byte 12 - 13** | **升降台电机速度** | 2 字节 (高字节在前) | 升降台电机转速，无符号整型。数值 1 = 0.1 转/秒 |
| **Byte 14** | **累加和低字节** | 1 字节 (校验和) | **字节 0-13 的累加和的低字节**（即 Sum % 256）。<br>*特殊规定*：若将此字节直接写入 `0x00`，系统将进入**调试模式**，底盘忽略校验检查直接执行该帧。 |

- **车辆运动方向与 Byte 3 方向字定义关系举例**：
  在车轮 A, B 为左侧，C, D 为右侧时，常用运动方向对应的 Byte 3 状态如下：
  - **小车前进**：`0x00`（所有车轮正转）
  - **小车后退**：`0x0F`（所有车轮反转，Bit 0-3 均为 1）
  - **原地左转**：`0x0C`（左轮反转，右轮正转；Bit 2-3 为 1，即 C、D 电机向后）
  - **原地右转**：`0x03`（左轮正转，右轮反转；Bit 0-1 为 1，即 A、B 电机向后）
  - **水平左平移**：`0x05`（车轮对角差动，Bit 0, 2 为 1，即 A、C 电机向后）
  - **水平右平移**：`0x0A`（车轮对角差动，Bit 1, 3 为 1，即 B、D 电机向后）

- **控制报文 16 进制帧指令示例**（以调试模式为例，校验和 Byte 14 设为 `0x00`）：
  - **整车停止**：
    `a5 5a 00 00 00 00 00 00 00 00 00 00 00 00 00`
  - **小车前进（速度 0.1 转/秒）**：
    `a5 5a 00 00 00 01 00 01 00 01 00 01 00 00 00`
  - **小车后退（速度 0.1 转/秒）**：
    `a5 5a 00 0f 00 01 00 01 00 01 00 01 00 00 00`
  - **原地右转**：
    `a5 5a 00 03 00 01 00 01 00 01 00 01 00 00 00`
  - **原地左转**：
    `a5 5a 00 0c 00 01 00 01 00 01 00 01 00 00 00`
  - **水平向右平移**：
    `a5 5a 00 0a 00 01 00 01 00 01 00 01 00 00 00`
  - **水平向左平移**：
    `a5 5a 00 05 00 01 00 01 00 01 00 01 00 00 00`

---

## 2. 软件运行栈与已验证环境

本系统软件栈已在以下配置的计算节点中经过了严格闭环测试：

- **操作系统**：Ubuntu 22.04 LTS (Ubuntu 20.04 亦部分兼容)
- **显卡驱动**：NVIDIA Driver 580.65.06 (或 >= 525.60.13 的任何显卡驱动)
- **物理硬件**：NVIDIA RTX 3090 (显存 24 GB) 或以上规格，系统主内存 32 GB 以上
- **物理机要求**：仿真运行与运动规划（OmniGibson 与 CuRobo）在运行中显存占用约为 **15.5 GB**，AnyGrasp 独立常驻约 **0.5 GB**。
- **核心软件环境**：
  - Python 3.10
  - ROS 2 Humble
  - NVIDIA Isaac Sim 4.5
  - OmniGibson / BEHAVIOR-1K
  - CuRobo (无碰全身规划库)
  - AnyGrasp SDK (三维几何抓取候选过滤)

---

## 3. 依赖安装步骤

整个系统的环境搭建分为核心系统（Conda 虚拟环境）、仿真底座（Isaac Sim 与 OmniGibson）、规划执行器（CuRobo）以及感知系统（AnyGrasp）四个部分。

### 3.1 创建与配置 conda 虚拟环境
1. 激活 Conda 初始化路径（请替换为您的实际 Conda 安装路径）：
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   ```
2. 使用提供的 `environment.yml`（在 `voltron` 或项目根目录下）创建专用虚拟环境：
   ```bash
   conda env create -f environment.yml -p /path/to/your/conda_envs/voltron
   conda activate /path/to/your/conda_envs/voltron
   ```
3. 在开发模式下以可编辑状态安装核心仓库（`voltron` 和 `hems` 记忆库）：
   ```bash
   # 安装 HEMS 记忆模块
   cd hems
   pip install -e ".[dev]"
   
   # 安装 Voltron 控制平面
   cd ../voltron
   pip install -e .
   ```

### 3.2 仿真底座环境配置
1. **安装 NVIDIA Isaac Sim 4.5**：按照 NVIDIA Omniverse 官方指导安装 Launcher 并下载 Isaac Sim 4.5，记录安装路径。
2. **下载并软链 BEHAVIOR 仿真数据集**：
   仿真资产及数据集需下载至容量充足的数据盘（切勿写入系统根分区 `/` 导致磁盘爆满）。
   ```bash
   export OMNIGIBSON_DATA_PATH=/path/to/your/data_disk/BEHAVIOR-1K/datasets
   export OMNIGIBSON_APPDATA_PATH=/path/to/your/data_disk/.cache/omnigibson
   ```
3. **软链接 HuggingFace 缓存目录**：
   将 HuggingFace 的模型缓存软链至数据盘，防根盘爆满（请替换为您的实际数据盘缓存路径）：
   ```bash
   ln -s /path/to/your/data_disk/.cache/huggingface ~/.cache/huggingface
   ```

### 3.3 安装 CuRobo 无碰撞轨迹规划库
CuRobo 需要特定版本的 PyTorch 和 CUDA 运行时支持。
1. 在激活的 conda 环境中安装配套依赖：
   ```bash
   pip install curobo --extra-index-url https://pypi.nvidia.com
   ```
2. 运行 CuRobo 自检以确保其能够正确调用 GPU 硬件加速：
   ```bash
   python -c "import curobo; print('CuRobo version:', curobo.__version__)"
   ```

### 3.4 配置 AnyGrasp 几何抓取感知服务
AnyGrasp 属于类别无关的 6-DoF 抓取姿态估计系统，通过独立的 gRPC/HTTP 服务与 Voltron 进行隔离通信。
1. 申请 AnyGrasp 官方 SDK 及其物理机器绑定的运行许可证（`license.lic`）。
2. 在环境变量中声明 AnyGrasp 的 SDK 根路径与运行 Python 环境：
   ```bash
   export ANYGRASP_PYTHON=/path/to/anygrasp_env/bin/python
   export ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk
   ```
3. 将 AnyGrasp 检测权重下载并放置于 `$ANYGRASP_SDK_ROOT/grasp_detection/log/checkpoint_detection.tar`，并在 `models/third_party_models.json` 中比对 SHA-256 哈希值确认其完整性。

---

## 4. 系统集成运行步骤

整个系统的启动和联调流程在物理机和仿真环境中存在对应关系，请务必保证启动顺序。

### 4.1 离线算法与单元测试
在不启动 Isaac 物理渲染器的情况下，可以使用 CPU 环境对意图理解模型和回归用例进行测试。
1. **运行全套单元测试**：
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate /path/to/your/conda_envs/voltron
   pytest -q tests
   ```
2. **对自然语言指令模型进行一键测试 (Dry Run)**：
   测试语言解析器（visiomind）对自然语言的槽位抽取与任务序列生成：
   ```bash
   python run_instruction_demo.py "把半个苹果放进包装箱" --dry-run
   # 或者测试更复杂的工件交互指令：
   python -m visiomind.decision.cli "帮我把左侧的滚柱放到料箱第三格"
   ```

### 4.2 仿真闭环系统运行 (Isaac Sim + OmniGibson)
在测试仿真场景时，请按照以下步骤分别开启后端服务和闭环智能体主进程：
1. **第一步：启动 AnyGrasp 独立感知服务**：
   ```bash
   cd xh/competition_code
   ./scripts/start_anygrasp_service.sh
   # 验证感知服务健康状态：
   curl -sS http://127.0.0.1:18090/health
   ```
   *预期健康输出必须包含：`{"status":"ok", "detector_loaded":true}`*。
2. **第二步：运行自然语言驱动的仿真闭环主进程**：
   ```bash
   ./scripts/run_demo.sh "把半个苹果放进包装箱"
   ```
3. **第三步（可选）：直接验证固定的任务实例**：
   若跳过语言理解层直接验证执行层与 CuRobo 放置，可输入任务 JSON 配置文件：
   ```bash
   python run_action_only_overlay.py \
     --config voltron/configs/half_apple_to_packing_box_place_inside_i10.json
   ```

### 4.3 真实物理机器人启动与集成步骤
若要在真实物理 ASR-01 机器人上部署 VisioMind，必须严格遵循以下系统集成服务启动顺序：
1. **步骤 1：激活物理网络总线与底盘串口**：
   - 插入底盘串口线（确认 `/dev/ttyUSB_chassis` 可读写权限：`sudo chmod 666 /dev/ttyUSB_chassis`）。
   - 开启 CAN-to-USB 控制卡（例如使用 `ip link set can0 up type can bitrate 1000000` 激活底盘及双臂 CAN 总线）。
2. **步骤 2：启动底盘 ROS 2 节点**：
   - 运行已实现的底盘串口网桥脚本 `scripts/chassis_serial_bridge.py`。该网桥自动订阅 `/cmd_vel` 与 `/lift_cmd` 话题，在底层解析麦克纳姆轮运动学，并将其打包为 15 字节串口帧，以 20ms 的间隔定时发送给底盘物理设备（`/dev/ttyUSB_chassis`），同时以 50Hz 频率读取反馈数据，发布 `/odom` 里程计与电梯关节状态 `/joint_states`。
3. **步骤 3：启动 OpenArm 关节控制器**：
   - 运行 `openarm_hardware` 节点与 ROS 2 Humble 的 `controller_manager`。
   - 激活 `/left_arm_controller` 与 `/right_arm_controller` 的 `FollowJointTrajectory` 动作服务器。
4. **步骤 4：启动相机传感器系统**：
   - 运行已实现的传感器集成启动文件 `scripts/sensor_integration.launch.py`。该 Launch 文件会一键拉起头部及双臂上的 3 个 USB 相机（`Microdia 0c45:636b`，基于 `usb_cam` 节点），以及胸部的深度相机（`Aurora 930 3251:1930`，基于 `realsense2_camera` 节点），开始广播彩色图像、深度图像和 `/camera/depth/color/points` 稠密三维彩色点云，并发布静态相机位姿变换。
5. **步骤 5：启动 HEMS 记忆数据库与 AnyGrasp 感知后台**：
   - 启动 HEMS Unified Memory System 本地独立 RPC 服务（占用 8070 端口）。
   - 启动 AnyGrasp 6-DoF 抓取推断服务（占用 18090 端口）。
6. **步骤 6：启动 Voltron Brain 主调度控制平面**：
   - 启动 `VoltronOrchestrator` 主进程。此时向主进程发送自然语言指令（例如“把圆柱滚柱放到料箱第二格”），系统将依据实时订阅的 ROS 2 点云与图像完成闭环作业。

---

## 5. 常见故障排查 (Troubleshooting)

| 故障现象 | 根本原因分析 | 推荐解决方案 |
| :--- | :--- | :--- |
| **AnyGrasp 连接失败 / `detector_loaded=false`** | 1. 18090 端口被占用。<br>2. 机器许可证 `license.lic` 与当前主板 MAC 地址不匹配。<br>3. 权重文件不存在或损坏。 | 1. 使用 `lsof -i:18090` 检查端口，杀掉冲突进程。<br>2. 重新向官方申请匹配当前硬件 MAC 的许可证。<br>3. 检验权重 MD5 值是否符合 `models/third_party_models.json` 要求。 |
| **CuRobo 运动规划器报 Out of Memory (OOM)** | NVIDIA RTX 3090 显存不足。<br>常见于 Isaac Sim 渲染与 CuRobo 并发占用。 | 1. 关闭占用显存的其它大型进程（如外部 PyTorch 训练、多余的仿真 Viewport 窗口）。<br>2. 修改 curobo 规划配置文件，降低轨迹采样的 attempts 数量，或在 headless 模式下运行仿真以释放显存。 |
| **机器人底盘在接近目标台面或料箱时卡死** | Nav2 的二维代价地图将料箱外壁误判为障碍物；或者 A* 规划路径的碰撞安全容差过高。 | 1. 检查 `configs/` 中 `scene_state_include_aabb` 是否设为 `true`。<br>2. 确认地图使用了不包含动态障碍物的 objectless 底图 `floor_trav_no_obj_*.png`。<br>3. 调整 Voltron 导航配置中的避障膨胀核腐蚀参数（建议 7x7 腐蚀，保持 0.30m 净空）。 |
| **末端执行器到达放置点上方，但 CuRobo 规划无解** | 机械臂规划时手臂关节限位发生奇异，或者目标放置格位被容器网格保守化成封闭的刚体。 | 1. 检查末端执行器的安全接近角度是否正确（避免无意义的腕部扭转）。<br>2. 启用 `semantic_top_entry` 功能，跳过容器侧壁的过保护碰撞检查，仅在重力下探阶段对料箱实际物理边界进行硬性避障。 |
| **仿真环境显示 task_success=true，但物体实际未落入指定格位** | BDDL 任务谓词采用了宽松的空间包围盒检测，未能感知物体在下落瞬间反弹、倾倒或卡在隔板边缘。 | 1. 必须以 VisioMind 的“物理双证据”为验收基准，禁用仅依靠环境谓词结束任务的机制。<br>2. 调整释放高度和微调速度，并在结构化输出中核对 `placement_verified=true` 与 `aabb_contained=true`。 |
| **退出 stage 时 Python 报段错误 (Exit Code 139)** | Isaac Sim 视口渲染器（Viewport）或 USD 场景 Teardown 阶段在个别 GPU 驱动下触发显存非法访问。 | 若 `process_data.jsonl` 与运行视频已正常落盘，该段错误属于环境清理层面的次要缺陷，系统日志已将其归类为 `cleanup_errors`，不记作任务执行失败。 |
