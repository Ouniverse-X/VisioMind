# Isaac Sim & 具身智能多智能体仿真 Demo 集

本目录提供了用于评审、合规性检查与算法能力验证的 Isaac Sim / BEHAVIOR 仿真运行轨迹视频集，涵盖工业工具收纳、家庭食品操作、具身 VLA 大模型端到端控制、3D 语义导航及长程跨房间协作：

## 目录清单

### 1. 核心评审演示（工业与家庭基准）

1. **工业工件抓取与格位放置演示 (`visiomind_industrial_demo.mp4`)**
   - **展示内容**：系统接收到“请把混杂工位上的钳子放进工具箱第三格”指令后，在混杂零件工位（`outfit_a_basic_toolbox` 场景）对钳子进行识别、AnyGrasp 6-DoF 抓取、携物安全导航以及精确对齐并投放至工具箱第 3 格的全流程物理执行过程，包含完整的片头说明卡、任务规划时序卡、实时状态机进度条以及终态验收物理证据卡。
   - **视频对应运行**：`plier_to_toolbox_cell3_industrial_i00_20260824_200729`
   - **规格**：720p (1280×720)，H.264 / 30 fps，时长 180.83 秒（3 分钟），大小 40.46 MB
   - **SHA-256**：`18ebed7ba2d53584a4d261019ff6f15ff7cc00809281f165c9224c7b4bcba248`

2. **家庭场景午餐盒摆放演示 (`visiomind_isaac_demo.mp4`)**
   - **展示内容**：完整的“半个苹果 -> 包装箱”拾取、携物导航与箱内放置验证运行。
   - **视频对应运行**：`half_apple_to_packing_box_place_inside_i10_20260823_193048`
   - **规格**：900×450 双视图，H.264 / 25 fps，时长 52.72 秒，大小 6.23 MB
   - **SHA-256**：`c6f5de1ad64431151ac314fe784456455c428c9d8831d0ddfab6883761bea1e9`

---

### 2. 拓展已验证 Demo（操作、VLA、语义导航与长程协作）

3. **半个苹果抓取与原位放置演示 (`half_apple_pick_and_place_demo.mp4`)**
   - **展示内容**：针对半个苹果进行 3D 点云匹配与 AnyGrasp 6-DoF 抓取姿态推断，机械臂完成稳定抬升、空间轨迹运送并在操作台上平稳原位释放放置的连续物理闭环。
   - **视频对应运行**：`half_apple_pick_up_anygrasp_i10_20260820_171822`
   - **规格**：640×256，H.264 High Profile / 10 fps，时长 60.50 秒，大小 2.23 MB
   - **SHA-256**：`4c23315c4028e55b10f491e59f739a0fcd452df9c450b7fc6128a889916bba27`

4. **家庭三明治食品抓取验证 (`club_sandwich_pick_up_demo.mp4`)**
   - **展示内容**：家庭厨房场景下的多物体混杂台面，系统准确锚定三明治目标，生成 6-DoF 稳定抓取并抬升，实现 100% 物理抓取验证（控制步 345 步）。
   - **视频对应运行**：`club_sandwich_pick_up_anygrasp_i20_20260816_195559`
   - **规格**：640×256，H.264 / 10 fps，时长 34.60 秒，大小 1.21 MB
   - **SHA-256**：`a79ac1fab84f81666ec4f770936f0d01ba74d028dcdb3cd35e3951bb65de9bb1`

5. **$\pi_{0.5}$ 具身 VLA 大模型端到端策略控制演示 (`turning_on_radio_pi05_vla_demo.mp4`)**
   - **展示内容**：基于 OpenPI / $\pi_{0.5}$ 扩散策略网络端到端接收视觉输入与自然语言指令，完成精准趋近按键并开启收音机，任务判定 `task_success=True` (100% 进度)。
   - **视频对应运行**：`turning_on_radio_pi05_001_20260402_171139`
   - **规格**：640×256，H.264 / 10 fps，时长 276.10 秒，大小 9.44 MB
   - **SHA-256**：`315b39699aed4e4b9f92e63f1d908a22f26f211e0bba42254ddc93d8212f6310`

6. **3D 开放词表语义多房间导航演示 (`hovsg_multi_room_nav_demo.mp4`)**
   - **展示内容**：结合 HOV-SG（分层开放词表 3D 场景图）与 Nav2 避障规划，在包含家具杂乱、狭窄门廊和推拉门的复杂家庭环境中进行语义目标定位与长程自主穿梭寻路。
   - **视频对应运行**：`hovsg_nav_bathroom_nav2_waypoint_all_doors_open_sliding_full_bathroom_door_wider_open_consistent_graph_clutter_bedroom_corner_test_i00_20260327_115007`
   - **规格**：1024×256 多视角拼合，H.264 / 10 fps，时长 164.80 秒，大小 9.04 MB
   - **SHA-256**：`2b9a32890d1beb9b690bf2dca221acd6f696156585830b07f31993cec66bd557`

7. **长程跨房间杂货搬运闭环演示 (`carrying_groceries_long_horizon_demo.mp4`)**
   - **展示内容**：多智能体端到端协同执行长程家务任务（抓取购物袋 $\rightarrow$ 跨越玄关和走廊 $\rightarrow$ 导航至厨房台面准备归位），展示多阶段复杂任务规划与状态机流转。
   - **视频对应运行**：`main_carrying_in_groceries_openpi_comet_i00_20260707_114239`
   - **规格**：900×360，H.264 / 10 fps，时长 552.10 秒，大小 21.32 MB
   - **SHA-256**：`ede50527c867b264eaac84c870e9509e434dc6027006ffdb2ef96674d618074b`

---

所有视频均不包含 AnyGrasp 权重、许可证、SDK 文件或其他机器隐私凭据。
