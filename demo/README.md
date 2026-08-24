# Isaac Sim Demo

本目录提供了用于评审和验证的 Isaac Sim 仿真运行视频：

1. **工业工件抓取与格位放置演示 (`visiomind_industrial_demo.mp4`)**
   - **展示内容**：系统接收到“现在请把钳子收纳至料箱的第3格”指令后，在混杂零件工位（outfit_a_basic_toolbox 场景）对钳子进行识别、AnyGrasp 6-DoF 抓取、携物 A* 导航以及精确对齐并投放至工具箱第 3 格的物理执行过程。
   - **视频对应运行**：`plier_to_toolbox_cell3_industrial_i00_20260823_210659`
   - **时长**：由 2044 个控制步物理仿真整合而成。
   - **SHA-256**：`0562b653bca699122d200bcea61e26127c5b1982278ae7c6ea174dfafad33849`

2. **午餐盒摆放演示 (`visiomind_isaac_demo.mp4`)**
   - **展示内容**：完整的“半个苹果 -> 包装箱”拾取与箱内放置验证运行。
   - **视频对应运行**：`half_apple_to_packing_box_place_inside_i10_20260823_193048`
   - **编码**：H.264 High Profile Level 3.1，900×450 双视图，25 fps
   - **时长**：52.72 秒
   - **SHA-256**：`c6f5de1ad64431151ac314fe784456455c428c9d8831d0ddfab6883761bea1e9`

视频均不包含 AnyGrasp 权重、许可证、SDK 文件或其他机器隐私凭据。
