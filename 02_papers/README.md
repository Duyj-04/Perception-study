# 论文索引与精读方法

1. 打开 `paper_index.csv`，按 `P0 → P1 → P2` 阅读；初始状态是“待核对”，不代表你之前没有读过。
2. 复制 `paper_note_template.md` 到对应子目录，例如 `learning_slam/DROID-SLAM.md`。
3. 每篇论文至少回答：问题、输入输出、状态/损失、网络和几何分工、代码、对照、失败边界。
4. P0 要达到“能推导、能追踪代码、能解释主要实验”，P1 要达到“机制和边界清楚”，P2 先建立技术图谱。
5. 仓库与论文可能更新：正式复现时再核对 commit、checkpoint 和官方依赖。

P0 是各主线的核心，不表示所有论文必须同时精读。按当前岗位选择一条深入，再用小实验连接其他方向：

- 深度/补全：相机与双目几何 → DPT → Depth Anything V2 → 米制模型（如 Metric3Dv2）→ NLSPN；RAFT-Stereo 和 UniFuse 分别支撑双目与全景专题。
- 三维重建：SfM/BA → DUSt3R → MASt3R/VGGT → 网络初值与几何优化；可微 BA 在普通 BA 基线之后。
- 渲染/3DGS：NeRF 基本表示与体渲染 → 原版 3DGS → 密度控制/压缩；Photo-SLAM、MonoGS 等作为在线系统拓展。
- 机器人：VINS-Mono / ORB-SLAM3 / Forster → XFeat → DROID / MASt3R-SLAM。

笔记目录按领域分类，深度新笔记放 `depth_estimation/`，NeRF/3DGS 放 `3dgs_implicit/`。先核对是否已有笔记和实验，阅读状态仍需真实证据更新。
