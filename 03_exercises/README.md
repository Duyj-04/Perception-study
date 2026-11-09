# 最小算法实验

目的不是堆仓库，而是验证每个关键理论节点。每个练习必须包含：假设、输入、预期、测试、运行命令、失败边界。

建议放可独立运行的小练习；大型训练与真实系统改动仍留在原有项目仓库。

## 练习入口

| 练习 | 要验证的问题 |
|---|---|
| [两视图几何](01_two_view_geometry/README.md) | 位姿、三角化及退化 |
| [BA 与 Jacobian](02_bundle_adjustment/README.md) | 数值优化、规范固定及导数 |
| [IMU 预积分](03_imu_preintegration/README.md) | 偏置修正与重新积分 |
| [性能测量](04_profiling/README.md) | 同步、端到端延迟与资源 |
| [深度与双目/全景](05_depth_estimation/README.md) | 深度含义、尺度、投影与评价 |
| [深度补全](06_depth_completion/README.md) | 缺测、稀疏度、传播与边界 |
| [独立 3DGS](07_3dgs_reconstruction/README.md) | 高斯渲染、已知位姿重建与留出视角 |
| [前馈重建与 BA](08_feedforward_ba/README.md) | 网络初值、几何优化与失败 |

新练习为计划说明，尚无运行结果。实际执行使用可复用环境和官方接口，按记录中的命令验收。
