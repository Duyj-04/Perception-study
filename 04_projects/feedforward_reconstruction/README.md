# 前馈三维重建与 BA 联合优化项目

状态：2026-11-08 新增计划，未登记运行结果。面向用户提供的前馈网络+几何优化 JD，先建立普通 BA 对照，再研究学习与优化协同。

## 基线与研究顺序

1. 选用 [DUSt3R](https://github.com/naver/dust3r)、MASt3R 或 [VGGT](https://github.com/facebookresearch/vggt) 的一个固定官方版本，记录输出坐标、尺度、相机参数、对应及置信度。
2. 先在支持的标定公开多视图场景验证；[VGGT 官方导出流程](https://github.com/facebookresearch/vggt#exporting-to-colmap-format) 提供可选 BA 入口，具体参数以实际版本为准。使用 [Ceres BA 教程](https://ceres-solver.readthedocs.io/latest/nnls_tutorial.html#bundle-adjustment) 理解残差与实现，接入前核实相机模型和参数约定。
3. 冻结输入、划分和计算预算，比较前馈初值、优化前后几何、有效重建支持与成本。重投影误差、相机姿态、深度、几何准确性/完整性分别报告；无真值指标不伪填。
4. 在真实可用数据上做光照、低重叠、动态或噪声分组。庭院/跨季节验证须确有数据，按场景/时段隔离；验证集选择阈值和高价值样本，测试集留作最终评价。
5. 后续只选一个协同机制研究：网络引导初值/权重、可微 BA 或优化结果回馈。明确可训练接口与损失；camera token 回馈是 JD 中的探索方向，不能假定现有模型已支持。

## 最小验收

完成 [前馈+BA 练习](../../03_exercises/08_feedforward_ba/README.md)，提供一组可复查基线与单变量对照、一个失败案例及说明。网络初值可能使 BA 更好，也可能带来错误关联；不预设提升。

可在数据、标定与许可一致时将网络/BA 输出作为 [3DGS 项目](../gaussian_reconstruction/README.md) 的相机初值对照，固定高斯训练条件；渲染收益和几何收益分别评价。与 [MASt3R-SLAM](../mast3r_slam/README.md) 的在线时序任务分开记实验。

## 外部入口与证据

- 代码路径 / commit / checkpoint：
- 图像与相机 / 数据划分 / GT：
- BA 状态、规范固定、残差与初始化：
- 命令 / 协议 / 日志 / 几何与性能结果：
- 个人贡献 / 负结果 / 下一步：
