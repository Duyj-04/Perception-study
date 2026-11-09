# 深度估计与补全

本专题覆盖单目、双目、多视图、稀疏补全及全景深度；目标是独立训练/评价一个模型并解释尺度、泛化和部署的边界。当前为学习计划，运行状态见项目及实验记录。

## 学习顺序

1. **几何与定义**：深度、逆深度、视差、相对/米制深度；针孔、畸变与双目校正；遮挡、低纹理和远距离误差。
2. **经典基线**：从标定已知的双目数据做匹配/三角化，理解 SGBM 的代价与平滑约束；再看学习代价体、相关性及 RAFT-Stereo 迭代更新。
3. **单目模型**：DPT 的编码与多尺度解码、Depth Anything V2 的训练数据和监督；分清相对版本与米制 checkpoint，再选一个米制模型理解内参与尺度。
4. **补全**：RGB+稀疏深度+有效性掩码，局部/非局部传播、边缘约束和置信度；从简单插值到 NLSPN，检查稀疏度、离群点和边界泄漏。
5. **全景**：ERP/Cubemap、球面射线、重采样与接缝、径向距离与光轴深度；以 UniFuse 为代表理解双表示融合，分别检查高纬、接缝与常规区域。
6. **训练和部署**：数据划分、伪标签噪声、监督掩码、尺度不变/梯度等损失；在验证集选择调优因素，之后分析蒸馏、剪枝、量化或输入分辨率的精度—资源取舍。

## 几何与评价约定

理想已校正、水平基线且两相机主点横坐标一致时，$z=f_xB/d$；$f_x$ 与视差单位为像素，$B$ 的单位决定 $z$ 的单位。实际内参不满足该约定时使用真实校正模型或 Q 矩阵，不能给未标定图片任意指定 $f,B$ 后宣称米制精度。

全景径向距离 $r$ 与光轴深度 $z$ 不同：对单位射线 $\mathbf b$，相机点为 $r\mathbf b$，该相机坐标的光轴分量为 $z=r b_z$。ERP 转虚拟透视时同步变换射线、外参与深度含义，记录坐标轴和无效背向区域。

在有效 GT 集合 $V$ 上，$\mathrm{AbsRel}=|V|^{-1}\sum|\hat z-z|/z$，$\mathrm{RMSE}=\sqrt{|V|^{-1}\sum(\hat z-z)^2}$；$\delta_1$ 使用 $\max(\hat z/z,z/\hat z)<1.25$ 的比例。报告单位、按图/全像素聚合、split、crop、mask、深度范围、插值及输出无效值处理。

相对深度模型的输出可能在逆深度域定义；若使用 GT 离线尺度/平移对齐，应标明对齐域和规则，结果作为 aligned 指标。米制模型的主结果保留未经 GT 后验对齐的输出；对齐后的诊断单列。评测不因某模型的失败像素而静默缩小共同支持。

补全输入稀疏测量与评价 GT 分开登记；从稠密 GT 下采样只能作为明确标识的合成采样实验，采样模式、密度、随机种子及评价区域均固定。若报告深度与逆深度误差，分别给出 m、mm 或 1/m 等单位。

全景先按数据集官方像素协议评价；额外球面面积加权可使用像素对应纬度的面积权重，需标明为补充指标，不能与未加权排行榜数字混写。

## 验收与入口

- [ ] 明确任务、相机、输出含义和尺度；完成标定双目与深度评价练习。
- [ ] 跑通一种单目或补全基线；记录正常和失败场景。
- [ ] 完成一个验证集选定的单变量实验，保持测试集与训练/调参隔离。
- [ ] 在实际设备对照精度、端到端延迟、峰值内存和输入预处理；区分导出成功与真实部署。

练习：[深度与双目](../../03_exercises/05_depth_estimation/README.md)、[补全](../../03_exercises/06_depth_completion/README.md)。项目：[深度调优与部署](../../04_projects/depth_estimation/README.md)。

## 核验过的原始资料入口

- [OpenCV 标定与三维重建](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)、[DPT 原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html)。
- [Depth Anything V2 作者仓库](https://github.com/DepthAnything/Depth-Anything-V2)、[Metric3D 作者仓库](https://github.com/YvanYin/Metric3D)。
- [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)、[NLSPN](https://github.com/zzangjinsun/NLSPN_ECCV20)、[UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)。

资料核对日期：2026-11-08。实际执行前固定 commit/checkpoint/环境与对应数据协议，不从文档预填运行成功。
