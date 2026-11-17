# 学习式特征与可微几何

## 学习范围

XFeat、SuperPoint、LightGlue、DROID；PyTorch 数据/训练/推理、监督与伪标签、相关体、循环更新、可微 BA、模型压缩。深度专题见 [深度估计](../depth_estimation/README.md)，表示与渲染见 [3DGS](../gaussian_splatting/README.md)。

## 基础学习入口

- [特征维度、Token与Attention基础](token_embedding_attention.md)：从4个输入数的投影解释384维，再逐步手算Q/K/V、softmax、加权聚合与多头维度。
- [可运行Notebook](token_embedding_attention.ipynb)：解释、代码、数值、图形和可修改参数；已保存一次从头执行的结果。
- [教学核验记录](token_attention_verification.json)：Conv2d/线性投影、NumPy/PyTorch及固定官方Attention的数值比较。

从工作区根目录使用本机已有环境：

```powershell
& 'D:\Conda\envs\torch_gpu_py310\python.exe' -m notebook '01_foundations/learning/token_embedding_attention.ipynb'
```

选择包含NumPy、Matplotlib、PyTorch的内核，并核对首个代码单元打印的解释器路径。算例均为手工或合成数据，不是深度模型实测；材料与执行结果已提供，个人理解仍需自测。

## 验收实验

把训练损失与任务结果区分：深度误差与尺度、三维几何一致性、留出视角渲染、SLAM ATE/RPE/鲁棒性；分别验证调优或压缩后的精度和资源变化。

## 笔记约定

每个专题先写：定义/坐标系/假设、关键推导、实现链接、数值验证、失败边界。将具体笔记作为独立 `.md` 文件放本目录。

## 本专题进度

- [ ] 基础推导
- [ ] 可运行小实验
- [ ] 与 SLAM 仓库真实实现对应
