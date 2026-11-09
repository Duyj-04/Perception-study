# 李群与非线性优化

## 学习范围

SO(3)/SE(3)/Sim(3)、左右扰动、Jacobian、GN/LM、BA、Schur、鲁棒核。

## 学习笔记

- [面向 SLAM 的线性代数基础](linear_algebra_for_slam.md)：秩、零空间、SVD、QR、特征值、正定性、条件数、退化与 Schur 消元，含可运行数值例子。
- [线性代数交互式 Notebook](linear_algebra_for_slam.ipynb)：按主题运行推导、数值实验、图形和自测；已保存一次运行输出。

从工作区根目录在 PowerShell 中启动本机已有的 Jupyter 环境：

```powershell
& 'D:\Conda\envs\torch_gpu_py310\python.exe' -m notebook '01_foundations/optimization/linear_algebra_for_slam.ipynb'
```

在 Notebook 中选择包含 NumPy、Matplotlib 的 Python 内核，从头运行时检查首个代码单元输出的解释器路径；Notebook 图示是合成例子，不是 SLAM 实测。

## 验收实验

实现小型 BA；对照解析和数值 Jacobian、观察退化。

## 笔记约定

每个专题先写：定义/坐标系/假设、关键推导、实现链接、数值验证、失败边界。将具体笔记作为独立 `.md` 文件放本目录。

## 本专题进度

- [ ] 基础推导
- [ ] 可运行小实验
- [ ] 与 SLAM 仓库真实实现对应
