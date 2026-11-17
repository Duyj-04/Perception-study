# 深度估计与补全文献

使用 [论文笔记模板](../paper_note_template.md)，并在 [论文索引](../paper_index.csv) 登记路径和真实阅读状态。

每篇先核对任务、输入传感器/相机、深度或逆深度、相对/米制尺度、训练数据/伪标签、损失和推理输出；记录 split、有效 mask、crop、深度范围、单位和是否 GT 对齐。补全另记稀疏点来源、采样与传播；双目另记校正/视差；全景另记球面距离与 ERP/Cubemap。

只登记原文支持的结论，作者结果与个人运行结果分开。Depth Anything V2 已完成首轮阅读并建立问答笔记，网络结构等内容继续补齐；具体进度见阅读笔记。

## 已下载原文

- [Depth Anything V2 — 本地 PDF](Yang_2024_NeurIPS_Depth_Anything_V2.pdf)：arXiv `2406.09414v2`，2024-10-20 修订，共 30 页，含正文、附录与参考文献。页面标注已被 NeurIPS 2024 接收；当前为首轮已读、问答核验中。
- [下载来源](https://arxiv.org/pdf/2406.09414v2)。已核验 PDF 格式、标题、作者及全部页面内容流可读。

## 中文阅读材料

- [Depth Anything V2 阅读笔记](Depth_Anything_V2_zh/reading_notes.md)：记录2026-11-14本轮理解、讨论修正、原文依据和待学习事项；包含关于模型性能比较的本人澄清。
- [Depth Anything V2 网络结构讲义](Depth_Anything_V2_zh/network_structure.md)：从patch/token到DINOv2、DPT和输出恢复；附固定源码版本及CPU随机参数形状核验。
- [Depth Anything V2 正文中英逐段对照](Depth_Anything_V2_zh/bilingual_main.md)：正文 p.1—10，48组正文对照，保留9幅图、6张表及双语图注；按用户提供的本地PDF整理。
- [Depth Anything V2 中文精读指南](Depth_Anything_V2_zh/paper.md)：按正文 p.1—10 的问题、方法与实验梳理，包含术语、训练流程图、关键消融表和阅读自测。
- [划词翻译与 PDF 双语阅读工具](Depth_Anything_V2_zh/translation_tools.md)：官方安装与使用入口。
- [整理范围与核验说明](Depth_Anything_V2_zh/translation_notes.md)：说明原文页码、图表来源与未逐段翻译的范围。
- [逐段对照版整理说明](Depth_Anything_V2_zh/bilingual_notes.md)：说明段落对齐、术语、图表排版和译注。
