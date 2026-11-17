# Depth Anything V2 网络结构：从图像到逐像素输出

**学习目标：** 能解释图像如何经过DINOv2和DPT形成稠密相对逆深度图，并把解释对应到源码及张量形状。

如果“为什么是384维”或Attention公式尚不清楚，先读01中的 [维度、Token与Attention基础](../../../01_foundations/learning/token_embedding_attention.md)，配合 [基础Notebook](../../../01_foundations/learning/token_embedding_attention.ipynb) 手算小矩阵，再回到本讲义。

**记录日期：** 2026-11-14。源码固定为官方提交 [`a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`](https://github.com/DepthAnything/Depth-Anything-V2/commit/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf)。下面以官方Small完整配置为例：`encoder='vits'`、`features=64`、`out_channels=[48,96,192,384]`、`use_clstoken=False`。

**核验类型：** 已在现有PyTorch环境中运行CPU随机参数结构检查，记录中间形状；输入为人为构造的常量图像，未加载预训练权重。它核验结构、预处理和接口，不提供深度准确率结果。

对应论文：p.8 §7.1，[S037](bilingual_main.md#S037)。原文只概述DINOv2编码器和DPT解码器，本讲义的逐层细节来自固定版本源码。

## 1. 先建立整体分工

```text
原图 [H0,W0,3]
  → 颜色/尺寸/数值预处理
网络输入 [B,3,H,W]
  → DINOv2：patch编码、位置编码、Transformer
四层图像特征，每层 [B,N,C]
  → DPT：恢复空间网格、构造多尺度特征、融合
单通道预测 [B,1,H,W]
  → 去掉通道维、恢复原图尺寸
单图输出 [H0,W0]
```

**DINOv2负责形成图像表示，DPT负责将这些表示解码为稠密预测。** 训练时损失函数决定这些参数应该怎样变化；推理时使用已经学到的参数执行这条计算链。

源码中的`self.pretrained`是成员名。创建网络结构和加载预训练权重是两步：官方README在创建模型后另外调用`load_state_dict`。本次验证只创建随机初始化的结构，所以不能将其输出当作已学会的深度预测。

## 2. 张量的轴分别是什么

| 记号 | 含义 | Small方图例子 |
|---|---|---|
| $B$ | 一个batch中的图像数 | 1 |
| $H,W$ | 网络实际接收的图像高、宽 | 518、518 |
| $C$ | 每个位置的特征维度，不是深度单位 | DINOv2-Small中为384 |
| $H_p,W_p$ | patch组成的空间网格 | 37、37 |
| $N$ | patch数量，$N=H_pW_p$ | 1369 |

图像数组常用`[H,W,3]`，PyTorch卷积常用`[B,C,H,W]`，Transformer的token序列常用`[B,N,C]`。轴顺序变化并不等于数据含义自动改变，需要知道每条轴保存什么。

一个token在这里是**某个图像patch对应的特征向量**。`384`表示这个向量有384个数，既不是384个像素，也不是384米；每个维度没有预先规定的物理单位或唯一语义。

## 3. 图像怎样变成patch token

该版本实际使用`patch_size=14`。以`518×518`输入为例：

$$
H_p=W_p=518/14=37,\qquad N=37\times37=1369.
$$

一个RGB patch有$14\times14\times3=588$个输入数。可以将其编码理解成一个可学习的线性投影：

$$
t_i=W_{\mathrm{emb}}\operatorname{vec}(P_i)+b,
\qquad W_{\mathrm{emb}}\in\mathbb R^{384\times588}.
$$

实际源码用`kernel_size=14, stride=14`的卷积同时处理所有patch，然后展平空间轴：

```text
[1,3,518,518]
  → Conv2d(3→384, kernel=14, stride=14)
[1,384,37,37]
  → flatten空间轴，再transpose
[1,1369,384]
```

patch投影得到的是特征，不是最终深度值。后续还需要在不同位置间交换信息，并通过解码头形成逐像素预测。

源码：[PatchEmbed.forward](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dinov2_layers/patch_embed.py#L69-L82)。

## 4. CLS和位置编码为何出现在序列里

编码器会在1369个patch token前加入一个可学习的CLS token，并加上位置编码，因此进入Transformer的形状为：

$$
[1,1370,384].
$$

CLS不对应一个实际图像patch，可以参与汇集上下文。位置编码提供空间位置信息，帮助网络区分同样内容出现在哪个位置；输入空间尺寸变化时，源码会插值处理patch位置编码。

这份Small配置没有register token。`get_intermediate_layers(..., return_class_token=True)`会将各层的patch token和CLS token拆开返回。后面的DPT头在`use_clstoken=False`时不直接使用CLS向量，但CLS此前仍参与了编码器中的信息交换。

源码：[prepare_tokens_with_masks](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dinov2.py#L212-L231)、[get_intermediate_layers](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dinov2.py#L297-L321)。

## 5. Transformer在这些向量上做什么

先区分两种计算：**Attention在不同token间交换信息，MLP处理每个token的特征通道。**

### 5.1 Attention：从哪些位置收集信息

对token特征$X$分别进行可学习投影，得到$Q,K,V$。单个注意力头可写为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^T}{\sqrt d}\right)V.
$$

$QK^T$描述位置之间的信息关联，softmax沿被关注的位置归一化，然后用这些权重组合$V$。这些权重不是深度值，也不是标签。

Small使用6个头，$384/6=64$。方图例子中，各头的$Q,K,V$形状为`[1,6,1370,64]`，关联矩阵形状为`[1,6,1370,1370]`，合并多头后回到`[1,1370,384]`。这些拆分形状按源码推导；运行记录另外捕获了联合QKV投影的实际形状`[1,1370,1152]`。

源码：[Attention.forward](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dinov2_layers/attention.py#L49-L62)。

### 5.2 MLP、归一化和残差

Small中的MLP先将每个token从384维扩展到1536维，经过GELU，再投影回384维：

```text
[B,N,384] → Linear → [B,N,1536] → GELU → Linear → [B,N,384]
```

这种MLP不直接混合token位置，位置间的信息交换发生在Attention中。Transformer块还包含LayerNorm、残差相加和LayerScale等操作，使特征可以在多层计算中逐步更新。这里的“残差”指网络特征的跳接，和几何优化中观测减预测的残差不是同一概念。

### 5.3 为什么要取四层

Small有12个Transformer块，Depth Anything V2选取零起始索引`[2,5,8,11]`，即第3、6、9、12个块的输出。四层代表不同计算深度的特征；在这里，它们的patch网格仍都为`37×37`。

传给DPT的是四组`(patch_tokens, cls_token)`：每组分别为`[1,1369,384]`和`[1,384]`。所以DPT循环里的`x=x[0]`是在元组中选取全部patch特征，**不是只保留第一个patch**。

源码追踪时，`DepthAnythingV2.forward`显式调用编码器的`get_intermediate_layers`，再把结果交给`DPTHead.forward`。不要把DINOv2自身的最终CLS输出接口，误当成这里送入深度头的patch特征。

这与CNN逐级下采样形成特征金字塔的情况不同：本例的四层ViT特征具有相同空间网格，下一步才由DPT构造不同尺寸的特征图。

## 6. DPT怎样把token变成特征金字塔

### 6.1 恢复空间网格

每一层patch token先从`[B,N,C]`变为`[B,C,N]`，再按照已知patch网格重排：

```text
[1,1369,384] → permute → [1,384,1369] → reshape → [1,384,37,37]
```

这个重排本身没有学习参数，只是把位置索引恢复为二维网格；它要求$N=H_pW_p$，不是随意把一个向量拉伸成图像。

### 6.2 调整通道与分辨率

Small完整配置将四层分别投影到`[48,96,192,384]`个通道，再进行不同的空间调整。以下数值已由运行中的hook捕获：

| 分支 | 输入网格 | 1×1卷积后通道 | 空间操作 | 输出特征图 |
|---|---|---:|---|---|
| 1 | 37×37 | 48 | 转置卷积，kernel=4、stride=4 | `[1,48,148,148]` |
| 2 | 37×37 | 96 | 转置卷积，kernel=2、stride=2 | `[1,96,74,74]` |
| 3 | 37×37 | 192 | Identity | `[1,192,37,37]` |
| 4 | 37×37 | 384 | 3×3卷积，stride=2、padding=1 | `[1,384,19,19]` |

1×1卷积主要在每个位置上组合通道，不直接扩大其空间邻域。转置卷积在此是可学习的上采样操作，不能理解为严格求逆原来的卷积。第4分支的37变19来自卷积尺寸公式，而不是把37直接除以2。

接着，`scratch.layer1_rn`至`layer4_rn`用3×3卷积将四支通道统一为64；空间尺寸分别保持148、74、37、19。

### 6.3 逐级融合

融合块将粗分支与较细分支结合，再把结果调整到下一层需要的尺寸：

| 源码模块 | 融合前的空间尺寸 | 输出形状 |
|---|---|---|
| `refinenet4` | 最粗分支19×19 | `[1,64,37,37]` |
| `refinenet3` | 上一路37×37＋第3分支37×37 | `[1,64,74,74]` |
| `refinenet2` | 上一路74×74＋第2分支74×74 | `[1,64,148,148]` |
| `refinenet1` | 上一路148×148＋第1分支148×148 | `[1,64,296,296]` |

最容易算错的是第一步：`refinenet4`明确指定目标为第3分支的37×37，因此**19变37，不是简单翻倍成38**。每次做特征相加时，两个输入的空间网格也必须匹配。

模块内部包含残差卷积、特征相加和双线性插值。上采样增加了输出位置数，最终能否恢复正确细节还取决于学习到的特征与监督；不能只凭尺寸变大就认为新增了可靠的几何信息。

源码：[DPTHead.forward](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py#L117-L150)、[FeatureFusionBlock.forward](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/util/blocks.py#L123-L148)。

## 7. 怎样得到最终逐像素输出

融合后的`[1,64,296,296]`还要经过输出层：

```text
[1,64,296,296]
  → output_conv1：64→32
[1,32,296,296]
  → 插值到patch网格×14
[1,32,518,518]
  → output_conv2：卷积、ReLU、输出1通道
[1,1,518,518]
  → DepthAnythingV2.forward的squeeze(1)
[1,518,518]
```

`squeeze(1)`只去掉长度为1的通道维，batch维保留。`infer_image`随后将结果插值回原图高宽，取出单图结果，转为NumPy数组`[H0,W0]`。

末端ReLU将负数截为0，不会把输出标定为米，也不是分类softmax。一通道表示每个位置预测一个标量；彩色深度可视化通常是后处理，不能据此把网络输出理解为三通道RGB。

源码：[DepthAnythingV2.forward](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py#L176-L184)、[infer_image](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py#L187-L194)。

## 8. 预处理为何也要读

`image2tensor`执行以下步骤：

1. 记录原图尺寸；将OpenCV的BGR转换为RGB，并除以255。
2. 缩放图像，尽量保持长宽比，同时约束高宽为14的倍数。
3. 按RGB通道的均值`[0.485,0.456,0.406]`、标准差`[0.229,0.224,0.225]`归一化。
4. 将`[H,W,3]`转为`[3,H,W]`的连续float32数组，再增加batch维。

这个归一化作用于**输入颜色**，不等于深度输出的尺度对齐。

`input_size=518`并不意味着所有推理图片都会变成518×518。实际验证的矩形例子是：

| 阶段 | 高×宽或形状 |
|---|---|
| 原图 | 480×640 |
| 网络输入 | `[1,3,518,686]` |
| patch网格 | 37×49，共1813个patch |
| 加CLS后的序列 | `[1,1814,384]` |
| DPT四分支空间尺寸 | 148×196、74×98、37×49、19×25 |
| 网络输出 | `[1,518,686]` |
| `infer_image`最终输出 | `[480,640]` |

518与686均为14的倍数。取整可能使长宽比与原图略有偏差，不能将配置中的`keep_aspect_ratio=True`理解为取整后比例也绝对不变。

论文p.8 §7.1描述的训练流程还包含随机裁剪到518×518；它与这里的单图推理预处理是不同环节。

源码：[image2tensor](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py#L196-L221)、[Resize.get_size](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/util/transform.py#L62-L107)。

## 9. 本次怎样核验，怎样重跑

使用已有环境：Python 3.10.20、PyTorch 2.11.0+cu128、OpenCV 4.10.0、NumPy 2.2.6。实际运行设备为CPU、4个Torch线程，使用作者的普通PyTorch Attention回退分支。模型参数数目为24,785,089，与官方Small约24.8M的量级一致。

| 核验项 | 实际检查 | 结果 |
|---|---|---|
| 方图518×518 | patch、CLS、四层特征、DPT分支/融合、模型与接口输出形状 | 通过 |
| 原图480×640 | 原生预处理为518×686，最终输出恢复480×640 | 通过 |
| 输入颜色与数值 | 构造已知BGR颜色，检查RGB通道、float32和归一化数值 | 通过 |
| 原生推理包装 | `infer_image`与同模型前向结果的尺寸恢复一致 | 通过 |
| batch=2的小输入 | 模型前向`[2,3,56,84]→[2,56,84]` | 通过 |
| 非14倍数高度 | 直接前向输入`[1,3,55,84]`由PatchEmbed断言拒绝 | 通过 |

这次验证使用随机初始化参数和常量图像，验证的是计算结构与接口。没有预训练模型精度、泛化、速度或部署结论。

在工作区根目录可重跑：

```powershell
& 'D:\Conda\envs\torch_gpu_py310\python.exe' '.\02_papers\depth_estimation\Depth_Anything_V2_zh\network_shape_check.py'
```

脚本会核对源码哈希，默认读取`.cache/depth_anything_v2_a561b849ebae/`中的固定版本。该缓存由现有忽略规则排除；如果缓存不存在，可把该官方提交的外部源码根目录传给`--source-root`。脚本不会自动联网、安装依赖或下载权重。

- [结构检查脚本](network_shape_check.py)
- [真实形状记录](network_shape_trace.json)
- [固定源码与逐文件哈希](network_source_manifest.json)

## 10. 按什么次序掌握这份结构

第一轮：先说明BCHW、token的含义，以及编码器和解码器的分工；能从518÷14算出37×37网格。

第二轮：解释CLS、Attention与MLP的作用；说明为什么四个中间层都有1369个patch token，以及`x[0]`取出的是什么。

第三轮：沿DPT表格逐级计算空间尺寸，重点检查37→19以及19→37的来源，再追踪如何恢复原图尺寸。

最后自测：

- [ ] `384`个通道分别表示384个像素或384种深度吗？
- [ ] 1369与1370为什么同时出现在这条链中？
- [ ] 四层Transformer特征为什么不自动构成四个空间分辨率？
- [ ] DPT里19×19为何变为37×37，而不是38×38？
- [ ] 输出恢复到原图大小，是否同时恢复了米制尺度？

理解程度仍需自己的复述和代码练习核验；本次结构记录不自动改变技能矩阵等级。继续记录在 [阅读笔记](reading_notes.md)。
