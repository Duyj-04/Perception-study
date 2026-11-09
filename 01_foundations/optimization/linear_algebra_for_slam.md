# 线性代数基础：从矩阵分解到 SLAM 退化与 Schur 消元

> 学习目标：不仅会算 SVD、QR 和特征值，还能解释优化中哪个状态方向没有被约束、为什么解会对噪声敏感，以及 BA 为什么可以先消去地图点。本文的矩阵均为教学用合成例子，不是实际项目结果。

> 学习入口：本文提供详细推导；[配套 Notebook](linear_algebra_for_slam.ipynb) 按相同主题提供可运行单元。建议按“矩阵与秩 → SVD → 特征值和正定性 → 条件数 → QR → SLAM 规范自由度 → Schur”学习；每节先手算，再运行验证。

## 0. 一张贯穿全文的图

```text
状态扰动 δx ──J──> 预测残差变化 Jδx
      │                    │
      │                    └─ 奇异值小：这一方向的变化几乎看不见
      └─ Jδx = 0：局部线性模型中，这个方向完全看不见

加权最小二乘 → JᵀWJ δx = -JᵀWr
                     │
                     ├─ 特征值/零空间：分析信息与退化
                     ├─ QR / SVD：求解或诊断线性子问题
                     └─ 分块 Schur：先消去地图点，再求位姿
```

学习顺序建议：第 1—3 节建立空间和 SVD 直觉；第 4—6 节学正定性、条件数和 QR；第 7—9 节联系优化和 Schur；最后动手复现第 10 节并做第 11 节。先动笔再看结论。

## 1. 从向量和矩阵开始：矩阵到底做了什么？

令 $A\in\mathbb R^{m\times n}$，输入 $x\in\mathbb R^n$，输出 $y=Ax\in\mathbb R^m$。把 $A$ 看成从状态变化到观测变化的线性映射：

$$
A=[a_1\;a_2\;\cdots\;a_n],\qquad Ax=x_1a_1+\cdots+x_na_n.
$$

第 $j$ 列 $a_j$ 告诉我们：只改变第 $j$ 个输入，输出沿什么方向改变。SLAM 中，$A$ 常常是某次线性化后的 Jacobian $J$；列对应状态扰动，行对应残差分量。

- **列空间** $\operatorname{Col}(A)$：所有可能的 $Ax$，即当前线性模型能解释的输出变化。
- **秩** $\operatorname{rank}(A)$：列空间维数，也等于独立行数；最多为 $\min(m,n)$。
- **零空间** $\operatorname{Null}(A)=\{v:Av=0\}$：输入改变但输出不变的方向。
- **秩—零度定理**：$\operatorname{rank}(A)+\dim\operatorname{Null}(A)=n$。
- **左零空间** $\operatorname{Null}(A^\top)$：与所有列正交的输出方向；维数为 $m-\operatorname{rank}(A)$。

若 $Av=0$，则对任意 $t$，$A(x+tv)=Ax$。所以仅凭当前线性测量，无法区分沿 $v$ 的不同状态。注意：这是**局部线性模型**的结论；是否对应全局非线性不可观性，还要检查模型和线性化点。

### 手算：一个不唯一的系统

$$
A=\begin{bmatrix}1&2\\2&4\end{bmatrix},\qquad
v=\begin{bmatrix}-2\\1\end{bmatrix},\qquad Av=0.
$$

第二列是第一列的两倍，故秩为 1，零空间由 $v$ 张成。若 $b=(3,6)^\top$，则所有精确解为 $(3,0)^\top+t(-2,1)^\top$；其中最小欧氏范数解为 $(3/5,6/5)^\top$。**残差为零不意味着解唯一。**

## 2. SVD：找出最强和最弱的输入方向

任意实矩阵都有奇异值分解：

$$
A=U\Sigma V^\top,\qquad Av_i=\sigma_i u_i,\qquad
\sigma_1\ge\sigma_2\ge\cdots\ge0.
$$

$V$ 的列 $v_i$ 是输入空间中的正交方向，$U$ 的列 $u_i$ 是相应输出方向；$\sigma_i$ 表示单位输入在该方向的放大倍数。可以想象：$V^\top$ 先转动输入，$\Sigma$ 按方向缩放，$U$ 再转动输出。对矩形矩阵，完整 SVD 的 $U,V$ 为方正交阵，$\Sigma$ 为矩形；精简 SVD 只保留所需列。

- $\sigma_i>0$：该方向在输出中可见；$\sigma_i$ 越小，单位输入引起的输出越小。
- $\sigma_i=0$ 对应的右奇异向量属于零空间；非零奇异值的数目等于秩。
- $A^\top A v_i=\sigma_i^2 v_i$，所以奇异值是 $A^\top A$ 特征值的非负平方根。
- 若 $m<n$，即使返回的 $\min(m,n)$ 个奇异值都非零，仍至少有 $n-m$ 维右零空间；检查完整 $V$ 或用秩—零度定理。
- 数值计算中“是否为零”取决于阈值和精度。常见规则是 $\sigma_i\le\tau\sigma_{\max}$ 视为数值零；阈值应随噪声、单位和任务说明，不能把某个固定阈值当成普适物理判据。

例如第 1 节的 $A$ 的奇异值为 $(5,0)$，右零空间方向是 $(-2,1)^\top/\sqrt5$。对秩亏最小二乘，SVD 可构造 Moore–Penrose 伪逆 $A^+$：令非零奇异值取倒数，零奇异值保留为零，$x^\dagger=A^+b$ 是**最小残差解中的最小欧氏范数解**。它为选择一个解提供数学规则，本身不会创造缺失的物理观测。

### 2.1 从头认识 U、Σ、V

设 $A\in\mathbb R^{m\times n}$，**完整 SVD** 的维度为

$$
A_{m\times n}=U_{m\times m}\Sigma_{m\times n}V^\top_{n\times n},\quad
U^\top U=I_m,\quad V^\top V=I_n.
$$

写作 $U=[u_1,\ldots,u_m]$、$V=[v_1,\ldots,v_n]$：$v_i\in\mathbb R^n$ 是输入空间的单位正交方向，$u_i\in\mathbb R^m$ 是输出空间的单位正交方向，$\Sigma$ 的非负对角元素是按降序排列的奇异值。正交矩阵保持长度与夹角；其作用可以包含反射，不一定只是旋转。对角形式的 $\Sigma$ 可为矩形，最多有 $\min(m,n)$ 个对角项。

对于配对的 $i\le\min(m,n)$，先将 $v_i$ 表示在 $V$ 这组输入基下：$V^\top v_i=e_i^{(n)}$。再经 $\Sigma$ 缩放：$\Sigma e_i^{(n)}=\sigma_i e_i^{(m)}$，最后 $Ue_i^{(m)}=u_i$。这里 $e_i^{(n)}$ 和 $e_i^{(m)}$ 分别是两个空间的标准基向量。因此：

$$
Av_i=U\Sigma V^\top v_i=\sigma_i u_i.
$$

一般输入 $x=\sum_i\alpha_i v_i$，其输出为配对分量之和 $Ax=\sum_{i=1}^{\min(m,n)}\sigma_i\alpha_i u_i$；若输入维数更大，额外的完整右奇异向量位于右零空间。奇异值为零时，$Av_i=0$，此时不能从零输出确定唯一的输出方向 $u_i$；SVD 仍可以选取一组正交的 $u_i$ 来补全 $U$。

### 2.2 一个输入方向与输出方向不同的例子

$$
A=\begin{bmatrix}0&-2\\3&0\end{bmatrix},\quad
U=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\quad
\Sigma=\begin{bmatrix}3&0\\0&2\end{bmatrix},\quad V=I_2.
$$

直接相乘可核对 $U\Sigma V^\top=A$。第一个输入方向 $v_1=(1,0)^\top$ 对应 $u_1=(0,1)^\top$，且 $Av_1=(0,3)^\top=3u_1$；第二个输入方向 $v_2=(0,1)^\top$ 对应 $u_2=(-1,0)^\top$，且 $Av_2=(-2,0)^\top=2u_2$。例如 $x=2v_1+v_2$，则 $Ax=6u_1+2u_2=(-2,6)^\top$。

奇异向量不总是唯一：一对 $u_i,v_i$ 同时变号仍是有效分解；重复奇异值和零奇异值也带来基的选择自由。因此代码核对应以重构 $A\approx U\Sigma V^\top$、正交性及方向关系为准，不要要求不同实现输出完全相同的符号。

## 3. 特征值与奇异值：不要混用

对方阵 $H$，若非零向量 $v$ 满足 $Hv=\lambda v$，则 $v$ 是特征向量、$\lambda$ 是特征值。一般非对称矩阵的特征值可以为负或复数，也可能无法由一组正交特征向量对角化；奇异值始终实且非负，矩形矩阵同样存在 SVD。

对实对称矩阵，存在正交特征分解 $H=Q\Lambda Q^\top$。此时其奇异值是 $|\lambda_i|$；只有 $H$ **半正定**时，奇异值才与特征值相等（按排序配对）。例如 $H=\operatorname{diag}(2,-1)$ 的特征值是 $2,-1$，奇异值是 $2,1$；观察奇异值本身无法区分该矩阵是正定还是不定。

在优化中，若 $H$ 是二次近似的对称矩阵，$v^\top Hv$ 描述沿 $v$ 的二次曲率：正曲率是碗形，零曲率是平坦方向，负曲率说明局部二次模型存在下降曲率方向。要注意状态尺度和噪声权重，否则各方向数值大小未必能直接做物理比较。

## 4. 正定、半正定与信息矩阵

对实对称矩阵 $H$：

| 性质 | 二次型定义（任意 $x\ne0$） | 特征值 | 直观 |
|---|---|---|---|
| 正定 $H\succ0$ | $x^\top Hx>0$ | 全部大于 0 | 所有非零方向都有正曲率 |
| 半正定 $H\succeq0$ | $x^\top Hx\ge0$ | 全部非负 | 允许平坦方向 |
| 不定 | 有方向为正，也有方向为负 | 同时有正、负 | 既有上升也有下降曲率 |

关键推导：若 $W\succeq0$，则

$$
H=J^\top WJ,\quad x^\top Hx=(Jx)^\top W(Jx)=\|W^{1/2}Jx\|^2\ge0.
$$

因此 $H$ 一定半正定；它正定**当且仅当** $W^{1/2}J$ 满列秩。即使 $J$ 满列秩，若 $W$ 抹掉了某些观测方向，$H$ 仍可能奇异。这个结论适用于上述 Gauss–Newton 形式；一般非线性目标的**真实 Hessian**还可能含残差的二阶导数项，不能无条件宣称半正定。

当误差模型为线性高斯、噪声协方差和观测模型正确、并且已经合理处理规范自由度时，$J^\top WJ$ 可以解释为局部信息矩阵，其逆与局部协方差近似有关。存在零空间时普通逆不存在；伪逆或加先验的结果依赖具体处理，不能当作未作假设的绝对位置不确定度。

## 5. 条件数：问题对误差有多敏感？

对可逆方阵，在 2 范数下：

$$
\kappa_2(A)=\|A\|_2\|A^{-1}\|_2
=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.
$$

满列秩矩形矩阵可用最大与最小列方向奇异值之比衡量；若存在未约束输入方向，则状态恢复问题没有唯一解，应报告秩亏/无穷敏感性。$m<n$ 时尤其不要仅拿库函数返回的短奇异值列表声称没有零空间。

**例子**：$A_\varepsilon=\operatorname{diag}(1,\varepsilon)$，取 $\varepsilon=10^{-3}$：它在精确运算中满秩，$\kappa_2(A_\varepsilon)=10^3$。若第二个观测只改变 $10^{-4}$，解的第二个分量就改变 $10^{-1}$。这不是计算器坏了：模型本身对该方向的观测很弱。

若 $J$ 满列秩，$H=J^\top J$ 的 2 范数条件数满足：

$$
\kappa_2(H)=\kappa_2(J)^2.
$$

因此 $\kappa_2(J)=10^3$ 时，正规方程矩阵的条件数为 $10^6$。这是**正规方程矩阵**的关系，不是说任何最小二乘问题的固有敏感性必然平方。形成 $J^\top J$ 还会发生舍入；一个求解算法是否稳定，与原问题是否病态是不同问题。残差很小，也不能保证病态问题的状态误差很小。

**单位很重要**：位置按米或毫米表示、角度按弧度、偏置尺度等，都会改变数值条件数。若 $\delta x=D\delta\tilde x$ 且 $D$ 可逆，则 $\tilde J=JD$：条件数可能明显变化，精确秩和零空间维数不变。先按噪声协方差白化残差，再用有物理意义的变量尺度报告诊断；非零缩放不能创造真实缺失的几何信息。

## 6. QR：不显式形成正规方程的最小二乘

考虑 $\min_x\|Ax-b\|_2$，$A\in\mathbb R^{m\times n}$、$m\ge n$ 且满列秩。精简 QR 分解为

$$
A=QR,\qquad Q^\top Q=I,\quad R\in\mathbb R^{n\times n}\text{ 为上三角阵}.
$$

最小二乘解满足 $Rx=Q^\top b$，只需三角回代。$Q$ 的列是列空间的正交基；最优残差与列空间正交。实践中通常用 Householder 等稳定实现，不必手写经典 Gram–Schmidt。

**手算例子**：

$$
A=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix},\qquad
b=\begin{bmatrix}1\\0\\0\end{bmatrix},\qquad
A^\top A=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$

$A$ 满列秩；其正规矩阵特征值为 $3,1$，因此 $A$ 的奇异值为 $\sqrt3,1$。解为 $x=(2/3,-1/3)^\top$，残差 $b-Ax=(1,-1,1)^\top/3$，且 $A^\top(b-Ax)=0$。第 10 节会使用 QR 求出同一结果。

| 方法 | 适用直觉 | 代价或边界 |
|---|---|---|
| 正规方程 + Cholesky | 权重/锚定正确、系统结构合适时能利用稀疏性 | 条件数平方；Cholesky 需满足相应正定条件 |
| QR | 直接处理最小二乘，不显式形成 $A^\top A$ | 基本无主元 QR 的三角回代假定满列秩；大稀疏系统有填充和内存代价 |
| 列主元 QR / SVD | 需要发现或处理秩亏时 | 需指定数值秩阈值；SVD 通常更昂贵 |

QR 不是“总能修复不可观测”：求解器可以选择代表解，却不能从缺失数据恢复不存在的信息。

### 6.1 先理解最小二乘，再理解 QR

当 $m\ge n$ 且观测有噪声时，$Ax=b$ 可能没有精确解，因此寻找 $x^\star=\arg\min_x\|Ax-b\|_2^2$。$Ax$ 是 $A$ 各列的线性组合，所以只能落在 $\operatorname{Col}(A)$；最优预测 $Ax^\star$ 是 $b$ 在该列空间的正交投影。令 $r=b-Ax^\star$，则 $A^\top r=0$：残差与每个列方向垂直，但残差自身不必为零。

精简 QR 的 $Q=[q_1,\ldots,q_n]$ 是与 $A$ 同列空间的正交单位基，$R$ 记录原始列在该基中的坐标。例如两列时：

$$
A=[a_1,a_2]=[q_1,q_2]
\begin{bmatrix}r_{11}&r_{12}\\0&r_{22}\end{bmatrix},\quad
a_1=r_{11}q_1,\quad a_2=r_{12}q_1+r_{22}q_2.
$$

因为残差与列空间垂直，$Q^\top(b-QRx^\star)=0$；利用 $Q^\top Q=I$ 得到 $Rx^\star=Q^\top b$。本式用三角回代求解的前提是 $A$ 满列秩，故 $R$ 可逆；它不意味着每个分量的残差都为零。

### 6.2 逐步算出本页例子的 Q 和 R

对上面的 $A=[a_1,a_2]$，用 Gram–Schmidt 说明思路：

1. $a_1=(1,1,0)^\top$，$\|a_1\|=\sqrt2$，故 $q_1=a_1/\sqrt2$，$r_{11}=\sqrt2$。
2. $a_2=(0,1,1)^\top$，$r_{12}=q_1^\top a_2=1/\sqrt2$。
3. 去掉 $a_2$ 沿 $q_1$ 的分量：$\tilde q_2=a_2-r_{12}q_1=(-1/2,1/2,1)^\top$。
4. $\|\tilde q_2\|=\sqrt{3/2}$，故 $q_2=\tilde q_2/\|\tilde q_2\|=(-1,1,2)^\top/\sqrt6$，$r_{22}=\sqrt{3/2}$。

于是

$$
Q=\begin{bmatrix}
1/\sqrt2&-1/\sqrt6\\1/\sqrt2&1/\sqrt6\\0&2/\sqrt6
\end{bmatrix},\quad
R=\begin{bmatrix}\sqrt2&1/\sqrt2\\0&\sqrt{3/2}\end{bmatrix}.
$$

对 $b=(1,0,0)^\top$，$Q^\top b=(1/\sqrt2,-1/\sqrt6)^\top$。先由三角系统第二行求 $x_2=-1/3$，再回代第一行得 $x_1=2/3$。$r=b-Ax^\star=(1,-1,1)^\top/3$ 非零，但 $A^\top r=0$。不同实现可以改变 $Q$ 的列及 $R$ 相应行的符号；应核对 $QR=A$ 和 $Q^\top Q=I$，不必要求逐元素同号。

## 7. 把线性代数放回 SLAM

对残差 $r(x)$ 在当前估计 $x_0$ 附近线性化：

$$
r(x_0\boxplus\delta x)\approx r_0+J\delta x,\qquad
\min_{\delta x}\frac12\|W^{1/2}(r_0+J\delta x)\|^2.
$$

其中 $\boxplus$ 表示按所用参数化施加扰动；位姿常在流形切空间更新。令 $\bar J=W^{1/2}J$、$\bar r=W^{1/2}r_0$，则

$$
H\delta x=g,\qquad H=\bar J^\top\bar J,\quad g=-\bar J^\top\bar r.
$$

实际系统可使用鲁棒核、先验或阻尼，上式是最基本的加权 Gauss–Newton 局部形式。诊断时记清你检查的是哪一次线性化、哪个状态参数化、白化前后，以及阻尼前后。

### 一个一维位姿图：规范自由度是怎样产生的？

假设只有一条相对位移观测 $z$：$r(x_1,x_2)=x_2-x_1-z$。则

$$
J=\begin{bmatrix}-1&1\end{bmatrix},\quad
H=J^\top J=\begin{bmatrix}1&-1\\-1&1\end{bmatrix},\quad
H\begin{bmatrix}1\\1\end{bmatrix}=0.
$$

把两位置同时加同一个数，相对位移不变；因此该问题只有相对位置，没有绝对原点。$H$ 的特征值为 $0,2$。固定 $x_1$ 或加入有明确参考含义的先验可选定坐标基准，但先验同时引入了假设；仅靠相对测量本身没有获得绝对位置。更复杂的 3D 图、单目 BA 或 VIO 的规范自由度依赖实际状态、传感器、先验与运动，不应机械套用统一数量。

**区分两种退化**：规范自由度是由问题整体变换不改变观测产生的结构性不唯一；低视差、弱激励、错误/缺失关联或断开的图也可能造成额外零方向或近零方向。一个单独的奇异值无法告诉你是哪一种：要看右奇异向量的状态分量、观测图、坐标尺度和实际运动，再设计对照实验。

LM 类阻尼例如 $H+\lambda I$（仅作示意）可让当前线性步更容易求，但它增加的是算法正则项；阻尼后可逆不证明原问题已经可观。应分别检查无阻尼和阻尼后的系统。

## 8. Schur Complement：先消去一部分变量

将线性系统分为保留变量 $y$（例如相机/位姿增量）和待消变量 $z$（例如地图点增量）：

$$
\begin{bmatrix}B&E\\E^\top&C\end{bmatrix}
\begin{bmatrix}\delta y\\\delta z\end{bmatrix}
=\begin{bmatrix}v\\w\end{bmatrix}.
$$

第二行是 $E^\top\delta y+C\delta z=w$。**在 $C$ 可逆的前提下**，

$$
\delta z=C^{-1}(w-E^\top\delta y).
$$

代回第一行得到

$$
\underbrace{(B-EC^{-1}E^\top)}_{S}\delta y
=v-EC^{-1}w,\qquad
\delta z=C^{-1}(w-E^\top\delta y).
$$

$S$ 就是 $C$ 关于整个块矩阵的 Schur 补。先解较小的 $S\delta y$，再回代恢复 $\delta z$；实际实现应**解线性方程**，而不是显式计算 $C^{-1}$。如果 $C$ 奇异或病态，不能直接套公式；需检查观测、规范约束、参数化和消元顺序，并在适当条件下使用秩揭示分解、明确假设的阻尼或伪逆。

### 手算并回代

$$
\begin{bmatrix}3&1\\1&2\end{bmatrix}
\begin{bmatrix}y\\z\end{bmatrix}
=\begin{bmatrix}5\\4\end{bmatrix}.
$$

$B=3,E=1,C=2$，所以 $S=3-1\cdot(1/2)\cdot1=5/2$，右侧为 $5-(1/2)4=3$。因此 $y=6/5$，回代 $z=7/5$；代回原系统恰好得到 $(5,4)^\top$。

### 为什么 BA 中有用？

在标准 BA 中，一个重投影因子涉及一帧相机和一个地图点，通常不同时涉及两个不同点。因此按相机/点分块后，$C$ 具有点块对角结构，可以分别求解各点块；但单视图的 2D 重投影通常不足以唯一约束 3D 点，低视差也可能让点块病态。先消去点后，**共同观测同一点的相机之间一般会产生耦合**；这就是填充的来源之一。消元顺序会改变填充、内存与运算成本，而不是改变原始线性系统本身。相机图很稠密时 Schur 后的系统也未必稀疏或便宜。

Schur 不自动修复全局 gauge：即使 $C$ 可逆，$S$ 仍可能奇异。若整体 $H\succ0$，则 $C\succ0$ 且 $S\succ0$；若整体仅半正定，$S$ 可以只半正定。Ceres 的 [Schur 求解说明](https://ceres-solver.readthedocs.io/latest/nnls_solving.html) 给出标准 BA 分块、稀疏和迭代求解形式。

## 9. Schur、边缘化与稀疏性不是同一个概念

| 操作 | 当前做什么 | 后续如何处理被消变量 |
|---|---|---|
| 线性求解中的 Schur 消元 | 在当前线性化系统中临时消去变量，解降维系统 | 回代恢复增量；下一轮原非线性变量仍可重新线性化 |
| 固定窗口边缘化 | 将旧状态的局部线性高斯信息压缩到剩余状态上的先验 | 旧状态通常被丢弃；先验依赖形成时的线性化点，之后不能简单重线性化旧变量 |
| LM 阻尼 | 修改当前线性子问题以控制更新步 | 可与 Schur 组合；阻尼不等于新增真实观测 |

在线性高斯、相关块可逆且变量约定一致时，对被消变量进行最小化与高斯边缘化，会得到相同的保留变量二次信息形式；边缘化还涉及归一化常数。非线性情形中，固定线性化先验只是在当时局部近似意义下成立。直接删除旧变量和相关约束，不等价于保留边缘化先验。

## 10. 可运行的数值验证

以下代码只依赖 NumPy，CPU 上运行；断言和输出验证上文的精确秩亏、近秩亏、QR、规范自由度、Schur 回代与填充。此代码验证小规模合成线性代数关系，**不验证实际 SLAM 系统**。在支持 `python` 和 NumPy 的环境中复制为脚本运行；如已在仓库根目录，可用第 11 节所列命令直接从本 Markdown 提取并执行。

```python
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# 1. 秩、零空间与最小范数解
A = np.array([[1., 2.], [2., 4.]])
u, s, vt = np.linalg.svd(A, full_matrices=True)
null = vt[-1]
b = np.array([3., 6.])
x = np.linalg.pinv(A) @ b
assert np.linalg.matrix_rank(A) == 1
assert np.allclose(A @ null, 0, atol=1e-12)
assert np.allclose(x, [0.6, 1.2])
print('rank-deficient SVD:', s, 'null:', null, 'min-norm:', x)

# 2. 近秩亏与正规方程条件数
for eps in (1e-1, 1e-3, 1e-6):
    j = np.diag([1., eps])
    h = j.T @ j
    cj, ch = np.linalg.cond(j), np.linalg.cond(h)
    assert np.isclose(ch, cj**2, rtol=1e-12)
    print(f'eps={eps:g}, cond(J)={cj:.3g}, cond(J.T@J)={ch:.3g}')

# 3. QR 最小二乘与残差正交
a = np.array([[1., 0.], [1., 1.], [0., 1.]])
b = np.array([1., 0., 0.])
q, r = np.linalg.qr(a, mode='reduced')
x = np.linalg.solve(r, q.T @ b)
residual = b - a @ x
assert np.allclose(x, [2/3, -1/3])
assert np.allclose(a.T @ residual, 0, atol=1e-12)
print('QR least squares:', x, 'residual:', residual)

# 4. 只有相对测量的 1D 位姿图
j = np.array([[-1., 1.]])
h = j.T @ j
assert np.allclose(h @ [1., 1.], 0)
assert np.allclose(np.linalg.eigvalsh(h), [0., 2.])
print('relative-pose eigenvalues:', np.linalg.eigvalsh(h))

# 5. Schur 消元与完整线性系统比较
h = np.array([[3., 1.], [1., 2.]])
g = np.array([5., 4.])
B, E, C = h[:1, :1], h[:1, 1:], h[1:, 1:]
v, w = g[:1], g[1:]
S = B - E @ np.linalg.solve(C, E.T)
rhs = v - E @ np.linalg.solve(C, w)
y = np.linalg.solve(S, rhs)
z = np.linalg.solve(C, w - E.T @ y)
x_schur = np.concatenate((y, z))
assert np.allclose(x_schur, np.linalg.solve(h, g))
assert np.allclose(x_schur, [1.2, 1.4])
print('Schur S:', S, 'solution:', x_schur)

# 6. 两个相机共视一个点，消点后产生相机间耦合
j = np.array([[1., 0., 0., 0.],
              [1., 0., 1., 0.],
              [0., 1., 1., 0.],
              [0., 1., 0., 1.]])
h = j.T @ j
B, E, C = h[:2, :2], h[:2, 2:], h[2:, 2:]
S = B - E @ np.linalg.solve(C, E.T)
assert np.allclose(S, [[1.5, -0.5], [-0.5, 0.5]])
print('camera Schur matrix:\n', S)

print('All numerical checks passed.')
```

最后一个玩具 Jacobian 的第一行是对第一个相机的锚定，其他三行模拟两个相机和两个点的关联；它只说明稀疏结构，不是完整 BA 投影模型。对同一例子移除第一行后，相机 Schur 矩阵有 $(1,1)^\top$ 零方向。

## 11. 自测题与答案

先独立完成，再对照答案。学习完成程度记在 [技能矩阵](../../00_planning/skill_matrix.md)，不因阅读或运行一次示例自动标为已掌握。

1. 对 $J=[-1\;1]$，求秩、零空间和 $J^\top J$ 的特征值。**答**：秩 1；零空间为 $\operatorname{span}\{(1,1)^\top\}$；特征值 $0,2$。
2. 若 $J$ 的奇异值为 $10,1,10^{-3}$ 且满列秩，求 $\kappa_2(J)$ 与 $\kappa_2(J^\top J)$。**答**：分别为 $10^4$、$10^8$；改变变量单位可能改变数值，但可逆缩放不改变精确秩。
3. $J^\top J$ 一定能 Cholesky 分解吗？**答**：它一定半正定，但仅满列秩时正定；普通无主元 Cholesky 对奇异矩阵不保证成功。
4. 只有两个相机相对位姿，优化残差为零，为什么全局位置仍不唯一？**答**：整体变换不改变相对观测；先识别具体模型的 gauge，再选择锚定或适合的表示。
5. 若点块 $C$ 可逆，是否保证完整系统可逆？**答**：不保证；Schur 补 $S$ 仍可能奇异，且此时完整系统也奇异。
6. 两个相机共享一个点，消去该点后会发生什么？**答**：相机块之间通常出现耦合；是否实际非零受具体 Jacobian 和数值结构影响。
7. 两组测试均显示很小残差，但一组的 $\sigma_{\min}$ 极小。可以判断哪组估计更准确吗？**答**：不能仅凭残差和奇异值断言实际误差；需结合噪声、尺度、真值或独立参考和完整模型。小奇异值提示某方向更敏感。

提取并运行本页代码（PowerShell，工作区根目录）：

```powershell
@'
from pathlib import Path
text = Path('01_foundations/optimization/linear_algebra_for_slam.md').read_text(encoding='utf-8')
code = text.split('```python\n', 1)[1].split('\n```', 1)[0]
exec(compile(code, '<linear-algebra-demo>', 'exec'))
'@ | python -
```

下一步建议：修改第 10 节 $\varepsilon$、删去玩具 BA 的锚定行，观察秩、零空间与 Schur 的变化；然后再进入 [BA 与 Jacobian 练习](../../03_exercises/02_bundle_adjustment/README.md)，用真实重投影残差和有限差分检验替代玩具矩阵。

## 12. 参考资料

- [MIT OCW：SVD 课程与讲义](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/positive-definite-matrices-and-applications/singular-value-decomposition/)：奇异值、正交基及子空间。
- [LAPACK Users' Guide：QR Factorization](https://netlib.org/lapack/lug/node40.html)：QR 和正交分解。
- [MIT OCW：Numerical Methods Week 2](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-2/)：条件数、最小二乘及数值稳定性。
- [Ceres：Solving Non-linear Least Squares](https://ceres-solver.readthedocs.io/latest/nnls_solving.html)：正规方程、BA 的 Schur 补及稀疏求解。
- [Ceres：Covariance Estimation](https://ceres-solver.readthedocs.io/latest/nnls_covariance.html)：秩亏、伪逆与 gauge 边界。
- [GTSAM：Fixed-Lag Smoother Example](https://borglab.github.io/gtsam/fixedlagsmootherexample/)：固定窗口的旧状态边缘化与先验。
