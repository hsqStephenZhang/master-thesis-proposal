# 高阶函数与部分求值如何构造可靠的张量编译变换？

**硕士论文开题报告**  
**研究重点：** 基于已实现的 Torch 前端，研究变换构造的适用条件与循环契约。  
**日期：** 2026 年 10 月 15 日。

## 摘要

张量编译器需要将模型运算转换为高效的循环和内存访问，同时维护源语言语义与变换合法性。MimIR [12] 与多层编译研究 [13] 已展示通过部分求值特化高阶定义的方法，其工程适用条件和构造成本有待案例验证。本文以作者已实现、并为 [13] 提供评估的 MimIR Torch 前端为基础，将算子分解、im2col 和 Winograd 的目标结构定义为库函数，并研究 split/fuse 的组合契约。研究以算法构造和循环契约为核心，辅以一条逐元素 tensor→buffer 转换路径，将融合和通用缓冲区复用作为边界案例。论文将结合前端执行结果、索引验证和同算法的受控实验，报告接口复用、合法性检查、编译成本与生成代码质量。

## 1. 引言

### 1.1 问题：构造替换程序需要哪些工作？

张量编译器把 PyTorch 模型逐步转换为机器代码。每个阶段处理一种抽象：模型算子、通用张量计算、循环或内存访问。一个编译器变换不仅要生成新代码，还要判断新代码是否保持源程序语义，以及变换是否值得执行。

以循环分块为例，下面两个程序计算相同的逐元素映射。分块后的程序额外处理不足一个块的尾部；假设 `N ≥ 0`、`b > 0`，且 `f` 是纯函数。

```text
原程序：                         分块后的程序：
for i in [0, N):                for io in [0, ceil(N / b)):
    y[i] = f(x[i])                  for ii in [0, b):
                                       i = io * b + ii
                                       if i < N:
                                           y[i] = f(x[i])
```

传统的重写实现通常通过编译器 API 构造新循环、替换索引、连接读写并维护图结构。另一种实现把分块后的程序写成接收循环体 `f` 和索引映射的高阶函数，再用部分求值消除静态抽象。两种方式都必须处理合法性；差别在于替换程序如何被描述和构造。

本文关注编译器 pass、算子库和调度库的开发者。这里的实现成本指他们维护的变换代码，包括驱动层和辅助设施，而不是打印出的 IR 文本长度。需要检验的问题是：哪些构造可以复用为普通函数，哪些工作仍需由图分析与转换驱动承担？

### 1.2 方法与已有工作

高阶函数（Higher-Order Function，HOF）接收或返回函数。部分求值（Partial Evaluation，PE）根据静态参数特化程序，为动态输入保留残余程序。对上述例子，循环体和分块方式可以是静态参数，数组内容则在运行时提供。

本文将替换计算写成 MimIR 库中的 `lam` 函数。函数体就是变换后的结构：它组合标量操作、张量运算、访问映射或其他库函数；驱动层负责选择定义、传参和连接替换结果。高阶参数与 PE 使这一层可以复用和特化。

MimIR 已有带依赖类型的函数表示和受控 PE 机制 [12]。《Principles for Multi-Level Compiler Design》已展示 PE lowering、`Tileable` 接口和以函数表示 schedule 的方法 [13]。本论文承接这些机制，为选定变换补充契约、组合检查与成本证据。

本文区分三类工作。算子级算法改写把卷积改成 im2col 加矩阵乘或 Winograd；循环变换调整迭代域与执行顺序；表示转换把张量值转换为 buffer 等目标表示。本文以算法改写和循环变换为核心，通过一条表示转换路径检验接口边界。第 4 节用源程序、目标函数和驱动链解释各自的构造过程。

### 1.3 已完成成果与论文计划

**作者已实现 MimIR Torch 前端。** 它导入 PyTorch FX 图，提供算子与 shape 适配、`torch.compile` 执行入口、模型导出、测试和评估设施，并为研究稿件 [13] 提供评估。前端的设计、实现与执行链路是论文的主要已完成贡献。

后续 12 周围绕两个核心问题展开：库函数适合构造哪些选定变换（RQ1），以及 split/fuse 的组合需要哪些可执行契约（RQ3）。RQ2 限于一条逐元素 tensor→buffer 路径；融合和通用缓冲区复用用于说明全局分析边界。

论文预期交付以下成果：

- **已有实现与更新评估：** 整理前端架构、个人贡献及固定工作负载上的执行结果。
- **RQ1：算法构造案例：** 对 decomposition、im2col、Winograd 建立源与目标对应，记录共享接口及算法专用前提；选 im2col 建立同算法的构造对照。
- **RQ3：循环契约：** 为矩形域上的 split、fuse 及其组合补充前提、尾块条件、拒绝路径与验证；以一例归约尾块检验贡献次数和顺序。
- **受限 RQ2 与成本证据：** 完成 rank-1 逐元素映射的 buffer 实现，测量新增驱动、特化和残余结构。

系统级实验检验前端和后端的整体行为，机制实验固定算法与调度来比较结果构造方式。

## 2. 背景与基本假设

### 2.1 MimIR 中的程序、接口与特化

MimIR 用统一的图表示类型与项。插件通过 axiom 声明操作，并提供归一化或 lowering 规则 [12]。算子可以先保持不透明，供高层分析识别；选定阶段再将其调用替换为实现函数的调用。

PE filter 控制是否在调用点执行 beta 归约。元素类型、rank、标量操作或索引函数等静态参数可以被特化，张量数值与部分维度长度则留在运行时。静态 rank 不意味着每个维度长度也已知。错误的展开策略可能造成代码膨胀或编译不终止，[12, §3.3] 对此有明确讨论。

一个显式 dictionary 是包含类型及其操作的记录，例如 `(T, zero, add, mul)`。其中的 `add`、`mul` 是高阶参数；从类型自动查找 dictionary 的 resolver 则是另一层可选机制。本文采用显式传参的 dictionary 接口。

MimIR 也能用 continuation 表示控制流。Continuation 的参数对应 SSA 中的合流值，包括循环携带的累加器 [12, §3.1.5, §3.2]。这解释了循环为何可以用函数描述，但不自动证明循环重排、并行化或存储复用合法。

### 2.2 正确性与效率是不同义务

类型检查能够约束参数、结果和索引映射的类型关系，但不能单独证明内存访问安全、浮点等价或 buffer 复用正确。[12] 的核心类型安全结果也不能直接外推到任意插件；[12, §3.4.1] 讨论了 axiom 和 normalizer 的安全边界。

对满足前提 `P` 的输入，一个精确变换需要保持可观察行为：

```text
P(s, x) 成立时：observe(F(s, x)) = observe(G(s, x))
```

其中，`s` 是静态参数，`x` 是动态输入。表示转换还需说明源值与目标表示的关系，带 effect 的变换需保持可观察内存行为。Winograd 等浮点算法使用独立的数值误差契约。

[13] 将 PE 变换的论证分为 PE 正确性、调用替换正确性和实现函数等价性。本文采用这一分层思路，并为具体变换补充前提和检查。正确性论证聚焦于本文选定的变换。

效率则分为编译器成本和生成程序成本。前者包括时间、节点数量与内存，后者包括分配、复制、向量化和运行时间。实现代码较短不能代替这两类测量。

## 3. 已完成工作：MimIR Torch 前端

本节说明作者已实现的前端及其与后续研究的关系。实现依据为 `mimir-frontend` 仓库快照 [15]；第 5.1 节报告端到端结果，附录 A 区分源码版本与运行环境。

### 3.1 从 FX 图到 MimIR 计算

前端以 PyTorch FX 图为输入，通过 MimIR Python bindings 构造函数和张量操作。核心链路是：

```text
PyTorch 模型 / FX 图
    → 图节点与输入签名处理
    → 算子参数和语义适配
    → 带 shape/type 信息的 MimIR Torch/tensor 图
    → 应用库中的算子实现，展开为较低层结构
    → MimIR 优化和 lowering
    → LLVM IR、JIT 编译与结果返回
```

`FXGraphTranslator` 维护 FX 节点到已转换值的映射，处理输入、参数、函数调用与输出，并通过转换表选择算子适配器。适配工作包括区分重载、规范化参数、保留标量与张量的区别，以及连接调用和返回。这些适配将 PyTorch 调用语义落实为 MimIR 的参数与类型关系。

`OperatorLibrary` 构造张量操作与所需函数，并连接 Torch 语义接口。当前实现涉及逐元素运算、归约、矩阵乘、卷积、池化、归一化及索引等算子类别。本文按 dtype、属性和输入形状报告各类操作的测试覆盖。

### 3.2 Shape、dtype 与源语言语义

PyTorch 的逻辑 shape 与 MimIR 的值类型并非始终逐维对应。例如，MimIR 归一化可能折叠长度为 1 的轴，但 PyTorch 的 `transpose`、`keepdim` 和 broadcast 仍按逻辑 rank 解释维度。前端通过 shape 记录与类型查询协调这两种视图。

现有 `ShapeEnv`、`shape_of` 和 `ShapeRules` 提供符号维度、broadcast、reduction 和维度操作的处理。本文分别检查类型信息、辅助映射和回退路径对符号身份与维度关系的保留情况。

算子适配还必须保持 dtype、负轴、`keepdim`、`alpha`、空张量和多结果等源语言行为。一个具体例子是 `aten.add` 的 `alpha` 参数：忽略它即使能生成类型正确的加法，也会改变结果。类似地，把有状态更新简单当作 identity，不能保证别名和后续读取的语义。

前端的 `DecompositionPolicy` 显式区分哪些操作交给 PyTorch 分解、哪些保留给 MimIR。该边界使模型导入不必先展开所有高层结构，也为比较不同 lowering 方案提供固定入口。外部规范如 StableHLO [11] 的类型范围和属性要求，也应与通用张量接口的能力分开。具体契约与测试用于验证插件的语义适配。

### 3.3 用库函数定义 Torch 算子语义

前端通过 Torch 插件的 `torch.mim` [16] 连接 PyTorch 操作与张量结构。下面保留两个有代表性的原文实现：带标量参数的加法，以及具有维度检查和结果 shape 的归约。Mish 和交叉熵组合见附录 C。

`A: Arithmetic` 携带元素类型与算术操作，`F: Floating` 提供浮点函数。`«s; T»` 表示 shape 为 `s`、元素类型为 `T` 的数组类型，`#` 是记录投影，花括号标记隐式参数。`anx lam` 注册可由驱动层引用的函数定义。摘录保留函数体，外围模块与依赖见源码。

#### 带 `alpha` 的加法

`torch.binary_impl.add`（`torch.mim:999–1006`）把 `lhs + alpha * rhs` 定义为两个张量库调用：

```mim
anx lam add (A: Arithmetic)
                       {r: Nat, s: «r; Nat»}
                       (lhs: «s; A#R#T», rhs: «s; A#R#T»,
                        alpha: A#R#T): «s; A#R#T» =
    let scaled_rhs = tensor.unary @(A#R#T, A#R#T)
        (torch_scale_scalar A alpha) @(r, s) rhs;
    tensor.binary @(A#R#T, A#R#T, A#R#T)
        A#R#add @(r, s) (lhs, scaled_rhs);
```

`torch_scale_scalar A alpha` 是部分应用后的标量函数，交给 `tensor.unary`；`A#R#add` 交给 `tensor.binary`。函数签名要求两输入 shape 和元素类型相同，前端在调用前完成 broadcast 与 dtype 适配。动态 `alpha` 留在残余程序中。

#### 带前提和结果 shape 的归约

`torch.reduction_impl.mean_dim`（`torch.mim:5207–5221`）将 mean 定义为归约与缩放：

```mim
anx lam mean_dim (F: Floating)
                            {r: Nat, s: «r; Nat»}
                            (dimension: I64)
                            (input: «s; F#A#R#T»)
                          : «torch_reduce_shape (s, torch_canonical_dim @r dimension); F#A#R#T» =
    runtime.require (torch_dim_in_range @r dimension, "mean.dim expects an in-range dim") body
    where
        lam body(): «torch_reduce_shape (s, torch_canonical_dim @r dimension); F#A#R#T» =
            let dim = torch_canonical_dim @r dimension;
            let reduced = torch_reduce_dim (F#A#R#add, F#A#R#_0) (s, dim) input;
            let count = F#from_nat (s#dim);
            tensor.unary @(F#A#R#T, F#A#R#T)
                (torch_div_scalar F#A count)
                @(r - 1, torch_reduce_shape (s, dim)) reduced;
    end;
```

函数签名通过 `torch_reduce_shape` 描述结果 shape；`runtime.require` 检查 signed dimension，并接收主体函数 `body`。随后，库函数规范化维度、求和并按元素数缩放。类型关系、前提检查和计算构造各有明确位置。

#### 源调用、目标函数与驱动

```text
源：torch.binary.add A @shape (lhs, rhs, alpha)
驱动：DecomposeByImpl::apply_impl 选择并应用 torch.binary_impl.add
目标：tensor.unary(scale_scalar A alpha, rhs)
          → tensor.binary(A.add, lhs, scaled_rhs)
```

源 axiom 保留操作名称与阶段边界，目标 `lam` 给出替换计算。Torch 驱动递归处理生成的待分解调用；张量插件的对应入口为 `Lower::lower_via_impl`。PE 特化函数体中的静态参数，高层 axiom 则在相应阶段由驱动连接到实现定义。公开库入口和内部 `_impl` 的可见性由插件接口管理。

### 3.4 执行、诊断与可复现评估

前端实现了名为 `mimir` 的 `torch.compile` backend。它调用 MimIR pipeline，生成 LLVM IR 并 JIT 编译，处理输入传递、结果打包和恢复。这条路径覆盖从图导入到结果返回的完整执行过程。

系统还提供编译缓存、实现指纹、调试产物、编译 manifest 与阶段 profiling。`audit.py` 记录编译分区、算子统计和执行信息；benchmark 与 KernelBench runner 使用独立子进程、资源限制和参考结果检查。这些设施既用于稿件 [13] 的评估，也支持本论文定位失败阶段和比较变换成本。

在检查的快照中，`src/mimir_frontend/` 下 10 个 Python 文件共 9,001 个物理行，不含子模块，包含空行与注释。该数字只帮助说明实现范围；实现规模与语义覆盖、执行结果分别报告。附录 A 记录版本与统计口径。

## 4. 技术方案：用 HOF 与 PE 构造变换

流程为“分析和选择—应用库定义—PE 特化—残余验证”。目标结构先作为库中的 `lam` 函数存在；驱动提供参数和上下文，PE 特化静态结构，验证阶段检查目标操作与接口。

| 研究问题 | 范围与可检验的问题 | 地位 |
|---|---|---|
| RQ1：适用范围 | 在 decomposition、im2col、Winograd 中，哪些构造可共享，哪些前提和结构必须专用？同算法对照中实现与编译成本如何变化？ | 核心 |
| RQ3：循环契约 | split/fuse 组合后，检查器能否维持迭代覆盖、写入唯一性、归约贡献与允许顺序，并拒绝不满足前提的调用？ | 核心 |
| RQ2：受限 lowering | 对 rank-1 逐元素 tensor→buffer 路径，表示参数化函数相对显式 buffer 实现增加多少驱动代码、特化时间和残余节点？ | 一条辅助路径 |

第 4.1–4.2 节给出已有算法原型，第 4.3–4.4 节明确计划新增的接口与验证，第 4.5 节讨论全局分析边界。

### 4.1 算子级算法改写：从卷积到 Im2col

本例完整展示源计算、替换算法与 `lower_via_impl` 的对应。设输入采用 NCHW 布局，权重采用 OIHW 布局；无 groups、无 bias，输入输出空间尺寸为正。本例使用实数乘加；浮点实现另按数值模式验证。为突出索引关系，伪代码采用 stride=dilation=1、3×3 kernel，`Xp` 是按 `(ph,pw)` 补零后的输入。

记 `X` 的 shape 为 `[N,Cin,H,W]`，分别表示 batch、输入通道、高和宽；`W` 的 shape 为 `[Cout,Cin,3,3]`。输出为 `[N,Cout,OH,OW]`，其中 `OH=H+2*ph-2`、`OW=W+2*pw-2`。

原始 cross-correlation 为：

```text
for n, co, oh, ow:
    acc = 0
    for ci in [0, Cin):
        for kh in [0, 3):
            for kw in [0, 3):
                acc += W[co, ci, kh, kw] * Xp[n, ci, oh+kh, ow+kw]
    Y[n, co, oh, ow] = acc
```

Im2col 保持输出语义，将窗口读取提前到列矩阵构造，再调用矩阵乘。目标算法作为库函数定义在 `tensor.mim` 中。下面是 `trans_im2col_impl` 的完整原文（卷积原型 [17]，`tensor.mim:2276`）：

```mim
priv anx lam trans_im2col_impl (R: Ring) {n cin cout h w kh kw: Nat}
                             (stride dilation padding: «2; Nat»)
                             (input: «n, cin, h, w; R#T», weight: «cout, cin, kh, kw; R#T»)
                           : «conv_shape (n, cout, (h, w), (kh, kw), stride, dilation, padding); R#T» =
    let (_, _, oh, ow) = conv_shape (n, cout, (h, w), (kh, kw), stride, dilation, padding);
    let k = cin * kh * kw;
    let p = oh * ow;
    let cols = im2col @(R#T) @(n, cin, h, w) ((kh, kw), stride, dilation, padding) (input, R#_0);
    let cols = compute_at 0 (reshape (k, n, p) (k, n * p) (transpose @(R#T, 3) (1_3, 0_3, 2_3) @(n, k, p) cols));
    let w2 = reshape (cout, cin, kh, kw) (cout, k) weight;
    let prod = product_2d R @(cout, k, n * p) (w2, cols);
    reshape (n, cout, p) (n, cout, oh, ow)
        (transpose @(R#T, 3) (1_3, 0_3, 2_3) @(cout, n, p) (reshape (cout, n * p) (cout, n, p) prod));
```

该函数体依次组合窗口提取、layout 变换、矩阵乘与输出恢复。令 `K=Cin*KH*KW`、`Q=N*OH*OW`，它构造的三个逻辑关系是：

```text
Col[k,q] = Xp[n,ci,oh*sh+kh*dh,ow*sw+kw*dw]
Z[co,q] = sum(k in [0,K), reshape(W,[Cout,K])[co,k] * Col[k,q])
Y[n,co,oh,ow] = Z[co,q]

k = (ci*KH+kh)*KW+kw
q = (n*OH+oh)*OW+ow
```

这里 `im2col`、`reshape`、`transpose`、`product_2d` 是实现可继续分层的库接口。与第 3.3 节的 Torch decomposition 一样，整个替换计算先由函数组合给出，再通过相应阶段展开。

原循环中的一个归约项 `(ci,kh,kw)` 与矩阵乘中的一个 `k` 一一对应；原输出位置 `(n,oh,ow)` 与列号 `q` 一一对应。因此，窗口值进入 `Col[k,q]` 后，乘上的仍是原权重 `W[co,ci,kh,kw]`。这给出索引对应，而不仅是“卷积可以变为 GEMM”的结论。

实际实现支持一般的合法 kernel、stride 和 dilation。对应读取为 `Xp[n,ci,oh*sh+kh*dh,ow*sw+kw*dw]`，`k` 使用 `KH`、`KW` 展平。输出尺寸由 `conv_shape` 计算；上述示例只固定了其中一组参数。

#### 调用替换如何对应到这三个阶段？

当前算法入口为显式的 `tensor.trans.im2col` 调用。应用与实现的关系如下：

```text
应用层：
    tensor.trans.im2col R @shape params (X, W)

Lower::rewrite_imm_App 选择该算子的实现：
    lower_via_impl(app, impl<tensor::trans_im2col_impl>())

lower_via_impl 保持参数次序，重写参数并重新应用：
    trans_im2col_impl R @shape params (X', W')

PE 展开后的算法结构：
    im2col / pad / reshape / transpose
        → product_2d
        → reshape / transpose，恢复 NCHW
```

这里的 `X'`、`W'` 是驱动重写后的动态输入。调用链对应卷积原型 [17]：其 `Lower::impl<Id>()` 暂设 `in_annex_` 后调用 `annex<Id>()`。前端子模块 [15] 的旧版本直接使用 `annex<Id>()`；以上示例统一采用卷积原型的 API 和补丁版本。

`trans_im2col_impl` 先生成 `[N,K,OH*OW]`，再经 transpose 和 reshape 得到 `[K,N*OH*OW]`。它用 `compute_at 0` 将列矩阵暂存在 consumer 根部，以便矩阵乘复用；权重变为 `[Cout,K]`。乘积 `[Cout,N*OH*OW]` 最后恢复为 `[N,Cout,OH,OW]`。

| 伪代码中的步骤 | 当前实现中的构造 | 需要核对的对应关系 |
|---|---|---|
| `Xp` 与窗口读取 | `pad`、`im2col` 的访问映射 | padding 值及 stride/dilation 偏移 |
| `Col[k,q]` | `im2col` 后的 transpose/reshape | `k=(ci*KH+kh)*KW+kw`；`q=(n*OH+oh)*OW+ow` |
| `W2 * Col` | `product_2d R @(Cout,K,N*OH*OW)` | 归约轴为同一 `K`，不混入 batch |
| `Z` 回到 `Y` | 输出 reshape/transpose | 恢复 NCHW，而非改变外部布局 |

PE 把已知的类型、shape 表达式、映射函数和标量操作接入上述算法。后续 lowering 生成循环、分配与内存访问，动态输入的列矩阵在运行时计算。该算法引入中间表示与物化成本，改变了原计算的组织方式。

上表连接逻辑阶段与函数定义。`lower_via_impl` 及后续阶段会继续展开较低层的实现。评估将同时保留替换前、展开后和循环生成后的 IR，以检查这些对应关系。

### 4.2 算子级算法改写：从卷积到 Winograd

Winograd 的源语义仍是第 4.1 节的 3×3 cross-correlation，但目标算法改变了乘法结构。F(2×2,3×3) 使用一个 4×4 输入 tile 计算 2×2 输出。设 `TH=ceil(OH/2)`、`TW=ceil(OW/2)`，输出 tile 号为 `t=(n*TH+th)*TW+tw`。

它使用以下固定变换矩阵，其中 `Bt` 表示 `Bᵀ`，`At` 表示 `Aᵀ`：

```text
G  = [ 1   0   0 ]    Bt = [ 1  0 -1  0 ]    At = [ 1  1  1  0 ]
     [ ½   ½   ½ ]         [ 0  1  1  0 ]         [ 0  1 -1 -1 ]
     [ ½  -½   ½ ]         [ 0 -1  1  0 ]
     [ 0   0   1 ]         [ 0  1  0 -1 ]
```

下面用函数式伪代码展示库函数的分层。为便于阅读，将实际 `trans_winograd_impl` 中的四段局部计算分别命名；`weight_transform`、`input_transform`、`output_transform` 是此处的摘要名称，仅用于解释现有函数体。完整原文见附录 B。

```text
lam conv_winograd(mode, format, shape, padding)(X, W) =
    let F = float_type(format)
    let (OH, OW) = output_shape(shape, padding)
    let TH = ceil(OH / 2)
    let TW = ceil(OW / 2)
    let Tiles = N * TH * TW
    let Xp = pad_for_tiles(X, padding, OH % 2, OW % 2)
    let U = compute_at(0, weight_transform(mode, W))       # [16,Cout,Cin]
    let V = compute_at(0, input_transform(mode, Xp))       # [16,Cin,Tiles]
    let M = bmm(float_ring(mode, F), U, V)                # [16,Cout,Tiles]
    output_transform(mode, M, [N,Cout,OH,OW])
```

三个 transform helper 的函数体对应以下数学映射。`t=(n*TH+th)*TW+tw`；`d[a,b]=Xp[n,ci,2*th+a,2*tw+b]`；`g` 是一个 3×3 权重块。

```text
weight_transform: U[:,:,co,ci] = G * g * transpose(G)
input_transform:  V[:,:,ci,t]  = Bt * d * transpose(Bt)
channel product:  M[a,b,co,t]  = sum(ci, U[a,b,co,ci] * V[a,b,ci,t])
output_transform: z            = At * M[:,:,co,t] * transpose(At)
                  Y[n,co,2*th+i,2*tw+j] = z[i,j]，仅写有效输出
```

当前实现直接在一个 `lam` 中定义 `weight_point`、`input_point`、`output_point`，通过 `map_reduce` 传入这些函数及其访问映射。主函数组合出变换后的算法；固定的小矩阵变换由一维标量函数组合完成。

对于一个输入通道，原算法在 2×2 输出窗口中需要 `4*9=36` 次乘法。Winograd 在变换域中使用 16 次逐元素乘法，再经输出变换恢复这四个输出；代价是额外加减、变换和中间存储。因此，它通过代数变换减少乘法，并引入额外变换成本。

其中，权重的一维变换也是源码中的普通 `lam`（`tensor.mim:2294`）：

```mim
lam winograd_weight (m: Nat) {pe: «2; Nat»} (a b c: math.F pe): «4; math.F pe» =
    let add = math.arith.add m @pe;
    let sub = math.arith.sub m @pe;
    lam half (x: math.F pe): math.F pe = math.arith.mul m @pe (math.conv.f2f pe (0.5:math.F64), x);
    (a, half (add (add (a, b), c)), half (add (sub (a, b), c)), c);
```

它把三个值变为四个变换分量；在两个方向组合后得到 `G*g*Gᵀ`。PE 处理这些函数应用和已知的运算参数，动态权重值仍在运行时参与计算。

当前 pipeline 通过下面的入口应用这个库定义：

```text
tensor.trans.winograd math_mode @format @shape params (X, W)
    → lower_via_impl(app, impl<tensor::trans_winograd_impl>())
    → trans_winograd_impl math_mode @format @shape params (X', W')
```

| 伪代码阶段 | 当前实现 | 对应布局 |
|---|---|---|
| `G*g*Gᵀ` | `weight_point` 调用一维 `winograd_weight`，由 `map_reduce` 构造 | `[4,4,Cout,Cin] → [16,Cout,Cin]` |
| `Bᵀ*d*B` | `input_point`、`winograd_input` 与输入访问映射 | `[4,4,Cin,N,TH,TW] → [16,Cin,Tiles]` |
| 通道归约 | `bmm R @(16,Cout,Cin,Tiles)` | `[16,Cout,Tiles]` |
| `Aᵀ*M*A` 与有效写入 | `output_point`、`winograd_output` 与输出映射 | `[N,Cout,OH,OW]` |

其中 `Tiles=N*TH*TW`。高阶部分体现在传给 `map_reduce` 的点函数和访问映射，以及归约所使用的标量操作。主函数组合这些库定义，保留算法阶段和输出布局的对应。

当前实现对变换后的 U、V 使用 `compute_at 0`，供 BMM 读取。奇数输出尺寸会在输入高端增加补零，使最后一个 4×4 tile 的读取有效；输出变换只写真实输出坐标。它要求 3×3 kernel、stride=dilation=1、正输入维度和非空输出。

Im2col 可以按相同窗口项解释其重排；Winograd 还使用代数恒等变换，并依赖减法和 ½ 因子。Winograd 的 API 因此使用浮点格式与数值模式，并通过独立误差契约验证舍入差异。

Torch decomposition、im2col 和 Winograd 采用相同的库组织：用 `lam` 定义目标计算，再选择定义并特化。各自的数学前提、目标层级与中间存储通过专用参数和检查表达。

这两个例子给出 RQ1 的初始判断：当替换算法结构规则、标量操作与映射能显式传入、前提可检查时，HOF/PE 适合承担结果构造。算法选择、物化收益和数值契约仍需单独建立。驱动显式选择算法，其性能适用范围由测量确定。

### 4.3 一条逐元素表示转换路径（受限 RQ2）

本节计划将 rank-1、连续存储、纯逐元素映射转换为 buffer 读写。输入只读，输出为调用者提供的独立缓冲区，长度为 `N`。这条路径固定分配策略；函数签名适配与读写构造是主要研究对象。

**源程序：**

```text
lam map_tensor(f, N)(x: Tensor[N,T]): Tensor[N,U] =
    tensor.unary(f, x)
```

**目标函数：** 下面是拟议接口的语义草图，`map_into`、`BufferOps` 和 `lower_unary_into` 为研究中的命名。

```text
lam map_into(Ops, f, N)(mem, src, dst) =
    Ops.for_range(N, mem, lam(mem, i) =
        let (mem, value) = Ops.read(mem, src, i)
        Ops.write(mem, dst, i, f(value)))

lam map_buffer(f, N)(mem, src: Ptr[T], dst: Ptr[U]) =
    map_into(BufferOps, f, N)(mem, src, dst)
```

`BufferOps` 提供带 memory token 的迭代、读取与写入函数。PE 特化该记录后，目标是顺序的循环、load 与 store。对照实现直接在 `map_buffer` 中写出同一循环与内存操作。二者使用相同输入表示、独立目标及循环顺序。

**驱动骨架：**

```text
lower_unary_into(source_call, value_map, destination):
    (f, N, x) = inspect_unary(source_call)
    src = value_map[x]
    check_rank1_contiguous(src, destination, N)
    require_separate_allocation(src, destination)
    (mem_out) = apply(map_buffer(f, N), mem_in, src, destination)
    value_map[source_call] = destination
    connect_memory(mem_out)
```

驱动维护值映射、目标传递和 memory token。入口对 `N` 与容量做检查；`N=0` 返回原 token。正确性关系是：目标执行后 `dst[i]=f(x[i])`，且输入内容保持原值。该约定也明确返回张量值如何对应到目标 buffer。

本路径的交付是一对可执行实现、共享驱动和阶段测量。实验改变 `N`、重复调用数量和图共享程度，比较新增接口/helper 行数、特化时间、节点数量以及残余的间接调用。多值表示、跨函数 ABI 转换和存储复用留作后续扩展。

### 4.4 Split/Fuse 的组合契约（核心 RQ3）

[13] 已提供 `Tileable`、`retile` 及函数式调度的构造基础。作者当前分支 [17] 已有 `fuse_par`、分块及部分 predicate 传播与局部检查。本论文将这些条件系统化为契约，补齐组合后的前提与拒绝路径，并增加覆盖和归约顺序的自动化验证。交付以矩形迭代域、纯逐元素计算和一例归约尾块为限。

结构化计算记录输出 shape `So`、迭代域 `D`、parallel/reduction 角色、读取映射 `Aj`、写入映射 `W`，以及标量体和重建函数。若 `φ:D'→D` 将新坐标映回原坐标，读取变为 `Aj∘φ`，写入变为 `W∘φ`；再应用 `χ` 后为 `φ∘χ`。输出 rank 与迭代 rank 分离，使 fuse 能保持原输出布局。

**源程序：** 采用第 1.1 节的逐元素映射 `y[i]=f(x[i])`，域为 `0≤i<N`。

**目标函数：** 以下 `loop`、`split_loop`、`fuse_loop` 为接口草图，`body` 包含原读写。

```text
lam split_loop(N, b, body)() =
    loop(ceildiv(N,b), lam(io) =
        loop(b, lam(ii) =
            let i = io*b + ii
            if i < N:
                body(i)))

# 将相邻独立轴 [A,B] 合为一轴；空域单独返回，B>0 时应用：
lam fuse_loop(A, B, body)() =
    loop(checked_mul(A,B), lam(t) = body(t/B, t%B))
```

Split 的 `φ(io,ii)=io*b+ii` 与尾块条件共同给出源域的唯一覆盖。Fuse 使用 `φ(t)=(t/B,t%B)`；原来的 `W(i,j)` 经组合保留输出 layout。所有读取和写入均位于有效分支内。

**驱动骨架：**

```text
schedule_split(compute, axis, b):
    contract = check_axis_factor_roles_and_extent(compute, axis, b)
    if contract is invalid or unresolved: return diagnostic
    (D_new, phi, valid) = build_split_map(contract)
    # tail 条件也映回原域；实现用短路控制流保护所有访问
    P_new(q) = valid(q) && compute.predicate(phi(q))
    target = retile(compute, D_new, phi, P_new)
    remap_axis_metadata(target, contract)
    return target

schedule_fuse(compute, consecutive_axes):
    check_rectangular_independent_axes_and_roles(compute, consecutive_axes)
    handle_empty_domain_or_check_positive_divisor()
    check_extent_product_overflow()
    return retile(compute, fused_domain, unflatten_map, mapped_predicate)
```

构造层复用 `retile`；新增契约层负责判断可接受的输入、传播 predicate 和迁移轴元数据。相依轴、未知合法性或不支持的归约角色返回诊断。小域穷举验证有效点覆盖、唯一写入和映射组合方向。

归约尾块使用更强的义务。对 `acc=fold(k∈[0,K),step)`，split 后按 `(ko,ki)` 字典序执行，`k=ko*b+ki`；有效点恰好调用一次 `step`，无效点返回原 `acc`。检查同时验证贡献次数和原顺序，浮点重排另受数值模式约束。最终 store 的 guard 仅保护写入；load 和归约更新各自需要有效控制流。

| 本文交付 | 具体检查与验证 |
|---|---|
| Split | 轴有效、`b>0`、extent/索引计算安全、有效覆盖与尾块控制流 |
| Fuse | 相邻矩形独立轴、角色可合并、空域处理、乘积不溢出、输出映射保留 |
| Split/Fuse 组合 | predicate 传播、映射复合、轴元数据一致；非法输入实际终止转换 |
| 一例归约 split | 每个贡献一次、顺序保留、无效 lane 维持累加器 |

TVM [8–9] 提供契约和诊断的参照。Vectorize、parallelize、cache 和 `compute_at` 的完整合法化属于扩展范围；本文只检查核心案例最终生成的循环与访问。

### 4.5 全局分析的边界案例

融合和通用缓冲区复用作为范围边界，用于解释局部函数构造所需的外部信息。本文不为它们设置独立 RQ、实现里程碑或性能实验组。

纯链与共享 producer 的差别可以用同一个函数例子展示：

```text
源图：
    t = map(f, x)
    y = map(g, t)
    z = map(h, t)

共享物化方案：
    lam materialized(f,g,h)(x) =
        let t = map(f,x)
        (map(g,t), map(h,t))

分别内联方案：
    lam inlined(f,g,h)(x) =
        (map(lam(v)=g(f(v)), x), map(lam(v)=h(f(v)), x))

驱动边界：
    plan = analyze_uses_effects_and_cost(source_graph)
    apply(plan.library_function, rewritten_arguments)
```

两个目标函数都可由库表达。驱动选择方案时还需要 use-count、effect 和成本信息；第二种方案重复执行 `f`。本例在论文中用于划清“选择什么”和“如何构造”的职责。

缓冲区复用则以 `t1=update(t0,i,v); r=read(t0,j)` 为反例：把 `t1` 原地写入 `t0` 需要证明后续对旧值的读取仍然安全。Memory token 表达顺序，alias、所有权和生命周期由额外分析提供。第 4.3 节通过独立目标避开这一复用决策；One-Shot Bufferization [7] 的分析/重写分离为后续扩展提供参照。

## 5. 实验评估：已有结果与后续验证

### 5.1 已有结果及其解释范围

前端已用于研究稿件 [13] 的端到端评估。2026 年 10 月 15 日在前端 `619a157` 上的运行记录显示，KernelBench Level 1 为 100/100，Level 2 为 99/100。两份原始 JSON 与日志已归档到附录 A 的材料目录；计数如下：

| 测试集 | PASS | FAIL | INVALID | 证据 |
|---|---:|---:|---:|---|
| Level 1，2026-10-15 | 100 | 0 | 0 | 100 个案例均完成参考结果比较 |
| Level 2，2026-10-15 | 99 | 1 | 0 | 唯一失败为 case 28；旧报告中的 3 个 INVALID 已修复 |

Level 2 case 28 为 `28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py`，在编译执行阶段失败，日志报告 LLVM 后端收到未处理的张量值。这项失败归入 tensor-to-memory/lowering 路径，需与 FX 导入错误区分。

上述计数按各套 100 个缩放 fixture 报告。原始结果文件记录逐例状态与阶段；本次运行的完整命令、依赖锁定和缓存设置仍需随实验环境归档。前端与 Torch 插件的 `scan.diff` 接口缺口另见附录 A；KernelBench 计数与前端单元测试覆盖分别报告。

历史报告用于说明演进：2026-08-24 的 Level 2 为 96 PASS、1 FAIL、3 INVALID；Level 3 的维护 fixture 为 27 PASS，另有 21 个缺少缩放 fixture、2 个依赖缺失。历史报告使用 `size-divisor=16`、240 秒超时、16 GiB 地址空间限制和 512 次 fixed-point 预算，其版本和条件保存在仓库报告 [15] 中。

卷积原型已有独立的端到端初步测量。下表来自 Apple M2 Pro 上的单线程 FP32 fast 运行；均为 3×3、stride=dilation=1、padding=1，计时包含变换和分配/释放，每次调用重新变换权重。

| Shape：`N,Cin,Cout,H,W` | Direct | Im2col | Winograd |
|---|---:|---:|---:|
| `4,128,128,28,28` | 25.19 ms | 38.85 ms | 18.83 ms |
| `4,512,512,7,7` | 54.29 ms | 34.84 ms | 22.26 ms |

Winograd 在这两组大通道案例中更快；batch=1 的六组几何配置在 1/4 线程下均慢于 direct，四线程扩展也有限。附录 D 给出全部 24 行，包括六组 batch=1 单线程的 direct/Winograd：0.1081/0.4254、0.6422/0.8321、5.1725/5.4117、19.9282/21.1382、0.1161/0.6232、0.0389/0.0990 ms。这些结果界定算法的初步适用范围；构造机制的成本另按第 5.3 节测量。

### 5.2 前端实现的评估

前端作为主要实现贡献，需要独立于后端优化进行评估。测试将分别回答：

- **导入是否正确：** 检查重载、标量参数、shape、dtype、逻辑 rank、多输出及不支持输入的诊断。
- **信息是否保留：** 比较静态和符号维度、共享符号、singleton 轴及 PyTorch 特有维度规则在生成图中的表示。
- **能否执行：** 使用 `torch.compile` 与 eager PyTorch 对比数值结果，并记录失败发生在 tracing、导入、lowering、代码生成还是运行阶段。
- **模型覆盖到哪里：** 区分图分区、graph break、原始模型和缩放 fixture，明确参与编译的图与输入。

报告将固定前端与 MimIR 的版本、decomposition policy、依赖、输入配置和资源预算。新增测试集中于既有前端接口、核心变换的前提与已知失败的分类。

### 5.3 变换构造的受控评估

研究使用两层实验。系统级实验从同一 Torch 前端导入模型，检验实际代码路径和性能。机制实验尽量固定算法、schedule、后端和优化选项，只改变结果构造方式，以隔离 HOF/PE 的成本。

| 问题 | 最低交付 | 主要证据 |
|---|---|---|
| 核心 RQ1 | add/mean_dim、im2col、Winograd 的对应；im2col 同算法构造对照 | 共享定义、专用前提、驱动规模、特化成本与残余结构 |
| 核心 RQ3 | 矩形域 split/fuse 及组合，一例归约尾块 | 点覆盖、唯一写入、贡献次数和顺序、拒绝行为 |
| 受限 RQ2 | rank-1 逐元素映射、独立目标 | 显式 buffer 与参数化接口的驱动/helper、时间和节点差异 |

“Direct 与 Winograd 谁更快”回答算法性能问题；“同一 Winograd 算法通过 HOF 或手工构造时成本如何”才回答构造机制问题。Im2col 对照将共享输入、输出布局、物化位置、矩阵乘和调度，只比较 C++ 显式构图与库函数应用；差异项随结果报告。

### 5.4 正确性检查与成本指标

索引测试覆盖长度为 1、空域、非整除因子、奇数输出、通道尾块、非法轴和零因子。小域穷举检查有效点覆盖、写入冲突和归约贡献次数；差分执行检查实际结果。浮点变换单独说明数值模式、参考值与误差口径。

逐元素 buffer 路径检查独立目标、输入保持、容量和空域。诊断测试检查非法输入实际终止转换；残余验证检查目标操作与边界类型。源级差分执行和 LLVM 验证分别提供语义与 IR 结构证据。

| 指标 | 统计口径 |
|---|---|
| 实现成本 | 构造专用代码及全部新增驱动/helper；测试、生成代码和已有设施分别列出 |
| 编译成本 | 各阶段时间、特化/重写次数、中间与残余节点、峰值内存 |
| 扩展性 | 重复实例、共享 DAG、schedule 深度和 unroll 因子增长时的变化 |
| 残余结构 | 函数/闭包、循环、guard、向量操作、分配、复制和并行任务 |
| 运行质量 | 冷编译、预热延迟、组件耗时、线程扩展、误差与分配字节数 |

LOC 采用统一规则；共享设施需计入，而非在一种实现中视作免费。可比的分支或图操作调用数作为辅助指标，不把跨语言圈复杂度直接当作质量排序。运行实验交替顺序、避免并行编译，报告中位数和波动，并明确缓存、权重预变换与复制成本。

如果同类定义可以复用、静态抽象被消除且成本可控，方法得到支持。如果通用接口仍需大量专用图重建，或特化显著膨胀，则应报告其优势有限。论文按测量结果报告收益与代价。

## 6. 相关工作与本论文定位

### 6.1 MimIR 与多层编译

POPL 2025 的 MimIR 论文 [12] 提供核心语言、插件、归一化、PE 和张量 `map_reduce` 的基础。本文复用这些核心机制，研究选定变换的构造接口、成本与附加前提。

《Principles for Multi-Level Compiler Design》[13] 已讨论 affine maps、符号 shape、PE lowering、`Tileable` 与 schedule-as-value，并提供实现规模和系统级评估。作者的 Torch 前端是其中评估的实现贡献。本文结合代码与协作记录说明成果沿革，将新增贡献集中于算法对应、split/fuse 契约与受控成本实验。

该稿件的部分规模统计排除了融合和缓冲区化。本文将这两类保留为分析边界，已有局部构造的统计用于解释对应的局部案例。

### 6.2 部分求值、函数式构造与调度语言

PE、typed interpreter 和 staging 已有长期研究 [1–3]；AnyDSL 使用特化构建高性能库 [10]。Elevate 将优化表达为函数式 rewrite strategy [14]。本文继承这些工作的高阶表达与函数组合方法。

本文围绕现有 Torch→MimIR 链路，为选定算法与循环变换建立可执行契约，并将接口复用、驱动负担和特化成本一起测量。

### 6.3 MLIR 与 TVM

MLIR 的 PDL dialect、PDLL 语言和 Transform dialect 提供声明式重写及变换编排 [5–6]。Dialect Conversion 管理合法性、值映射和类型适配 [4]。本文比较可执行高阶定义与这些构造接口的职责、复用方式和成本。

One-Shot Bufferization已将分析和重写分离 [7]。本文借助这一分离说明第 4.5 节的构造边界。

TVM 的结构化 schedule 与算术分析提供了前提检查和错误处理的参考 [8–9]。本文以选定 API 的契约为对照。OpenBLAS 等运行库用于辅助判断 kernel 质量，构造机制通过同算法实验评估。

## 7. 后续工作计划与范围

前端作为已有主要实现成果进行整理与评估。后续 12 周以核心 RQ1、RQ3 和一条辅助 RQ2 路径为主计划：

| 时间 | 工作与交付 | 验收点 |
|---|---|---|
| 第 1–2 周 | 固定三套代码与数据记录，补齐运行命令和 `scan.diff` 依赖记录；整理前端贡献与当前结果 | 前端环境清单；算法原型补丁可应用；结果分母清楚 |
| 第 3–4 周 | RQ1：完成三类算法的源/目标/驱动对应，建立 im2col 同算法构造对照 | 对应表、数值契约和一对匹配实现 |
| 第 5–7 周 | RQ3：split/fuse 前提、predicate 组合、错误路径；加入一例归约尾块 | 小域穷举、差分执行与非法输入测试 |
| 第 8 周 | 受限 RQ2：rank-1 映射、独立目标、显式 buffer 与参数化接口 | 两种实现生成匹配循环，输入保持 |
| 第 9–10 周 | 运行受控编译成本实验，核对残余 IR 与生成代码质量 | 版本化结果、阶段成本、规模增长曲线 |
| 第 11–12 周 | 分析边界、整理可复现材料并完成写作 | 论证、材料清单和论文 |

范围固定为 CPU、现有 LLVM 后端、选定 Torch 工作负载、矩形迭代域及独立目标的逐元素 lowering。融合与通用缓冲区复用仅作边界讨论；完整调度合法化、通用 alias/SMT 求解、自动调优、GPU 和新 TVM 插件列为后续方向。

主要风险与收缩策略如下：

| 风险 | 应对 | 进度不足时保留的最小交付 |
|---|---|---|
| PE 特化膨胀 | 限制展开，记录阶段节点与时间，保留重复实例/共享图对照 | 一种匹配的算法构造对照及规模数据 |
| 合法性范围扩大 | 对未知前提拒绝；先限定矩形域和同角色轴 | split/fuse 核心覆盖、尾块和组合测试 |
| 前端与后端缺陷混淆 | 固定输入，按 tracing/导入/lowering/执行分类 | 当前前端结果与核心变换案例 |
| 共享设施掩盖成本 | 构造、驱动、helper 分项统计，并给总量 | 完整成本账目与残余结构 |
| 表示接口超过一周预算 | 固定 rank、连续布局和独立目标 | 显式 buffer 基线、接口草图与失败证据 |

收缩始终优先保障 RQ1 和 RQ3。研究期间保持边界案例的讨论性质，避免新增实现目标挤占核心验证。

## 8. 结论

本论文以作者已实现、并用于多层编译研究评估的 Torch→MimIR 前端为基础，研究高阶函数和部分求值构造张量编译变换的适用条件。前端覆盖模型导入、语义适配、编译执行与评估，是主要已完成成果。

后续研究集中于两项交付：用 decomposition、im2col 和 Winograd 说明库函数与目标算法的对应，以及为 split/fuse 建立可执行的组合契约。一条逐元素表示转换路径用于检验接口与驱动成本；共享 producer 和缓冲区复用则解释全局分析的边界。

论文通过受控实验分别衡量接口复用、编译成本和生成代码质量。读者可据此判断选定变换需要哪些函数参数、检查与外部分析，并了解这些构造方式的收益及代价。

## 参考资料

1. Neil D. Jones、Carsten K. Gomard、Peter Sestoft。*Partial Evaluation and Automatic Program Generation*。Prentice Hall，1993。
2. Jacques Carette、Oleg Kiselyov、Chung-chieh Shan。“Finally tagless, partially evaluated: Tagless staged interpreters for simpler typed languages.” *Journal of Functional Programming*，2009。
3. Tiark Rompf、Martin Odersky。“Lightweight Modular Staging: A Pragmatic Approach to Runtime Code Generation and Compiled DSLs.” *GPCE*，2010。
4. MLIR。[Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)。
5. MLIR。[PDL dialect](https://mlir.llvm.org/docs/Dialects/PDLOps/) 与 [PDLL](https://mlir.llvm.org/docs/PDLL/)。
6. MLIR。[Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/)。
7. MLIR。[Bufferization](https://mlir.llvm.org/docs/Bufferization/) 与 [Ownership-based Buffer Deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/)。
8. Apache TVM。[结构化循环调度实现](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/src/s_tir/schedule/primitive/loop_transformation.cc)。该版本使用 `s_tir` 命名。
9. Apache TVM。[Arithmetic analyzer 接口](https://github.com/apache/tvm/blob/main/include/tvm/arith/analyzer.h)。
10. [AnyDSL 项目](https://anydsl.github.io/)，包括 staging 与高性能库工作。
11. OpenXLA。[StableHLO 规范](https://openxla.org/stablehlo/spec)。本文使用它区分外部规范要求与通用张量接口能力。
12. Roland Leißa、Marcel Ullrich、Joachim Meyer、Sebastian Hack。*MimIR: An Extensible and Type-Safe Intermediate Representation for the DSL Age*。*PACMPL*，9，POPL，Article 4，2025，31 页。[DOI: 10.1145/3704840](https://doi.org/10.1145/3704840)。
13. *Principles for Multi-Level Compiler Design*。研究稿件，§7–8。
14. Bastian Hagedorn 等。“Achieving high-performance the functional way: a functional pearl on expressing high-performance optimizations as rewrite strategies.” *PACMPL*，4，ICFP，Article 92，2020。
15. [MimIR Torch 前端](https://github.com/hsqStephenZhang/mimir-frontend/tree/619a15776d91a645fba997c059a3001e8e778dea)。Commit `619a15776d91a645fba997c059a3001e8e778dea`；历史报告见 `docs/kernelbench_*_report.md`。
16. [MimIR Torch 插件](https://github.com/hsqStephenZhang/mimir-torch-plugin/tree/6c41f8eb2313cbb923c2cb92a0a5a6b4d4f9a23d)。Commit `6c41f8eb2313cbb923c2cb92a0a5a6b4d4f9a23d`；`torch.mim` 与 `src/phase/lower.cpp`。
17. [MimIR 卷积原型仓库](https://github.com/hsqStephenZhang/MimIR)。基线 commit `09bfc92ae80bef5a2acdbcb449bbd909a84afdbf` 由附录 A 的 Git bundle 提供，再应用 `conv-winograd.patch`。

## 附录 A：代码、数据与可复现状态

材料目录：[artifacts/pe-hof-2026-10-15](artifacts/pe-hof-2026-10-15/README.md)。它包含代码补丁、逐文件 SHA-256、前端原始结果和卷积完整数据。以下三个快照分别服务于前端评估、源码讲解与卷积实验。

| 材料 | 仓库与版本 | 复现状态 |
|---|---|---|
| Torch 前端 | [15]，`619a157`；MimIR 子模块 [`3a31afc`](https://github.com/hsqStephenZhang/MimIR/tree/3a31afcdef1e00db51ea0de175de4c4b2228443e) | 原始结果已归档；运行依赖锁定待补齐 |
| Torch 算子摘录 | [16]，`6c41f8e` | add、mean_dim、mish、交叉熵定义可逐字定位 |
| 卷积算法原型 | [17]，`09bfc92` + [代码补丁](artifacts/pe-hof-2026-10-15/conv-winograd.patch) | Git bundle 恢复基线，补丁检查通过；包含 Winograd、lowering 修复、测试与测量驱动 |

**前端依赖缺口。** 前端 `operators.py` 请求 `%torch.scan.diff` 和 `%torch.scan.diff_i64`，而上述 Torch 插件快照缺少这两个接口；已知有六个前端测试受影响。因此这组三个源码版本尚未构成完整兼容的前端测试环境。2026-10-15 的 KernelBench 结果已保存在 `frontend/level1.json`、`frontend/level2.json` 及对应日志中；`frontend/manifest.json` 区分作者提供的运行版本和事后检查的依赖，完整运行参数与依赖锁定为待补材料。该问题是论文环境固定阶段的验收项。

**实现规模。** 前端 `src/mimir_frontend/*.py` 共 10 个文件、9,001 个物理行，含空行和注释；tests、scripts 和子模块另计。主要模块为 `translator.py`、`operators.py`、`shape_rules.py`、`utils.py`、`backend.py`、`decomposition.py`、`audit.py`、`model_export.py`。Torch 插件 `torch.mim` 为 7,386 个物理行，按独立仓库统计。

**卷积源码。** Im2col 已存在于本地基线 `09bfc92`；材料中的 `mimir-conv-base.bundle` 提供该提交，前置提交为远端 feature/par 的 `c56d58681549d478afd893ed5b8915d2bfbda979`。Winograd 来自补丁。两者独立于前端子模块，实验直接从 MimIR 编译入口运行。补丁 SHA-256 为 `c7c584ce2ebd4f5cc441f64a9656fef42a044f9d65b417e7765fc963dd3fa12c`。应用方式：

```sh
git clone https://github.com/hsqStephenZhang/MimIR.git mimir-conv
cd mimir-conv
# BASE_BUNDLE 指向材料目录中的 mimir-conv-base.bundle
git bundle verify "$BASE_BUNDLE"
git fetch "$BASE_BUNDLE" HEAD
git checkout --detach 09bfc92ae80bef5a2acdbcb449bbd909a84afdbf
# PATCH 指向材料目录中的 conv-winograd.patch
git apply --check "$PATCH"
git apply "$PATCH"
```

之后按基线构建说明安装依赖、初始化子模块并构建编译器。材料 README 给出测量脚本和编译器路径的配置方式。SHA-256 标识保存的补丁；补丁检查与完整重建分别记录。

**卷积数据。** `conv/results/` 保存 12 份独立运行 CSV，`conv/table.md` 与附录 D 保存 24 行配置汇总。`conv/build-b*-t*.json` 记录原测量编译器、插件和 LLVM IR 的 hash。平台为 Apple M2 Pro（8P+4E）、32 GiB、Release、LLVM 22.1.4；FP32 fast，clang++ `-O3 -mcpu=native`。每次调用计入 padding、变换、矩阵乘、packing、分配和输出释放；无持久化权重缓存。

每个程序先与 double 标量参考比较全部输出，再校准计时批次约 5 ms，每次运行交替三算法顺序，取 7 轮中位数；3 次独立运行再取中位数。单轮样本未保存，表格报告现有中位数；线程未绑核，异构核与系统负载会影响测量。Direct 是同一卷积原型的 `tensor.conv`。

## 附录 B：Winograd 的完整库函数定义

以下原文摘自卷积原型 [17] 加附录 A 补丁后的 `src/mim/plug/tensor/tensor.mim:2294–2397`，包括三个一维 helper 与完整 `trans_winograd_impl`。外围依赖见同文件的 `Ring`、`map_reduce`、`proj_map` 定义。第 4.2 节的具名阶段用于解释这一函数体。

```mim
lam winograd_weight (m: Nat) {pe: «2; Nat»} (a b c: math.F pe): «4; math.F pe» =
    let add = math.arith.add m @pe;
    let sub = math.arith.sub m @pe;
    lam half (x: math.F pe): math.F pe = math.arith.mul m @pe (math.conv.f2f pe (0.5:math.F64), x);
    (a, half (add (add (a, b), c)), half (add (sub (a, b), c)), c);

lam winograd_input (m: Nat) {pe: «2; Nat»} (a b c d: math.F pe): «4; math.F pe» =
    let add = math.arith.add m @pe;
    let sub = math.arith.sub m @pe;
    (sub (a, c), add (b, c), sub (c, b), sub (b, d));

lam winograd_output (m: Nat) {pe: «2; Nat»} (a b c d: math.F pe): «2; math.F pe» =
    let add = math.arith.add m @pe;
    let sub = math.arith.sub m @pe;
    (add (add (a, b), c), sub (sub (b, c), d));
///
/// ### trans_winograd_impl
///
/// U=GgGᵀ and V=BᵀdB are staged before the sixteen channel contractions, M=UV.
/// All batch/spatial tiles share a GEMM column axis; AᵀMA restores the valid NCHW output points.

priv anx lam trans_winograd_impl (m: Nat) {pe: «2; Nat»} {n cin cout h w kh kw: Nat}
                               (stride dilation padding: «2; Nat»)
                               (input: «n, cin, h, w; math.F pe», weight: «cout, cin, kh, kw; math.F pe»)
                             : «conv_shape (n, cout, (h, w), (kh, kw), stride, dilation, padding); math.F pe» =
    let F = math.F pe;
    let (_, _, oh, ow) = conv_shape (n, cout, (h, w), (kh, kw), stride, dilation, padding);
    let th = (oh + 1) / 2;
    let tw = (ow + 1) / 2;
    let tiles = n * th * tw;
    let lo = (0, 0, padding#0_2, padding#1_2);
    let hi = (0, 0, padding#0_2 + oh % 2, padding#1_2 + ow % 2);
    let ph = h + padding#0_2 + hi#2_4;
    let pw = w + padding#1_2 + hi#3_4;
    let padded = pad @(F, 4) (n, cin, h, w) (0, lo, hi) (input, 0:F);

    let coord4 = (0:I8, 1:I8, 2:I8, 3:I8);
    let coord2 = (0:I8, 1:I8);
    let T9 = (F, F, F, F, F, F, F, F, F, I8, I8);
    let T16 = (F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, F, I8, I8);

    fun weight_point (_: F, (x0 x1 x2 x3 x4 x5 x6 x7 x8: F, i j: I8))@tt: F =
        let ii = core.conv.u 4 i;
        let jj = core.conv.u 4 j;
        let r0 = (winograd_weight m @pe (x0, x1, x2))#jj;
        let r1 = (winograd_weight m @pe (x3, x4, x5))#jj;
        let r2 = (winograd_weight m @pe (x6, x7, x8))#jj;
        return ((winograd_weight m @pe (r0, r1, r2))#ii);
    let us = (4, 4, cout, cin);
    let umaps = ‹a: 9; lm (o: «4; affine.Index»): «4; affine.Index» =
        (o#2_4, o#3_4, affine.lit (core.bitcast Nat a / 3), affine.lit (core.bitcast Nat a % 3))›;
    let u = map_reduce @11 @(F, 4, 4) (us, us)
        @(T9, vec.cat @(9, 2) @Nat (‹9; 4›, (1, 1)),
          tuple.cat @(9, 2) @(‹9; «4; Nat»›, ‹2; Nat›) (‹9; (cout, cin, kh, kw)›, (4, 4)))
        (weight_point, ⊥: F) affine.id
        (tuple.cat @(9, 2) @(‹9; «4; affine.Index» → «4; affine.Index»›, ‹2; «4; affine.Index» → affine.Index›) (umaps, (proj_map @(4, 1) 0, proj_map @(4, 1) 1)))
        (tuple.cat @(9, 2) @(‹9; «cout, cin, kh, kw; F»›, ‹2; «4; I8»›) (‹9; weight›, (coord4, coord4)));
    let u = compute_at 0 (reshape us (16, cout, cin) u);

    fun input_point (_: F, (x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15: F, i j: I8))@tt: F =
        let ii = core.conv.u 4 i;
        let jj = core.conv.u 4 j;
        let r0 = (winograd_input m @pe (x0, x1, x2, x3))#jj;
        let r1 = (winograd_input m @pe (x4, x5, x6, x7))#jj;
        let r2 = (winograd_input m @pe (x8, x9, x10, x11))#jj;
        let r3 = (winograd_input m @pe (x12, x13, x14, x15))#jj;
        return ((winograd_input m @pe (r0, r1, r2, r3))#ii);
    let vs = (4, 4, cin, n, th, tw);
    let vmaps = ‹a: 16; lm (o: «6; affine.Index»): «4; affine.Index» =
        (o#3_6, o#2_6,
         affine.op.add (affine.semiop.mul (o#4_6, 2), affine.lit (core.bitcast Nat a / 4)),
         affine.op.add (affine.semiop.mul (o#5_6, 2), affine.lit (core.bitcast Nat a % 4)))›;
    let v = map_reduce @18 @(F, 6, 6) (vs, vs)
        @(T16, vec.cat @(16, 2) @Nat (‹16; 4›, (1, 1)),
          tuple.cat @(16, 2) @(‹16; «4; Nat»›, ‹2; Nat›) (‹16; (n, cin, ph, pw)›, (4, 4)))
        (input_point, ⊥: F) affine.id
        (tuple.cat @(16, 2) @(‹16; «6; affine.Index» → «4; affine.Index»›, ‹2; «6; affine.Index» → affine.Index›) (vmaps, (proj_map @(6, 1) 0, proj_map @(6, 1) 1)))
        (tuple.cat @(16, 2) @(‹16; «n, cin, ph, pw; F»›, ‹2; «4; I8»›) (‹16; padded›, (coord4, coord4)));
    let v = compute_at 0 (reshape vs (16, cin, tiles) v);
    let R: Ring = (F, 0:F, math.arith.add m @pe, math.arith.mul m @pe);
    let prod = bmm R @(16, cout, cin, tiles) (u, v);

    fun output_point (_: F, (x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15: F, i j: I8))@tt: F =
        let ii = core.conv.u 2 i;
        let jj = core.conv.u 2 j;
        let r0 = (winograd_output m @pe (x0, x1, x2, x3))#jj;
        let r1 = (winograd_output m @pe (x4, x5, x6, x7))#jj;
        let r2 = (winograd_output m @pe (x8, x9, x10, x11))#jj;
        let r3 = (winograd_output m @pe (x12, x13, x14, x15))#jj;
        return ((winograd_output m @pe (r0, r1, r2, r3))#ii);
    let ys = (n, cout, oh, ow);
    let ymaps = ‹a: 16; lm (o: «4; affine.Index»): «3; affine.Index» =
        (affine.lit (core.bitcast Nat a), o#1_4,
         affine.op.add (affine.semiop.mul (o#0_4, th * tw),
             affine.op.add (affine.semiop.mul (affine.semiop.floordiv (o#2_4, 2), tw),
                            affine.semiop.floordiv (o#3_4, 2))))›;
    lam row_index (o: «4; affine.Index»): affine.Index = affine.semiop.rem (o#2_4, 2);
    lam col_index (o: «4; affine.Index»): affine.Index = affine.semiop.rem (o#3_4, 2);
    map_reduce @18 @(F, 4, 4) (ys, ys)
        @(T16, vec.cat @(16, 2) @Nat (‹16; 3›, (1, 1)),
          tuple.cat @(16, 2) @(‹16; «3; Nat»›, ‹2; Nat›) (‹16; (16, cout, tiles)›, (2, 2)))
        (output_point, ⊥: F) affine.id
        (tuple.cat @(16, 2) @(‹16; «4; affine.Index» → «3; affine.Index»›, ‹2; «4; affine.Index» → affine.Index›) (ymaps, (row_index, col_index)))
        (tuple.cat @(16, 2) @(‹16; «16, cout, tiles; F»›, ‹2; «2; I8»›) (‹16; prod›, (coord2, coord2)));
```


## 附录 C：补充 Torch 算子摘录

以下原文来自 Torch 插件 [16]，省略外围模块与导入。

### C.1 标量 Mish 与张量提升

`torch_mish_scalar`（`torch.mim:4971–4975`）与 `torch.activation_impl.mish`（`5081–5085`）分别定义标量语义和张量遍历：

```mim
lam torch_mish_scalar (F: Floating)
                      (x: F#A#R#T): F#A#R#T =
    let softplus = torch_softplus_scalar F
        (F#from_nat 1, F#from_nat 20) x;
    F#A#R#mul (x, torch_tanh_scalar F softplus);
```

```mim
anx lam mish (F: Floating)
    {r: Nat, s: «r; Nat»}
    (input: «s; F#A#R#T»): «s; F#A#R#T» =
    tensor.unary @(F#A#R#T, F#A#R#T)
        (torch_mish_scalar F) @(r, s) input;
```

标量函数复用 softplus、tanh 和乘法，张量层复用 `tensor.unary`。

### C.2 组合式交叉熵

`torch.loss_impl.cross_entropy_mean_2d`（`torch.mim:6637–6651`）实现 `(N,C)` logits、`(N)` class target 的 mean-reduced 子集：

```mim
anx lam cross_entropy_mean_2d (F: Floating)
    {n c: Nat}
    (input: «n, c; F#A#R#T», target: «n; I64»): F#A#R#T =
    let log_probs = normalization.log_softmax F
        @(2, (n, c)) (1I64, ff) input;
    let target_2d = tensor.reshape @(I64, 1, 2)
        (n,) (n, 1) target;
    let selected_2d = indexing.gather
        @(F#A#R#T, 2, (n, c), (n, 1))
        1I64 (log_probs, target_2d);
    let selected = tensor.reshape @(F#A#R#T, 2, 1)
        (n, 1) (n,) selected_2d;
    let losses = tensor.unary @(F#A#R#T, F#A#R#T)
        (torch_neg_scalar F#A) @(1, (n,)) selected;
    reduction.mean_all F @(1, (n,)) losses;
```

函数组合 log-softmax、reshape、gather、取负与 mean；源接口与各子算子在相应阶段连接到实现。

### C.3 Add 的阶段接口

`torch.binary.add`（`torch.mim:893–896`）的 axiom 与第 3.3 节的目标函数对应：

```mim
axm add: [A: Arithmetic]
                  → {r: Nat, s: «r; Nat»}
                  → [lhs: «s; A#R#T», rhs: «s; A#R#T», alpha: A#R#T]
                  → «s; A#R#T»;
```

## 附录 D：卷积完整初步测量

6 组几何配置 × batch 1/4 × 编译及运行线程设置 1/4，共 24 行；每行比较三个算法。条件见附录 A，原始 CSV 和汇总脚本见材料目录。比值大于 1 表示 Winograd 相对 direct 更快。

| Shape (N,Cin,Cout,H,W) | Threads | Direct ms | Im2col ms | Winograd ms | Direct / Winograd |
|---|---:|---:|---:|---:|---:|
| 1,16,16,32,32 | 1 | 0.1081 | 0.2702 | 0.4254 | 0.25× |
| 1,32,64,28,28 | 1 | 0.6422 | 1.4263 | 0.8321 | 0.77× |
| 1,128,128,28,28 | 1 | 5.1725 | 8.9977 | 5.4117 | 0.96× |
| 1,512,512,7,7 | 1 | 19.9282 | 10.2071 | 21.1382 | 0.94× |
| 1,3,64,32,32 | 1 | 0.1161 | 0.1915 | 0.6232 | 0.19× |
| 1,13,17,17,19 | 1 | 0.0389 | 0.1344 | 0.0990 | 0.39× |
| 1,16,16,32,32 | 4 | 0.1071 | 0.1606 | 0.3887 | 0.28× |
| 1,32,64,28,28 | 4 | 0.6412 | 0.7575 | 0.8383 | 0.76× |
| 1,128,128,28,28 | 4 | 5.1817 | 3.8258 | 5.3860 | 0.96× |
| 1,512,512,7,7 | 4 | 19.8575 | 10.2175 | 21.1719 | 0.94× |
| 1,3,64,32,32 | 4 | 0.1153 | 0.1189 | 0.6131 | 0.19× |
| 1,13,17,17,19 | 4 | 0.0391 | 0.1363 | 0.0995 | 0.39× |
| 4,16,16,32,32 | 1 | 0.5013 | 1.2388 | 1.4538 | 0.34× |
| 4,32,64,28,28 | 1 | 3.1186 | 5.7788 | 7.5481 | 0.41× |
| 4,128,128,28,28 | 1 | 25.1853 | 38.8540 | 18.8297 | 1.34× |
| 4,512,512,7,7 | 1 | 54.2909 | 34.8413 | 22.2639 | 2.44× |
| 4,3,64,32,32 | 1 | 0.4328 | 0.8131 | 2.6215 | 0.17× |
| 4,13,17,17,19 | 1 | 0.1520 | 0.5974 | 0.3559 | 0.43× |
| 4,16,16,32,32 | 4 | 0.5055 | 0.6423 | 1.3946 | 0.36× |
| 4,32,64,28,28 | 4 | 3.1304 | 3.4280 | 7.4283 | 0.42× |
| 4,128,128,28,28 | 4 | 25.1230 | 16.7236 | 18.7171 | 1.34× |
| 4,512,512,7,7 | 4 | 54.2802 | 34.8766 | 22.3429 | 2.43× |
| 4,3,64,32,32 | 4 | 0.4312 | 0.5018 | 2.4664 | 0.17× |
| 4,13,17,17,19 | 4 | 0.1526 | 0.5979 | 0.3548 | 0.43× |
