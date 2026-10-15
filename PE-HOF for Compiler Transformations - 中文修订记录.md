# 中文 Proposal 修订记录

**日期：** 2026 年 10 月 15 日。  
**正文：** [中文修订版](<PE-HOF for Compiler Transformations - 中文修订版.md>)。  
**依据：** Sebastian Hack，[Writing a Thesis/Paper](https://compilers.cs.uni-saarland.de/people/hack/writing.php)。本次已阅读页面正文；网页标注最后修改时间为 2025 年 7 月 21 日。

本文保存写作调整、评审对照和文档核对记录，避免把修订过程放进正式 proposal 的研究论述。英文修订稿与最初原稿不在本次修改范围内。

## 1. 根据写作指南进行的调整

指南列出摘要、引言、背景、技术主体、评估、相关工作和结论的常见结构，也明确允许根据内容调整。本文是开题报告，因此将尚未开展的实验写成计划，不把预期结果写成已完成结果。

| 指南要求 | 原中文稿的具体问题 | 本次调整 |
|---|---|---|
| 摘要短、清楚，覆盖动机、问题、方法、结果和结论 | 三段背景与方法说明较长，贡献定位不集中 | 压缩为五句；以已有 frontend 和初步实验说明结果基础，最后陈述研究目标 |
| 引言快速说明问题、贡献与结果 | 贡献在靠后章节；读者需跨章节收集主线 | 用逐元素循环分块引入问题；第 1.3 节列出已完成成果与三组后续任务 |
| 用简明例子说明问题，并复用例子解释方法 | 主要是机制描述，缺少原算法与 `_impl` 的具体对应 | 第 4.1 节给出完整 im2col 前后伪代码、索引和调用链；第 4.2 节给出 Winograd 的阶段与实现对应 |
| 背景只提供理解后文所需的信息 | 类型、PE、安全边界和已有论文论述多处重复 | 合并为第 2 节；保留静态/动态、dictionary、控制流和正确性义务 |
| 技术主体精确、可复现 | 多次强调原则，但实现与证据来源散落 | 单列前端架构；算法映射注明函数和布局；附录列出代码快照、数据目录与条件 |
| 相关工作需要比较相同与不同之处 | 多处仅列参考或反复否定过度主张 | 第 6 节逐组比较 MimIR、PE/staging、MLIR、TVM 与本文问题的关系 |
| 结论让读者知道能期待什么 | 缺少独立结论 | 第 8 节总结已完成成果、初步发现、适用条件和后续要建立的证据 |
| 主动语态、短句、简单用语 | 多个长段落夹杂“可能、应该、不是”等重复限定 | 用具体主体说明驱动层、构造层与检查器做什么；限制集中在相关义务和范围中 |
| 只看引言和结论也能了解工作 | frontend 容易被读成辅助平台 | 将前端写入标题、摘要、引言贡献列表、独立技术章节和结论 |

“五句摘要”采用了指南的建议，不把建议误写成所有论文必须遵守的格式。中文不机械套用英文冠词、连字符等语言规则，但遵循清楚、简洁和面向读者的原则。

## 2. 根据本轮补充明确的技术与贡献叙述

### 2.1 Torch 前端是主要实现贡献

已检查 `/Users/zc/courses/compiler/ml-compiler/mimir-frontend`，包括 README、核心源码、测试入口与历史覆盖报告。前端主仓库检查版本为 `619a15776d91a645fba997c059a3001e8e778dea`。

新版第 3 节分别说明 FX 导入、算子和 shape 适配、JIT/backend、缓存与诊断、测试和 benchmark。它们构成已完成的实现工作，不再只在评估部分称作“平台”。现有前端曾用于《Principles for Multi-Level Compiler Design》的 evaluation；本文保留这一成果关系，不把它归为未来待实现事项。

只读源码检查还发现，README 中存在早期的简短算子支持列表，旧 dynamic shape 文档也保留了阶段计划。新版依据当前源码说明模块职责，并将报告中的覆盖结果绑定到各自版本，没有把不同日期的描述拼成当前覆盖率。

### 2.2 算法改写必须看见源和目标

Im2col 示例以卷积为源，分别列出列矩阵、矩阵乘和 NCHW 恢复，明确：

```text
k = (ci*KH + kh)*KW + kw
q = (n*OH + oh)*OW + ow
```

随后将伪代码映射到 `trans.im2col`、`lower_via_impl`、`trans_im2col_impl`、`compute_at 0` 和 `product_2d`。文中区分逻辑算法阶段与最终残余 IR，不声称所有高层名称都能在最终 dump 中保留。

Winograd 示例列出 G、Bᵀ、Aᵀ 矩阵和四个阶段，并将它们映射到 `weight_point`、`input_point`、BMM 和 `output_point`。文中区分实数等价与浮点误差，也区分算法改写和循环重索引。

当前 C++ helper 的实际名称是 `lower_via_impl`；`rewrite_via_template` 是原稿对方法的称呼。新版没有创建一个不存在的 API 来解释实现。

### 2.3 已有论文与新研究的边界

POPL 2025 MimIR 论文提供类型、插件和 PE 基础；本地文件 `ssa-is-still-fp.pdf` 的题名是《Principles for Multi-Level Compiler Design》，其第 7 节已经讨论 PE lowering 与 `Tileable`。正文明确承接这些结果。

前端设计与实现是作者已完成的贡献。后续研究则集中在变换分类、表示转换接口、循环契约、分析依赖和成本证据。稿件共享的系统结果与 thesis 的受控实验分开报告，既不重复声称既有机制，也不抹去已经完成的个人工作。

## 3. PR #1 评审意见与新版对应

依据：[proposal PR #1](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1) 的总体评审、行内评论和作者回复。下表重新映射到本次正文位置；删除的旧例子不必为了逐项回复而重新塞回研究主体。

| 评审意见 | 本次处理 | 正文位置 |
|---|---|---|
| [明确做什么、不做什么和如何评估](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#pullrequestreview-4721563960) | 已完成前端、计划贡献、评估与范围分别列出 | 第 1.3、3、5、7 节 |
| [不能从语义分散或对 C++ 的批评推断多数 pass 都难写](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3602206042) | 从构造步骤与合法性工作解释问题，用案例检验表达范围 | 第 1.1、4 节 |
| [IR 不面向手写，为什么关注 boilerplate](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3612678048) | 研究对象是 pass、算子库和调度库作者维护的代码 | 第 1.1 节 |
| [from/to operators 含义不清](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3612689694) | 用源卷积与替换算法逐项解释语义对应 | 第 4.1、4.2 节 |
| [PDL 和 PDLL 不是两个 dialect](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3612696829) | 区分 PDL dialect 和 PDLL 语言 | 第 6.3 节 |
| [MimIR Def 也由 C++ 对象实现](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613145197) | 不再用“是否是内存对象”建立差别，比较构造接口和求值机制 | 第 2、4.1、6.3 节 |
| [避免反复批评 C++](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613155585) | 删除情绪化判断，以受控成本比较为依据 | 第 1、5.3、6.3 节 |
| [redundant/messy 需要量化](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3629044627) | LOC 计入共享设施，辅以时间、节点和可比图操作统计 | 第 5.4 节 |
| [解释 rewrite_via_template 做了什么](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613280465) | 展示源调用、helper、实现调用及算法展开 | 第 4.1、4.2 节 |
| [could 不等于已完成成果](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613287201) | 已完成成果和后续目标使用不同状态，历史结果标注版本 | 第 1.3、3、5、7 节 |
| [sort lowering 是插件选择，不是核心规定](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613321757) | sort 不再作为主体案例；保留“核心机制/插件实现/源规范”区分 | 第 2.1、3.2 节 |
| [dictionary 含义不清](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3614196723) | 定义为显式操作记录，与自动查找机制分开 | 第 2.1 节 |
| [compare_type 在已有比较函数时是否冗余](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613821568) | 主体不再依赖该旧结构；若保留标签，需说明元数据/选择用途，不能默认与函数一致 | 本记录及第 2.1 节的接口原则 |
| [为何限制为 StableHLO 类型，量化具体缺什么](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613977264) | 通用接口与外部规范分开；量化若纳入需定义存储/计算类型、scale 和 zero-point，本轮不新增完整量化目标 | 第 3.2、7 节及本记录 |
| [resolver 是否必要](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613983879) | 显式 dictionary 已足够表达，自动解析为可选层 | 第 2.1、7 节 |
| [有符号/无符号是插件设计，不是核心类型系统缺陷](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3613998222) | dtype 支持按实现契约说明，不再据此提出核心类型系统不足 | 第 2.1、3.2 节 |
| [新建 torch 插件为什么能解决语义问题](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3629077807) | 用 alpha、rank、dtype、状态更新等实际义务说明，新插件本身不是保证 | 第 3.2 节 |
| [先转 TVM IR 的收益是什么](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#discussion_r3629098397) | 不设为核心步骤，仅在回答明确对照问题时考虑 | 第 7 节 |
| [重复内容、术语与句子结构影响可读性](https://github.com/hsqStephenZhang/master-thesis-proposal/pull/1#pullrequestreview-4721563960) | 重写摘要和引言，合并背景，补结论，移出过程记录 | 全文 |

## 4. 核对记录

- 已核对写作指南全文；没有把其建议全部解释成硬性论文格式。
- 已对照当前 MimIR 的 `trans_im2col_impl`、`trans_winograd_impl` 与 `lower_via_impl` 检查名称、布局和调用关系。
- 使用独立的小规模计算核对了文中 im2col 索引及 Winograd 三个矩阵：多 batch/通道、奇数尺寸、valid 与非对称 padding、全 singleton 案例均与 direct cross-correlation 一致。这是伪代码核对，不是新一轮编译器测试或 benchmark。
- 前端描述来自只读代码检查；历史覆盖结果使用各报告注明的 commit。本次没有运行或修改 frontend。
- 最终检查包括摘要句数、章节与链接、公式代码块、参考条目，以及原稿和英文稿未改变。

正文从修订前的 15,048 个汉字缩减为约 8,600 个，仍保留四个研究问题，并新增前端技术章节及两个算法对应实例。这是上一轮写作结构调整的统计；本轮补入源码摘录后的篇幅不沿用此数字。


## 5. 本轮补充：用 Library 函数呈现变换后的结构

本轮保持论文主线，但将 decomposition、im2col 与 Winograd 的目标结构统一呈现为 library 中的 `lam`/function 定义。源 axiom 可保留为阶段入口，驱动层选择并应用库定义；PE 特化函数体，后续阶段处理尚未展开的低层接口。本文没有修改任何插件接口或计算实现。

### 源码摘录

Torch 定义来自前端 worktree 的 `MimIR/extra/torch/torch.mim`，独立仓库为 `mimir-torch-plugin`，检查版本 `6c41f8eb2313cbb923c2cb92a0a5a6b4d4f9a23d`。

| 定义 | 源行 | 展示的内容 |
|---|---:|---|
| `torch.binary_impl.add` | 999–1006 | `alpha` 参数、部分应用的标量函数与 unary/binary 组合 |
| `torch_mish_scalar`、`torch.activation_impl.mish` | 4971–4975、5081–5085 | 标量 library 复用与 tensor 提升分层 |
| `torch.reduction_impl.mean_dim` | 5207–5221 | shape 关系、signed dim 检查、归约与缩放 |
| `torch.loss_impl.cross_entropy_mean_2d` | 6637–6651 | log-softmax、gather、negation、mean 的复合实现 |
| `torch.binary.add` | 893–896 | 源阶段接口与 impl 的类型对应 |

正文摘录上述函数体原文，保留其外围模块名与依赖说明。Torch 递归 decomposition 驱动的实际名称是 `DecomposeByImpl::apply_impl`；tensor 的对应 helper 是 `Lower::lower_via_impl`，文中没有混称二者。

### 算法改写的函数定义

- 第 4.1 节完整摘录 `trans_im2col_impl`，以函数组合代替仅列命令式目标循环；保留源卷积、展平关系和调用替换过程。
- 第 4.2 节使用带 `lam` 的函数式摘要，解释 Winograd 的 library 分层，并摘录真实 `winograd_weight`。摘要中的 helper 名字明确标记为论文层命名，不冒充现有 API。
- 附录 B 保留 Winograd 三个一维 helper 与 `trans_winograd_impl` 的完整连续源码，使读者能对照其 map_reduce、shape、访问映射和 BMM。

本轮以逐字源码对照、源码位置、Markdown 结构和引用检查验证文档。没有执行插件重建、frontend 测试或新 benchmark，现有历史结果与数值验证范围不变。


## 6. 本轮评审：收缩范围、补齐骨架与可复现材料

### 6.1 主计划与技术叙述

主计划改为 RQ1（库函数构造的适用范围）与 RQ3（split/fuse 契约）。RQ2 固定为 rank-1、连续布局、独立目标的逐元素 tensor→buffer 路径；fusion、通用 bufferization 不再作为独立 RQ 或实现里程碑。第 7 节重新安排 12 周验收点，并将风险收缩限制在上述主范围内。

保留第 1.1 节的分块例子、第 4.1 节的完整 im2col 定义、索引等式与对应表、第 2.2 节的正确性/效率义务，以及第 5.3 节的算法性能/构造成本区分。第 4.3、4.4 节新增明确标记的源→目标函数→驱动草图，说明它们是计划实现的接口。第 4.4 节将新增工作定位为已有 Tileable/retile 之上的前提检查、predicate 组合和自动化验证。

正文只保留 add、mean_dim 两个 Torch 摘录，mish、交叉熵与 add axiom 移到附录 C；完整 Winograd 保留附录 B。标题改成研究问题，摘要直接引用 [12][13]，统一库/前端/张量等中文术语，减少防御性否定句。

### 6.2 证据更新

找到原始运行材料：

`/tmp/claude-501/-Users-zc-courses-compiler-ml-compiler-mimir-frontend/f3d27865-6dcc-47cd-8dd9-371628be1d19/scratchpad/level1.json` 与 `level2.json`，对应日志位于同目录。文件时间为 2026-10-15；逐例计数确认为 Level 1 100 PASS、Level 2 99 PASS/1 FAIL/0 INVALID。唯一失败为 case 28，阶段 compile_execute，错误是 LLVM 后端收到未处理的 tensor 值。已将结果逐字复制到 proposal 的本地 artifacts 目录。运行源码版本 619a157 由作者确认；原始文件没有完整运行环境，不能用文件时间或事后 HEAD 补造命令和依赖 lock。

前端源码请求 `%torch.scan.diff`/`diff_i64`，插件 6c41f8e 未定义它们；作者报告六个前端单测受影响。正文明确将这点和 KernelBench 结果分开。尚需作者给出当日完整命令、fixture/缓存配置和运行依赖记录。

卷积 12 份 CSV 已独立计算每组三次中位数，逐项匹配 24 行完整表。全部 batch=1 行加入附录 D。原始单轮样本未保存，只能报告已存中位数。

### 6.3 源码快照与 API 核对

前端 MimIR 子模块 3a31afc 使用 `annex<...>()`；卷积 worktree 的 `Lower::impl<Id>()` 包装 `annex<Id>()` 并暂设 in_annex_，所以第 4.1/4.2 节统一按卷积原型写 `impl<...>()`。这属于版本差异，未修改代码来强行统一。

卷积基线 09bfc92 已包含 im2col。将当前五个 tracked 源文件增量、新增 lit 测试与四份 benchmark 驱动导出为 `artifacts/pe-hof-2026-10-15/conv-winograd.patch`，以 SHA-256 固定；在独立基线 archive 上通过 `git apply --check`、实际应用和逐文件字节一致性检查。由于基线 09bfc92 尚未包含在 remote-tracking 分支中，另导出小型 Git bundle，以已知远端 feature/par 提交 c56d586 为前置。在新仓库从前置和 bundle 恢复的精确 commit 为 09bfc92，补丁检查同样通过。没有修改源 worktree/index、创建 commit 或推送。

原测量 build.py 的 clang++ 绝对路径属于环境设置，材料 README 明确要求复现时替换为对应 LLVM 22.1.4 工具路径。材料包含基线、补丁、结果、原 build/IR hashes 与文件校验单；本轮未重建、运行测试或 benchmark。

### 6.4 引用元数据与起草信息

正文删除 Anonymous Author(s)、本地路径、用户提供等起草信息。[13] 暂按《Principles for Multi-Level Compiler Design》，研究稿件，§7–8 引用。已向作者询问应使用的作者署名与投稿状态；答复前不推测。正式交稿前需由作者/导师确认这两项元数据。

本地 PDF 对应关系：`/Users/zc/thesis/lumh25.pdf` 为 [12]；`/Users/zc/thesis/ssa-is-still-fp.pdf` 为 [13]。参考稿件的占位 venue/DOI 不作为发表事实。写作指南与 PR 意见对照留在本记录，正文不再放修订过程说明。

### 6.5 本轮验证范围

核对 Markdown 代码围栏、17 条引用、各 API 名称与来源，逐字检查重新安置的函数摘录，验证正文无 /Users 或 /tmp 路径、无 RQ4 交付；核验前端状态计数、卷积汇总和补丁适用性。英文版本与原稿保持不变。源代码测试结果沿用历史记录，本轮完成的是文档和材料校验。
