# Conv2d Winograd F(2×2,3×3)

实现入口为 `tensor.trans.winograd`，显式选择算法；`tensor.conv` 不自动切换算法。
支持 NCHW 输入、OIHW 权重、groups=1、无 bias，要求 3×3 kernel、stride=dilation=(1,1)、正输入/通道/batch 维度以及非空输出。
padding=(padH,padW) 分别表示高度和宽度两侧的对称零填充，可取不同值。

```mim
plugin tensor;
plugin math;
extern fun conv_wino (x: «4,128,28,28; math.F32», w: «128,128,3,3; math.F32»): «4,128,28,28; math.F32» =
    return (tensor.trans.winograd math.mode.fast @math.f32 @(4,128,128,28,28,3,3)
        ((1,1),(1,1),(1,1)) (tensor.buf x, tensor.buf w));
```

## 算法和数据布局

每个 4×4 输入 tile 产生 2×2 输出。对每对输入、输出通道：

```text
U = G g Gᵀ
V = Bᵀ d B
M[s,co,tile] = Σci U[s,co,ci] × V[s,ci,tile]     s=0…15
Y = Aᵀ M A
```

```text
G  = [ 1    0    0 ]       Bᵀ = [ 1  0 -1  0 ]       Aᵀ = [ 1  1  1  0 ]
     [ ½    ½    ½ ]            [ 0  1  1  0 ]             [ 0  1 -1 -1 ]
     [ ½   -½    ½ ]            [ 0 -1  1  0 ]
     [ 0    0    1 ]            [ 0  1  0 -1 ]
```

两个方向分别应用小 SSA 元组上的一维变换。输入、权重、输出变换均使用已有 `map_reduce`，读操作通过 affine maps 表达，无新增 barrier。
U 按 `[16,Cout,Cin]`、V 按 `[16,Cin,N×ceil(OH/2)×ceil(OW/2)]` materialize，复用 `bmm` 做 16 个 channel contraction。
输出变换只写有效 NCHW 坐标；奇数 OH/OW 会增加输入高端的零填充，保证最后的完整 4×4 读取合法。
目前每次调用都重新变换权重，没有持久化预变换权重接口。

主中间量 U/V/M 分别包含 `16×Cout×Cin`、`16×Cin×Tiles`、`16×Cout×Tiles` 个元素，另有 padded input、packing scratch 和输出。
乘法从每 tile 每通道对 36 次降到 16 次；这是卷积乘法的计数，不是端到端加速保证。

## 数值行为和验证

`math.mode.none` 也不保证与 direct conv 逐位相等：Winograd 改变了浮点计算顺序，并包含减法和 ½ 因子。
当前运行测试覆盖 FP32 fast/none 和 FP64 none，以 double 标量 cross-correlation 校验每个输出。
FP32 误差界为 `2e-5×(1+Σ|xw|)`，FP64 为 `2e-12×(1+Σ|xw|)`；这些是测试容差，不是对任意输入的数学误差上界。
包含随机、零权重、非对称稀疏权重、抵消输入四类数据，检查 finite 输出。
覆盖多 batch、奇数尾块、valid/asymmetric padding、单通道、Cin13/Cout17、Cin130、全标量输入输出，以及 runtime 1/4 线程。
非法 kernel/stride/dilation/空输出/零 batch 有编译失败测试。
最终完整 lit：534 项发现，527 通过、7 unsupported；CTest 46/46 通过；pre-commit 和 git diff --check 通过。

实现还修复了三个由新用例暴露的 lowering 问题：

- 常量/splat buffer materialization 使用 fresh memory token，防止跨函数共享闭合 allocation 后内存依赖串到另一个函数。
- 全 singleton 的 scalar 输入在 padding 前 materialize 成 rank-zero buffer。
- affine literal 在 annex bootstrap 时完成降级，防止其缓存结果在外部函数中留下未降级的 affine.lit。已有 `buffer/tensor_map_reduce.mim` 能回归此问题。

## 测量方法

Apple M2 Pro（8P+4E），32 GiB；同一当前工作树的 Release 编译器，LLVM 22.1.4，clang++ `-O3 -mcpu=native`。
所有 variant 均为 FP32 fast，3×3、stride=dilation=1、padding=1。
6 组 geometry × batch 1/4 × 编译及运行线程设置 1/4，共 24 个配置，每个配置比较 direct、im2col、Winograd。
这里的 direct 是当前分支的 `tensor.conv`，不是另一次 master 测量。

每次调用包含 padding、变换、矩阵乘、packing、分配和输出恢复；计时包含释放输出，无预变换权重缓存。
每个二进制先用独立 double 标量参考验证全部输出，再进行 warm timing。
每轮交错 variant 顺序，校准每个计时 batch 约 5 ms，7 轮取中位数；独立运行 3 次再取中位数。
配置运行顺序轮换；无线程 affinity 固定，因此异构核调度和系统负载仍可能影响结果。
CSV 保存每次独立运行的 7 轮中位数，未保存全部单轮样本。
GFLOPS 使用 direct conv 的等效 FLOP 数，不能解释为 Winograd 实际执行的算术吞吐。
参考缓存校验完整 geometry 和输入/权重 bit fingerprint；参考计算和缓存读取均不计时。

构建、运行、生成汇总：

```sh
python3 docs/benchmarks/conv-winograd/build.py 1 1
python3 docs/benchmarks/conv-winograd/build.py 1 4
python3 docs/benchmarks/conv-winograd/build.py 4 1
python3 docs/benchmarks/conv-winograd/build.py 4 4
python3 docs/benchmarks/conv-winograd/run.py
python3 docs/benchmarks/conv-winograd/summarize.py
```

编译命令、LLVM IR、schedule report、编译器及 IR hash 位于 `.test/winograd/bench-b*-t*/`；原始测量见 `results/`，完整对比见 [table.md](table.md)。

## 当前结论和下一步

当前实现首先提供可验证的 Winograd 路径，尚不能替代默认调度。
最终三次独立运行的中位数：单线程 batch4、Cin=Cout=128、28×28 为 direct 25.1853 ms / im2col 38.8540 ms / Winograd 18.8297 ms；batch4、Cin=Cout=512、7×7 为 54.2909 / 34.8413 / 22.2639 ms。
相对 direct 分别为 1.34× 和 2.44×；小通道和本次所有 batch1 案例仍落后。
4 线程 128-channel 案例中，im2col 16.7236 ms，优于 Winograd 的 18.7171 ms，不能据此把 Winograd 选作通用默认。
现有 BMM 对部分 shape 可以在 16 个变换分量间并行；packing 的部分路径仍是串行，4 线程通常没有明显端到端收益。

下一步应分别测量 weight/input/output transform 和 contraction，用实际 schedule/LLVM IR 定位：

1. 让变换分量在 PE 时固定，消除运行时小元组选取，并沿连续 tile 维向量化；减少重复 tap 读取和中间缓冲区流量。
2. 调整 16 个 contraction 的任务切分，复用不可变 U/V，对 channel/tile 尾块保留 predicate，避免共享可变 packing scratch。
3. 再考虑预变换权重接口、tile pipeline 和 shape 分派；当前测量不足以设定自动选择阈值。

文档、benchmark harness 和 measurements 仅保存在本地，不提交。
