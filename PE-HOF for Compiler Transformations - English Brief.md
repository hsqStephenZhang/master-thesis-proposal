# Expressing Tensor Compiler Transformations with Higher-Order Functions and Partial Evaluation: Case Studies in Lowering and Loop Transformation

**Master's Thesis Proposal (brief version)**  
**Date:** 16 October 2026. The full proposal with source excerpts, appendices, and reproducibility details is a separate document.

## Abstract

A tensor compiler rewrites model operators into efficient loops and memory accesses while preserving the semantics of the source language. In the MLIR and TVM paths considered here, lowering and loop-transformation primitives are compiler code, and the rank and shape correspondence between an operator's signature and its implementation is maintained across separate definitions. MimIR and recent work on multi-level compilation already demonstrate another route: implementation functions, the tiling interface, and schedules are typed values in the IR, and partial evaluation specializes them into loops [1, 2]. This thesis studies the source/target correspondence and explicit preconditions of selected transformations, the construction cost relative to an existing C++ implementation, and the schedules an expert can express through this route. Building on the author's MimIR Torch frontend, this thesis expresses three layers of lowering with higher-order functions, from operator decomposition through convolution algorithm substitution to buffer operations materialized as loops, and uses the buffer-to-loop layer for a controlled cost comparison between C++ and higher-order-function construction. For loop transformations, it builds expert schedules for GEMM, GEMV, and CONV by hand through a schedule API, with single-threaded OpenBLAS as the performance reference, and documents composition contracts for split, axis fusion (fuse), and interchange, supported by targeted regression, rejection, and differential-execution tests.

## 1. Introduction

### 1.1 The Problem

A tensor compiler turns a PyTorch model into machine code step by step. This thesis distinguishes two kinds of work. **Lowering** states how a high-level computation is expressed with lower-level operations, down to loops and memory accesses. **Loop transformation** reorganizes the iteration domain, loop order, and storage arrangement of a computation to improve efficiency. Both construct a target program and check semantic preconditions. This thesis calls a MimIR function that gives the target computation a **library function**, and the C++ pass code that selects it, passes arguments, and connects the result a **driver**.

**Lowering.** The thesis takes three cases, ordered by abstraction level. The highest layer decomposes Torch operators into tensor operations. The middle layer substitutes an algorithm within one plugin: im2col and Winograd recompute a 3×3 convolution with window extraction, a matrix product, or fixed transforms, under preconditions on layout, groups, and sizes. The lowest layer materializes a buffer operation as explicit loops. `map_reduce_post` declares iteration, reduction, and an epilogue over buffers; the implementation reuses the loop construct `mr_nest`, composes the reads and the element function into `cell`, the epilogue and the write-back into `wb`, and hands both to `nest`:

```text
btensor.map_reduce_post(...)(mem, dst, inputs, post_inputs)
    → nest(init, cell, wb)(mem, dst)
    → affine.For + buffer.read/write + element computation
```

All three cases ask how the target program expresses the semantics of the source operation and how the preconditions are stated and checked. The driver selects; the library function gives the composition.

**Loop transformation.** For `C[M,N] = A[M,K]·B[K,N]`, a developer wants a schedule API that expresses OpenBLAS-style cache blocking, packing, register blocking, and vectorization:

```text
t' = gemm_schedule(nc, kc, rows, pack_a, pack_b)(t)
    # t:  structured loop representation (iteration domain, element function, access maps)
    # t': nc/kc cache blocks, packed A/B panels, rows×16 register accumulator, vectorized inner loop
```

Each primitive reorganizes loops, composes indices, and arranges accumulator lifetimes; coverage, tail blocks, and reduction order are legality obligations that composition must maintain. In MimIR a schedule acts on `Tileable`, a structured representation that keeps the iteration domain and access maps, and a lowering then materializes the loops. Loop transformation therefore happens before loops exist; this thesis presents lowering first because that is the order in which a reader understands them.

**Existing practice.** Torch-MLIR, TVM Relax, and MLIR linalg implement lowering and scheduling primitives in C++ or Python and maintain the preconditions there (Section 6). MimIR declares operations as axioms in plugins with normalization or lowering rules [1], and [2] writes `Tileable` and schedules as functions in the IR. This thesis adopts these mechanisms and studies how to compose target programs from higher-order functions, which type relations the checker verifies, and which obligations still need explicit contracts.

### 1.2 Method

A higher-order function (HOF) takes or returns functions. Partial evaluation (PE) specializes a program with respect to its static arguments and keeps a **residual program** for the dynamic inputs [3]. Here, arithmetic operations, access maps, element functions, and schedule parameters are static; tensor contents and dynamic scalars remain at runtime. Each substituted computation is a `lam` in the MimIR library whose body is the transformed structure. MimIR provides dependent types and controlled PE [1]; [2] shows PE lowering, `Tileable`, and schedules as functions. This thesis adds case-specific contracts, targeted tests of selected compositions, and controlled construction-cost evidence.

### 1.3 Completed Results, Research Questions, and Contributions

**Completed.**

- **Torch frontend and Torch plugin.** The frontend imports PyTorch FX graphs [4], adapts operators and shapes, provides a `torch.compile` entry point [5] and evaluation facilities, and served the evaluation of manuscript [2]. The Torch plugin [6], also by the author, defines operator decompositions as library functions in `torch.mim`. On 15 October 2026, 100/100 scaled cases of KernelBench [7] Level 1 and 99/100 of Level 2 passed the reference comparison, all with static shapes (Section 3.1).
- **im2col and Winograd prototypes** as library functions in `tensor.mim` [8], verified against a double-precision scalar reference, with preliminary measurements delimiting where each algorithm pays off.
- **Buffer-to-loop lowering** in `.mim` on a separate branch [9], replacing the corresponding C++ graph construction; regression tests pass.
- **GEMM schedule prototype.** The prototype's `dot_trace` schedule, a function on `Tileable`, brings single-threaded FP32 GEMM to a geometric-mean ratio ρ of 1.08 against OpenBLAS over 17 shapes (median 0.98; 16 of 17 at or above 0.8), versus 0.38 for upstream MimIR's default schedule (Section 5.1).

**Research questions.** The thesis has six months in total; the remaining four months (16 weeks) follow two questions, one per kind of work:

- **RQ1, construction and cost of lowering (Section 3):** How do HOFs compose lower-level operations with existing functions across the three layers, which structures are reusable, and which preconditions need dedicated checks? Construction cost is measured on the buffer-to-loop case by a controlled C++/HOF comparison with element function and schedule fixed; a comparison for the convolution algorithms is an extension item.
- **RQ2, expressiveness and contracts of loop transformation (Section 4):** Can the schedule API express OpenBLAS-style blocking and data reuse by hand and bring GEMM/GEMV/CONV close to their single-threaded references? Which preconditions and predicate-propagation rules do the selected split/fuse/interchange compositions require, and how are coverage, reduction order, and tail-block safety checked in the implementation and tested?

**Contributions.**

- The Torch→MimIR frontend and Torch plugin (completed).
- Source/target correspondence, preconditions, and numeric contracts for three layers of lowering, plus the controlled C++/HOF cost comparison on the buffer-to-loop case (RQ1).
- The `gemm_schedule` expert interface and the performance and structure comparison of GEMM/GEMV/im2col-CONV against single-threaded OpenBLAS (RQ2).
- A contract table, checked preconditions, rejection paths, and targeted regression and differential-execution tests for selected split/fuse/interchange compositions (RQ2).

The thesis addresses developers of compiler passes, operator libraries, and schedule libraries; implementation cost means the code they maintain, not the length of IR text.

## 2. Background

**Specialization in MimIR.** Plugins declare operations with axioms and supply normalization or lowering rules [1]. An operator stays opaque for high-level analyses until a chosen stage replaces its calls by calls to an implementation function. The PE filter controls whether a call site is beta-reduced: element types, ranks, scalar operations, and index functions are specialized, while tensor values and some extents remain at runtime. A wrong filter can make MimIR diverge [1, §3.3], and excessive unfolding causes code bloat. The type-safety result of [1] does not extend to arbitrary plugins [1, §3.4.1]. Continuation parameters correspond to loop-carried values [1, §3.1.5, §3.2], which explains why loops can be described by functions but does not prove that reordering or storage reuse is legal.

**`Tileable` and schedules as functions.** [2, §7] rewrites MLIR's `TilingInterface` as a MimIR type `Tileable`. It records the iteration domain `Sr`, the result shape `So`, the output axis `oax` each parallel axis writes, the vector axis `vdim`, the unrolled axes `unroll`, and a function `retile` that rebuilds `map_reduce` for a new domain, schedule, and access maps. The first `Ro` axes are parallel and the last `Rr` are reduction axes; this thesis calls this classification the **role** of an axis. Conditions such as "`Sr` has length `Ro + Rr`" are typing constraints. A schedule is an ordinary function of type `Tileable → Tileable`; [2] implements strip-mining, interchange, and selectors for `vdim` and `unroll`, and the C++ lowering makes no scheduling decision. The author's convolution prototype [8] adds `fuse_iter`, `cache_read` (packing an operand block into a contiguous panel), and `compute_at` (materializing a producer at a consumer loop level), and distinguishes the output rank `Ro` from the number of parallel axes `Rp`.

**Obligations a transformation carries itself.** Type checking constrains the relations between parameters, results, and index maps; it does not prove memory safety, floating-point equivalence, or correct buffer reuse. With source `F`, target `G`, static arguments `s`, and dynamic inputs `x`, an exact transformation must satisfy `P(s, x) ⇒ observe(F(s, x)) = observe(G(s, x))` under its precondition `P`. Floating-point algorithms such as Winograd use a numeric error contract instead. [2] splits the argument into PE correctness, call-replacement correctness, and equivalence of the implementation function; this thesis adopts that layering.

## 3. Lowering: From Torch Operators to Explicit Loops (RQ1)

All three cases share one flow: analysis and selection, application of the library definition, PE specialization, residual verification. Table 1 shows the three layers in one pipeline.

| Layer | Source | Driver | Library definition / residual after PE |
| --- | --- | --- | --- |
| Frontend import | PyTorch FX graph | Torch frontend (Python) | MimIR `torch.*` calls with shape/type information |
| Operator decomposition | `torch.*` calls | `DecomposeByImpl::apply_impl` | `torch.*_impl` functions composing tensor operations |
| Algorithm substitution | `tensor.trans.im2col` / `tensor.trans.winograd` | `Lower::lower_via_impl` | Tensor operations implementing the chosen convolution algorithm |
| Buffer-to-loop lowering | `btensor.map_reduce_post` | Bind metadata and chain readers | `map_reduce_post_impl`: scheduled `nest(init, cell, wb)`, loops, and buffer reads/writes |

*Table 1. The lowering pipeline. Each row is one layer; the columns separate the source operation, the driver that selects a definition and passes arguments (Python for import, C++ for lowering), and the library function whose residual after PE is the target program. The schedule of Section 4 enters as a function on `Tileable` before layer 3 materializes the loops.*

### 3.1 Torch Operator Decomposition

The frontend [10] imports FX graphs through the MimIR Python bindings and selects an adapter per operator. Adaptation distinguishes overloads, keeps scalars apart from tensors, and preserves dtype, negative axes, `keepdim`, `alpha`, empty tensors, and multiple results. PyTorch's logical shape and MimIR's value type do not always agree dimension by dimension (MimIR may fold length-1 axes); shape records and one set of shape rules reconcile the two views.

The Torch plugin's `torch.mim` [6] defines each operator as a library function, and library functions compose other library functions. `torch.loss_impl.cross_entropy_mean_2d` (`torch.mim:6637–6651`) defines the mean cross-entropy loss over `[n, c]` logits and `n` class indices as six library calls. The explanatory excerpt writes `T` as a separate parameter for readability; throughout the example it denotes the element type `F#A#R#T` of the floating-point dictionary `F`:

```mim
anx lam cross_entropy_mean_2d (F: Floating) {n c: Nat, T: *}
    (input: «n, c; T», target: «n; I64»): T =
    let log_probs   = normalization.log_softmax F @(2, (n, c)) (1I64, ff) input;
    let target_2d   = tensor.reshape @(I64, 1, 2) (n,) (n, 1) target;
    let selected_2d = indexing.gather @(T, 2, (n, c), (n, 1)) 1I64 (log_probs, target_2d);
    let selected    = tensor.reshape @(T, 2, 1) (n, 1) (n,) selected_2d;
    let losses      = tensor.unary @(T, T) (torch_neg_scalar F#A) @(1, (n,)) selected;
    reduction.mean_all F @(1, (n,)) losses;
```

Three kinds of composition appear. `log_softmax` and `mean_all` are other Torch library functions, themselves defined over tensor operations, so the decomposition is layered rather than flat. `reshape` and `gather` are tensor-plugin operations whose shape arguments `(n, c)` and `(n, 1)` are static and checked against the types of `input` and `target`. `torch_neg_scalar F#A` is a scalar function `T → T` from the arithmetic part of the dictionary, partially applied and lifted to tensors by `tensor.unary`; activations such as `mish` are built the same way from scalar `softplus`, `tanh`, and `mul`. After PE, `log_softmax` and `mean_all` have unfolded into their own `map_reduce` and `unary` calls, and only the tensor contents remain dynamic. For this example, `DecomposeByImpl::apply_impl` connects a `torch.loss.cross_entropy_mean_2d` call to `torch.loss_impl.cross_entropy_mean_2d` and recurses on the calls that still await decomposition. The general `torch.loss.cross_entropy` entry has a separate implementation for its target variants and options. The frontend also provides a `torch.compile` backend with JIT execution, per-stage profiling, and KernelBench and benchmark runners with reference checks.

**Limitation: symbolic shapes.** Support is partial. Import maps a symbolic dimension to a MimIR `Nat` variable, and the shape rules carry symbols through expand, reduction, transpose, and select. But the `torch.compile` execution path accepts only static input shapes (KernelBench runs with `dynamic=False`), several adapters reject dynamic cases (split by dynamic sizes, dynamic `narrow`, negative indices into a dynamic dimension), and the shape rules do not prove relations between symbols (broadcasting two symbols keeps the left one unchecked). All evidence in this proposal is therefore for static shapes; closing these gaps is follow-up work (Section 7).

### 3.2 Algorithm Substitution: im2col and Winograd

Assume NCHW input, OIHW weights, no groups, no bias, and positive sizes; this is the precondition `P`. Im2col [11] moves the window reads into a column matrix `Col[k,q]` with `k=(ci·KH+kh)·KW+kw` and `q=(n·OH+oh)·OW+ow`, calls a matrix product `Z[co,q] = Σ_k W2[co,k]·Col[k,q]`, and recovers NCHW. The library function `trans_im2col_impl` in `tensor.mim` [8] composes `im2col`, `reshape`, `transpose`, `product_2d`, and the output recovery; `compute_at 0` keeps the column matrix at the root of the consumer. The entry `tensor.trans.im2col` is replaced through `Lower::lower_via_impl`, and PE plugs the known types, shapes, maps, and scalar operations into the algorithm. For real arithmetic, the source/target relation is:

```text
Y[n,co,oh,ow] = Σ_ci,kh,kw W[co,ci,kh,kw] · Xp[n,ci,oh*sh+kh*dh,ow*sw+kw*dw]
              = Σ_k W2[co,k] · Col[k,q(n,oh,ow)]
```

Here `Xp` is the zero-padded input, `(sh,sw)` is stride, and `(dh,dw)` is dilation. Each source reduction term has the same weight and input value at its flattened `k`; each source output position has its corresponding column `q`. Both index equations were checked with independent small-scale computations covering multiple batches and channels, odd sizes, asymmetric padding, and singleton axes.

Winograd F(2×2,3×3) [12] keeps the same source semantics but changes the multiplication structure: each 4×4 input tile is transformed to 16 points, the channel reduction happens pointwise there, and the result is transformed back to 2×2 outputs, cutting convolution multiplications per 2×2 output tile and input/output channel pair from 36 to 16. `trans_winograd_impl` composes padding, the weight and input transforms as `map_reduce` point functions, a batched product, and the output transform. Because it depends on subtraction and factors of ½, its API takes a floating-point format and numeric mode, and the tests accept `|out − ref| ≤ ε·(1 + Σ|x·w|)` with `ε = 2e-5` for FP32 against a double-precision reference.

The current API lets the caller explicitly choose `tensor.trans.im2col` or `tensor.trans.winograd`; the lowering driver connects that entry to its implementation. `tensor.conv` keeps its direct path. Measurements delimit where the extra memory traffic or transform arithmetic pays off (Section 5.1); automatic algorithm selection is follow-up work. Initial judgment on RQ1: when the substituted algorithm has a regular structure, scalar operations and maps can be passed explicitly, and the preconditions are checkable, HOF/PE is suited to carry the construction. Algorithm selection, materialization benefit, and the numeric contract are established separately.

### 3.3 From Buffer Operations to Explicit Loops (Cost Comparison)

The source operation `btensor.map_reduce_post(...)(f, init, post) map_out (maps, post_maps) (mem, dst, inputs, post_inputs)` already uses buffers, but one operation still describes iteration, reduction, and epilogue. The implementation [9] reuses `f`, `post`, the loop construct `mr_nest`, and the schedule `sched` acting on it; this is the junction with Section 4, since the lowering partially evaluates the nest builder with the schedule stored in the `Tileable`. Omitting dependent types and CPS continuations, the `.mim` implementation is:

```text
lam map_reduce_post_impl(..., f, init, post, read_inputs, read_post, write_output)(mem, dst) =
    lam cell(mem, acc, i) =
        let (mem, values) = read_inputs(mem, i)
        f(mem, acc, values)
    lam wb(mem, dst, acc, (i_out, i_red)) =
        let (mem, o) = affine.map(map_out, (i_out, 0_red), mem)
        let (mem, extra) = read_post(mem, o)
        let (mem, value) = post(mem, acc, extra)
        let mem = write_output(mem, dst, o, value)
        (mem, dst)
    let nest = sched(...)(mr_nest(...))
    nest(init, cell, wb)(mem, dst)
```

`cell` and `wb` reuse the supplied `f` and `post` and wrap them with coordinate interpretation, reads and writes, and memory-token passing. The read functions and `write_output` use the shared buffer access adapters to project logical coordinates onto the axes retained by `buffer.Buf` normalization. The caller allocates `dst` before the operation; this lowering preserves that destination-passing convention. After PE, the residual program contains concrete loops, reads, scalar computation, and write-back; the function abstraction is eliminated. The driver chain is `tensor.lower_to_mem → btensor.lower_map_reduce → buffer.lower_ptr`; the C++ driver still binds static metadata and chains the heterogeneous read functions. With the same schedule fixed, the C++ construction and the MimIR composition must produce the same access maps, accumulator updates, epilogue, and memory order; Section 5.2 compares the two paths under identical element function, allocation strategy, `sched`, and `mr_nest`.

## 4. Loop Transformation: Expert Schedules and Contracts (RQ2)

### 4.1 Motivation and Workloads

OpenBLAS [13] gives a concrete reference for single-threaded GEMM/GEMV: cache blocking, packing, register blocking, and vectorization along the contiguous direction. The thesis asks whether an expert can obtain the required loop nest, data layout, and local storage by composing schedule functions, with PE and the backend generating efficient code. Multi-threading is out of scope.

| Workload | Structures the manual schedule must express | Reference |
|---|---|---|
| GEMM | multi-level blocking, operand packing, register accumulators, vectorization | single-threaded OpenBLAS GEMM |
| GEMV | vectorization along the contiguous direction, reduction blocking, output reuse | single-threaded OpenBLAS GEMV, same layout and transposition |
| CONV | with im2col fixed, materialization and reuse of intermediates, reusing the GEMM schedule | same im2col transform and allocation strategy, plus OpenBLAS GEMM |

Section 5.1 gives the starting points. For GEMM, upstream MimIR's default schedule reaches a geometric-mean ρ of 0.38 against OpenBLAS and the prototype's `dot_trace` 1.08, with one shape below 0.8. For CONV, im2col is slower than direct in 10 of 12 single-threaded configurations: direct's `tensor.conv` and the matrix product unfolded from im2col both enter existing default scheduling paths. The im2col path has not yet been given the dedicated expert schedule proposed here. RQ2 asks whether manual scheduling closes this gap; if not, Section 5.2 reports both and separates algorithm from schedule.

### 4.2 Schedule Sketch for GEMM

The prototype's current schedule `dot_trace (nc, rows, kblk, packed)` applies three steps: `dot_nblk` blocks N by `nc` outside M; `dot_schedule_rb` builds a `rows×16` register block with the innermost K unrolled by 4; `dot_kblk` strip-mines K and hoists the reduction only when the K×nc panel exceeds the cache. `cache_read` packs B into `«N/16, K, 16»` panels. This corresponds to OpenBLAS's `nc`/`kc` cache blocks, `mr×nr` register block, and packed panels, but `kc` has no explicit interface, only B is packed, and `dot_params` chooses every parameter internally. The proposed expert interface `gemm_schedule (nc kc rows: Nat, pack_a pack_b: Bool)` reuses these combinators, makes the parameters explicit, and adds packing of A with the same `cache_read`. GEMV uses `dot_schedule_kvec` (contraction axis as vector axis with a horizontal sum, unroll-and-jam of output cells), likewise with explicit parameters.

### 4.3 Core Contracts: Split, Fuse, and Interchange

Here fuse means merging two adjacent iteration axes into one (`fuse_iter`), unrelated to operator fusion. The contract model distinguishes the iteration domain `D`, output domain `O`, axis roles, read maps `Aj`, write coordinates, and two predicates: `P_fold` for reading inputs and updating an accumulator, and `P_write` for writing an output. Keeping these predicates separate matters for an empty reduction: when `K=0`, no fold step runs, but each valid output still receives `init`.

A transformation supplies an integer-coordinate map `φ: ℤ^r' → ℤ^r`. On the valid transformed fold domain, `φ` must map into the original valid domain, preserving each required contribution. Reads use `Aj∘φ`. With the output layout fixed, write coordinates are reindexed correspondingly; when a parallel split changes the result shape or layout, `retile` also uses the output-coordinate map `omega`. The following equations describe the coordinate maps; complete contracts include the predicates and schedule metadata:

```text
split(N,b):       new extents (ceil(N/b), b); φ(io,ii) = io*b + ii
                  fold-valid: io*b+ii < N
                  preconditions: b > 0; size and coordinate arithmetic do not overflow
fuse(A,B):        new extent A*B; φ(t) = (t/B, t%B)
                  preconditions: adjacent same-role axes; B > 0; A*B does not overflow
interchange(p):   new extent j = old extent p[j]; φ(o)[i] = o[p_inverse[i]]
                  preconditions: p is a role-preserving bijection of the iteration axes
```

The supported empty-domain cases and rejected zero factors are recorded per API. For example, the stated fuse equation uses `B>0`; the current implementation also handles an empty fused domain with a safe divisor. The contract table will distinguish that supported case from zero factors that cause an error.

Each contract requires the schedule function to check preconditions, construct the new domain and maps, call `retile`, and update `vdim`, `oax`, and any affected unroll or partition metadata. A fold predicate combines the new coordinate-validity condition with the original fold predicate under `φ`; validity is established before any input access or accumulator update. Write validity is propagated separately at write-back. An invalid reduction-tail point leaves `acc` and the memory token unchanged. The dependent argument type already constrains the length of an `interchange` permutation, and the prototype checks axis roles, but it still accepts `(0,0,2)`; range and uniqueness checks with rejection tests are planned work.

For the loop-transformation experiments, element computations are pure, input buffers are read-only, destinations are independent, and scratch storage is private to the relevant loop instance. A transformation preserves the contribution order for each output unless the selected numeric mode permits the stated reassociation. Callbacks with additional shared effects require preservation of their effect order and lie outside these reordering cases. The buffer-to-loop comparison separately checks that wrapping existing callbacks preserves their memory-token order.

TVM's schedule primitives and arithmetic analysis [14] are references for preconditions and diagnostics. Validation uses targeted compositions, rejection tests, and differential execution, including non-divisible tails and empty reductions (Section 5.3). Expert schedules are the primary deliverable. Search over their explicit parameters and supported structural choices, including MetaSchedule-style mutation [15], is follow-up work.

**Boundary cases.** Operator fusion and general buffer reuse are scope boundaries only. Choosing between materializing a shared producer and inlining it twice needs use-count, effect, and cost information from analysis; in-place buffer updates need alias and lifetime analysis beyond memory tokens. Both target programs are expressible as library functions; the thesis separates "what to choose" from "how to construct" and sets no RQ or milestone for them.

## 5. Evaluation

### 5.1 Existing Results

**Frontend.** The run of 15 October 2026 (frontend commit `619a157`) gives KernelBench Level 1 100/100 and Level 2 99/100 on static-shape cases scaled with size-divisor=16; the only failure (`28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply`) lies in the tensor-to-memory path, not in import. These counts speak to correctness of import and execution, not performance. [2, §8.3] evaluated on the then-supported subset of Level 3; the thesis reruns Level 3 by version after fixing the environment in weeks 1–2.

**Convolution.** Preliminary single-threaded FP32 measurements on an Apple M2 Pro (3×3, stride 1, padding 1, timing including transforms and allocation):

| Shape `N,Cin,Cout,H,W` | Direct | Im2col | Winograd |
|---|---:|---:|---:|
| `4,128,128,28,28` | 25.19 ms | 38.85 ms | 18.83 ms |
| `4,512,512,7,7` | 54.29 ms | 34.84 ms | 22.26 ms |

Winograd is 1.3 to 2.4× faster than direct in these two large-channel cases and slower in the other 10 configurations; im2col is slower than direct in 10 of 12. The losses concentrate on small channels, batch 1, and small spatial sizes. These cases motivate checking the im2col schedule alongside transform and allocation costs in RQ2.

**GEMM.** Two single-threaded FP32 measurements against Homebrew OpenBLAS 0.3.34 on the same machine give the starting point, with `ρ = T_OpenBLAS / T_MimIR`. Harness A (the author's `gemm-bench`, 12 October 2026, median of nine interleaved batches) measured upstream MimIR master `2d21e0d`, whose `product_2d` uses the register-blocked `dot_schedule` of [2] without cache blocking or packing: geometric mean ρ 0.38 over 16 shapes. Harness B (the prototype's own harness, 16 October 2026, best of N calls, branch `feature/par` at `c56d586` with working-tree edits) measured `dot_trace`: geometric mean ρ 1.08 over 17 shapes, median 0.98, 16 of 17 at or above 0.8.

| Shape `M×K×N` | ρ, upstream default schedule | ρ, prototype `dot_trace` |
|---|---:|---:|
| `512×512×512` | 0.35 | 0.94 |
| `1024×1024×1024` | 0.30 | 0.91 |
| `2048×2048×2048` | 0.29 | 0.88 |
| `8×2304×512` | 0.57 | 1.37 |
| `64×130×512` | 0.30 | 0.98 |
| `1024×64×1024` | – | 0.69 |

The two runs provide preliminary starting points for the default and prototype schedules. The prototype exposes cache blocking and packing as schedule functions, which RQ2 makes explicit through `gemm_schedule`; the K=64 case remains a gap to investigate. Both harnesses are single laptop runs without core pinning and differ in timing method and conditions; controlled attribution uses the common protocol of Section 5.2. Raw data and provenance are archived with the full proposal.

### 5.2 Planned Experiments

**Frontend.** Tests answer whether import is correct (overloads, scalars, shape, dtype, logical rank, diagnostics), whether information is preserved (symbolic dimensions, shared symbols, singleton axes), whether execution matches eager PyTorch and where failures occur, and how far model coverage reaches. Weeks 1–2 consolidate the existing tests and add the failure classification report.

**Construction cost (RQ1).** The paired source snapshots are C++ baseline `0a89de908d` and HOF variant `3702ae8023` [9]. The comparison fixes the element function, access maps, allocation, `sched`, `mr_nest`, and compiler build options; it varies shape, number of calls, and graph sharing. It records the code maintained in each variant, compile cost, and residual structure, including access coordinates, accumulator updates, epilogue placement, and memory order. A same-algorithm C++ counterpart of im2col is an extension item.

**Expressiveness (RQ2).** GEMM/GEMV fix dtype, shape, layout, transposition, and alpha/beta against single-threaded OpenBLAS; CONV fixes im2col so the reference uses the same transform and allocation with OpenBLAS GEMM. The workload set (6 to 8 static shapes each, including non-divisible and small sizes) and the OpenBLAS build are fixed in advance; the `if_static` fallback of [2] for symbolic extents is not measured. `ρ` is reported per shape with a geometric mean; `ρ ≥ 0.8` counts as close, and lower cases are kept and explained by the generated structure, with the gap attributed to interface, lowering, code generation, or runtime.

### 5.3 Correctness Checks and Cost Metrics

Targeted regression cases cover length 1, empty output domains, empty reductions, non-divisible factors, odd outputs, channel tail blocks, illegal axes, and zero factors. Named compositions include split followed by interchange, split followed by fuse, and combinations with existing tail predicates. Rejection tests cover duplicate or out-of-range permutation entries, role mixing, and unsupported parameter combinations. Differential execution compares the resulting kernels with the unscheduled computation or an independent reference under the stated numeric mode and tolerance. Order-sensitive reduction cases and IR checks test the preservation of contribution order and the placement of guards before loads and updates. Each contract-table entry identifies its implementation check and associated tests.

The code-cost ledger separates the C++ driver, MimIR implementation functions, and access adapters. Shared facilities such as `mr_nest` appear once in a common inventory, with their reuse reported for both variants; new or changed code is charged to its variant. C++ code retained for heterogeneous input binding and callback adaptation remains part of the HOF variant's cost. This distinguishes moving construction into the library from reducing the total code that must be maintained.

Compiler-cost metrics are time per stage, rewrites, intermediate and residual nodes, and peak memory. Generated-program metrics are allocations, copies, vectorization, and runtime. The construction-cost comparison reports compile-time and node ratios rather than setting a pass threshold; residual checks record whether closures or superfluous allocations remain. If the generic interface still needs large dedicated graph reconstruction, or specialization cost grows markedly with instances, the advantage is reported as limited.

## 6. Related Work

The MimIR paper [1] provides the core language, plugins, normalization, PE, and tensor `map_reduce`; [2] discusses affine maps, symbolic shapes, PE lowering, `Tileable`, and schedules as values, with the author's frontend behind its evaluation. This thesis adds source/target correspondence, the performance test of expert schedules, the split/fuse/interchange contracts, and controlled cost experiments.

AnyDSL [16] writes high-performance libraries as ordinary functions specialized by PE; MimIR places this mechanism in a dependently typed IR. Halide [17] separates algorithm from schedule in a schedule language; here the schedule is a function in the IR. Elevate [18] expresses optimizations as rewrite strategies on the source program, whereas these schedules act on IR values and are eliminated by PE. This thesis proposes no new mechanism; it tests the engineering boundaries of these mechanisms on the Torch→MimIR chain.

MLIR's Transform dialect orchestrates transformations written in C++ [19], and One-Shot Bufferization separates analysis from rewriting [20]. By Table 1 of [2], linalg's 81 named operations take about 8.6k lines in three languages, versus 431 lines in one language for the 33 operations of MimIR's tensor plugin. Torch-MLIR's decompositions are C++ rewrite patterns [21] and TVM Relax's legalization is Python over TE/TOPI [22]; both verify generated IR but do not express the rank/shape correspondence between signature and implementation in one type. TVM's schedule primitives act on TIR with C++ legality checks [14]; here they are IR functions, with legality carried by types and explicit contracts.

## 7. Plan, Scope, and Risks

| Weeks | Work | Acceptance |
|---|---|---|
| 1–2 | Fix code, data, and the OpenBLAS baseline; close the `scan.diff` plugin gap; consolidate frontend tests and rerun Level 3; CONV reference harness (C++ im2col + OpenBLAS GEMM); fix the shape set | environment and measurement checklist; test report; harness runs; shape set recorded |
| 3–4 | RQ1: lowering correspondences and contracts for the three cases; RQ2: basic parameter checks and tail regressions before schedule tuning | correspondence tables; IR samples; duplicate/range/zero-factor rejection tests and non-divisible-tail tests pass |
| 5–8 | RQ2: `gemm_schedule` with A packing; validate each selected composition, measure GEMM and iterate; extend to GEMV and im2col-CONV | correctness tests pass before timing; GEMM table and gap analysis; GEMV/CONV schedule reports |
| 9–10 | RQ2: extend split/fuse/interchange composition regressions, predicate propagation and reduction-order checks; document supported cases | targeted regression, rejection, and differential tests pass; contract table matches implementation |
| 11–12 | RQ1: controlled C++/HOF comparison of the buffer-to-loop lowering | cost ledger, memory order, residual structure |
| 13 | Extension: C++ counterpart of im2col for one shape; dropped if behind | comparison or recorded reason for dropping |
| 14 | GEMV/CONV measurements and gap attribution; verify all residual IR | full tables, geometric mean of `ρ`, gap classification |
| 15–16 | Boundary analysis, reproducible materials, writing | thesis and materials list |

Scope is fixed to a single CPU thread, the existing LLVM backend, the selected workloads, expert schedules for GEMM/GEMV/im2col-CONV, the three core contracts, and the buffer-to-loop lowering, all with static shapes. Multi-threading, auto-tuning, end-to-end symbolic shapes, complete schedule legalization, alias/SMT solving, and GPUs are follow-up directions. Risks and fallbacks: PE blow-up (limit unfolding, record per-stage nodes); the im2col C++ counterpart exceeding budget (drop it, RQ1 cost evidence from the buffer case only); mixed sources of performance gaps (attribute per shape with schedule and LLVM IR); legality scope creep (rectangular domains and same-role axes first). Reduction always protects RQ1 and RQ2.

## 8. Conclusion

This thesis studies when tensor compiler transformations can be built from higher-order functions and partial evaluation, organized as lowering and loop transformation. The author's Torch→MimIR frontend and plugin carry the highest layer of lowering and pass 100/100 and 99/100 of the scaled KernelBench Level 1 and 2 cases. Four preliminary findings: decomposition, im2col, and Winograd fit one library organization with shape relations in types and additional checks for semantic preconditions; algorithm substitution has shape-dependent tradeoffs and im2col still needs dedicated expert tuning of its unfolded matrix product; type checking covers rank relations while permutation validity, coverage, and reduction order must be added as explicit contracts; and the prototype's `dot_trace` schedule already reaches a geometric-mean ρ of 1.08 against OpenBLAS on single-threaded GEMM, where upstream's default reaches 0.38. All evidence is for static shapes. The remaining work establishes the lowering correspondences and the buffer-to-loop cost comparison (RQ1), and the `gemm_schedule` interface, the OpenBLAS comparison, and the split/fuse/interchange contracts with implementation checks and targeted regression tests (RQ2). Afterwards, developers of passes, operator libraries, and schedule libraries should be able to judge which transformations can be written as ordinary MimIR functions handed to partial evaluation, what they still have to check themselves, and what code quality such schedules reach.

**Materials.** Code snapshots, raw results, and run conditions are archived locally (not tracked in Git) in `artifacts/pe-hof-2026-10-15` and `artifacts/gemm-2026-10-16` and documented in the full proposal.

## References

1. Roland Leißa, Marcel Ullrich, Joachim Meyer, Sebastian Hack. *MimIR: An Extensible and Type-Safe Intermediate Representation for the DSL Age*. *PACMPL*, 9, POPL, Article 4, 2025. [DOI: 10.1145/3704840](https://doi.org/10.1145/3704840).
2. *Principles for Multi-Level Compiler Design*. Anonymous submission, 2026. Cited for its §3 and §7–8.
3. Neil D. Jones, Carsten K. Gomard, Peter Sestoft. *Partial Evaluation and Automatic Program Generation*. Prentice Hall, 1993.
4. James K. Reed et al. "torch.fx: Practical Program Capture and Transformation for Deep Learning in Python." *MLSys*, 2022.
5. Jason Ansel et al. "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation." *ASPLOS*, 2024.
6. [MimIR Torch plugin](https://github.com/hsqStephenZhang/mimir-torch-plugin/tree/6c41f8eb2313cbb923c2cb92a0a5a6b4d4f9a23d), commit `6c41f8e`.
7. Anne Ouyang et al. "KernelBench: Can LLMs Write Efficient GPU Kernels?" arXiv:2502.10517, 2025.
8. [MimIR convolution prototype](https://github.com/hsqStephenZhang/MimIR), baseline `09bfc92` with `conv-winograd.patch`; GEMM schedule measured on branch `feature/par` at `c56d586`.
9. [MimIR buffer-to-loop lowering](https://github.com/hsqStephenZhang/MimIR/tree/3702ae80230bf173fe1ba0a7dc03222596e95366), commit `3702ae8023`, branch `feature/btensor-map-reduce-hof`; paired C++ baseline `0a89de908d`.
10. [MimIR Torch frontend](https://github.com/hsqStephenZhang/mimir-frontend/tree/619a15776d91a645fba997c059a3001e8e778dea), commit `619a157`.
11. Kumar Chellapilla, Sidd Puri, Patrice Simard. "High Performance Convolutional Neural Networks for Document Processing." *ICFHR*, 2006.
12. Andrew Lavin, Scott Gray. "Fast Algorithms for Convolutional Neural Networks." *CVPR*, 2016.
13. [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/tree/06f28d0942ee8eefce9aad7989d98743d2999932), commit `06f28d0`; measurements use the Homebrew 0.3.34 build.
14. Apache TVM. [Structured loop schedule primitives](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/src/s_tir/schedule/primitive/loop_transformation.cc) and [arithmetic analyzer](https://github.com/apache/tvm/blob/main/include/tvm/arith/analyzer.h).
15. Apache TVM MetaSchedule, commit `def37e3`: space generators, evolutionary search, and tile-size mutation.
16. Roland Leißa et al. "AnyDSL: A Partial Evaluation Framework for Programming High-Performance Libraries." *PACMPL*, 2, OOPSLA, Article 119, 2018.
17. Jonathan Ragan-Kelley et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines." *PLDI*, 2013.
18. Bastian Hagedorn et al. "Achieving high-performance the functional way: a functional pearl on expressing high-performance optimizations as rewrite strategies." *PACMPL*, 4, ICFP, Article 92, 2020.
19. MLIR. [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/).
20. MLIR. [Bufferization](https://mlir.llvm.org/docs/Bufferization/).
21. [Torch-MLIR](https://github.com/llvm/torch-mlir), `lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp` and `lib/Conversion/TorchToLinalg/`.
22. [Apache TVM Relax legalization](https://github.com/apache/tvm/tree/main/python/tvm/relax/transform/legalize_ops).
