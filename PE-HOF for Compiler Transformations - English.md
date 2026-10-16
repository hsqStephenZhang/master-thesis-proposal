# Expressing Tensor Compiler Transformations with Higher-Order Functions and Partial Evaluation: Case Studies in Lowering and Loop Transformation

**Master's Thesis Proposal**  
**Focus:** Starting from a Torch frontend, examine the expressiveness, contracts, and cost of building lowerings and expert schedules from higher-order functions.  
**Date:** 16 October 2026.

## Abstract

A tensor compiler rewrites model operators into efficient loops and memory accesses while preserving the semantics of the source language. Existing systems write lowering and loop-transformation primitives in compiler code, so the rank and shape correspondence between an operator's signature and its implementation is not expressed in one type. MimIR and recent work on multi-level compilation show another route: implementation functions, the tiling interface, and schedules are all dependently typed functions in the IR, and partial evaluation specializes them into loops. Which transformations this route supports, which contracts a transformation must still carry itself, and whether schedules built this way approach the performance of reference libraries have not been examined on cases. Building on the author's MimIR Torch frontend, this thesis expresses three layers of lowering with higher-order functions, from operator decomposition through convolution algorithm substitution to buffer operations materialized as loops, and uses the buffer-to-loop layer for a controlled cost comparison between C++ and higher-order-function construction. For loop transformations, it builds expert schedules for GEMM, GEMV, and CONV by hand through a schedule API, with single-threaded OpenBLAS as the performance reference, and establishes composition contracts and small-domain exhaustive checks for split, axis fusion (fuse), and interchange.

## 1. Introduction

### 1.1 The Problem: Lowering and Loop Transformation

A tensor compiler turns a PyTorch model into machine code step by step. This thesis distinguishes two kinds of work. Lowering states how a high-level computation is expressed with lower-level operations, down to loops and memory accesses. Loop transformation reorganizes the iteration domain, loop order, and storage arrangement of a computation to improve execution efficiency. Both kinds of work construct a target program and check the corresponding semantic preconditions.

This thesis calls a MimIR function that gives the target computation a **library function**, and the C++ pass code that selects a library function, passes arguments, and connects the result a **driver**. Later sections use these two words to separate, for each transformation, "who gives the computation" from "who makes the choice".

**First kind, lowering: restating the computational semantics.** These transformations re-express the computation of a source operation with another set of operations. This thesis takes three cases, ordered by abstraction level. The highest layer is the decomposition of Torch operators into tensor operations; for example, addition with `alpha` becomes a scaling composed with an addition (Section 3.1). The middle layer is algorithm substitution within one plugin: im2col and Winograd recompute a 3×3 convolution with window extraction, matrix multiplication, or fixed transforms, and only hold under preconditions on layout, groups, and sizes (Sections 3.2–3.3). The lowest layer is from buffer operations to explicit loops. `map_reduce_post` declares iteration, reduction, and an epilogue over buffers; its computational semantics must be given with explicit loops and element reads and writes. Section 3.4 reuses the existing `mr_nest` to build the loops. It composes the reads and the element function into `cell`, composes the epilogue and the write-back into `wb`, and hands both to `nest`:

```text
btensor.map_reduce_post(...)(mem, dst, inputs, post_inputs)
    → nest(init, cell, wb)(mem, dst)
    → affine.For + buffer.read/write + element computation
```

All three cases answer the same question: how does the target program express the computational semantics of the source operation, and how are the preconditions stated and checked? The driver selects the implementation; the library function gives the target composition.

**Second kind, loop transformation: reorganizing loop structure.** For the matrix product `C[M,N] = A[M,K]·B[K,N]`, a developer wants a schedule API that expresses OpenBLAS-style cache blocking, packing, register blocking, and vectorization (the proposed interface of Section 4):

```text
t' = gemm_schedule(nc, kc, rows, pack_a, pack_b)(t)
    # t:  the structured loop representation of the product (iteration domain, element function, access maps)
    # t': nc/kc cache blocks, packed A/B panels, a rows×16 register accumulator, vectorized innermost loop
```

Each primitive must reorganize loops, compose indices, and arrange the lifetime of accumulators. Coverage, tail blocks, and reduction order are legality obligations that composition must maintain. Floating-point reordering is further constrained by the numeric mode. The generated structure and the performance relative to OpenBLAS together test the expressiveness of this schedule API.

This thesis organizes its research around these two responsibilities. A loop transformation can act on explicit loops, or on a structured representation that keeps the iteration domain and access maps. MimIR's current `Tileable` is the latter: a schedule function first describes the target loop structure, and a lowering like that of Section 3.4 then materializes those loops according to the schedule. In the actual pipeline, therefore, loop transformation happens before loops are materialized. This thesis presents lowering before loop transformation because that is the order in which a reader understands them, not the order in which they execute.

**Existing practice and the route of this thesis.** Torch-MLIR, TVM Relax, MLIR linalg, and TVM typically implement lowering and scheduling primitives in C++ or Python and maintain the corresponding preconditions; Section 7.3 compares them. MimIR organizes operations of different levels into plugins, declares the name and type of an operation with an axiom, and supplies normalization or lowering rules [12]. [13] goes further and writes `Tileable` and schedules as functions in the IR. This thesis adopts these mechanisms. It studies how to compose target programs from higher-order functions, which type relations the checker can verify, and which semantic obligations still need explicit contracts. Later sections test construction reuse, driver responsibility, compile cost, and generated code quality on the two kinds of cases.

### 1.2 Method and Prior Work

A higher-order function (HOF) takes or returns functions. Partial evaluation (PE) specializes a program with respect to its static arguments and keeps a **residual program** for the dynamic inputs, that is, the part left to run at runtime after specialization. In the examples of Section 1.1, the arithmetic operations, access maps, element functions, and schedule parameters are static arguments; tensor contents and dynamic scalars remain at runtime.

This thesis writes the substituted computation as a `lam` function in the MimIR library: the function body is the transformed structure, and the driver selects the definition, passes arguments, and connects the result. MimIR provides dependent types and controlled PE [12]. [13] has already shown PE lowering, the `Tileable` interface, and schedules represented as functions. This thesis adds contracts, composition checks, and cost evidence for the selected transformations.

The lowerings of Section 1.1 comprise Torch operator decomposition (Section 3.1), compositional implementations of convolution algorithms (Sections 3.2–3.3), and the loop semantics of buffer operations (Section 3.4); loop transformation corresponds to the schedule API and its contracts (Section 4). Each kind of work maps to one research question (Section 1.3). Section 3 presents, by abstraction level, the source program, target function, and driver chain of each lowering; Section 4 discusses loop transformation. The buffer lowering also provides a controlled comparison between the existing C++ and HOF implementations.

### 1.3 Completed Results and Thesis Plan

**Completed results.**

- **MimIR Torch frontend and Torch plugin (Section 3.1):** The frontend imports PyTorch FX graphs [27], provides operator and shape adaptation, a `torch.compile` entry point [28], model export, tests, and evaluation facilities, and provided the evaluation for the research manuscript [13]. The Torch plugin [16] defines operator decompositions as library functions in `torch.mim`; the author implemented it as well. In the run of 15 October 2026, 100 of 100 scaled cases of KernelBench [25] Level 1 and 99 of 100 of Level 2 passed the reference comparison (Section 6.1). These runs use static shapes; symbolic shapes are handled by the import and shape-rule layer but not by the JIT execution path (Section 3.1). One interface gap remains between the three snapshots of frontend, Torch plugin, and MimIR; see Appendix A.
- **im2col and Winograd prototypes (Sections 3.2–3.3):** Both convolution algorithms are implemented as library functions in `tensor.mim`, and all outputs are compared against a double-precision scalar reference. Preliminary measurements (Section 6.1) delimit where each algorithm pays off relative to direct convolution.
- **Lowering buffer operations to loops (Section 3.4):** On a separate branch, `.mim` code composes element reads, `cell`, `wb`, and the existing loop constructs, replacing the corresponding C++ graph construction; the regression tests pass.
- **GEMM schedule prototype (Sections 4.2 and 6.1):** The prototype's `dot_trace` schedule, written as a function on `Tileable`, brings single-threaded FP32 GEMM to a geometric-mean ρ of 1.08 against OpenBLAS over 17 shapes (median 0.98; 16 of 17 at or above 0.8), versus 0.38 for the default schedule of upstream MimIR master.

**Contributions the thesis will add.** The thesis has a total time limit of six months; the frontend and the prototypes have consumed part of that effort. The remaining four months (16 weeks) are organized around two research questions that correspond one-to-one to the two kinds of work in Section 1.1. This section is the only place that defines the research questions; later sections only refer to them:

- **RQ1, construction and cost of lowering (core, Section 3):** From Torch operator decomposition and convolution algorithm substitution to the loop implementation of buffer operations, how do HOFs compose lower-level operations with existing functions? Which structures are reusable, and which preconditions need dedicated checks? The evidence is source/target correspondence, type relations, and driver size. Construction cost is measured on the buffer-to-loop case by a controlled C++/HOF comparison: with the element function and schedule fixed, compare driver size, compile time, and residual structure. A cost comparison for the convolution algorithms is an extension item.
- **RQ2, expressiveness and contracts of loop transformation (core, Section 4):** Can the schedule API express OpenBLAS-style blocking and data reuse by hand, and bring the selected GEMM/GEMV/CONV close to their reference implementations on a single thread? How do split/fuse/interchange compositions maintain coverage, reduction order, and tail-block safety? The deliverables are a unified schedule interface, a contract table, rejection paths, and a small-domain exhaustive checker.

**Contributions.** After completion, the following four items can be cited independently:

- The Torch→MimIR frontend and Torch plugin: FX graph import, operator and shape adaptation, the `torch.compile` entry point, and evaluation facilities (completed).
- Source/target correspondence for three layers of lowering: the library functions, preconditions, and numeric contracts of operator decomposition, im2col/Winograd, and buffer-to-loop, plus the controlled C++/HOF cost comparison on the buffer-to-loop case (RQ1).
- The `gemm_schedule` expert interface and the performance and structure comparison of GEMM/GEMV/im2col-CONV against single-threaded OpenBLAS (RQ2).
- The contract table, rejection paths, and small-domain exhaustive checker for split/fuse/interchange (RQ2).

This thesis addresses developers of compiler passes, operator libraries, and schedule libraries. Implementation cost means the construction code, drivers, and supporting facilities they maintain, not the length of IR text.

## 2. Background and Basic Assumptions

### 2.1 Programs, Interfaces, and Specialization in MimIR

MimIR represents types and terms in one unified graph. Plugins declare operations with axioms and supply normalization or lowering rules [12]. An operator can stay opaque at first so that high-level analyses recognize it; at a chosen stage its calls are replaced by calls to the implementation function. The core type-safety result of [12] does not extend directly to arbitrary plugins; [12, §3.4.1] discusses the safety boundary of axioms and normalizers.

The PE filter controls whether a call site is beta-reduced. Static arguments such as element type, rank, scalar operations, or index functions can be specialized; tensor values and some extents remain at runtime. A static rank does not mean that every extent is known. A wrong filter can make MimIR diverge; [12, §3.3] discusses this with a recursive `pow`. Excessive unfolding also causes code bloat, which the cost measurements of this thesis record.

An explicit dictionary is a record of a type and its operations, for example `(T, zero, add, mul)`, where `add` and `mul` are higher-order arguments. A resolver that looks up the dictionary from the type is another, optional layer; this thesis uses the dictionary interface with explicit arguments.

MimIR also represents control flow with continuations. The parameters of a continuation correspond to the merged values of SSA, including loop-carried accumulators [12, §3.1.5, §3.2]. This explains why loops can be described by functions, but it does not automatically prove that reordering, parallelizing, or storage reuse of a loop is legal.

**`Tileable` and schedules as functions.** [13, §7] rewrites MLIR's `TilingInterface` as a type `Tileable` in MimIR. It records the iteration domain `Sr` of a structured computation, the result shape `So`, the output axis `oax` that each parallel axis writes, the vector axis `vdim`, the unrolled axes `unroll`, and a function `retile`. The first `Ro` axes of the iteration domain are parallel axes and the last `Rr` are reduction axes; this thesis calls the class an axis belongs to its **role**. `retile` rebuilds `map_reduce` according to the new iteration domain, schedule, and access maps.

Consistency conditions such as "the length of `Sr` equals `Ro + Rr`" are expressed by dependent types, so an implementation with inconsistent ranks does not type-check. A schedule is an ordinary function of type `Tileable → Tileable`. [13] implements strip-mining (of a parallel or reduction axis), loop interchange, and combinators that choose `vdim` and `unroll`, and writes the schedule of a matrix product by function composition. At lowering time the C++ driver makes no scheduling decision; it partially evaluates the nest builder with the schedule stored in the `Tileable` and obtains the loop nest.

The convolution prototype of this thesis [17] adds several combinators on this basis: `strip_mine_*`, `fuse_iter`, `cache_read` (which rearranges an operand block into a contiguous panel), and `compute_at` (which **materializes** a producer at a chosen loop level of the consumer, that is, allocates and writes the intermediate result at that level). The schedule sketches of Section 4 use them directly. The prototype distinguishes the output rank `Ro` from the number of parallel axes `Rp`; in [13] the two coincide.

### 2.2 Obligations a Transformation Must Carry Itself

Section 2.1 describes the mechanisms MimIR provides; this section describes what these mechanisms do not take over from a transformation. Type checking constrains the type relations between parameters, results, and index maps. By itself it does not prove memory safety, floating-point equivalence, or correct buffer reuse. Each transformation states and checks these properties itself. This section fixes the way they are stated, for later transformations to refer to.

Let `F` be the source program and `G` the target program. For inputs satisfying the precondition `P`, an exact transformation must preserve observable behavior:

```text
If P(s, x) holds:  observe(F(s, x)) = observe(G(s, x))
```

Here `s` are the static arguments and `x` the dynamic inputs. A representation change must additionally state the relation between source values and the target representation. A transformation with effects must preserve observable memory behavior. Floating-point algorithms such as Winograd do not satisfy bitwise equivalence; they use a separate numeric error contract instead, defined by a numeric mode, reference values, and an error criterion; Section 3.3 gives the concrete criterion.

[13] splits the argument for a PE transformation into PE correctness, call-replacement correctness, and equivalence of the implementation function. This thesis adopts this layering and adds preconditions and checks for the concrete transformations. The correctness argument focuses on the transformations selected in this thesis.

## 3. Lowering: From Torch Operators to Explicit Loops (Core RQ1)

This section follows the abstraction levels of Section 1.1 from high to low through the three lowering cases: Torch operator decomposition (Section 3.1), algorithm substitution for convolution (Sections 3.2–3.3), and buffer operations to explicit loops (Section 3.4). The three cases share one flow: analysis and selection, application of the library definition, PE specialization, and residual verification. The target structure first exists as a `lam` function in the library. The driver supplies arguments and context, PE specializes the static structure, and the verification stage checks the target operations and interfaces. Section 4 discusses loop transformation; Section 5 discusses the boundary of global analysis.

| Layer | Source | Driver | Library definition / residual after PE |
| --- | --- | --- | --- |
| Frontend import | PyTorch FX graph | Torch frontend (Python) | MimIR `torch.*` calls with shape/type information |
| Operator decomposition | `torch.*` calls | `DecomposeByImpl::apply_impl` | `torch.*_impl` functions composing tensor operations |
| Algorithm substitution | `tensor.trans.im2col` / `tensor.trans.winograd` | `Lower::lower_via_impl` | Tensor operations implementing the chosen convolution algorithm |
| Buffer-to-loop lowering | `btensor.map_reduce_post` | Bind metadata and chain readers | `map_reduce_post_impl`: scheduled `nest(init, cell, wb)`, loops, and buffer reads/writes |

*Table 1. The lowering pipeline. Each row is one layer; the columns separate the source operation, the C++ driver that selects a definition and passes arguments, and the library function whose residual after PE is the target program. The schedule of Section 4 enters as a function on `Tileable` before layer 3 materializes the loops.*


### 3.1 Torch Operator Decomposition: Frontend and Library Functions

The highest layer of lowering is carried jointly by the author's Torch frontend [15] and Torch plugin [16]. The frontend imports PyTorch FX graphs and adapts operator semantics and shapes. The plugin's `torch.mim` defines, as library functions, the decomposition of each operator into tensor operations; the C++ driver `DecomposeByImpl` selects and applies these definitions. The frontend is also the execution entry point and test facility for the whole chain: it supplies model import and execution for the end-to-end correctness evaluation of Section 6.2. The cost comparison of Section 3.4 and the schedule experiments of Section 4 run directly from the MimIR compiler entry, bypassing the frontend, but reuse its measurement and diagnostic facilities. The implementation basis is the `mimir-frontend` repository snapshot [15]; Appendix A separates source versions from the runtime environment.

#### From the FX Graph to MimIR Computation

The frontend takes a PyTorch FX graph as input and builds functions and tensor operations through the MimIR Python bindings. The core chain is:

```text
PyTorch model / FX graph
    → graph node and input signature handling
    → operator argument and semantic adaptation
    → MimIR Torch/tensor graph with shape/type information
    → application of the library operator implementations, unfolding to lower-level structure
    → MimIR optimization and lowering
    → LLVM IR, JIT compilation, and result return
```

The translator maintains a map from FX nodes to converted values and selects an adapter for each operator through a conversion table. Adaptation includes distinguishing overloads, normalizing arguments, keeping scalars apart from tensors, and connecting calls and returns. The operator library builds tensor operations and connects them to the Torch semantic interface; it currently covers elementwise operations, reductions, matrix products, convolutions, pooling, normalization, and indexing.

#### Shape, dtype, and Source-Language Semantics

PyTorch's logical shape and MimIR's value type do not always correspond dimension by dimension. For example, MimIR normalization may fold axes of length 1, while PyTorch's `transpose`, `keepdim`, and broadcast still interpret dimensions by logical rank. The frontend reconciles these two views with shape records and type queries, and handles symbolic dimensions, broadcast, reduction, and dimension operations with one set of shape rules.

Operator adaptation must also preserve source-language behavior such as dtype, negative axes, `keepdim`, `alpha`, empty tensors, and multiple results. Take the `alpha` argument of `aten.add`: ignoring it still produces a well-typed addition, but changes the result. The frontend also states explicitly which operations are handed to PyTorch decomposition and which are kept for MimIR. This boundary means that model import need not unfold every high-level structure first, and gives a fixed entry point for comparing lowering strategies.

**Limitation: symbolic shapes.** Support for symbolic shapes is partial. The import layer maps a symbolic input dimension to a MimIR `Nat` variable, and the shape rules carry these symbols through expand, reduction, transpose, select, squeeze, and static split, as the unit tests check. Three gaps remain. First, the `torch.compile` execution path and the model-export flow accept only fully static input shapes; the KernelBench and benchmark runners therefore compile with `dynamic=False`, and every count in Section 6.1 is a static-shape result. Second, several adapters reject dynamic cases with `NotImplementedError`: split by dynamic sizes, a dynamic `narrow` offset, negative indices into a dynamic dimension, and non-contiguous strides with dynamic extents. Third, the shape rules do not prove relations between symbols: broadcasting two symbolic dimensions keeps the left one without a runtime check, slicing a symbolic extent loses the sliced extent, and concatenation delegates the result extent to the plugin. The dependent-type argument of this thesis, that the rank/shape correspondence between signature and implementation lives in one type, is therefore examined on static shapes; symbolic extents are exercised only in the import layer and its tests. Closing these gaps is listed as follow-up work (Section 8).

#### Defining Torch Operator Semantics with Library Functions

The frontend connects PyTorch operations to tensor structures through `torch.mim` of the Torch plugin [16]. Below, one implementation is kept verbatim (addition with a scalar argument) and one reduction is shown by signature only (with dimension check and result shape).

`A: Arithmetic` carries the element type and arithmetic operations; `F: Floating` supplies floating-point functions. `«s; T»` is the array type of shape `s` and element type `T`, `#` is record projection, and braces mark implicit parameters. `anx lam` registers a function definition that a driver can refer to. The excerpts keep the function body; see the sources for the enclosing module and dependencies.

**Addition with `alpha`.** `torch.binary_impl.add` (`torch.mim:999–1006`) defines `lhs + alpha * rhs` as two tensor library calls:

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

`torch_scale_scalar A alpha` is a partially applied scalar function handed to `tensor.unary`; `A#R#add` is handed to `tensor.binary`. The signature requires both inputs to have the same shape and element type; the frontend completes broadcast and dtype adaptation before the call. A dynamic `alpha` stays in the residual program.

**A reduction with precondition and result shape.** `torch.reduction_impl.mean_dim` (`torch.mim:5207–5221`) defines mean as a reduction followed by scaling; only the signature and the precondition check are shown:

```mim
anx lam mean_dim (F: Floating)
                            {r: Nat, s: «r; Nat»}
                            (dimension: I64)
                            (input: «s; F#A#R#T»)
                          : «torch_reduce_shape (s, torch_canonical_dim @r dimension); F#A#R#T» =
    runtime.require (torch_dim_in_range @r dimension, "mean.dim expects an in-range dim") body
    // body: canonicalize the dimension, sum with torch_reduce_dim, then scale by s#dim
```

The signature describes the result shape with `torch_reduce_shape`. `runtime.require` checks the signed dimension and receives the body function `body`. Type relations, precondition checks, and computation construction each have a definite place.

**Source call, target function, and driver.**

```text
Source:  torch.binary.add A @shape (lhs, rhs, alpha)
Driver:  DecomposeByImpl::apply_impl selects and applies torch.binary_impl.add
Target:  tensor.unary(scale_scalar A alpha, rhs)
             → tensor.binary(A.add, lhs, scaled_rhs)
```

The source axiom keeps the operation name and the stage boundary; the target `lam` gives the substituted computation. The Torch driver recursively processes the generated calls that still await decomposition; the corresponding entry of the tensor plugin is `Lower::lower_via_impl`. PE specializes the static arguments in the function body, and the driver connects the high-level axiom to its implementation definition at the corresponding stage. The visibility of public library entries and internal `_impl` definitions is managed by the plugin interface.

#### Execution, Diagnostics, and Reproducible Evaluation

The frontend implements a `torch.compile` backend named `mimir`. It invokes the MimIR pipeline, generates LLVM IR, JIT-compiles it, and handles input passing, result packing, and recovery. This path covers the complete execution from graph import to result return.

The system also provides a compile cache, implementation fingerprints, debug artifacts, a compile manifest, and per-stage profiling. An audit script records compile partitions, operator statistics, and execution information. The benchmark and KernelBench runners use separate subprocesses, resource limits, and reference-result checks. These facilities served the evaluation of manuscript [13] and support this thesis in locating failing stages and comparing transformation costs. Implementation size and counting rules are in Appendix A.

### 3.2 Algorithm Substitution: From Convolution to Im2col

This example shows the full correspondence between the source computation, the substituted algorithm, and `lower_via_impl`. Assume NCHW input layout and OIHW weight layout, no groups, no bias, and positive input and output spatial sizes. These conditions form the precondition `P` of Section 2.2. The example uses real-valued multiply-add; the floating-point implementation is verified separately under a numeric mode. To highlight the index relations, the pseudocode uses stride=dilation=1 and a 3×3 kernel; `Xp` is the input zero-padded by `(ph,pw)`.

Let `X` have shape `[N,Cin,H,W]` (batch, input channels, height, width) and `W` have shape `[Cout,Cin,3,3]`. The output is `[N,Cout,OH,OW]` with `OH=H+2*ph-2` and `OW=W+2*pw-2`.

The original cross-correlation is:

```text
for n, co, oh, ow:
    acc = 0
    for ci in [0, Cin):
        for kh in [0, 3):
            for kw in [0, 3):
                acc += W[co, ci, kh, kw] * Xp[n, ci, oh+kh, ow+kw]
    Y[n, co, oh, ow] = acc
```

Im2col [24] keeps the output semantics, moves the window reads forward into the construction of a column matrix, and then calls a matrix product. The target algorithm is defined as a library function in `tensor.mim`. Below is the full text of `trans_im2col_impl` (convolution prototype [17], `tensor.mim:2276`):

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

The body composes window extraction, layout changes, a matrix product, and output recovery in sequence. With `K=Cin*KH*KW` and `Q=N*OH*OW`, it builds three logical relations:

```text
Col[k,q] = Xp[n,ci,oh*sh+kh*dh,ow*sw+kw*dw]     # sh,sw are the strides, dh,dw the dilations; all 1 in this example
Z[co,q] = sum(k in [0,K), reshape(W,[Cout,K])[co,k] * Col[k,q])
Y[n,co,oh,ow] = Z[co,q]

k = (ci*KH+kh)*KW+kw
q = (n*OH+oh)*OW+ow
```

Here `im2col`, `reshape`, `transpose`, and `product_2d` are library interfaces whose implementations can be layered further. As with the Torch decomposition of Section 3.1, the whole substituted computation is first given by function composition and then unfolded at the corresponding stage.

One reduction term `(ci,kh,kw)` of the original loop corresponds one-to-one to one `k` of the matrix product; one original output position `(n,oh,ow)` corresponds one-to-one to one column `q`. Therefore, once a window value enters `Col[k,q]`, it is still multiplied by the original weight `W[co,ci,kh,kw]`. The actual implementation supports general legal kernels, strides, and dilations, with the output size computed by `conv_shape`; the example above fixes one set of parameters. The author has checked both index equations with independent small-scale computations: cases with multiple batches/channels, odd sizes, valid and asymmetric padding, and all-singleton axes agree with direct cross-correlation.

#### Call Replacement and Driver Chain

The entry is an explicit `tensor.trans.im2col` call; the driver chain is the same as in Section 3.1:

```text
Source:  tensor.trans.im2col R @shape params (X, W)
Driver:  Lower::rewrite_imm_App → lower_via_impl(app, impl<tensor::trans_im2col_impl>())
Target:  trans_im2col_impl R @shape params (X', W'); after PE it unfolds to im2col/reshape/transpose → product_2d → NCHW recovery
```

`X'` and `W'` are the dynamic inputs after the driver's rewrite. The call chain follows the API of the convolution prototype [17]; version differences are in Appendix A.

| Step in the pseudocode | Construct in the current implementation | Correspondence to verify |
|---|---|---|
| `Xp` and window reads | access maps of `pad` and `im2col` | padding value and stride/dilation offsets |
| `Col[k,q]` | transpose/reshape after `im2col` | `k=(ci*KH+kh)*KW+kw`; `q=(n*OH+oh)*OW+ow` |
| `W2 * Col` | `product_2d R @(Cout,K,N*OH*OW)` | the reduction axis is the same `K`, with no batch mixed in |
| `Z` back to `Y` | output reshape/transpose | recovers NCHW rather than changing the external layout |

PE plugs the known types, shape expressions, map functions, and scalar operations into this algorithm. The column matrix is kept by `compute_at 0` at the root of the consumer for reuse by the matrix product. The evaluation keeps the IR before replacement, after unfolding, and after loop generation to verify the correspondences in the table.

### 3.3 Algorithm Substitution: From Convolution to Winograd

Winograd [23] keeps the 3×3 cross-correlation of Section 3.2 as source semantics; the target algorithm changes the multiplication structure. F(2×2,3×3) transforms each 4×4 input tile through fixed matrices to 16 points, performs a pointwise channel reduction in the transformed domain, and transforms back to a 2×2 output. The multiplications per 2×2 output window drop from 36 to 16, at the cost of extra additions, transforms, and intermediate storage. The algorithm itself belongs to the algorithm library implemented in MimIR; this section only describes the library organization it shares with im2col. The full definition is in the patch of the Appendix A materials (`tensor.mim:2294–2397`).

The structure of `trans_winograd_impl` is as follows (the stages are summary names; `TH=ceil(OH/2)`, `TW=ceil(OW/2)`, `Tiles=N*TH*TW`):

```text
lam conv_winograd(mode, format, shape, padding)(X, W) =
    let Xp = pad_for_tiles(X, padding, OH % 2, OW % 2)
    let U = compute_at(0, weight_transform(mode, W))     # G·g·Gᵀ         [16,Cout,Cin]
    let V = compute_at(0, input_transform(mode, Xp))     # Bᵀ·d·B         [16,Cin,Tiles]
    let M = bmm(float_ring(mode, F), U, V)              # pointwise channel reduction  [16,Cout,Tiles]
    output_transform(mode, M, [N,Cout,OH,OW])           # Aᵀ·m·A, writes only valid outputs
```

The three transforms are point functions passed to `map_reduce` (`weight_point`, `input_point`, `output_point`) together with their access maps; the fixed matrices are composed from one-dimensional scalar `lam`s (such as `winograd_weight`). The entry `tensor.trans.winograd` applies this definition through `lower_via_impl`, as in Section 3.2. The preconditions are a 3×3 kernel, stride=dilation=1, positive input dimensions, and a non-empty output. Odd output sizes are zero-padded at the high end of the input so that the last tile's reads are valid.

The difference from im2col lies in preconditions and verification. Im2col can explain its rearrangement window term by window term. Winograd additionally uses algebraic identities and depends on subtraction and factors of ½, so its API uses a floating-point format and a numeric mode, and rounding differences are verified under the numeric error contract of Section 2.2. The tests in the materials make this contract concrete as an acceptance criterion: with a double-precision scalar cross-correlation as reference, every output must satisfy `|out − ref| ≤ ε·(1 + Σ|x·w|)`, with `ε = 2e-5` for FP32 and `2e-12` for FP64. The test inputs cover singleton axes, multiple batches, asymmetric padding, odd output tiles, channel tail blocks, and FP64. This is a test criterion for selected inputs, not an error bound for arbitrary inputs.

Both algorithms have a known range of applicability. Im2col materializes the `[K,Q]` column matrix, so its memory traffic is about `KH·KW` times that of the original input; it only pays off when the subsequent matrix product is large enough to be amortized by blocking and vectorization. Winograd's saved multiplications must offset the additions of three transforms and the reads and writes of the `[16,·,·]` intermediate tensors; it only pays off when `Cin·Cout` is large enough for the channel reduction to dominate [23]. Configurations with small channels, small batch, and small spatial sizes are expected to fall behind direct convolution; production libraries select the algorithm by shape for the same reason. This thesis leaves algorithm selection to the driver and delimits each algorithm's applicable range by measurement (Section 6.1).

Torch decomposition, im2col, and Winograd use the same library organization: define the target computation with a `lam`, then select the definition and specialize it. Their mathematical preconditions, target level, and intermediate storage are expressed through dedicated parameters and checks. These two examples give an initial judgment on RQ1: when the substituted algorithm has a regular structure, when scalar operations and maps can be passed explicitly, and when the preconditions are checkable, HOF/PE is suited to carry the construction of the result. Algorithm selection, the benefit of materialization, and the numeric contract still have to be established separately.

### 3.4 From Buffer Operations to Explicit Loops (Cost Comparison)

This section takes the implemented lowering of `btensor.map_reduce_post` [20] as a concrete example of lowering. The source operation already uses buffers for inputs and outputs, but iteration, reduction, and the epilogue are still described by one operation. The target program gives its computational semantics with explicit loops, element reads, accumulator updates, and result writes. The lowering idea is the same as in Sections 3.1 and 3.2–3.3: compose the semantics of the source operation from the operations and existing functions of the target level. In this case, the existing element function and the chosen schedule are reused, and iteration and reduction are implemented explicitly by loops.

This implementation builds on the existing loop constructs: `mr_nest` provides the nested loops and accumulator passing, and `sched` is a schedule function acting on it that determines loop order and blocking. As described in [13, §7], this is exactly the junction between the schedule of Section 4 and the lowering of this section: the schedule first decides the loop structure on the `Tileable`, and the lowering then partially evaluates the nest builder with that schedule to materialize the loops. The new `.mim` implementation composes the element reads and `f` into `cell`, composes the output map, `post`, and the write into `wb`, and hands both to the resulting `nest`. What MimIR expresses, therefore, is the whole composition logic of the lowering, including how the existing computation and loop constructs are reused.

**Source operation:** the leading groups of type, shape, and schedule arguments are omitted below.

```text
btensor.map_reduce_post(...)(f, init, post) map_out (maps, post_maps)
    (mem, dst, inputs, post_inputs)
    → [mem, dst]
```

`f` receives the accumulator and the input elements; `post` receives the reduction result and extra inputs. Both are reused unchanged in the transformation. Buffer handles are passed as operands, and access maps describe how to read each buffer from the loop coordinates. The existing implementation covers multi-dimensional maps, reductions, heterogeneous inputs, and the epilogue.

**Target implementation:** the semantic pseudocode below corresponds to `map_reduce_post_impl` of [20]; dependent types, the conversion of loop counters to bounded indices, and the CPS continuations are omitted. At write-back time, `i_out` is the output coordinate, and the reduction coordinates are zeroed by the existing convention before being passed to `map_out`.

```text
lam map_reduce_post_impl(..., f, init, post, read_inputs, read_post)(mem, dst) =
    lam cell(mem, acc, i) =
        let (mem, values) = read_inputs(mem, i)
        f(mem, acc, values)
    lam wb(mem, dst, acc, (i_out, i_red)) =
        let (mem, o) = affine.map(map_out, (i_out, 0_red), mem)
        let (mem, extra) = read_post(mem, o)
        let (mem, value) = post(mem, acc, extra)
        buffer.write(mem, dst, o, value)
    let nest = sched(...)(mr_nest(...))
    nest(init, cell, wb)(mem, dst)
```

`cell` and `wb` are function wrappers that connect computation to storage. They keep the bodies of `f` and `post` and add coordinate interpretation, reads and writes, and memory-token passing around the calls. The read function is itself composed from `read_input`, `finish_reader`, and `input_reader`, which read the elements in turn according to the static input description. `nest` plugs these functions into their loop positions and decides when to call `cell` and when to hand the accumulated result to `wb`.

For example, for the classic loop schedule with a two-dimensional output and a one-dimensional reduction, the composition has the following semantics; other schedules are given by the same `mr_nest` interface:

```text
for i, j:
    acc = init
    for k:
        (mem, acc) = cell(mem, acc, (i, j, k))
    (mem, dst) = wb(mem, dst, acc, (i, j, 0))
```

After PE specializes the function composition, the residual program contains the concrete loops, reads, scalar computation, and write-back. What is eliminated is the static function abstraction; the memory operations and loop semantics introduced by the wrapper layer remain in the target program.

| Construct | Reused or new responsibility | Target semantics |
|---|---|---|
| `f`, `post` | reuse of the existing computation functions | accumulator update and epilogue |
| `mr_nest` and `sched` | reuse of the existing nested-loop construct and schedule | iteration order, loop-carried values, and write-back timing |
| read functions and `cell` | composition of access maps, reads, and `f` in MimIR | element reads and computation at each loop point |
| `wb` | composition of the output map, `post`, and the write in MimIR | output update after the reduction completes |
| `map_reduce_post_impl` | application of `nest(init, cell, wb)` | the complete loop and read/write program |

**Driver chain and level boundaries:**

```text
tensor.lower_to_mem → btensor.map_reduce_post (buffer operation)
btensor.lower_map_reduce → applies map_reduce_post_impl
                         → affine.For + buffer.read/write + element computation
buffer.lower_ptr → mem.Ptr / lea / load / store
```

[20] moves `cell`, `wb`, and the element reads from the original `LowerMapReduce::lower_map_reduce_post` into `btensor.mim`. The current C++ driver still binds static metadata, chains the heterogeneous read functions according to the input description, and adapts the already scalarized callbacks. Allocation of the target buffer and the type/value mapping are handled by the preceding `tensor.lower_to_mem`. Afterwards, `buffer.lower_ptr` turns element accesses into pointer operations, and the affine lowering turns loops into control flow. Section 5 discusses the global analysis needed for storage reuse.

With the same schedule fixed, the C++ graph construction and the MimIR composition should keep the same access maps, accumulator updates, epilogue, and memory operation order. The tests cover multi-dimensional indices, heterogeneous inputs, logical singleton axes, empty domains, and the memory chain. Section 6.3 uses these two existing paths for a controlled comparison: with the element function, allocation strategy, `sched`, and `mr_nest` fixed, it varies shape, number of calls, and degree of graph sharing, and compares driver size, compile cost, and residual structure. This provides an executable case for "implementing lowering by HOF composition" and makes the responsibilities of the composition layer and the external conversion driver explicit.

## 4. Loop Transformation: Expressiveness and Contracts of Expert Manual Schedules (Core RQ2)

### 4.1 Motivation: Testing the Expressiveness of Higher-Order Transformations with Expert Schedules

The second kind of work in Section 1.1 is loop transformation. It is usually done through a schedule API whose goal is to let developers organize high-performance computation. The GEMM/GEMV implementations of OpenBLAS [19] give a concrete reference: cache blocking, packing, register blocking, and vectorization along the contiguous access direction together determine single-threaded performance. This thesis wants to compose these structures from schedule primitives implemented as HOFs. On a selected CPU, shape set, and numeric setting, single-threaded GEMM/GEMV should approach OpenBLAS, and CONV should approach an optimized reference implementation of the same algorithm. Multi-threading is handled by a separate OpenMP implementation and is out of scope.

This turns expressiveness into a testable question: can an expert, while preserving operator semantics, obtain the required loop nest, data layout, and local storage by composing schedule functions, and does PE plus the backend generate efficient code from it? The experiments keep the schedule definition, the actual schedule report, and the LLVM IR, verify the key structures, and then measure runtime performance. Structural correspondence and performance results together provide empirical evidence for the expressiveness of HOF transformations. Performance gaps are attributed to the schedule interface, lowering, code generation, or runtime.

| Workload | Key structures the manual schedule must express | Performance reference |
|---|---|---|
| GEMM | multi-level loop blocking, operand packing, register accumulators, and vectorization | single-threaded OpenBLAS GEMM with the same computational semantics |
| GEMV | vectorization along the contiguous access direction, reduction blocking, and output reuse | single-threaded OpenBLAS GEMV with the same layout and transposition |
| CONV | with the im2col algorithm fixed, arrange materialization and reuse of intermediates, reusing the GEMM schedule | an end-to-end implementation with the same im2col transform, layout recovery, and allocation strategy, plus OpenBLAS GEMM |

Sections 3.2–3.3 decide the algorithm structure; this section decides the loop and storage arrangement inside the chosen algorithm. CONV first uses the existing im2col path; its reference implementation is defined in Section 6.3. Appendix B shows that the current im2col path is slower than direct in 10 of 12 single-threaded configurations. Section 3.3 gives the expectation at the algorithm level; beyond that, there is a gap at the schedule level: direct's `tensor.conv` already uses the register-blocked map_reduce schedule of [13], while the matrix product unfolded from im2col is not yet scheduled. RQ2 asks exactly whether manual scheduling can close this gap. For GEMM itself, Section 6.1 reports the starting point: upstream MimIR's default schedule reaches a geometric-mean ρ of 0.38 against single-threaded OpenBLAS, and the prototype's `dot_trace` schedule 1.08 (median 0.98), with one shape below 0.8. If the scheduled im2col CONV is still slower than direct, Section 6.3 reports both and separates the gap into algorithm and schedule.

### 4.2 Schedule Sketch: Blocking, Packing, and Register Accumulation for Single-Threaded GEMM

The convolution prototype [17] already writes schedules on `Tileable` as ordinary functions. The **current schedule** is `dot_trace (nc, rows, kblk, packed)`, which applies three steps in turn to `C[M,N]=A[M,K]·B[K,N]`; `dot_params` chooses the parameters automatically from `target.cache_elems` and `target.acc_regs`:

```text
t1 = dot_nblk (r_out, nb, na, nc) t    # strip_mine_par(N, nc) + interchange: block N by nc, move the block loop outside M
t2 = dot_schedule_rb (r1, rows) t1     # strip_mine_par(N,16) + partition_par(M,rows) + interchange + vectorize,
                                       #   ending with a call to dot_k: a rows×16 register block, innermost K unrolled by 4
t3 = dot_kblk (r_out, na, nc) t2       # only when the K×nc panel exceeds cache_elems: strip_mine_red + hoist_red,
                                       #   the K block size is computed inside dot_kb; the register block reads/writes the accumulator panel between K blocks
b_packed = cache_read b (N, 16, 0, level)   # packed path: B is rearranged into «N/16, K, 16» panels
```

Here `r_out` is the output rank, `nb` and `na` are the numbers of free dimensions of the right and left operands, which decide outside which loops the N block loop moves, and `r1` is the output rank after blocking N. Without K blocking, the accumulator of the register block stays in registers across the whole of K. With K blocking, the accumulator panel is written back at the end of each K block. These three steps correspond to the `nc`/`kc` cache blocks, the `mr×nr` register block, and the packed panels of OpenBLAS GEMM, but `kc` has no explicit interface and only the right operand is packed.

The **proposed expert interface** is `gemm_schedule (nc kc rows: Nat, pack_a pack_b: Bool)`. It reuses the combinators above, hands `nc`, `kc`, `rows`, and the packing choice to the expert explicitly (`kc` replaces the internal computation of `dot_kb`), and uses the same `cache_read` to add packing of the left operand block. The experiments compare the generated structure and performance item by item against the blocking parameters of OpenBLAS. GEMV uses `dot_schedule_kvec`: the contraction dimension becomes the vector dimension with a horizontal sum, and the output cells are unrolled-and-jammed in two dimensions. This thesis makes its parameters explicit as well.

### 4.3 Core Loop Contracts: Composing Split, Fuse, and Interchange

[13] provides the construction basis of `Tileable`, `retile`, and functional schedules. [17] has `strip_mine_*`, `fuse_iter`, `interchange`, and partial predicate propagation. In this section, fuse means merging two adjacent axes into one (`fuse_iter`); it is unrelated to the operator fusion of Section 5. This thesis systematizes the preconditions of these three transformations into contracts, completes the rejection paths after composition, and adds automated verification of coverage and reduction order. Verification concentrates on rectangular domains, pure elementwise computations, and one reduction tail-block case.

A structured computation records the iteration domain `D`, the roles of the axes, the read maps `Aj`, the write map `W`, and the predicate `P`. Each transformation gives `φ:D'→D`, which maps new coordinates back to the original ones; the reads and writes become `Aj∘φ` and `W∘φ`, and the composition of two transformations is `φ₁∘φ₂`. With the elementwise map `y[i]=f(x[i])` and its two-dimensional form as source programs, the sketches of the three transformations are:

```text
split(N, b):        D' = [ceil(N/b), b]   φ(io,ii) = io*b + ii   P'(io,ii) = io*b+ii < N
fuse(A, B):         D' = [A*B]            φ(t) = (t/B, t%B)      precondition: B > 0, A*B does not overflow
interchange(perm):  D' = D∘perm           φ(o) = o∘perm⁻¹        precondition: perm is a permutation (length Rp+Rr, i.e. the prototype's number of parallel axes plus reduction axes, see Section 2.1; no duplicates, in range) and does not mix roles
```

The driver skeleton is the same for all three: check the preconditions (axis valid, factor/role/adjacency, extent and index arithmetic safe) and return a diagnostic if they fail; build `D'`, `φ`, and `P' = valid ∧ P∘φ`; call `retile`; migrate the axis metadata (`vdim`, `oax`). The `interchange` of [17] already uses `refly.check` to reject role-mixing permutations at compile time and migrates `vdim` and `oax` with the axes. But it does not check that `perm` really is a permutation: `(0,0,2)` satisfies the role condition but has no inverse. Adding the length, range, and uniqueness checks, together with rejection tests for duplicate, missing, and out-of-range axes, is the first deliverable of the contract layer. The checks for split and fuse are completed in the same way.

The reduction tail block uses an obligation stronger than the observable equivalence of Section 2.2. For `acc=fold(k∈[0,K),step)`, after a split the execution follows the lexicographic order of `(ko,ki)`, every valid point calls `step` exactly once, and invalid points return the original `acc`. The guard of the store only protects the write; the load and the reduction update each need valid control flow. An interchange that permutes the order among reduction axes changes the floating-point summation order and is constrained by the numeric mode. Moving a reduction axis outside a parallel axis changes the lifetime of the accumulator and is checked separately, as in the `hoist_red` case.

| Deliverable | Concrete checks and verification |
|---|---|
| Split | axis valid, `b>0`, index arithmetic safe, valid coverage and tail-block control flow |
| Fuse | adjacent rectangular independent axes, consistent roles, empty domain, product does not overflow, output map preserved |
| Interchange | `perm` is a legal permutation (length, uniqueness, range), no role mixing, `vdim`/`oax` migrate with the axes, reduction order constrained by the numeric mode |
| Composition | predicate propagation, map composition, consistent axis metadata; illegal inputs actually terminate the transformation |
| One reduction split | each contribution once, order preserved, invalid lanes keep the accumulator |

TVM [8–9] provides a reference for contracts and diagnostics. The other interfaces in the sketch above are checked on the selected cases through numeric results and the actually generated structure. The small-domain exhaustive checker (Section 6.4) verifies coverage, unique writes, and contribution counts.

### 4.4 Follow-up Direction: From Expert Schedule Skeletons to Auto-Tuning

This thesis first establishes manual schedules that an expert controls, that are explainable, and whose performance approaches optimized reference implementations; it considers automatic parameter selection afterwards. TVM MetaSchedule [18] shows the connecting path: schedule rules or custom schedule functions generate the design space, and evolutionary search mutates the sampled decisions. The `nc`, `kc`, `rows`, and unroll factors in the sketch above are exactly such decisions. Exposing them as samplable parameters and reusing the legality checks is work that follows this thesis.

## 5. Boundary Cases for Global Analysis

Operator fusion and general buffer reuse serve as scope boundaries that explain the external information local function construction needs. This thesis sets no separate RQ, implementation milestone, or performance experiment group for them.

The difference between a pure chain and a shared producer can be shown with one function example:

```text
Source graph:
    t = map(f, x)
    y = map(g, t)
    z = map(h, t)

Shared materialization:
    lam materialized(f,g,h)(x) =
        let t = map(f,x)
        (map(g,t), map(h,t))

Separate inlining:
    lam inlined(f,g,h)(x) =
        (map(lam(v)=g(f(v)), x), map(lam(v)=h(f(v)), x))

Driver boundary:
    plan = analyze_uses_effects_and_cost(source_graph)
    apply(plan.library_function, rewritten_arguments)
```

Both target functions can be expressed by the library. To choose between them, the driver also needs use-count, effect, and cost information; the second variant executes `f` twice. In the thesis this example separates the responsibilities of "what to choose" and "how to construct".

Buffer reuse takes `t1=update(t0,i,v); r=read(t0,j)` as a counterexample: writing `t1` in place into `t0` requires proving that later reads of the old value remain safe. Memory tokens express ordering; aliasing, ownership, and lifetime come from additional analyses. Section 3.4 keeps the allocation decisions of the existing bufferization and compares only the construction of the local composition. The analysis/rewrite separation of One-Shot Bufferization [7] is a reference for later extensions.

## 6. Experimental Evaluation: Existing Results and Further Verification

### 6.1 Existing Results and Their Scope of Interpretation

The frontend has been used for the end-to-end evaluation of manuscript [13]. The run of 15 October 2026 (the `src/` of the frontend working tree equals commit `619a157`) shows 100/100 for KernelBench [25] Level 1 and 99/100 for Level 2. Each set has 100 scaled cases; the original problems are scaled down with size-divisor=16. These counts therefore speak to the correctness of import and execution, not to performance at the original scale. They are also static-shape results: the runner compiles with `dynamic=False` (Section 3.1). The two raw JSON files and logs are archived in the materials directory of Appendix A; the counts are:

| Test set | PASS | FAIL | INVALID | Evidence |
|---|---:|---:|---:|---|
| Level 1, 2026-10-15 | 100 | 0 | 0 | all 100 cases completed the reference comparison |
| Level 2, 2026-10-15 | 99 | 1 | 0 | the only failure is case 28; the 3 INVALID of the old report are fixed |

Level 2 case 28 is `28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py`; it fails in the compile-and-execute stage, and the log reports that the LLVM backend received an unhandled tensor value. This failure belongs to the tensor-to-memory/lowering path and must be distinguished from FX import errors.

The raw result files record the per-case status and stage. The two test sets ran concurrently, so per-case durations are not timing data. Run parameters, component versions, and the machine are in Appendix A and in `frontend/manifest.json` of the materials. KernelBench counts and frontend unit-test coverage are reported separately.

Historical reports document the evolution: on 2026-08-24, Level 2 had 96 PASS, 1 FAIL, 3 INVALID; the maintained fixtures of Level 3 had 27 PASS, with 21 missing scaled fixtures and 2 missing dependencies. The CPU evaluation of [13, §8.3] used exactly the subset of Level 3 models supported at the time; this thesis does not treat this historical count as the current result, and reruns Level 3 by version after the environment is fixed in weeks 1–2 (Sections 6.2 and 8). The historical reports used `size-divisor=16`, a 240-second timeout, a 16 GiB address-space limit, and a budget of 512 fixed-point iterations; their versions and conditions are kept in the repository reports [15].

The convolution prototype has independent preliminary end-to-end measurements. The table below comes from single-threaded FP32 fast runs on an Apple M2 Pro; all are 3×3, stride=dilation=1, padding=1, timing includes the transform and allocation/deallocation, and the weights are re-transformed on every call.

| Shape: `N,Cin,Cout,H,W` | Direct | Im2col | Winograd |
|---|---:|---:|---:|
| `4,128,128,28,28` | 25.19 ms | 38.85 ms | 18.83 ms |
| `4,512,512,7,7` | 54.29 ms | 34.84 ms | 22.26 ms |

Winograd is 1.3 to 2.4 times faster than direct in these two large-channel, batch=4 cases and slower in the other 10 single-threaded configurations. Im2col is slower than direct in 10 of 12 single-threaded configurations. Appendix B gives all 12 rows.

This distribution matches the algorithmic expectation of Section 3.3. Both algorithms trade extra memory traffic or transform arithmetic for fewer multiplications and only pay off when the channel reduction is large enough. The losing configurations concentrate on small channels (`Cin=3`, `16`, `32`), batch=1, and small spatial sizes, exactly the range where both algorithms are known not to win. This thesis does not treat the losses in these configurations as a problem of the construction method. These results delimit the applicable range of the algorithms, and the driver selects the algorithm by shape. The part where im2col still trails direct on large-channel configurations has the schedule gap pointed out in Section 4 and is the object of RQ2. The cost of the construction mechanism is measured separately in Section 6.3.

**GEMM.** Two measurements document the state of single-threaded FP32 GEMM before the thesis starts. Both ran on the same Apple M2 Pro against Homebrew OpenBLAS 0.3.34 with one thread (its `DYNAMIC_ARCH` dispatch selects the `neoversen1` kernel), and both report `ρ = T_OpenBLAS / T_MimIR` as defined in Section 6.3. Raw data and provenance are in `artifacts/gemm-2026-10-16/`.

Harness A is the author's multi-backend benchmark `gemm-bench` (run of 12 October 2026, harness commit `1682059`). It measured upstream MimIR master `2d21e0d` (12 October 2026), whose `tensor.product_2d` uses the register-blocked `dot_schedule` of [13] without cache blocking or packing. Each backend runs nine batches of at least 100 ms interleaved with a PyTorch baseline, and the median per-call time is reported; MimIR's time includes allocating and freeing the output, OpenBLAS writes into a preallocated buffer. Over the 16 GEMM shapes, ρ ranges from 1.44 (64³) to 0.26 (2000³) with a geometric mean of 0.38; from 256³ upwards MimIR sustains 22–35 GFLOP/s against OpenBLAS's 74–98.

Harness B is the `.test/gemm_bench` harness of the convolution prototype [17], run on 16 October 2026 on branch `feature/par` at `c56d586` with uncommitted edits to `tensor.mim` and `lower.cpp` (the diff stat is archived). Its `product_2d` uses the `dot_trace` schedule of Section 4.2: N blocking, a rows×16 register block, K blocking when the panel exceeds the cache, and a packed right operand. The harness reports the best of N calls per shape with a hot cache; MimIR's time again includes allocating and freeing the output. Over 17 shapes, ρ has a geometric mean of 1.08 and a median of 0.98; 16 of 17 shapes are at or above the 0.8 threshold of Section 6.3. The table shows the shapes both harnesses cover plus two prototype-only shapes; Appendix C lists every row.

| Shape `M×K×N` | OpenBLAS ms (harness A) | upstream master ms | ρ | OpenBLAS ms (harness B) | prototype ms | ρ |
|---|---:|---:|---:|---:|---:|---:|
| `128×128×128` | 0.078 | 0.092 | 0.85 | 0.065 | 0.045 | 1.44 |
| `512×512×512` | 2.908 | 8.364 | 0.35 | 2.671 | 2.837 | 0.94 |
| `1024×1024×1024` | 23.422 | 78.816 | 0.30 | 22.098 | 24.190 | 0.91 |
| `2048×2048×2048` | 191.857 | 657.874 | 0.29 | 174.832 | 199.827 | 0.88 |
| `8×2304×512` | 0.339 | 0.597 | 0.57 | 0.307 | 0.224 | 1.37 |
| `64×130×512` | 0.101 | 0.336 | 0.30 | 0.089 | 0.091 | 0.98 |
| `1024×64×1024` | – | – | – | 1.329 | 1.927 | 0.69 |
| `1024×1024×1008` | – | – | – | 21.982 | 23.670 | 0.93 |

Three readings follow. First, the difference between the two MimIR versions, about 3× on large shapes, is the effect of cache blocking and packing written as schedule functions; RQ2 makes these choices explicit through `gemm_schedule` instead of leaving them to `dot_params`. Second, the prototype is within 0.84–1.01 of OpenBLAS on square shapes from 384 upwards and ahead on small and skinny shapes, where OpenBLAS's fixed overhead dominates. The only shape below 0.8 is `1024×64×1024` (ρ 0.69): with K=64 the register block cannot amortize the packed panel, which is the case an explicit `kc` targets. Third, both harnesses are single runs on a laptop without core pinning, harness B reports best-of rather than median and measured a dirty working tree, and the two harnesses differ in timing scope, so their absolute numbers are not interchangeable. The thesis replaces both with the fixed protocol of Section 6.3.

### 6.2 Evaluation of the Torch Operator Decomposition Layer (Frontend)

The frontend carries the highest layer of lowering and is the main implementation contribution; it needs an evaluation independent of backend optimization. The tests answer, separately:

- **Is import correct:** check overloads, scalar arguments, shape, dtype, logical rank, multiple outputs, and diagnostics for unsupported inputs.
- **Is information preserved:** compare the representation of static and symbolic dimensions, shared symbols, singleton axes, and PyTorch-specific dimension rules in the generated graph.
- **Does it execute:** compare numeric results of `torch.compile` with eager PyTorch and record whether a failure occurs in tracing, import, lowering, code generation, or execution.
- **How far does model coverage reach:** distinguish graph partitions, graph breaks, original models, and scaled fixtures, and state which graphs and inputs take part in compilation.

The existing unit tests and the KernelBench runner already cover the main paths of the first three items. Weeks 1–2 consolidate these tests, complete the report that classifies failures by stage, and rerun Level 3 (Section 8). The report fixes the versions of frontend and MimIR, the decomposition policy, dependencies, input configuration, and resource budget. New tests concentrate on the existing frontend interfaces, the preconditions of the core transformations, and the classification of known failures.

### 6.3 Controlled Evaluation of Transformation Construction

The research uses two layers of experiments. System-level experiments import models through the same Torch frontend and test the actual code paths and performance. Mechanism experiments fix the algorithm, schedule, backend, and optimization options as far as possible and change only the way the result is constructed, to isolate the cost of HOF/PE. The scope and evidence of the two research questions are defined in Section 1.3; this section explains how the experiments produce that evidence.

"Which is faster, direct or Winograd" answers an algorithm performance question; "what does it cost to construct the same Winograd algorithm through HOFs or by hand" answers the construction mechanism question. The main experiment for construction cost is the C++/HOF comparison of the existing buffer-to-loop lowering of Section 3.4: both constructions reuse the same element function, access maps, allocation strategy, and loop schedule, and only the construction method is compared. This comparison is the main evidence for the construction cost of RQ1. The construction cost of the convolution algorithm lowering would additionally require a same-algorithm C++ counterpart of im2col; it is implemented as an extension item. If implemented, the comparison shares inputs, output layout, materialization points, matrix product, and schedule; any differences are reported with the results.

The manual schedule experiments test the expressiveness of RQ2. GEMM/GEMV fix dtype, shape, layout, transposition, and alpha/beta semantics against OpenBLAS [19], with OpenBLAS fixed to a single thread. CONV fixes the im2col algorithm so that the reference implementation and MimIR use the same window transform, output recovery, and allocation strategy, with the reference calling OpenBLAS GEMM. All experiments are single-threaded and use static shapes; for a symbolic extent that does not normalize, the `if_static` selector of [13] falls back to a tile size of 1, a case this thesis does not measure. Shapes and versions are reported item by item so that gains from different algorithms are not credited to loop scheduling.

Relative performance is reported as `ρ=T_reference/T_MimIR`. The workload set is fixed before the experiments and written to the materials directory: 6 to 8 representative shapes each for GEMM, GEMV, and CONV, including non-divisible sizes and small cases. OpenBLAS is fixed to one build (version, target kernel, single thread) and recorded. `ρ` is reported per shape with a geometric mean. `ρ ≥ 0.8` counts as close; cases below it are not dropped but answered with the generated structure and a gap analysis. The matrix-product/matrix-vector part and the end-to-end timing including transforms, packing, allocation, and deallocation are reported separately. Each result corresponds to an actual schedule and LLVM IR, to locate whether the expected structure was realized.

This performance experiment provides evidence for schedule expressiveness and generated code quality. The HOF/C++ construction comparison under the same algorithm and schedule continues to measure compile cost independently. The two groups of evidence answer "can high-performance schedules be expressed" and "how much does it cost to construct such programs", respectively.

### 6.4 Correctness Checks and Cost Metrics

Index tests cover length 1, empty domains, non-divisible factors, odd outputs, channel tail blocks, illegal axes, and zero factors. Small-domain exhaustive checks verify valid-point coverage, write conflicts, and reduction contribution counts; differential execution checks actual results. The exhaustive checker is delivered as a standalone test tool: given an iteration domain, maps, and predicate, it outputs violations of coverage, unique writes, and contribution counts. Floating-point transformations state their numeric mode, reference values, and error criterion separately.

The buffer-to-loop lowering checks multi-dimensional access maps, accumulators, the epilogue, the memory chain, and empty domains; elementwise maps additionally check input preservation. Diagnostic tests check that illegal inputs actually terminate the transformation; residual verification checks target operations and boundary types. Source-level differential execution and LLVM verification provide semantic and IR-structure evidence, respectively.

Cost metrics are divided into compiler cost and generated-program cost. The former includes time, node counts, and memory; the latter includes allocations, copies, vectorization, and runtime. Shorter implementation code cannot replace these two kinds of measurement; the table below lists the counting rules.

| Metric | Counting rule |
|---|---|
| Implementation cost | construction-specific code and all new drivers/helpers; tests, generated code, and existing facilities listed separately |
| Compile cost | time per stage, number of specializations/rewrites, intermediate and residual nodes, peak memory |
| Scalability | change under growth of repeated instances, shared DAGs, schedule depth, and unroll factors |
| Residual structure | functions/closures, loops, guards, vector operations, allocations, and copies |
| Runtime quality | cold compile, warm-up latency, component time, error, and allocated bytes |

LOC follows one uniform rule; shared facilities are counted rather than treated as free in one implementation. Comparable counts of branch or graph-operation calls serve as an auxiliary metric, and cross-language cyclomatic complexity is not used directly as a quality ranking. Runs alternate order, avoid parallel compilation, report medians and variation, and state caching, weight pre-transformation, and copy costs explicitly.

If definitions of the same kind can be reused, static abstractions are eliminated, and the cost is controllable, the method is supported. This thesis sets no pass line for cost in advance; it reports, item by item under the same algorithm and schedule, the compile-time ratio and intermediate-node ratio of HOF construction relative to C++ graph construction, and whether closures or superfluous allocations remain in the residual program. If the generic interface still requires large amounts of dedicated graph reconstruction, or if specialization cost grows markedly with the number of instances, the thesis reports that the advantage is limited. The thesis reports benefits and costs according to the measurements.

## 7. Related Work and Positioning of This Thesis

### 7.1 MimIR and Multi-Level Compilation

The POPL 2025 MimIR paper [12] provides the foundation of the core language, plugins, normalization, PE, and the tensor `map_reduce`. This thesis reuses these core mechanisms and studies the construction interfaces, costs, and additional preconditions of the selected transformations.

"Principles for Multi-Level Compiler Design" [13] already discusses affine maps, symbolic shapes, PE lowering, `Tileable`, and schedules as values, and provides implementation sizes and a system-level evaluation. The author's Torch frontend is the implementation contribution behind that evaluation. This thesis documents the lineage of these results from the code and the collaboration record, and concentrates its new contributions on the source/target correspondence of lowering, the performance test of expert schedules, the split/fuse/interchange contracts, and the controlled cost experiments.

### 7.2 Partial Evaluation, Functional Construction, and Schedule Languages

PE, typed interpreters, and staging have a long research history [1–3]. AnyDSL [10] uses partial evaluation to write high-performance libraries as ordinary functions, with the user specifying the specialization points in the source language. MimIR inherits this mechanism but places it in a dependently typed IR, so that shapes and interfaces become part of the types. Halide [26] first separated algorithm from schedule, with the user describing the execution strategy in a schedule language. The schedule API of this thesis follows this division, but the schedule is a function in the IR rather than a separate language. Elevate [14] expresses optimizations as functional rewrite strategies and guarantees traceability through strategy composition. It acts on rewrites of the source program, whereas the schedules of this thesis act on IR values such as `Tileable` and are eliminated by PE.

Compared with this work, this thesis proposes no new specialization or rewriting mechanism; it tests the engineering boundaries of these mechanisms on the existing Torch→MimIR chain: which preconditions and contracts the selected transformations need, whether expert schedules reach reference-library performance, and how much interface reuse, driver burden, and specialization cost each amount to.

### 7.3 MLIR and TVM

MLIR's PDL dialect, the PDLL language, and the Transform dialect provide declarative rewriting and transformation orchestration [5–6]. Dialect Conversion manages legality, value mapping, and type adaptation [4]. According to Table 1 of [13], linalg's 81 named operations take about 8.6k lines across three languages, whereas the 33 operations of MimIR's tensor plugin take 431 lines in one language. Torch-MLIR's operator conversion layer consists of C++ rewrite patterns with type converters and operation verifiers [21]. TVM Relax's legalization consists of Python functions that reuse TE/TOPI [22]. Both give the substituted computation in compiler code and verify the well-formedness of the generated IR; the rank/shape correspondence between operator signature and substituted implementation is not expressed in one type. This thesis distinguishes three things: the static constraints of the interface type, the verification of generated IR, and the correctness of the transformation semantics. MimIR differs in the first: the rank/shape correspondence is written in the type of the implementation function and checked at application time. The third is still carried by the contracts of Section 2.2. The corresponding layer in this thesis is a `lam` in the IR plus a driver; what is compared is the responsibility, reuse, and cost of the construction interface.

One-Shot Bufferization separates analysis from rewriting [7]. This thesis uses that separation to explain the construction boundary of Section 5.

TVM's structured scheduling and arithmetic analysis provide a reference for the core contracts and diagnostics [8–9]. MetaSchedule [18] connects schedule rules, traces, and mutation to search. TVM's schedule primitives act on TIR with legality checks implemented in C++; the schedule primitives of this thesis are functions in the IR, with legality carried by types and explicit contracts. Section 4 explains how the two connect in exposing parameters.

## 8. Work Plan and Scope

The frontend is consolidated and evaluated as the main existing implementation result. The thesis has a total time limit of six months. The frontend, the im2col/Winograd prototypes, and the buffer-to-loop lowering branch are complete; the remaining four months (16 weeks) follow RQ1 and RQ2 as the main plan:

| Time | Work and deliverables | Acceptance point |
|---|---|---|
| Weeks 1–2 | Fix code, data, and the OpenBLAS baseline; close the `scan.diff` gap; consolidate the frontend tests of Section 6.2 and rerun KernelBench Level 3; implement the C++ im2col + OpenBLAS GEMM reference harness for CONV; consolidate the existing GEMM/GEMV/CONV manual schedule prototypes and fix the representative shape set | checklist of frontend environment and performance measurement standards; frontend test classification report and Level 3 results; CONV reference harness runs; prototypes reproducible; shape set written to the materials directory |
| Weeks 3–4 | RQ1: consolidate the lowering correspondences of operator decomposition, convolution algorithms, and buffer-to-loop, and state composition and semantic contracts | correspondence tables and numeric contracts; IR samples before replacement, after unfolding, and after loop generation |
| Weeks 5–8 | RQ2 schedules: implement the `gemm_schedule` expert interface and add left-operand packing; measure GEMM first and iterate the schedule by the gap; then extend to GEMV and im2col-CONV | GEMM performance table and gap analysis; GEMV/CONV schedule definitions and actual schedule reports |
| Weeks 9–10 | RQ2 contracts: precondition checks, rejection paths, predicate propagation, and reduction tail-block verification for split/fuse/interchange; deliver the small-domain exhaustive checker | rejection tests for illegal inputs; exhaustive and differential tests pass; contract table matches the implementation item by item |
| Weeks 11–12 | RQ1 cost comparison: consolidate the buffer-to-loop lowering, build the controlled C++/HOF comparison on the existing multi-dimensional maps and reductions, and measure compile cost and residual structure | same element function and schedule reused; cost ledger, memory order, and residual structure comparison |
| Week 13 | Extension item: same-algorithm C++ construction counterpart of im2col, fixing a single shape and matrix-product entry first; dropped if behind schedule | single-shape construction cost comparison, or an explicit record of why it was dropped |
| Week 14 | Complete GEMV/CONV performance measurements and gap attribution for the three workloads; verify all residual IR | complete performance tables for the three workloads, geometric mean of `ρ`, and gap classification |
| Weeks 15–16 | Analyze boundaries, assemble reproducible materials, and finish writing | argument, materials list, and thesis |

The scope is fixed to a single CPU thread, the existing LLVM backend, the selected Torch workloads, expert schedules for GEMM/GEMV/im2col-CONV, the core split/fuse/interchange contracts, and the composition and cost of the existing buffer-to-loop lowering. The implementation reuses the existing prototypes, and performance verification follows the representative shape set fixed in advance. All experiments use static shapes. Fusion and general buffer reuse serve as boundary discussion. Multi-threaded scheduling (handled by a separate OpenMP implementation), auto-tuning, complete schedule legalization, general alias/SMT solving, GPUs, end-to-end symbolic shapes (the execution path, the dynamic-case adapters, and relations between symbols; Section 3.1), and a new TVM plugin are listed as follow-up directions.

The main risks and reduction strategies are:

| Risk | Response | Minimum deliverable kept if behind schedule |
|---|---|---|
| PE specialization blow-up | limit unfolding, record per-stage nodes and time, keep the repeated-instance/shared-graph comparison | one matched algorithm construction comparison with scaling data |
| im2col C++ counterpart exceeds budget | extension item occupying only week 13, started after RQ2 and the RQ1 cost comparison are done; fix a single shape and matrix-product entry first | drop the counterpart; keep qualitative evidence for the convolution algorithm lowering; RQ1 cost evidence comes only from the buffer-to-loop case |
| mixed sources of manual schedule performance gaps | fix algorithm and measurement standard; separate interface, backend, and runtime factors using the schedule and LLVM IR | representative shapes, actual structure, and gap analysis kept for GEMM/GEMV/CONV |
| legality scope expands | reject unknown preconditions; restrict to rectangular domains and same-role axes first | core coverage, tail-block, and composition tests for split/fuse/interchange |
| frontend and backend defects confused | fix inputs; classify by tracing/import/lowering/execution | current frontend results and core transformation cases |
| shared facilities hide cost | count construction, driver, and helpers separately and give totals | complete cost ledger and residual structure |
| buffer-to-loop cost comparison exceeds the two-week budget | fix one set of multi-dimensional maps and one reduction from the existing regression cases, with the same allocation and schedule | existing `.mim` implementation versus the C++ counterpart; cost and residual structure of the fixed cases |

Reduction always gives priority to RQ1 and RQ2. During the research, the boundary cases remain discussion only, so that no new implementation targets crowd out the core verification.

## 9. Conclusion

This thesis studies the conditions under which tensor compiler transformations can be constructed from higher-order functions and partial evaluation, organized by the two kinds of work, lowering and loop transformation. The author's Torch→MimIR frontend carries the highest layer of lowering, covering model import, semantic adaptation, compilation and execution, and evaluation; it is the main completed result and has been used for the evaluation of the multi-level compilation study [13]. On the scaled cases of KernelBench Level 1 and Level 2 it passes 100/100 and 99/100 reference comparisons, and the only failure lies in the tensor-to-memory lowering rather than in import.

The existing prototypes give four preliminary findings. First, decomposition, im2col, and Winograd can be expressed with one library organization, with their preconditions and numeric contracts entering the function signature through dedicated parameters and checks. Second, Winograd beats direct convolution only on large-channel, batch=4 configurations, and im2col is slower than direct in most single-threaded configurations. The former matches the known applicable range of the two algorithms; the latter additionally reflects the gap that the matrix product unfolded from im2col is not yet scheduled. The benefit of algorithm substitution depends on algorithm selection and on the schedule that follows. Third, the prototype's `interchange` accepts the illegal permutation `(0,0,2)`, which shows that type checking covers the rank relations, but contracts such as permutation validity, coverage, and reduction order must still be added explicitly. Fourth, the prototype's `dot_trace` schedule already brings single-threaded GEMM to a geometric-mean ρ of 1.08 against OpenBLAS (median 0.98), where upstream MimIR's default schedule reaches 0.38; RQ2 turns this implicit schedule into the explicit `gemm_schedule` interface and extends it to GEMV and CONV.

The further research follows the lowering and loop transformation of Section 1.1: the former shows, through decomposition, im2col/Winograd, and the buffer-to-loop implementation, how to compose lower-level operations and existing functions to express the source computation; the latter expresses expert schedules for GEMM/GEMV/CONV by hand through the schedule API, aiming at performance close to optimized references, and establishes composition contracts for split/fuse/interchange. The buffer-to-loop implementation also provides controlled evidence for construction cost. Shared producers and buffer reuse explain the boundary of global analysis.

After reading this thesis, developers of compiler passes, operator libraries, or schedule libraries should be able to judge which transformations can be written as ordinary MimIR functions and handed to partial evaluation, which precondition checks and external analyses they still have to carry after writing them as functions, and what generated-code quality schedules constructed this way can reach. Auto-tuning, multi-threaded scheduling, end-to-end symbolic shapes, and general buffer reuse are follow-up directions built on top of this.

## References

1. Neil D. Jones, Carsten K. Gomard, Peter Sestoft. *Partial Evaluation and Automatic Program Generation*. Prentice Hall, 1993.
2. Jacques Carette, Oleg Kiselyov, Chung-chieh Shan. "Finally tagless, partially evaluated: Tagless staged interpreters for simpler typed languages." *Journal of Functional Programming*, 2009.
3. Tiark Rompf, Martin Odersky. "Lightweight Modular Staging: A Pragmatic Approach to Runtime Code Generation and Compiled DSLs." *GPCE*, 2010.
4. MLIR. [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/).
5. MLIR. [PDL dialect](https://mlir.llvm.org/docs/Dialects/PDLOps/) and [PDLL](https://mlir.llvm.org/docs/PDLL/).
6. MLIR. [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/).
7. MLIR. [Bufferization](https://mlir.llvm.org/docs/Bufferization/) and [Ownership-based Buffer Deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/).
8. Apache TVM. [Structured loop schedule implementation](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/src/s_tir/schedule/primitive/loop_transformation.cc). This version uses the `s_tir` naming.
9. Apache TVM. [Arithmetic analyzer interface](https://github.com/apache/tvm/blob/main/include/tvm/arith/analyzer.h).
10. Roland Leißa, Klaas Boesche, Sebastian Hack, Arsène Pérard-Gayot, Richard Membarth, Philipp Slusallek, André Müller, Bertil Schmidt. "AnyDSL: A Partial Evaluation Framework for Programming High-Performance Libraries." *PACMPL*, 2, OOPSLA, Article 119, 2018. Project page: [anydsl.github.io](https://anydsl.github.io/).
11. OpenXLA. [StableHLO specification](https://openxla.org/stablehlo/spec). This thesis uses it to separate the requirements of an external specification from the capabilities of a generic tensor interface.
12. Roland Leißa, Marcel Ullrich, Joachim Meyer, Sebastian Hack. *MimIR: An Extensible and Type-Safe Intermediate Representation for the DSL Age*. *PACMPL*, 9, POPL, Article 4, 2025, 31 pages. [DOI: 10.1145/3704840](https://doi.org/10.1145/3704840).
13. *Principles for Multi-Level Compiler Design*. Anonymous submission, 2026; author information omitted as required by the submission. This thesis cites its §3 and §7–8.
14. Bastian Hagedorn et al. "Achieving high-performance the functional way: a functional pearl on expressing high-performance optimizations as rewrite strategies." *PACMPL*, 4, ICFP, Article 92, 2020.
15. [MimIR Torch frontend](https://github.com/hsqStephenZhang/mimir-frontend/tree/619a15776d91a645fba997c059a3001e8e778dea). Commit `619a15776d91a645fba997c059a3001e8e778dea`; historical reports in `docs/kernelbench_*_report.md`.
16. [MimIR Torch plugin](https://github.com/hsqStephenZhang/mimir-torch-plugin/tree/6c41f8eb2313cbb923c2cb92a0a5a6b4d4f9a23d). Commit `6c41f8eb2313cbb923c2cb92a0a5a6b4d4f9a23d`; `torch.mim` and `src/phase/lower.cpp`.
17. [MimIR convolution prototype repository](https://github.com/hsqStephenZhang/MimIR). Baseline commit `09bfc92ae80bef5a2acdbcb449bbd909a84afdbf` provided by the Git bundle of Appendix A, with `conv-winograd.patch` applied on top.
18. Apache TVM MetaSchedule, commit `def37e30bd4785114ce3ffa6e059da1c36bf2656`. [Rule-generated design space](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/python/tvm/s_tir/meta_schedule/space_generator/post_order_apply.py), [custom schedule functions](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/python/tvm/s_tir/meta_schedule/space_generator/schedule_fn.py), [evolutionary search](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/src/s_tir/meta_schedule/search_strategy/evolutionary_search.cc), and [tile decision mutation](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/python/tvm/s_tir/meta_schedule/mutator/mutate_tile_size.py).
19. [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/tree/06f28d0942ee8eefce9aad7989d98743d2999932), source reference commit `06f28d0942ee8eefce9aad7989d98743d2999932`; `driver/level3/gemm.c`, `driver/level2/gemv_thread.c`, and the target-specific `kernel/`. The actual measurements additionally fix the build and configuration used.
20. [MimIR buffer-operation-to-loop lowering implementation](https://github.com/hsqStephenZhang/MimIR/tree/640537a4cbd5141d9a69352a659e46ad77cf5040). Commit `640537a4cbd5141d9a69352a659e46ad77cf5040`, branch `feature/btensor-map-reduce-hof`; `src/mim/plug/btensor/btensor.mim`, `src/mim/plug/btensor/phase/lower_map_reduce.cpp`, and `lit/btensor/map_reduce_post_library.mim`.
21. [Torch-MLIR](https://github.com/llvm/torch-mlir). The decomposition patterns in `lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp` and the conversions in `lib/Conversion/TorchToLinalg/`, all in C++.
22. [Apache TVM Relax legalization](https://github.com/apache/tvm/tree/main/python/tvm/relax/transform/legalize_ops). `python/tvm/relax/transform/legalize_ops/` maps Relax operators to TE computations in Python.
23. Andrew Lavin, Scott Gray. "Fast Algorithms for Convolutional Neural Networks." *CVPR*, 2016.
24. Kumar Chellapilla, Sidd Puri, Patrice Simard. "High Performance Convolutional Neural Networks for Document Processing." *ICFHR*, 2006.
25. Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini. "KernelBench: Can LLMs Write Efficient GPU Kernels?" arXiv:2502.10517, 2025.
26. Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman Amarasinghe. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines." *PLDI*, 2013.
27. James K. Reed, Zachary DeVito, Horace He, Ansley Ussery, Jason Ansel. "torch.fx: Practical Program Capture and Transformation for Deep Learning in Python." *MLSys*, 2022.
28. Jason Ansel et al. "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation." *ASPLOS*, 2024.

## Appendix A: Code, Data, and Reproducibility Status

Local materials directory (not tracked in Git): `artifacts/pe-hof-2026-10-15`. See the [reproducibility notes](artifacts/pe-hof-2026-10-15/README.md). It contains the code patches, per-file SHA-256 sums, the raw frontend results, and the complete convolution data. The following snapshots serve the frontend evaluation, the source walkthrough, the convolution experiments, and the buffer-to-loop lowering, respectively.

| Material | Repository and version | Reproducibility status |
|---|---|---|
| Torch frontend | [15], `619a157`; MimIR submodule [`3a31afc`](https://github.com/hsqStephenZhang/MimIR/tree/3a31afcdef1e00db51ea0de175de4c4b2228443e) | raw results and run conditions archived (`frontend/manifest.json`) |
| Torch operator excerpts | [16], `6c41f8e` | the add and mean_dim definitions can be located verbatim |
| Convolution algorithm prototype | [17], `09bfc92` + local `conv-winograd.patch` | Git bundle restores the baseline, patch check passes; includes Winograd, lowering fixes, tests, and the measurement driver |
| Buffer-to-loop lowering | [20], `640537a4cb`; compared against master `0a89de908d` | committed on a separate branch; includes the `.mim` implementation, C++ adaptation, and passing regression tests |
| GEMM measurements (Section 6.1, Appendix C) | harness A: `gemm-bench` `1682059` on upstream MimIR `2d21e0d`; harness B: [17] `c56d586` plus working-tree edits | `artifacts/gemm-2026-10-16/`: summary CSV, run log, harness sources, diff stat, and provenance JSON for each harness |

**Frontend dependency gap.** The frontend's `operators.py` requests `%torch.scan.diff` and `%torch.scan.diff_i64`, which the Torch plugin snapshot above lacks; six frontend tests are known to be affected. The three source versions therefore do not yet form a fully compatible frontend test environment. These two interfaces are only used by the adaptation of `torch.diff`; the KernelBench Level 1/2 cases contain no `torch.diff`, and the run logs show no request for them, so the results of 2026-10-15 are unaffected by this gap. The KernelBench results of 2026-10-15 are kept in `frontend/level1.json`, `frontend/level2.json`, and the corresponding logs. The run conditions were `size-divisor=16`, a 300-second timeout, a 12 GB address-space limit, a budget of 512 fixed-point iterations, and `--allow-invalid`; BLAS single-threaded (`OMP_NUM_THREADS=1`), the two test sets run concurrently, each with its own empty JIT cache. Software versions: MimIR submodule `3a31afc` (Release, Homebrew clang 22.1.4), Torch plugin `6c41f8e`, runtime plugin `a9958ea`, KernelBench `423217d`, PyTorch 2.12.0, Python 3.14.8; fixtures from Lighthouse `d330e762`, identical to the copy the repository vendored afterwards; the machine is the same as for the convolution measurements below. `frontend/manifest.json` records the complete commands, environment variables, and dependency lock (`uv.lock` as of `619a157`). Closing the `scan.diff` gap is an acceptance item of the environment-fixing phase of the thesis.

**Implementation size.** The frontend's `src/mimir_frontend/*.py` comprises 10 files and 9,001 physical lines, including blank lines and comments; tests, scripts, and submodules are counted separately. The main modules are `translator.py`, `operators.py`, `shape_rules.py`, `utils.py`, `backend.py`, `decomposition.py`, `audit.py`, and `model_export.py`. The Torch plugin's `torch.mim` has 7,386 physical lines, counted in its own repository. These numbers only indicate the scope of the implementation; implementation size, semantic coverage, and execution results are reported separately.

**Convolution sources.** Im2col already exists in the local baseline `09bfc92`; `mimir-conv-base.bundle` in the materials provides that commit, whose parent is `c56d58681549d478afd893ed5b8915d2bfbda979` on the remote feature/par branch. Winograd comes from the patch. Both are independent of the frontend submodule, and the experiments run directly from the MimIR compiler entry. The `lower_via_impl(app, impl<…>())` of Sections 3.2–3.3 uses the API of this prototype: `Lower::impl<Id>()` temporarily sets `in_annex_` and then calls `annex<Id>()`; the older version in the frontend submodule [15] calls `annex<Id>()` directly. The SHA-256 of the patch is `c7c584ce2ebd4f5cc441f64a9656fef42a044f9d65b417e7765fc963dd3fa12c`. To apply it:

```sh
git clone https://github.com/hsqStephenZhang/MimIR.git mimir-conv
cd mimir-conv
# BASE_BUNDLE points to mimir-conv-base.bundle in the materials directory
git bundle verify "$BASE_BUNDLE"
git fetch "$BASE_BUNDLE" HEAD
git checkout --detach 09bfc92ae80bef5a2acdbcb449bbd909a84afdbf
# PATCH points to conv-winograd.patch in the materials directory
git apply --check "$PATCH"
git apply "$PATCH"
```

Afterwards, install dependencies, initialize submodules, and build the compiler following the baseline build instructions. The materials README describes how to configure the measurement scripts and the compiler path. The SHA-256 identifies the saved patch; the patch check and the full rebuild are recorded separately.

**Buffer-to-loop lowering sources.** [20] is based on master `0a89de908d` and is independent of the convolution prototype and the frontend submodule; check out its fixed commit, initialize submodules recursively, then build and run `test-all` following the repository instructions. `lit/btensor/map_reduce_post_library.mim` covers heterogeneous inputs, different output element types, logical singleton axes, reductions with memory side effects, dynamic loop bounds, and zero iterations. Local verification of this snapshot: C++ tests 50/50, lit 542 passed and 7 skipped, Mim and AST round-trip checks passed. The construction cost experiment of RQ1 is recorded separately according to Sections 6.3–6.4.

**Convolution data.** `conv/results/` holds the CSVs of 12 independent runs. `conv/table.md` holds the summary of all 24 configuration rows (6 geometric configurations × batch 1/4 × compile-and-run thread setting 1/4); Appendix B lists only the 12 single-threaded rows used in this thesis. `conv/build-b*-t*.json` records the hashes of the compiler, plugins, and LLVM IR of the original measurement. The platform is an Apple M2 Pro (8P+4E), 32 GiB, Release, LLVM 22.1.4; FP32 fast, clang++ `-O3 -mcpu=native`. Every call includes padding, transforms, matrix product, packing, allocation, and output deallocation; there is no persistent weight cache.

Each program first compares all outputs against a double-precision scalar reference, then calibrates the timing batch to about 5 ms, alternates the order of the three algorithms in each run, and takes the median of 7 rounds; the median of 3 independent runs is taken again. Single-round samples were not saved, and the tables report the existing medians; threads were not pinned, and heterogeneous cores and system load affect the measurements. Direct is the `tensor.conv` of the same convolution prototype.

## Appendix B: Preliminary Single-Threaded Convolution Measurements

6 geometric configurations × batch 1/4, 12 rows in total, all with compile and run thread settings of 1; each row compares the three algorithms. Conditions are in Appendix A, raw CSVs and the summary script in the materials directory; the other 12 rows with the 4-thread setting are kept in `conv/table.md` of the materials and are not used in this thesis. Both ratio columns have direct as numerator; a value above 1 means that algorithm is faster than direct.

| Shape (N,Cin,Cout,H,W) | Direct ms | Im2col ms | Winograd ms | Direct / Im2col | Direct / Winograd |
|---|---:|---:|---:|---:|---:|
| 1,16,16,32,32 | 0.1081 | 0.2702 | 0.4254 | 0.40× | 0.25× |
| 1,32,64,28,28 | 0.6422 | 1.4263 | 0.8321 | 0.45× | 0.77× |
| 1,128,128,28,28 | 5.1725 | 8.9977 | 5.4117 | 0.57× | 0.96× |
| 1,512,512,7,7 | 19.9282 | 10.2071 | 21.1382 | 1.95× | 0.94× |
| 1,3,64,32,32 | 0.1161 | 0.1915 | 0.6232 | 0.61× | 0.19× |
| 1,13,17,17,19 | 0.0389 | 0.1344 | 0.0990 | 0.29× | 0.39× |
| 4,16,16,32,32 | 0.5013 | 1.2388 | 1.4538 | 0.40× | 0.34× |
| 4,32,64,28,28 | 3.1186 | 5.7788 | 7.5481 | 0.54× | 0.41× |
| 4,128,128,28,28 | 25.1853 | 38.8540 | 18.8297 | 0.65× | 1.34× |
| 4,512,512,7,7 | 54.2909 | 34.8413 | 22.2639 | 1.56× | 2.44× |
| 4,3,64,32,32 | 0.4328 | 0.8131 | 2.6215 | 0.53× | 0.17× |
| 4,13,17,17,19 | 0.1520 | 0.5974 | 0.3559 | 0.25× | 0.43× |

Im2col is faster than direct only in the two `Cin=Cout=512` rows, and Winograd only in the two rows with batch=4 and `Cin=Cout≥128`. The losses in the remaining configurations match the algorithmic applicability range described in Section 3.3.

## Appendix C: Preliminary Single-Threaded GEMM Measurements

Conditions are in Section 6.1 and in `artifacts/gemm-2026-10-16/*/provenance.json`. ρ is OpenBLAS time over MimIR time; values above 1 mean MimIR is faster.

**Harness A: `gemm-bench`, 2026-10-12, upstream MimIR master `2d21e0d`.** Median of nine interleaved batches. TVM 0.27 with its default LLVM schedule rules and IREE with data tiling are listed for context only.

| Shape `M×K×N` | PyTorch ms | OpenBLAS ms | MimIR ms | TVM ms | IREE ms | MimIR GFLOP/s | OpenBLAS GFLOP/s | ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `64×64×64` | 0.0012 | 0.0196 | 0.0136 | 0.0433 | 0.0122 | 38.6 | 26.7 | 1.44 |
| `128×128×128` | 0.0052 | 0.0781 | 0.0918 | 0.4587 | 0.0582 | 45.7 | 53.7 | 0.85 |
| `256×256×256` | 0.0309 | 0.4551 | 1.0638 | 1.4027 | 0.4253 | 31.5 | 73.7 | 0.43 |
| `512×512×512` | 0.1925 | 2.9077 | 8.3637 | 13.3145 | 3.2014 | 32.1 | 92.3 | 0.35 |
| `1000×1000×1000` | 1.4266 | 23.4084 | 64.0121 | 66.2217 | 20.5541 | 31.2 | 85.4 | 0.37 |
| `1024×1024×1024` | 1.6582 | 23.4216 | 78.8163 | 36.9359 | 23.2502 | 27.2 | 91.7 | 0.30 |
| `2000×2000×2000` | 13.9962 | 181.4930 | 706.7687 | 323.4869 | 172.9204 | 22.6 | 88.2 | 0.26 |
| `2048×2048×2048` | 15.5139 | 191.8570 | 657.8740 | 878.6175 | 191.4267 | 26.1 | 89.5 | 0.29 |
| `8×2304×512` | 0.0331 | 0.3390 | 0.5973 | 0.8101 | 1.1023 | 31.6 | 55.7 | 0.57 |
| `64×130×512` | 0.0090 | 0.1013 | 0.3357 | 0.1724 | 0.1022 | 25.4 | 84.1 | 0.30 |
| `64×49×1024` | 0.0099 | 0.0855 | 0.2595 | 0.1736 | 0.0860 | 24.8 | 75.1 | 0.33 |
| `64×147×1024` | 0.0205 | 0.2387 | 0.7444 | 0.4826 | 0.2397 | 25.9 | 80.7 | 0.32 |
| `64×245×1024` | 0.0328 | 0.3697 | 1.2235 | 0.8116 | 0.3669 | 26.2 | 86.9 | 0.30 |
| `64×2048×2048` | 1.0263 | 6.1891 | 23.1650 | 16.4082 | 11.2886 | 23.2 | 86.7 | 0.27 |
| `2048×512×128` | 0.2328 | 2.7450 | 7.7974 | 7.8658 | 3.6467 | 34.4 | 97.8 | 0.35 |
| `2049×2047×513` | 3.9778 | 47.8500 | 163.6658 | 899.3922 | 50.1097 | 26.3 | 89.9 | 0.29 |

Geometric mean of ρ over the 16 shapes: 0.38.

**Harness B: prototype `.test/gemm_bench`, 2026-10-16, branch `feature/par` at `c56d586` plus working-tree edits.** Best of N calls per shape; the last three rows come from `shapes2.txt`.

| Shape `M×K×N` | OpenBLAS ms | MimIR ms | OpenBLAS GFLOP/s | MimIR GFLOP/s | ρ |
|---|---:|---:|---:|---:|---:|
| `64×64×64` | 0.020 | 0.005 | 26.2 | 101.5 | 3.88 |
| `128×128×128` | 0.065 | 0.045 | 64.8 | 93.1 | 1.44 |
| `256×256×256` | 0.417 | 0.356 | 80.4 | 94.1 | 1.17 |
| `384×384×384` | 1.214 | 1.196 | 93.3 | 94.7 | 1.01 |
| `512×512×512` | 2.671 | 2.837 | 100.5 | 94.6 | 0.94 |
| `1000×1000×1000` | 20.273 | 24.003 | 98.7 | 83.3 | 0.84 |
| `1024×1024×1024` | 22.098 | 24.190 | 97.2 | 88.8 | 0.91 |
| `2048×2048×2048` | 174.832 | 199.827 | 98.3 | 86.0 | 0.88 |
| `8×2304×512` | 0.307 | 0.224 | 61.4 | 84.3 | 1.37 |
| `64×130×512` | 0.089 | 0.091 | 95.7 | 94.0 | 0.98 |
| `20×100×512` | 0.025 | 0.023 | 82.3 | 87.8 | 1.07 |
| `1024×1024×64` | 1.399 | 1.396 | 95.9 | 96.1 | 1.00 |
| `64×1024×1024` | 1.382 | 1.480 | 97.2 | 90.7 | 0.93 |
| `1024×64×1024` | 1.329 | 1.927 | 101.0 | 69.7 | 0.69 |
| `256×256×240` | 0.351 | 0.331 | 89.6 | 95.1 | 1.06 |
| `512×512×496` | 2.598 | 2.710 | 100.1 | 95.9 | 0.96 |
| `1024×1024×1008` | 21.982 | 23.670 | 96.2 | 89.3 | 0.93 |

Geometric mean of ρ over the 17 shapes: 1.08; median 0.98; 16 of 17 at or above 0.8.
