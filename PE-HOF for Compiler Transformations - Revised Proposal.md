# Partial Evaluation and Higher-Order Functions for Compiler Transformations

## Revised Master's Thesis Proposal

**Working subtitle:** Composable lowering and tensor scheduling in MimIR, with explicit boundaries for graph analysis and memory planning.

**Revision date:** 15 October 2026.  
**Basis:** “MimIR as a powerful DSL Design Doc,” proposal repository revision `3220668`, and the current MimIR tensor implementation.  
**Status:** Research proposal. Existing prototypes, preliminary measurements, and proposed extensions are distinguished below. All unlabelled algorithm sketches are explanatory pseudocode, not claims about an available API.

## Abstract

Compiler transformations often combine two different activities: deciding which transformation is legal and profitable, and constructing the transformed program. Modern frameworks provide substantial infrastructure for both, but the construction of a replacement computation frequently requires compiler-specific manipulation of operations, regions, types, and SSA values. An alternative is to describe that computation as a typed, parameterized program whose higher-order arguments encode scalar operations, index maps, loop bodies, and representation choices. Partial evaluation can specialize this program and remove its abstraction overhead.

This thesis investigates the scope and engineering requirements of that alternative in MimIR. It asks which transformations naturally admit such descriptions, how higher-order definitions can support both same-representation decomposition and representation-changing lowering, and how tensor loop transformations can be expressed as compositions of domain and access-map transformations. It also studies where graph-wide information remains necessary, using fusion and bufferization as boundary cases.

The central hypothesis is that higher-order functions (HOFs) and partial evaluation (PE) can provide a reusable language for constructing substantial classes of transformations, while a separate driver supplies graph traversal, representation conversion, legality checks, and selected analysis results. This is a hypothesis about a useful division of responsibilities, not a claim that function inlining replaces compiler analysis. The evaluation will measure semantic correctness, expressiveness, implementation effort, compiler cost, and generated-code quality separately. Existing tensor schedules, im2col, and Winograd provide starting points; selected fusion and bufferization examples will test the limits of the approach.

## 1. Motivation and Research Gap

A compiler lowering might replace one tensor operation with a small graph of simpler operations, or replace it with loops, scalar arithmetic, and memory accesses. Much of the replacement has a regular computational structure: it is parameterized by element types, tensor dimensions, scalar operations, indexing functions, or loop bodies. This suggests describing the replacement directly as a program rather than reconstructing every node through a compiler API.

For example, a convolution-to-im2col decomposition is a composition of padding, window extraction, reshaping, and matrix multiplication. A tiled loop nest can be described by a new iteration domain and a map back to the original indices. A fused elementwise consumer can be expressed by composing scalar functions. HOFs provide a way to abstract over these bodies and maps, while PE can specialize their static parameters.

This opportunity should be stated without overstating limitations of existing frameworks. MLIR already supports declarative operation definitions, interfaces, rewrite patterns, PDL/PDLL, a Transform dialect, and dialect conversion with type converters and legality targets [4–7]. TVM provides a structured tensor scheduling interface, explicit schedule errors, and arithmetic analysis [8–9]. PyTorch's decomposition infrastructure also uses ordinary programming-language functions. The research question is therefore not whether other frameworks can produce the same IR, nor whether C++ is intrinsically incapable of expressing transformations safely.

The question is whether a typed executable representation, combined with specialization, yields a more composable and economical way to describe selected transformations **without merely moving equivalent boilerplate into a hidden driver or paying excessive specialization costs**. The proposal will compare responsibilities and measurements, rather than infer superiority from shorter examples or from the implementation language.

MimIR is a suitable setting because it represents functions and types within a common IR, supports dependent relationships between parameters and result types, and already uses PE to specialize higher-order computations. However, well-typedness alone does not establish bounds safety, dependence preservation, floating-point equivalence, or safe buffer reuse. These obligations remain part of the research problem.

## 2. Central Hypothesis and Terminology

### 2.1 The hypothesis

For a transformation with an explicit, bounded description of its required context, the transformed computation can often be constructed by specializing a reusable higher-order definition. A compiler driver can supply that context without containing the full replacement algorithm.

Conceptually:

```text
inspect/analyze source graph
    → select a legal transformation and its parameters
    → instantiate a typed higher-order definition
    → specialize and simplify
    → validate the residual target program
```

“Bounded description” does not mean that the context is always small or that its analysis is local. It means that the construction interface explicitly states what information it consumes. A template that implicitly requires arbitrary access to use-def chains has not eliminated graph analysis; it has hidden it.

### 2.2 Three distinct uses of HOFs

The thesis distinguishes three mechanisms that are easily conflated:

1. **Parameterized implementation:** an operation is implemented by a function receiving scalar operations, types, shapes, or component implementations.
2. **Computation transformation:** a function receives a structured computation and returns another one, for example by composing its access maps with an index transformation.
3. **Compiler orchestration:** a driver visits IR nodes, manages mappings and analyses, chooses implementations, and applies replacements.

The first two are the main subjects of the thesis. The third may also be written in a functional language, but doing so is a different claim from eliminating its graph-processing responsibilities through PE.

Not every parameterized function is higher-order. A decomposition receiving only shapes is a staged implementation; it becomes higher-order when it abstracts over functions such as scalar combiners, indexing maps, loop bodies, or reconstruction continuations. The evaluation will report which abstraction each case actually uses.

### 2.3 What PE contributes

PE evaluates the parts of a program determined by static information and leaves a residual program for dynamic inputs. Beta reduction and inlining are important mechanisms, but they do not themselves imply dependence analysis, algebraic theorem proving, or profitable scheduling.

For this thesis, element types, ranks, selected schedules, and many indexing functions are typically static. Tensor values and some extents remain dynamic. A symbolic extent may participate in a statically constructed expression without having a known numeric value. Static rank and dynamic dimension size are therefore separate concerns.

A successful specialization should eliminate avoidable dictionaries, closures, and dispatch when their choices are static. Dynamic choices must either remain explicit, select a documented fallback, or be rejected by a stage that requires static information. They must not trigger uncontrolled unrolling or enumeration of tensor elements during compilation.

### 2.4 What “efficient” means

Two independent requirements will be evaluated:

- **Compiler efficiency:** specialization time, temporary and residual IR size, repeated traversal, cache behavior, and peak compiler memory.
- **Program efficiency:** allocations, copies, loop structure, vectorization, parallel work distribution, memory traffic, and end-to-end execution time.

A 100-line implementation can still compile slowly or generate a slow kernel. Conversely, an effective kernel does not demonstrate that its transformation interface is general or safe.

## 3. Research Questions

**RQ1 — Suitability.** Which rewrite and transformation classes are naturally expressed through HOFs and PE, and which properties determine the fit?

**RQ2 — Lowering.** How can an HOF-based construction layer support efficient dialect conversion, including type- and representation-changing cases, without duplicating a conventional conversion engine inside every definition?

**RQ3 — Loop transformation.** How can tensor loop transformations be composed through HOFs while making their legality, tail handling, and generated loop structure explicit?

**RQ4 — Global context.** How far can the same construction approach be used for fusion and bufferization, and which graph analyses and planning decisions must remain explicit?

These questions share one objective: identify a useful interface between **transformation construction** and **transformation selection and validation**.

## 4. RQ1: A Suitability Model for Higher-Order Transformations

The initial classification uses four dimensions:

- **Required context:** parameters of one operation, a structured producer-consumer fragment, or information about the enclosing graph/function.
- **Representation change:** identical source/result representation, a local adaptation, or a coordinated change to values and function boundaries.
- **Semantic obligations:** pure value equivalence, reduction ordering, bounds, effects, aliasing, or synchronization.
- **Specialization behavior:** whether static structure can be resolved with bounded code growth and whether dynamic data remains residual.

These dimensions matter more than an absolute distinction between “local” and “global.” A global plan may be executed by compact HOFs, while a local rewrite may have difficult numerical or effect semantics.

| Transformation family | HOF/PE construction opportunity | Information not supplied by PE alone |
|---|---|---|
| Operator decomposition, such as im2col convolution | Compose lower-level computations and parameterized scalar operations | Shape/type preconditions, algorithm selection, target support |
| Scalar implementation or dtype specialization | Pass explicit operation dictionaries and specialize them | Dtype promotion, valid instances, numerical policy |
| Structured loop generation | Abstract over bounds, bodies, access maps, and accumulators | Dependence and effect assumptions |
| Split, fuse, and interchange of structured domains | Compose coordinate maps and reconstruction closures | Domain legality, reduction order, schedule compatibility |
| Vectorized or parallel variants | Select specialized loop bodies and execution primitives | Alignment, tail semantics, race freedom, profitability |
| Pure elementwise producer-consumer fusion | Compose scalar functions and access maps | Sharing, recomputation, effects, and cost |
| Reduction epilogue fusion | Compose a post-reduction function | Completion of the reduction, additional operand maps, sharing |
| Tensor-to-buffer construction | Instantiate destination-passing or allocating loop bodies | Alias/liveness decisions, representation migration, lifetime |
| Global reuse, placement, or fusion selection | Instantiate a selected plan | Use-def analysis, ownership, conflicting uses, optimization policy |
| Arbitrary CFG or recursive graph rewriting | HOFs may construct replacement fragments | Binder/region remapping, recursion, traversal, global invariants |

The initial hypothesis is strongest when the replacement is structurally regular, its context is explicit, and its static parameters control structure rather than large data. It is weaker when the required abstraction is essentially arbitrary graph inspection or when specialization duplicates large shared computations.

The thesis will not claim that the harder classes are mathematically inexpressible using HOFs. Instead, it will ask whether doing so retains the intended benefits of clarity, reuse, and efficient specialization.

## 5. RQ2: Efficient Dialect Conversion and Lowering

### 5.1 Same-representation decomposition

The simplest case preserves the value-level type of the source operation. An opaque operation can remain available for analysis until an explicit lowering phase selects its implementation. The driver extracts its arguments and applies a parameterized implementation; PE constructs lower-level operations.

MimIR's existing `Lower::lower_via_impl` follows this structure: it gathers the curried arguments, rewrites them, and applies an implementation annex. Tensor implementations can thereby reuse scalar bodies, access maps, and other implementations. This is an existing mechanism, not a proposed replacement for all conversion infrastructure.

Keeping the operation opaque until the relevant phase is important. Unconditional early unfolding could destroy recognizable matrix products, lose fusion opportunities, or commit to a representation before target information is available.

### 5.2 Representation-changing conversion

Changing `Tensor<T,s>` into a buffer handle or descriptor is a different problem. Inlining a function that still expects a tensor does not, by itself, produce a correctly typed bufferized caller.

Let `R(T)` denote the target representation of a source type `T`. A representation-aware implementation has the conceptual signature:

```text
lower_op_R : R(T1) × … × R(Tn) → R(U)
```

This notation is schematic. A real conversion may map one source value to multiple target values, include runtime metadata, and thread effects. `R` is not assumed to be a simple textual substitution of type names.

The proposed architecture separates the following responsibilities:

| Conversion driver | Higher-order implementation layer |
|---|---|
| Source-to-target value mapping and graph traversal | Computational structure of the replacement |
| Type conversion and representation descriptors | Scalar algorithms, loop bodies, and access maps |
| Function, continuation, call, and return adaptation | Construction under an explicitly supplied representation |
| Recursive stubs, binder scopes, and sharing | Reusable implementations for families of operations |
| Materialization at permitted boundaries | Explicit reads, writes, allocation requests, or destination use |
| Legality checking and conversion failure | Preconditions and residual target computation |

Two construction styles will be compared on a limited subset:

- **Explicit target implementations:** a tensor operation has a buffer-oriented implementation receiving converted inputs and a destination or allocation service.
- **Representation-parameterized definitions:** a semantic implementation receives an interface providing operations such as indexed read, destination construction, loop execution, and result finalization. Specialization selects a concrete interpretation.

The second style is related to typed interpreter and staging techniques [2–3]. It is a research option, not an assertion that the current tensor plugin already implements a general representation-polymorphic interpreter. The experiment should establish whether its additional abstraction reduces duplication enough to justify interface complexity and specialization cost.

### 5.3 Effects and boundaries

Effects must be represented explicitly where they become observable. An effectful construction may conceptually have the form:

```text
lower_update_R : Memory × R(Input) × Destination → Memory × R(Result)
```

The memory token orders operations; it is not an alias proof, an ownership certificate, or a guarantee of allocation freshness. The driver and allocation interface must preserve distinct allocation identities even if their shape and initializer parameters are identical.

The same discipline applies to control-flow joins, recursive functions, and public ABIs. The core experiment will support a clearly specified intraprocedural subset and selected call-boundary adaptations. General interprocedural bufferization and arbitrary ABI conversion are outside the required scope.

### 5.4 Avoiding excessive compilation cost

The implementation will investigate:

1. **Phase-controlled specialization:** expand only the chosen abstraction level and retain opaque primitives where they remain useful.
2. **Memoized graph reconstruction:** preserve shared source values and create recursive stubs before descending into cycles.
3. **Context-correct reuse:** key reusable artifacts by relevant representation, target, policy, and static parameters. Do not reuse a conversion result across semantically different contexts.
4. **Bounded specialization:** unroll small static control structures, while leaving large tensor domains as loops.
5. **Selective reconstruction:** avoid rebuilding unchanged graph fragments and repeatedly specializing equivalent implementation instances.
6. **Residual validation:** check the allowed operations and boundary types after conversion rather than treating successful beta reduction as successful legalization.

Allocation-producing computations require special care: reusing an implementation definition is different from merging two runtime allocation instances. Likewise, caches used during library/annex rewriting must not cause later program rewriting to reuse an unlowered result.

### 5.5 Correctness objective

For pure decomposition, the objective is observational equivalence under the operation's stated preconditions. For representation-changing lowering, the objective is preservation under an explicit source/target representation relation, together with valid effect ordering and memory access.

Dependent types can encode relationships such as matching input/output element types or shape preservation. They do not establish arbitrary algebraic identities or target layout compatibility. Numerical algorithms such as Winograd additionally require a documented floating-point contract; exact bitwise equality is not an appropriate default claim.

## 6. RQ3: Tensor Loop Transformations as Composition

### 6.1 Structured computation model

The starting point is a structured computation with:

- output shape `So`, independent of the number of iteration dimensions;
- iteration domain `D`, divided into parallel and reduction dimensions;
- operand access maps `Aj`;
- an output map `W`;
- a scalar body, initialization, reduction step, and optional epilogue;
- write-validity and reduction-validity predicates;
- explicit schedule information and a reconstruction function.

MimIR's current `Tileable` packages much of this information and retains operation-specific details inside its `retile` closure. This is a concrete HOF-based reconstruction mechanism. It should not be presented as a fully verified schedule calculus: the existing API has checks and structural restrictions, but complete legality guarantees are not established.

Separating output rank from iteration rank is essential. Fusing two loop dimensions should not automatically flatten the output tensor, and splitting one iteration dimension should not necessarily change the external result layout.

### 6.2 Reindexing and composition

For an index-preserving transformation, let `φ : D' → D` map new iteration coordinates to old ones. Reads become:

```text
Aj' = Aj ∘ φ
```

For a transformation preserving output layout, writes become `W' = W ∘ φ`. An explicit output-layout transformation additionally applies a map `ψ` to the old output coordinates; in the current API this may also depend on the new loop coordinates.

If a second transformation maps `D''` to `D'` through `χ`, the combined index map is `φ ∘ χ`. PE can simplify the resulting higher-order composition when the maps are known.

Map composition alone is insufficient for correctness. For a pure parallel computation, valid new iterations must cover each required original output computation without unintended omissions or conflicting writes. For a reduction, the transformation must also preserve the required contribution multiplicity, initialization, and ordering, or explicitly rely on an allowed reassociation policy.

### 6.3 Split, tails, and predicate placement

A split by positive tile size `b` uses:

```text
i = io × b + ii
0 ≤ io < ceil(N / b),  0 ≤ ii < b
valid(io, ii) = io × b + ii < N
```

An exact split can omit the predicate when divisibility is established. A partitioned split retains it or generates a peeled tail. For a reduction tail, the predicate must suppress the invalid contribution and its loads, not merely the final output store.

This distinction matters under strict evaluation: a value-level `select(valid, load(p), zero)` is not sufficient if the load has already been evaluated. The lowering must guard the access, use an operation with genuine masked-access semantics, or prove an alternative access safe. A neutral element also does not justify executing arbitrary invalid loads or replacing an accumulator by zero in an invalid lane.

The proposed contract therefore records write and fold validity separately, composes them through domain changes, and requires the generated nest to discharge them correctly. Uniform predicate semantics need not force all kernels to retain identical branches; a later pass can peel or simplify guards after establishing equivalence.

### 6.4 Fuse and interchange

For two adjacent independent dimensions with extents `A` and `B`, fusion can use:

```text
t ∈ [0, A×B)
i = t / B
j = t % B
```

This requires a valid rectangular domain or a separately represented validity predicate, safe extent arithmetic, and correct handling of empty domains without evaluating division by zero. Fusion must also update schedule metadata and preserve the logical output map.

Interchange permutes iteration coordinates and their associated metadata. For structured pure parallel dimensions, legality is often established by the representation. Reordering reductions or effectful accesses needs stronger assumptions. In particular, associativity alone does not justify arbitrary permutation of noncommutative operations, and floating-point reassociation requires an explicit numerical policy.

### 6.5 API contracts and comparison with TVM

TVM is a reference for making schedule preconditions and errors explicit, not a promise that every TVM primitive will be implemented. The thesis will examine this selected API set:

| API family | Required contract |
|---|---|
| Exact split | Valid axis, positive factor, exact domain coverage, divisibility when required |
| Partitioned split / tail peel | Ceil-divided bounds, composed guards, safe loads, correct accumulator behavior |
| Iteration fusion | Compatible adjacent dimensions, reversible indexing over valid points, updated metadata |
| Interchange | Valid permutation, preserved bounds and dependence/reduction ordering |
| Vectorization | Legal lane accesses, alignment/layout assumptions, masks or a safe tail path |
| Parallelization | Independent writes or an explicit reduction scheme, valid scratch ownership |
| `cache_read` / `cache_write` | Correct layout maps, destination and writeback semantics, valid storage scope |
| `compute_at` and staging | Producer coverage, availability, placement and lifetime, acceptable recomputation |

These API families do not all reduce to pure coordinate changes. Storage staging and placement are included specifically to identify the boundary between domain composition and analysis-dependent construction.

Preconditions will be classified as: encoded by types/representation; established by a static checker; left as an explicit runtime guard where appropriate; or unresolved and rejected/fallen back. Unknown is not equivalent to proven safe. Runtime guards can handle selected size conditions; they are not a universal substitute for dependence or alias analysis.

Existing `refly.check` calls should not be assumed to provide this complete contract. In the inspected implementation, a false literal check logs an error and leaves normalization unresolved; that behavior alone does not establish a reliable conversion-failure or runtime-assertion mechanism. Diagnostics and exit behavior must be tested separately.

### 6.6 Generating efficient loop nests

Static schedule descriptions should specialize to explicit loops and primitives rather than leave a runtime schedule interpreter. However, full eager expansion can cause code growth. The implementation will compare selected expansion boundaries and check whether higher-order maps and dictionaries disappear without unrolling entire tensor domains.

Schedule reports and generated LLVM IR will be inspected alongside timings. They can reveal scalarized accumulators, redundant transform work, missed vectorization, excessive scratch allocation, and ineffective parallel task grids that a high-level schedule description conceals.

## 7. RQ4: Fusion and Bufferization as Boundary Cases

### 7.1 Fusion: construction versus selection

For a pure pointwise chain,

```text
producer(i) = f(x(A(i)))
consumer(j) = g(producer(B(j)), y(C(j)))
```

substitution gives:

```text
fused(j) = g(f(x(A(B(j)))), y(C(j)))
```

HOF composition naturally describes the fused scalar body and the composed access maps. But it does not decide whether removing the producer materialization is safe or profitable.

Consider a diamond in which a producer feeds two consumers. Inlining into both may duplicate an expensive reduction. Keeping a materialized producer may be preferable; alternatively, a combined loop may serve both consumers. Choosing among these options requires use-def information, effect information, and an explicit policy. A single-consumer rule is a useful conservative heuristic, not a universal fusion criterion.

Reduction epilogue fusion is a second case: `relu(sum_k(...))` applies `relu` after the completed reduction. Applying it during each reduction step changes the computation. The template must expose the epilogue boundary, while the driver checks operand mappings, sharing, and legal placement.

The proposed experiment will retain a graph-level fusion selector and move selected fused-construction logic into reusable HOFs. It will cover a pure chain, a diamond/shared producer, and a reduction epilogue. Existing `Fuse` consumer counts and shared-producer tracking provide a concrete starting point. The thesis will measure whether this separation simplifies construction without obscuring the necessary analysis.

### 7.2 Bufferization: representation conversion versus reuse

A conservative bufferization allocates a distinct destination and copies values when necessary. It can preserve tensor value semantics under explicit assumptions about readable input views, fresh allocations, valid bounds, and lifetime. It is already a meaningful representation-changing lowering, but it does not solve allocation minimization.

In-place bufferization additionally asks whether an existing buffer can safely serve as a destination. For example:

```text
t0 = input tensor
t1 = update(t0, i, v)
r  = read(t0, j)
```

Writing `t1` into `t0`'s storage may change `r` when the old value remains observable. Passing a memory token through the update orders effects but does not prove that reusing the storage preserves tensor semantics.

A reusable construction interface can receive an explicit destination and a selected plan:

```text
analysis and policy → allocate/copy/reuse plan
plan + converted inputs + memory → specialized buffer computation
```

The core thesis will implement or extract the allocate-and-copy construction and one restricted reuse case, subject to a specified safety check. General alias analysis, allocation placement, interprocedural ownership, and optimal memory planning are excluded from the minimum deliverable. Lifetime/deallocation remains a separate obligation even after a reuse decision.

This separation is consistent with MLIR One-Shot Bufferization's distinction between analysis and rewriting [7]. The proposed contribution is to investigate HOF-based execution of selected plans and reuse of semantic definitions, not to claim that the analysis/rewrite split is new.

### 7.3 The expected boundary

Fusion and bufferization are not inherently unsuitable for HOFs. What HOFs and PE do not automatically supply is the missing global information. There are three possible outcomes to investigate:

- graph analysis plus HOF construction is simpler and efficient;
- a richer structured representation moves some obligations into explicit interfaces, reducing repeated analysis;
- the required interfaces become as complicated as direct graph rewriting, so the construction approach has limited value for that case.

The third outcome is useful evidence, rather than a result to hide.

## 8. Prototype Basis and Proposed Work

The source proposal contains several valuable experiments, including triangular masking, StableHLO-style sort signatures, explicit numeric/comparison dictionaries, symbolic shapes, and template-based decomposition. These will be retained as supporting examples where their implementations can be reproduced. They will not all become independent thesis deliverables.

The inspected tensor implementation provides the following more immediate basis:

| Area | Existing evidence | Work required for this thesis |
|---|---|---|
| Decomposition | `Lower::lower_via_impl` and tensor `_impl` definitions | Specify the contract, isolate driver costs, compare equivalent implementations |
| Loop composition | `Tileable`, `retile`, separate output/iteration ranks, split/fuse/interchange APIs | Audit selected legality conditions, document invariants, test composition and failure behavior |
| Convolution algorithms | im2col and a local Winograd F(2×2,3×3) implementation | Use as construction and performance case studies; avoid claiming universal speedup |
| Fusion | C++ `Fuse`, access-map composition, consumer counts and shared tracking | Extract selected construction into HOFs and preserve graph-aware selection |
| Bufferization | `LowerToMem`, buffer/btensor layers, boundary adaptation and memory threading | Make representation and allocation contracts explicit; compare restricted construction alternatives |
| Semantic interfaces | Numeric dictionaries, shape-dependent types, original sort/triu examples | Use a small subset to test composability and clarify which constraints remain external |

Current buffer reuse policy is conservative: `reuse_in_place` returns false. This is not an existing general liveness/alias-based reuse implementation. Similarly, the original draft's runtime-check and comparison-witness sketches should not be treated as a verified, complete refinement-type system or a complete implementation of StableHLO semantics.

### Preliminary Winograd observation

The current local Winograd prototype consists of approximately 135 physical lines for the algorithm, API declaration, parameter checks, and lowering registration, excluding tests, documentation, and supporting lowering fixes. This is an implementation-size observation, not an equivalently scoped LOC comparison with MLIR or TVM.

On the current Apple M2 Pro measurements, single-threaded FP32 convolution with batch 4, 128 input/output channels, and 28×28 spatial extent takes about 25.19 ms for direct convolution, 38.85 ms for im2col, and 18.83 ms for Winograd. With batch 4, 512 channels, and 7×7 extent, the corresponding times are 54.29, 34.84, and 22.26 ms. All use a 3×3 kernel, unit stride/dilation, and padding 1; measurements include transforms and allocation/free, with no persistent transformed-weight cache.

Winograd is slower than direct convolution in all measured batch-1 cases in this small suite, and four-thread scaling is limited. These preliminary results motivate separating description economy from generated-code quality. They are neither a general algorithm-selection policy nor evidence that PE outperforms another compiler. The final evaluation will pin source revisions, patches, compiler options, and the full measurement protocol.

## 9. Evaluation Plan

### 9.1 Case studies and controlled comparisons

The minimum evaluation comprises four groups:

1. **Construction and lowering:** a pointwise decomposition, a structured reduction, im2col convolution, and Winograd. Include one same-representation and one representation-changing path.
2. **Loop scheduling:** split, partition/tail handling, fuse, and interchange, plus vectorization/parallelization as execution cases. Test nontrivial compositions, not only isolated primitives.
3. **Fusion:** a pure chain, a diamond/shared producer, and a reduction epilogue.
4. **Bufferization:** fresh destination construction, a conflicting old-value use requiring a copy, and a restricted case where reuse is established safe.

The most important baseline is an equivalent construction inside the same MimIR pipeline: the same analysis, algorithm, schedule, target backend, and optimization settings, with only the construction mechanism varied where feasible. This helps isolate HOF/PE costs from algorithmic improvements or unrelated backend differences.

MLIR and TVM serve as design and contract references. For a small matched subset, their implementations can also inform implementation-effort comparisons. Runtime cross-framework results will be labelled end-to-end system comparisons unless algorithms and schedules are demonstrably matched. OpenBLAS can contextualize the quality of GEMM/GEMV kernels; it is not a direct baseline for the compiler-construction mechanism.

### 9.2 Correctness and legality tests

Validation will combine unit tests, executable differential tests, negative tests, and transformation composition tests:

- extent-one and empty-domain behavior where supported; rejection where excluded;
- divisible and nondivisible factors, large factors, odd spatial extents, and channel tails such as 130;
- static and symbolic dimensions, with explicit behavior for unresolved preconditions;
- invalid axes, duplicate permutations, zero tile factors, and unsupported iterator mixing;
- nested compositions such as partition → interchange → fuse, checking output layout and contributions;
- shared producers, effectful operations excluded from pure fusion, and epilogues executed after reduction;
- shared constants across functions, distinct allocation identity, conflicting tensor uses, and necessary copies;
- single- and multithreaded execution, checking disjoint writes and scratch ownership;
- floating-point policies, cancellation-heavy inputs, and documented tolerances for numerical algorithms.

Small-domain exhaustive enumeration can check index-map coverage, contribution multiplicity, and bounds for the supported schedule subset. Such tests complement a written argument for the invariants; they are not a proof for all symbolic domains. LLVM verification and runtime memory checking, where supported, will be used to detect malformed IR and memory errors. The verifier alone does not prove source-level equivalence.

### 9.3 Measurements

| Dimension | Measurements and controls |
|---|---|
| Expressiveness | Supported transform classes, required context, restrictions, and duplicated definitions |
| Implementation effort | Construction-specific LOC plus total required driver/helper code; reuse sites; tests counted separately |
| Compiler cost | Phase time, specialization/rewrite counts, intermediate/residual node counts, peak memory, scaling with graph sharing and schedule depth |
| Residual structure | Remaining closures/dictionaries, loop nests, guards, calls, vector operations, allocations, copies, and task grids |
| Runtime quality | Warm end-to-end latency, selected component timings, thread scaling, allocated bytes, and error relative to reference |

LOC will use a stated counting rule and include new shared helpers amortized over the evaluated cases. Existing generic infrastructure must be identified rather than treated as free in one implementation and charged to the other. LOC and observed implementation effort are evidence about these cases, not a controlled usability study.

Compiler benchmarks will include repeated operation instances, a shared producer DAG, deeper schedule composition, and increasing unroll factors. Results should report when specialization ceases to scale and distinguish cold compilation from reuse of cached implementation instances.

Runtime benchmarks will vary shape and batch, include tails, alternate measurement order, and avoid concurrent compilation. Reports will distinguish compile-time thread settings from runtime thread counts, record the machine and compiler, and report variability as well as medians. Weight pretransformation, allocation/free, and layout conversion must be included or excluded consistently and explicitly.

### 9.4 Falsifiable expectations

The approach is supported if multiple selected transformations share semantic construction components, if static HOF abstractions disappear without excessive compiler cost, and if equivalent schedules produce competitive residual code.

The approach is weakened if an ostensibly generic template needs operation-specific driver reconstruction of comparable complexity, if type conversion forces extensive duplicate implementations, or if specialization causes large compile-time/code-size growth. Fusion and bufferization results must report whether the residual analysis/interface burden outweighs the construction benefit.

No fixed speedup is assumed. Failure to outperform an existing system is compatible with useful results about expressiveness or compilation cost, but it cannot be reported as a performance advantage.

## 10. Expected Contributions

1. A transformation taxonomy explaining where HOF/PE descriptions are useful, including explicit counterexamples and required context.
2. A reusable construction interface for selected lowerings, with a documented separation between same-representation decomposition and representation-changing conversion.
3. A compositional model for a bounded set of tensor loop transformations, including output/iteration-rank separation, predicate semantics, and explicit legality contracts.
4. An analysis-guided construction experiment for selected fusion and bufferization cases, establishing where global information remains necessary.
5. A reproducible evaluation of implementation structure, specialization cost, correctness, and residual-code performance.

The novelty sought is the design and empirical characterization of these interfaces in a typed, executable IR. HOFs, staging, explicit dictionaries, index-map composition, and the separation of analysis from rewriting are established techniques and must be credited as such.

## 11. Scope, Work Plan, and Risks

### Required scope

The thesis targets CPU tensor kernels through the existing LLVM backend. It covers a bounded structured tensor subset and selected intraprocedural graph patterns. It does not require a new general typeclass solver, a complete refinement-type system, a complete symbolic theorem prover, all StableHLO operators, automatic schedule search, or a GPU compiler.

A small arithmetic checker and explicit guards are permitted where needed. The objective is a reliable contract for the selected transforms, not full parity with every TVM scheduling primitive.

### Indicative 12-week work plan

| Period | Deliverable |
|---|---|
| Weeks 1–2 | Freeze case studies and revisions; reproduce prototypes; specify taxonomy, semantic assumptions, and baselines |
| Weeks 3–4 | Isolate same-representation construction; prototype the representation-changing interface on a restricted subset |
| Weeks 5–6 | State schedule invariants; audit selected APIs; implement negative and composition tests; inspect residual nests |
| Weeks 7–8 | Extract selected fusion construction and fresh-destination bufferization; add one restricted reuse experiment |
| Weeks 9–10 | Measure compiler scaling and runtime quality; run controlled ablations and investigate regressions |
| Weeks 11–12 | Consolidate results, limitations, reproducible artifacts, and thesis writing |

The schedule is relative to the agreed project start. If time becomes constrained, general reuse analysis and representation-parameterized interpretation are reduced to bounded experiments. The core deliverables remain the taxonomy, same-representation lowering, one explicit representation-changing path, compositional scheduling, and honest boundary-case results.

### Main risks and mitigations

**Specialization explosion.** Keep large domains residual, limit unrolling, profile intermediate graph size, and measure sharing explicitly.

**Overly ambitious semantic guarantees.** State the supported subset, use exact reasoning for coordinate transformations where possible, and separate numerical error contracts from exact transformations.

**Driver complexity disguised as simplicity.** Include all required new infrastructure in implementation comparisons and expose analysis/representation interfaces explicitly.

**Incomplete diagnostics.** Test rejected programs and process failure behavior, rather than assuming a logged message constitutes enforcement.

**Unfair performance attribution.** Hold algorithms and schedules constant in mechanism comparisons; label broader system benchmarks separately.

**Unfinished global analyses.** Use conservative, explicitly scoped policies. Report the restriction and its allocation/recomputation cost instead of claiming optimal fusion or bufferization.

## 12. Related Work and References

The implementation work should be positioned relative to PE and staged DSLs, not only relative to MLIR. Typed embeddings and staged interpreters already demonstrate how function abstraction can disappear through specialization [1–3]. AnyDSL is particularly relevant to MimIR's technical lineage [10].

MLIR provides reference designs for pattern application, conversion legality, type migration, transform orchestration, and bufferization [4–7]. The distinction is the proposed use of executable higher-order construction within MimIR, not an alleged absence of declarative tools or extensibility in MLIR. TVM provides a reference for structured schedule contracts and arithmetic reasoning [8–9].

1. Neil D. Jones, Carsten K. Gomard, and Peter Sestoft. *Partial Evaluation and Automatic Program Generation*. Prentice Hall, 1993.
2. Jacques Carette, Oleg Kiselyov, and Chung-chieh Shan. “Finally tagless, partially evaluated: Tagless staged interpreters for simpler typed languages.” *Journal of Functional Programming*, 2009.
3. Tiark Rompf and Martin Odersky. “Lightweight Modular Staging: A Pragmatic Approach to Runtime Code Generation and Compiled DSLs.” *GPCE*, 2010.
4. MLIR. [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/).
5. MLIR. [PDL dialect](https://mlir.llvm.org/docs/Dialects/PDLOps/) and [PDLL](https://mlir.llvm.org/docs/PDLL/).
6. MLIR. [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/).
7. MLIR. [Bufferization](https://mlir.llvm.org/docs/Bufferization/) and [Ownership-based Buffer Deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/).
8. Apache TVM. [Structured loop scheduling implementation](https://github.com/apache/tvm/blob/def37e30bd4785114ce3ffa6e059da1c36bf2656/src/s_tir/schedule/primitive/loop_transformation.cc). This inspected snapshot uses `s_tir`; naming differs across versions.
9. Apache TVM. [Arithmetic analyzer interface](https://github.com/apache/tvm/blob/main/include/tvm/arith/analyzer.h).
10. [AnyDSL project](https://anydsl.github.io/), including its staging and high-performance library work.
11. OpenXLA. [StableHLO specification](https://openxla.org/stablehlo/spec), used for the original proposal's sort and comparison examples.

The listed online documents are reference entry points. The design discussion also uses locally inspected source snapshots; it does not imply that current online documentation is identical to those snapshots.

## Appendix A. Evidence and Reproducibility Record

**Original proposal:** `MimIR as a powerful DSL Design Doc.md`, proposal repository revision `3220668`.

**MimIR base revision:** `09bfc92ae80bef5a2acdbcb449bbd909a84afdbf`, plus local, uncommitted Winograd and supporting lowering changes. A reproducible submission must include that patch or a later pinned revision containing it. Relevant source locations:

- `src/mim/plug/tensor/phase/lower.cpp`: `lower_via_impl` and phase-controlled implementation application.
- `src/mim/plug/tensor/tensor.mim`: `Tileable`, `mr_iter_tileable`, `retile`, schedule functions, im2col and local Winograd implementations.
- `src/mim/plug/tensor/phase/fuse.cpp` and `include/mim/plug/tensor/phase/fuse.h`: graph-aware fusion and sharing bookkeeping.
- `src/mim/plug/tensor/phase/lower_to_mem.cpp` and its header: representation/boundary conversion, allocation construction, and the conservative reuse policy.
- `src/mim/plug/affine/phase/lower_index.cpp`: affine lowering, including the local shared-literal fix.
- `src/mim/plug/refly/normalizers.cpp`: current check-normalization behavior.
- `docs/rewriting.md`: graph sharing, scoped mappings, and recursive stub reconstruction.

**TVM local snapshot:** `def37e30bd4785114ce3ffa6e059da1c36bf2656`. Inspected `src/s_tir/schedule/primitive/loop_transformation.cc` and `include/tvm/arith/analyzer.h`. These sources are design references; there is no claim of a completed schedule-API equivalence audit in this proposal revision.

**LLVM/MLIR local snapshot:** `0212229ac05d60314db7dfb8ae89c11a015bb7bc`. Inspected `mlir/docs/DialectConversion.md` and `mlir/docs/Bufferization.md`.

**Local preliminary benchmark artifacts:** `docs/benchmarks/conv-winograd/README.md`, `table.md`, `results/`, and `.test/winograd/bench-b*-t*/` in the MimIR worktree. The preliminary suite uses six geometries, batch 1/4, and compile/runtime thread settings 1/4. Each independent run reports a median of seven calibrated warm timing rounds; the table takes the median across three runs. Complete per-round timings were not retained, so that dataset cannot support retrospective per-round variance analysis.

**Prototype validation:** the local Winograd work completed 527 lit tests with seven unsupported cases, 46 CTest tests, and executable checks against scalar references. This establishes a useful starting point; it is not evidence that all proposed interfaces, all original sort semantics, or all symbolic schedule contracts have already been implemented or verified.

## Appendix B. Changes from the Previous Proposal

- Replaced the broad claim that MimIR is a more powerful DSL framework with four bounded, testable research questions about transformation construction.
- Removed unsupported assertions about the proportion of MLIR bugs, inherent C++ unsafety as the primary cause of compiler design problems, and guaranteed performance parity through PE.
- Retained explicit dictionaries, dependent shape relationships, and operator examples, but separated typing from semantic verification and optional elaboration.
- Split dialect conversion into same-representation decomposition and coordinated representation-changing lowering.
- Made compile-time efficiency a first-class evaluation target alongside generated-code performance.
- Turned loop scheduling into a question about compositional domains, access maps, reduction order, predicate placement, and legality contracts.
- Reframed fusion and bufferization as analysis-guided construction experiments, with explicit graph sharing, aliasing, destination, freshness, and lifetime obligations.
- Added controlled baselines, failure criteria, scope limits, a work plan, and an evidence record distinguishing current prototypes from proposed work.
- Deferred a general GPU DSL, a complete frontend/typeclass system, automatic tuning, and agent-driven compiler rewriting to future work.
