# MimIR as a powerful DSL Design Doc

(the examples in MimIR has been either implemented or prototyped with basic lit tests)

## Motivation

1. Whats' the pain point of MLIR? 
2. What's the solution in other compiler frameworks like pytorch's inductor+aten?
3. How can MimIR do better than MLIR and aten?
4. Why cannot MLIR and pytorch do that?  **Descriptive rewrite**
5. What's a good DSL framework? Is MLIR a good one?

### Problems with MLIR

MLIR is a Meta IR framework, it works like a language like cpp, which gives you the ability to define your own operations(types) and organize them into dialects(libraries like STL), but it lacks support for polymorphism and macro, which means you cannot avoid the boilerplate code in frontend language like what template did for cpp and generic types did for rust. Instead, you have to write them in MLIR passes in cpp, which is error prone and lack clear semantics. 

At this time, a great majority bugs in MLIR and its downstream projects like [torch-mir](https://github.com/llvm/torch-mlir) are caused by this deficiency of language design(TODO, list more examples). When writing a high-level tensor operator decomposition, you have to keep in mind or check the semantics of the from/to operator(s) from time to time, and the semantics is **scattered** across documentations, tablegen source code and even the cpp implementation source code. And the **segmentation fault** is everywhere in cpp implementations due to cpp's language drawbacks.

[PDL](https://mlir.llvm.org/docs/Dialects/PDLOps/) and PDLL are dialects that aim at making structural graph rewriting declarative. Their core values are handles to payload IR entities such as operations, values, attributes, and MLIR types. A PDL pattern can match a graph, constrain these handles, construct replacement operations, and delegate additional checks or mutations to registered native functions.

**Current downstream usage also suggests a relatively narrow practical role for PDL(TODO: ask MLIR guys about the reasons)**. IREE uses PDL in a small number of custom-dispatch, target-specialization, and testing cases. Representative patterns still delegate semantic matching and transformation to native functions through `pdl.apply_native_constraint` and `pdl.apply_native_rewrite`. This is evidence about the present engineering boundary, not proof that PDL could never encode the same final IR.

So at this time, to write passes and transforms is painful in MLIR and other frameworks. **we observed that a great majority of passes follow the approach of `template` and `hole filling`.** 

### pytorch's approach

PyTorch reduces some of this burden by keeping many graph-level decompositions in Python while relying on ATen schemas, dispatch, fake/meta implementations, symbolic shapes, and native kernels underneath. This is productive, but the semantics of one operator may still be **distributed** across several definitions.

that's why we wish to move the implementation and semantics into the frontend language like MimIR.

### Rationale of this design

the hole drawbacks in the current DSL designs can be tracked down to one fact: the expressiveness of frontend language is not full exploited. but what does it mean to exploit the expressiveness of frontend languages?  it's about construct the IRs in the compiler implementation language's object representation, like `Operation` objects in MLIR, where `Operation` is just a memory object in cpp, `relax/tensor IR` objects in TVM. But actually, the IRs could be constructed in another approach: function + inline. 

the traditional manual IR manipulation approach was bad in the following aspects:
1.  it was implemented in compiler implementation language like cpp, which is notorious for its memory safety and developer unfriendliness, developer could **easily panic the compiler** by casting an `Operation` in MLIR to an `xxxInterface` and using them without nullptr checking. 
2. the types are not **composable**. this leads to several serious issues, user could easily broke the semantics of operator since many constraints could not be expressed in frontend language's type system. top make the compiler reject ill-constructed IRs either from inputs or transformation outputs, the developers have to implement `Verify` interface for operator to tell if the current `Operation` is valid, but it was still implemented in cpp, `Verify` implementer could miss some important invariants/constraints. even with the facility of tablegen, developers still have to write a decent amount of boilerplate code. furthermore, to debug the code rejected by `Verify` is also painful: you would spend many time locating the problem.

for example, we could implement triu/tril lower via a composition of axioms that is complicated that is more suitable to be implemented in cpp and mimir lambdas. while in [torch-mlir](https://github.com/llvm/torch-mlir/blob/a220e1c965978c0ace015b9434cd669d55f62f1d/lib/Conversion/TorchToStablehlo/Basic.cpp#L2113), it looks redundant and messy.

```mimir

/// PyTorch-style triangular masking over the last two dimensions. `diagonal` is signed:
/// positive values move the retained diagonal upward, negative values move it downward.
axm %tensor.torch_triu: {T: *}
                       → [zero: T, diagonal: I64]
                       → {r: Nat, s: «r; Nat»}
                       → [input: «s; T»]
                       → «s; T»;

axm %tensor.torch_tril: {T: *}
                       → [zero: T, diagonal: I64]
                       → {r: Nat, s: «r; Nat»}
                       → [input: «s; T»]
                       → «s; T»;

lam %tensor.triangular_pointwise_elem_impl {T: *, r: Nat}
                                           (s: «r; Nat»)
                                           (upper: Bool, zero: T, diagonal: I64)
                                           (out_iters: «r; I64»)
                                           (input: «s; T»)
                                         : T =
    let row_axis = %core.idx r 0 (%core.nat.sub (r, 2));
    let col_axis = %core.idx r 0 (%core.nat.sub (r, 1));
    let row = out_iters#row_axis;
    let col = out_iters#col_axis;
    let shifted_row = %core.wrap.add 0 (row, diagonal);
    let keep_upper = %core.icmp.sge (col, shifted_row);
    let keep_lower = %core.icmp.sle (col, shifted_row);
    let keep = %core.select (upper, keep_upper, keep_lower);
    let coords = ‹d: r; %core.conv.u (s#d) (out_iters#d)›;
    let elem = %tensor.get @(T, r, s) (input, coords);
    %core.select @T (keep, elem, zero);

lam %tensor.torch_triu_impl {T: *}
                            (zero: T, diagonal: I64)
                            {r: Nat, s: «r; Nat»}
                            (input: «s; T»)
                          : «s; T» =
    let body = %tensor.triangular_pointwise_elem_impl @(T, r) s (tt, zero, diagonal);
    %tensor.pointwise_loop @(«s; T», T, r) s body input;

lam %tensor.torch_tril_impl {T: *}
                            (zero: T, diagonal: I64)
                            {r: Nat, s: «r; Nat»}
                            (input: «s; T»)
                          : «s; T» =
    let body = %tensor.triangular_pointwise_elem_impl @(T, r) s (ff, zero, diagonal);
    %tensor.pointwise_loop @(«s; T», T, r) s body input;

/// Structure-only pointwise loop. The body receives raw I64 loop counters and the original input tuple.
/// LowerMapReduce turns this into nested %affine.For loops and stores each computed element at the identity output
/// coordinate.
axm %tensor.pointwise_loop: {Tin T: *, r: Nat}
                          → [s_out: «r; Nat»]
                          → [body: [«r; I64»] → [Tin] → T]
                          → [inputs: Tin]
                          → «s_out; T»;
```

thus, we introduce the `rewrite_via_template` design, where we could construct a template function that's generic over dtypes and tensor shapes and other type parameters. it's a pure function. from a template point of view, the parameters including the type parameters serves as **holes**, when the caller pass the parameters, the holes are filled and we a complete template that is equivalent to the hand-written IRs.

## Other strengths of MimIR

beside the convenience brought by partial evaluation, MimIR has other strength that would made the analysis and transform easy.

### Semantics

let's understand MimIR's ability at capturing the semantics via a example of [`stablehlo.sort`](https://openxla.org/stablehlo/spec#sort) operator

the implementation code could be found [here]()

##### StableHLO Sort in MimIR

The `triu`/`tril` example demonstrates that MimIR can express both the type-level
shape contract and the implementation of a tensor operator in one composable
definition. StableHLO `sort` is a more demanding case study because its semantics
are split between two parts: the constraints on the tensor operands and the
semantics of the comparator supplied to the sort.

##### Constraints and Implementation

A tensor operator has two major parts:

1. **Constraints.** These describe valid input and output types, shape invariants,
   and restrictions on attributes. For `sort`, the constraints include a non-empty
   variadic input list, identical input and output shapes, a common sort dimension,
   and a comparator whose argument types match every input element type.
2. **Implementation.** This describes how valid values are transformed. MimIR
   lowers `sort` to a stable bubble sort over every one-dimensional slice orthogonal
   to the selected dimension. All variadic operands are carried in one accumulator,
   so a single swap keeps keys and payload tensors synchronized.

The StableHLO dimension is a signed constant and may be negative. The importer
normalizes a negative dimension by adding the rank. The MimIR operator then uses
`Idx r`, which makes the normalized range constraint intrinsic to the type of the
dimension argument.

The rank remains a canonical parameter: an inequality such as `r >= 2` should not
be encoded as `batch_rank + 2`, because the current normalizer cannot automatically
prove those arithmetic forms definitionally equivalent.

Instead, the operator keeps its canonical `r` and `n` parameters and expresses the
inequality as a checked continuation:

```mimir
lam %runtime.require {T: *, n: Nat}
                     (condition: Bool, message: «n; I8»)
                     (body: [] → T): T =
    if condition then body() else %runtime.fail @(T, n) message;

%runtime.require (%core.ncmp.ge (r, 2), "triu requires rank >= 2") checked;
```

The thunk is essential under strict evaluation: the implementation is constructed
only after the condition has been checked. A closed condition is reduced by partial
evaluation (`true` removes the guard and `false` becomes a static failure); a dynamic
condition remains an explicit control-flow check. This gives a practical first
version of refinement-like preconditions without adding a `requires` keyword.

For statement-style checks, MimIR exposes a separate effectful operation:

```mimir
%runtime.assert:
    [%mem.M a, Bool, «n; I8»] → %mem.M a
```

The explicit `%mem.M` token sequences the assertion with subsequent effects. It
does not represent a memory mutation; it is an ordering dependency. Pure tensor
operators use `require`, while bufferized or effectful code can use `assert`.

##### Comparator Semantics

StableHLO `compare` selects its comparison type from the element type:

- signed integers use `SIGNED`;
- unsigned integers and booleans use `UNSIGNED`;
- floating-point values use `FLOAT` or `TOTALORDER`;
- complex values use `FLOAT` semantics for equality-style comparisons.

In an MLIR-style representation, these rules are usually checked in a verifier
separate from the comparator body. It is easy for a lowering to accept the right
function signature while accidentally using signed comparison for a boolean or an
unsigned integer. The `stablehlo_compare` implementation makes the policy explicit
through a type-indexed dictionary:

```mim
let %tensor.StableHloCompare = [
    T: *,
    compare_type: Nat,
    compare: [T, T] -> Bool,
];
```

The witness carries the element type, the StableHLO comparison policy, and the
primitive comparator together. A sort comparator can then be composed from
`%tensor.stablehlo_compare` rather than directly naming `%core.icmp` or
`%math.cmp`:

```mim
let I32_SIGNED =
    (I32, %tensor.compare_type.signed_int, %core.icmp.sl @i32);

lam less (lhs rhs: I32): Bool =
    %tensor.stablehlo_compare I32_SIGNED (lhs, rhs);
```

This is a **typeclass**-like design, but it does not require a general-purpose typeclass
solver. MimIR already uses **explicit** dictionaries for structures such as
`%tensor.Ring`. A future resolver can construct canonical witnesses from the
element type: `Bool` and unsigned integers resolve to unsigned comparison, signed
integers to signed comparison, and floating-point types to floating comparison.

##### What Is Captured Today

It is useful to separate StableHLO's constraints from the parts of the semantics
that are already represented by MimIR:

**Captured by the type and implementation**

- **C2, input/output types:** `Ts` is shared by the input and result packs, so
  `type(inputs...) = type(results...)` is enforced by construction.
- **C3, shapes:** every input and result uses the same shape `s`.
- **C5, comparator signature:** the comparator has exactly `2 * n` arguments with
  the interleaved order `lhs0, rhs0, lhs1, rhs1, ...`, and each argument type is
  selected from `Ts`.
- **Lockstep sorting:** all variadic inputs are carried through the same swap, so
  keys and payloads remain synchronized.
- **Sorting direction:** adjacent elements are exchanged when
  `comparator(rhs, lhs)` is true, which implements non-descending order for the
  StableHLO comparator convention.
- **Stability:** the implementation is stable. This is valid for both values of
  `is_stable`, because StableHLO only requires stability when it is true.

**Captured by checks or importer conventions**

- **C1, non-empty inputs:** `%runtime.require (n >= 1, ...)` is used. A literal
  failure is a compile-time error; a dynamic condition remains a runtime check.
- **C4, dimension:** the importer normalizes a negative dimension to a non-negative
  value. The MimIR operator then uses `dimension : Idx r`, which captures the
  normalized range `0 <= dimension < r`. The original signed range check
  `-r <= dimension < r` remains an importer responsibility.
- **Comparator ordering:** strict-weak-ordering is treated as a precondition. It is
  not checked dynamically because validating it would require an unbounded number
  of comparator calls.

`%runtime.assert` is the statement-style form for effectful code:

```mim
%runtime.assert:
    [%mem.M a, Bool, «n; I8»] → %mem.M a
```

Pure tensor expressions use continuation-based `%runtime.require`. Its failure is
backend-neutral: the host LLVM backend may materialize the message and call
`puts`/`abort`, while an NVPTX backend can lower the same failure to `llvm.trap` or a
device assertion ABI.

**Unsupported or TODO**

- **Input element-type domain:** `Ts` is currently polymorphic over `*`; the
  implementation does not yet reject element types that StableHLO `sort` does not
  support, nor does it model per-tensor quantized tensors.
- **Comparator typeclass resolution:** an explicit comparison witness can express
  `SIGNED`, `UNSIGNED`, `FLOAT`, and `TOTALORDER`, but users can still construct an
  inconsistent witness. A canonical typeclass resolver must reject such combinations
  and select the policy from the element type.
- **Complete `stablehlo.compare` semantics:** IEEE floating-point NaN behavior,
  `TOTALORDER`, complex lexicographic comparison, and quantized comparison require
  additional primitives and resolver support.
- **Constant attributes:** StableHLO requires `dimension` and `is_stable` to be
  constants. The current signature permits a general `Bool` for `is_stable`; this
  should be tightened when constant-attribute witnesses are introduced.
- **Runtime executable diagnostics:** the IR preserves dynamic failure messages and
  lowers host failure through the LLVM hook, but the direct-style LLVM emitter still
  needs to handle a non-returning failure appearing in a value-returning expression
  before end-to-end message/abort execution is complete.

##### Remaining Extensions

The dictionary currently makes the comparison policy **explicit**, but users can still
manually construct an **inconsistent** tuple, such as an unsigned policy paired with a
signed comparator. A stronger type system could prevent this in two steps:

1. distinguish signed and unsigned integer types, rather than representing both only
   as an `Idx` carrier;
2. add **compile-time instance resolution** for canonical `StableHloCompare` witnesses.

That resolver would reject unsupported combinations during elaboration, while
partial evaluation would erase the witness and leave only the selected primitive
comparison. `TOTALORDER`, complex lexicographic comparison, and quantized
dequantize-compare require additional primitives before they can be represented
without weakening StableHLO semantics.

##### Constraints at Type Level

Not every operator constraint belongs in the type-level signature. A useful division
observed in the `triu`/`tril` and `sort` implementations is:

- dependent types express equalities and shape relationships that are part of the
  representation, such as input/output shape preservation, variadic element types,
  comparator arity, and `dimension : Idx r`;
- type-indexed witnesses express semantic choices such as StableHLO's
  `compare_type` mapping;
- ordinary predicates express inequalities and other checks whose truth may depend
  on dynamic metadata, such as `rank >= 2` and non-empty variadic input lists;
- partial evaluation removes predicates proved from literals, while dynamic
  predicates lower to target-specific control flow and a non-returning failure.

This avoids imposing arithmetic theorem proving on the normalizer. A future
refinement-type or proof-argument extension could move selected predicates into the
interface, but it should be introduced only when call-site proof obligations provide
clear value beyond the current checked-continuation encoding.

#### dynamic shapes

dynamic shape computations have become critical in modern machine learning workloads, especially in emerging large language models, to support them, different frameworks designed **new** IRs or attached different metadata. for example, TVM proposed [Relax IR](https://arxiv.org/pdf/2311.02103) that made symbolic expression a first class citizen in the IR by [sympy](https://github.com/sympy/sympy), projects based on MLIR like [wave](https://github.com/iree-org/wave) also used sympy in their python frontend and MLIR's `AffineExpr` in the MLIR side, but as the author had claimed, [there is no good way to represent them in upstream MLIR](https://discourse.llvm.org/t/mlir-open-meeting-symbolic-expressions-and-symbolic-tensors-2025-05-15/86223)

thanks to MimIR's dependent type system, we support symbolic shape **by construction** without touching the language. we simply need to developer some arithmetic analysis on the `Def` (everything is a `Def` in MimIR's world) to get the divisibility, boundary informations and consume them later in stages like codegen where the code generator needs to know if a dynamic shape value is divisible by the machine's vector size.

TODO: arithmetic analyzer algorithms & usage in later pipeline stages

## Design and Implementation at language level

a IR pass consists of two major parts: analysis + rewrite/transform (the LOC ratio is about `1:10` to `1:5`). we made an observation that a great deal of rewrite could be viewed as: defining template with holes + collecting for parameters + filling the holes with the arguments.
Another way to understand this approach is `pure function + partial evaluation(inline)`. 

to have a fully useful `template + partial evaluation(inline)` compiler frontend, we need to address several issues, some involving extending the ability in MimIR. But thanks to its functionality and expressive dependent type system, these challenges could be solved by lightweight extensions.

### `rewrite_via_template` implementation

see [lower_via_impl](https://github.com/mimir/mimir/blob/68182ea69b5ddc1221b1b4c3503938e37f567c87/src/mim/plug/tensor/phase/lower.cpp#L15) in current tensor plugin. we could lift it to a more widely used tool!

### Simple Interface System Implementation(proposal)

MimIR already supports parametric polymorphism, but it does not currently provide typeclass-style dispatch. Tensor operators therefore encode operations such as add and mul in explicit dictionaries, which requires manual instantiation at every call site and does not scale well.

A lightweight solution is to add **compile-time interfaces with type-indexed instances**:

```mimir
interface Ring<T> {
    zero: T;
    add: (T, T) -> T;
    mul: (T, T) -> T;
}

instance Ring<i32> = {
    zero = 0;
    add = %core.add.i32;
    mul = %core.mul.i32;
};
```

A polymorphic operator can request an interface implicitly:

`lam dot [T: *] (R: Ring<T>) (...) : T = ...`

At elaboration time, MimIR resolves `Ring<T>` using the concrete type:

```mimir
Ring<i32> -> instance Ring<i32>
Ring<f32> -> instance Ring<f32>
```

The resolved dictionary is then passed explicitly in the elaborated IR and can be erased or specialized during partial evaluation.

The same mechanism can express StableHLO comparison semantics:

```mimir
interface StableHloCompare<T> {
    compare_type: CompareType;
    compare: (T, T) -> Bool;
}

instance StableHloCompare<i32> = SIGNED;
instance StableHloCompare<u64> = UNSIGNED;
instance StableHloCompare<Bool> = UNSIGNED;
instance StableHloCompare<f32> = FLOAT;
instance StableHloCompare<f64> = FLOAT;
```

### Onnx/Torch dialect  - Frontend for high-level operators

to import operators from pytorch, there are two workable approaches:
1. we could easily get the computational graph(`fx.Graph`) representation of a pytorch model by pytorch's api, and we could then traverse its nodes by DAG order and do a mapping MimIR's python bindings. it's done in pure python without touching the MimIR and cpp.
2. add a new `torch` plugin and do `torch => tensor` conversion.

while the first one sounds nice at first, we have shown that the **semantics** of operators and decompositions are **ill-defined** in this approach, so we chose to implement it via a mimir `torch` dialect, like what torch-mlir did. Compared with `torch-mlir`, we enjoy the strengths that has been declared above.

### Dynamic shape implementation and usage across the pipeline

#### Fusion

TODO: implement fusion by `analysis + template define + rewrite_via_template`, list some examples and why we could achieve result as performant as TVM's. the **producer-consumer** style fusion could be expressed by continuation and high-order function, so after partial evaluation, we could get the fused kernel. (Question: **can we avoid heavy usage of cpp while let mimir handle the ir structure transform**?)

#### Arithmetic Analysis

TODO: brief TVM style [arithmetic analyzer](https://github.com/apache/tvm/blob/main/include/tvm/arith/analyzer.h) implementation description.

#### Shape instantiation and dispatch

to make the dynamic shape model run, we need a workable dispatch approach.

TODO: how to do the dynamic shape codegen(parsing the dynamic shape from program's argument and do runtime dispatch)

### Template based Transforms

(we left the gpu/machine level transforms case study to the next DSL section)

#### tensor level transforms case study

there are at least these transforms in MLIR that could be expressed in MimIR's `rewrite_via_template` 

1. WinogradConv2D.cpp
2. TosaToLinalg.cpp (decompostion)
3. PackAndUnpackPatterns.cpp
4. SplitReduction.cpp
5. TransposeConv2D.cpp

 the mimir defines the **template** and the cpp is responsible for module/global level **analysis** and **apply** the collected arguments to the template.

## DSL + template rewrite (could be another thesis topic)

**Note: this is unrelated to the generate ML compiler approach: this gives hardware and compiler developer a powerful toolbox to write high performance kernels! yes, we can make MimIR yet another DSL to write performant GEMM kernel.(Actually, this approach is easier for verification, since the transforms/higher-order functions/templates are developed and applied by experts directly in IR and we need lesser analysis and auto-tuning)**

many DSLs these days adopted a hybrid approach: it allows experts to define performant kernels like `gemv`, `gemm`, `attention`, `flashattention` with a high-level DSL and compile the DSL into lower level IRs, like what [tilelang](https://github.com/tile-ai/tilelang) would compile its DSL into TVM's relax IR. this approach give the compiler **more fusion opportunities** since the kernels are not provided by external libraries like `cuBlas`, they could be fused with other operators like `gemm` and `relu` could be fused together and save the cost of kernel launches and temporal buffer allocations.

thanks to the **expressiveness** of `rewrite_via_template`, we could achieve similar performance without sacrificing the semantics.

(these had already been partially implemented in https://github.com/AnyDSL/stincilla and https://github.com/AnyDSL/thorin/blob/d60f0e31e70270bb1242d12c050563d0befa8ad2/src/thorin/be/llvm/llvm.h#L87, but we need some more convincing proofs that this DSL approach is more powerful)

to make it work, we need to define the following templates as higher-order functions like [what tilelang had done](https://tilelang.com/autoapi/tilelang/language/loop/index.html):

- parallel
- vectorized
- unroll
- pipelined
- persistent
- partition

`pipelined`  would look like

```mimir
// TODO: verify we could benefit from pipelined approach
// see https://gist.github.com/hsqStephenZhang/3543a078396539ac7672ca847f011c5b
```

dev phases:
1. for quick prototype, it could be easier to define a TVM TensorIR dialect/plugin, then we could do exact what tilelang had done: compile it to TensorIR and enjoy the optimizations of TVM since the optimizations in native MimIR are still pretty poor
2. **rewrite** MLIR/tilelang/TVM transforms in MimIR!

## Another thought

LLMs are great at rewrite as long as it has a powerful **Oracle** that not only would tell if the result is correct but also what the problems are(like type check error message). the rewrite into rust of great project like **[react](https://github.com/react/react/pull/36173)**/[bun](https://github.com/oven-sh/bun/pull/30412) had proved the feasibility and potential.

Can we drive the agents to **rewrite** imperative transforms of MLIR/Tilelang/TVM into the **descriptive** transforms in MimIR in a descriptive way with efficiency?
