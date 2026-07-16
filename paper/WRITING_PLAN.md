# Writing Plan: Double-Blocked Factorizations on Hygon DCUs

## 1. Paper Objective

Write an English conference paper centered on the design and implementation of
double-blocked Cholesky and pivoted LU factorizations for the Hygon K100_AI DCU.
The external performance baseline is the official rocSOLVER library.

The paper does not claim that double blocking is a new concept. Its intended
contribution is a factorization-specific double-blocked framework for LU and
Cholesky, a correctness-preserving LU permutation mapping, and an empirical
study of the framework on the Hygon HIP software stack. Iterative blocked TRSM,
K-blocked SYRK, and M-tiled GEMM are general execution transformations; they are
not claimed as new or K100_AI-specific algorithms.

## 2. Central Claim

The working claim is:

> Double blocking decouples the granularity of dependency-bound panel
> factorization from the granularity of panel preparation. The separation
> provides a common factorization framework in which POTRF and GETRF retain the
> same two-level schedule while using factorization-specific update and data
> movement policies. Measurements on K100_AI determine whether general
> BLAS-level transformations should be enabled for the generated shapes.

This claim must be supported by the final ablation study and stage profiles.

## 3. Scope and Claim Boundaries

### In scope

- Dense square Cholesky factorization of symmetric positive-definite matrices.
- Dense square LU factorization with partial pivoting.
- FP32 as the primary precision.
- Hygon K100_AI, HIP, rocBLAS, and rocSOLVER.
- End-to-end comparison with `rocsolver_spotrf` and `rocsolver_sgetrf`.
- Internal ablations of the double-blocked implementations.
- Runtime, throughput, stage breakdown, and numerical backward error.
- Exact-shape primitive controls used to derive fixed K100_AI dispatch rules.

### Out of scope unless additional evidence is collected

- Claims of superiority over every alternative factorization schedule.
- Claims of portability to arbitrary GPU architectures.
- Claims about the internal algorithm used by rocSOLVER.
- FP64 performance claims without a complete FP64 evaluation.
- Performance comparisons beyond matrix sizes supported reliably by the
  rocSOLVER baseline.
- Claims that double blocking itself is novel.
- Claims that blocked TRSM or tiled SYRK/GEMM are inherently K100_AI-specific.
- Architecture-specific conclusions inferred from algebra alone rather than
  exact-shape and end-to-end measurements.

## 4. Stable Terminology

Use the following names consistently:

- `DB-LU`: algorithmic name for the double-blocked LU factorization.
- `DB-Cholesky`: algorithmic name for the double-blocked Cholesky
  factorization.
- `DB-GETRF` and `DB-POTRF`: implementation and benchmark labels for DB-LU
  and DB-Cholesky, respectively.
- `B` or `NB`: outer block width.
- `b`: inner panel width.
- `T_K`: reduction-dimension tile for SYRK.
- `T_M`: row-dimension tile for GEMM.
- `b_t`: iterative TRSM block size.
- `b_p`: column tile used by the batched permutation workspace.

Use "outer block" and "inner panel" in prose. Avoid switching between
"micro block", "leaf block", and "inner block" unless they denote distinct
levels.

## 5. Writing Phases

### Manuscript narrative standard

- Motivation must establish a measurable conflict before introducing a method.
- Methods must describe matrix regions, dependencies, and invariants before
  discussing API calls or tuning constants.
- Each execution transformation must state the bottleneck, algebraic
  transformation, correctness condition, and measured selection criterion.
- Generic execution transformations must remain separate from algorithmic and
  platform-specific contribution claims.
- Implementation details belong after the Methods and Operation-Level Mapping
  sections and should explain only decisions needed to reproduce the
  algorithms.
- Preliminary logs and parameter-search histories must not appear as the main
  narrative. They may supply an ablation or sensitivity figure after the
  experiment protocol is frozen.
- The abstract and conclusion receive numerical claims only after the complete
  evaluation is generated from one source commit.

### Phase A: Algorithm manuscript

- [x] Fix the paper scope and central claim.
- [x] Audit the current POTRF and GETRF implementations.
- [x] Create the manuscript structure.
- [x] Draft the abstract with result placeholders.
- [x] Draft the introduction and background.
- [x] Draft the unified double-blocking framework.
- [x] Draft DB-POTRF and DB-GETRF algorithms.
- [x] Draft the reference implementation section.
- [x] Create an empty evaluation section.
- [x] Draft related work positioning, limitations, and conclusion.

### Phase B: Experiment freeze

- [ ] Finish all algorithm changes intended for the paper.
- [ ] Add an exact-shape left/unit-lower TRSM control matching DB-LU; the
  existing right-side TRSM benchmark is not a valid substitute.
- [ ] Archive generated TRSM, SYRK, and GEMM shapes with their leading
  dimensions.
- [ ] Run correctness tests at small and non-power-of-two sizes.
- [ ] Select one source commit for all reported experiments.
- [ ] Record the DTK, compiler, rocBLAS, rocSOLVER, driver, and device versions.
- [ ] Freeze the parameter-selection policy before the scaling run.
- [x] Audit the benchmark timer and error-verification paths.
- [x] Report mean, median, sample standard deviation, minimum, and maximum.
- [x] Create one script that reproduces every reported experiment.

### Phase C: Unified evaluation

- [ ] Run primitive shape characterization.
- [ ] Compare untiled and transformed TRSM, SYRK, and GEMM at generated shapes.
- [ ] Retain only transformations with stable stage and end-to-end effects.
- [ ] Run end-to-end POTRF scaling.
- [ ] Run POTRF ablations.
- [ ] Run POTRF stage profiles.
- [ ] Run end-to-end GETRF scaling.
- [ ] Run GETRF ablations.
- [ ] Run GETRF stage profiles.
- [ ] Run correctness and backward-error experiments.
- [ ] Parse raw logs into machine-readable tables.
- [ ] Generate all figures from checked-in plotting scripts.

Only the panel-update conflict and a compact set of primitive shape
measurements should be moved into Section III as motivation. End-to-end
performance, ablations, stage profiles, and accuracy belong in Section VII.
This separation preserves the paper's analysis-method-evaluation narrative.

### Phase D: Final paper

- [ ] Replace every `TBD` and `RESULTS_TODO` marker.
- [ ] Add complete references and verify citation metadata.
- [ ] Rewrite the abstract using final measured results.
- [ ] Check that every performance claim is backed by a figure or table.
- [ ] Check that every algorithmic claim matches the committed implementation.
- [ ] Convert the Markdown manuscript to the target conference LaTeX template.
- [ ] Perform final language, layout, and reproducibility reviews.

## 6. Planned Evaluation Matrix

The following protocol is provisional until Phase B is complete.

### Common protocol

- Device: one K100_AI DCU selected with `HIP_VISIBLE_DEVICES`.
- Matrix sizes: 8192, 16384, 24576, 32768, and 40960.
- Warmup iterations: 2.
- Profile iterations: 10.
- Precision: FP32.
- Inputs: the same generated matrix is restored before every timed method.
- Timing: GPU execution only, measured with HIP events on the execution stream.
- Correctness: performed outside the timed region.
- Residual policy: deterministic 4096-entry sampling during scaling, plus full
  Frobenius residuals at 4096 and a non-power-of-two representative size.
- External baseline: official rocSOLVER only.
- Large unsupported rocSOLVER cases are excluded rather than counted as wins.

### POTRF methods

1. rocSOLVER POTRF.
2. Basic DoubleBlock POTRF.
3. DoubleBlock plus K-blocked SYRK.
4. DB-POTRF: K-blocked SYRK plus adaptive M-tiled GEMM.

### GETRF methods

1. rocSOLVER GETRF with partial pivoting.
2. Basic DoubleBlock GETRF with immediate pivot application and iterative
   blocked outer TRSM.
3. DoubleBlock plus batched pivoting.
4. DB-GETRF: batched pivoting plus M-tiled inner GEMM.

Iterative blocked outer TRSM is common to the DoubleBlock GETRF variants. A
direct rocBLAS outer-TRSM run is retained only as an implementation ablation;
it is not a separate factorization method.

### Parameter policy

The final scaling comparison must use a deterministic policy rather than
selecting the best configuration independently at every matrix size. Tuning
experiments must use a separate dataset and be reported separately.

## 7. Required Figures and Tables

- Figure 1: measured panel-update conflict as block width changes.
- Figure 2: untiled-versus-transformed K100_AI GEMM, SYRK, and TRSM behavior
  at exact factorization shapes.
- Figure 3: aligned DB-LU and DB-Cholesky double-blocking workflows.
- Figure 4: delayed LU pivot regions and two-stage gather/scatter permutation.
- Figure 5: POTRF throughput versus matrix size.
- Figure 6: GETRF throughput versus matrix size.
- Figure 7: POTRF and GETRF stage breakdowns.
- Table 1: hardware and software environment.
- Table 2: fixed parameter policy.
- Table 3: end-to-end runtime, throughput, speedup, and backward error.
- Table 4: ablation results.

## 8. Claim Ledger

### Supported by the current implementation

- Both factorizations use an outer block and an inner panel.
- DB-POTRF provides K-blocked SYRK and conditional M-tiled GEMM updates.
- DB-GETRF provides delayed pivot composition and two-stage permutation.
- DB-GETRF uses iterative blocked outer TRSM with GEMM updates and block TRSM
  calls.
- Both implementations use rocSOLVER for local diagonal or panel
  factorization and rocBLAS for BLAS-3 operations.
- The TRSM, SYRK, and GEMM transformations are general and parameterized
  independently of the double-blocked dependency graph.

### Requires the unified evaluation

- End-to-end speedup over rocSOLVER.
- Stability of speedup over the complete matrix-size range.
- The contribution of each retained execution transformation to final
  performance.
- Whether each general execution transformation should remain enabled on
  K100_AI after direct primitive, stage, and end-to-end controls.
- The final recommended values or policies for `B`, `b`, `T_K`, `T_M`,
  `b_t`, and `b_p`.
- Numerical equivalence within the reported backward-error tolerance.

### Do not claim

- "The first double-blocked factorization."
- Universal superiority over alternative blocked schedules.
- "Architecture independent."
- "K100_AI-specific TRSM/SYRK/GEMM algorithm."
- "rocSOLVER uses a particular internal blocking algorithm" without source
  evidence.
- "No numerical cost" without reporting backward error.

## 9. Reference Coverage Backlog

References remain intentionally absent until the algorithm and evaluation
narrative is frozen. The final bibliography must cover all of the following
areas rather than relying mainly on OSLA:

- foundational blocked LU and Cholesky formulations and stability;
- recursive, inner-blocked, and multi-level dense factorizations;
- MAGMA and comparable hybrid CPU-GPU solver libraries;
- GPU-resident LU and Cholesky implementations and vendor solver libraries;
- tiled and task-based systems, including representative DAG runtimes;
- GPU pivot search, row interchange, batched permutation, and fused panels;
- communication-avoiding and tournament-pivoting LU, clearly distinguished
  from standard partial pivoting;
- OSLA as the closest double-blocking precedent;
- rocBLAS, rocSOLVER, HIP, and Hygon platform documentation.

Each Related Work paragraph must receive at least one primary citation, and
claims about a named system must be checked against its original paper or
official documentation before numbering the bibliography.
