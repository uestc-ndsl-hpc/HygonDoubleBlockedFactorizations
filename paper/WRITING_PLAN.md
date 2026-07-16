# Writing Plan: Double-Blocked Factorizations on Hygon DCUs

## 1. Paper Objective

Write an English conference paper centered on the design and implementation of
double-blocked Cholesky and pivoted LU factorizations for the Hygon K100_AI DCU.
The external performance baseline is the official rocSOLVER library.

The paper does not claim that double blocking is a new concept. Its intended
contribution is the architecture-oriented realization of double blocking in the
Hygon HIP software stack, together with factorization-specific mechanisms that
make the structure effective on K100_AI.

## 2. Central Claim

The working claim is:

> Double blocking decouples the granularity of dependency-bound panel
> factorization from the granularity of high-throughput matrix updates. On
> Hygon K100_AI, this separation enables a common factorization framework in
> which POTRF and GETRF can use different update, triangular-solve, and data
> movement policies while retaining the same two-level schedule.

This claim must be supported by the final ablation study and stage profiles.

## 3. Scope and Claim Boundaries

### In scope

- Dense square Cholesky factorization of symmetric positive-definite matrices.
- Dense square LU factorization with partial pivoting.
- FP32 as the primary precision.
- Hygon K100_AI, HIP, rocBLAS, and rocSOLVER.
- End-to-end comparison with `rocsolver_spotrf` and `rocsolver_sgetrf`.
- Internal ablations of the proposed double-blocked implementations.
- Runtime, throughput, stage breakdown, and numerical backward error.

### Out of scope unless additional evidence is collected

- Claims of superiority over every blocked or recursive factorization.
- Claims of portability to arbitrary GPU architectures.
- Claims about the internal algorithm used by rocSOLVER.
- FP64 performance claims without a complete FP64 evaluation.
- Performance comparisons beyond matrix sizes supported reliably by the
  rocSOLVER baseline.
- Claims that double blocking itself is novel.

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
- `b_t`: recursive TRSM leaf size.
- `b_p`: column tile used by the batched permutation workspace.

Use "outer block" and "inner panel" in prose. Avoid switching between
"micro block", "leaf block", and "inner block" unless they denote distinct
levels.

## 5. Writing Phases

### Manuscript narrative standard

- Motivation must establish a measurable conflict before introducing a method.
- Methods must describe matrix regions, dependencies, and invariants before
  discussing API calls or tuning constants.
- Each optimization must state the bottleneck, algebraic transformation,
  correctness condition, and expected performance effect.
- Implementation details belong after the Methods and Optimizations sections
  and should explain only decisions needed to reproduce the algorithms.
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
- [x] Draft the Hygon implementation section.
- [x] Create an empty evaluation section.
- [x] Draft related work positioning, limitations, and conclusion.

### Phase B: Experiment freeze

- [ ] Finish all algorithm changes intended for the paper.
- [ ] Run correctness tests at small and non-power-of-two sizes.
- [ ] Select one source commit for all reported experiments.
- [ ] Record the DTK, compiler, rocBLAS, rocSOLVER, driver, and device versions.
- [ ] Freeze the parameter-selection policy before the scaling run.
- [x] Audit the benchmark timer and error-verification paths.
- [x] Report mean, median, sample standard deviation, minimum, and maximum.
- [x] Create one script that reproduces every reported experiment.

### Phase C: Unified evaluation

- [ ] Run primitive shape characterization.
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
2. Basic DoubleBlock GETRF with immediate pivot application and recursive
   outer TRSM.
3. DoubleBlock plus batched pivoting.
4. DB-GETRF: batched pivoting plus M-tiled inner GEMM.

Recursive outer TRSM is common to the current DoubleBlock GETRF variants. It
must not be presented as an additional optimization after batched pivoting
unless a direct-TRSM control variant is implemented before the experiment
freeze.

### Parameter policy

The final scaling comparison must use a deterministic policy rather than
selecting the best configuration independently at every matrix size. Tuning
experiments must use a separate dataset and be reported separately.

## 7. Required Figures and Tables

- Figure 1: measured panel-update conflict as block width changes.
- Figure 2: selected K100_AI GEMM, SYRK, and TRSM shape behavior.
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
- DB-GETRF uses recursive outer TRSM with GEMM updates and leaf TRSM calls.
- Both implementations use rocSOLVER for local diagonal or panel
  factorization and rocBLAS for BLAS-3 operations.

### Requires the unified evaluation

- End-to-end speedup over rocSOLVER.
- Stability of speedup over the complete matrix-size range.
- The contribution of each optimization to final performance.
- The final recommended values or policies for `B`, `b`, `T_K`, `T_M`,
  `b_t`, and `b_p`.
- Numerical equivalence within the reported backward-error tolerance.

### Do not claim

- "The first double-blocked factorization."
- "Universally faster than blocked right-looking algorithms."
- "Architecture independent."
- "rocSOLVER uses a particular internal blocking algorithm" without source
  evidence.
- "No numerical cost" without reporting backward error.
