# Double-Blocked LU and Cholesky Factorizations for Hygon DCUs

**Authors:** To be added

**Affiliation:** To be added

## Abstract

Blocked one-sided factorizations alternate between dependency-bound panel
operations and compute-intensive matrix updates. On a GPU, these two stages
favor different granularities: a narrow panel limits the critical path, whereas
a wide update exposes more regular Level-3 BLAS operations. A conventional
single-level blocked algorithm couples both requirements to one block size and
does not allow them to be tuned independently. This paper presents
double-blocked LU and Cholesky factorizations for the Hygon K100_AI DCU. An
outer panel controls coarse-grained matrix updates and data movement, while an
inner block controls local GETRF or POTRF operations. For LU factorization, we
delay the application of panel pivots, compose the ordered swaps into a
conflict-free permutation, and apply it through an ordered gather/scatter
kernel pair. For Cholesky factorization, we preserve the triangular update
structure while regularizing selected SYRK and GEMM shapes. Recursive TRSM and
row-tiled GEMM further adapt the algorithms to the execution characteristics
of the Hygon BLAS stack. The implementation uses HIP, rocBLAS, and rocSOLVER
and preserves the conventional partial-pivoting LU and Cholesky factorizations.
<!-- Insert the final rocSOLVER comparison and accuracy results here. -->

**Index Terms:** dense linear algebra, LU factorization, Cholesky
factorization, double blocking, GPU, Hygon DCU

## I. Introduction

One-sided matrix factorizations are central to dense numerical linear algebra.
LU factorization with partial pivoting is used to solve general linear systems,
while Cholesky factorization provides a lower-cost and numerically stable
factorization for symmetric positive-definite matrices. Modern implementations
organize both algorithms around Level-3 BLAS operations, allowing the majority
of their floating-point work to execute as matrix multiplication or rank-k
updates [2]-[6].

The presence of BLAS-3 operations does not by itself guarantee high accelerator
utilization. A factorization advances through a sequence of panels. The local
factorization of each panel lies on the critical path and includes operations
with limited parallelism or irregular data movement. The update associated
with that panel contains more arithmetic, but its dimensions are determined by
the panel width. A small block limits the work assigned to each
dependency-bound operation at the cost of narrow update shapes. A large block
makes the update more regular but increases panel cost and delays the next
factorization step.

This conflict becomes more pronounced when a factorization is expressed as a
composition of vendor-library calls. The library controls the execution of an
individual POTRF, GETRF, TRSM, SYRK, or GEMM operation, but the application
controls the dimensions and order of those calls. Consequently, algorithm-level
blocking and library-level kernel optimization address different parts of the
execution. Factorization performance therefore depends on exposing suitable
operation shapes without unnecessarily increasing the dependency-bound portion
of the algorithm.

Double blocking addresses this conflict by introducing an outer width $B$ and
an inner width $b$, with $b \leq B$. The inner width bounds local
factorization, while the outer width determines the unit of panel preparation
and delayed update. The concept has recently been applied to one-sided
factorizations in OSLA [1]. However, realizing the method on the Hygon software
stack raises two implementation issues that are not addressed by direct API
substitution. Our implementation expresses LU pivoting without a cooperative
grid-wide barrier and selects the resulting GEMM, SYRK, and TRSM call shapes
for rocBLAS on K100_AI.

We present double-blocked LU and Cholesky implementations designed for this
environment. Both algorithms use a panel-oriented schedule that prepares only
the next required inner or outer panel. Their specializations differ where the
mathematics differs. LU accumulates partial-pivoting information and delays
external row movement. Cholesky separates the symmetric diagonal update from
the rectangular update below it. A third execution granularity is then used to
partition an unfavorable reduction or row dimension without changing the
factorization schedule.

The contributions of this paper are as follows:

- We analyze how the panel-update conflict appears in LU and Cholesky
  factorization on Hygon DCUs and use the analysis to derive a common
  two-level blocking schedule.
- We implement a double-blocked LU factorization that retains standard partial
  pivoting while batching external row permutations. A two-kernel
  gather/scatter design uses stream ordering instead of cooperative grid
  synchronization.
- We implement a double-blocked Cholesky factorization with separate
  inner-panel and outer-panel updates, K-blocked SYRK, and conditional M-tiled
  GEMM.
- We integrate recursive TRSM, reusable permutation workspaces, and
  shape-aware BLAS dispatch into a HIP implementation based on rocBLAS and
  rocSOLVER.
- We evaluate the proposed methods against the official rocSOLVER GETRF and
  POTRF routines using end-to-end scaling, stage-level ablation, and numerical
  residuals.
  <!-- Add measured speedups after the unified evaluation. -->

The remainder of this paper is organized as follows. Section II reviews the
factorizations and their update schedules. Section III analyzes the conflict
between panel factorization, matrix updates, and LU row movement. Section IV
presents the double-blocked algorithms. Section V describes the
Hygon-oriented optimizations, and Section VI summarizes the implementation.
Section VII is reserved for the unified experimental evaluation. Section VIII
discusses related work, and Section IX concludes the paper.

## II. Background

### A. One-Sided Matrix Factorizations

#### 1) LU Factorization

For a nonsingular matrix $A \in \mathbb{R}^{n \times n}$, LU factorization
with partial pivoting computes

$$
PA = LU,
$$

where $P$ is a permutation matrix, $L$ is unit lower triangular, and $U$ is
upper triangular. A blocked step factors a tall panel, applies the resulting
row pivots, solves for a block row of $U$, and updates the remaining matrix by
GEMM. Its leading-order arithmetic cost is

$$
F_{\mathrm{LU}}(n) = \frac{2}{3}n^3 + O(n^2).
$$

Partial pivoting couples panel factorization to matrix-wide data movement. The
pivot selected for one panel row can refer to any active row below it. Although
the pivot sequence is ordered and data-dependent, a matrix region that is not
consumed until the panel completes needs only the final composed permutation.

#### 2) Cholesky Factorization

For a symmetric positive-definite matrix $A$, Cholesky factorization computes

$$
A = LL^T,
$$

where $L$ is lower triangular. With

$$
A =
\begin{bmatrix}
A_{11} & A_{21}^T \\
A_{21} & A_{22}
\end{bmatrix},
$$

one block step is

$$
L_{11} = \operatorname{chol}(A_{11}),
$$

$$
L_{21} = A_{21}L_{11}^{-T},
$$

and

$$
A_{22} \leftarrow A_{22} - L_{21}L_{21}^T.
$$

The three operations map to POTRF, TRSM, and SYRK. Cholesky requires no
pivoting and has the leading-order cost

$$
F_{\mathrm{Chol}}(n) = \frac{1}{3}n^3 + O(n^2).
$$

The absence of row exchanges makes its control flow more regular than LU, but
the lower-triangular destination distinguishes its diagonal SYRK update from
the rectangular GEMM updates used elsewhere in the panel.

### B. Left-Looking and Right-Looking Schedules

A right-looking algorithm uses a newly factorized panel to update the complete
trailing matrix. This exposes a large update immediately but repeatedly touches
matrix regions that will not be factorized until much later. A left-looking
algorithm accumulates contributions from previously computed panels when the
next panel is required. It postpones updates to regions that are not yet needed
but places a wider-history operation on the panel-preparation path.

The algorithms in this paper use a panel-oriented left-looking schedule. After
an inner block has been factorized, only the next inner panel inside the current
outer block is prepared. At the outer boundary, only the next outer panel is
prepared. This schedule is particularly useful for LU because it also defines
when delayed pivots must become visible.

### C. Hygon DCU Software Stack

The target platform is a Hygon K100_AI DCU programmed through HIP. rocBLAS
provides GEMM, SYRK, and TRSM, while rocSOLVER provides the local GETRF and
POTRF operations. These routines are treated as external library primitives;
the analysis makes no assumption about their internal blocking. Our algorithms
control the dimensions, order, and data regions supplied to the library.

## III. Performance Analysis and Design Motivation

### A. The Panel-Update Conflict

Let $b$ be the panel width in a conventional blocked factorization. Its
execution time can be written schematically as

$$
T_{\mathrm{single}}(b) =
T_{\mathrm{panel}}(b) +
\sum_q \frac{F_q}{P_q(\mathcal{S}_q(b))} +
T_{\mathrm{transform}}(b),
$$

where $F_q$ is the arithmetic work of update $q$,
$P_q(\mathcal{S}_q)$ is the measured throughput for shape
$\mathcal{S}_q$, and $T_{\mathrm{transform}}$ includes non-BLAS work such as
LU row permutations.

No monotonic relationship is expected between $b$ and total factorization
time. Increasing $b$ reduces the number of panels and can provide wider update
shapes, but it also increases the work and latency assigned to each local
factorization. In LU, the aggregate work performed inside tall GETRF panels
grows approximately as $\Theta(n^2b)$ when $b \ll n$. In Cholesky, larger
diagonal blocks increase the aggregate local POTRF work and postpone the next
block column. A single block width therefore represents a compromise between
the two stages.

Double blocking introduces $B$ and $b$:

$$
T_{\mathrm{double}}(B,b) =
T_{\mathrm{panel}}(B,b) +
T_{\mathrm{update}}(B,b) +
T_{\mathrm{transform}}(B,b) +
T_{\mathrm{schedule}}(B,b).
$$

The purpose of $B$ is not to enlarge the local GETRF or POTRF call. It enlarges
the region over which previously computed factors are accumulated, while $b$
continues to bound local factorization.

> **[Figure 1 required: panel-update conflict.]** Plot, on the same K100_AI
> platform, panel time and update throughput as the single-level block width
> increases. The figure must establish the opposing trends that motivate
> independent $B$ and $b$ values. Populate it only from the unified experiment
> run, using the $b=B$ degeneration or isolated panel/update shapes rather than
> introducing another end-to-end library baseline.

### B. Update Geometry

The geometry of the updates follows directly from the panel-oriented schedule.
Suppose an inner panel ends at column $Q$ and the current outer panel begins at
$K$. Preparing the next panel requires a GEMM whose reduction dimension is
$Q-K$, whose output width is at most $b$, and whose row dimension can approach
$n-Q$. The operation is therefore generally rectangular even when the original
matrix is square.

At an outer boundary, the next panel has width at most $B$. For Cholesky, its
diagonal part is updated by SYRK and the region below it is updated by GEMM.
For LU, a triangular solve prepares the upper part and GEMM updates the lower
part. These outer operations contain most of the history accumulated so far
and can be wider than an inner panel when $B>b$.

The relevant performance quantity is thus not an isolated square-GEMM peak but
the throughput of the exact factorization shapes, including their leading
dimensions. Section VII will report this shape characterization on K100_AI.

> **[Figure 2 required: exact-shape primitive performance.]** Report GEMM,
> SYRK, and TRSM throughput for the shapes generated by Algorithms 2 and 3,
> including leading dimensions. Mark the shape cliffs that motivate $T_M$,
> $T_K$, and the recursive TRSM leaf. This figure belongs to the unified
> evaluation rather than the preliminary tuning history.

### C. Pivoting as a Data-Movement Bottleneck

For an LU panel of width $b$, partial pivoting produces $b$ ordered row swaps.
Applying each swap immediately to a wide external region requires many
large-stride accesses in column-major storage and launches work at the
granularity of a single exchange. The arithmetic cost is negligible, but the
operation introduces data movement and ordering constraints.

The ordered swaps define a final permutation after the panel completes.
Therefore, if an external matrix region is not read during the panel
factorization, the swaps can be composed first and applied to that region in
two streaming passes: one read pass and one write pass. Double blocking
increases the scope over which this delayed application is legal. It does not
change pivot selection; it changes only when the resulting permutation becomes
visible.

### D. Design Requirements

The analysis motivates four design constraints:

1. Local GETRF and POTRF dimensions must remain independent of the width used
   for coarse-grained updates.
2. Only the next data-dependent panel should be updated.
3. LU must preserve the exact order of partial-pivoting swaps while batching
   their application to independent regions.
4. A device-specific execution tile must be allowed to regularize a BLAS shape
   without redefining $B$ or $b$.

## IV. Double-Blocking Methods

### A. Common Two-Level Schedule

The matrix is partitioned into outer panels of width $B$, each containing inner
blocks of width $b$. Let $I$ be the first column of the current outer panel,
$I_e=\min(I+B,n)$, and $J$ the first column of the current inner block. The
common schedule is shown in Algorithm 1. Matrix ranges are half-open: for
example, $A[I:J,K:Q]$ excludes rows $J$ and columns $Q$.

> **[Figure 3 required: double-blocking workflows.]** Use two aligned
> subfigures for DB-LU and DB-Cholesky. In both, label the outer width $B$,
> inner width $b$, current inner block, next inner panel, next outer panel,
> computed history, and untouched region. In the LU subfigure, mark the
> regions receiving inner and outer pivot permutations. In the Cholesky
> subfigure, distinguish POTRF, TRSM, inner GEMM, outer SYRK, and outer GEMM.

~~~text
Algorithm 1: Common double-blocking schedule

Input: A, n, outer width B, inner width b

for I = 0, B, 2B, ... while I < n do
    I_e = min(I + B, n)

    for J = I, I + b, ... while J < I_e do
        w = min(b, I_e - J)
        factor the inner block beginning at J
        apply any factorization-specific transformation needed inside [I, I_e)
        prepare the next inner panel, if one exists
    end for

    apply any transformations delayed to the outer boundary
    prepare the next outer panel, if one exists
end for
~~~

The second blocking level changes the temporal grouping of updates while
preserving the factorization dependencies and leading-order arithmetic.

### B. Double-Blocked LU Factorization

At inner position $J$, DB-LU factors the tall panel
$A_{J:n,J:J+w}$ with partial pivoting. The panel operation returns local
factors and a pivot sequence $p_J$. The corresponding row transformation is
applied to two regions inside the current outer panel: columns already computed
since $I$ and columns that remain to be prepared before $I_e$. The left history
before $I$ and columns after $I_e$ are not needed by the next inner step and
can remain unpermuted.

rocSOLVER reports each pivot as a one-based row index relative to the active
submatrix beginning at $J$. DB-LU adds the panel offset before storing the
entry in the final LAPACK-compatible pivot vector.

Let $Q=J+w$ and $R=\min(Q+b,I_e)$. If $Q<I_e$, the next inner panel is
prepared by

$$
U_{I:Q,Q:R} =
L_{I:Q,I:Q}^{-1}A_{I:Q,Q:R},
$$

followed by

$$
A_{Q:n,Q:R} \leftarrow
A_{Q:n,Q:R} - L_{Q:n,I:Q}U_{I:Q,Q:R}.
$$

At $I_e$, pivots generated in the completed outer panel are applied to the
earlier column history. The pivots accumulated from the beginning of the
factorization are applied to the next outer panel. Its upper part is then
obtained from

$$
U_{0:I_e,I_e:R_e} =
L_{0:I_e,0:I_e}^{-1}A_{0:I_e,I_e:R_e},
$$

where $R_e=\min(I_e+B,n)$, and its lower part is updated by

$$
A_{I_e:n,I_e:R_e} \leftarrow
A_{I_e:n,I_e:R_e}
-L_{I_e:n,0:I_e}U_{0:I_e,I_e:R_e}.
$$

~~~text
Algorithm 2: Double-blocked LU with partial pivoting

Input: A, n, B, b
Output: packed L and U, pivot vector ipiv

for I = 0, B, 2B, ... while I < n do
    I_e = min(I + B, n)

    for J = I, I + b, ... while J < I_e do
        w = min(b, I_e - J)
        Q = J + w

        p_J = GETRF(A[J:n, J:J+w])
        record p_J in ipiv
        apply p_J to A[J:n, I:J] and A[J:n, Q:I_e]

        if Q < I_e then
            R = min(Q + b, I_e)
            U[I:Q, Q:R] = TRSM(L[I:Q, I:Q], A[I:Q, Q:R])
            A[Q:n, Q:R] -= L[Q:n, I:Q] * U[I:Q, Q:R]
        end if
    end for

    apply the current outer pivots to A[I:n, 0:I]
    apply all accumulated pivots to A[0:n, I_e:min(I_e+B,n)]

    if I_e < n then
        prepare the upper next-outer panel with recursive TRSM
        update the lower next-outer panel with GEMM
    end if
end for
~~~

### C. Double-Blocked Cholesky Factorization

At inner position $J$, DB-Cholesky factors the diagonal block
$A_{J:J+w,J:J+w}$ and computes the complete block column below it:

$$
L_{J+w:n,J:J+w}
=
A_{J+w:n,J:J+w}L_{J:J+w,J:J+w}^{-T}.
$$

Let $Q=J+w$ and $R=\min(Q+b,I_e)$. The next inner panel is prepared by
accumulating contributions from the history inside the current outer panel:

$$
A_{Q:n,Q:R} \leftarrow
A_{Q:n,Q:R}
-L_{Q:n,I:Q}L_{Q:R,I:Q}^T.
$$

At the outer boundary, let $R_e=\min(I_e+B,n)$. Symmetry divides preparation
of the next outer panel into a diagonal and an off-diagonal operation:

$$
A_{I_e:R_e,I_e:R_e} \leftarrow
A_{I_e:R_e,I_e:R_e}
-L_{I_e:R_e,0:I_e}L_{I_e:R_e,0:I_e}^T,
$$

and

$$
A_{R_e:n,I_e:R_e} \leftarrow
A_{R_e:n,I_e:R_e}
-L_{R_e:n,0:I_e}L_{I_e:R_e,0:I_e}^T.
$$

The first update is SYRK and writes only the lower triangle. The second is
GEMM and writes the rectangular region below the diagonal block.

The inner update is evaluated as a rectangular GEMM. Its top square is computed
in full, but the following POTRF reads only the lower triangle; the additional
upper-triangular values therefore do not affect the factorization.

~~~text
Algorithm 3: Double-blocked Cholesky factorization

Input: symmetric positive-definite A, n, B, b
Output: lower triangular L stored in A

for I = 0, B, 2B, ... while I < n do
    I_e = min(I + B, n)

    for J = I, I + b, ... while J < I_e do
        w = min(b, I_e - J)
        Q = J + w

        POTRF(A[J:J+w, J:J+w])
        A[Q:n, J:J+w] =
            A[Q:n, J:J+w] * inverse(transpose(A[J:J+w, J:J+w]))

        if Q < I_e then
            R = min(Q + b, I_e)
            A[Q:n, Q:R] -=
                A[Q:n, I:Q] * transpose(A[Q:R, I:Q])
        end if
    end for

    if I_e < n then
        R_e = min(I_e + B, n)
        update A[I_e:R_e, I_e:R_e] with SYRK
        update A[R_e:n, I_e:R_e] with GEMM
    end if
end for
~~~

### D. Algebraic Equivalence and Complexity

DB-Cholesky changes only the grouping of rank-k contributions. Every block
product contributing to a destination is applied before that destination is
factorized. Therefore, in exact arithmetic, Algorithm 3 produces the same
lower triangular factor as the corresponding blocked Cholesky factorization.

DB-LU preserves the pivot selected by each GETRF panel and preserves the order
of those swaps. Delayed application is restricted to columns that are not read
before the composed permutation is applied. Thus, delaying a permutation does
not alter later pivot searches or update operands. Algorithm 2 remains a
partial-pivoting LU factorization.

The leading arithmetic costs remain $2n^3/3$ for LU and $n^3/3$ for
Cholesky. Tiling an update changes its partition but not its leading work.
Pivot composition and application add $O(n^2)$ integer and data-movement work.
The permutation workspace uses $O(nb_p)$ values, where $b_p$ is the number of
columns processed in one permutation tile.

When $b=B$, the inner loop contains one block and the method degenerates to a
single-level panel schedule. This property provides a controlled way to
measure whether the second blocking level is beneficial.

## V. Optimizations for Hygon DCUs

The two-level schedule determines when each matrix region becomes available,
but it does not prescribe how an individual permutation, triangular solve, or
matrix update is executed. This section describes the operation-level choices
used in the Hygon implementation.

### A. Delayed and Batched Pivot Application

The raw pivot vector returned by a GETRF panel encodes an ordered sequence of
swaps. DB-LU copies this vector to the host and applies the swaps to an identity
index array. If the resulting permutation maps target row $d_t$ to source row
$s_t$, fixed points are removed and the compact source and target arrays are
uploaded to the device.

> **[Figure 4 required: delayed pivot application.]** The left half should
> contrast immediate row swaps after every inner panel with the regions updated
> by DB-LU at inner and outer boundaries. The right half should show the
> ordered pivot sequence, CPU-composed source/target mapping, gather kernel,
> temporary tile, and scatter kernel. Emphasize that the two kernel launches
> form the global read-before-write barrier.

A naive in-place parallel permutation is unsafe because one destination can
overwrite a source needed by another thread. OSLA resolves this dependency
with a read-synchronize-write cooperative kernel [1]. The Hygon implementation
uses two separate HIP kernels instead. For a tile of columns, the gather kernel
computes

$$
T_{c,t}=A_{s_t,c},
$$

and a subsequent scatter kernel computes

$$
A_{d_t,c}=T_{c,t}.
$$

The kernel-launch boundary supplies the required global ordering: all reads
complete before any destination is overwritten. Column tiling bounds temporary
storage and allows one workspace to serve both inner and outer permutations.

### B. Recursive Triangular Solve

The outer LU panel requires a unit-lower-triangular solve whose triangular
dimension grows with the factorization. A single large TRSM may correspond to
a low-throughput library shape. We recursively partition

$$
L =
\begin{bmatrix}
L_{11} & 0 \\
L_{21} & L_{22}
\end{bmatrix},
\qquad
X =
\begin{bmatrix}
X_1 \\
X_2
\end{bmatrix}.
$$

The recursion evaluates

$$
X_1=L_{11}^{-1}B_1,
$$

$$
B_2 \leftarrow B_2-L_{21}X_1,
$$

and

$$
X_2=L_{22}^{-1}B_2.
$$

Only leaf nodes call rocBLAS TRSM; the update between leaves is GEMM. The leaf
size $b_t$ is independent of both $B$ and $b$.

### C. K-Blocked SYRK

The diagonal update in DB-Cholesky has the form

$$
C \leftarrow C-AA^T.
$$

When the reduction dimension is large, the implementation partitions the
columns of $A$ into chunks of width $T_K$:

$$
C \leftarrow C-\sum_q A_qA_q^T.
$$

Each chunk is submitted as a lower-triangular SYRK and accumulated into the
same destination. The mathematical work is unchanged, apart from floating-point
reassociation. This execution tile is used only for the outer diagonal update;
it does not redefine the Cholesky block structure.

### D. M-Tiled GEMM

Both factorizations produce GEMMs with a large row dimension and a panel-width
column dimension. DB-LU uses

$$
C \leftarrow C-AB,
$$

whereas DB-Cholesky uses

$$
C \leftarrow C-AB^T.
$$

In either case, row slabs of the destination are independent. DB-LU therefore
uses

$$
C_i \leftarrow C_i-A_iB,
$$

and DB-Cholesky uses

$$
C_i \leftarrow C_i-A_iB^T.
$$

Each $C_i$ has at most $T_M$ rows. DB-LU applies this transformation to the
inner-panel update. DB-Cholesky enables it only when the row and reduction
dimensions exceed measured thresholds, avoiding additional launch overhead for
cases in which tiling provides no measured benefit.

The values of $T_K$, $T_M$, and the dispatch thresholds are selected from
primitive shape measurements rather than from the outer block size. This
keeps two decisions separate: $B$ and $b$ express the dependency graph, while
the execution tiles describe the call shapes submitted to the target BLAS
implementation.

## VI. Implementation

### A. Library Mapping and Execution Order

The implementation is written in C++ and HIP. Matrices use column-major storage
with a configurable leading dimension. Local factorization calls map to
rocsolver_sgetrf or rocsolver_spotrf in FP32 and to their FP64 counterparts.
Matrix updates and triangular solves map to rocBLAS. The double-blocked
factorizations are denoted DB-GETRF and DB-POTRF in the implementation.

All GPU library calls and custom kernels are submitted to the stream associated
with the rocBLAS handle. To isolate the effects of blocking, pivot batching,
and operation shaping, the implementation uses a single ordered stream.
Multi-stream look-ahead is outside the scope of this paper.

### B. Addressing and Workspace Management

Block addresses are formed with the column-major mapping

$$
\operatorname{ptr}(A,i,j)=A+j\cdot lda+i.
$$

Custom matrix-index calculations use size-aware arithmetic. Vendor routine
dimensions retain the integer interface required by rocBLAS and rocSOLVER.

The pivot source, target, and temporary buffers are allocated once and reused
throughout DB-LU. The temporary buffer stores at most one $b_p$-column
permutation tile. Memory allocation and matrix generation are excluded from
the timed factorization.

### C. Measurement and Numerical Validation

End-to-end timing uses HIP events on the execution stream. Before every timed
method, the same deterministically generated input matrix is restored on the
device. All individual iteration times are retained; the evaluation reports
the arithmetic mean, median, sample standard deviation, minimum, and maximum.
GETRF throughput uses $2n^3/3$ operations and POTRF throughput uses $n^3/3$
operations.

Numerical validation is performed outside the timed region. Scaling runs use
a deterministic set of 4096 entries to estimate the relative residuals of
$PA-LU$ and $A-LL^T$. Separate correctness runs explicitly form $L$ and $U$
or $L$ and reconstruct the complete product with FP64 rocBLAS GEMM. They
report the full Frobenius residual at representative power-of-two and
non-power-of-two sizes. The final evaluation records both policies and does
not compare a full residual from one method with a sampled residual from
another.

## VII. Experimental Evaluation

### A. Experimental Setup

<!-- Complete after the source commit, DTK version, device, timing policy,
     matrix generator, and fixed parameter policy are frozen. -->

### B. End-to-End Performance

<!-- Add rocSOLVER versus DB-LU and rocSOLVER versus DB-Cholesky scaling. -->

### C. Ablation Study

<!-- LU: basic double blocking, batched pivots, and M-tiled inner GEMM.
     Cholesky: basic double blocking, K-blocked SYRK, and adaptive M tiling. -->

### D. Stage and Shape Analysis

<!-- Add exact-shape BLAS measurements and end-to-end stage breakdowns. -->

### E. Numerical Accuracy

<!-- Add backward errors and full residual checks at representative sizes. -->

## VIII. Related Work

Blocked LU and Cholesky algorithms use Level-3 BLAS to improve data reuse [2],
[3]. GPU libraries and accelerator-oriented systems extend these formulations
through hybrid execution, inner blocking, and GPU-resident kernels [5], [6].
At the application level, their panel and update dimensions still have to be
selected for the target device and library.

OSLA applies double blocking to LU, Cholesky, and QR factorization on NVIDIA
GPUs and combines the structure with specialized panel and mixed-precision
optimizations [1]. The present work follows its two-level blocking strategy but
studies a different software and hardware environment. We focus
on native-precision LU and Cholesky on K100_AI. In particular, our LU
implementation replaces OSLA's cooperative read-synchronize-write permutation
with ordered gather and scatter HIP kernels and combines it with rocBLAS shape
regularization.

Communication-avoiding LU reduces panel communication by modifying pivot
selection or organizing it as a reduction [7]. DB-LU does not change the pivot
rule. It retains standard partial pivoting and delays only the application of
the resulting permutation to matrix regions that are not yet data-dependent.

rocBLAS and rocSOLVER provide the vendor operations used by our implementation
[8], [9]. They are also the basis of the official comparison in Section VII.
Our contribution lies at the factorization schedule and data-transformation
level rather than in replacing the vendor GEMM or local factorization kernels.

## IX. Conclusion

This paper described double-blocked LU and Cholesky factorizations for Hygon
DCUs. The design is derived from the conflict between narrow
dependency-bound panels and wide matrix updates. An outer panel width controls
coarse-grained accumulation and data movement, while an inner width bounds
local factorization. LU uses the resulting delay to batch partial-pivoting row
movement, and Cholesky uses it to separate symmetric and rectangular updates.
Recursive TRSM, K-blocked SYRK, and M-tiled GEMM provide an additional
execution granularity without changing the mathematical partition.

The resulting HIP implementation retains the standard numerical
factorizations and uses rocBLAS and rocSOLVER as computational building blocks.
<!-- Insert the final performance and accuracy conclusion after evaluation. -->

## References

[1] L. Shi, R. Zhan, G. Zou, H. Wang, G. Min, H. Duan, and S. Zhang,
"OSLA: A High Performance One-Sided Linear Algebra Library on GPU
Architectures," IEEE Transactions on Parallel and Distributed Systems, 2026,
doi: 10.1109/TPDS.2026.3692274.

[2] E. Anderson, Z. Bai, C. Bischof, L. S. Blackford, J. Demmel,
J. Dongarra, J. Du Croz, A. Greenbaum, S. Hammarling, A. McKenney, and
D. Sorensen, LAPACK Users' Guide, 3rd ed. Philadelphia, PA, USA: SIAM, 1999.

[3] J. W. Demmel, N. J. Higham, and R. S. Schreiber, "Stability of block LU
factorization," Numerical Linear Algebra with Applications, vol. 2, no. 2,
pp. 173-190, 1995.

[4] N. J. Higham, "Cholesky factorization," Wiley Interdisciplinary Reviews:
Computational Statistics, vol. 1, no. 2, pp. 251-254, 2009.

[5] A. Haidar, A. Abdelfattah, S. Tomov, and J. Dongarra, "High-performance
Cholesky factorization for GPU-only execution," in Proceedings of GPGPU-10,
2017, pp. 42-52.

[6] E. Agullo, C. Augonnet, J. Dongarra, M. Faverge, J. Langou, H. Ltaief,
and S. Tomov, "LU factorization for accelerator-based systems," in
Proceedings of AICCSA, 2011, pp. 217-224.

[7] J. Demmel, L. Grigori, M. Hoemmen, and J. Langou,
"Communication-optimal parallel and sequential QR and LU factorizations:
theory and practice," SIAM Journal on Scientific Computing, vol. 34, no. 1,
pp. A206-A239, 2012.

[8] AMD ROCm Software, "rocBLAS documentation." [Online]. Available:
https://rocm.docs.amd.com/projects/rocBLAS/

[9] AMD ROCm Software, "rocSOLVER documentation." [Online]. Available:
https://rocm.docs.amd.com/projects/rocSOLVER/
