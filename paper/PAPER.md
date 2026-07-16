# Double-Blocked LU and Cholesky Factorizations on Hygon DCUs

**Authors:** To be added

**Affiliation:** To be added

## Abstract

Blocked one-sided factorizations alternate between dependency-bound panel
operations and compute-intensive matrix updates. On a GPU, these two stages
favor different granularities: a narrow panel limits the critical path, whereas
a wide update exposes more regular Level-3 BLAS operations. A conventional
single-level blocked algorithm couples both requirements to one block size and
does not allow them to be tuned independently. A double-blocked formulation
separates these requirements through an outer width for coarse-grained panel
preparation and an inner width for local GETRF or POTRF operations. In the LU
path, panel pivots are delayed where dependencies permit, composed into a
conflict-free permutation, and applied by an ordered gather/scatter kernel
pair. In the Cholesky path, the lower-triangular update is retained while inner
and outer panel preparation remain distinct. Iterative blocked TRSM, K-blocked
SYRK, and M-tiled GEMM are treated as general execution transformations rather
than Hygon-specific algorithms. Exact-shape measurements on the K100_AI
rocBLAS stack define the evidence required to select their inclusion and
parameters. The corresponding HIP mapping retains conventional
partial-pivoting LU and Cholesky definitions.
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
updates.

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
factorizations in OSLA. Mapping the method to the Hygon software stack raises
two implementation questions that are not answered by direct API substitution:
LU permutation requires global read-before-write ordering, and the generated
GEMM, SYRK, and TRSM shapes may interact differently with the target rocBLAS
backend.

The algebraic forms of blocked TRSM, K-blocked SYRK, and M-tiled GEMM are
portable across GPU platforms. Consequently, their equations do not establish
K100_AI specificity. Platform-dependent conclusions must instead come from the
measured throughput surfaces of the exact generated shapes, the crossover
thresholds derived from those measurements, and the contribution of each
stage to end-to-end time.

The design consists of LU and Cholesky specializations under one panel-oriented
schedule. Only the next required inner or outer panel is prepared. LU
accumulates partial-pivoting information and delays external row movement,
whereas Cholesky separates the symmetric diagonal update from the rectangular
update below it. Additional execution tiles partition unfavorable triangular,
reduction, or row dimensions without changing the factorization schedule.

The technical scope comprises the following elements:

- A common two-level schedule derived from the panel-update granularity
  conflict in LU and Cholesky factorization.
- A partial-pivoting DB-LU formulation with delayed external permutations and
  an ordered gather/scatter mapping that avoids cooperative grid
  synchronization.
- A DB-Cholesky formulation with distinct inner- and outer-panel updates and
  separate symmetric-diagonal and rectangular-off-diagonal preparation.
- A HIP execution mapping that parameterizes general TRSM, SYRK, and GEMM
  transformations independently of the two factorization block widths.
- An evaluation protocol that connects K100_AI exact-shape measurements to
  fixed dispatch rules, stage-level effects, end-to-end rocSOLVER comparisons,
  and numerical residuals.
  <!-- Add numerical results only after the unified evaluation. -->

The remainder of this paper is organized as follows. Section II reviews the
factorizations and their update schedules. Section III analyzes the conflict
between panel factorization, matrix updates, and LU row movement. Section IV
presents the double-blocked algorithms. Section V describes the
operation-level mapping and selection rules, and Section VI specifies the
reference implementation. Section VII defines the experimental evaluation.
Section VIII
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

### B. Panel-Oriented Deferred Updates

A panel-oriented schedule need not update every region as soon as a new factor
becomes available. Contributions may remain deferred until their destination
panel becomes the next dependency on the factorization path. This reduces
updates to matrix regions that are not yet required, while increasing the
history accumulated by each panel-preparation operation.

The double-blocked schedule prepares only the next inner panel inside the
current outer block. At an outer boundary, it prepares only the next outer
panel. For LU, the same boundaries determine when a delayed permutation must
become visible to subsequent pivot searches and updates.

### C. Hygon DCU Software Stack

The target platform is a Hygon K100_AI DCU programmed through HIP. rocBLAS
provides GEMM, SYRK, and TRSM, while rocSOLVER provides the local GETRF and
POTRF operations. These routines are treated as external library primitives;
the analysis makes no assumption about their internal blocking. The
double-blocked schedules control the dimensions, order, and data regions
supplied to the library.

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
> increases. The figure should quantify the opposing effects associated with
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
dimensions. Section VII will report this shape characterization on K100_AI and
connect it to the time of the corresponding factorization stages.

> **[Figure 2 required: exact-shape primitive performance.]** Report GEMM,
> SYRK, and TRSM throughput for the exact shapes generated by Algorithms 2 and
> 3, including leading dimensions. Compare each untiled vendor call with its
> iterative or tiled counterpart across the dimensions reached during the
> factorization. Derive candidate values of $b_t$, $T_K$, $T_M$, and the
> crossover thresholds from a separate tuning set. The figure characterizes
> K100_AI behavior; it does not make the underlying transformations
> K100_AI-specific. A transformation with no stable advantage must not be
> retained solely to strengthen the method narrative.

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
4. An execution tile selected for the target BLAS backend must be allowed to
   regularize a call shape without redefining $B$ or $b$.

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

At $I_e$, pivots generated in the current outer panel are applied to the
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
        prepare the upper next-outer panel with the selected TRSM policy
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
        A[Q:n, J:J+w] = right_side_TRSM(
            A[J:J+w, J:J+w], A[Q:n, J:J+w])

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

## V. Operation-Level Mapping

The two-level schedule determines when each matrix region becomes available,
but it does not prescribe how an individual permutation, triangular solve, or
matrix update is executed. The pivot mapping below is specific to the LU data
dependency, whereas blocked TRSM and tiled SYRK/GEMM are general BLAS-level
transformations. None of the latter is treated as a new or Hygon-specific
algorithm. Their role is to expose alternative call shapes whose usefulness
must be established empirically on the target stack.

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
with a read-synchronize-write cooperative kernel. The HIP mapping instead uses
two ordered kernels. For a tile of columns, the gather kernel
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

### B. Iterative Blocked Triangular Solve

The outer LU panel requires a unit-lower-triangular solve whose triangular
dimension grows with the factorization. A single large TRSM may correspond to
a low-throughput library shape. The iterative alternative partitions the
triangular dimension into fixed-width blocks of at most $b_t$. At block
position $p$, let $L_{pp}$ be the current diagonal block, $B_p$ the
corresponding block rows of the right-hand side, and $B_r$ the rows below it.
The blocked forward substitution evaluates

$$
X_p=L_{pp}^{-1}B_p,
$$

$$
B_r \leftarrow B_r-L_{rp}X_p.
$$

The blocks are processed from top to bottom. Each diagonal block invokes
rocBLAS TRSM, and each remaining-row update invokes GEMM before the next block
is solved. This fixed-order loop exposes regular BLAS-3 work without changing
the triangular dependency order. The execution block size $b_t$ is independent
of both $B$ and $b$.

### C. K-Blocked SYRK

The diagonal update in DB-Cholesky has the form

$$
C \leftarrow C-AA^T.
$$

The K-blocked alternative partitions the columns of $A$ into chunks of width
$T_K$:

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

In either case, row slabs of the destination are independent. The M-tiled
alternative for DB-LU evaluates

$$
C_i \leftarrow C_i-A_iB,
$$

and DB-Cholesky uses

$$
C_i \leftarrow C_i-A_iB^T.
$$

Each $C_i$ has at most $T_M$ rows. The candidate transformation applies to the
DB-LU and DB-Cholesky inner-panel updates. Thresholds on the row and reduction
dimensions prevent extra tiled launches from being introduced
unconditionally.

### E. Platform-Dependent Selection

The factorization widths $B$ and $b$ express the dependency graph. In contrast,
$b_t$, $T_K$, $T_M$, and the dispatch thresholds describe alternative call
shapes submitted to a BLAS implementation. Keeping the two parameter groups
separate prevents a favorable primitive shape from being misinterpreted as a
new factorization structure.

Selection on K100_AI proceeds through three direct controls: one large TRSM
versus iterative blocked TRSM, one full-reduction SYRK versus K-blocked SYRK,
and one untiled GEMM versus M-tiled GEMM. Candidate parameters are selected on
a tuning dataset and then frozen before scaling experiments. Retention requires
a stable reduction in the relevant factorization stage and a measurable
end-to-end effect beyond run-to-run variability. Otherwise the untiled vendor
operation remains the final policy. This process can establish a
platform-dependent choice, but not platform-specific novelty of the underlying
transformation.

## VI. Reference Implementation

### A. Library Mapping and Execution Order

The reference implementation uses C++ and HIP. Matrices use column-major
storage with a configurable leading dimension. Local factorization calls map to
rocsolver_sgetrf or rocsolver_spotrf in FP32 and to their FP64 counterparts.
Matrix updates and triangular solves map to rocBLAS. The double-blocked
factorizations are denoted DB-GETRF and DB-POTRF in the implementation.

All GPU library calls and custom kernels are submitted to the stream associated
with the rocBLAS handle. To isolate the effects of blocking, pivot batching,
and operation shaping, execution is restricted to a single ordered stream.
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
device. All individual iteration times are retained. The reported summary
statistics are the arithmetic mean, median, sample standard deviation, minimum,
and maximum.
GETRF throughput uses $2n^3/3$ operations and POTRF throughput uses $n^3/3$
operations.

Numerical validation remains outside the timed region. Scaling runs use
a deterministic set of 4096 entries to estimate the relative residuals of
$PA-LU$ and $A-LL^T$. Separate correctness runs explicitly form $L$ and $U$
or $L$ and reconstruct the complete product with FP64 rocBLAS GEMM. Full
Frobenius residuals are recorded at representative power-of-two and
non-power-of-two sizes. Sampled and full residuals remain separately labeled
and are not compared as equivalent quantities.

## VII. Experimental Evaluation

### A. Experimental Setup

The evaluation uses one K100_AI device and records the source commit, DTK,
compiler, rocBLAS, rocSOLVER, driver, device state, and fixed parameter policy.
Parameter tuning and reported scaling use disjoint result sets. Every timed
method restores the same input, and timing variability is reported together
with the central estimate.

<!-- Insert the frozen environment table and final parameter table. -->

### B. End-to-End Performance

The external comparison is limited to rocSOLVER GETRF versus DB-LU and
rocSOLVER POTRF versus DB-Cholesky. Runtime, effective throughput, variability,
and backward error are reported at the same matrix sizes. Internal tuning
controls are not presented as additional external baselines.

<!-- Insert the final scaling tables and figures. -->

### C. Ablation Study

Each operation-level transformation is evaluated against a direct control while
holding $B$, $b$, the input, and all unrelated parameters fixed. The LU
controls are immediate versus batched pivot application, one large outer TRSM
versus iterative blocked TRSM, and untiled versus M-tiled inner GEMM. The
Cholesky controls are one full-reduction outer SYRK versus K-blocked SYRK and
untiled versus conditionally M-tiled inner GEMM.

For each pair, the evaluation reports primitive throughput, corresponding
factorization-stage time, and end-to-end time. A primitive-only gain is not
sufficient evidence: if stage time does not decrease beyond run-to-run
variability, the transformation is excluded from the final policy and from the
performance contribution statement.

<!-- Insert one-factor-at-a-time ablation tables after parameter freeze. -->

### D. Stage and Shape Analysis

Figure 2 contains three exact-shape controls drawn from the factorization call
trace: direct and iterative LU outer TRSM, full and K-blocked Cholesky outer
SYRK, and untiled and M-tiled inner GEMM for both factorizations. The axes cover
the dimensions and leading dimensions reached by the scaling experiments,
rather than only square or peak-throughput cases. Candidate crossover points
are selected from a tuning run and then applied unchanged to the reported
scaling set.

Stage profiles map those primitive results back to panel factorization, pivot
movement, TRSM, SYRK, and GEMM time. This analysis can support statements about
the observed K100_AI and rocBLAS combination. Without a second architecture it
does not support claims that the transformations themselves are
architecture-specific.

<!-- Insert Figure 2, threshold table, and final stage breakdowns. -->

### E. Numerical Accuracy

Sampled backward errors accompany every scaling point. Full FP64-accumulated
Frobenius residuals are reported at one power-of-two and one non-power-of-two
size. Tiled and untiled variants use the same acceptance threshold, and a
performance result is discarded if its factorization reports nonzero `info` or
fails the residual criterion.

<!-- Insert the final accuracy table. -->

## VIII. Related Work

### A. Blocked and Multi-Level Factorizations

Conventional blocked LU and Cholesky formulations aggregate rank-1 updates into
Level-3 BLAS operations. Recursive formulations and inner blocking introduce
additional hierarchy to match cache levels or accelerator kernels. These
methods establish that panel and update granularities need not coincide. The
framework considered here makes two widths, $B$ and $b$, explicit and applies
the same panel-preparation schedule to both partial-pivoting LU and Cholesky.
The generic blocked TRSM and tiled update formulas are therefore part of the
implementation space rather than standalone algorithmic contributions.

### B. Hybrid and GPU-Resident Solver Libraries

MAGMA and related hybrid solvers assign latency-sensitive panel work to a CPU
while large trailing updates execute on a GPU. Vendor libraries such as
cuSOLVER and rocSOLVER, together with GPU-resident research implementations,
also provide factorizations that keep numerical work and matrix data on the
accelerator. The reference mapping in this study keeps GETRF, POTRF, TRSM,
SYRK, and GEMM on one DCU and uses rocSOLVER for local factorization. Host
participation in DB-LU is limited to composing pivot permutations; it does not
perform the numerical panel factorization.

### C. Tiled and Task-Based Factorizations

PLASMA, Chameleon, PaRSEC, StarPU, and related systems express dense
factorizations as directed acyclic graphs of tile operations. Their runtimes
can overlap panels and updates through look-ahead and schedule work across
heterogeneous resources. The present schedule does not introduce a task
runtime or multi-stream look-ahead. It uses a single ordered stream and defers
work at panel boundaries, making the effects of $B$, $b$, pivot movement, and
BLAS call shaping directly observable.

### D. Pivoting on Accelerators

Accelerator LU implementations have explored GPU-resident pivot search,
batched row interchange, fused panel kernels, tournament pivoting, and
communication-avoiding pivot rules. Strategies that change pivot selection
alter the numerical algorithm. DB-LU instead retains the ordered partial-pivot
sequence returned by GETRF and delays only its application to regions that are
not yet data-dependent. The gather-then-scatter mapping addresses the
read-before-write hazard of applying a composed permutation in parallel.

### E. GPU Cholesky Factorization

GPU Cholesky implementations range from hybrid blocked methods to GPU-only,
tiled, recursive, and task-scheduled formulations. Their performance commonly
depends on POTRF panel cost, TRSM shape, and the balance between SYRK and GEMM
updates. DB-Cholesky differs in the explicit separation between inner-panel
preparation and the diagonal and off-diagonal portions of the next outer panel.
K-blocked SYRK and M-tiled GEMM remain generic execution alternatives whose
value on K100_AI must be established by the controls in Section VII.

### F. Double Blocking and Vendor Primitives

OSLA is the closest structural precedent because it applies double blocking to
multiple one-sided factorizations on NVIDIA GPUs and combines the schedule with
specialized panel and mixed-precision mechanisms. The scope here is
native-precision LU and Cholesky on a Hygon K100_AI. The LU mapping uses ordered
gather and scatter HIP kernels rather than a cooperative permutation kernel,
while local factorization and BLAS-3 operations remain vendor primitives.

rocBLAS and rocSOLVER define the available primitive interface, and rocSOLVER
supplies the external end-to-end baselines. No assumption is made about the
internal blocking of those libraries. The positioning is therefore limited to
the factorization schedule, correctness-preserving pivot transformation, and
empirical selection of generic execution alternatives on the target stack; it
does not include a new GEMM kernel or a claim of universally superior GPU
tiling.

## IX. Conclusion

Double blocking separates the width of local factorization from the width of
coarse-grained panel preparation. Within this structure, LU delay boundaries
also delimit legal batching of partial-pivoting row movement, while Cholesky
separates symmetric diagonal updates from rectangular off-diagonal updates.
When retained by the platform controls, iterative blocked TRSM, K-blocked SYRK,
and M-tiled GEMM act as execution tiles inside the two-level schedule rather
than as additional factorization levels or device-specific algorithms.

The HIP mapping retains the standard numerical definitions and uses rocBLAS
and rocSOLVER as computational building blocks.
<!-- Insert the final performance and accuracy conclusion after evaluation. -->
