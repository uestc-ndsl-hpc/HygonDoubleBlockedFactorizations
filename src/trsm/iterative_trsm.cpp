#include "trsm/iterative_trsm.h"
#include <hip/hip_runtime.h>

extern "C" {

void iterative_trsm_right_looking(
    rocblas_handle handle,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb)
{
    if (n <= 0 || m <= 0) return;

    float one = 1.0f;
    float neg_one = -1.0f;

    // Right-TRSM: split m (triangular dimension is on RIGHT)
    for (int i = 0; i < m; i += nb) {

        int current_nb = (m - i < nb) ? (m - i) : nb;

        // ======================================================
        // 1. Solve panel:
        //    X_i * L_ii^T = B_i
        // ======================================================
        rocblas_strsm(
            handle,
            rocblas_side_right,              // ✔ FIX: right side
            rocblas_fill_lower,
            rocblas_operation_transpose,     // ✔ FIX: match L^T
            rocblas_diagonal_non_unit,
            n, current_nb, &one,
            A + (size_t)i * lda + i, lda,
            B + (size_t)i * ldb, ldb);

        // ======================================================
        // 2. Update trailing matrix:
        //    B_j -= X_i * L_ji^T
        // ======================================================
        if (i + current_nb < m) {

            int remaining = m - (i + current_nb);

            rocblas_sgemm(
                handle,
                rocblas_operation_none,
                rocblas_operation_transpose,  // ✔ FIX: L^T consistency
                n, remaining, current_nb,
                &neg_one,

                B + (size_t)i * ldb, ldb,     // X_i
                A + (size_t)i * lda + (i + current_nb), lda, // L_ji

                &one,
                B + (size_t)(i + current_nb) * ldb, ldb);
        }
    }

    return;
}

}
