#include "potrf/recursive_potrf.h"

#include "trsm/recursive_trsm.h"
#include "syrk/batched_syrk.h"

#include <rocsolver/rocsolver.h>

extern "C" {

// ======================================================
// 单精度
// ======================================================

void recursive_potrf(rocblas_handle handle,
                     float* A,
                     int lda,
                     int n,
                     int nb,
                     int* info)
{
    if (n <= 0)
        return;

    // ==================================================
    // Base case
    // ==================================================

    if (n <= nb)
    {
        rocsolver_spotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info);

        return;
    }

    // ==================================================
    // Split
    // ==================================================

    int n1 = n / 2;
    int n2 = n - n1;

    float* A11 = A;

    float* A21 = A + n1;

     float* A22 = A + (size_t)n1 * (size_t)lda + n1;

    // ==================================================
    // POTRF(A11)
    // ==================================================

    recursive_potrf(
        handle,
        A11,
        lda,
        n1,
        nb,
        info);

    // ==================================================
    // TRSM
    // ==================================================

    recursive_trsm(
        handle,
        A11,
        lda,
        A21,
        lda,
        n1,
        n2,
        nb);

    // ==================================================
    // SYRK
    // ==================================================

    batched_syrk_bottom_up(
        handle,
        A21,
        lda,
        A22,
        lda,
        n2,
        n1,
        nb);

    // ==================================================
    // POTRF(A22)
    // ==================================================

    recursive_potrf(
        handle,
        A22,
        lda,
        n2,
        nb,
        info);
}

// ======================================================
// 双精度
// ======================================================

void recursive_potrf_double(rocblas_handle handle,
                            double* A,
                            int lda,
                            int n,
                            int nb,
                            int* info)
{
    if (n <= 0)
        return;

    if (n <= nb)
    {
        rocsolver_dpotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info);

        return;
    }

    int n1 = n / 2;
    int n2 = n - n1;

    double* A11 = A;

    double* A21 = A + n1;

    double* A22 = A + (size_t)n1 * (size_t)lda + n1;

    recursive_potrf_double(
        handle,
        A11,
        lda,
        n1,
        nb,
        info);

    recursive_trsm_double(
        handle,
        A11,
        lda,
        A21,
        lda,
        n1,
        n2,
        nb);

    batched_syrk_bottom_up_double(
        handle,
        A21,
        lda,
        A22,
        lda,
        n2,
        n1,
        nb);

    recursive_potrf_double(
        handle,
        A22,
        lda,
        n2,
        nb,
        info);
}

} // extern "C"