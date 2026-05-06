#ifndef ITERATIVE_TRSM_RIGHT_H
#define ITERATIVE_TRSM_RIGHT_H

#include <rocblas/rocblas.h>

extern "C" {
    // 单精度迭代右视 TRSM
    void iterative_trsm_right_looking(rocblas_handle handle,
                                      float* A, int lda,
                                      float* B, int ldb,
                                      int n, int m, int nb);
}

#endif