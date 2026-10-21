#pragma once

// Minimal LP64 CBLAS declaration for SciPy's privately prefixed OpenBLAS.
// This adapter changes the symbol name only; all arithmetic is CPU DGEMM.
// Use only with a library verified to export scipy_cblas_dgemm (not ILP64).
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
extern "C" void scipy_cblas_dgemm(
    CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    int, int, int, double, const double*, int,
    const double*, int, double, double*, int);
#define cblas_dgemm scipy_cblas_dgemm
