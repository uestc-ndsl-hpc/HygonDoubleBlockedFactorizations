#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

unsigned int get_random_seed();

template <typename T>
void generate_spd_matrix(rocblas_handle handle, T* d_A, int n, int lda, unsigned int seed = get_random_seed());

template <typename T>
void generate_syrk_matrices(rocblas_handle handle, T* d_A, T* d_C, int n, int k, int lda, int ldc, unsigned int seed = get_random_seed());

template <typename T>
void generate_stable_lower_triangular(T* A, int dim, int lda, unsigned int seed = get_random_seed());

template <typename T>
void generate_random_matrix(T* B, int n, int m, int ldb, unsigned int seed = get_random_seed());