#include "utils/matrix/matrix_generator.h"
#include <random>
#include <vector>
#include <type_traits>
#include <algorithm>
#include "utils/common/error_check.h"
#include "kernels/add_damping.h"

unsigned int get_random_seed()
{
    std::random_device rd;
    return rd();
}

template <typename T>
void generate_spd_matrix(rocblas_handle handle, T* d_A, int n, int lda, unsigned int seed)
{
    size_t num_elements = (size_t)n * (size_t)lda;
    size_t bytes = num_elements * sizeof(T);

    std::vector<T> h_R(num_elements);
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    for (size_t i = 0; i < num_elements; ++i) {
        h_R[i] = static_cast<T>(norm_dist(gen));
    }

    T* d_R = nullptr;
    HIP_CHECK(hipMalloc(&d_R, bytes));
    HIP_CHECK(hipMemcpy(d_R, h_R.data(), bytes, hipMemcpyHostToDevice));

    T alpha = static_cast<T>(1.0);
    T beta  = static_cast<T>(0.0);
    
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(handle, 
                                    rocblas_operation_none, rocblas_operation_transpose,
                                    n, n, n, 
                                    &alpha, 
                                    d_R, lda, 
                                    d_R, lda, 
                                    &beta, 
                                    d_A, lda));
    } else if constexpr (std::is_same_v<T, double>) {
        ROCBLAS_CHECK(rocblas_dgemm(handle, 
                                    rocblas_operation_none, rocblas_operation_transpose,
                                    n, n, n, 
                                    &alpha, 
                                    d_R, lda, 
                                    d_R, lda, 
                                    &beta, 
                                    d_A, lda));
    }

    HIP_CHECK(hipFree(d_R));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    if constexpr (std::is_same_v<T, float>) {
        launch_add_damping(d_A, n, lda, 1.0f, stream);
    } else {
        launch_add_damping_double(d_A, n, lda, 1.0, stream);
    }
    
    HIP_CHECK(hipStreamSynchronize(stream));

    return;
}

template <typename T>
void generate_syrk_matrices(rocblas_handle handle, T* d_A, T* d_C, int n, int k, int lda, int ldc, unsigned int seed)
{
    generate_spd_matrix<T>(handle, d_C, n, ldc, seed);

    size_t num_elements_A = (size_t)lda * (size_t)k; 
    size_t bytes_A = num_elements_A * sizeof(T);
    std::vector<T> h_A(num_elements_A);
    
    std::mt19937 gen(seed + 1);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    for (size_t i = 0; i < num_elements_A; ++i) {
        h_A[i] = static_cast<T>(norm_dist(gen));
    }

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes_A, hipMemcpyHostToDevice));

    return;
}

template <typename T>
void generate_stable_lower_triangular(T* A, int dim, int lda, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < dim; i++) {
            if (i < j) {
                A[j * lda + i] = static_cast<T>(0.0);
            }
            else if (i == j) {
                A[j * lda + i] = static_cast<T>(dim + dist(gen));
            }
            else {
                A[j * lda + i] = static_cast<T>(dist(gen));
            }
        }
    }

    return;
}

template <typename T>
void generate_random_matrix(T* B, int n, int m, int ldb, unsigned int seed)
{
    std::mt19937 gen(seed + 1);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            B[j * ldb + i] = static_cast<T>(dist(gen));
        }
    }

    return;
}

template void generate_spd_matrix<float>(rocblas_handle, float*, int, int, unsigned int);
template void generate_spd_matrix<double>(rocblas_handle, double*, int, int, unsigned int);

template void generate_syrk_matrices<float>(rocblas_handle, float*, float*, int, int, int, int, unsigned int);
template void generate_syrk_matrices<double>(rocblas_handle, double*, double*, int, int, int, int, unsigned int);

template void generate_stable_lower_triangular<float>(float*, int, int, unsigned int);
template void generate_stable_lower_triangular<double>(double*, int, int, unsigned int);

template void generate_random_matrix<float>(float*, int, int, int, unsigned int);
template void generate_random_matrix<double>(double*, int, int, int, unsigned int);