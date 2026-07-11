#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "utils/benchmark/stage_breakdown_profiler.h"
#include "utils/common/error_check.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace {

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

double potrf_work(int n)
{
    return static_cast<double>(n) * n * n / 3.0;
}

double trsm_work(int rows, int cols)
{
    return static_cast<double>(rows) * cols * cols;
}

double syrk_work(int n, int k)
{
    return static_cast<double>(n) * n * k;
}

template <typename T>
void rocsolver_potrf(rocblas_handle handle, int n, T* A, int lda, int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(handle, rocblas_fill_lower, n, A, lda, info));
    } else {
        ROCBLAS_CHECK(rocsolver_dpotrf(handle, rocblas_fill_lower, n, A, lda, info));
    }

    return;
}

template <typename T>
void rocblas_trsm_right_lower_transpose(rocblas_handle handle,
                                        int rows,
                                        int cols,
                                        const T* A,
                                        int lda,
                                        T* B,
                                        int ldb)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_strsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            rows,
            cols,
            &one,
            A,
            lda,
            B,
            ldb));
    } else {
        ROCBLAS_CHECK(rocblas_dtrsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            rows,
            cols,
            &one,
            A,
            lda,
            B,
            ldb));
    }

    return;
}

template <typename T>
void rocblas_syrk_lower_update(rocblas_handle handle,
                               int n,
                               int k,
                               const T* A,
                               int lda,
                               T* C,
                               int ldc)
{
    if (n <= 0 || k <= 0) {
        return;
    }

    const T neg_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_ssyrk(
            handle,
            rocblas_fill_lower,
            rocblas_operation_none,
            n,
            k,
            &neg_one,
            A,
            lda,
            &one,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dsyrk(
            handle,
            rocblas_fill_lower,
            rocblas_operation_none,
            n,
            k,
            &neg_one,
            A,
            lda,
            &one,
            C,
            ldc));
    }

    return;
}

template <typename T>
void blocked_right_looking_potrf_profiled(rocblas_handle handle,
                                          hipStream_t stream,
                                          StageBreakdownProfiler& profiler,
                                          T* A,
                                          int lda,
                                          int n,
                                          int nb,
                                          int* info)
{
    if (n <= 0) {
        return;
    }

    if (nb <= 0) {
        nb = n;
    }

    for (int k = 0; k < n; k += nb) {
        const int panel_size = std::min(nb, n - k);
        const int trailing_size = n - k - panel_size;

        T* A_kk = block_ptr(A, lda, k, k);

        profiler.measure(
            stream,
            "POTRF",
            potrf_work(panel_size),
            [&]() {
                rocsolver_potrf(handle, panel_size, A_kk, lda, info);
            });

        if (trailing_size <= 0) {
            continue;
        }

        T* A_21 = block_ptr(A, lda, k + panel_size, k);
        T* A_22 = block_ptr(A, lda, k + panel_size, k + panel_size);

        profiler.measure(
            stream,
            "TRSM",
            trsm_work(trailing_size, panel_size),
            [&]() {
                rocblas_trsm_right_lower_transpose(
                    handle,
                    trailing_size,
                    panel_size,
                    A_kk,
                    lda,
                    A_21,
                    lda);
            });

        profiler.measure(
            stream,
            "SYRK",
            syrk_work(trailing_size, panel_size),
            [&]() {
                rocblas_syrk_lower_update(
                    handle,
                    trailing_size,
                    panel_size,
                    A_21,
                    lda,
                    A_22,
                    lda);
            });
    }

    return;
}

int read_int_arg(int argc, char** argv, const std::string& name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name.c_str()) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }

    return fallback;
}

} // namespace

int main(int argc, char** argv)
{
    const int n = read_int_arg(argc, argv, "--n", 32768);
    const int nb = read_int_arg(argc, argv, "--nb", 2048);
    const int warmup = read_int_arg(argc, argv, "--warmup", 1);
    const int iters = read_int_arg(argc, argv, "--iters", 3);

    if (n <= 0 || nb <= 0 || warmup < 0 || iters <= 0) {
        std::cerr << "Usage: profile_blocked_potrf "
                  << "--n N --nb NB [--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const size_t num_elements = static_cast<size_t>(n) * n;
    const size_t bytes = num_elements * sizeof(float);

    DeviceBuffer<float> d_A_orig(num_elements);
    DeviceBuffer<float> d_A(num_elements);
    DeviceBuffer<int> d_info(1);

    std::cout << "Generating SPD matrix: N=" << n << "\n";
    generate_spd_matrix<float>(handle, d_A_orig, n, n);

    std::cout << "Profiling Blocked-RL POTRF: N=" << n
              << ", nb=" << nb
              << ", warmup=" << warmup
              << ", iters=" << iters << "\n";

    for (int i = 0; i < warmup; ++i) {
        StageBreakdownProfiler warmup_profiler;
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        blocked_right_looking_potrf_profiled(
            handle,
            stream,
            warmup_profiler,
            d_A.get(),
            n,
            n,
            nb,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    StageBreakdownProfiler profiler("GFLOPS");
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        blocked_right_looking_potrf_profiled(
            handle,
            stream,
            profiler,
            d_A.get(),
            n,
            n,
            nb,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    profiler.print(
        "Blocked-RL POTRF Stage Breakdown (N=" + std::to_string(n) +
        ", nb=" + std::to_string(nb) + ")",
        110);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return EXIT_SUCCESS;
}
