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

double gemm_work(int rows, int cols, int k)
{
    return 2.0 * static_cast<double>(rows) * cols * k;
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
void rocblas_gemm_nt_update(rocblas_handle handle,
                            int rows,
                            int cols,
                            int k,
                            const T* A,
                            int lda,
                            const T* B,
                            int ldb,
                            T* C,
                            int ldc)
{
    if (rows <= 0 || cols <= 0 || k <= 0) {
        return;
    }

    const T neg_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            rows,
            cols,
            k,
            &neg_one,
            A,
            lda,
            B,
            ldb,
            &one,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            rows,
            cols,
            k,
            &neg_one,
            A,
            lda,
            B,
            ldb,
            &one,
            C,
            ldc));
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
void update_next_inner_panel_profiled(rocblas_handle handle,
                                      hipStream_t stream,
                                      StageBreakdownProfiler& profiler,
                                      T* A,
                                      int lda,
                                      int n,
                                      int outer,
                                      int panel_start,
                                      int outer_end,
                                      int b)
{
    if (panel_start >= outer_end) {
        return;
    }

    const int panel_cols = std::min(b, outer_end - panel_start);
    const int rows = n - panel_start;
    const int k = panel_start - outer;

    profiler.measure(
        stream,
        "Inner GEMM",
        gemm_work(rows, panel_cols, k),
        [&]() {
            rocblas_gemm_nt_update(
                handle,
                rows,
                panel_cols,
                k,
                block_ptr(A, lda, panel_start, outer),
                lda,
                block_ptr(A, lda, panel_start, outer),
                lda,
                block_ptr(A, lda, panel_start, panel_start),
                lda);
        });

    return;
}

template <typename T>
void prepare_next_outer_panel_profiled(rocblas_handle handle,
                                       hipStream_t stream,
                                       StageBreakdownProfiler& profiler,
                                       T* A,
                                       int lda,
                                       int n,
                                       int next_outer,
                                       int nb)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(nb, n - next_outer);
    const int below_start = next_outer + next_size;
    const int below_rows = n - below_start;

    profiler.measure(
        stream,
        "Outer SYRK",
        syrk_work(next_size, next_outer),
        [&]() {
            rocblas_syrk_lower_update(
                handle,
                next_size,
                next_outer,
                block_ptr(A, lda, next_outer, 0),
                lda,
                block_ptr(A, lda, next_outer, next_outer),
                lda);
        });

    profiler.measure(
        stream,
        "Outer GEMM",
        gemm_work(below_rows, next_size, next_outer),
        [&]() {
            rocblas_gemm_nt_update(
                handle,
                below_rows,
                next_size,
                next_outer,
                block_ptr(A, lda, below_start, 0),
                lda,
                block_ptr(A, lda, next_outer, 0),
                lda,
                block_ptr(A, lda, below_start, next_outer),
                lda);
        });

    return;
}

template <typename T>
void double_blocked_potrf_profiled(rocblas_handle handle,
                                   hipStream_t stream,
                                   StageBreakdownProfiler& profiler,
                                   T* A,
                                   int lda,
                                   int n,
                                   int nb,
                                   int b,
                                   int* info)
{
    for (int outer = 0; outer < n; outer += nb) {
        const int outer_size = std::min(nb, n - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += b) {
            const int inner_size = std::min(b, outer_end - inner);
            const int panel_start = inner + inner_size;
            const int trailing_rows = n - panel_start;

            T* A_ii = block_ptr(A, lda, inner, inner);

            profiler.measure(
                stream,
                "POTRF",
                potrf_work(inner_size),
                [&]() {
                    rocsolver_potrf(handle, inner_size, A_ii, lda, info);
                });

            profiler.measure(
                stream,
                "TRSM",
                trsm_work(trailing_rows, inner_size),
                [&]() {
                    rocblas_trsm_right_lower_transpose(
                        handle,
                        trailing_rows,
                        inner_size,
                        A_ii,
                        lda,
                        block_ptr(A, lda, panel_start, inner),
                        lda);
                });

            update_next_inner_panel_profiled(
                handle,
                stream,
                profiler,
                A,
                lda,
                n,
                outer,
                panel_start,
                outer_end,
                b);
        }

        prepare_next_outer_panel_profiled(
            handle,
            stream,
            profiler,
            A,
            lda,
            n,
            outer_end,
            nb);
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
    const int nb = read_int_arg(argc, argv, "--nb", 8192);
    const int b = read_int_arg(argc, argv, "--b", 512);
    const int warmup = read_int_arg(argc, argv, "--warmup", 1);
    const int iters = read_int_arg(argc, argv, "--iters", 3);

    if (n <= 0 || nb <= 0 || b <= 0 || b > nb) {
        std::cerr << "Usage: profile_double_blocked_potrf "
                  << "--n N --nb NB --b B [--warmup W] [--iters I]\n";
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

    std::cout << "Profiling DoubleBlock POTRF: N=" << n
              << ", nb=" << nb
              << ", b=" << b
              << ", warmup=" << warmup
              << ", iters=" << iters << "\n";

    for (int i = 0; i < warmup; ++i) {
        StageBreakdownProfiler warmup_profiler;
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        double_blocked_potrf_profiled(
            handle,
            stream,
            warmup_profiler,
            d_A.get(),
            n,
            n,
            nb,
            b,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    StageBreakdownProfiler profiler("GFLOPS");
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        double_blocked_potrf_profiled(
            handle,
            stream,
            profiler,
            d_A.get(),
            n,
            n,
            nb,
            b,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    profiler.print(
        "DoubleBlock POTRF Stage Breakdown (N=" + std::to_string(n) +
        ", nb=" + std::to_string(nb) +
        ", b=" + std::to_string(b) + ")",
        110);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return EXIT_SUCCESS;
}
