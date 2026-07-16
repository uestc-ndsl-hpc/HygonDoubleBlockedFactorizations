#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include "getrf_batched_pivot_internal.h"
#include "getrf_pivot_internal.h"
#include "utils/benchmark/stage_breakdown_profiler.h"
#include "utils/common/error_check.h"
#include "utils/memory/device_buffer.h"

namespace {

using getrf_pivot_internal::apply_panel_pivots;
using getrf_pivot_internal::block_ptr;
using getrf_pivot_internal::factor_panel;
using getrf_pivot_internal::gemm_nn_update;
using getrf_pivot_internal::trsm_left_lower_unit;

double panel_getrf_work(int rows, int cols)
{
    const double m = static_cast<double>(rows);
    const double n = static_cast<double>(cols);
    return m * n * n - n * n * n / 3.0;
}

double trsm_work(int rows, int cols)
{
    return static_cast<double>(rows) * rows * cols;
}

double gemm_work(int rows, int cols, int k)
{
    return 2.0 * static_cast<double>(rows) * cols * k;
}

template <typename T>
void trsm_left_lower_unit_recursive_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    int rows,
    int cols,
    const T* L,
    int ldl,
    T* B,
    int ldb,
    int leaf_size)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (leaf_size <= 0 || rows <= leaf_size) {
        profiler.measure(
            stream,
            "Outer TRSM leaf",
            trsm_work(rows, cols),
            [&]() {
                trsm_left_lower_unit(
                    handle, rows, cols, L, ldl, B, ldb);
            });
        return;
    }

    const int top_rows = rows / 2;
    const int bottom_rows = rows - top_rows;

    trsm_left_lower_unit_recursive_profiled(
        handle,
        stream,
        profiler,
        top_rows,
        cols,
        L,
        ldl,
        B,
        ldb,
        leaf_size);

    profiler.measure(
        stream,
        "Outer TRSM GEMM",
        gemm_work(bottom_rows, cols, top_rows),
        [&]() {
            gemm_nn_update(
                handle,
                bottom_rows,
                cols,
                top_rows,
                block_ptr(L, ldl, top_rows, 0),
                ldl,
                B,
                ldb,
                B + top_rows,
                ldb);
        });

    trsm_left_lower_unit_recursive_profiled(
        handle,
        stream,
        profiler,
        bottom_rows,
        cols,
        block_ptr(L, ldl, top_rows, top_rows),
        ldl,
        B + top_rows,
        ldb,
        leaf_size);

    return;
}

template <typename T>
void generate_dense_matrix(std::vector<T>& matrix)
{
    std::mt19937 gen(1234567890u);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (T& value : matrix) {
        value = static_cast<T>(dist(gen));
    }

    return;
}

template <typename T>
void prepare_next_inner_panel_profiled(rocblas_handle handle,
                                       hipStream_t stream,
                                       StageBreakdownProfiler& profiler,
                                       T* A,
                                       int lda,
                                       int n,
                                       int outer,
                                       int next_inner,
                                       int outer_end,
                                       int inner_nb)
{
    if (next_inner >= outer_end) {
        return;
    }

    const int next_size = std::min(inner_nb, outer_end - next_inner);
    const int history_size = next_inner - outer;

    profiler.measure(
        stream,
        "Inner TRSM",
        trsm_work(history_size, next_size),
        [&]() {
            trsm_left_lower_unit(
                handle,
                history_size,
                next_size,
                block_ptr(A, lda, outer, outer),
                lda,
                block_ptr(A, lda, outer, next_inner),
                lda);
        });

    profiler.measure(
        stream,
        "Inner GEMM",
        gemm_work(n - next_inner, next_size, history_size),
        [&]() {
            gemm_nn_update(
                handle,
                n - next_inner,
                next_size,
                history_size,
                block_ptr(A, lda, next_inner, outer),
                lda,
                block_ptr(A, lda, outer, next_inner),
                lda,
                block_ptr(A, lda, next_inner, next_inner),
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
                                       int outer_nb,
                                       int trsm_leaf)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(outer_nb, n - next_outer);

    trsm_left_lower_unit_recursive_profiled(
        handle,
        stream,
        profiler,
        next_outer,
        next_size,
        A,
        lda,
        block_ptr(A, lda, 0, next_outer),
        lda,
        trsm_leaf);

    profiler.measure(
        stream,
        "Outer GEMM",
        gemm_work(n - next_outer, next_size, next_outer),
        [&]() {
            gemm_nn_update(
                handle,
                n - next_outer,
                next_size,
                next_outer,
                block_ptr(A, lda, next_outer, 0),
                lda,
                block_ptr(A, lda, 0, next_outer),
                lda,
                block_ptr(A, lda, next_outer, next_outer),
                lda);
        });

    return;
}

template <typename T>
void double_blocked_getrf_profiled(rocblas_handle handle,
                                   hipStream_t stream,
                                   StageBreakdownProfiler& profiler,
                                   T* A,
                                   int lda,
                                   int n,
                                   int outer_nb,
                                   int inner_nb,
                                   int trsm_leaf,
                                   int* ipiv,
                                   int* info)
{
    for (int outer = 0; outer < n; outer += outer_nb) {
        const int outer_size = std::min(outer_nb, n - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += inner_nb) {
            const int inner_size = std::min(inner_nb, outer_end - inner);
            const int next_inner = inner + inner_size;

            profiler.measure(
                stream,
                "Panel GETRF",
                panel_getrf_work(n - inner, inner_size),
                [&]() {
                    factor_panel(
                        handle,
                        n - inner,
                        inner_size,
                        block_ptr(A, lda, inner, inner),
                        lda,
                        ipiv + inner,
                        info);
                });

            profiler.measure(
                stream,
                "Pivot propagation",
                0.0,
                [&]() {
                    apply_panel_pivots(
                        handle,
                        A,
                        lda,
                        n,
                        inner,
                        inner_size,
                        ipiv + inner);
                });

            prepare_next_inner_panel_profiled(
                handle,
                stream,
                profiler,
                A,
                lda,
                n,
                outer,
                next_inner,
                outer_end,
                inner_nb);
        }

        prepare_next_outer_panel_profiled(
            handle,
            stream,
            profiler,
            A,
            lda,
            n,
            outer_end,
            outer_nb,
            trsm_leaf);
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

std::string read_string_arg(int argc,
                            char** argv,
                            const std::string& name,
                            const std::string& fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name.c_str()) == 0) {
            return argv[i + 1];
        }
    }

    return fallback;
}

struct BatchedStageObserver {
    hipStream_t stream;
    StageBreakdownProfiler& profiler;

    template <typename Func>
    void measure(const char* name, double work, Func&& func)
    {
        profiler.measure(stream, name, work, std::forward<Func>(func));

        return;
    }
};

} // namespace

int main(int argc, char** argv)
{
    const int n = read_int_arg(argc, argv, "--n", 40960);
    const int outer_nb = read_int_arg(
        argc,
        argv,
        "--outer-nb",
        read_int_arg(argc, argv, "--nb", n / 2));
    const int inner_nb = read_int_arg(
        argc,
        argv,
        "--inner-nb",
        read_int_arg(argc, argv, "--b", outer_nb / 2));
    const int trsm_leaf = read_int_arg(argc, argv, "--trsm-leaf", 512);
    const int requested_pivot_column_tile = read_int_arg(
        argc, argv, "--pivot-tile", 2048);
    const int inner_gemm_m_tile = read_int_arg(
        argc, argv, "--inner-gemm-m-tile", 0);
    const std::string pivot_mode = read_string_arg(
        argc, argv, "--pivot-mode", "immediate");
    const int warmup = read_int_arg(argc, argv, "--warmup", 1);
    const int iters = read_int_arg(argc, argv, "--iters", 3);

    if (n <= 0 || outer_nb <= 0 || inner_nb <= 0 || inner_nb > outer_nb
        || trsm_leaf <= 0 || requested_pivot_column_tile <= 0
        || inner_gemm_m_tile < 0
        || (pivot_mode != "immediate" && pivot_mode != "batched")
        || warmup < 0 || iters <= 0) {
        std::cerr << "Usage: profile_double_blocked_getrf "
                  << "--n N --outer-nb NB --inner-nb B "
                  << "[--trsm-leaf L] [--pivot-tile T] "
                  << "[--inner-gemm-m-tile TM] "
                  << "[--pivot-mode immediate|batched] "
                  << "[--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    const int pivot_column_tile = std::min(
        n, requested_pivot_column_tile);

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const size_t element_count = static_cast<size_t>(n) * n;
    const size_t bytes = element_count * sizeof(float);
    std::vector<float> h_A(element_count);
    generate_dense_matrix(h_A);

    DeviceBuffer<float> d_A_orig(element_count);
    DeviceBuffer<float> d_A(element_count);
    DeviceBuffer<int> d_ipiv(n);
    DeviceBuffer<int> d_info(1);
    const bool use_batched_pivot = pivot_mode == "batched";
    const size_t pivot_capacity = use_batched_pivot
        ? static_cast<size_t>(n)
        : 0;
    DeviceBuffer<int> d_pivot_sources(pivot_capacity);
    DeviceBuffer<int> d_pivot_targets(pivot_capacity);
    DeviceBuffer<float> d_pivot_swap(
        pivot_capacity * static_cast<size_t>(pivot_column_tile));
    std::vector<int> h_pivots(pivot_capacity);
    std::vector<int> h_permutation(pivot_capacity);
    std::vector<int> h_sources(pivot_capacity);
    std::vector<int> h_targets(pivot_capacity);
    getrf_batched_pivot_workspace batched_workspace{
        d_pivot_sources.get(),
        d_pivot_targets.get(),
        d_pivot_swap.get(),
        h_pivots.data(),
        h_permutation.data(),
        h_sources.data(),
        h_targets.data(),
        use_batched_pivot ? n : 0,
        pivot_column_tile
    };

    HIP_CHECK(hipMemcpy(
        d_A_orig, h_A.data(), bytes, hipMemcpyHostToDevice));

    std::cout << "Profiling pivoted DoubleBlock GETRF: N=" << n
              << ", outer_nb=" << outer_nb
              << ", inner_nb=" << inner_nb
              << ", trsm_leaf=" << trsm_leaf
              << ", pivot_mode=" << pivot_mode
              << ", pivot_tile=" << pivot_column_tile
              << ", inner_gemm_m_tile=" << inner_gemm_m_tile
              << ", warmup=" << warmup
              << ", iters=" << iters << "\n";

    for (int iteration = 0; iteration < warmup; ++iteration) {
        StageBreakdownProfiler warmup_profiler;
        HIP_CHECK(hipMemcpy(
            d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemset(d_ipiv, 0, static_cast<size_t>(n) * sizeof(int)));
        HIP_CHECK(hipMemset(d_info, 0, sizeof(int)));
        if (use_batched_pivot) {
            BatchedStageObserver observer{stream, warmup_profiler};
            if (!getrf_batched_pivot_internal::double_blocked_getrf(
                    handle,
                    observer,
                    d_A.get(),
                    n,
                    n,
                    outer_nb,
                    inner_nb,
                    trsm_leaf,
                    inner_gemm_m_tile,
                    d_ipiv.get(),
                    d_info.get(),
                    batched_workspace)) {
                std::cerr << "Invalid pivot returned during warmup\n";
                return EXIT_FAILURE;
            }
        } else {
            double_blocked_getrf_profiled(
                handle,
                stream,
                warmup_profiler,
                d_A.get(),
                n,
                n,
                outer_nb,
                inner_nb,
                trsm_leaf,
                d_ipiv.get(),
                d_info.get());
        }
    }

    StageBreakdownProfiler profiler("GFLOPS");
    for (int iteration = 0; iteration < iters; ++iteration) {
        HIP_CHECK(hipMemcpy(
            d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemset(d_ipiv, 0, static_cast<size_t>(n) * sizeof(int)));
        HIP_CHECK(hipMemset(d_info, 0, sizeof(int)));
        if (use_batched_pivot) {
            BatchedStageObserver observer{stream, profiler};
            if (!getrf_batched_pivot_internal::double_blocked_getrf(
                    handle,
                    observer,
                    d_A.get(),
                    n,
                    n,
                    outer_nb,
                    inner_nb,
                    trsm_leaf,
                    inner_gemm_m_tile,
                    d_ipiv.get(),
                    d_info.get(),
                    batched_workspace)) {
                std::cerr << "Invalid pivot returned during profiling\n";
                return EXIT_FAILURE;
            }
        } else {
            double_blocked_getrf_profiled(
                handle,
                stream,
                profiler,
                d_A.get(),
                n,
                n,
                outer_nb,
                inner_nb,
                trsm_leaf,
                d_ipiv.get(),
                d_info.get());
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    profiler.print(
        "Pivoted DoubleBlock GETRF Stage Breakdown (N=" + std::to_string(n)
        + ", outer_nb=" + std::to_string(outer_nb)
        + ", inner_nb=" + std::to_string(inner_nb)
        + ", trsm_leaf=" + std::to_string(trsm_leaf)
        + ", pivot_mode=" + pivot_mode
        + ", pivot_tile=" + std::to_string(pivot_column_tile)
        + ", inner_gemm_m_tile=" + std::to_string(inner_gemm_m_tile)
        + ")",
        118);

    std::cout << "Profiled stage time per iteration: "
              << std::fixed << std::setprecision(3)
              << profiler.total_time_ms() / static_cast<double>(iters)
              << " ms\n";

    int info = 0;
    HIP_CHECK(hipMemcpy(
        &info, d_info, sizeof(int), hipMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "GETRF reported info=" << info << "\n";
    }

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return info == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
