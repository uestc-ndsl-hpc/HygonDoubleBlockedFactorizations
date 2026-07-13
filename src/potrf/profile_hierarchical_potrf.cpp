#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "utils/benchmark/stage_breakdown_profiler.h"
#include "utils/common/error_check.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace {

constexpr int kGemmMThreshold = 32768;
constexpr int kTrsmUpdateTile = 8192;

float* block_ptr(float* A, int lda, int row, int col)
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

void rocsolver_potrf(
    rocblas_handle handle,
    int n,
    float* A,
    int lda,
    int* info)
{
    ROCBLAS_CHECK(rocsolver_spotrf(
        handle,
        rocblas_fill_lower,
        n,
        A,
        lda,
        info));

    return;
}

void rocblas_trsm_right_lower_transpose(
    rocblas_handle handle,
    int rows,
    int cols,
    const float* A,
    int lda,
    float* B,
    int ldb)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const float one = 1.0f;
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

    return;
}

void rocblas_gemm_nt_update(
    rocblas_handle handle,
    int rows,
    int cols,
    int k,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float* C,
    int ldc)
{
    if (rows <= 0 || cols <= 0 || k <= 0) {
        return;
    }

    const float neg_one = -1.0f;
    const float one = 1.0f;
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

    return;
}

void rocblas_syrk_lower_update(
    rocblas_handle handle,
    int n,
    int k,
    const float* A,
    int lda,
    float* C,
    int ldc)
{
    if (n <= 0 || k <= 0) {
        return;
    }

    const float neg_one = -1.0f;
    const float one = 1.0f;
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

    return;
}

void gemm_nt_update_m_tiled_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    const std::string& stage,
    int rows,
    int cols,
    int k,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float* C,
    int ldc,
    int m_tile)
{
    if (rows <= 0 || cols <= 0 || k <= 0) {
        return;
    }

    if (m_tile <= 0 || m_tile >= rows) {
        profiler.measure(
            stream,
            stage,
            gemm_work(rows, cols, k),
            [&]() {
                rocblas_gemm_nt_update(
                    handle, rows, cols, k, A, lda, B, ldb, C, ldc);
            });
        return;
    }

    for (int row = 0; row < rows; row += m_tile) {
        const int tile_rows = std::min(m_tile, rows - row);
        profiler.measure(
            stream,
            stage,
            gemm_work(tile_rows, cols, k),
            [&]() {
                rocblas_gemm_nt_update(
                    handle,
                    tile_rows,
                    cols,
                    k,
                    A + row,
                    lda,
                    B,
                    ldb,
                    C + row,
                    ldc);
            });
    }

    return;
}

void blocked_right_looking_panel_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    int n,
    int leaf_size,
    int* info)
{
    for (int k = 0; k < n; k += leaf_size) {
        const int diagonal_size = std::min(leaf_size, n - k);
        const int trailing_size = n - k - diagonal_size;
        float* diagonal = block_ptr(A, lda, k, k);

        profiler.measure(
            stream,
            "Local POTRF",
            potrf_work(diagonal_size),
            [&]() {
                rocsolver_potrf(
                    handle, diagonal_size, diagonal, lda, info);
            });

        if (trailing_size <= 0) {
            continue;
        }

        float* below = block_ptr(A, lda, k + diagonal_size, k);
        float* trailing =
            block_ptr(A, lda, k + diagonal_size, k + diagonal_size);

        profiler.measure(
            stream,
            "Local TRSM",
            trsm_work(trailing_size, diagonal_size),
            [&]() {
                rocblas_trsm_right_lower_transpose(
                    handle,
                    trailing_size,
                    diagonal_size,
                    diagonal,
                    lda,
                    below,
                    lda);
            });

        profiler.measure(
            stream,
            "Local SYRK",
            syrk_work(trailing_size, diagonal_size),
            [&]() {
                rocblas_syrk_lower_update(
                    handle,
                    trailing_size,
                    diagonal_size,
                    below,
                    lda,
                    trailing,
                    lda);
            });
    }

    return;
}

void recursive_trsm_update_gemm_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    float* B,
    int ldb,
    int rows,
    int left_cols,
    int right_cols)
{
    if (rows <= kTrsmUpdateTile &&
        left_cols <= kTrsmUpdateTile &&
        right_cols <= kTrsmUpdateTile) {
        profiler.measure(
            stream,
            "Recursive TRSM GEMM",
            gemm_work(rows, right_cols, left_cols),
            [&]() {
                rocblas_gemm_nt_update(
                    handle,
                    rows,
                    right_cols,
                    left_cols,
                    B,
                    ldb,
                    A + left_cols,
                    lda,
                    B + static_cast<size_t>(left_cols) * ldb,
                    ldb);
            });
        return;
    }

    for (int row = 0; row < rows; row += kTrsmUpdateTile) {
        const int tile_rows = std::min(kTrsmUpdateTile, rows - row);

        for (int col = 0; col < right_cols; col += kTrsmUpdateTile) {
            const int tile_cols =
                std::min(kTrsmUpdateTile, right_cols - col);

            for (int kk = 0; kk < left_cols; kk += kTrsmUpdateTile) {
                const int tile_k =
                    std::min(kTrsmUpdateTile, left_cols - kk);

                profiler.measure(
                    stream,
                    "Recursive TRSM GEMM",
                    gemm_work(tile_rows, tile_cols, tile_k),
                    [&]() {
                        rocblas_gemm_nt_update(
                            handle,
                            tile_rows,
                            tile_cols,
                            tile_k,
                            B + static_cast<size_t>(kk) * ldb + row,
                            ldb,
                            A + static_cast<size_t>(kk) * lda +
                                left_cols + col,
                            lda,
                            B + static_cast<size_t>(left_cols + col) * ldb +
                                row,
                            ldb);
                    });
            }
        }
    }

    return;
}

void recursive_trsm_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    float* B,
    int ldb,
    int rows,
    int cols,
    int leaf_size)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (cols <= leaf_size) {
        profiler.measure(
            stream,
            "Recursive TRSM leaf",
            trsm_work(rows, cols),
            [&]() {
                rocblas_trsm_right_lower_transpose(
                    handle, rows, cols, A, lda, B, ldb);
            });
        return;
    }

    const int left_cols = cols / 2;
    const int right_cols = cols - left_cols;

    recursive_trsm_profiled(
        handle,
        stream,
        profiler,
        A,
        lda,
        B,
        ldb,
        rows,
        left_cols,
        leaf_size);

    recursive_trsm_update_gemm_profiled(
        handle,
        stream,
        profiler,
        A,
        lda,
        B,
        ldb,
        rows,
        left_cols,
        right_cols);

    recursive_trsm_profiled(
        handle,
        stream,
        profiler,
        block_ptr(A, lda, left_cols, left_cols),
        lda,
        B + static_cast<size_t>(left_cols) * ldb,
        ldb,
        rows,
        right_cols,
        leaf_size);

    return;
}

void update_panel_from_current_outer_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    int n,
    int outer,
    int panel_start,
    int panel_size,
    int syrk_k_tile,
    int gemm_m_tile)
{
    const int k = panel_start - outer;
    if (k <= 0 || panel_size <= 0) {
        return;
    }

    const int effective_k_tile = syrk_k_tile > 0 ? syrk_k_tile : k;
    for (int k_offset = 0; k_offset < k; k_offset += effective_k_tile) {
        const int tile_k = std::min(effective_k_tile, k - k_offset);
        profiler.measure(
            stream,
            "Inter-panel SYRK",
            syrk_work(panel_size, tile_k),
            [&]() {
                rocblas_syrk_lower_update(
                    handle,
                    panel_size,
                    tile_k,
                    block_ptr(A, lda, panel_start, outer + k_offset),
                    lda,
                    block_ptr(A, lda, panel_start, panel_start),
                    lda);
            });
    }

    const int below_start = panel_start + panel_size;
    const int below_rows = n - below_start;
    if (below_rows <= 0) {
        return;
    }

    const int effective_m_tile =
        below_rows > kGemmMThreshold ? gemm_m_tile : 0;

    gemm_nt_update_m_tiled_profiled(
        handle,
        stream,
        profiler,
        "Inter-panel GEMM",
        below_rows,
        panel_size,
        k,
        block_ptr(A, lda, below_start, outer),
        lda,
        block_ptr(A, lda, panel_start, outer),
        lda,
        block_ptr(A, lda, below_start, panel_start),
        lda,
        effective_m_tile);

    return;
}

void factor_and_solve_panel_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    int n,
    int panel_start,
    int panel_size,
    int leaf_size,
    int* info)
{
    float* diagonal = block_ptr(A, lda, panel_start, panel_start);
    blocked_right_looking_panel_profiled(
        handle,
        stream,
        profiler,
        diagonal,
        lda,
        panel_size,
        leaf_size,
        info);

    const int below_start = panel_start + panel_size;
    const int below_rows = n - below_start;
    if (below_rows <= 0) {
        return;
    }

    recursive_trsm_profiled(
        handle,
        stream,
        profiler,
        diagonal,
        lda,
        block_ptr(A, lda, below_start, panel_start),
        lda,
        below_rows,
        panel_size,
        leaf_size);

    return;
}

void prepare_next_outer_panel_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    int n,
    int next_outer,
    int outer_block_size,
    int syrk_k_tile,
    int gemm_m_tile)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(outer_block_size, n - next_outer);
    const int below_start = next_outer + next_size;
    const int below_rows = n - below_start;

    for (int k_offset = 0; k_offset < next_outer;
         k_offset += syrk_k_tile) {
        const int tile_k = std::min(syrk_k_tile, next_outer - k_offset);
        profiler.measure(
            stream,
            "Outer SYRK",
            syrk_work(next_size, tile_k),
            [&]() {
                rocblas_syrk_lower_update(
                    handle,
                    next_size,
                    tile_k,
                    block_ptr(A, lda, next_outer, k_offset),
                    lda,
                    block_ptr(A, lda, next_outer, next_outer),
                    lda);
            });
    }

    const int effective_m_tile =
        below_rows > kGemmMThreshold ? gemm_m_tile : 0;
    gemm_nt_update_m_tiled_profiled(
        handle,
        stream,
        profiler,
        "Outer GEMM",
        below_rows,
        next_size,
        next_outer,
        block_ptr(A, lda, below_start, 0),
        lda,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, below_start, next_outer),
        lda,
        effective_m_tile);

    return;
}

void hierarchical_potrf_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    int n,
    int outer_block_size,
    int panel_block_size,
    int leaf_size,
    int syrk_k_tile,
    int gemm_m_tile,
    int* info)
{
    for (int outer = 0; outer < n; outer += outer_block_size) {
        const int outer_size = std::min(outer_block_size, n - outer);
        const int outer_end = outer + outer_size;

        for (int panel = outer; panel < outer_end;
             panel += panel_block_size) {
            const int panel_size =
                std::min(panel_block_size, outer_end - panel);

            update_panel_from_current_outer_profiled(
                handle,
                stream,
                profiler,
                A,
                lda,
                n,
                outer,
                panel,
                panel_size,
                syrk_k_tile,
                gemm_m_tile);

            factor_and_solve_panel_profiled(
                handle,
                stream,
                profiler,
                A,
                lda,
                n,
                panel,
                panel_size,
                leaf_size,
                info);
        }

        prepare_next_outer_panel_profiled(
            handle,
            stream,
            profiler,
            A,
            lda,
            n,
            outer_end,
            outer_block_size,
            syrk_k_tile,
            gemm_m_tile);
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
    const int n = read_int_arg(argc, argv, "--n", 40960);
    const int nb = read_int_arg(argc, argv, "--nb", 8192);
    const int panel_block_size =
        read_int_arg(argc, argv, "--panel-b", 8192);
    const int leaf_size = read_int_arg(argc, argv, "--leaf-b", 512);
    const int syrk_k_tile =
        read_int_arg(argc, argv, "--syrk-k-tile", 8192);
    const int gemm_m_tile =
        read_int_arg(argc, argv, "--gemm-m-tile", 8192);
    const int warmup = read_int_arg(argc, argv, "--warmup", 1);
    const int iters = read_int_arg(argc, argv, "--iters", 3);

    if (n <= 0 || nb <= 0 || panel_block_size <= 0 ||
        panel_block_size > nb || leaf_size <= 0 ||
        leaf_size > panel_block_size || syrk_k_tile <= 0 ||
        gemm_m_tile <= 0 || warmup < 0 || iters <= 0) {
        std::cerr
            << "Usage: profile_hierarchical_potrf "
            << "--n N --nb NB --panel-b PB --leaf-b IB "
            << "--syrk-k-tile KT --gemm-m-tile MT "
            << "[--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(
        handle, rocblas_pointer_mode_host));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const size_t num_elements = static_cast<size_t>(n) * n;
    const size_t bytes = num_elements * sizeof(float);

    DeviceBuffer<float> d_A_orig(num_elements);
    DeviceBuffer<float> d_A(num_elements);
    DeviceBuffer<int> d_info(1);

    std::cout << "Generating SPD matrix: N=" << n << "\n";
    generate_spd_matrix<float>(handle, d_A_orig, n, n);

    std::cout << "Profiling Hierarchical DoubleBlock POTRF: N=" << n
              << ", nb=" << nb
              << ", panel_b=" << panel_block_size
              << ", leaf_b=" << leaf_size
              << ", syrk_k_tile=" << syrk_k_tile
              << ", gemm_m_tile=" << gemm_m_tile
              << ", warmup=" << warmup
              << ", iters=" << iters << "\n";

    for (int i = 0; i < warmup; ++i) {
        StageBreakdownProfiler warmup_profiler("GFLOPS");
        HIP_CHECK(hipMemcpy(
            d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        hierarchical_potrf_profiled(
            handle,
            stream,
            warmup_profiler,
            d_A.get(),
            n,
            n,
            nb,
            panel_block_size,
            leaf_size,
            syrk_k_tile,
            gemm_m_tile,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    StageBreakdownProfiler profiler("GFLOPS");
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipMemcpy(
            d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        hierarchical_potrf_profiled(
            handle,
            stream,
            profiler,
            d_A.get(),
            n,
            n,
            nb,
            panel_block_size,
            leaf_size,
            syrk_k_tile,
            gemm_m_tile,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    profiler.print(
        "Hierarchical DoubleBlock POTRF Stage Breakdown (N=" +
            std::to_string(n) +
            ", nb=" + std::to_string(nb) +
            ", panel_b=" + std::to_string(panel_block_size) +
            ", leaf_b=" + std::to_string(leaf_size) +
            ", syrk_k_tile=" + std::to_string(syrk_k_tile) +
            ", gemm_m_tile=" + std::to_string(gemm_m_tile) + ")",
        110);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return EXIT_SUCCESS;
}
