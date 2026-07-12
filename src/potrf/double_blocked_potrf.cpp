#include "potrf/double_blocked_potrf.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <rocsolver/rocsolver.h>

#include "gemm/update_gemm.h"
#include "potrf/inverse_panel_kernels.h"
#include "potrf/profile_inverse_panel_potrf.h"
#include "syrk/kblocked_syrk.h"
#include "trsm/double_blocked_trsm.h"
#include "trsm/right_lower_trsm.h"
#include "utils/benchmark/stage_breakdown_profiler.h"
#include "utils/common/error_check.h"

namespace {

constexpr int kPotrfDbTrsmMicroBlock = 1024;
constexpr int kInnerGemmMTile = 8192;
constexpr int kInnerGemmMThreshold = 32768;
constexpr int kInnerGemmKThreshold = 5632;

double potrf_work(int n)
{
    return static_cast<double>(n) * n * n / 3.0;
}

double panel_gemm_work(int rows, int cols)
{
    return 2.0 * static_cast<double>(rows) * cols * cols;
}

double inner_update_work(int rows, int cols, int k)
{
    return 2.0 * static_cast<double>(rows) * cols * k;
}

double outer_update_work(int diagonal_size, int below_rows, int k)
{
    return static_cast<double>(diagonal_size) * diagonal_size * k +
        2.0 * static_cast<double>(below_rows) * diagonal_size * k;
}

template <typename Func>
void run_stage(StageBreakdownProfiler* profiler,
               hipStream_t stream,
               const char* name,
               double work,
               Func&& function)
{
    if (profiler == nullptr) {
        function();
        return;
    }

    profiler->measure(stream, name, work, std::forward<Func>(function));

    return;
}

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
const T* block_ptr(const T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
void rocsolver_potrf(rocblas_handle handle, int n, T* A, int lda, int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info));
    } else {
        ROCBLAS_CHECK(rocsolver_dpotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info));
    }

    return;
}

template <typename T>
void dispatch_trsm_right_lower_transpose(
    rocblas_handle handle,
    int rows,
    int cols,
    const T* A,
    int lda,
    T* B,
    int ldb)
{
    if constexpr (std::is_same_v<T, float>) {
        trsm_right_lower_transpose(handle, rows, cols, A, lda, B, ldb);
    } else {
        trsm_right_lower_transpose_double(
            handle, rows, cols, A, lda, B, ldb);
    }

    return;
}

template <typename T>
void dispatch_double_blocked_trsm_right_lower_transpose(
    rocblas_handle handle,
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

    const int trsm_b = std::min(kPotrfDbTrsmMicroBlock, cols);

    if constexpr (std::is_same_v<T, float>) {
        double_blocked_trsm(
            handle,
            const_cast<T*>(A),
            lda,
            B,
            ldb,
            rows,
            cols,
            cols,
            trsm_b);
    } else {
        double_blocked_trsm_double(
            handle,
            const_cast<T*>(A),
            lda,
            B,
            ldb,
            rows,
            cols,
            cols,
            trsm_b);
    }

    return;
}

template <typename T>
void dispatch_gemm_nt_update(
    rocblas_handle handle,
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
    if constexpr (std::is_same_v<T, float>) {
        gemm_nt_update(handle, rows, cols, k, A, lda, B, ldb, C, ldc);
    } else {
        gemm_nt_update_double(
            handle, rows, cols, k, A, lda, B, ldb, C, ldc);
    }

    return;
}

template <typename T>
void dispatch_gemm_nt_update_m_tiled(
    rocblas_handle handle,
    int rows,
    int cols,
    int k,
    const T* A,
    int lda,
    const T* B,
    int ldb,
    T* C,
    int ldc,
    int m_tile)
{
    if constexpr (std::is_same_v<T, float>) {
        gemm_nt_update_m_tiled(
            handle,
            rows,
            cols,
            k,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m_tile);
    } else {
        gemm_nt_update_m_tiled_double(
            handle,
            rows,
            cols,
            k,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            m_tile);
    }

    return;
}

template <typename T>
void dispatch_syrk_lower_update_kblocked(
    rocblas_handle handle,
    int n,
    int k,
    const T* A,
    int lda,
    T* C,
    int ldc,
    int k_tile)
{
    if constexpr (std::is_same_v<T, float>) {
        syrk_lower_update_kblocked(handle, n, k, A, lda, C, ldc, k_tile);
    } else {
        syrk_lower_update_kblocked_double(
            handle, n, k, A, lda, C, ldc, k_tile);
    }

    return;
}

template <typename T>
void dispatch_pack_lower_triangular(const T* source,
                                    int source_ld,
                                    T* destination,
                                    int destination_ld,
                                    int n,
                                    hipStream_t stream)
{
    if constexpr (std::is_same_v<T, float>) {
        pack_lower_triangular_for_inverse(
            source, source_ld, destination, destination_ld, n, stream);
    } else {
        pack_lower_triangular_for_inverse_double(
            source, source_ld, destination, destination_ld, n, stream);
    }

    return;
}

template <typename T>
void dispatch_zero_strict_upper(T* matrix,
                                int lda,
                                int n,
                                hipStream_t stream)
{
    if constexpr (std::is_same_v<T, float>) {
        zero_strict_upper_triangle(matrix, lda, n, stream);
    } else {
        zero_strict_upper_triangle_double(matrix, lda, n, stream);
    }

    return;
}

template <typename T>
void dispatch_trtri(rocblas_handle handle,
                    int n,
                    T* matrix,
                    int lda,
                    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_strtri(
            handle,
            rocblas_fill_lower,
            rocblas_diagonal_non_unit,
            n,
            matrix,
            lda,
            info));
    } else {
        ROCBLAS_CHECK(rocsolver_dtrtri(
            handle,
            rocblas_fill_lower,
            rocblas_diagonal_non_unit,
            n,
            matrix,
            lda,
            info));
    }

    return;
}

template <typename T>
void dispatch_inverse_panel_gemm(rocblas_handle handle,
                                 int rows,
                                 int cols,
                                 const T* panel,
                                 int panel_ld,
                                 const T* inverse,
                                 int inverse_ld,
                                 T* workspace,
                                 int workspace_ld)
{
    const T one = T(1);
    const T zero = T(0);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            rows,
            cols,
            cols,
            &one,
            panel,
            panel_ld,
            inverse,
            inverse_ld,
            &zero,
            workspace,
            workspace_ld));
    } else {
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            rows,
            cols,
            cols,
            &one,
            panel,
            panel_ld,
            inverse,
            inverse_ld,
            &zero,
            workspace,
            workspace_ld));
    }

    return;
}

template <typename T>
void solve_panel_with_inverse(rocblas_handle handle,
                              hipStream_t stream,
                              const T* diagonal,
                              int diagonal_ld,
                              T* panel,
                              int panel_ld,
                              int rows,
                              int cols,
                              T* inverse_workspace,
                              int inverse_ld,
                              T* panel_workspace,
                              int panel_workspace_rows,
                              int* local_info,
                              int* global_info,
                              int global_offset,
                              StageBreakdownProfiler* profiler)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    run_stage(
        profiler,
        stream,
        "Pack+TRTRI",
        potrf_work(cols),
        [&]() {
            dispatch_pack_lower_triangular(
                diagonal,
                diagonal_ld,
                inverse_workspace,
                inverse_ld,
                cols,
                stream);
            dispatch_trtri(
                handle, cols, inverse_workspace, inverse_ld, local_info);
            record_first_factorization_error(
                local_info, global_info, global_offset, stream);

            // TRTRI only defines the selected triangle. Dense GEMM requires
            // the unstored half to be explicitly zero.
            dispatch_zero_strict_upper(
                inverse_workspace, inverse_ld, cols, stream);
        });

    for (int row_offset = 0; row_offset < rows;
         row_offset += panel_workspace_rows) {
        const int tile_rows =
            std::min(panel_workspace_rows, rows - row_offset);
        T* panel_tile = panel + row_offset;

        run_stage(
            profiler,
            stream,
            "Panel GEMM",
            panel_gemm_work(tile_rows, cols),
            [&]() {
                dispatch_inverse_panel_gemm(
                    handle,
                    tile_rows,
                    cols,
                    panel_tile,
                    panel_ld,
                    inverse_workspace,
                    inverse_ld,
                    panel_workspace,
                    panel_workspace_rows);
            });

        run_stage(
            profiler,
            stream,
            "Copyback",
            0.0,
            [&]() {
                HIP_CHECK(hipMemcpy2DAsync(
                    panel_tile,
                    static_cast<size_t>(panel_ld) * sizeof(T),
                    panel_workspace,
                    static_cast<size_t>(panel_workspace_rows) * sizeof(T),
                    static_cast<size_t>(tile_rows) * sizeof(T),
                    cols,
                    hipMemcpyDeviceToDevice,
                stream));
            });
    }

    return;
}

template <typename T>
void update_next_inner_panel(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int outer,
    int panel_start,
    int outer_end,
    int b,
    int gemm_m_tile,
    int gemm_m_threshold,
    int gemm_k_threshold)
{
    if (panel_start >= outer_end) {
        return;
    }

    const int panel_cols = std::min(b, outer_end - panel_start);
    const int rows = n - panel_start;
    const int k = panel_start - outer;

    const bool use_m_tiling =
        gemm_m_tile > 0 && rows > gemm_m_threshold && k >= gemm_k_threshold;

    dispatch_gemm_nt_update_m_tiled(
        handle,
        rows,
        panel_cols,
        k,
        block_ptr(A, lda, panel_start, outer),
        lda,
        block_ptr(A, lda, panel_start, outer),
        lda,
        block_ptr(A, lda, panel_start, panel_start),
        lda,
        use_m_tiling ? gemm_m_tile : 0);

    return;
}

template <typename T>
void prepare_next_outer_panel(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int next_outer,
    int nb,
    int syrk_k_tile)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(nb, n - next_outer);
    const int below_start = next_outer + next_size;
    const int below_rows = n - below_start;

    dispatch_syrk_lower_update_kblocked(
        handle,
        next_size,
        next_outer,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, next_outer, next_outer),
        lda,
        syrk_k_tile);

    dispatch_gemm_nt_update(
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

    return;
}

template <typename T>
void prepare_next_outer_panel_2d(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int next_outer,
    int nb,
    int syrk_k_tile,
    int column_tile,
    int gemm_m_tile,
    int gemm_k_tile)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(nb, n - next_outer);
    column_tile = column_tile > 0
        ? std::min(column_tile, next_size)
        : next_size;
    gemm_k_tile = gemm_k_tile > 0
        ? std::min(gemm_k_tile, next_outer)
        : next_outer;

    // Partition the lower part of the next block column. Each diagonal tile
    // uses SYRK; the rectangle below it uses GEMM, tiled only in M and K.
    for (int col_offset = 0; col_offset < next_size;
         col_offset += column_tile) {
        const int cols = std::min(column_tile, next_size - col_offset);
        const int col = next_outer + col_offset;

        dispatch_syrk_lower_update_kblocked(
            handle,
            cols,
            next_outer,
            block_ptr(A, lda, col, 0),
            lda,
            block_ptr(A, lda, col, col),
            lda,
            syrk_k_tile);

        const int row_start = col + cols;
        const int rows = n - row_start;
        if (rows <= 0) {
            continue;
        }

        const int effective_m_tile = gemm_m_tile > 0
            ? std::min(gemm_m_tile, rows)
            : rows;

        for (int k_offset = 0; k_offset < next_outer;
             k_offset += gemm_k_tile) {
            const int k_chunk =
                std::min(gemm_k_tile, next_outer - k_offset);

            for (int row_offset = 0; row_offset < rows;
                 row_offset += effective_m_tile) {
                const int tile_rows =
                    std::min(effective_m_tile, rows - row_offset);

                dispatch_gemm_nt_update(
                    handle,
                    tile_rows,
                    cols,
                    k_chunk,
                    block_ptr(A, lda, row_start + row_offset, k_offset),
                    lda,
                    block_ptr(A, lda, col, k_offset),
                    lda,
                    block_ptr(A, lda, row_start + row_offset, col),
                    lda);
            }
        }
    }

    return;
}

template <typename T>
void double_blocked_potrf_impl(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int nb,
    int b,
    int* info,
    bool use_double_blocked_trsm,
    int syrk_k_tile,
    int inner_gemm_m_tile,
    int inner_gemm_m_threshold,
    int inner_gemm_k_threshold,
    int outer_column_tile = 0,
    int outer_gemm_m_tile = 0,
    int outer_gemm_k_tile = 0,
    T* inverse_workspace = nullptr,
    int inverse_ld = 0,
    T* panel_workspace = nullptr,
    int panel_workspace_rows = 0,
    int* local_info = nullptr,
    bool require_inverse_panel = false,
    StageBreakdownProfiler* profiler = nullptr)
{
    if (n <= 0) {
        return;
    }

    if (nb <= 0) {
        nb = n;
    }

    if (b <= 0) {
        b = nb;
    }

    b = std::min(b, nb);

    const bool use_inverse_panel =
        inverse_workspace != nullptr && inverse_ld >= b &&
        panel_workspace != nullptr && panel_workspace_rows > 0 &&
        local_info != nullptr && local_info != info;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    if (require_inverse_panel && !use_inverse_panel) {
        HIP_CHECK(hipMemsetAsync(info, 0xff, sizeof(int), stream));
        return;
    }

    if (use_inverse_panel) {
        HIP_CHECK(hipMemsetAsync(info, 0, sizeof(int), stream));
    }

    for (int outer = 0; outer < n; outer += nb) {
        const int outer_size = std::min(nb, n - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += b) {
            const int inner_size = std::min(b, outer_end - inner);
            const int panel_start = inner + inner_size;
            const int trailing_rows = n - panel_start;

            T* A_ii = block_ptr(A, lda, inner, inner);

            int* factor_info = use_inverse_panel ? local_info : info;
            run_stage(
                profiler,
                stream,
                "POTRF",
                potrf_work(inner_size),
                [&]() {
                    rocsolver_potrf(
                        handle, inner_size, A_ii, lda, factor_info);

                    if (use_inverse_panel) {
                        record_first_factorization_error(
                            factor_info, info, inner, stream);
                    }
                });

            if (use_inverse_panel) {
                solve_panel_with_inverse(
                    handle,
                    stream,
                    A_ii,
                    lda,
                    block_ptr(A, lda, panel_start, inner),
                    lda,
                    trailing_rows,
                    inner_size,
                    inverse_workspace,
                    inverse_ld,
                    panel_workspace,
                    panel_workspace_rows,
                    local_info,
                    info,
                    inner,
                    profiler);
            } else if (use_double_blocked_trsm) {
                dispatch_double_blocked_trsm_right_lower_transpose(
                    handle,
                    trailing_rows,
                    inner_size,
                    A_ii,
                    lda,
                    block_ptr(A, lda, panel_start, inner),
                    lda);
            } else {
                dispatch_trsm_right_lower_transpose(
                    handle,
                    trailing_rows,
                    inner_size,
                    A_ii,
                    lda,
                    block_ptr(A, lda, panel_start, inner),
                    lda);
            }

            if (panel_start < outer_end) {
                const int panel_cols =
                    std::min(b, outer_end - panel_start);
                const int inner_k = panel_start - outer;
                run_stage(
                    profiler,
                    stream,
                    "Inner Update",
                    inner_update_work(
                        n - panel_start, panel_cols, inner_k),
                    [&]() {
                        update_next_inner_panel(
                            handle,
                            A,
                            lda,
                            n,
                            outer,
                            panel_start,
                            outer_end,
                            b,
                            inner_gemm_m_tile,
                            inner_gemm_m_threshold,
                            inner_gemm_k_threshold);
                    });
            }
        }

        if (outer_end < n) {
            const int next_size = std::min(nb, n - outer_end);
            const int below_rows = n - outer_end - next_size;
            run_stage(
                profiler,
                stream,
                "Outer Update",
                outer_update_work(next_size, below_rows, outer_end),
                [&]() {
                    if (outer_column_tile > 0) {
                        prepare_next_outer_panel_2d(
                            handle,
                            A,
                            lda,
                            n,
                            outer_end,
                            nb,
                            syrk_k_tile,
                            outer_column_tile,
                            outer_gemm_m_tile,
                            outer_gemm_k_tile);
                    } else {
                        prepare_next_outer_panel(
                            handle,
                            A,
                            lda,
                            n,
                            outer_end,
                            nb,
                            syrk_k_tile);
                    }
                });
        }
    }

    return;
}

} // namespace

extern "C" {

void double_blocked_potrf(rocblas_handle handle,
                          float* A,
                          int lda,
                          int n,
                          int nb,
                          int b,
                          int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, 0, 0, 0, 0);

    return;
}

void double_blocked_potrf_dbtrsm(rocblas_handle handle,
                                 float* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int b,
                                 int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, true, 0, 0, 0, 0);

    return;
}

void double_blocked_potrf_kblocked_syrk(rocblas_handle handle,
                                        float* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int b,
                                        int syrk_k_tile,
                                        int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, syrk_k_tile, 0, 0, 0);

    return;
}

void double_blocked_potrf_kblocked_syrk_adaptive_gemm(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold);

    return;
}

void double_blocked_potrf_outer_panel_2d(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int column_tile,
    int gemm_m_tile,
    int gemm_k_tile,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold,
        column_tile,
        gemm_m_tile,
        gemm_k_tile);

    return;
}

void double_blocked_potrf_inverse_panel(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    float* inverse_workspace,
    int inverse_ld,
    float* panel_workspace,
    int panel_workspace_rows,
    int* local_info,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold,
        0,
        0,
        0,
        inverse_workspace,
        inverse_ld,
        panel_workspace,
        panel_workspace_rows,
        local_info,
        true);

    return;
}

void double_blocked_potrf_double(rocblas_handle handle,
                                 double* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int b,
                                 int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, 0, 0, 0, 0);

    return;
}

void double_blocked_potrf_double_dbtrsm(rocblas_handle handle,
                                        double* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int b,
                                        int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, true, 0, 0, 0, 0);

    return;
}

void double_blocked_potrf_double_kblocked_syrk(rocblas_handle handle,
                                               double* A,
                                               int lda,
                                               int n,
                                               int nb,
                                               int b,
                                               int syrk_k_tile,
                                               int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, syrk_k_tile, 0, 0, 0);

    return;
}

void double_blocked_potrf_double_kblocked_syrk_adaptive_gemm(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold);

    return;
}

void double_blocked_potrf_double_outer_panel_2d(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int column_tile,
    int gemm_m_tile,
    int gemm_k_tile,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold,
        column_tile,
        gemm_m_tile,
        gemm_k_tile);

    return;
}

void double_blocked_potrf_double_inverse_panel(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    double* inverse_workspace,
    int inverse_ld,
    double* panel_workspace,
    int panel_workspace_rows,
    int* local_info,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold,
        0,
        0,
        0,
        inverse_workspace,
        inverse_ld,
        panel_workspace,
        panel_workspace_rows,
        local_info,
        true);

    return;
}

} // extern "C"

void double_blocked_potrf_inverse_panel_profiled(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    float* inverse_workspace,
    int inverse_ld,
    float* panel_workspace,
    int panel_workspace_rows,
    int* local_info,
    int* info,
    StageBreakdownProfiler& profiler)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold,
        0,
        0,
        0,
        inverse_workspace,
        inverse_ld,
        panel_workspace,
        panel_workspace_rows,
        local_info,
        true,
        &profiler);

    return;
}
