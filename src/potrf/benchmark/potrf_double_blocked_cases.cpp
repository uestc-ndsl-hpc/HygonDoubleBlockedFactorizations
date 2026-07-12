#include "potrf/benchmark/potrf_benchmark_common.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>

#include "potrf/double_blocked_potrf.h"
#include "potrf/tiled_right_looking_potrf.h"
#include "utils/memory/device_buffer.h"

namespace potrf_benchmark {
namespace {

template <typename T>
void call_double_blocked_potrf(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int b,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf(handle, A, lda, n, nb, b, info);
    } else {
        double_blocked_potrf_double(handle, A, lda, n, nb, b, info);
    }

    return;
}

template <typename T>
void call_double_blocked_potrf_dbtrsm(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int b,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf_dbtrsm(handle, A, lda, n, nb, b, info);
    } else {
        double_blocked_potrf_double_dbtrsm(handle, A, lda, n, nb, b, info);
    }

    return;
}

template <typename T>
void call_double_blocked_potrf_kblocked_syrk(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int b,
    int syrk_k_tile,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf_kblocked_syrk(
            handle, A, lda, n, nb, b, syrk_k_tile, info);
    } else {
        double_blocked_potrf_double_kblocked_syrk(
            handle, A, lda, n, nb, b, syrk_k_tile, info);
    }

    return;
}

template <typename T>
void call_double_blocked_potrf_kblocked_syrk_adaptive_gemm(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int b,
    int syrk_k_tile,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf_kblocked_syrk_adaptive_gemm(
            handle, A, lda, n, nb, b, syrk_k_tile, info);
    } else {
        double_blocked_potrf_double_kblocked_syrk_adaptive_gemm(
            handle, A, lda, n, nb, b, syrk_k_tile, info);
    }

    return;
}

template <typename T>
void call_double_blocked_potrf_outer_panel_2d(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int b,
    int syrk_k_tile,
    int column_tile,
    int gemm_m_tile,
    int gemm_k_tile,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf_outer_panel_2d(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            column_tile,
            gemm_m_tile,
            gemm_k_tile,
            info);
    } else {
        double_blocked_potrf_double_outer_panel_2d(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            column_tile,
            gemm_m_tile,
            gemm_k_tile,
            info);
    }

    return;
}

template <typename T>
void call_double_blocked_potrf_inverse_panel(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int b,
    int syrk_k_tile,
    T* inverse_workspace,
    int inverse_ld,
    T* panel_workspace,
    int panel_workspace_rows,
    int* local_info,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf_inverse_panel(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            inverse_workspace,
            inverse_ld,
            panel_workspace,
            panel_workspace_rows,
            local_info,
            info);
    } else {
        double_blocked_potrf_double_inverse_panel(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            inverse_workspace,
            inverse_ld,
            panel_workspace,
            panel_workspace_rows,
            local_info,
            info);
    }

    return;
}

template <typename T>
void call_tiled_right_looking_double_blocked_potrf(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int outer_block_size,
    int update_tile_size,
    int potrf_leaf_size,
    int trsm_leaf_size,
    int update_k_tile,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        tiled_right_looking_double_blocked_potrf(
            handle,
            A,
            lda,
            n,
            outer_block_size,
            update_tile_size,
            potrf_leaf_size,
            trsm_leaf_size,
            update_k_tile,
            info);
    } else {
        tiled_right_looking_double_blocked_potrf_double(
            handle,
            A,
            lda,
            n,
            outer_block_size,
            update_tile_size,
            potrf_leaf_size,
            trsm_leaf_size,
            update_k_tile,
            info);
    }

    return;
}

template <typename T>
void run_double_blocked_pair(PotrfBenchmarkContext<T>& context, int nb, int b)
{
    const std::string nb_b = std::to_string(nb) + "/" + std::to_string(b);

    context.runner.execute(
        "DoubleBlock",
        nb_b,
        context.reset,
        [&]() {
            call_double_blocked_potrf<T>(
                context.handle,
                context.d_A,
                context.n,
                context.lda,
                nb,
                b,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    context.runner.execute(
        "DoubleBlock+DBTRSM",
        nb_b + "/1024",
        context.reset,
        [&]() {
            call_double_blocked_potrf_dbtrsm<T>(
                context.handle,
                context.d_A,
                context.n,
                context.lda,
                nb,
                b,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    context.runner.execute(
        "DoubleBlock+KBSYRK",
        nb_b + "/" + std::to_string(nb),
        context.reset,
        [&]() {
            call_double_blocked_potrf_kblocked_syrk<T>(
                context.handle,
                context.d_A,
                context.n,
                context.lda,
                nb,
                b,
                nb,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    context.runner.execute(
        "DoubleBlock+KBSYRK+MGEMM",
        nb_b + "/" + std::to_string(nb) + "/8192",
        context.reset,
        [&]() {
            call_double_blocked_potrf_kblocked_syrk_adaptive_gemm<T>(
                context.handle,
                context.d_A,
                context.n,
                context.lda,
                nb,
                b,
                nb,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    {
        constexpr int inverse_panel_row_tile = 8192;
        const int panel_workspace_rows =
            std::min(inverse_panel_row_tile, context.n);
        DeviceBuffer<T> inverse_workspace(
            static_cast<size_t>(b) * static_cast<size_t>(b));
        DeviceBuffer<T> panel_workspace(
            static_cast<size_t>(panel_workspace_rows) *
            static_cast<size_t>(b));
        DeviceBuffer<int> inverse_local_info(1);

        context.runner.execute(
            "DoubleBlock+InvPanel",
            nb_b + "/" + std::to_string(nb) + "/" +
                std::to_string(panel_workspace_rows),
            context.reset,
            [&]() {
                call_double_blocked_potrf_inverse_panel<T>(
                    context.handle,
                    context.d_A,
                    context.n,
                    context.lda,
                    nb,
                    b,
                    nb,
                    inverse_workspace.get(),
                    b,
                    panel_workspace.get(),
                    panel_workspace_rows,
                    inverse_local_info.get(),
                    context.d_info);
            },
            context.verify,
            context.calc_gflops);
    }

    constexpr int outer_gemm_m_tile = 8192;
    constexpr int outer_gemm_k_tile = 8192;

    for (int column_tile : context.outer_panel_column_tiles) {
        if (column_tile <= 0 || column_tile > nb) {
            continue;
        }

        const std::string parameters =
            nb_b + "/" + std::to_string(nb) + "/" +
            std::to_string(column_tile) + "/" +
            std::to_string(outer_gemm_m_tile) + "/" +
            std::to_string(outer_gemm_k_tile);

        context.runner.execute(
            "DoubleBlock+OuterPanel2D",
            parameters,
            context.reset,
            [&]() {
                call_double_blocked_potrf_outer_panel_2d<T>(
                    context.handle,
                    context.d_A,
                    context.n,
                    context.lda,
                    nb,
                    b,
                    nb,
                    column_tile,
                    outer_gemm_m_tile,
                    outer_gemm_k_tile,
                    context.d_info);
            },
            context.verify,
            context.calc_gflops);
    }

    const std::vector<int> default_update_tiles = {
        std::min(4096, nb)};
    const std::vector<int>& update_tiles =
        context.tiled_rl_update_tiles.empty()
        ? default_update_tiles
        : context.tiled_rl_update_tiles;

    constexpr int trsm_leaf_size = 1024;
    const int update_k_tile = std::min(8192, nb);

    for (int update_tile_size : update_tiles) {
        if (update_tile_size <= 0) {
            continue;
        }

        const std::string parameters =
            nb_b + "/" + std::to_string(update_tile_size) + "/" +
            std::to_string(trsm_leaf_size) + "/" +
            std::to_string(update_k_tile);

        context.runner.execute(
            "DoubleBlock+TiledRL",
            parameters,
            context.reset,
            [&]() {
                call_tiled_right_looking_double_blocked_potrf<T>(
                    context.handle,
                    context.d_A,
                    context.n,
                    context.lda,
                    nb,
                    update_tile_size,
                    b,
                    trsm_leaf_size,
                    update_k_tile,
                    context.d_info);
            },
            context.verify,
            context.calc_gflops);
    }

    return;
}

} // namespace

template <typename T>
void run_double_blocked_cases(PotrfBenchmarkContext<T>& context)
{
    if (context.micro_blocks.empty()) {
        const int osla_nb = context.n / 2;
        const int osla_b = context.n / 4;

        run_double_blocked_pair(context, osla_nb, osla_b);

        return;
    }

    const std::vector<int>& double_nbs = context.double_nbs.empty()
        ? context.nbs
        : context.double_nbs;

    for (int nb : double_nbs) {
        for (int b : context.micro_blocks) {
            if (nb <= 0 || b <= 0 || b > nb) {
                continue;
            }

            run_double_blocked_pair(context, nb, b);
        }
    }

    return;
}

template void run_double_blocked_cases<float>(PotrfBenchmarkContext<float>& context);
template void run_double_blocked_cases<double>(PotrfBenchmarkContext<double>& context);

} // namespace potrf_benchmark
