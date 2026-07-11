#include "potrf/benchmark/potrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "potrf/double_blocked_potrf.h"
#include "potrf/hierarchical_double_blocked_potrf.h"

namespace potrf_benchmark {
namespace {

constexpr int kDefaultTailBlockSize = 2048;
constexpr int kDefaultHierarchicalPanelSize = 2048;
constexpr int kHierarchicalGemmMTile = 8192;

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
void call_double_blocked_potrf_tail_blocked(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int b,
    int syrk_k_tile,
    int tail_block_size,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf_tail_blocked(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            tail_block_size,
            info);
    } else {
        double_blocked_potrf_double_tail_blocked(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            tail_block_size,
            info);
    }

    return;
}

template <typename T>
void run_tail_blocked_case(
    PotrfBenchmarkContext<T>& context,
    int nb,
    int b,
    int tail_block_size)
{
    if (tail_block_size <= 0) {
        return;
    }

    const std::string parameters =
        std::to_string(nb) + "/" + std::to_string(b) + "/" +
        std::to_string(nb) + "/8192/" + std::to_string(tail_block_size);

    context.runner.execute(
        "DoubleBlock+TailRL",
        parameters,
        context.reset,
        [&]() {
            call_double_blocked_potrf_tail_blocked<T>(
                context.handle,
                context.d_A,
                context.n,
                context.lda,
                nb,
                b,
                nb,
                tail_block_size,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    return;
}

template <typename T>
void run_hierarchical_case(
    PotrfBenchmarkContext<T>& context,
    int nb,
    int panel_block_size,
    int potrf_leaf_size)
{
    if constexpr (std::is_same_v<T, float>) {
        if (panel_block_size <= 0 || panel_block_size > nb ||
            potrf_leaf_size <= 0 || potrf_leaf_size > panel_block_size) {
            return;
        }

        const std::string parameters =
            std::to_string(nb) + "/" + std::to_string(panel_block_size) +
            "/" + std::to_string(potrf_leaf_size) + "/" +
            std::to_string(nb) + "/" +
            std::to_string(kHierarchicalGemmMTile);

        context.runner.execute(
            "DoubleBlock+Hier",
            parameters,
            context.reset,
            [&]() {
                hierarchical_double_blocked_potrf(
                    context.handle,
                    context.d_A,
                    context.lda,
                    context.n,
                    nb,
                    panel_block_size,
                    potrf_leaf_size,
                    nb,
                    kHierarchicalGemmMTile,
                    context.d_info);
            },
            context.verify,
            context.calc_gflops);
    } else {
        (void)context;
        (void)nb;
        (void)panel_block_size;
        (void)potrf_leaf_size;
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

    if (context.panel_blocks.empty()) {
        run_hierarchical_case(
            context, nb, kDefaultHierarchicalPanelSize, b);
    } else {
        for (int panel_block_size : context.panel_blocks) {
            run_hierarchical_case(context, nb, panel_block_size, b);
        }
    }

    if (context.tail_nbs.empty()) {
        run_tail_blocked_case(
            context, nb, b, kDefaultTailBlockSize);
    } else {
        for (int tail_nb : context.tail_nbs) {
            run_tail_blocked_case(context, nb, b, tail_nb);
        }
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
