#include "potrf/benchmark/potrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "potrf/double_blocked_potrf.h"

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
    int gemm_m_tile,
    int gemm_m_threshold,
    int gemm_k_threshold,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_potrf_kblocked_syrk_adaptive_gemm_configured(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            gemm_m_tile,
            gemm_m_threshold,
            gemm_k_threshold,
            info);
    } else {
        double_blocked_potrf_double_kblocked_syrk_adaptive_gemm_configured(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            syrk_k_tile,
            gemm_m_tile,
            gemm_m_threshold,
            gemm_k_threshold,
            info);
    }

    return;
}

template <typename T>
void run_double_blocked_pair(PotrfBenchmarkContext<T>& context, int nb, int b)
{
    const std::string nb_b = std::to_string(nb) + "/" + std::to_string(b);
    const int syrk_k_tile = context.syrk_k_tile > 0
        ? context.syrk_k_tile
        : nb;

    if (context.test_double_basic) {
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
    }

    if (context.test_double_dbtrsm) {
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
    }

    if (context.test_double_kbsyrk) {
        context.runner.execute(
            "DoubleBlock+KBSYRK",
            nb_b + "/" + std::to_string(syrk_k_tile),
            context.reset,
            [&]() {
                call_double_blocked_potrf_kblocked_syrk<T>(
                    context.handle,
                    context.d_A,
                    context.n,
                    context.lda,
                    nb,
                    b,
                    syrk_k_tile,
                    context.d_info);
            },
            context.verify,
            context.calc_gflops);
    }

    if (context.test_double_mgemm) {
        const std::string parameters = nb_b
            + "/" + std::to_string(syrk_k_tile)
            + "/" + std::to_string(context.gemm_m_tile)
            + "/" + std::to_string(context.gemm_m_threshold)
            + "/" + std::to_string(context.gemm_k_threshold);
        context.runner.execute(
            "DoubleBlock+KBSYRK+MGEMM",
            parameters,
            context.reset,
            [&]() {
                call_double_blocked_potrf_kblocked_syrk_adaptive_gemm<T>(
                    context.handle,
                    context.d_A,
                    context.n,
                    context.lda,
                    nb,
                    b,
                    syrk_k_tile,
                    context.gemm_m_tile,
                    context.gemm_m_threshold,
                    context.gemm_k_threshold,
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
