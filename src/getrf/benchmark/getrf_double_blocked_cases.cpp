#include "getrf/benchmark/getrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "getrf/double_blocked_getrf.h"
#include "getrf/double_blocked_getrf_npvt.h"

namespace getrf_benchmark {
namespace {

template <typename T>
void call_double_blocked_getrf(rocblas_handle handle,
                               T* A,
                               int lda,
                               int n,
                               int nb,
                               int b,
                               int trsm_leaf,
                               bool pivoting,
                               int* ipiv,
                               int* info)
{
    if (pivoting) {
        if constexpr (std::is_same_v<T, float>) {
            double_blocked_getrf(
                handle, A, lda, n, nb, b, trsm_leaf, ipiv, info);
        } else {
            double_blocked_getrf_double(
                handle, A, lda, n, nb, b, trsm_leaf, ipiv, info);
        }
    } else {
        if constexpr (std::is_same_v<T, float>) {
            double_blocked_getrf_npvt(handle, A, lda, n, nb, b, info);
        } else {
            double_blocked_getrf_npvt_double(
                handle, A, lda, n, nb, b, info);
        }
    }

    return;
}

template <typename T>
void call_double_blocked_getrf_batched_pivot(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int nb,
    int b,
    int trsm_leaf,
    int inner_gemm_m_tile,
    int* ipiv,
    int* info,
    getrf_batched_pivot_workspace* workspace)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_getrf_batched_pivot(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            trsm_leaf,
            inner_gemm_m_tile,
            ipiv,
            info,
            workspace);
    } else {
        double_blocked_getrf_double_batched_pivot(
            handle,
            A,
            lda,
            n,
            nb,
            b,
            trsm_leaf,
            inner_gemm_m_tile,
            ipiv,
            info,
            workspace);
    }

    return;
}

} // namespace

template <typename T>
void run_double_blocked_cases(GetrfBenchmarkContext<T>& context)
{
    const std::vector<int> default_double_nbs = {context.n / 2};
    const std::vector<int> default_micro_blocks = {context.n / 4};
    const std::vector<int>& double_nbs = context.double_nbs.empty()
        ? default_double_nbs
        : context.double_nbs;
    const std::vector<int>& micro_blocks = context.micro_blocks.empty()
        ? default_micro_blocks
        : context.micro_blocks;

    for (int nb : double_nbs) {
        for (int b : micro_blocks) {
            if (nb <= 0 || b <= 0 || b > nb) {
                continue;
            }

            const std::string parameters =
                std::to_string(nb) + "/" + std::to_string(b)
                + (context.pivoting
                    ? "/" + std::to_string(context.trsm_leaf)
                    : "");
            if (context.test_double_basic) {
                context.runner.execute(
                    context.pivoting
                        ? "DoubleBlock-LU"
                        : "DoubleBlock-LU-NPVT",
                    parameters,
                    context.reset,
                    [&]() {
                        call_double_blocked_getrf(
                            context.handle,
                            context.d_A,
                            context.lda,
                            context.n,
                            nb,
                            b,
                            context.trsm_leaf,
                            context.pivoting,
                            context.d_ipiv,
                            context.d_info);
                    },
                    context.verify,
                    context.calc_gflops);
            }

            if (context.pivoting) {
                const std::string batched_parameters = parameters + "/"
                    + std::to_string(context.pivot_column_tile);
                if (context.test_batched_pivot) {
                    context.runner.execute(
                        "DoubleBlock-LU+BatchedPivot",
                        batched_parameters,
                        context.reset,
                        [&]() {
                            call_double_blocked_getrf_batched_pivot(
                                context.handle,
                                context.d_A,
                                context.lda,
                                context.n,
                                nb,
                                b,
                                context.trsm_leaf,
                                0,
                                context.d_ipiv,
                                context.d_info,
                                context.batched_pivot_workspace);
                        },
                        context.verify,
                        context.calc_gflops);
                }

                if (context.test_mgemm && context.inner_gemm_m_tile > 0) {
                    const std::string tiled_parameters = batched_parameters
                        + "/" + std::to_string(context.inner_gemm_m_tile);
                    context.runner.execute(
                        "DoubleBlock-LU+BatchedPivot+MGEMM",
                        tiled_parameters,
                        context.reset,
                        [&]() {
                            call_double_blocked_getrf_batched_pivot(
                                context.handle,
                                context.d_A,
                                context.lda,
                                context.n,
                                nb,
                                b,
                                context.trsm_leaf,
                                context.inner_gemm_m_tile,
                                context.d_ipiv,
                                context.d_info,
                                context.batched_pivot_workspace);
                        },
                        context.verify,
                        context.calc_gflops);
                }
            }
        }
    }

    return;
}

template void run_double_blocked_cases<float>(
    GetrfBenchmarkContext<float>& context);
template void run_double_blocked_cases<double>(
    GetrfBenchmarkContext<double>& context);

} // namespace getrf_benchmark
