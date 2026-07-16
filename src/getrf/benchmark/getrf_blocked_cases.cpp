#include "getrf/benchmark/getrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "getrf/blocked_getrf.h"
#include "getrf/blocked_getrf_npvt.h"

namespace getrf_benchmark {
namespace {

template <typename T>
void call_blocked_getrf(rocblas_handle handle,
                        T* A,
                        int lda,
                        int n,
                        int nb,
                        bool pivoting,
                        int* ipiv,
                        int* info)
{
    if (pivoting) {
        if constexpr (std::is_same_v<T, float>) {
            blocked_right_looking_getrf(
                handle, A, lda, n, nb, ipiv, info);
        } else {
            blocked_right_looking_getrf_double(
                handle, A, lda, n, nb, ipiv, info);
        }
    } else {
        if constexpr (std::is_same_v<T, float>) {
            blocked_right_looking_getrf_npvt(
                handle, A, lda, n, nb, info);
        } else {
            blocked_right_looking_getrf_npvt_double(
                handle, A, lda, n, nb, info);
        }
    }

    return;
}

} // namespace

template <typename T>
void run_blocked_cases(GetrfBenchmarkContext<T>& context)
{
    for (int nb : context.nbs) {
        if (nb <= 0) {
            continue;
        }

        context.runner.execute(
            context.pivoting ? "Blocked-RL-LU" : "Blocked-RL-LU-NPVT",
            std::to_string(nb),
            context.reset,
            [&]() {
                call_blocked_getrf(
                    context.handle,
                    context.d_A,
                    context.lda,
                    context.n,
                    nb,
                    context.pivoting,
                    context.d_ipiv,
                    context.d_info);
            },
            context.verify,
            context.calc_gflops);
    }

    return;
}

template void run_blocked_cases<float>(GetrfBenchmarkContext<float>& context);
template void run_blocked_cases<double>(
    GetrfBenchmarkContext<double>& context);

} // namespace getrf_benchmark
