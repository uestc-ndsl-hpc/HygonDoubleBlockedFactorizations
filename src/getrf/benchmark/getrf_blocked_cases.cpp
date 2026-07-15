#include "getrf/benchmark/getrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "getrf/blocked_getrf_npvt.h"

namespace getrf_benchmark {
namespace {

template <typename T>
void call_blocked_getrf_npvt(rocblas_handle handle,
                             T* A,
                             int lda,
                             int n,
                             int nb,
                             int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        blocked_right_looking_getrf_npvt(
            handle, A, lda, n, nb, info);
    } else {
        blocked_right_looking_getrf_npvt_double(
            handle, A, lda, n, nb, info);
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
            "Blocked-RL-LU-NPVT",
            std::to_string(nb),
            context.reset,
            [&]() {
                call_blocked_getrf_npvt(
                    context.handle,
                    context.d_A,
                    context.lda,
                    context.n,
                    nb,
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
