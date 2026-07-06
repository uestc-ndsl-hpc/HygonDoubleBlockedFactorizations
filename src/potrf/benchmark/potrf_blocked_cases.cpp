#include "potrf/benchmark/potrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "potrf/blocked_potrf.h"

namespace potrf_benchmark {
namespace {

template <typename T>
void call_blocked_right_looking_potrf(
    rocblas_handle handle,
    T* A,
    int n,
    int lda,
    int nb,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        blocked_right_looking_potrf(handle, A, lda, n, nb, info);
    } else {
        blocked_right_looking_potrf_double(handle, A, lda, n, nb, info);
    }

    return;
}

} // namespace

template <typename T>
void run_blocked_cases(PotrfBenchmarkContext<T>& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "Blocked-RL",
            std::to_string(nb),
            context.reset,
            [&]() {
                call_blocked_right_looking_potrf<T>(
                    context.handle,
                    context.d_A,
                    context.n,
                    context.lda,
                    nb,
                    context.d_info);
            },
            context.verify,
            context.calc_gflops);
    }

    return;
}

template void run_blocked_cases<float>(PotrfBenchmarkContext<float>& context);
template void run_blocked_cases<double>(PotrfBenchmarkContext<double>& context);

} // namespace potrf_benchmark
