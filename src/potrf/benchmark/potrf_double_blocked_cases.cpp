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

} // namespace

template <typename T>
void run_double_blocked_cases(PotrfBenchmarkContext<T>& context)
{
    for (int nb : context.nbs) {
        for (int b : context.micro_blocks) {
            if (nb <= 0 || b <= 0 || b > nb) {
                continue;
            }

            context.runner.execute(
                "DoubleBlock",
                std::to_string(nb) + "/" + std::to_string(b),
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
    }

    return;
}

template void run_double_blocked_cases<float>(PotrfBenchmarkContext<float>& context);
template void run_double_blocked_cases<double>(PotrfBenchmarkContext<double>& context);

} // namespace potrf_benchmark
