#include "potrf/benchmark/potrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "potrf/recursive_potrf.h"

namespace potrf_benchmark {
namespace {

template <typename T>
void call_recursive_potrf(
    rocblas_handle handle,
    StreamPool* pool,
    T* A,
    int n,
    int lda,
    int nb,
    int* info,
    int precision_type)
{
    if constexpr (std::is_same_v<T, float>) {
        if (precision_type == 2) {
            recursive_potrf_mixed_driver(handle, pool, A, lda, n, nb, info);
        } else {
            recursive_potrf(handle, pool, A, lda, n, nb, info);
        }
    } else {
        recursive_potrf_double(handle, pool, A, lda, n, nb, info);
    }
}

} // namespace

template <typename T>
void run_recursive_cases(PotrfBenchmarkContext<T>& context)
{
    for (int nb : context.nbs) {
        std::string method_name =
            (context.precision_type == 2) ? "Mixed-Recursive" : "Recursive";

        context.runner.execute(
            method_name,
            std::to_string(nb),
            context.reset,
            [&]() {
                call_recursive_potrf<T>(
                    context.handle,
                    context.pool,
                    context.d_A,
                    context.n,
                    context.lda,
                    nb,
                    context.d_info,
                    context.precision_type);
            },
            context.verify,
            context.calc_gflops);
    }
}

template void run_recursive_cases<float>(PotrfBenchmarkContext<float>& context);
template void run_recursive_cases<double>(PotrfBenchmarkContext<double>& context);

} // namespace potrf_benchmark
