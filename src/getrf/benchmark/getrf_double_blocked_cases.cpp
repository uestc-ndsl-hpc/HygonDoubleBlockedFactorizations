#include "getrf/benchmark/getrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "getrf/double_blocked_getrf_npvt.h"

namespace getrf_benchmark {
namespace {

template <typename T>
void call_double_blocked_getrf_npvt(rocblas_handle handle,
                                    T* A,
                                    int lda,
                                    int n,
                                    int nb,
                                    int b,
                                    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_getrf_npvt(handle, A, lda, n, nb, b, info);
    } else {
        double_blocked_getrf_npvt_double(
            handle, A, lda, n, nb, b, info);
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
                std::to_string(nb) + "/" + std::to_string(b);
            context.runner.execute(
                "DoubleBlock-LU-NPVT",
                parameters,
                context.reset,
                [&]() {
                    call_double_blocked_getrf_npvt(
                        context.handle,
                        context.d_A,
                        context.lda,
                        context.n,
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

template void run_double_blocked_cases<float>(
    GetrfBenchmarkContext<float>& context);
template void run_double_blocked_cases<double>(
    GetrfBenchmarkContext<double>& context);

} // namespace getrf_benchmark
