#include "potrf/benchmark/potrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "potrf/double_blocked_left_looking_potrf.h"

namespace potrf_benchmark {
namespace {

constexpr int kOuterGemmMTile = 8192;

template <typename T>
void call_left_looking(rocblas_handle handle,
                       T* A,
                       int lda,
                       int n,
                       int nb,
                       int b,
                       int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_left_looking_potrf(
            handle, A, lda, n, nb, b, info);
    } else {
        double_blocked_left_looking_potrf_double(
            handle, A, lda, n, nb, b, info);
    }

    return;
}

template <typename T>
void call_left_looking_optimized(rocblas_handle handle,
                                 T* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int b,
                                 int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        double_blocked_left_looking_potrf_optimized(
            handle, A, lda, n, nb, b, nb, kOuterGemmMTile, info);
    } else {
        double_blocked_left_looking_potrf_double_optimized(
            handle, A, lda, n, nb, b, nb, kOuterGemmMTile, info);
    }

    return;
}

template <typename T>
void run_pair(PotrfBenchmarkContext<T>& context, int nb, int b)
{
    const std::string nb_b = std::to_string(nb) + "/" + std::to_string(b);

    context.runner.execute(
        "DoubleBlock-LL",
        nb_b,
        context.reset,
        [&]() {
            call_left_looking<T>(
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

    context.runner.execute(
        "DoubleBlock-LL+MTile",
        nb_b + "/" + std::to_string(kOuterGemmMTile),
        context.reset,
        [&]() {
            call_left_looking_optimized<T>(
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

    return;
}

} // namespace

template <typename T>
void run_double_blocked_left_looking_cases(PotrfBenchmarkContext<T>& context)
{
    const std::vector<int>& double_nbs = context.double_nbs.empty()
        ? context.nbs
        : context.double_nbs;
    const std::vector<int> default_micro_blocks = {512};
    const std::vector<int>& micro_blocks = context.micro_blocks.empty()
        ? default_micro_blocks
        : context.micro_blocks;

    for (int nb : double_nbs) {
        for (int b : micro_blocks) {
            if (nb <= 0 || b <= 0 || b > nb) {
                continue;
            }
            run_pair(context, nb, b);
        }
    }

    return;
}

template void run_double_blocked_left_looking_cases<float>(
    PotrfBenchmarkContext<float>& context);
template void run_double_blocked_left_looking_cases<double>(
    PotrfBenchmarkContext<double>& context);

} // namespace potrf_benchmark
