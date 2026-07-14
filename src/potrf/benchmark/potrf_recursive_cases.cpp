#include "potrf/benchmark/potrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include "potrf/fast_recursive_potrf.h"
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

    return;
}

template <typename T>
void call_recursive_potrf_tiled_trsm(
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
        if (precision_type != 2) {
            recursive_potrf_tiled_trsm(handle, pool, A, lda, n, nb, info);
        }
    }

    return;
}

template <typename T>
void call_fast_recursive_potrf(rocblas_handle handle,
                               T* A,
                               int n,
                               int lda,
                               int leaf_size,
                               int base_nb,
                               int strassen_min_size,
                               int syrk_k_tile,
                               int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        fast_recursive_strassen_potrf(
            handle,
            A,
            lda,
            n,
            leaf_size,
            base_nb,
            strassen_min_size,
            syrk_k_tile,
            info);
    } else {
        fast_recursive_strassen_potrf_double(
            handle,
            A,
            lda,
            n,
            leaf_size,
            base_nb,
            strassen_min_size,
            syrk_k_tile,
            info);
    }

    return;
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

        if constexpr (std::is_same_v<T, float>) {
            if (context.precision_type != 2) {
                context.runner.execute(
                    "RecTiledTRSM",
                    std::to_string(nb),
                    context.reset,
                    [&]() {
                        call_recursive_potrf_tiled_trsm<T>(
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
    }

    return;
}

template <typename T>
void run_fast_recursive_case(PotrfBenchmarkContext<T>& context)
{
    if (context.precision_type == 2) {
        return;
    }

    const std::string parameters =
        std::to_string(context.fast_recursive_leaf) + "/" +
        std::to_string(context.fast_recursive_base_nb) + "/" +
        std::to_string(context.strassen_min_size) + "/" +
        std::to_string(context.fast_recursive_syrk_k_tile);

    context.runner.execute(
        "FastRecursiveStrassen",
        parameters,
        context.reset,
        [&]() {
            call_fast_recursive_potrf<T>(
                context.handle,
                context.d_A,
                context.n,
                context.lda,
                context.fast_recursive_leaf,
                context.fast_recursive_base_nb,
                context.strassen_min_size,
                context.fast_recursive_syrk_k_tile,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    return;
}

template void run_recursive_cases<float>(PotrfBenchmarkContext<float>& context);
template void run_recursive_cases<double>(PotrfBenchmarkContext<double>& context);
template void run_fast_recursive_case<float>(PotrfBenchmarkContext<float>& context);
template void run_fast_recursive_case<double>(PotrfBenchmarkContext<double>& context);

} // namespace potrf_benchmark
