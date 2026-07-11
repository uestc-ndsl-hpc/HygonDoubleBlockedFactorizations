#include "syrk/benchmark/syrk_benchmark_common.h"

#include <algorithm>
#include <string>
#include <vector>

#include "syrk/kblocked_syrk.h"

namespace syrk_benchmark {

void run_kblocked_cases(SyrkBenchmarkContext& context)
{
    std::vector<int> k_tiles = context.k_tiles;
    if (k_tiles.empty()) {
        k_tiles.push_back(std::min(8192, context.k));
    }

    for (int k_tile : k_tiles) {
        if (k_tile <= 0) {
            continue;
        }

        context.runner.execute(
            "K-blocked (FP32)",
            std::to_string(k_tile),
            context.reset,
            [&]() {
                syrk_lower_update_kblocked(
                    context.handle,
                    context.n,
                    context.k,
                    context.d_A,
                    context.lda,
                    context.d_C,
                    context.ldc,
                    k_tile);
            },
            context.verify,
            context.calc_gflops);
    }

    return;
}

} // namespace syrk_benchmark
