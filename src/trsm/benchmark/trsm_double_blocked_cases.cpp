#include "trsm/benchmark/trsm_benchmark_common.h"

#include <algorithm>
#include <string>

#include "trsm/double_blocked_trsm.h"

namespace trsm_benchmark {

void run_double_blocked_cases(TrsmBenchmarkContext& context)
{
    if (context.micro_blocks.empty()) {
        const int osla_nb = std::max(1, context.m / 2);
        const int osla_b = std::max(1, context.m / 4);

        context.runner.execute(
            "DoubleBlock (FP32)",
            std::to_string(osla_nb) + "/" + std::to_string(std::min(osla_b, osla_nb)),
            context.reset,
            [&]() {
                double_blocked_trsm(
                    context.handle,
                    context.d_A,
                    context.lda,
                    context.d_B,
                    context.ldb,
                    context.n,
                    context.m,
                    osla_nb,
                    osla_b);
            },
            context.verify,
            context.calc_gflops);

        return;
    }

    for (int nb : context.nbs) {
        for (int b : context.micro_blocks) {
            if (nb <= 0 || b <= 0 || b > nb) {
                continue;
            }

            context.runner.execute(
                "DoubleBlock (FP32)",
                std::to_string(nb) + "/" + std::to_string(b),
                context.reset,
                [&]() {
                    double_blocked_trsm(
                        context.handle,
                        context.d_A,
                        context.lda,
                        context.d_B,
                        context.ldb,
                        context.n,
                        context.m,
                        nb,
                        b);
                },
                context.verify,
                context.calc_gflops);
        }
    }

    return;
}

} // namespace trsm_benchmark
