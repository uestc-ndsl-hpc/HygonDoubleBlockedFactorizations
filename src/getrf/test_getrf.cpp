#include <cstdlib>
#include <cstring>

#include "getrf/benchmark/getrf_benchmark_common.h"

namespace {

enum class BlockParameter {
    Nb,
    DoubleNb,
    MicroBlock
};

} // namespace

int main(int argc, char** argv)
{
    getrf_benchmark::GetrfBenchmarkOptions options;

    if (argc >= 2) {
        options.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        options.precision_type = std::atoi(argv[2]);
    }
    if (argc >= 4 && std::atoi(argv[3]) != 0) {
        options.error_mode = ErrorMode::Backward;
    }

    BlockParameter parameter = BlockParameter::Nb;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--double-nb") == 0) {
            parameter = BlockParameter::DoubleNb;
            continue;
        }
        if (std::strcmp(argv[i], "--b") == 0) {
            parameter = BlockParameter::MicroBlock;
            continue;
        }

        const int value = std::atoi(argv[i]);
        if (parameter == BlockParameter::DoubleNb) {
            options.double_nbs.push_back(value);
        } else if (parameter == BlockParameter::MicroBlock) {
            options.micro_blocks.push_back(value);
        } else {
            options.nbs.push_back(value);
        }
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(2048);
    }

    getrf_benchmark::run_getrf_benchmarks(options);

    return 0;
}
