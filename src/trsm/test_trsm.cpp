#include <cstdlib>
#include <cstring>

#include "trsm/benchmark/trsm_benchmark_common.h"

namespace {

bool is_bool_flag(const char* arg)
{
    return std::strcmp(arg, "0") == 0 || std::strcmp(arg, "1") == 0;
}

} // namespace

int main(int argc, char** argv)
{
    trsm_benchmark::TrsmBenchmarkOptions options;
    int error_input = 0;

    if (argc >= 2) {
        options.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        options.m = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        error_input = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        options.test_recursive = std::atoi(argv[4]) != 0;
    }
    if (argc >= 6) {
        options.test_rocblas = std::atoi(argv[5]) != 0;
    }
    if (argc >= 7) {
        options.test_iterative = std::atoi(argv[6]) != 0;
    }

    int nb_start = 7;

    if (argc >= 8) {
        options.test_half = std::atoi(argv[7]) != 0;
        nb_start = 8;
    }

    if (argc > nb_start && is_bool_flag(argv[nb_start])) {
        options.test_double_blocked = std::atoi(argv[nb_start]) != 0;
        ++nb_start;
    }

    enum class BlockParameter {
        Nb,
        DoubleNb,
        MicroBlock
    };

    BlockParameter block_parameter = BlockParameter::Nb;
    for (int i = nb_start; i < argc; ++i) {
        if (std::strcmp(argv[i], "--double-nb") == 0) {
            block_parameter = BlockParameter::DoubleNb;
            continue;
        }
        if (std::strcmp(argv[i], "--b") == 0) {
            block_parameter = BlockParameter::MicroBlock;
            continue;
        }

        if (block_parameter == BlockParameter::DoubleNb) {
            options.double_nbs.push_back(std::atoi(argv[i]));
        } else if (block_parameter == BlockParameter::MicroBlock) {
            options.micro_blocks.push_back(std::atoi(argv[i]));
        } else {
            options.nbs.push_back(std::atoi(argv[i]));
        }
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(256);
    }

    options.error_mode = parse_benchmark_error_mode(error_input);
    trsm_benchmark::run_trsm_benchmarks(options);

    return 0;
}
