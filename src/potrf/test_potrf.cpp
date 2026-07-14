#include <cstdlib>
#include <cstring>
#include <vector>

#include "potrf/benchmark/potrf_benchmark_common.h"

namespace {

bool is_bool_flag(const char* arg)
{
    return std::strcmp(arg, "0") == 0 || std::strcmp(arg, "1") == 0;
}

} // namespace

int main(int argc, char** argv)
{
    potrf_benchmark::PotrfBenchmarkOptions options;
    int error_input = 0;

    if (argc >= 2) {
        options.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        options.test_rocsolver = std::atoi(argv[2]) != 0;
    }
    if (argc >= 4) {
        options.test_recursive = std::atoi(argv[3]) != 0;
    }
    if (argc >= 5) {
        error_input = std::atoi(argv[4]);
    }
    if (argc >= 6) {
        options.precision_type = std::atoi(argv[5]);
    }

    int nb_start = 6;
    bool has_blocked_flag = false;
    bool has_double_blocked_flag = false;

    if (argc > nb_start && is_bool_flag(argv[nb_start])) {
        options.test_blocked = std::atoi(argv[nb_start]) != 0;
        has_blocked_flag = true;
        ++nb_start;
    }

    if (argc > nb_start && is_bool_flag(argv[nb_start])) {
        options.test_double_blocked = std::atoi(argv[nb_start]) != 0;
        has_double_blocked_flag = true;
        ++nb_start;
    }

    if (has_blocked_flag && !has_double_blocked_flag) {
        options.test_double_blocked = options.test_blocked;
    }

    enum class BlockParameter {
        Nb,
        DoubleNb,
        MicroBlock
    };

    BlockParameter block_parameter = BlockParameter::Nb;
    for (int i = nb_start; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fast-recursive") == 0) {
            options.test_fast_recursive = true;
            continue;
        }
        if (std::strcmp(argv[i], "--fast-leaf") == 0 && i + 1 < argc) {
            options.fast_recursive_leaf = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--fast-base-nb") == 0 && i + 1 < argc) {
            options.fast_recursive_base_nb = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--strassen-min") == 0 && i + 1 < argc) {
            options.strassen_min_size = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--fast-syrk-k-tile") == 0 && i + 1 < argc) {
            options.fast_recursive_syrk_k_tile = std::atoi(argv[++i]);
            continue;
        }
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
    potrf_benchmark::run_potrf_benchmarks(options);

    return 0;
}
