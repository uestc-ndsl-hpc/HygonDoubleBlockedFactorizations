#include <cstdlib>
#include <cstring>

#include "syrk/benchmark/syrk_benchmark_common.h"

int main(int argc, char** argv)
{
    syrk_benchmark::SyrkBenchmarkOptions options;
    int error_input = 0;

    if (argc >= 2) {
        options.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        options.k = std::atoi(argv[2]);
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
        options.test_batched = std::atoi(argv[6]) != 0;
    }
    if (argc >= 8) {
        options.test_mixed = std::atoi(argv[7]) != 0;
    }

    bool parse_k_tiles = false;
    for (int i = 8; i < argc; ++i) {
        if (std::strcmp(argv[i], "--k-tile") == 0) {
            parse_k_tiles = true;
            continue;
        }

        if (parse_k_tiles) {
            options.k_tiles.push_back(std::atoi(argv[i]));
        } else {
            options.nbs.push_back(std::atoi(argv[i]));
        }
    }
    if (options.nbs.empty()) {
        options.nbs.push_back(256);
    }

    options.error_mode = parse_benchmark_error_mode(error_input);
    syrk_benchmark::run_syrk_benchmarks(options);

    return 0;
}
