#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "getrf/benchmark/getrf_benchmark_common.h"

namespace {

enum class BlockParameter {
    Nb,
    DoubleNb,
    MicroBlock,
    TrsmBlock,
    PivotTile,
    InnerGemmMTile,
    OuterGemmMTile
};

bool parse_methods(const char* value,
                   getrf_benchmark::GetrfBenchmarkOptions& options)
{
    options.test_rocsolver = false;
    options.test_blocked = false;
    options.test_double_basic = false;
    options.test_batched_pivot = false;
    options.test_mgemm = false;

    std::stringstream input(value);
    std::string method;
    bool found = false;
    while (std::getline(input, method, ',')) {
        if (method == "all") {
            options.test_rocsolver = true;
            options.test_blocked = true;
            options.test_double_basic = true;
            options.test_batched_pivot = true;
            options.test_mgemm = true;
        } else if (method == "rocsolver") {
            options.test_rocsolver = true;
        } else if (method == "blocked") {
            options.test_blocked = true;
        } else if (method == "double" || method == "basic") {
            options.test_double_basic = true;
        } else if (method == "batched") {
            options.test_batched_pivot = true;
        } else if (method == "mgemm") {
            options.test_mgemm = true;
        } else {
            return false;
        }
        found = true;
    }

    return found;
}

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
    bool methods_explicit = false;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pivot") == 0
            || std::strcmp(argv[i], "-p") == 0) {
            options.pivoting = true;
            continue;
        }
        if (std::strcmp(argv[i], "--npvt") == 0) {
            options.pivoting = false;
            continue;
        }
        if (std::strcmp(argv[i], "--full-backward-error") == 0
            || std::strcmp(argv[i], "--full-residual") == 0) {
            options.full_backward_error = true;
            continue;
        }
        if (std::strcmp(argv[i], "--full-backward-error-max-n") == 0
            || std::strcmp(argv[i], "--full-residual-max-n") == 0) {
            if (i + 1 >= argc) {
                std::cerr
                    << "Missing value after --full-backward-error-max-n.\n";
                return EXIT_FAILURE;
            }
            options.full_backward_error_max_n = std::atoi(argv[++i]);
            parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--methods") == 0) {
            if (i + 1 >= argc || !parse_methods(argv[++i], options)) {
                std::cerr << "Invalid --methods value. Use a comma-separated "
                             "subset of rocsolver,blocked,double,batched,mgemm.\n";
                return EXIT_FAILURE;
            }
            methods_explicit = true;
            parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--double-nb") == 0) {
            parameter = BlockParameter::DoubleNb;
            continue;
        }
        if (std::strcmp(argv[i], "--b") == 0) {
            parameter = BlockParameter::MicroBlock;
            continue;
        }
        if (std::strcmp(argv[i], "--outer-trsm") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --outer-trsm.\n";
                return EXIT_FAILURE;
            }
            const char* mode = argv[++i];
            if (std::strcmp(mode, "direct") == 0) {
                options.outer_trsm_mode = getrf_outer_trsm_direct;
            } else if (std::strcmp(mode, "iterative") == 0) {
                options.outer_trsm_mode = getrf_outer_trsm_iterative;
            } else {
                std::cerr << "Invalid --outer-trsm value. Use direct or "
                             "iterative.\n";
                return EXIT_FAILURE;
            }
            parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--trsm-block") == 0
            || std::strcmp(argv[i], "--trsm-leaf") == 0) {
            parameter = BlockParameter::TrsmBlock;
            continue;
        }
        if (std::strcmp(argv[i], "--pivot-tile") == 0) {
            parameter = BlockParameter::PivotTile;
            continue;
        }
        if (std::strcmp(argv[i], "--inner-gemm-m-tile") == 0) {
            parameter = BlockParameter::InnerGemmMTile;
            continue;
        }
        if (std::strcmp(argv[i], "--outer-gemm-m-tile") == 0) {
            parameter = BlockParameter::OuterGemmMTile;
            continue;
        }

        const int value = std::atoi(argv[i]);
        if (parameter == BlockParameter::DoubleNb) {
            options.double_nbs.push_back(value);
        } else if (parameter == BlockParameter::MicroBlock) {
            options.micro_blocks.push_back(value);
        } else if (parameter == BlockParameter::TrsmBlock) {
            options.trsm_block = value;
            parameter = BlockParameter::Nb;
        } else if (parameter == BlockParameter::PivotTile) {
            options.pivot_column_tile = value;
            parameter = BlockParameter::Nb;
        } else if (parameter == BlockParameter::InnerGemmMTile) {
            options.inner_gemm_m_tile = value;
            parameter = BlockParameter::Nb;
        } else if (parameter == BlockParameter::OuterGemmMTile) {
            options.outer_gemm_m_tile = value;
            parameter = BlockParameter::Nb;
        } else {
            options.nbs.push_back(value);
        }
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(2048);
    }
    if (options.trsm_block <= 0) {
        std::cerr << "--trsm-block must be positive.\n";
        return EXIT_FAILURE;
    }
    if (options.inner_gemm_m_tile < 0 || options.outer_gemm_m_tile < 0) {
        std::cerr << "GEMM M-tile sizes must be nonnegative.\n";
        return EXIT_FAILURE;
    }

    if (!options.pivoting) {
        if (methods_explicit
            && (options.test_batched_pivot || options.test_mgemm)) {
            std::cerr << "batched and mgemm require --pivot.\n";
            return EXIT_FAILURE;
        }
        options.test_batched_pivot = false;
        options.test_mgemm = false;
    }
    if (options.full_backward_error) {
        if (options.full_backward_error_max_n <= 0
            || options.n > options.full_backward_error_max_n) {
            std::cerr << "Full normwise backward error requested for N="
                      << options.n
                      << ", but --full-backward-error-max-n="
                      << options.full_backward_error_max_n << ".\n";
            return EXIT_FAILURE;
        }
        options.error_mode = ErrorMode::Backward;
    }

    getrf_benchmark::run_getrf_benchmarks(options);

    return 0;
}
