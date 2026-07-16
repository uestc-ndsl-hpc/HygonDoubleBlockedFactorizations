#include <cstdlib>
#include <cstring>

#include "getrf/benchmark/getrf_benchmark_common.h"

namespace {

enum class BlockParameter {
    Nb,
    DoubleNb,
    MicroBlock,
    TrsmLeaf,
    PivotTile,
    InnerGemmMTile
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
        if (std::strcmp(argv[i], "--pivot") == 0
            || std::strcmp(argv[i], "-p") == 0) {
            options.pivoting = true;
            continue;
        }
        if (std::strcmp(argv[i], "--npvt") == 0) {
            options.pivoting = false;
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
        if (std::strcmp(argv[i], "--trsm-leaf") == 0) {
            parameter = BlockParameter::TrsmLeaf;
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

        const int value = std::atoi(argv[i]);
        if (parameter == BlockParameter::DoubleNb) {
            options.double_nbs.push_back(value);
        } else if (parameter == BlockParameter::MicroBlock) {
            options.micro_blocks.push_back(value);
        } else if (parameter == BlockParameter::TrsmLeaf) {
            options.trsm_leaf = value;
            parameter = BlockParameter::Nb;
        } else if (parameter == BlockParameter::PivotTile) {
            options.pivot_column_tile = value;
            parameter = BlockParameter::Nb;
        } else if (parameter == BlockParameter::InnerGemmMTile) {
            options.inner_gemm_m_tile = value;
            parameter = BlockParameter::Nb;
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
