#pragma once

namespace hygon::gemm {

constexpr int kK100InnerGemmMTile = 8192;

constexpr int select_k100_inner_gemm_m_tile(int rows, int cols, int k)
{
    const bool baseline_window = rows > 32768 && k >= 5632;

    // These narrow windows avoid measured rocBLAS FP32 NT kernel cliffs on
    // K100_AI. Adjacent shapes are deliberately left untiled.
    const bool low_k_32k_window =
        cols == 512 && rows + k == 32768 &&
        k >= 512 && k <= 3072 && k % 512 == 0;
    const bool high_k_16k_window =
        rows == 16896 && cols == 512 && k == 7680;

    return baseline_window || low_k_32k_window || high_k_16k_window
        ? kK100InnerGemmMTile
        : 0;
}

} // namespace hygon::gemm
