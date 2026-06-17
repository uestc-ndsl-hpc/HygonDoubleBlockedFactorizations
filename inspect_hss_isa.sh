#!/usr/bin/env bash
set -u
set -o pipefail

ROCM_PATH=${ROCM_PATH:-/opt/dtk-26.04}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build"}
OUT_DIR=${OUT_DIR:-"${ROOT_DIR}/results/fp32/gemm"}
ASM_LOG=${ASM_LOG:-"${OUT_DIR}/hss_isa.txt"}

find_executable() {
    if [ -n "${EXEC:-}" ]; then
        echo "${EXEC}"
        return
    fi

    local candidates=(
        "${BUILD_DIR}/src/gemm/benchmark_gemm"
        "${BUILD_DIR}/benchmark_gemm"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -x "${candidate}" ]; then
            echo "${candidate}"
            return
        fi
    done

    echo "${candidates[0]}"
}

find_objdump() {
    local candidates=(
        "${ROCM_PATH}/llvm/bin/llvm-objdump"
        "${ROCM_PATH}/bin/llvm-objdump"
        "/opt/rocm/llvm/bin/llvm-objdump"
        "/opt/rocm/bin/llvm-objdump"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -x "${candidate}" ]; then
            echo "${candidate}"
            return
        fi
    done

    if command -v llvm-objdump >/dev/null 2>&1; then
        command -v llvm-objdump
    fi
}

OBJDUMP=$(find_objdump)

mkdir -p "${OUT_DIR}"

find_isa_inputs() {
    if [ -n "${ISA_INPUT:-}" ]; then
        echo "${ISA_INPUT}"
        return
    fi

    local direct_candidates=(
        "${BUILD_DIR}/src/kernels/libhygon_kernels.a"
        "${BUILD_DIR}/libhygon_kernels.a"
        "$(find_executable)"
    )

    local candidate
    for candidate in "${direct_candidates[@]}"; do
        if [ -f "${candidate}" ] || [ -x "${candidate}" ]; then
            echo "${candidate}"
        fi
    done

    if [ -d "${BUILD_DIR}/src/kernels/CMakeFiles" ]; then
        find "${BUILD_DIR}/src/kernels/CMakeFiles" -type f \
            \( -name 'hss_gemm*.o' -o -name 'hss_gemm*.obj' \) 2>/dev/null | sort
    fi
}

mapfile -t ISA_INPUTS < <(find_isa_inputs)
EXEC=$(find_executable)

echo "=================================================="
echo "HSS ISA inspection"
echo "DTK environment : ${ROCM_PATH}"
echo "Executable      : ${EXEC}"
echo "llvm-objdump    : ${OBJDUMP:-not found}"
echo "Output          : ${ASM_LOG}"
echo "ISA input       : ${ISA_INPUT:-auto}"
echo "=================================================="

if [ ! -x "${EXEC}" ]; then
    echo "Error: executable not found or not executable: ${EXEC}"
    exit 1
fi

if [ -z "${OBJDUMP:-}" ]; then
    echo "Error: llvm-objdump not found. Set ROCM_PATH or install LLVM objdump."
    exit 1
fi

if [ "${#ISA_INPUTS[@]}" -eq 0 ]; then
    echo "Error: no ISA input found. Set ISA_INPUT to a .o, .a, .hsaco, or executable."
    exit 1
fi

: > "${ASM_LOG}"
for input in "${ISA_INPUTS[@]}"; do
    if [ ! -f "${input}" ] && [ ! -x "${input}" ]; then
        continue
    fi

    {
        echo "=================================================="
        echo "Input: ${input}"
        echo "=================================================="
    } >> "${ASM_LOG}"

    if "${OBJDUMP}" --offloading -d --demangle "${input}" >> "${ASM_LOG}" 2>/dev/null; then
        continue
    fi

    "${OBJDUMP}" -d --demangle "${input}" >> "${ASM_LOG}" 2>/dev/null || true
done

count_pattern() {
    local pattern=$1
    grep -E -c "${pattern}" "${ASM_LOG}" || true
}

mixed_count=$(count_pattern 'v_(mad|fma)_mix_f32')
fma_count=$(count_pattern 'v_(fma|mac|mad)_f32')
cvt_count=$(count_pattern 'v_cvt_f32_f16')
mfma_count=$(count_pattern 'v_mfma|v_wmma')

echo ""
echo "Instruction counts"
echo "  mixed f16/f32 fma : ${mixed_count}"
echo "  fp32 fma/mad/mac  : ${fma_count}"
echo "  f16->f32 convert  : ${cvt_count}"
echo "  matrix instr      : ${mfma_count}"

echo ""
echo "Mixed instruction context"
mixed_context=$(grep -n -E 'v_(mad|fma)_mix_f32' "${ASM_LOG}" | head -40 || true)
if [ -n "${mixed_context}" ]; then
    echo "${mixed_context}"
else
    echo "  none"
fi

echo ""
echo "Convert/FMA context"
cvt_context=$(grep -n -E 'v_cvt_f32_f16|v_(fma|mac|mad)_f32' "${ASM_LOG}" | head -80 || true)
if [ -n "${cvt_context}" ]; then
    echo "${cvt_context}"
else
    echo "  none"
fi

echo ""
echo "=================================================="
if [ "${mixed_count}" -eq 0 ]; then
    echo "Result: no mixed FMA instruction found; inspect hss_mixed_fma lowering next."
else
    echo "Result: mixed FMA instructions are present."
fi
echo "=================================================="
