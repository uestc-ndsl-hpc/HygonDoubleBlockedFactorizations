#!/usr/bin/env bash
set -u
set -o pipefail

ROCM_PATH=${ROCM_PATH:-/opt/dtk-26.04}
export ROCM_PATH
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"

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
    if [ -n "${OBJDUMP:-}" ]; then
        echo "${OBJDUMP}"
        return
    fi

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

emit_with_objdump() {
    if [ -z "${OBJDUMP:-}" ]; then
        return 1
    fi

    if [ "${#ISA_INPUTS[@]}" -eq 0 ]; then
        return 1
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
}

detect_gpu_arch() {
    if [ -n "${GPU_ARCH:-}" ]; then
        echo "${GPU_ARCH%%:*}"
        return
    fi

    if command -v hipconfig >/dev/null 2>&1; then
        local arch
        arch=$(hipconfig --amdgpu-target 2>/dev/null | awk '/gfx/ {print $1; exit}' | cut -d: -f1)
        if [ -n "${arch}" ]; then
            echo "${arch}"
            return
        fi
    fi

    if command -v rocminfo >/dev/null 2>&1; then
        local arch
        arch=$(rocminfo 2>/dev/null | awk '/Name: *gfx/ {print $2; exit}' | cut -d: -f1)
        if [ -n "${arch}" ]; then
            echo "${arch}"
            return
        fi
    fi

    if command -v hipinfo >/dev/null 2>&1; then
        hipinfo 2>/dev/null | awk '/gcnArchName/ && /gfx/ {print $2; exit}' | cut -d: -f1
    fi
}

emit_with_save_temps() {
    local hipcc="${HIPCC:-${ROCM_PATH}/bin/hipcc}"
    if [ ! -x "${hipcc}" ] && command -v hipcc >/dev/null 2>&1; then
        hipcc=$(command -v hipcc)
    fi

    if [ ! -x "${hipcc}" ]; then
        return 1
    fi

    local arch
    arch=$(detect_gpu_arch)
    local target_id="${HYGON_AMDGPU_TARGET_ID:-${arch}}"
    local enable_mfma="${HYGON_ENABLE_MFMA_KERNEL:-1}"
    local temp_dir="${OUT_DIR}/hss_isa_temps"
    rm -rf "${temp_dir}"
    mkdir -p "${temp_dir}"

    local arch_args=()
    if [ -n "${target_id}" ]; then
        arch_args=(--offload-arch="${target_id}")
    fi

    echo "llvm-objdump not found; using hipcc -save-temps=obj fallback."
    echo "hipcc           : ${hipcc}"
    echo "GPU arch        : ${arch:-auto}"
    echo "Target id       : ${target_id:-auto}"
    echo "MFMA kernel     : ${enable_mfma}"
    echo "Temp dir        : ${temp_dir}"

    (
        cd "${temp_dir}" &&
        "${hipcc}" -O3 -std=c++17 -D__HIP_PLATFORM_AMD__ \
            -DHYGON_ENABLE_MFMA_KERNEL="${enable_mfma}" \
            -DHYGON_AMDGPU_TARGET_ID="\"${target_id}\"" \
            -I"${ROCM_PATH}/include" \
            -I"${ROOT_DIR}/include" \
            "${arch_args[@]}" \
            -save-temps=obj \
            -c "${ROOT_DIR}/src/kernels/hss_gemm.hip" \
            -o hss_gemm.o
    )

    : > "${ASM_LOG}"
    {
        echo "=================================================="
        echo "hipcc save-temps output: ${temp_dir}"
        echo "=================================================="
    } >> "${ASM_LOG}"

    find "${temp_dir}" -type f \
        \( -name '*.s' -o -name '*.ll' -o -name '*.isa' -o -name '*.mir' \) \
        -print -exec sed -n '1,200000p' {} \; >> "${ASM_LOG}"
}

if ! emit_with_objdump; then
    if ! emit_with_save_temps; then
        echo "Error: neither llvm-objdump nor hipcc save-temps path is available."
        echo "Set OBJDUMP=/path/to/objdump or HIPCC=/path/to/hipcc."
        exit 1
    fi
fi

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
