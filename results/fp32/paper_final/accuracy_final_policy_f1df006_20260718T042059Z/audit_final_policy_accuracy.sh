#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 SWEEP_DIRECTORY" >&2
    exit 2
fi

sweep_dir=$1
accuracy_dir=${sweep_dir}/accuracy
expected_sizes=(1024 2048 4096 8192 16384 24576)

validate_csv() {
    local csv_path=$1
    local limit=$2
    shift 2

    awk -F, -v limit="${limit}" \
        -v m1="$1" -v p1="$2" \
        -v m2="$3" -v p2="$4" \
        -v m3="$5" -v p3="$6" \
        -v m4="$7" -v p4="$8" '
        function unquote(value) {
            sub(/^"/, "", value)
            sub(/"$/, "", value)
            return value
        }
        BEGIN {
            expected_method[1] = m1
            expected_parameter[1] = p1
            expected_method[2] = m2
            expected_parameter[2] = p2
            expected_method[3] = m3
            expected_parameter[3] = p3
            expected_method[4] = m4
            expected_parameter[4] = p4
        }
        NR == 1 {
            if (NF != 15 ||
                $14 != "residual_metric" ||
                $15 != "residual_value") {
                exit 10
            }
            next
        }
        {
            row = NR - 1
            method = unquote($2)
            parameter = unquote($3)
            if (NF != 15 ||
                method != expected_method[row] ||
                parameter != expected_parameter[row] ||
                $14 != "normwise_backward_error" ||
                $15 !~ /^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/) {
                exit 11
            }
            value = $15 + 0
            if (value < 0 || value > limit) {
                exit 12
            }
            if (row == 1 || value < min_value) {
                min_value = value
            }
            if (row == 1 || value > max_value) {
                max_value = value
            }
            rows++
        }
        END {
            if (rows != 4) {
                exit 13
            }
            printf "%s rows=%d min=%.12g max=%.12g limit=%.12g policy=matched\n",
                FILENAME, rows, min_value, max_value, limit
        }
    ' "${csv_path}"
}

[[ -s "${sweep_dir}/manifest.log" ]]
grep -q "Paper experiment run completed" "${sweep_dir}/manifest.log"
grep -q "Accuracy policy     : identical to the scaling dispatch policy" \
    "${sweep_dir}/manifest.log"

checked=0
for n in "${expected_sizes[@]}"; do
    potrf_nb=8192
    if (( potrf_nb > n )); then
        potrf_nb=${n}
    fi

    getrf_nb=$((n / 2))
    getrf_b=$((n / 4))
    if (( getrf_b > 4096 )); then
        getrf_b=4096
    fi

    pivot_tile=2048
    if (( pivot_tile > n )); then
        pivot_tile=${n}
    fi

    potrf_log=${accuracy_dir}/potrf_N${n}.log
    potrf_csv=${accuracy_dir}/potrf_N${n}_timings.csv
    getrf_log=${accuracy_dir}/getrf_N${n}.log
    getrf_csv=${accuracy_dir}/getrf_N${n}_timings.csv

    for log_path in "${potrf_log}" "${getrf_log}"; do
        [[ -s "${log_path}" ]]
        grep -q "Normwise backward error" "${log_path}"
        grep -q "full matrix, factorization-precision product, FP64 norm accumulation" \
            "${log_path}"
    done

    validate_csv "${potrf_csv}" 1e-5 \
        "rocSOLVER" "N/A" \
        "DoubleBlock" "${potrf_nb}/1024" \
        "DoubleBlock+KBSYRK" "${potrf_nb}/1024/8192" \
        "DoubleBlock+KBSYRK+MGEMM" \
        "${potrf_nb}/1024/8192/8192/32768/5632"

    validate_csv "${getrf_csv}" 5e-4 \
        "rocSOLVER GETRF" "N/A" \
        "DoubleBlock-LU" "${getrf_nb}/${getrf_b}/iterative/2048" \
        "DoubleBlock-LU+BatchedPivot" \
        "${getrf_nb}/${getrf_b}/iterative/2048/${pivot_tile}" \
        "DoubleBlock-LU+BatchedPivot+MGEMM" \
        "${getrf_nb}/${getrf_b}/iterative/2048/${pivot_tile}/8192"

    checked=$((checked + 2))
done

artifact_count=$(
    find "${accuracy_dir}" -maxdepth 1 -type f \
        \( -name 'potrf_N*.log' -o -name 'getrf_N*.log' \
           -o -name 'potrf_N*_timings.csv' -o -name 'getrf_N*_timings.csv' \) \
        -size +0c | wc -l
)

[[ "${checked}" -eq 12 ]]
[[ "${artifact_count}" -eq 24 ]]
echo "AUDIT PASSED: ${checked} CSVs match the final policy; ${artifact_count} artifacts are nonempty"
