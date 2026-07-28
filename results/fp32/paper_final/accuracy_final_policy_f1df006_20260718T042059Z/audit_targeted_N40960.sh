#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 TARGETED_DIRECTORY" >&2
    exit 2
fi

target_dir=$1
accuracy_dir=${target_dir}/accuracy

validate_csv() {
    local csv_path=$1
    local limit=$2
    local expected_method=$3
    local expected_parameter=$4

    awk -F, -v limit="${limit}" \
        -v expected_method="${expected_method}" \
        -v expected_parameter="${expected_parameter}" '
        function unquote(value) {
            sub(/^"/, "", value)
            sub(/"$/, "", value)
            return value
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
            method = unquote($2)
            parameter = unquote($3)
            if (NF != 15 ||
                $14 != "normwise_backward_error" ||
                $15 !~ /^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/) {
                exit 11
            }
            if (NR == 2 && (parameter != "N/A" ||
                            method !~ /^rocSOLVER/)) {
                exit 12
            }
            if (NR == 3 && (method != expected_method ||
                            parameter != expected_parameter)) {
                exit 13
            }
            value = $15 + 0
            if (value < 0 || value > limit) {
                exit 14
            }
            if (NR == 2 || value < min_value) {
                min_value = value
            }
            if (NR == 2 || value > max_value) {
                max_value = value
            }
            rows++
        }
        END {
            if (rows != 2) {
                exit 15
            }
            printf "%s rows=%d min=%.12g max=%.12g limit=%.12g policy=matched\n",
                FILENAME, rows, min_value, max_value, limit
        }
    ' "${csv_path}"
}

[[ -s "${target_dir}/manifest.log" ]]
grep -q "Paper experiment run completed" "${target_dir}/manifest.log"
grep -q "Accuracy policy     : identical to the scaling dispatch policy" \
    "${target_dir}/manifest.log"

for log_path in \
    "${accuracy_dir}/potrf_N40960.log" \
    "${accuracy_dir}/getrf_N40960.log"; do
    [[ -s "${log_path}" ]]
    grep -q "Normwise backward error" "${log_path}"
    grep -q "full matrix, factorization-precision product, FP64 norm accumulation" \
        "${log_path}"
done

validate_csv "${accuracy_dir}/potrf_N40960_timings.csv" 1e-5 \
    "DoubleBlock+KBSYRK+MGEMM" \
    "8192/1024/8192/8192/32768/5632"

validate_csv "${accuracy_dir}/getrf_N40960_timings.csv" 5e-4 \
    "DoubleBlock-LU+BatchedPivot+MGEMM" \
    "20480/4096/iterative/2048/2048/8192"

artifact_count=$(
    find "${accuracy_dir}" -maxdepth 1 -type f -size +0c | wc -l
)
[[ "${artifact_count}" -eq 4 ]]

echo "AUDIT PASSED: N=40960 final methods match the performance policy; 4 artifacts are nonempty"
