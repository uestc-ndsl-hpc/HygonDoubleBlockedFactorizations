#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 SWEEP_DIRECTORY" >&2
    exit 2
fi

sweep_dir=$1
accuracy_dir=${sweep_dir}/accuracy
expected_sizes=(1024 2048 4096 8192 16384 24576)

[[ -f "${sweep_dir}/manifest.log" ]]
grep -q "Paper experiment run completed" "${sweep_dir}/manifest.log"

checked=0
for n in "${expected_sizes[@]}"; do
    for factorization in potrf getrf; do
        log_path=${accuracy_dir}/${factorization}_N${n}.log
        csv_path=${accuracy_dir}/${factorization}_N${n}_timings.csv

        [[ -s "${log_path}" ]]
        [[ -s "${csv_path}" ]]
        grep -q "Normwise backward error" "${log_path}"
        grep -q "full matrix, factorization-precision product, FP64 norm accumulation" "${log_path}"

        if [[ "${factorization}" == "potrf" ]]; then
            limit=1e-5
        else
            limit=5e-4
        fi

        awk -F, -v limit="${limit}" '
            NR == 1 {
                if (NF != 15 ||
                    $14 != "residual_metric" ||
                    $15 != "residual_value") {
                    exit 10
                }
                next
            }
            {
                if (NF != 15 ||
                    $14 != "normwise_backward_error" ||
                    $15 !~ /^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$/) {
                    exit 11
                }
                value = $15 + 0
                if (value < 0 || value > limit) {
                    exit 12
                }
                if (rows == 0 || value < min_value) {
                    min_value = value
                }
                if (rows == 0 || value > max_value) {
                    max_value = value
                }
                rows++
            }
            END {
                if (rows != 4) {
                    exit 13
                }
                printf "%s rows=%d min=%.12g max=%.12g limit=%.12g\n",
                    FILENAME, rows, min_value, max_value, limit
            }
        ' "${csv_path}"
        checked=$((checked + 1))
    done
done

artifact_count=$(
    find "${accuracy_dir}" -maxdepth 1 -type f \
        \( -name 'potrf_N*.log' -o -name 'getrf_N*.log' \
           -o -name 'potrf_N*_timings.csv' -o -name 'getrf_N*_timings.csv' \) \
        -size +0c | wc -l
)

[[ "${checked}" -eq 12 ]]
[[ "${artifact_count}" -eq 24 ]]
echo "AUDIT PASSED: ${checked} benchmark CSVs and ${artifact_count} nonempty log/CSV artifacts"
