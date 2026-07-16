# Final Experiment Protocol

This protocol is the executable companion to `PAPER.md`. All reported results
must come from one clean Git commit and one K100_AI device. Runs are serial.

## 1. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8 --target \
  benchmark_potrf benchmark_getrf \
  profile_double_blocked_potrf profile_double_blocked_getrf
```

## 2. Candidate Parameter Policy

The unified script currently records the following candidate policy:

- DB-POTRF: `B=8192`, `b=1024`, `T_K=8192`, `T_M=8192`, with M tiling
  enabled for `M>32768` and `K>=5632`.
- DB-GETRF: `B=N/2`, `b=min(N/4,4096)`, iterative TRSM block `2048`, pivot
  column tile `2048`, and inner-GEMM row tile `8192`.

These values may be changed only before the source commit is frozen. A single
policy must then be used for every matrix size; per-size best-of-search results
are tuning data, not end-to-end scaling results.

## 3. Correctness Gate

First inspect the selected GPU and confirm that it is idle. Then run only the
full-residual checks:

```bash
PAPER_DEVICE=1 \
PAPER_CONFIRM_IDLE=1 \
PAPER_SECTIONS=correctness \
PAPER_RUN_ID=correctness_gate \
bash run_paper_experiments.sh
```

This tests `N=4096` and the non-power-of-two size `N=5000`. It reconstructs
the complete `A-LL^T` and `PA-LU` products with FP64 rocBLAS GEMM outside the
timed region. The correctness run uses a `1024` update tile and zero crossover
thresholds so that the K-blocked and M-tiled branches execute at these smaller
sizes. Do not start scaling if any method reports nonzero `info`, a non-finite
residual, or an FP32 relative Frobenius residual above `2e-6`.

## 4. Scaling and Profiles

After the correctness gate and parameter freeze:

```bash
PAPER_DEVICE=1 \
PAPER_CONFIRM_IDLE=1 \
PAPER_SECTIONS=scaling,profiles \
PAPER_RUN_ID=paper_final \
PAPER_WARMUP_ITERS=2 \
PAPER_PROFILE_ITERS=10 \
bash run_paper_experiments.sh
```

Scaling uses `N=8192,16384,24576,32768,40960`. The script compares rocSOLVER
with the basic and optimized double-blocked variants required by the ablation;
it does not select a different configuration at each size. The profile section
runs the final POTRF, the immediate- and batched-pivot GETRF schedules, and a
direct-versus-iterative outer-TRSM control at `N=40960`.

## 5. Artifacts

Each end-to-end run writes:

- a human-readable log containing runtime, throughput, speedup, residual, and
  timing variability;
- a CSV containing every measured iteration and its summary statistics;
- a manifest containing the Git commit, dirty state, toolchain, device, NUMA
  binding, and fixed parameters.

Do not combine measurements from different commits, DTK versions, or devices
in one figure. Keep tuning, correctness, scaling, and profile results in
separate directories.
