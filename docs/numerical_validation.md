# Full CPU FP64 residual validation

The factorization remains FP32 on the accelerator. The optional validation path
copies the original FP32 input and computed factors to the CPU, converts values
to FP64 **before multiplication**, and evaluates every residual entry using CPU
DGEMM. Subtraction and Frobenius-norm accumulation also use FP64. It computes
`||PA-LU||F/||A||F` or `||A-LL^T||F/||A||F`, not the error of an FP64
factorization. The CSV metric is `cpu_fp64_full_relative_residual`.

The implementation packs bounded tiles and skips mathematically zero triangular
products. It does not reuse an FP32 GPU product. At order 40960, the two FP32 host
matrix copies alone occupy 12.5 GiB; plan additional memory for input generation,
BLAS and the benchmark. CPU validation is outside the GPU timing interval and
uses a separate factorization. Existing benchmark defaults are unchanged.

## Build and test

Use the project's DTK build configuration with a **host** CBLAS library:

```sh
cmake -S . -B build -DHYGON_CPU_FP64_USE_CBLAS=ON
cmake --build build --target benchmark_getrf benchmark_potrf -j8
cmake -S tests/cpu_fp64_residual -B build-cpu-tests -DHYGON_CPU_FP64_USE_CBLAS=ON
cmake --build build-cpu-tests
ctest --test-dir build-cpu-tests --output-on-failure
```

If necessary, provide `-DHYGON_CPU_CBLAS_INCLUDE_DIR=/path/to/headers` and
`-DHYGON_CPU_CBLAS_LIBRARY=/path/to/libopenblas.so` in both configurations.
Without CBLAS, a portable, single-threaded CPU implementation is available for
small tests but is too slow for the published large-order validation sweep.
See `tests/cpu_fp64_residual/README.md` for the optional SciPy LP64 adapter.

## Commands and parameter policy

Use an available device allocated to you. Do not start a run on an occupied
device. Set `HIP_VISIBLE_DEVICES` to its physical index, avoid conflicting
visibility remapping, and set `OPENBLAS_NUM_THREADS=8` and `OMP_NUM_THREADS=8`.
The collection used eight CPU BLAS threads per independent job and CPU NUMA
binding local to each selected accelerator. No factorization spanned devices.

Example for one original-seed input at order 40960:

```sh
export HIP_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export BENCHMARK_WARMUP_ITERS=0 BENCHMARK_PROFILE_ITERS=1
# Select new CSV and log paths; do not overwrite an earlier run.
export BENCHMARK_CSV_PATH=getrf_N40960_seed1234567890.csv
./build/benchmark_getrf 40960 0 1 --pivot --methods rocsolver,mgemm \
  --double-nb 20480 --b 4096 --outer-trsm iterative --trsm-block 2048 \
  --pivot-tile 2048 --inner-gemm-m-tile 8192 --seed 1234567890 \
  --cpu-fp64-residual --cpu-residual-tile 512 --full-residual-max-n 40960

export BENCHMARK_CSV_PATH=potrf_N40960_seed20260716.csv
./build/benchmark_potrf 40960 0 0 2 0 2048 \
  --methods rocsolver,kbsyrk,recursive --skip-rec-tiled-trsm \
  --double-nb 8192 --b 1024 --syrk-k-tile 8192 --seed 20260716 \
  --cpu-fp64-residual --cpu-residual-tile 512 --full-residual-max-n 40960
```

The explicit maximum-order option permits full validation above its default
safety limit. For all tested orders, LU uses `B=min(n/2,20480)` and
`b=min(n/4,4096)`, with TRSM and pivot tiles 2048 and GEMM M-tile 8192.
Cholesky uses `B=8192`, `b=1024`, and SYRK K-tile 8192. The recursive baseline
uses leaf size 2048 and excludes RecTiledTRSM.

## Published coverage and interpretation

The October 2026 supplement contains 45 method/order/seed results from 18
benchmark invocations and 18 distinct matrix inputs:

- Orders 8192, 16384, 24576, 32768 and 40960 with original seeds
  GETRF=1234567890 and POTRF=20260716: 25 method results
- At orders 8192 and 24576, seeds +1 and +2: 20 additional method results
- LU: rocSOLVER and final DB. Cholesky: rocSOLVER, final DB and recursive

All methods within a routine/order/seed use the same input. GETRF uses uniform
entries on [-0.5,0.5]; POTRF uses the FP32 matrix generated as `RR^T+I`, where R
has standard normal entries. The residual is against the actual stored input,
not a regenerated ideal FP64 matrix. This extends seed coverage within the
same families; it is not a condition-number study or a general stability proof.

The collection used zero warmups and one timed sample per method, followed by
validation. **These timings do not replace the paper's ten-run performance
measurements.** All 45 validation factorizations returned `info=0` with finite
residuals. Exact commands, seeds, raw logs, raw CSVs, backend metadata, tested
source hashes and export provenance are in
`results/fp32/paper_final/supplements_20261021/fp64_residual/`.

Recheck the public evidence with:

```sh
python3 scripts/verify_paper_supplements.py
```

The verifier also recomputes the stage-spread statistics and checks the
rocSOLVER boundary probes. Source hashes normalize CRLF to LF. For later source
revisions, `--skip-source-check` checks archived data without claiming the new
source is identical to the collection snapshot.
