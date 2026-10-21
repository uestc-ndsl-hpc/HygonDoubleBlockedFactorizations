env HIP_VISIBLE_DEVICES=1 BENCHMARK_WARMUP_ITERS=0 BENCHMARK_PROFILE_ITERS=1 <REPOSITORY>/build/benchmark_getrf 46341 0 1 2048 --pivot --methods rocsolver --pivot-tile 2048 
