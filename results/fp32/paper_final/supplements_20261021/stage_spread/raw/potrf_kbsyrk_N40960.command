env HIP_VISIBLE_DEVICES=1 <BUILD_DIR>/profile_double_blocked_potrf_stage_samples --n 40960 --nb 8192 --b 1024 --syrk-k-tile 8192 --inner-gemm-m-tile 0 --warmup 1 --iters 10 
