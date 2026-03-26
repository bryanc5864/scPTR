#!/bin/bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
exec python -u /home/bcheng/scPTR/analyses/run_deep_benchmark.py "$@"
