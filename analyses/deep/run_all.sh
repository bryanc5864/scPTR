#!/bin/bash
# Run all DeepPTR benchmark scripts sequentially.
# Usage: bash analyses/deep/run_all.sh
set -e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""

cd "$(dirname "$0")"

for script in 01_*.py 02_*.py 03_*.py 04_*.py 05_*.py 06_*.py 07_*.py 08_*.py 09_*.py 10_*.py 11_*.py; do
    echo ""
    echo "========================================"
    echo "Running: $script"
    echo "========================================"
    python -u "$script" || echo "FAILED: $script"
done

echo ""
echo "All scripts complete. Results in output/deep_benchmarks/"
