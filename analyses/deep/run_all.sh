#!/bin/bash
# Run all DeepPTR benchmark scripts sequentially.
# Usage: bash analyses/deep/run_all.sh [script_numbers...]
# Examples:
#   bash analyses/deep/run_all.sh           # run all
#   bash analyses/deep/run_all.sh 12 13 14  # run only these
set -e
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""

cd "$(dirname "$0")"

if [ $# -gt 0 ]; then
    # Run specific scripts
    for num in "$@"; do
        script=$(ls ${num}_*.py 2>/dev/null)
        if [ -n "$script" ]; then
            echo ""
            echo "========================================"
            echo "Running: $script"
            echo "========================================"
            python -u "$script" || echo "FAILED: $script"
        else
            echo "No script matching ${num}_*.py"
        fi
    done
else
    # Run all
    for script in [0-9][0-9]_*.py; do
        echo ""
        echo "========================================"
        echo "Running: $script"
        echo "========================================"
        python -u "$script" || echo "FAILED: $script"
    done
fi

echo ""
echo "Complete. Results in output/deep_benchmarks/"
