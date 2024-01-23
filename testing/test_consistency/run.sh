#!/bin/bash

set -e

WD=$(pwd)

cd ../test_tutorial && ./prepare.sh && cd "$WD"

python run_benchmarks.py
python check_results.py
