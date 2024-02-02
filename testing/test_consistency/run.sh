#!/bin/bash

set -e

WD=$(pwd)

cd ../test_tutorial && ./prepare.sh && cd "$WD"
rm -rf ./data

python run_benchmarks.py
python check_results.py
