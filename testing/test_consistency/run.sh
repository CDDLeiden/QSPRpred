#!/bin/bash

set -e

python run_benchmarks.py
python check_results.py