#!/bin/bash

set -e

WD=$(pwd)

# -x: stop on first failure, -v: verbose
pytest -xv ../qsprpred/benchmarks
pytest -xv ../qsprpred/data
pytest -xv ../qsprpred/extra/data
pytest -xv ../qsprpred/extra/gpu
pytest -xv ../qsprpred/extra/models
pytest -xv ../qsprpred/plotting
cd test_cli && ./run.sh && cd "$WD"
cd test_tutorial && ./run.sh && cd "$WD"
cd test_consistency && ./run.sh && cd "$WD"