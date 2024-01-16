#!/bin/bash

set -e

WD=$(pwd)
RUN_EXTRAS=${QSPPRED_TEST_EXTRAS:-true}
echo "Running tests with RUN_EXTRAS=$RUN_EXTRAS"

# -x: stop on first failure, -v: verbose
pytest -xv ../qsprpred/benchmarks
pytest -xv ../qsprpred/data
pytest -xv ../qsprpred/plotting
# only run extras if explicitly requested
if [ "$RUN_EXTRAS" == "true" ]; then
    pytest -xv ../qsprpred/extra/data/descriptors
    pytest -xv ../qsprpred/extra/data/sampling
    pytest -xv ../qsprpred/extra/data/tables
    pytest -xv ../qsprpred/extra/gpu
    pytest -xv ../qsprpred/extra/models
fi
cd test_cli && ./run.sh && cd "$WD"
cd test_consistency && ./run.sh && cd "$WD"
cd test_tutorial && ./run.sh && cd "$WD"