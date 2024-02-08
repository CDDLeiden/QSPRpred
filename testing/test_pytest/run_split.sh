#!/bin/bash

set -e

PYTEST_ARGS="-xv"
REPO_ROOT=$(git rev-parse --show-toplevel)
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/benchmarks
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/data
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/models
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/plotting
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/utils
# only run extras if explicitly requested
if [ "$QSPPRED_TEST_EXTRAS" == "true" ]; then
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/data/descriptors
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/data/sampling
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/data/tables
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/gpu
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/models
fi