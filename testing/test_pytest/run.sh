#!/bin/bash

set -e

PYTEST_ARGS="-xv"
REPORT_FILE="pytest_report.xml"
REPO_ROOT=$(git rev-parse --show-toplevel)
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/benchmarks --junitxml="$REPORT_FILE"
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/data --junitxml="$REPORT_FILE"
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/plotting --junitxml="$REPORT_FILE"
# only run extras if explicitly requested
if [ "$QSPPRED_TEST_EXTRAS" == "true" ]; then
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/data/descriptors --junitxml="$REPORT_FILE"
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/data/sampling --junitxml="$REPORT_FILE"
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/data/tables --junitxml="$REPORT_FILE"
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/gpu --junitxml="$REPORT_FILE"
  pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred/extra/models --junitxml="$REPORT_FILE"
fi