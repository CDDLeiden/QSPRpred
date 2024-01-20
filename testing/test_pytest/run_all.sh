#!/bin/bash

set -e

PYTEST_ARGS="-xv"
REPO_ROOT=$(git rev-parse --show-toplevel)
IGNORE_ARGS="--ignore=$REPO_ROOT/qsprpred/extra"
REPORT_FILE="test_report.xml"

# add extra if QSPPRED_TEST_EXTRAS is true
if [ "$QSPPRED_TEST_EXTRAS" == "true" ]; then
  IGNORE_ARGS=""
fi
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred --junitxml="$REPORT_FILE" "$IGNORE_ARGS"
