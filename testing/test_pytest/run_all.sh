#!/bin/bash

set -e

PYTEST_ARGS="-xv"
IGNORE_ARGS="--ignore=qsprpred/extra"
# ignore extra if QSPPRED_TEST_EXTRAS is true
if [ "$QSPPRED_TEST_EXTRAS" == "true" ]; then
  IGNORE_ARGS=""
fi
REPORT_FILE="pytest_report.xml"
REPO_ROOT=$(git rev-parse --show-toplevel)
pytest "${PYTEST_ARGS}" "$REPO_ROOT"/qsprpred --junitxml="$REPORT_FILE" "$IGNORE_ARGS"