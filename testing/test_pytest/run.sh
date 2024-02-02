#!/bin/bash

set -e

export QSPPRED_TEST_SPLIT_UNITS=${QSPPRED_TEST_SPLIT_UNITS:-false}
export QSPPRED_TEST_EXTRAS=${QSPPRED_TEST_EXTRAS:-true}
echo "Setting QSPPRED_TEST_SPLIT_UNITS=$QSPPRED_TEST_SPLIT_UNITS"
echo "Setting QSPPRED_TEST_EXTRAS=$QSPPRED_TEST_EXTRAS"

# run all if QSPPRED_TEST_SPLIT_UNITS is false, otherwise run split
if [ "$QSPPRED_TEST_SPLIT_UNITS" != "true" ]; then
  echo "Running all tests at once."
  ./run_all.sh
else
  echo "Running tests in split units."
  ./run_split.sh
fi