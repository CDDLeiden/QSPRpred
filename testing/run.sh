#!/bin/bash

set -e

WD=$(pwd)
export QSPPRED_TEST_TUTORIAL=${QSPPRED_TEST_TUTORIAL:-true}
echo "Setting QSPPRED_TEST_TUTORIAL=$QSPPRED_TEST_TUTORIAL"
export QSPPRED_TEST_CLI=${QSPPRED_TEST_CLI:-false}
echo "Setting QSPPRED_TEST_CLI=$QSPPRED_TEST_CLI"
export QSPPRED_TEST_PYTEST=${QSPPRED_TEST_PYTEST:-true}
echo "Setting QSPPRED_TEST_PYTEST=$QSPPRED_TEST_PYTEST"

if [ "$QSPPRED_TEST_PYTEST" == "true" ]; then
  cd test_pytest && ./run.sh && cd "$WD"
fi
if [ "$QSPPRED_TEST_CLI" == "true" ]; then
  cd test_cli && ./run.sh && cd "$WD"
fi
cd test_tutorial && ./prepare.sh && cd "$WD"
cd test_consistency && ./run.sh && cd "$WD"
if [ "$QSPPRED_TEST_TUTORIAL" == "true" ]; then
  cd test_tutorial && ./run.sh && cd "$WD"
fi