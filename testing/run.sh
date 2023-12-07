#!/bin/bash

set -e

WD=$(pwd)

pytest -x .. # -x: stop on first failure
cd test_cli && ./run.sh && cd "$WD"
cd test_tutorial && ./run.sh && cd "$WD"
cd test_consistency && ./run.sh && cd "$WD"