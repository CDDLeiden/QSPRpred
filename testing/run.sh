#!/bin/bash

set -e

WD=$(pwd)

pytest ..
cd test_cli && ./run.sh && cd "$WD"
cd test_tutorial && ./run.sh && cd "$WD"