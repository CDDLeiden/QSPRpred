#!/usr/bin/env bash

# stop if anything goes wrong
set -e

rm -rf _build

export PYTHONPATH=`pwd`/../

# configure
mv ./api/modules.rst modules.rst.bak
rm -rf api
mkdir api
mv modules.rst.bak ./api/modules.rst
sphinx-apidoc -o ./api/ ../qsprpred/

# make
make html
