#!/bin/bash

# Run all jupyter notebooks in the tutorial directory
set -e

# prepare tutorial data
./prepare.sh

# Run each notebook in the directory
export TUTORIAL_BASE="../../tutorials"
find $TUTORIAL_BASE -name "*.ipynb" | while read notebook
do
    jupyter nbconvert --to notebook --execute "$notebook"
done