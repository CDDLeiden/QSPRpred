#!/bin/bash

# Run all jupyter notebooks in the tutorial directory
set -e

# prepare tutorial data
./prepare.sh

# Run each notebook in the directory
export TUTORIAL_BASE="../../tutorials"
cd $TUTORIAL_BASE
find . -name "*.ipynb" | while read notebook
do
    # skip converted if file ends with 'nbconvert.ipynb'
    if [[ "$notebook" == *"nbconvert.ipynb" ]]; then
        continue
    fi

    if [[ "$notebook" == *"advanced"* ]] && [[ "${QSPR_TEST_TUTORIAL_ALL:-false}" != "true" ]]; then
        continue
    fi

    # run normally
    jupyter nbconvert --to notebook --execute "$notebook"
done