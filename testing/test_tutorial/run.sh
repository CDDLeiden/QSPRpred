#!/bin/bash

# Run all jupyter notebooks in the tutorial directory
set -e

export TUTORIAL_BASE="../../tutorials"

# Create the dataset
python "${TUTORIAL_BASE}/tutorial_data/create_tutorial_data.py"

# Run each notebook in the directory
find $TUTORIAL_BASE -name "*.ipynb" | while read notebook
do
    jupyter nbconvert --to notebook --execute "$notebook"
done