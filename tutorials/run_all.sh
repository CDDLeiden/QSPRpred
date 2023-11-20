#!/bin/bash

# Run all jupyter notebooks in the tutorial directory
set -e

# Create the dataset
python tutorial_data/create_tutorial_data.py

# Run each notebook in the directory
find . -name "*.ipynb" | while read notebook
do  
    jupyter nbconvert --to notebook --execute "$notebook"
done

# check consistency of models
#python check_results.py