#!/bin/bash

# Run all jupyter notebooks in the tutorial directory
set -e

# Run each notebook (order matters so let's do it manually)
jupyter nbconvert --to notebook --execute data_preparation.ipynb
jupyter nbconvert --to notebook --execute data_preparation_advanced.ipynb
jupyter nbconvert --to notebook --execute tutorial_training.ipynb
jupyter nbconvert --to notebook --execute tutorial_usage.ipynb