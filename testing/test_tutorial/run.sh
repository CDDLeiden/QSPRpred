#!/bin/bash

# TODO: move consistency tests here after tutorial refactoring

set -e

TUTORIAL_BASE="../../tutorial"

# Run each notebook (order matters so let's do it manually)
jupyter nbconvert --to notebook --execute ${TUTORIAL_BASE}/data_preparation.ipynb
jupyter nbconvert --to notebook --execute ${TUTORIAL_BASE}/data_preparation_advanced.ipynb
jupyter nbconvert --to notebook --execute ${TUTORIAL_BASE}/data_splitting.ipynb
jupyter nbconvert --to notebook --execute ${TUTORIAL_BASE}/tutorial_training.ipynb
jupyter nbconvert --to notebook --execute ${TUTORIAL_BASE}/tutorial_usage.ipynb
jupyter nbconvert --to notebook --execute ${TUTORIAL_BASE}/adding_new_components.ipynb
jupyter nbconvert --to notebook --execute ${TUTORIAL_BASE}/pcm.ipynb

# check consistency of models
python check_results.py