#!/bin/bash

set -e

CONDA_ROOT=/opt/conda/bin
ACTIVATE_CMD="source ${CONDA_ROOT}/activate"
ENV_NAME="qsprpred"
RUN_CMD="${ACTIVATE_CMD} && conda activate ${ENV_NAME}"
WD=`pwd`

# setting up environments
echo "Creating environment: ${ENV_NAME}"
bash -c "${ACTIVATE_CMD} && conda create -n ${ENV_NAME} python=${PYTHON_VERSION}"
bash -c "${RUN_CMD} && conda install -c bioconda clustalo"
bash -c "${RUN_CMD} && conda install -c biocore mafft"
bash -c "${RUN_CMD} && conda install cudatoolkit"
bash -c "${RUN_CMD} && python -m pip install cupy-cuda11x" # FIXME: do not hardcode version here
bash -c "${RUN_CMD} && python -m pip install jupyterlab pytest"
git clone "${QSPRPRED_REPO}"
cd QSPRpred
git checkout "${QSPRPRED_REVISION}"
bash -c "${RUN_CMD} && python -m pip install .[full]"
echo "Checking for CUDA..."
bash -c "${RUN_CMD} && python -m python -c 'import torch; print(torch.cuda.is_available())'"
echo "Checking for qsprpred version..."
bash -c "${RUN_CMD} && python -m python -c 'import qsprpred; print(qsprpred.__version__)'"

# running tests
echo "Running tests..."
cd testing
bash -c "${RUN_CMD} && ./run.sh"

echo "All tests finished successfully. Exiting..."
