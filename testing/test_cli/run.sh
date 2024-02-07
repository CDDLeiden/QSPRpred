#!/bin/bash

set -e

export PYTHONPATH=".."

# input data and base directory
export TEST_BASE="."
export TEST_DATA="${TEST_BASE}/data/test_data_large.tsv"

function cleanup() {
  rm -rf ${TEST_BASE}/data/backup_*;
  rm -rf ${TEST_BASE}/data/*.log;
  rm -rf ${TEST_BASE}/data/*.json;
  rm -rf ${TEST_BASE}/data/*.pkl;
  rm -rf ${TEST_BASE}/qspr;
  rm -rf ${TEST_BASE}/logs;
}

cleanup

# default values of some common parameters
export SMILES='SMILES'
export TRAIN_BATCH=32
export TRAIN_GPUS=0
export N_CPUS=2
export OPTIMIZATION='bayes'
export SEARCH_SPACE='./data/search_space/search_space_test.json'
export N_TRIALS=2

###############
# DATA #
###############
python -m qsprpred.data_CLI \
-de \
-i ${TEST_BASE}/${TEST_DATA} \
-o ${TEST_BASE}/qspr/data/ \
-ncpu ${N_CPUS} \
-sm  ${SMILES} \
-pr  CL fu \
-th '{"CL":[6.5],"fu":[0.3]}' \
-tr '{"CL":"log"}' \
-sp 'time' \
-stc 'Year of first disclosure' \
-st 2000 \
-fe Morgan \
-fe RDkit \
-pd ../../qsprpred/utils/testing/test_files/test_predictor/RFC_SINGLECLASS/RFC_SINGLECLASS_meta.json \
-lv 0.01 \
-hc 0.9 \
-fv 0.0

###############
# MODELLING #
###############
python -m qsprpred.model_CLI \
-o ${TEST_BASE}/qspr/models \
-de \
-dp ${TEST_BASE}/qspr/data/CL_fu_SINGLECLASS/CL_fu_SINGLECLASS_meta.json \
-ncpu ${N_CPUS} \
--model_types RF \
-s \
-op ${OPTIMIZATION} \
-ss ${SEARCH_SPACE} \
-nt ${N_TRIALS} \
-me

###############
# PREDICTING #
###############
python -m qsprpred.predict_CLI \
-de \
-i ${TEST_DATA} \
-o ${TEST_BASE}/qspr/predictions.tsv \
-ncpu ${N_CPUS} \
-mp ${TEST_BASE}/qspr/models/RF_CL_fu_SINGLECLASS/RF_CL_fu_SINGLECLASS_meta.json \
-pr \
-fv 0.0

echo "All tests finished without errors."

cleanup