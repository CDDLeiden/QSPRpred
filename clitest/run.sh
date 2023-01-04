set -e

export PYTHONPATH=".."

# input data and base directory
export TEST_BASE="."
export TEST_DATA='test_data_large.tsv'

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
export SEARCH_SPACE='data/search_space/search_space_test'
export N_TRIALS=2

###############
# DATA #
###############
python -m qsprpred.data_CLI \
-b ${TEST_BASE} \
-d \
-i ${TEST_DATA} \
-ncpu ${N_CPUS} \
-sm  ${SMILES} \
-pr  CL \
-th '{"CL":[6.5],"fu":[0,0.2,0.5,4]}' \
-lt '{"CL":true,"fu":false}' \
-sp 'time' \
-stc 'Year of first disclosure' \
-st 2000 \
-fe Morgan \
-fe RDkit \
-fp ../qsprpred/data/test_files/test_predictor/PLS_REG_GABAAalpha.json \
-lv 0.01 \
-hc 0.9

###############
# MODELLING #
###############
python -m qsprpred.model_CLI \
-b ${TEST_BASE} \
-d \
-ncpu ${N_CPUS} \
-pr CL \
-m RF \
-s \
-o ${OPTIMIZATION} \
-ss ${SEARCH_SPACE} \
-nt ${N_TRIALS} \
-me

###############
# PREDICTING #
###############
python -m qsprpred.predict_CLI \
-b ${TEST_BASE} \
-d \
-i ${TEST_DATA} \
-ncpu ${N_CPUS} \
-sm  ${SMILES} \
-pr CL \
-m RF \

cleanup