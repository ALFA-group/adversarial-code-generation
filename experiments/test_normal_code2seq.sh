#!/bin/bash 

GPU=$1 \
DATASET_NAME=datasets/adversarial/$2/ast-paths/$4/ \
RESULTS_OUT=final-results/code2seq/$4/normal-model/$2 \
MODELS_IN=$5  \
  time make test-model-code2seq


## eval on original data
# GPU=$1 \
# DATASET_NAME=datasets/transformed/preprocessed/ast-paths/sri/py150/transforms.Identity \
# RESULTS_OUT=final-results/code2seq/$4/normal-model/$2 \
# MODELS_IN=$5  \
#   time make test-model-code2seq