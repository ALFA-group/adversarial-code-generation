#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

trap "echo 'CTRL-C Pressed. Quiting...'; exit;" SIGINT SIGTERM

dataset=${1}
shift

set -x  
MODELS_OUT="trained-models/${dataset}/normal" \
DATASET_NAME="datasets/adversarial/just-one-step-attacks/tokens/${dataset}" \
ARGS="$@ --regular_training" \
  time make train-model-seq2seq


MODELS_OUT="trained-models/${dataset}/adversarial-one-step" \
DATASET_NAME="datasets/adversarial/just-one-step-attacks/tokens/${dataset}" \
ARGS="$@" \
  time make train-model-seq2seq


MODELS_OUT="trained-models/${dataset}/adversarial-all" \
DATASET_NAME="datasets/adversarial/all-attacks/tokens/${dataset}" \
ARGS="$@" \
  time make train-model-seq2seq
set +x
