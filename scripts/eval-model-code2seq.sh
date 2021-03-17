#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ -z "${GPU}" ]]; then
  echo "Error: please specify a GPU (GPU=<device>)"
  exit 1
fi

if [[ -z "${DATASET}" ]]; then
  echo "Error: please specfiy dataset (DATASET=c2s/java-small)"
  exit 1
fi

if [[ -z "${DEPTH}" ]]; then
  echo "Error: please specfiy depth (DEPTH=1 or DEPTH=5)"
  exit 1
fi

if [[ -z "${IS_ADV}" ]]; then
  echo "Error: please specfiy is adversarial (IS_ADV=true or IS_ADV=false)"
  exit 1
fi

MODEL="normal"
if [ "${IS_ADV}" = "true" ]; then
  if [[ -z "${ADV_TYPE}" ]]; then
    echo "Error: please specfiy adversarial model type (ADV_TYPE=gradient or ADV_TYPE=random)"
    exit 1
  fi

  MODEL="adv-${ADV_TYPE}"
fi

if [ "${IS_AUG}" = "true" ]; then
  MODEL="augmented"
fi

REGEX="(transforms\.Identity|depth-${DEPTH}-.*)"
if [ "${DEPTH}" = "1" ]; then
  REGEX="transforms\.\w+"
fi

GPU="${GPU}" \
NO_RANDOM="false" \
NO_GRADIENT="false" \
NO_TEST="false" \
AVERLOC_JUST_TEST="true" \
SHORT_NAME="test-depth-${DEPTH}-attack" \
DATASET="${DATASET}" \
MODELS_IN=${MODELS_IN} \
TRANSFORMS="${REGEX}" \
  time make extract-adv-dataset-ast-paths

GPU="${GPU}" \
ARGS="--no-attack" \
DATASET_NAME=datasets/transformed/preprocessed/ast-paths/${DATASET}/transforms.Identity \
RESULTS_OUT=final-results/code2seq/${DATASET}/${MODEL}-model/no-attack \
MODELS_IN=${MODELS_IN} \
  time make test-model-code2seq

LS_COUNT=$(ls datasets/adversarial/test-depth-${DEPTH}-attack/ast-paths/${DATASET}/random-targeting | wc -l)
NUM_T=$(($LS_COUNT - 1))

GPU="${GPU}" \
ARGS="${NUM_T}" \
DATASET_NAME=datasets/adversarial/test-depth-${DEPTH}-attack/ast-paths/${DATASET}/random-targeting \
RESULTS_OUT=final-results/code2seq/${DATASET}/${MODEL}-model/depth-${DEPTH}-random-attack \
MODELS_IN=${MODELS_IN} \
  time make test-model-code2seq

GPU="${GPU}" \
ARGS="${NUM_T}" \
DATASET_NAME=datasets/adversarial/test-depth-${DEPTH}-attack/ast-paths/${DATASET}/gradient-targeting \
RESULTS_OUT=final-results/code2seq/${DATASET}/${MODEL}-model/depth-${DEPTH}-gradient-attack \
MODELS_IN=${MODELS_IN} \
  time make test-model-code2seq
