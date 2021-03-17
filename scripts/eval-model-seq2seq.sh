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

CHECKPOINT="Best_F1"
MODEL="normal"
if [ "${IS_ADV}" = "true" ]; then
  if [[ -z "${ADV_TYPE}" ]]; then
    echo "Error: please specfiy adversarial model type (ADV_TYPE=gradient or ADV_TYPE=random)"
    exit 1
  fi

  CHECKPOINT="Latest"
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
CHECKPOINT="${CHECKPOINT}" \
NO_RANDOM="false" \
NO_GRADIENT="false" \
NO_TEST="false" \
AVERLOC_JUST_TEST="true" \
SHORT_NAME="test-depth-${DEPTH}-attack" \
DATASET="${DATASET}" \
MODELS_IN="${MODELS_IN}" \
TRANSFORMS="${REGEX}" \
  time make extract-adv-dataset-tokens


GPU="${GPU}" \
ARGS='--batch_size 1' \
CHECKPOINT="${CHECKPOINT}" \
DATASET_NAME=datasets/adversarial/test-depth-${DEPTH}-attack/tokens/${DATASET}/random-targeting \
RESULTS_OUT=final-results/seq2seq/${DATASET}/${MODEL}-model/depth-${DEPTH}-random-attack \
MODELS_IN="${MODELS_IN}" \
  time make test-model-seq2seq

GPU="${GPU}" \
ARGS='--batch_size 1' \
CHECKPOINT="${CHECKPOINT}" \
DATASET_NAME=datasets/adversarial/test-depth-${DEPTH}-attack/tokens/${DATASET}/gradient-targeting \
RESULTS_OUT=final-results/seq2seq/${DATASET}/${MODEL}-model/depth-${DEPTH}-gradient-attack \
MODELS_IN="${MODELS_IN}" \
  time make test-model-seq2seq
