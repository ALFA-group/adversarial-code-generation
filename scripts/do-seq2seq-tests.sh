#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

OUTPUT_DIR="results/01-25-2020"

the_set="$1"
shift

MODELS="adversarial-one-step adversarial-all normal"

TESTS_SOURCE="datasets/transformed/preprocessed/tokens"

trap "echo 'CTRL-C Pressed. Quiting...'; exit;" SIGINT SIGTERM

for the_model in ${MODELS}; do
  for the_test in $(find ${TESTS_SOURCE}/${the_set} -mindepth 1 -maxdepth 1 -type d); do
    results_dir="${OUTPUT_DIR}/${the_set}/${the_model}/$(basename ${the_test})"
    mkdir -p "${results_dir}"

    set -x  
    RESULTS_OUT="${results_dir}" \
    MODELS_IN="trained-models/${the_set}/${the_model}" \
    DATASET_NAME="${the_test}" \
    ARGS="$@ --save" \
      time  make test-model-seq2seq 
    set +x

  done
done
