#!/bin/bash

DS_NAME=$1

for MODEL in normal adversarial-one-step adversarial-all; do
  for T in 7 8; do
   
    FAKE_DS_NAME=all-attacks

    if [ "${T}" = "7" ]; then
      FAKE_DS_NAME=just-one-step-attacks
    fi

    echo "Running attack.py for:"
    echo "  + MODEL = ${MODEL}"
    echo "  + T = ${T}"
    echo "  + DATASET = ${DS_NAME}"
   
    export RESULTS_OUT=c2s-test-results/${DS_NAME}/${MODEL}/${FAKE_DS_NAME}
    export DATASET_NAME=datasets/adversarial/all-attacks/ast-paths/${DS_NAME}
    export MODELS_IN=../../../mnt/trained-models/code2seq/${DS_NAME}/${MODEL}

    ARGS="${T} ${ARGS}" time make test-model-code2seq
    
  done
done

