#!/bin/bash

DS_NAME=$1

for MODEL in normal adversarial-one-step adversarial-all; do
  for DATASET in just-one-step-attacks all-attacks; do
   
    echo "Running attack.py for:"
    echo "  + MODEL = ${MODEL}"
    echo "  + DATASET TYPE = ${DATASET}"
    echo "  + DATASET = ${DS_NAME}"
   
    export RESULTS_OUT=test-results/${DS_NAME}/${MODEL}/${DATASET}
    export DATASET_NAME=datasets/adversarial/${DATASET}/tokens/${DS_NAME}
    export MODELS_IN=../../../mnt/trained-models/seq2seq/${DS_NAME}/${MODEL}

    if [ -f $"{RESULTS_OUT}/attacked_metrics.txt" ]; then
      echo "Found ${RESULTS_OUT}/attacked_metrics.txt:"
      echo "  - Skipping!"
      continue
    fi

    time make test-model-seq2seq
    
  done
done

