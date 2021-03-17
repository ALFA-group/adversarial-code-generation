#!/bin/bash

set -ex

THE_DS=$1

DATASET_NAME=datasets/transformed/preprocessed/tokens/${THE_DS}/transforms.Identity \
RESULTS_OUT=test-results/${THE_DS}/normal/normal \
MODELS_IN=${2}trained-models/seq2seq/${THE_DS}/normal \
ARGS="--no-attack --save ${ARGS}" \
time make test-model-seq2seq

DATASET_NAME=datasets/transformed/preprocessed/tokens/${THE_DS}/transforms.Identity \
RESULTS_OUT=test-results/${THE_DS}/adversarial-all/normal \
MODELS_IN=${2}trained-models/seq2seq/${THE_DS}/adversarial-all \
ARGS="--no-attack --save ${ARGS}" \
time make test-model-seq2seq

DATASET_NAME=datasets/transformed/preprocessed/tokens/${THE_DS}/transforms.Identity \
RESULTS_OUT=test-results/${THE_DS}/adversarial-one-step/normal \
MODELS_IN=${2}trained-models/seq2seq/${THE_DS}/adversarial-one-step \
ARGS="--no-attack --save ${ARGS}" \
time make test-model-seq2seq

