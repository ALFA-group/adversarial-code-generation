#!/bin/bash

DS_NAME="${1}"

DATASET_NAME=datasets/transformed/preprocessed/tokens/${DS_NAME}/transforms.Identity/test.tsv \
RESULTS_OUT=ig-results/${DS_NAME}/normal/complete \
MODELS_IN=trained-models/seq2seq/${DS_NAME}/normal \
time make do-integrated-gradients-seq2seq

DATASET_NAME=datasets/transformed/preprocessed/tokens/${DS_NAME}/transforms.Identity/test.tsv \
RESULTS_OUT=ig-results/${DS_NAME}/adversarial-one-step/complete \
MODELS_IN=trained-models/seq2seq/${DS_NAME}/adversarial-one-step \
time make do-integrated-gradients-seq2seq

DATASET_NAME=datasets/transformed/preprocessed/tokens/${DS_NAME}/transforms.Identity/test.tsv \
RESULTS_OUT=ig-results/${DS_NAME}/adversarial-all/complete \
MODELS_IN=trained-models/seq2seq/${DS_NAME}/adversarial-all \
time make do-integrated-gradients-seq2seq
