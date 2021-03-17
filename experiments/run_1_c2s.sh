#!/bin/bash
NUM_REPLACE=1000
MODEL_NAME=final-models/code2seq/sri/py150/normal/model
RUN_SETUP=false
FIND_EXACT_MATCHES=false

# no attack
# bash experiments/adv_attack_c2s.sh 0 2 1 false 1 1 false 1 false false false test-transforms.Rename-py150-no-attack transforms.Rename 0.5 0.5 0.01 sri/py150 1 $MODEL_NAME $NUM_REPLACE 1 $RUN_SETUP $FIND_EXACT_MATCHES

# z-1 site rand, pgd 3
bash experiments/adv_attack_c2s.sh 1 2 1 false 1 1 true 3 false false false test-transforms.Rename-py150-z_1_rand_pgd3 transforms.Rename 0.5 0.5 0.01 sri/py150 1 $MODEL_NAME $NUM_REPLACE 1 $RUN_SETUP $FIND_EXACT_MATCHES
