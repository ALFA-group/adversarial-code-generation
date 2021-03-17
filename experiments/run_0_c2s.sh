NUM_REPLACE=1000
MODEL_NAME="final-models/code2seq/sri/py150/normal/model"
RUN_SETUP=false
FIND_EXACT_MATCHES=false
TRANSFORM_NAME='transforms.Rename'
DATASET_NAME='sri/py150'
DATASET_NAME_SMALL='py150'

# no attack
# ./experiments/attack_and_test_c2s.sh 0 2 1 false 1 1 false 1 false false false test-transforms.Rename-py150-no-attack transforms.Rename 0.5 0.5 0.01 sri/py150 1 $MODEL_NAME $NUM_REPLACE 1 $RUN_SETUP $FIND_EXACT_MATCHES

# z-1 site rand, pgd 3
# bash experiments/attack_and_test_c2s.sh 0 2 1 false 1 1 true 3 false false false test-transforms.Rename-py150-z_1_rand_pgd3 transforms.Rename 0.5 0.5 0.01 sri/py150 1 $MODEL_NAME $NUM_REPLACE 1 $RUN_SETUP $FIND_EXACT_MATCHES

# no attack, random replace
bash experiments/attack_and_test_c2s.sh 0 2 1 false 1 1 false 1 false true false v2-2-z_rand_1-pgd_rand_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1 $RUN_SETUP $FIND_EXACT_MATCHES

