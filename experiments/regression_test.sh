# provide arguments in this order
# 1. GPU
# 2. attack_version
# 3. n_alt_iters
# 4. z_optim
# 5. z_init
# 6. z_epsilon
# 7. u_optim
# 8. u_pgd_epochs (v2) / pgd_epochs (v3)
# 9. u_accumulate_best_replacements
# 10. u_rand_update_pgd
# 11. use_loss_smoothing
# 12. short_name
# 13. src_field
# 14. u_learning_rate (v3)
# 15. z_learning_rate (v3)
# 16. smoothing_param (v3)
# 17. dataset (sri/py150 or c2s/java-small)
# 18. vocab_to_use 
# 19. model_in
# 20. number of replacement tokens

TRANSFORM_NAME="transforms.Combined"
DATASET_NAME="sri/py150"
DATASET_NAME_SMALL="py150"
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/normal"
NUM_REPLACE=1500

# v2 tests
./experiments/attack_and_test_seq2seq.sh 1 2 1 false 1 1 true 3 false false false test-v2-3-z_rand_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

./experiments/attack_and_test_seq2seq.sh 1 2 3 true 1 1 true 3 false false false test-v2-8-z_o_1-pgd_3_no-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

./experiments/attack_and_test_seq2seq.sh 1 2 3 true 1 1 true 3 false false true test-v2-9-z_o_1-pgd_3_smooth-$TRANSFORM_NAME-$DATASET_NAME_SMALL $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

# v3 tests
./experiments/attack_and_test_seq2seq.sh 1 3 0 true 1 1 true 1 false false false test-v3-z_o_1-pgd_1_no $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1

./experiments/attack_and_test_seq2seq.sh 1 3 0 true 1 1 true 1 false false true test-v3-z_o_1-pgd_1_smooth $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1
