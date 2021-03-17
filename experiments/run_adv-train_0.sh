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
# 21. exact_matches (1 or 0)

TRANSFORM_NAME="transforms.Combined"
#DATASET_NAME="sri/py150"
DATASET_NAME="c2s/java-small"
DATASET_NAME_SMALL="javasmall"
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/adversarial"
NUM_REPLACE=1500

#./experiments/adv_train_seq2seq.sh 0 2 1 false 1 0.4 false 1 false false false v2-1-no-attack $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0

./experiments/adv_train_seq2seq.sh 0 2 1 false 1 0.4 false 1 false true false v2-2-rand-attack $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0

./experiments/adv_train_seq2seq.sh 0 2 3 true 1 0.4 true 3 false false true v2-5-z1-pgd3-smooth $TRANSFORM_NAME 0.5 0.5 1.0 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 0
