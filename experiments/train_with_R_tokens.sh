DATASET_NAME="sri/py150"
# DATASET_NAME="c2s/java-small"
TRANSFORM_NAME="transforms.RenameVarParam"
MODEL_NAME="final-models/seq2seq/$DATASET_NAME/normal"
NUM_REPLACE=1500
# change averloc_just_test in adv_attack.sh to false

./experiments/adv_attack.sh 0 2 1 false 1 1 false 1 false false false train-with-R-tokens-$TRANSFORM_NAME-$DATASET_NAME $TRANSFORM_NAME 0.5 0.5 0.01 $DATASET_NAME 1 $MODEL_NAME $NUM_REPLACE 1


# ARGS="--regular_training --epochs 10" \
# GPU=0 \
# MODELS_OUT=final-models/seq2seq/$DATASET_NAME-R-$TRANSFORM_NAME/ \
# DATASET_NAME=datasets/adversarial/train-with-R-tokens-$TRANSFORM_NAME-$DATASET_NAME/tokens/$DATASET_NAME/ \
# time make train-model-seq2seq


