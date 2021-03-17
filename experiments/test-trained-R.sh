

DATASET=sri/py150 \
MODEL=seq2seq \
GPU=0 \
SHORT_NAME=$DATASET-trained-with-R \
ARGS='--batch_size 1' \
CHECKPOINT="Best_F1" \
SRC_FIELD="transforms.RenameVarParam" \
DATASET_NAME=datasets/adversarial/train-with-R-tokens-transforms.RenameVarParam-$DATASET/tokens/sri/py150/ \
RESULTS_OUT=final-results/seq2seq/${DATASET}/${SHORT_NAME} \
MODELS_IN=final-models/seq2seq/$DATASET-R-transforms.RenameVarParam/normal \
  time make test-model-seq2seq

