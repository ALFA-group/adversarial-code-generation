DATASET=$4 \
MODEL=seq2seq \
GPU=$1 \
SHORT_NAME="$2" \
ARGS='--batch_size 1' \
CHECKPOINT="Best_F1" \
SRC_FIELD="$3" \
DATASET_NAME=datasets/adversarial/${SHORT_NAME}/tokens/${DATASET}/gradient-targeting \
RESULTS_OUT=final-results/seq2seq/${DATASET}/${SHORT_NAME} \
MODELS_IN=$5 \
  time make test-model-seq2seq

