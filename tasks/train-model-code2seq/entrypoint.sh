#!/bin/bash

set -ex
rm -rf /mnt/outputs/model
mkdir -p /mnt/outputs/model

FOLDER=/mnt/inputs
export PYTHONPATH="$PYTHONPATH:/code2seq/:/seq2seq/"
NUM_REPLACEMENTS=1000

if [ "$(head -n1 "${FOLDER}/data.train.c2s" | tr -cd ' ' | wc -c)" = "1001" ]; then
  mkdir -p /staging
  cp $FOLDER/data.* /staging
  python3 -m data_preprocessing.preprocess_code2seq_data data code2seq --collect-vocabulary --data_folder /staging --num_replacements $NUM_REPLACEMENTS
  cp /staging/vocabulary.pkl /mnt/outputs/model
fi

python3 /code2seq/train.py ${DATASET_SHORT_NAME} code2seq --expt_dir /mnt/outputs/model --data_folder /staging


# if [ "${1}" = "--regular_training" ]; then
#   shift
#   python3 -u /code2seq/code2seq.py \
#     -d "${FOLDER}/data" \
#     -te "${FOLDER}/data.val.c2s" \
#     -s /mnt/outputs/model \
#     $@
# elif [ "${1}" = "--augmented_training" ]; then
#   shift
#   python /app/augment.py
#   python3 -u /code2seq/code2seq.py \
#     -d "${FOLDER}/data" \
#     -te "${FOLDER}/data.val.c2s" \
#     -td "${FOLDER}" \
#     -t "2" \
#     -s /mnt/outputs/model \
#     $@
# elif [ "${1}" = "--adv_fine_tune" ]; then
#   SELECTED_MODEL=$(
#     find /mnt/outputs \
#       -type f \
#       -name "model_iter*" \
#     | awk -F'.' '{ print $1 }' \
#     | sort -t 'r' -k 2 -n -u \
#     | tail -n1
#   )

#   shift
#   T="${1}"
#   shift
#   python3 -u /code2seq/code2seq.py \
#     -d "${FOLDER}/data" \
#     -te "${FOLDER}/data.val.c2s" \
#     -td "${FOLDER}" \
#     -t "${T}" \
#     -l ${SELECTED_MODEL} \
#     -s /mnt/outputs/model \
#     $@
# else
#   T="${1}"
#   shift
#   python3 -u /code2seq/code2seq.py \
#     -d "${FOLDER}/data" \
#     -te "${FOLDER}/data.val.c2s" \
#     -td "${FOLDER}" \
#     -t "${T}" \
#     -s /mnt/outputs/model \
#     $@
# fi
