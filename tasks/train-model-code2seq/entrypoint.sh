#!/bin/bash

set -ex
rm -rf /mnt/outputs/model
mkdir -p /mnt/outputs/model

FOLDER=/mnt/inputs

if [ "$(head -n1 "${FOLDER}/data.train.c2s" | tr -cd ' ' | wc -c)" = "1001" ]; then
  echo "Stripping first column hashes..."
  mkdir -p /staging
  cat "${FOLDER}/data.train.c2s" | cut -d ' ' -f2- > /staging/data.train.c2s
  cat "${FOLDER}/data.test.c2s" | cut -d ' ' -f2- > /staging/data.test.c2s
  cat "${FOLDER}/data.val.c2s" | cut -d ' ' -f2- > /staging/data.val.c2s
  cp "${FOLDER}/data.dict.c2s" /staging/data.dict.c2s
  FOLDER=/staging
fi

if [ "${1}" = "--regular_training" ]; then
  shift
  python3 -u /code2seq/code2seq.py \
    -d "${FOLDER}/data" \
    -te "${FOLDER}/data.val.c2s" \
    -s /mnt/outputs/model \
    $@
elif [ "${1}" = "--augmented_training" ]; then
  shift
  python /app/augment.py
  python3 -u /code2seq/code2seq.py \
    -d "${FOLDER}/data" \
    -te "${FOLDER}/data.val.c2s" \
    -td "${FOLDER}" \
    -t "2" \
    -s /mnt/outputs/model \
    $@
elif [ "${1}" = "--adv_fine_tune" ]; then
  SELECTED_MODEL=$(
    find /mnt/outputs \
      -type f \
      -name "model_iter*" \
    | awk -F'.' '{ print $1 }' \
    | sort -t 'r' -k 2 -n -u \
    | tail -n1
  )

  shift
  T="${1}"
  shift
  python3 -u /code2seq/code2seq.py \
    -d "${FOLDER}/data" \
    -te "${FOLDER}/data.val.c2s" \
    -td "${FOLDER}" \
    -t "${T}" \
    -l ${SELECTED_MODEL} \
    -s /mnt/outputs/model \
    $@
else
  T="${1}"
  shift
  python3 -u /code2seq/code2seq.py \
    -d "${FOLDER}/data" \
    -te "${FOLDER}/data.val.c2s" \
    -td "${FOLDER}" \
    -t "${T}" \
    -s /mnt/outputs/model \
    $@
fi
