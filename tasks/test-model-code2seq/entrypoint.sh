#!/bin/bash

set -ex
export PATH_PREFIX=true

FOLDER=/mnt/inputs

SELECTED_MODEL=$(
  find /models \
    -type f \
    -name "model_iter*" \
  | awk -F'.' '{ print $1 }' \
  | sort -t 'r' -k 2 -n -u \
  | tail -n1
)

if [ "$(head -n1 "${FOLDER}/data.test.c2s" | tr -cd ' ' | wc -c)" = "1001" ]; then
  echo "Stripping first column hashes..."
  mkdir -p /staging
  cat "${FOLDER}/data.train.c2s" | cut -d ' ' -f2- > /staging/data.train.c2s
  cat "${FOLDER}/data.test.c2s" | cut -d ' ' -f2- > /staging/data.test.c2s
  cat "${FOLDER}/data.val.c2s" | cut -d ' ' -f2- > /staging/data.val.c2s
  cp "${FOLDER}/data.dict.c2s" /staging/data.dict.c2s
  FOLDER=/staging
fi

if [ "${1}" = "--no-attack" ]; then
  echo "Skipping attack.py"
  shift
  python3 /code2seq/code2seq.py \
    --load ${SELECTED_MODEL} \
    --test ${FOLDER}/data.test.c2s | tee /mnt/outputs/log-normal.txt
elif [ "${1}" = "--individual" ]; then
  echo "Individual."
  ls /mnt/inputs
  mkdir -p /staging
  shift
  cat "${FOLDER}/${1}.test.c2s" > /staging/data.test.c2s
  cp "${FOLDER}/data.dict.c2s" /staging/data.dict.c2s
  FOLDER=/staging
  shift
  python3 /code2seq/code2seq.py \
    --load ${SELECTED_MODEL} \
    --test ${FOLDER}/data.test.c2s | tee /mnt/outputs/log-normal.txt
else
  T="${1}"
  shift
  python3 /code2seq/code2seq.py \
    --load ${SELECTED_MODEL} \
    -td ${FOLDER} \
    --adv_eval \
    -t "${T}" \
    -bs 32
  python3 /seq2seq/seq2seq/evaluator/metrics.py \
    --f_true /mnt/outputs/true_target \
    --f_pred /mnt/outputs/predicted_target | tee /mnt/outputs/attack_metrics.txt
fi
