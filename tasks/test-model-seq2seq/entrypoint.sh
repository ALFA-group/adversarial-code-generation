#!/bin/sh

set -ex

TEST_FILE=/mnt/inputs/test.tsv

if grep -qF 'from_file' "${TEST_FILE}"; then
  echo "Stripping first column hashes..."
  cat "${TEST_FILE}" | cut -f2- > /inputs.tsv
  TEST_FILE=/inputs.tsv
fi

if [ "${1}" = "--no-attack" ]; then
  echo "Skipping attack.py"
  shift

  python /model/evaluate.py \
    --data_path "${TEST_FILE}" \
    --expt_dir /models/lstm \
    --output_dir /mnt/outputs \
    --load_checkpoint "${CHECKPOINT}" \
      $@ \
  | tee /mnt/outputs/log.txt
else
  echo "Attack mode .."
  python /model/evaluate.py \
    --data_path "${TEST_FILE}" \
    --expt_dir /models/lstm \
    --output_dir /mnt/outputs \
    --load_checkpoint "${CHECKPOINT}" \
    --src_field_name "${SRC_FIELD}" --save  


  #python /model/attack_batched.py \
  #  --data_path "${TEST_FILE}" \
  #  --expt_dir /models/lstm \
  #  --output_dir /mnt/outputs \
  #  --load_checkpoint "${CHECKPOINT}" $@ \
  #| tee /mnt/outputs/log-attacked.txt
fi

if [ -f /mnt/inputs/baseline.tsv ]; then
  cat /mnt/inputs/baseline.tsv | cut -f2- > /baseline-fixed.tsv
  
  python /model/evaluate.py \
    --data_path /baseline-fixed.tsv \
    --expt_dir /models/lstm \
    --output_dir /mnt/outputs \
    --load_checkpoint "${CHECKPOINT}" \
    $@
fi

