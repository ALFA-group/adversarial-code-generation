#!/bin/sh

set -ex

mkdir -p /mnt/outputs

TEST_FILE=/mnt/inputs.tsv

if grep -qF 'from_file' "${TEST_FILE}"; then
  echo "Stripping first column hashes..."
  cat "${TEST_FILE}" | cut -f2- > /inputs.tsv
  TEST_FILE=/inputs.tsv
fi

python /model/evaluate.py \
  --data_path "${TEST_FILE}" \
  --expt_dir /models/lstm \
  --output_dir /mnt/outputs \
  --load_checkpoint Best_F1 \
  --save \
  --attributions \
  $@
