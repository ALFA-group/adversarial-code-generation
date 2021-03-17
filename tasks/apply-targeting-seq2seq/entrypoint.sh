#!/bin/sh

set -ex

mkdir -p /mnt/post-processed

cat /mnt/inputs/test.tsv | python /app/preprocess.py \
  > /mnt/post-processed/test.tsv

# python /model/target.py --expt_dir=/models/lstm --load_checkpoint=Best_F1 --data_path=/mnt/post-processed/test.tsv --batch_size=5

python /model/target.py --expt_dir=/models/lstm --load_checkpoint=Best_F1 --data_path=/mnt/post-processed/test.tsv --batch_size=128 \
  > /mnt/post-processed/test-targets.tsv

cat /mnt/post-processed/test.tsv | python /app/postprocess.py /mnt/post-processed/test-targets.tsv \
  > /mnt/outputs/test.tsv
