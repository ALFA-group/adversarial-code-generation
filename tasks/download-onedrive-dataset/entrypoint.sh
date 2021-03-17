#!/bin/bash

echo "Downloading One Drive dataset"

echo "  - Downloading 'test.jsonl.gz'..."
wget -O /mnt/test.jsonl.gz "${TEST_FILE_URL}"
echo "    + Finished!"

echo "  - Creating 'train.jsonl.gz'..."
wget -O /mnt/train.jsonl.gz "${TRAIN_FILE_URL}"
echo "    + Finished!"

echo "  - Creating 'valid.jsonl.gz'..."
wget -O /mnt/valid.jsonl.gz "${VALID_FILE_URL}"
echo "    + Finished!"

echo "  + Dataset downloaded!"
