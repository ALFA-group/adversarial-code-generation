#!/bin/bash

echo "Setting up dataset from '${DATASET_URL}'..."

echo "  - Downloading raw dataset..."
TMPFILE=$(mktemp)
curl ${DATASET_URL} -o "${TMPFILE}"
unzip -qq "${TMPFILE}" -d "/tmp"
rm "${TMPFILE}"
echo "    + Downloaded!"

echo "  - Creating 'test.jsonl.gz'..."
cat /tmp/*/final/jsonl/test/*.jsonl.gz \
  | gzip -cd \
  | jq -c '{ granularity: "method", language: .language, code: .code }' \
  | gzip \
>> /mnt/test.jsonl.gz
echo "    + Created!"

echo "  - Creating 'train.jsonl.gz'..."
cat /tmp/*/final/jsonl/train/*.jsonl.gz \
  | gzip -cd \
  | jq -c '{ granularity: "method", language: .language, code: .code }' \
  | gzip \
>> /mnt/train.jsonl.gz
echo "    + Created!"

echo "  - Creating 'valid.jsonl.gz'..."
cat /tmp/*/final/jsonl/valid/*.jsonl.gz \
  | gzip -cd \
  | jq -c '{ granularity: "method", language: .language, code: .code }' \
  | gzip \
>> /mnt/valid.jsonl.gz
echo "    + Created!"

echo "  + Dataset created!"
