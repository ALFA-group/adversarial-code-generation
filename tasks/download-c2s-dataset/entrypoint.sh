#!/bin/bash

echo "Setting up dataset from '${DATASET_URL}'..."

echo "  - Downloading raw dataset..."
curl "${DATASET_URL}" | tar -xz -C "/tmp"
echo "    + Downloaded!"

echo "  - Creating 'test.jsonl.gz'..."
find /tmp/*/test -type f | python3 /app.py | gzip > /mnt/test.jsonl.gz
echo "    + Created!"

echo "  - Creating 'train.jsonl.gz'..."
find /tmp/*/training -type f | python3 /app.py | gzip > /mnt/train.jsonl.gz
echo "    + Created!"

echo "  - Creating 'valid.jsonl.gz'..."
find /tmp/*/validation -type f | python3 /app.py | gzip > /mnt/valid.jsonl.gz
echo "    + Created!"

echo "  + Dataset created!"
