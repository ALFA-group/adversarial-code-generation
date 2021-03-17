#!/bin/bash

cat /mnt/inputs/test.jsonl.gz \
  | gzip -cd \
  | jq -r '.from_file' \
  | python3 /app/app.py \
  | gzip \
> /mnt/inputs/baseline.jsonl.gz
