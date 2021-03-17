#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DATASETS="$(
  find "${DIR}/../datasets/normalized" -mindepth 3 -maxdepth 3 -type f -name "*.jsonl.gz"
)"

for dataset in $DATASETS; do
  cat "${dataset}" | gzip -cd \
    | ~/jq -r '.target_tokens[]' \
    | sort \
    | uniq -c \
    | sort -nr \
    | awk '{ print $2 }' \
  > "$(dirname "${dataset}")/$(basename "${dataset}" .jsonl.gz).targets.histo.txt"
done
