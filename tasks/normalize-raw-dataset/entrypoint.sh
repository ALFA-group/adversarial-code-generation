#!/bin/bash

INPUT_DIR=/mnt/inputs
OUTPUT_DIR=/mnt/outputs

echo "Starting dataset normalization..."

mkdir -p "${OUTPUT_DIR}"

python3 /src/function-parser/function_parser/parser_cli.py "${1}" "${2}"

echo "  + Dataset normalization complete!"

if [ "$#" == "5" ]; then
  echo "Trimming datasets..."

  echo "Sizes:"
  echo "  + Train ${3}, valid ${4}, test ${5}"

  get_seeded_random()
  {
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
      </dev/zero 2>/dev/null
  }

  cat /mnt/outputs/train.jsonl.gz \
    | gzip -cd \
    | sort \
    | shuf --random-source=<(get_seeded_random 123) \
    | head -n"${3}" \
    | gzip > /mnt/outputs/train-trimmed.jsonl.gz

  cat /mnt/outputs/valid.jsonl.gz \
    | gzip -cd \
    | sort \
    | shuf --random-source=<(get_seeded_random 1234) \
    | head -n"${4}" \
    | gzip > /mnt/outputs/valid-trimmed.jsonl.gz

  cat /mnt/outputs/test.jsonl.gz \
    | gzip -cd \
    | sort \
    | shuf --random-source=<(get_seeded_random 12345) \
    | head -n"${5}" \
    | gzip > /mnt/outputs/test-trimmed.jsonl.gz
  
  rm -f /mnt/outputs/train.jsonl.gz
  rm -f /mnt/outputs/valid.jsonl.gz
  rm -f /mnt/outputs/test.jsonl.gz

  mv /mnt/outputs/train-trimmed.jsonl.gz /mnt/outputs/train.jsonl.gz
  mv /mnt/outputs/valid-trimmed.jsonl.gz /mnt/outputs/valid.jsonl.gz
  mv /mnt/outputs/test-trimmed.jsonl.gz /mnt/outputs/test.jsonl.gz

  echo "  + Dataset trim complete!"
fi
