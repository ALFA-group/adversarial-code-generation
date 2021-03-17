#!/bin/bash

set -ex

mkdir -p /mnt/staging

NUM_T=$(($# - 1))

echo "[Step 1/2] Preparing data..."

if [ "${AVERLOC_JUST_TEST}" = "true" ]; then
    python3 /app/app.py test $@
else
    if [ "${NO_TEST}" != "true" ]; then
        python3 /app/app.py test $@
    fi
    python3 /app/app.py train $@
fi

FLAGS=""
if [ "${NO_RANDOM}" != "true" ]; then
  FLAGS="--random"
  mkdir -p /mnt/outputs/random-targeting
  cp /mnt/inputs/transforms.Identity/data.dict.c2s /mnt/outputs/random-targeting/data.dict.c2s

  if [ "${AVERLOC_JUST_TEST}" = "true" ]; then
    cp /mnt/staging/data0.test.c2s /mnt/outputs/random-targeting/data0.test.c2s
  else
    if [ "${NO_TEST}" != "true" ]; then
      cp /mnt/staging/data0.test.c2s /mnt/outputs/random-targeting/data0.test.c2s
    fi
    cat /mnt/inputs/transforms.Identity/data.val.c2s | cut -d' ' -f2- > /mnt/outputs/random-targeting/data.val.c2s
    cp /mnt/staging/data0.train.c2s /mnt/outputs/random-targeting/data0.train.c2s
  fi
fi

if [ "${NO_GRADIENT}" != "true" ]; then
  mkdir -p /mnt/outputs/gradient-targeting
  cp /mnt/inputs/transforms.Identity/data.dict.c2s /mnt/outputs/gradient-targeting/data.dict.c2s
  
  if [ "${AVERLOC_JUST_TEST}" = "true" ]; then
    cp /mnt/staging/data0.test.c2s /mnt/outputs/gradient-targeting/data0.test.c2s
  else
    if [ "${NO_TEST}" != "true" ]; then
      cp /mnt/staging/data0.test.c2s /mnt/outputs/gradient-targeting/data0.test.c2s
    fi
    cat /mnt/inputs/transforms.Identity/data.val.c2s | cut -d' ' -f2- > /mnt/outputs/gradient-targeting/data.val.c2s
    cp /mnt/staging/data0.train.c2s /mnt/outputs/gradient-targeting/data0.train.c2s
  fi
else
  FLAGS="--no_gradient"
  if [ "${NO_RANDOM}" != "true" ]; then
    FLAGS="--random --no_gradient --batch_size 128"
  fi
fi

echo "  + Done!"

echo "[Step 2/2] Doing taregting..."

SELECTED_MODEL=$(
  find /models \
    -type f \
    -name "model_iter*" \
  | awk -F'.' '{ print $1 }' \
  | sort -t 'r' -k 2 -n -u \
  | tail -n1
)

if [ "${AVERLOC_JUST_TEST}" != "true" ]; then
  for t in $(seq 1 ${NUM_T}); do
    time python3 /model/gradient_attack.py \
      --data /mnt/staging/data${t}.train.c2s \
      --load "${SELECTED_MODEL}" \
      --output_json_path /mnt/staging/${t}-targets-train.json \
      ${FLAGS}

    if [ "${NO_GRADIENT}" != "true" ]; then
      python3 /model/replace_tokens.py \
        --source_data_path /mnt/staging/data${t}.train.c2s \
        --dest_data_path /mnt/outputs/gradient-targeting/data${t}.train.c2s \
        --mapping_json /mnt/staging/${t}-targets-train-gradient.json
    fi

    if [ "${NO_RANDOM}" != "true" ]; then
      python3 /model/replace_tokens.py \
      --source_data_path /mnt/staging/data${t}.train.c2s \
      --dest_data_path /mnt/outputs/random-targeting/data${t}.train.c2s \
      --mapping_json /mnt/staging/${t}-targets-train-random.json
    fi
  done
fi

if [ "${NO_TEST}" != "true" ]; then
  for t in $(seq 1 ${NUM_T}); do
    time python3 /model/gradient_attack.py \
      --data /mnt/staging/data${t}.test.c2s \
      --load "${SELECTED_MODEL}" \
      --output_json_path /mnt/staging/${t}-targets-test.json \
      ${FLAGS}

    if [ "${NO_GRADIENT}" != "true" ]; then
      python3 /model/replace_tokens.py \
        --source_data_path /mnt/staging/data${t}.test.c2s \
        --dest_data_path /mnt/outputs/gradient-targeting/data${t}.test.c2s \
        --mapping_json /mnt/staging/${t}-targets-test-gradient.json
    fi

    if [ "${NO_RANDOM}" != "true" ]; then
      python3 /model/replace_tokens.py \
      --source_data_path /mnt/staging/data${t}.test.c2s \
      --dest_data_path /mnt/outputs/random-targeting/data${t}.test.c2s \
      --mapping_json /mnt/staging/${t}-targets-test-random.json
    fi
  done
fi

echo "  + Done!"
