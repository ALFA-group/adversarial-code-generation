#!/bin/bash

MAX_DATA_CONTEXTS=1000
MAX_CONTEXTS=200
SUBTOKEN_VOCAB_SIZE=15000 # Standardize with our seq2seq model
TARGET_VOCAB_SIZE=5000 # Standardize with our seq2seq model 
NUM_THREADS=$(nproc)
PYTHON=python3
JUST_TEST=no

mkdir -p /mnt/outputs

if [ ! -f "/mnt/inputs/train.jsonl.gz" ]; then
  echo "Just test.jsonl.gz... fixing up for just test extraction"
  cp /mnt/inputs/test.jsonl.gz /mnt/inputs/train.jsonl.gz
  cp /mnt/inputs/test.jsonl.gz /mnt/inputs/valid.jsonl.gz
  JUST_TEST=yes
fi

if [ "${1}" == "java" ]; then

  TRAIN_DATA_FILE=/mnt/outputs/train.raw.txt
  VAL_DATA_FILE=/mnt/outputs/valid.raw.txt
  TEST_DATA_FILE=/mnt/outputs/test.raw.txt
  BASELINE_DATA_FILE=/mnt/outputs/baseline.raw.txt
  EXTRACTOR_JAR=/code2seq/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar

  echo "Extracting paths from validation set..."
  ${PYTHON} /code2seq/JavaExtractor/extract.py --dir /mnt/inputs/valid.jsonl.gz --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${VAL_DATA_FILE}
  echo "Finished extracting paths from validation set"
  echo "Extracting paths from test set..."
  ${PYTHON} /code2seq/JavaExtractor/extract.py --dir /mnt/inputs/test.jsonl.gz --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${TEST_DATA_FILE}
  echo "Finished extracting paths from test set"
  echo "Extracting paths from training set..."
  ${PYTHON} /code2seq/JavaExtractor/extract.py --dir /mnt/inputs/train.jsonl.gz --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} | shuf > ${TRAIN_DATA_FILE}
  echo "Finished extracting paths from training set"

  EXTRA_FLAG=""
  if [ -f /mnt/inputs/baseline.jsonl.gz ]; then
    echo "Extracting paths from baseline set..."
    ${PYTHON} /code2seq/JavaExtractor/extract.py --dir /mnt/inputs/baseline.jsonl.gz --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar ${EXTRACTOR_JAR} > ${BASELINE_DATA_FILE}
    echo "Finished extracting paths from baseline set"
    EXTRA_FLAG="--baseline_data ${BASELINE_DATA_FILE}"
  fi

  TARGET_HISTOGRAM_FILE=/mnt/outputs/histo.tgt.c2s
  SOURCE_SUBTOKEN_HISTOGRAM=/mnt/outputs/histo.ori.c2s
  NODE_HISTOGRAM_FILE=/mnt/outputs/histo.node.c2s

  echo "Creating histograms from the training data"
  cat ${TRAIN_DATA_FILE} | cut -d' ' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
  cat ${TRAIN_DATA_FILE} | cut -d' ' -f3- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_SUBTOKEN_HISTOGRAM}
  cat ${TRAIN_DATA_FILE} | cut -d' ' -f3- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}

  ${PYTHON} /code2seq/preprocess.py --train_data ${TRAIN_DATA_FILE} ${EXTRA_FLAG} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
    --max_contexts ${MAX_CONTEXTS} --max_data_contexts ${MAX_DATA_CONTEXTS} --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
    --target_vocab_size ${TARGET_VOCAB_SIZE} --subtoken_histogram ${SOURCE_SUBTOKEN_HISTOGRAM} \
    --node_histogram ${NODE_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name /mnt/outputs/data

  # If all went well, the raw data files can be deleted, because preprocess.py creates new files 
  # with truncated and padded number of paths for each example.
  rm -f ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE} ${BASELINE_DATA_FILE}

elif [ "${1}" == "python" ]; then

  mkdir -p /mnt/outputs/step1
  mkdir -p /mnt/outputs/step2

  # Stage files
  echo "Staging files..."
  echo "  - Staging test.jsonl.gz..."
  cp /mnt/inputs/test_site_map.json /mnt/outputs
  cat /mnt/inputs/test.jsonl.gz | gzip -cd | python3 /pystage.py > /mnt/outputs/step1/test.json
  echo "  + Complete"
  echo "  - Staging train.jsonl.gz..."
  cp /mnt/inputs/train_site_map.json /mnt/outputs
  cat /mnt/inputs/train.jsonl.gz | gzip -cd | python3 /pystage.py > /mnt/outputs/step1/train.json
  echo "  + Complete"
  echo "  - Staging valid.jsonl.gz..."
  cp /mnt/inputs/valid_site_map.json /mnt/outputs
  cat /mnt/inputs/valid.jsonl.gz | gzip -cd | python3 /pystage.py > /mnt/outputs/step1/valid.json
  echo "  + Complete"

  EXTRA_FLAG=""
  if [ -f /mnt/inputs/baseline.jsonl.gz ]; then
    echo "  - Staging baseline.jsonl.gz..."
    cat /mnt/inputs/baseline.jsonl.gz | gzip -cd | python3 /pystage.py > /mnt/outputs/step1/baseline.json
    EXTRA_FLAG="--baseline_data /mnt/outputs/step2/baseline_output_file.txt"
    echo "  + Complete"
  fi

  # Python extractor
  python3 /code2seq/Python150kExtractor/extract.py \
    --n_jobs="$(nproc)" \
    --seed="12345" \
    --data_dir=/mnt/outputs/step1 \
    --max_path_width=2 \
    --max_path_length=8 \
    --output_dir=/mnt/outputs/step2

  # Modified preprocess
  data_dir=/mnt/outputs/step2
  mkdir -p "${data_dir}"
  train_data_file=$data_dir/train_output_file.txt
  valid_data_file=$data_dir/valid_output_file.txt
  test_data_file=$data_dir/test_output_file.txt

  echo "Creating histograms from the training data..."

  TARGET_HISTOGRAM_FILE=/mnt/outputs/histo.tgt.c2s
  SOURCE_SUBTOKEN_HISTOGRAM=/mnt/outputs/histo.ori.c2s
  NODE_HISTOGRAM_FILE=/mnt/outputs/histo.node.c2s

  cut <"${train_data_file}" -d' ' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${TARGET_HISTOGRAM_FILE}"
  cut <"${train_data_file}" -d' ' -f3- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${SOURCE_SUBTOKEN_HISTOGRAM}"
  cut <"${train_data_file}" -d' ' -f3- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' >"${NODE_HISTOGRAM_FILE}"
  echo "  + Histograms created"

  echo "Preprocessing..."
  python /code2seq/preprocess.py \
    --train_data "${train_data_file}" ${EXTRA_FLAG} \
    --val_data "${valid_data_file}" \
    --test_data "${test_data_file}" \
    --max_contexts ${MAX_CONTEXTS} \
    --max_data_contexts ${MAX_DATA_CONTEXTS} \
    --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
    --target_vocab_size ${TARGET_VOCAB_SIZE} \
    --target_histogram "${TARGET_HISTOGRAM_FILE}" \
    --subtoken_histogram "${SOURCE_SUBTOKEN_HISTOGRAM}" \
    --node_histogram "${NODE_HISTOGRAM_FILE}" \
    --output_name /mnt/outputs/data

  echo "  + Preprocessing finished"

  rm -rf /mnt/outputs/step1
  rm -rf /mnt/outputs/step2

else

  echo "Lanugage: ${1} not supported for ast-paths extraction."
  exit 1

fi

if [ "${JUST_TEST}" = "yes" ]; then
  
  echo "Removing extra files (used for just-test extraction)..."
  rm -f /mnt/inputs/train.jsonl.gz
  rm -f /mnt/inputs/valid.jsonl.gz
  rm -f /mnt/outputs/data.train.c2s
  rm -f /mnt/outputs/data.val.c2s
  echo "  + Cleanup Done!"

fi
