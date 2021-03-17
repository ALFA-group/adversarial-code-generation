#!/bin/bash

set -ex
export PYTHONPATH="$PYTHONPATH:/code2seq/:/seq2seq/"
mkdir -p /mnt/staging

# NUM_T=$(($# - 1))
NUM_T=1


TEST_NAME="data"
FOLDER=/mnt/inputs/transforms.Rename
IDENTITY_DATA_PATH=/mnt/inputs/transforms.Identity


if [ "${EXACT_MATCHES}" == "1" ]; then

	if [ "${FIND_EXACT_MATCHES}" == "true" ]; then
		cp /models/vocabulary.pkl $IDENTITY_DATA_PATH
		python3 -m data_preprocessing.preprocess_code2seq_data data code2seq --data_folder $IDENTITY_DATA_PATH --just_test --test_name $TEST_NAME

        python3 /code2seq/get_exact_matches.py \
	    --orig_data_path $IDENTITY_DATA_PATH \
    	--data_path $FOLDER \
	    --checkpoint /models/Latest.ckpt \
    	--batch_size 1 \
	    --split "test" \
    	--vocab_path $IDENTITY_DATA_PATH/vocabulary.pkl
    fi

	TEST_NAME="small"

fi

echo "$FOLDER"
if [ "${RUN_SETUP}" == "true" ]; then
    cp /models/vocabulary.pkl $FOLDER
    python3 -m data_preprocessing.preprocess_code2seq_data data code2seq --data_folder $FOLDER --just_test --test_name $TEST_NAME
fi

if [ "${Z_OPTIM}" == "true" ]; then
    FLAGS+=" --z_optim"
fi

if [ "${U_OPTIM}" == "true" ]; then
    FLAGS+=" --u_optim"
fi

if [ "${U_ACCUMULATE_BEST_REPLACEMENTS}" == "true" ]; then
    FLAGS+=" --u_accumulate_best_replacements"
fi

if [ "${USE_LOSS_SMOOTHING}" == "true" ]; then
    FLAGS+=" --use_loss_smoothing"
fi

if [ "${U_RAND_UPDATE_PGD}" == "true" ]; then
    FLAGS+=" --u_rand_update_pgd"
fi

OUTPUT_TEST_FNAME="test1.tsv"
if [ "${EXACT_MATCHES}" == "1" ]; then
	FLAGS+=" --exact_matches"
	OUTPUT_TEST_FNAME="test1_small.tsv"
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
	echo "averloc just test"
  for t in $(seq 1 ${NUM_T}); do
    time CUDA_LAUNCH_BLOCKING=0 python3 /code2seq/gradient_attack.py \
    --data_path /mnt/outputs/train.tsv \
    --expt_dir /models \
    --load_checkpoint "${CHECKPOINT}" \
    --save_path /mnt/outputs/targets-train.json \
	--batch_size 8 \
    --n_alt_iters ${N_ALT_ITERS} \
    --z_init ${Z_INIT} \
    --u_pgd_epochs ${U_PGD_EPOCHS} \
    --z_epsilon ${Z_EPSILON} \
    --attack_version ${ATTACK_VERSION} \
    --u_learning_rate ${U_LEARNING_RATE} \
    --z_learning_rate ${Z_LEARNING_RATE} \
    --u_learning_rate ${U_LEARNING_RATE} \
    --smoothing_param ${SMOOTHING_PARAM} \
	--vocab_to_use ${VOCAB_TO_USE} \
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
    time CUDA_LAUNCH_BLOCKING=0 python3 /code2seq/gradient_attack.py \
    --data_path $FOLDER/test/ \
    --expt_dir /models/Latest.ckpt \
	--vocab /$FOLDER/vocabulary.pkl \
    --load_checkpoint ${CHECKPOINT} \
    --save_path /mnt/outputs/targets-test.json \
    --n_alt_iters ${N_ALT_ITERS} \
    --z_init ${Z_INIT} \
    --u_pgd_epochs ${U_PGD_EPOCHS} \
    --z_epsilon ${Z_EPSILON} \
    --attack_version ${ATTACK_VERSION} \
    --u_learning_rate ${U_LEARNING_RATE} \
    --z_learning_rate ${Z_LEARNING_RATE} \
    --u_learning_rate ${U_LEARNING_RATE} \
    --smoothing_param ${SMOOTHING_PARAM} \
	--vocab_to_use ${VOCAB_TO_USE} \
    ${FLAGS}


    if [ "${NO_GRADIENT}" != "true" ]; then
      python3 /code2seq/replace_tokens.py \
        --source_data_path $FOLDER/small.test.c2s \
        --dest_data_path /mnt/outputs/gradient.test.c2s \
        --mapping_json /mnt/outputs/targets-test-gradient.json \
        --transform_name $TRANSFORMS
    fi

    if [ "${NO_RANDOM}" != "true" ]; then
      python3 /code2seq/replace_tokens.py \
      --source_data_path /mnt/staging/data${t}.test.c2s \
      --dest_data_path /mnt/outputs/random-targeting/data${t}.test.c2s \
      --mapping_json /mnt/staging/${t}-targets-test-random.json
    fi
  done
fi

echo "  + Done!"

