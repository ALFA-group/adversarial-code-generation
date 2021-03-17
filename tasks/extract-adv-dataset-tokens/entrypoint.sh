#!/bin/bash

set -ex

echo "[Step 1/2] Preparing data..."

FLAGS="--distinct"
if [ "${NO_RANDOM}" != "true" ]; then
  FLAGS="--random --distinct"
  mkdir -p /mnt/outputs/random-targeting
  if [ "${AVERLOC_JUST_TEST}" != "true" ]; then
    cat /mnt/inputs/transforms.Identity/valid.tsv | awk -F'\t' '{ print $2 "\t" $3 }' > /mnt/outputs/random-targeting/valid.tsv
  fi
fi
if [ "${NO_GRADIENT}" != "true" ]; then
  mkdir -p /mnt/outputs/gradient-targeting
  if [ "${AVERLOC_JUST_TEST}" != "true" ]; then
    cat /mnt/inputs/transforms.Identity/valid.tsv | awk -F'\t' '{ print $2 "\t" $3 }' > /mnt/outputs/gradient-targeting/valid.tsv
  fi
else
  FLAGS="--no_gradient --distinct"
  if [ "${NO_RANDOM}" != "true" ]; then
    FLAGS="--random --no_gradient --distinct"
  fi
fi

if [ "${AVERLOC_JUST_TEST}" != "true" ]; then
    python3 /app/app.py train $@
fi

if [ "${NO_TEST}" != "true" ]; then
  python3 /app/app.py test $@
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

OUTPUT_TEST_FNAME="test.tsv"
if [ "${EXACT_MATCHES}" == "1" ]; then
	FLAGS+=" --exact_matches"
	OUTPUT_TEST_FNAME="test_small.tsv"
fi


echo ${FLAGS}

echo "  + Done!"

echo "[Step 2/2] Doing targeting..."
echo "EXPERIMENT: ${SHORT_NAME}"
echo "EPOCH: ${CURRENT_ATTACK_EPOCH}"

if [ "${AVERLOC_JUST_TEST}" != "true" ]; then
    time CUDA_LAUNCH_BLOCKING=0 python3 /model/gradient_attack.py \
    --data_path /mnt/outputs/train.tsv \
    --expt_dir /models/lstm \
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
      --source_data_path /mnt/outputs/train.tsv \
      --dest_data_path /mnt/outputs/gradient-targeting/train.tsv \
      --mapping_json /mnt/outputs/targets-train-gradient.json 
	   
  fi

  if [ "${NO_RANDOM}" != "true" ]; then
    python3 /model/replace_tokens.py \
      --source_data_path /mnt/outputs/train.tsv \
      --dest_data_path /mnt/outputs/random-targeting/train.tsv \
      --mapping_json /mnt/outputs/targets-test-random.json

  fi
fi
if [ "${NO_TEST}" != "true" ]; then
    time CUDA_LAUNCH_BLOCKING=0 python3 /model/gradient_attack.py \
    --data_path /mnt/outputs/test.tsv \
    --expt_dir /models/lstm \
    --load_checkpoint "${CHECKPOINT}" \
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
    python3 /model/replace_tokens.py \
      --source_data_path /mnt/outputs/${OUTPUT_TEST_FNAME} \
      --dest_data_path /mnt/outputs/gradient-targeting/test.tsv \
      --mapping_json /mnt/outputs/targets-test-gradient.json 

	mv /mnt/outputs/stats.json /mnt/outputs/gradient-targeting/stats.json
    
  fi

  if [ "${NO_RANDOM}" != "true" ]; then
    python3 /model/replace_tokens.py \
      --source_data_path /mnt/outputs/${OUTPUT_TEST_FNAME} \
      --dest_data_path /mnt/outputs/random-targeting/test.tsv \
      --mapping_json /mnt/outputs/targets-test-random.json 
  fi
fi

#rm -f /mnt/outputs/test.tsv
#rm -f /mnt/outputs/train.tsv
#rm -f /mnt/outputs/targets-test-random.json
#rm -f /mnt/outputs/targets-test-gradient.json
#rm -f /mnt/outputs/targets-train-random.json
#rm -f /mnt/outputs/targets-train-gradient.json

echo "  + Done!"
