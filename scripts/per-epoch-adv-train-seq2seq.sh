#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# if [[ -z "${GPU}" ]]; then
#   echo "Error: please specify a GPU (GPU=<device>)"
#   exit 1
# fi

# if [[ -z "${DATASET}" ]]; then
#   echo "Error: please specfiy dataset (DATASET=c2s/java-small)"
#   exit 1
# fi

# trap "echo '[ADV-TRAIN]  - CTRL-C Pressed. Quiting...'; exit;" SIGINT SIGTERM

pushd "${DIR}/.." > /dev/null

# if [ "${ADV_TYPE}" = "random" ]; then
#   export NO_RANDOM="false"
#   export NO_GRADIENT="true"
#   export TARGETING="random"
# elif [ "${ADV_TYPE}" = "gradient" ]; then
#   export NO_RANDOM="true"
#   export NO_GRADIENT="false"
#   export TARGETING="gradient"
# else
#   echo "Error: please specify either random or gradient as ADV_TYPE."
#   exit 1
# fi

export GPU="${GPU}"
export LAMB=0.4
export NUM_T=8
export MAX_EPOCHS=5
export DATASET="${DATASET}"
export TRANSFORMS="${TRANSFORMS}"
export TARGETING="gradient"

 echo "[ADV-TRAIN] Copying over normal model for first-round targeting..."
 mkdir -p "final-models/seq2seq/${SHORT_NAME}/${DATASET}/${TRANSFORMS}/adversarial/lstm"
 docker run -it --rm \
   -v "${DIR}/../final-models/seq2seq/${DATASET}/normal":/mnt/inputs \
   -v "${DIR}/../final-models/seq2seq/${SHORT_NAME}/${DATASET}/${TRANSFORMS}/adversarial":/mnt/outputs \
   debian:9 \
     bash -c 'cp -r /mnt/inputs/lstm/checkpoints /mnt/outputs/lstm && mv /mnt/outputs/lstm/checkpoints/Best_F1 /mnt/outputs/lstm/checkpoints/Latest'
 echo "[ADV-TRAIN]   + Done!"
 echo "[ADV-TRAIN] Starting loop..."

export DO_FINETUNE="--adv_fine_tune"

STATS_FILE="final-models/seq2seq/${SHORT_NAME}/${DATASET}/${TRANSFORMS}/adversarial/stats.txt"

STARTTIME=$(date +%s)

echo "$SHORT_NAME ADVERSARIAL TRAINING" | tee $STATS_FILE

for epoch in $(seq 1 ${MAX_EPOCHS}); do
echo "[ADV-TRAIN]   + Epoch ${epoch} targeting begins..."
echo "[ADV-TRAIN]   + Experiment: $SHORT_NAME"

CURRENT_TIME_1=$(date +%s) 

GPU=${GPU} \
ATTACK_VERSION=${ATTACK_VERSION} \
N_ALT_ITERS=${N_ALT_ITERS} \
Z_OPTIM=${Z_OPTIM} \
U_OPTIM=${U_OPTIM} \
Z_INIT=${Z_INIT} \
U_PGD_EPOCHS=${U_PGD_EPOCHS} \
U_ACCUMULATE_BEST_REPLACEMENTS=${U_ACCUMULATE_BEST_REPLACEMENTS} \
USE_LOSS_SMOOTHING=${USE_LOSS_SMOOTHING} \
U_RAND_UPDATE_PGD=${U_RAND_UPDATE_PGD} \
Z_EPSILON=${Z_EPSILON} \
U_LEARNING_RATE=${U_LEARNING_RATE} \
Z_LEARNING_RATE=${Z_LEARNING_RATE} \
SMOOTHING_PARAM=${SMOOTHING_PARAM} \
VOCAB_TO_USE=${VOCAB_TO_USE} \
EXACT_MATCHES=${EXACT_MATCHES} \
CHECKPOINT="Latest" \
NO_RANDOM="true" \
NO_GRADIENT="false" \
NO_TEST="true" \
SHORT_NAME="${SHORT_NAME}" \
DATASET=${DATASET} \
TRANSFORMS="${TRANSFORMS}" \
MODELS_IN=final-models/seq2seq/${SHORT_NAME}/${DATASET}/${TRANSFORMS}/adversarial \
NUM_REPLACEMENTS=${NUM_REPLACEMENTS} \
CURRENT_ATTACK_EPOCH=$epoch \
  time make extract-adv-dataset-tokens

CURRENT_TIME_2=$(date +%s)
ADV_ATTACK_TIME=$(($CURRENT_TIME_2-$CURRENT_TIME_1))
echo "Adversarial attacking epoch $epoch time elapsed: $(($ADV_ATTACK_TIME/3600)):$(($ADV_ATTACK_TIME/60%60)):$(($ADV_ATTACK_TIME%60))" | tee -a $STATS_FILE


echo "[ADV-TRAIN]     + Targeting complete!"
echo "[ADV-TRAIN]   + Epoch ${epoch} training begins..."

ARGS="${DO_FINETUNE} --batch_size 16 --epochs 1 --lamb ${LAMB}" \
SHORT_NAME="${SHORT_NAME}" \
GPU="${GPU}" \
MODELS_OUT=final-models/seq2seq/${SHORT_NAME}/${DATASET}/${TRANSFORMS} \
DATASET_NAME=datasets/adversarial/${SHORT_NAME}/tokens/${DATASET}/${TARGETING}-targeting \
  time make adv-train-model-seq2seq

CURRENT_TIME_3=$(date +%s)
TRAINING_TIME=$(($CURRENT_TIME_3-$CURRENT_TIME_2))
echo "Training epoch $epoch time elapsed: $(($TRAINING_TIME/3600)):$(($TRAINING_TIME/60%60)):$(($TRAINING_TIME%60))" | tee -a $STATS_FILE
echo "[ADV-TRAIN]     + Training epoch complete!"


rm -rf ${DIR}/../datasets/adversarial/${SHORT_NAME}

done

ENDTIME=$(date +%s)
ELAPSED_TIME=$(($ENDTIME - $STARTTIME))
echo "Total time elapsed: $(($ELAPSED_TIME/3600)):$(($ELAPSED_TIME/60%60)):$(($ELAPSED_TIME%60))" | tee -a $STATS_FILE 

echo "[ADV-TRAIN]  + Training finished!"

popd > /dev/null
