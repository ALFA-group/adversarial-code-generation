#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ -z "${GPU}" ]]; then
  echo "Error: please specify a GPU (GPU=<device>)"
  exit 1
fi

if [[ -z "${DATASET}" ]]; then
  echo "Error: please specfiy dataset (DATASET=c2s/java-small)"
  exit 1
fi

trap "echo '[ADV-TRAIN]  - CTRL-C Pressed. Quiting...'; exit;" SIGINT SIGTERM

pushd "${DIR}/.." > /dev/null

if [ "${ADV_TYPE}" = "random" ]; then
  export NO_RANDOM="false"
  export NO_GRADIENT="true"
  export TARGETING="random"
elif [ "${ADV_TYPE}" = "gradient" ]; then
  export NO_RANDOM="true"
  export NO_GRADIENT="false"
  export TARGETING="gradient"
else
  echo "Error: please specify either --random or --gradient as first arg."
  exit 1
fi

export GPU="${GPU}"
export LAMB=0.4
export NUM_T=8
export MAX_EPOCHS=10
export DATASET="${DATASET}"

 echo "[ADV-TRAIN] Copying over normal model for first-round targeting..."
 mkdir -p "final-models/seq2seq/pe${MAX_EPOCHS}-${TARGETING}-l${LAMB}/${DATASET}/adversarial/lstm"
 docker run -it --rm \
   -v "${DIR}/../final-models/seq2seq/${DATASET}/normal":/mnt/inputs \
   -v "${DIR}/../final-models/seq2seq/pe${MAX_EPOCHS}-${TARGETING}-l${LAMB}/${DATASET}/adversarial":/mnt/outputs \
   debian:9 \
     bash -c 'cp -r /mnt/inputs/lstm/checkpoints /mnt/outputs/lstm && mv /mnt/outputs/lstm/checkpoints/Best_F1 /mnt/outputs/lstm/checkpoints/Latest'
 echo "[ADV-TRAIN]   + Done!"
 echo "[ADV-TRAIN] Starting loop..."

#export DO_FINETUNE="--adv_fine_tune"

for epoch in $(seq 1 ${MAX_EPOCHS}); do
echo "[ADV-TRAIN]   + Epoch ${epoch} targeting begins..."

CHECKPOINT="Latest" \
NO_RANDOM="${NO_RANDOM}" \
NO_GRADIENT="${NO_GRADIENT}" \
NO_TEST="true" \
SHORT_NAME="d1-train-epoch-${epoch}-l${LAMB}-${TARGETING}" \
GPU="${GPU}" \
DATASET="${DATASET}" \
MODELS_IN=final-models/seq2seq/pe${MAX_EPOCHS}-${TARGETING}-l${LAMB}/${DATASET}/adversarial \
TRANSFORMS='transforms\.\w+' \
  time make extract-adv-dataset-tokens

echo "[ADV-TRAIN]     + Targeting complete!"
echo "[ADV-TRAIN]   + Epoch ${epoch} training begins..."

ARGS="--resume --load_checkpoint Latest --batch_size 16 --epochs 1 --lamb ${LAMB}" \
SHORT_NAME="d1-train-epoch-${epoch}-l${LAMB}-${TARGETING}" \
GPU="${GPU}" \
MODELS_OUT=final-models/seq2seq/pe${MAX_EPOCHS}-${TARGETING}-l${LAMB}/${DATASET} \
DATASET_NAME=datasets/adversarial/${SHORT_NAME}/tokens/${DATASET}/${TARGETING}-targeting \
  time make adv-train-model-seq2seq
echo "[ADV-TRAIN]     + Training epoch complete!"

done

echo "[ADV-TRAIN]  + Training finished!"

popd > /dev/null
